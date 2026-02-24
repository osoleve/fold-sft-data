#!/usr/bin/env python3
"""Validate tier0 geometry SFT dataset quality."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REQUIRED_KEYS = {
    "id",
    "family",
    "category",
    "difficulty",
    "source_module",
    "source_test",
    "source_function",
    "prompt_body",
    "prompt",
    "ground_truth",
    "verify_expr",
    "tags",
    "split",
}

EXPECTED_TOTAL = 694
EXPECTED_FAMILIES = {
    "spec_to_code": 222,
    "translation": 222,
    "bugfix": 100,
    "composition": 150,
}
EXPECTED_SOURCE_MODULE = "lattice/geometry/geometry.ss"
EXPECTED_SOURCE_TEST = "lattice/geometry/test-geometry.ss"
TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")
SCHEME_FENCE_RE = re.compile(r"```scheme\s*(.*?)```", re.DOTALL)
MAPS_PATH = Path(__file__).resolve().parent / "_bootstrap_geometry_maps.json"


def load_maps() -> Tuple[List[str], Dict[str, str]]:
    if not MAPS_PATH.exists():
        raise FileNotFoundError(f"missing bootstrap maps: {MAPS_PATH}")
    data = json.loads(MAPS_PATH.read_text(encoding="utf-8"))
    function_order = [str(x) for x in data["function_order"]]
    difficulty = {str(k): str(v) for k, v in data["difficulty"].items()}
    return function_order, difficulty


FUNCTION_ORDER, DIFFICULTY = load_maps()
EXPECTED_SOURCE_FUNCTIONS = set(FUNCTION_ORDER)


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as ex:
                raise ValueError(f"{path}:{i}: invalid json: {ex}") from ex
    return rows


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def quote_scheme_string(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def normalize_ws(s: str) -> str:
    return " ".join(s.split())


def strip_doc_forms_scheme(code: str) -> str:
    lines = [line for line in code.splitlines() if not line.strip().startswith("(doc ")]
    return "\n".join(lines)


def extract_first_arg_from_binary_call(expr: str, op: str) -> str | None:
    expr = expr.strip()
    prefix = f"({op} "
    if not expr.startswith(prefix):
        return None

    i = len(prefix)
    start = i
    depth = 0
    in_string = False
    escaped = False
    while i < len(expr):
        ch = expr[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                if depth == 0:
                    return expr[start:i].strip()
                depth -= 1
            elif ch.isspace() and depth == 0:
                return expr[start:i].strip()
        i += 1
    return None


def basic_checks(rows: List[Dict[str, object]]) -> None:
    ids: List[str] = []
    prompts: List[str] = []

    for i, row in enumerate(rows, start=1):
        missing = REQUIRED_KEYS - set(row.keys())
        assert_true(not missing, f"row {i} missing keys: {sorted(missing)}")

        sid = str(row["id"])
        family = str(row["family"])
        source_fn = str(row["source_function"])
        prompt_body = str(row["prompt_body"])
        prompt = str(row["prompt"])
        gt = str(row["ground_truth"])
        verify = str(row["verify_expr"])

        assert_true(sid, f"row {i}: invalid id")
        assert_true(len(prompt_body.strip()) >= 20, f"row {i}: prompt_body too short")
        assert_true(len(prompt.strip()) >= 25, f"row {i}: prompt too short")
        assert_true(gt.strip(), f"row {i}: empty ground_truth")
        assert_true(verify.strip(), f"row {i}: empty verify_expr")
        assert_true(isinstance(row["tags"], list) and row["tags"], f"row {i}: tags missing")
        assert_true(str(row["split"]) in {"train", "eval"}, f"row {i}: invalid split")
        assert_true("<TODO>" not in gt, f"row {i}: unresolved TODO in ground_truth")
        assert_true(str(row["source_module"]) == EXPECTED_SOURCE_MODULE, f"row {i}: unexpected source_module")
        assert_true(str(row["source_test"]) == EXPECTED_SOURCE_TEST, f"row {i}: unexpected source_test")
        assert_true(source_fn in EXPECTED_SOURCE_FUNCTIONS, f"row {i}: unexpected source_function {source_fn}")

        if family in {"spec_to_code", "translation", "bugfix"}:
            assert_true(gt.lstrip().startswith("(define "), f"row {i}: non-composition GT should be a define form")
            assert_true(source_fn in prompt, f"row {i}: prompt does not mention source function")
        elif family == "composition":
            assert_true(not gt.lstrip().startswith("(define "), f"row {i}: composition should be expression")
            tokens = set(TOKEN_RE.findall(gt))
            assert_true(source_fn in tokens, f"row {i}: composition source function missing from ground_truth")
        else:
            raise ValueError(f"row {i}: unknown family {family}")

        if family == "bugfix":
            assert_true("Known issue:" in prompt_body, f"row {i}: weak bugfix prompt (missing Known issue)")
            assert_true(bool(SCHEME_FENCE_RE.search(prompt_body)), f"row {i}: bugfix prompt missing scheme fence")

        gt_norm = normalize_ws(gt)
        verify_norm = normalize_ws(verify)
        assert_true(gt_norm != verify_norm, f"row {i}: tautological verify equals ground_truth")
        assert_true(f"(equal? {gt_norm} {gt_norm})" not in verify_norm, f"row {i}: tautological equal? verify")

        ids.append(sid)
        prompts.append(prompt)

    id_dups = [x for x, c in Counter(ids).items() if c > 1]
    prompt_dups = [x for x, c in Counter(prompts).items() if c > 1]
    assert_true(not id_dups, f"duplicate ids: {id_dups[:5]}")
    assert_true(not prompt_dups, f"duplicate prompts: {len(prompt_dups)}")


def near_duplicate_prompt_checks(rows: List[Dict[str, object]]) -> None:
    grouped: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for r in rows:
        key = (str(r["family"]), str(r["source_function"]))
        grouped.setdefault(key, []).append((str(r["id"]), normalize_ws(str(r["prompt_body"]))))

    bad: List[Tuple[str, str, float]] = []
    for pairs in grouped.values():
        for i in range(len(pairs)):
            sid_a, pa = pairs[i]
            for j in range(i + 1, len(pairs)):
                sid_b, pb = pairs[j]
                ratio = difflib.SequenceMatcher(a=pa, b=pb).ratio()
                if ratio >= 0.995:
                    bad.append((sid_a, sid_b, ratio))

    assert_true(
        not bad,
        "near-duplicate prompt_body within (family, source_function): "
        + ", ".join(f"{a}/{b}={r:.3f}" for a, b, r in bad[:10]),
    )


def composition_gt_verify_coherence_checks(rows: List[Dict[str, object]]) -> None:
    bad: List[str] = []
    for row in rows:
        if str(row["family"]) != "composition":
            continue
        gt_norm = normalize_ws(str(row["ground_truth"]))
        verify = str(row["verify_expr"]).strip()
        arg = None
        for op in ("equal?", "="):
            parsed = extract_first_arg_from_binary_call(verify, op)
            if parsed is not None:
                arg = normalize_ws(parsed)
                break
        if arg is None:
            continue
        if arg != gt_norm:
            bad.append(str(row["id"]))

    assert_true(not bad, f"composition ground_truth/verify mismatch: {bad[:10]}")


def trivial_chez_translation_checks(rows: List[Dict[str, object]]) -> None:
    bad: List[Tuple[str, float]] = []
    for row in rows:
        if str(row["family"]) != "translation":
            continue
        tags = {str(t) for t in row["tags"]}
        if "chez-to-fold" not in tags:
            continue
        prompt_body = str(row["prompt_body"])
        m = SCHEME_FENCE_RE.search(prompt_body)
        assert_true(m is not None, f"row {row['id']}: missing scheme fence")
        chez_src = strip_doc_forms_scheme(m.group(1))
        gt = strip_doc_forms_scheme(str(row["ground_truth"]))
        ratio = difflib.SequenceMatcher(a=normalize_ws(chez_src), b=normalize_ws(gt)).ratio()
        if ratio >= 0.99:
            bad.append((str(row["id"]), ratio))

    assert_true(
        not bad,
        "trivial chez-to-fold translation detected (ratio>=0.99): "
        + ", ".join(f"{sid}={r:.3f}" for sid, r in bad[:10]),
    )


def split_checks(train_rows: List[Dict[str, object]], eval_rows: List[Dict[str, object]]) -> None:
    train_ids = {str(r["id"]) for r in train_rows}
    eval_ids = {str(r["id"]) for r in eval_rows}
    overlap = train_ids & eval_ids
    assert_true(not overlap, f"train/eval overlap: {sorted(list(overlap))[:10]}")

    eval_fns = {str(r["source_function"]) for r in eval_rows}
    missing = sorted(EXPECTED_SOURCE_FUNCTIONS - eval_fns)
    assert_true(not missing, f"eval split missing source_function coverage: {missing}")


def family_distribution_checks(rows: List[Dict[str, object]]) -> None:
    counts = Counter(str(r["family"]) for r in rows)
    assert_true(dict(counts) == EXPECTED_FAMILIES, f"family counts mismatch: {dict(counts)}")


def source_function_distribution_checks(rows: List[Dict[str, object]]) -> None:
    counts = Counter(str(r["source_function"]) for r in rows)
    assert_true(set(counts.keys()) == EXPECTED_SOURCE_FUNCTIONS, "source_function set mismatch")

    # Baseline guaranteed by construction: 3 spec + 3 translation + 1 bugfix + 2 composition = 9.
    for fn in EXPECTED_SOURCE_FUNCTIONS:
        assert_true(counts[fn] >= 9, f"source_function {fn} should have >=9 samples, found {counts[fn]}")


def all_file_checks(all_rows: List[Dict[str, object]], train_rows: List[Dict[str, object]], eval_rows: List[Dict[str, object]]) -> None:
    expected = train_rows + eval_rows
    assert_true(len(all_rows) == len(expected), "all.jsonl row count mismatch")

    by_id_expected = {str(r["id"]): r for r in expected}
    by_id_all = {str(r["id"]): r for r in all_rows}
    assert_true(set(by_id_expected.keys()) == set(by_id_all.keys()), "all.jsonl ids mismatch")

    for sid, row in by_id_all.items():
        assert_true(row == by_id_expected[sid], f"all.jsonl row mismatch for id={sid}")


def build_executable_script(rows: Iterable[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append('(load "core/lang/module.ss")')
    lines.append('(load "lattice/geometry/geometry.ss")')
    lines.append("(define failures '())")
    lines.append("(define (record sid reason) (set! failures (cons (cons sid reason) failures)))")

    for row in rows:
        sid = quote_scheme_string(str(row["id"]))
        family = str(row["family"])
        gt = quote_scheme_string(str(row["ground_truth"]))
        verify = quote_scheme_string(str(row["verify_expr"]))

        lines.append("(guard (ex [else (record " + sid + " \"exception\")])")
        lines.append("  (let ([gt-expr (read (open-input-string " + gt + "))]")
        lines.append("        [verify-expr (read (open-input-string " + verify + "))])")
        if family != "composition":
            lines.append("    (eval gt-expr (interaction-environment))")
        else:
            lines.append("    (eval gt-expr (interaction-environment))")
        lines.append("    (unless (equal? (eval verify-expr (interaction-environment)) #t)")
        lines.append("      (record " + sid + " \"returned-false\"))))")

    lines.append("(if (null? failures)")
    lines.append('    (begin (display "VALIDATION_OK\\n"))')
    lines.append("    (begin")
    lines.append('      (display "VALIDATION_FAILED\\n")')
    lines.append("      (write (reverse failures))")
    lines.append("      (newline)")
    lines.append("      (exit 1)))")
    return "\n".join(lines) + "\n"


def executable_checks(rows: List[Dict[str, object]], dataset_dir: Path) -> Tuple[bool, str, str]:
    script = build_executable_script(rows)
    script_path = dataset_dir / ".tmp_validate_geometry_exec.ss"
    script_path.write_text(script, encoding="utf-8")
    try:
        proc = subprocess.run(
            ["scheme", "--quiet", "--script", str(script_path)],
            cwd=str(dataset_dir.parent.parent),
            capture_output=True,
            text=True,
        )
        return proc.returncode == 0, proc.stdout, proc.stderr
    finally:
        script_path.unlink(missing_ok=True)


def extract_buggy_definition(prompt_body: str) -> str | None:
    m = SCHEME_FENCE_RE.search(prompt_body)
    return m.group(1).strip() if m else None


def bugfix_negative_checks(rows: List[Dict[str, object]], dataset_dir: Path) -> None:
    failures: List[str] = []
    root_dir = dataset_dir.parent.parent

    for row in rows:
        if str(row["family"]) != "bugfix":
            continue
        sid = str(row["id"])
        buggy = extract_buggy_definition(str(row["prompt_body"]))
        assert_true(buggy is not None, f"bugfix row {sid}: missing buggy definition fence")

        buggy_q = quote_scheme_string(str(buggy))
        verify_q = quote_scheme_string(str(row["verify_expr"]))

        script = "\n".join(
            [
                '(load "core/lang/module.ss")',
                '(load "lattice/geometry/geometry.ss")',
                "(define parsed",
                "  (guard (ex [else 'parse-error])",
                "    (read (open-input-string " + buggy_q + "))))",
                "(if (eq? parsed 'parse-error)",
                '    (begin (display "PARSE_ERROR\\n") (exit 1))',
                "    (begin",
                "      (let ([result",
                "              (guard (ex [else 'exception])",
                "                (begin",
                "                  (eval parsed (interaction-environment))",
                "                  (eval (read (open-input-string " + verify_q + ")) (interaction-environment))))])",
                "        (if (equal? result #t)",
                '          (begin (display "BUG_PASSED\\n") (exit 1))',
                '          (begin (display "BUG_FAILED_AS_EXPECTED\\n"))))))',
            ]
        )

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".ss", delete=False) as tf:
            tf.write(script)
            path = Path(tf.name)
        try:
            proc = subprocess.run(
                ["scheme", "--quiet", "--script", str(path)],
                cwd=str(root_dir),
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                failures.append(sid)
        finally:
            path.unlink(missing_ok=True)

    assert_true(not failures, f"bugfix negatives failed for ids: {failures[:20]}")


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "total": len(rows),
        "families": dict(sorted(Counter(str(r["family"]) for r in rows).items())),
        "difficulty": dict(sorted(Counter(str(r["difficulty"]) for r in rows).items())),
        "splits": dict(sorted(Counter(str(r["split"]) for r in rows).items())),
        "source_functions": len({str(r["source_function"]) for r in rows}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate tier0 geometry dataset")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Dataset directory (default: script directory)",
    )
    args = parser.parse_args()

    dataset_dir = args.dir.resolve()
    all_path = dataset_dir / "all.jsonl"
    train_path = dataset_dir / "train.jsonl"
    eval_path = dataset_dir / "eval.jsonl"

    if not all_path.exists() or not train_path.exists() or not eval_path.exists():
        print(f"missing dataset files in {dataset_dir}", file=sys.stderr)
        return 2

    all_rows = load_jsonl(all_path)
    train_rows = load_jsonl(train_path)
    eval_rows = load_jsonl(eval_path)
    rows = train_rows + eval_rows

    assert_true(len(rows) == EXPECTED_TOTAL, f"unexpected total rows: {len(rows)}")

    basic_checks(rows)
    near_duplicate_prompt_checks(rows)
    composition_gt_verify_coherence_checks(rows)
    trivial_chez_translation_checks(rows)
    split_checks(train_rows, eval_rows)
    family_distribution_checks(rows)
    source_function_distribution_checks(rows)
    all_file_checks(all_rows, train_rows, eval_rows)

    ok, stdout, stderr = executable_checks(rows, dataset_dir)
    if not ok:
        print("Executable validation failed.", file=sys.stderr)
        print(stdout, file=sys.stderr)
        print(stderr, file=sys.stderr)
        return 1

    bugfix_negative_checks(rows, dataset_dir)

    report = {"status": "ok", "summary": summarize(rows)}
    report_path = dataset_dir / "validation-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    if stdout.strip():
        print(stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
