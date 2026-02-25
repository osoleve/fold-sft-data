#!/usr/bin/env python3
"""Validate tier1 egraph match SFT dataset quality."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
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

EXPECTED_TOTAL = 96
EXPECTED_FAMILIES = {
    "spec_to_code": 24,
    "translation": 24,
    "bugfix": 16,
    "composition": 32,
}
EXPECTED_SOURCE_MODULE = "lattice/egraph/match.ss"
EXPECTED_SOURCE_TEST = "lattice/egraph/test-match.ss"
EXPECTED_SOURCE_FUNCTIONS = {
    "pattern-var?",
    "subst-try-extend",
    "subst-merge",
    "ematch-pattern",
    "ematch",
    "pattern-apply",
    "apply-rule",
    "apply-rules",
}
TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")
SCHEME_FENCE_RE = re.compile(r"```scheme\s*(.*?)```", re.DOTALL)


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


def defined_function_name(code: str) -> str | None:
    m = re.search(r"\(define\s+\(([^\s)]+)", code)
    return m.group(1) if m else None


def basic_checks(rows: List[Dict[str, object]]) -> None:
    ids: List[str] = []
    prompts: List[str] = []

    for i, row in enumerate(rows, start=1):
        missing = REQUIRED_KEYS - set(row.keys())
        assert_true(not missing, f"row {i} missing keys: {sorted(missing)}")

        sid = row["id"]
        prompt_body = row["prompt_body"]
        prompt = row["prompt"]
        gt = row["ground_truth"]
        verify = row["verify_expr"]
        family = row["family"]
        split = row["split"]
        source_module = row["source_module"]
        source_test = row["source_test"]
        source_fn = row["source_function"]

        assert_true(isinstance(sid, str) and sid, f"row {i}: invalid id")
        assert_true(isinstance(prompt_body, str) and len(prompt_body.strip()) >= 20, f"row {i}: prompt_body too short")
        assert_true(isinstance(prompt, str) and len(prompt.strip()) >= 25, f"row {i}: prompt too short")
        assert_true(isinstance(gt, str) and gt.strip(), f"row {i}: empty ground_truth")
        assert_true(isinstance(verify, str) and verify.strip(), f"row {i}: empty verify_expr")
        assert_true(isinstance(row["tags"], list) and row["tags"], f"row {i}: tags missing")
        assert_true(split in {"train", "eval"}, f"row {i}: invalid split {split}")
        assert_true("<TODO>" not in gt, f"row {i}: unresolved TODO in ground_truth")
        assert_true(source_module == EXPECTED_SOURCE_MODULE, f"row {i}: unexpected source_module")
        assert_true(source_test == EXPECTED_SOURCE_TEST, f"row {i}: unexpected source_test")
        assert_true(source_fn in EXPECTED_SOURCE_FUNCTIONS, f"row {i}: unexpected source_function {source_fn}")

        if family in {"spec_to_code", "translation", "bugfix"}:
            assert_true(gt.lstrip().startswith("(define ("), f"row {i}: expected function definition")
            fn_name = defined_function_name(str(gt))
            assert_true(fn_name == source_fn, f"row {i}: source_function mismatch in ground_truth ({fn_name} != {source_fn})")
            assert_true(str(source_fn) in str(prompt), f"row {i}: prompt does not mention source function")
        elif family == "composition":
            assert_true(not gt.lstrip().startswith("(define ("), f"row {i}: composition should be expression")
            tokens = set(TOKEN_RE.findall(str(gt)))
            assert_true(source_fn in tokens, f"row {i}: composition source_function not used in ground_truth")
        else:
            raise ValueError(f"row {i}: unknown family {family}")

        if family == "bugfix":
            assert_true("Known issue:" in str(prompt_body), f"row {i}: weak bugfix prompt (missing explicit issue)")
            assert_true(bool(SCHEME_FENCE_RE.search(str(prompt_body))), f"row {i}: bugfix prompt missing scheme fenced block")

        gt_norm = normalize_ws(str(gt))
        verify_norm = normalize_ws(str(verify))
        assert_true(gt_norm != verify_norm, f"row {i}: tautological verify_expr equals ground_truth")
        assert_true(f"(equal? {gt_norm} {gt_norm})" not in verify_norm, f"row {i}: tautological equal? verify")
        assert_true(f"(eq? {gt_norm} {gt_norm})" not in verify_norm, f"row {i}: tautological eq? verify")

        ids.append(str(sid))
        prompts.append(str(prompt))

    id_dups = [x for x, c in Counter(ids).items() if c > 1]
    prompt_dups = [x for x, c in Counter(prompts).items() if c > 1]
    assert_true(not id_dups, f"duplicate ids: {id_dups[:5]}")
    assert_true(not prompt_dups, f"duplicate prompts: {len(prompt_dups)} duplicates")


def near_duplicate_prompt_checks(rows: List[Dict[str, object]]) -> None:
    by_group: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for row in rows:
        key = (str(row["family"]), str(row["source_function"]))
        by_group.setdefault(key, []).append((str(row["id"]), normalize_ws(str(row["prompt_body"]))))

    near_dups: List[Tuple[str, str, float]] = []
    for grouped in by_group.values():
        for i in range(len(grouped)):
            sid_a, prompt_a = grouped[i]
            for j in range(i + 1, len(grouped)):
                sid_b, prompt_b = grouped[j]
                ratio = difflib.SequenceMatcher(a=prompt_a, b=prompt_b).ratio()
                if ratio >= 0.985:
                    near_dups.append((sid_a, sid_b, ratio))

    assert_true(
        not near_dups,
        "near-duplicate prompts detected (ratio>=0.985): "
        + ", ".join(f"{a}/{b}={r:.3f}" for a, b, r in near_dups[:8]),
    )


def composition_gt_verify_coherence_checks(rows: List[Dict[str, object]]) -> None:
    bad: List[Tuple[str, str]] = []
    for row in rows:
        if str(row["family"]) != "composition":
            continue
        verify = str(row["verify_expr"]).strip()
        gt_norm = normalize_ws(str(row["ground_truth"]))
        op = ""
        arg = None
        for candidate in ("equal?", "="):
            parsed = extract_first_arg_from_binary_call(verify, candidate)
            if parsed is not None:
                op = candidate
                arg = parsed
                break
        if arg is None:
            continue
        if normalize_ws(arg) != gt_norm:
            bad.append((str(row["id"]), op))

    assert_true(
        not bad,
        "composition ground_truth/verify mismatch for wrapper-style checks: "
        + ", ".join(f"{sid}({op})" for sid, op in bad[:8]),
    )


def trivial_chez_translation_checks(rows: List[Dict[str, object]]) -> None:
    bad: List[Tuple[str, float]] = []
    for row in rows:
        if str(row["family"]) != "translation":
            continue
        tags = set(str(t) for t in row["tags"])
        if "chez-to-fold" not in tags:
            continue

        prompt_body = str(row["prompt_body"])
        match = SCHEME_FENCE_RE.search(prompt_body)
        assert_true(match is not None, f"row {row['id']}: missing scheme block in chez-to-fold prompt")
        chez_src = strip_doc_forms_scheme(match.group(1))
        gt = strip_doc_forms_scheme(str(row["ground_truth"]))
        ratio = difflib.SequenceMatcher(a=normalize_ws(chez_src), b=normalize_ws(gt)).ratio()
        if ratio >= 0.95:
            bad.append((str(row["id"]), ratio))

    assert_true(
        not bad,
        "trivial chez-to-fold translations detected (similarity>=0.95): "
        + ", ".join(f"{sid}={ratio:.3f}" for sid, ratio in bad[:8]),
    )


def split_checks(train_rows: List[Dict[str, object]], eval_rows: List[Dict[str, object]]) -> None:
    train_ids = {r["id"] for r in train_rows}
    eval_ids = {r["id"] for r in eval_rows}
    overlap = train_ids & eval_ids
    assert_true(not overlap, f"train/eval overlap: {sorted(list(overlap))[:10]}")

    all_functions = {str(r["source_function"]) for r in train_rows + eval_rows}
    eval_functions = {str(r["source_function"]) for r in eval_rows}
    missing_eval_functions = sorted(all_functions - eval_functions)
    assert_true(
        not missing_eval_functions,
        f"eval split missing source_function coverage: {missing_eval_functions}",
    )


def family_distribution_checks(rows: List[Dict[str, object]]) -> None:
    family_counts = Counter(str(r["family"]) for r in rows)
    assert_true(dict(family_counts) == EXPECTED_FAMILIES, f"family counts mismatch: {dict(family_counts)}")


def source_function_distribution_checks(rows: List[Dict[str, object]]) -> None:
    sf_counts = Counter(str(r["source_function"]) for r in rows)
    assert_true(set(sf_counts.keys()) == EXPECTED_SOURCE_FUNCTIONS, "source_function set mismatch")
    for fn, count in sf_counts.items():
        assert_true(count == 12, f"source_function {fn} should have 12 samples, found {count}")


def all_file_checks(
    all_rows: List[Dict[str, object]],
    train_rows: List[Dict[str, object]],
    eval_rows: List[Dict[str, object]],
) -> None:
    expected = train_rows + eval_rows
    assert_true(len(all_rows) == len(expected), "all.jsonl row count mismatch")

    by_id_expected = {str(r["id"]): r for r in expected}
    by_id_all = {str(r["id"]): r for r in all_rows}
    assert_true(set(by_id_expected.keys()) == set(by_id_all.keys()), "all.jsonl ids mismatch")

    for sid, row in by_id_all.items():
        assert_true(row == by_id_expected[sid], f"all.jsonl row mismatch for id={sid}")


def build_scheme_validation_script(rows: Iterable[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append('(load "core/lang/module.ss")')
    lines.append('(load "lattice/egraph/match.ss")')
    lines.append("(define failures '())")
    lines.append("(define (record-failure sid reason)")
    lines.append("  (set! failures (cons (cons sid reason) failures)))")

    for row in rows:
        sid = quote_scheme_string(str(row["id"]))
        verify_expr = str(row["verify_expr"])
        lines.append("(guard (ex")
        lines.append(f"        [else (record-failure {sid} \"exception\")])")
        lines.append(f"  (unless (equal? {verify_expr} #t)")
        lines.append(f"    (record-failure {sid} \"returned-false\")))")

    lines.append("(if (null? failures)")
    lines.append('    (begin (display "VALIDATION_OK\\n"))')
    lines.append("    (begin")
    lines.append('      (display "VALIDATION_FAILED\\n")')
    lines.append("      (write (reverse failures))")
    lines.append("      (newline)")
    lines.append("      (exit 1)))")
    return "\n".join(lines) + "\n"


def executable_checks(rows: List[Dict[str, object]], dataset_dir: Path) -> Tuple[bool, str, str]:
    script_text = build_scheme_validation_script(rows)
    script_path = dataset_dir / ".tmp_validate_egraph_match_sft.ss"
    script_path.write_text(script_text, encoding="utf-8")
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
    match = SCHEME_FENCE_RE.search(prompt_body)
    if not match:
        return None
    return match.group(1).strip()


def build_buggy_negative_script(buggy_def: str, verify_expr: str) -> str:
    lines: List[str] = []
    lines.append('(load "core/lang/module.ss")')
    lines.append('(load "lattice/egraph/match.ss")')
    lines.append("(define result")
    lines.append("  (guard (ex [else 'exception])")
    lines.append("    (begin")
    lines.append(buggy_def)
    lines.append(verify_expr)
    lines.append("    )))")
    lines.append("(if (equal? result #t)")
    lines.append("    (begin (display \"BUGGY_VERIFY_TRUE\\n\") (exit 1))")
    lines.append("    (begin (display \"BUGGY_VERIFY_FALSE_OR_EXCEPTION\\n\")))")
    return "\n".join(lines) + "\n"


def bugfix_negative_checks(rows: List[Dict[str, object]], dataset_dir: Path) -> None:
    failures: List[str] = []
    root_dir = dataset_dir.parent.parent

    for row in rows:
        if str(row["family"]) != "bugfix":
            continue

        sid = str(row["id"])
        prompt_body = str(row["prompt_body"])
        buggy_def = extract_buggy_definition(prompt_body)
        assert_true(buggy_def is not None, f"bugfix row {sid}: unable to extract buggy definition")

        script = build_buggy_negative_script(str(buggy_def), str(row["verify_expr"]))
        script_path = dataset_dir / f".tmp_buggy_negative_{sid}.ss"
        script_path.write_text(script, encoding="utf-8")
        try:
            proc = subprocess.run(
                ["scheme", "--quiet", "--script", str(script_path)],
                cwd=str(root_dir),
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                failures.append(sid)
        finally:
            script_path.unlink(missing_ok=True)

    assert_true(
        not failures,
        f"buggy implementations unexpectedly satisfy verify_expr: {failures[:8]}",
    )


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "total": len(rows),
        "families": dict(sorted(Counter(str(r["family"]) for r in rows).items())),
        "difficulty": dict(sorted(Counter(str(r["difficulty"]) for r in rows).items())),
        "splits": dict(sorted(Counter(str(r["split"]) for r in rows).items())),
        "source_functions": len({str(r["source_function"]) for r in rows}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate egraph/match SFT dataset")
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
    joined_rows = train_rows + eval_rows

    assert_true(len(joined_rows) == EXPECTED_TOTAL, f"unexpected total rows: {len(joined_rows)}")

    basic_checks(joined_rows)
    near_duplicate_prompt_checks(joined_rows)
    composition_gt_verify_coherence_checks(joined_rows)
    trivial_chez_translation_checks(joined_rows)
    split_checks(train_rows, eval_rows)
    family_distribution_checks(joined_rows)
    source_function_distribution_checks(joined_rows)
    all_file_checks(all_rows, train_rows, eval_rows)

    ok, stdout, stderr = executable_checks(joined_rows, dataset_dir)
    if not ok:
        print("Executable validation failed.", file=sys.stderr)
        print(stdout, file=sys.stderr)
        print(stderr, file=sys.stderr)
        return 1

    bugfix_negative_checks(joined_rows, dataset_dir)

    report = {
        "status": "ok",
        "summary": summarize(joined_rows),
    }
    print(json.dumps(report, indent=2))
    if stdout.strip():
        print(stdout.strip())

    report_path = dataset_dir / "validation-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
