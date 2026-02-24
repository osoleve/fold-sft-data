#!/usr/bin/env python3
"""Validate query aho-corasick SFT dataset quality."""

from __future__ import annotations

import argparse
import json
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
    "prompt",
    "ground_truth",
    "verify_expr",
    "tags",
    "split",
}


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as ex:
                raise ValueError(f"{path}:{idx}: invalid json: {ex}") from ex
    return rows


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def quote_scheme_string(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def basic_checks(rows: List[Dict[str, object]]) -> None:
    ids: List[str] = []
    prompts: List[str] = []

    for idx, row in enumerate(rows, start=1):
        missing = REQUIRED_KEYS - set(row.keys())
        assert_true(not missing, f"row {idx} missing keys: {sorted(missing)}")

        sid = row["id"]
        prompt = row["prompt"]
        ground_truth = row["ground_truth"]
        verify_expr = row["verify_expr"]
        family = row["family"]
        split = row["split"]

        assert_true(isinstance(sid, str) and sid, f"row {idx}: invalid id")
        assert_true(isinstance(prompt, str) and len(prompt.strip()) >= 25, f"row {idx}: prompt too short")
        assert_true(isinstance(ground_truth, str) and ground_truth.strip(), f"row {idx}: empty ground_truth")
        assert_true(isinstance(verify_expr, str) and verify_expr.strip(), f"row {idx}: empty verify_expr")
        assert_true(isinstance(row["tags"], list) and row["tags"], f"row {idx}: tags missing")
        assert_true(split in {"train", "eval"}, f"row {idx}: invalid split {split}")
        assert_true("<TODO>" not in ground_truth, f"row {idx}: unresolved TODO in ground_truth")

        if family in {"spec_to_code", "translation", "bugfix"}:
            assert_true(ground_truth.lstrip().startswith("(define ("), f"row {idx}: expected function definition")
        elif family == "composition":
            assert_true(not ground_truth.lstrip().startswith("(define ("), f"row {idx}: composition should be expression")
        else:
            raise ValueError(f"row {idx}: unknown family {family}")

        ids.append(str(sid))
        prompts.append(str(prompt))

    id_dups = [x for x, c in Counter(ids).items() if c > 1]
    prompt_dups = [x for x, c in Counter(prompts).items() if c > 1]
    assert_true(not id_dups, f"duplicate ids: {id_dups[:5]}")
    assert_true(not prompt_dups, f"duplicate prompts: {len(prompt_dups)} duplicates")


def split_checks(train_rows: List[Dict[str, object]], eval_rows: List[Dict[str, object]]) -> None:
    train_ids = {row["id"] for row in train_rows}
    eval_ids = {row["id"] for row in eval_rows}
    overlap = train_ids & eval_ids
    assert_true(not overlap, f"train/eval overlap: {sorted(list(overlap))[:10]}")

    all_functions = {str(row["source_function"]) for row in train_rows + eval_rows}
    eval_functions = {str(row["source_function"]) for row in eval_rows}
    missing_eval_functions = sorted(all_functions - eval_functions)
    assert_true(
        not missing_eval_functions,
        f"eval split missing source_function coverage: {missing_eval_functions}",
    )


def all_file_checks(
    all_rows: List[Dict[str, object]],
    train_rows: List[Dict[str, object]],
    eval_rows: List[Dict[str, object]],
) -> None:
    expected = train_rows + eval_rows
    assert_true(len(all_rows) == len(expected), "all.jsonl row count mismatch")

    by_id_expected = {str(row["id"]): row for row in expected}
    by_id_all = {str(row["id"]): row for row in all_rows}
    assert_true(set(by_id_expected.keys()) == set(by_id_all.keys()), "all.jsonl ids mismatch")

    for sid, row in by_id_all.items():
        assert_true(row == by_id_expected[sid], f"all.jsonl row mismatch for id={sid}")


def build_scheme_validation_script(rows: Iterable[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append('(load "core/lang/module.ss")')
    lines.append("(require 'aho-corasick)")
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
    script_path = dataset_dir / ".tmp_validate_query_aho_corasick_sft.ss"
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


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "total": len(rows),
        "families": dict(sorted(Counter(str(row["family"]) for row in rows).items())),
        "difficulty": dict(sorted(Counter(str(row["difficulty"]) for row in rows).items())),
        "splits": dict(sorted(Counter(str(row["split"]) for row in rows).items())),
        "source_functions": len({str(row["source_function"]) for row in rows}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate query aho-corasick SFT dataset")
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

    basic_checks(joined_rows)
    split_checks(train_rows, eval_rows)
    all_file_checks(all_rows, train_rows, eval_rows)

    ok, stdout, stderr = executable_checks(joined_rows, dataset_dir)
    if not ok:
        print("Executable validation failed.", file=sys.stderr)
        print(stdout, file=sys.stderr)
        print(stderr, file=sys.stderr)
        return 1

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
