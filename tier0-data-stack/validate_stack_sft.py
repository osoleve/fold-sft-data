#!/usr/bin/env python3
"""Validate Tier-0 stack SFT dataset quality."""

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


def basic_checks(rows: List[Dict[str, object]]) -> None:
    ids = []
    prompts = []

    for i, row in enumerate(rows, start=1):
        missing = REQUIRED_KEYS - set(row.keys())
        assert_true(not missing, f"row {i} missing keys: {sorted(missing)}")

        sid = row["id"]
        prompt = row["prompt"]
        gt = row["ground_truth"]
        verify = row["verify_expr"]
        family = row["family"]
        split = row["split"]

        assert_true(isinstance(sid, str) and sid, f"row {i}: invalid id")
        assert_true(isinstance(prompt, str) and len(prompt.strip()) >= 25, f"row {i}: prompt too short")
        assert_true(isinstance(gt, str) and gt.strip(), f"row {i}: empty ground_truth")
        assert_true(isinstance(verify, str) and verify.strip(), f"row {i}: empty verify_expr")
        assert_true(isinstance(row["tags"], list) and row["tags"], f"row {i}: tags missing")
        assert_true(split in {"train", "eval"}, f"row {i}: invalid split {split}")
        assert_true("<TODO>" not in gt, f"row {i}: unresolved TODO in ground_truth")

        if family in {"spec_to_code", "translation", "bugfix"}:
            assert_true(gt.lstrip().startswith("(define ("), f"row {i}: expected function definition")
        elif family == "composition":
            assert_true(not gt.lstrip().startswith("(define ("), f"row {i}: composition should be expression")
        else:
            raise ValueError(f"row {i}: unknown family {family}")

        ids.append(sid)
        prompts.append(prompt)

    id_dups = [x for x, c in Counter(ids).items() if c > 1]
    prompt_dups = [x for x, c in Counter(prompts).items() if c > 1]
    assert_true(not id_dups, f"duplicate ids: {id_dups[:5]}")
    assert_true(not prompt_dups, f"duplicate prompts: {len(prompt_dups)} duplicates")


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


def build_scheme_validation_script(rows: Iterable[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append('(load "core/lang/module.ss")')
    lines.append("(require 'stack)")
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
    lines.append("    (begin (display \"VALIDATION_OK\\n\"))")
    lines.append("    (begin")
    lines.append("      (display \"VALIDATION_FAILED\\n\")")
    lines.append("      (write (reverse failures))")
    lines.append("      (newline)")
    lines.append("      (exit 1)))")
    return "\n".join(lines) + "\n"


def executable_checks(rows: List[Dict[str, object]], dataset_dir: Path) -> Tuple[bool, str, str]:
    script_text = build_scheme_validation_script(rows)
    script_path = dataset_dir / ".tmp_validate_stack_sft.ss"
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
        "families": dict(sorted(Counter(str(r["family"]) for r in rows).items())),
        "difficulty": dict(sorted(Counter(str(r["difficulty"]) for r in rows).items())),
        "splits": dict(sorted(Counter(str(r["split"]) for r in rows).items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate stack SFT dataset")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Dataset directory (default: script directory)",
    )
    args = parser.parse_args()

    dataset_dir = args.dir.resolve()
    train_path = dataset_dir / "train.jsonl"
    eval_path = dataset_dir / "eval.jsonl"

    if not train_path.exists() or not eval_path.exists():
        print(f"missing dataset files in {dataset_dir}", file=sys.stderr)
        return 2

    train_rows = load_jsonl(train_path)
    eval_rows = load_jsonl(eval_path)
    all_rows = train_rows + eval_rows

    basic_checks(all_rows)
    split_checks(train_rows, eval_rows)

    ok, stdout, stderr = executable_checks(all_rows, dataset_dir)
    if not ok:
        print("Executable validation failed.", file=sys.stderr)
        print(stdout, file=sys.stderr)
        print(stderr, file=sys.stderr)
        return 1

    report = {
        "status": "ok",
        "summary": summarize(all_rows),
    }
    print(json.dumps(report, indent=2))
    if stdout.strip():
        print(stdout.strip())

    report_path = dataset_dir / "validation-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
