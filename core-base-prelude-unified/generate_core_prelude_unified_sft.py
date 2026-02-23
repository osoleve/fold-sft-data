#!/usr/bin/env python3
"""Generate unified SFT samples for core/base/prelude.ss.

This combines:
- data/core-base-prelude
- data/core-base-prelude-extended

into a single consolidated dataset with one deterministic split.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

OUT_DIR = Path(__file__).resolve().parent
ALL_PATH = OUT_DIR / "all.jsonl"
TRAIN_PATH = OUT_DIR / "train.jsonl"
EVAL_PATH = OUT_DIR / "eval.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.json"

SOURCE_MODULE = "core/base/prelude.ss"
SOURCE_TEST = "core/base/test-prelude.ss"

INPUT_DATASETS = [
    OUT_DIR.parent / "core-base-prelude" / "all.jsonl",
    OUT_DIR.parent / "core-base-prelude-extended" / "all.jsonl",
]

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
                row = json.loads(line)
            except json.JSONDecodeError as ex:
                raise ValueError(f"{path}:{i}: invalid json: {ex}") from ex

            missing = REQUIRED_KEYS - set(row.keys())
            if missing:
                raise ValueError(f"{path}:{i}: missing keys {sorted(missing)}")
            if row["source_module"] != SOURCE_MODULE:
                raise ValueError(f"{path}:{i}: unexpected source_module {row['source_module']}")
            rows.append(row)
    return rows


def spread_indices(n: int, k: int) -> Set[int]:
    if k <= 0:
        return set()
    if k >= n:
        return set(range(n))
    if k == 1:
        return {n // 2}

    idxs = {round(i * (n - 1) / (k - 1)) for i in range(k)}
    cursor = 0
    while len(idxs) < k:
        if cursor not in idxs:
            idxs.add(cursor)
        cursor += 1
    return idxs


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    rows: List[Dict[str, object]] = []
    for input_path in INPUT_DATASETS:
        if not input_path.exists():
            raise FileNotFoundError(f"missing input dataset: {input_path}")
        rows.extend(load_jsonl(input_path))

    # Remove old split assignment; we'll produce one consolidated split.
    samples: List[Dict[str, object]] = []
    for row in rows:
        sample = dict(row)
        sample.pop("split", None)
        sample["source_test"] = SOURCE_TEST
        samples.append(sample)

    # Deterministic ordering by id keeps generation stable.
    samples.sort(key=lambda r: str(r["id"]))

    ids = [str(r["id"]) for r in samples]
    dup_ids = [k for k, v in Counter(ids).items() if v > 1]
    if dup_ids:
        raise ValueError(f"duplicate ids in merged dataset: {dup_ids[:5]}")

    prompts = [str(r["prompt"]) for r in samples]
    dup_prompts = [k for k, v in Counter(prompts).items() if v > 1]
    if dup_prompts:
        raise ValueError("duplicate prompts in merged dataset")

    by_family: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for s in samples:
        by_family[str(s["family"])].append(s)

    # Family totals are expected from the two source datasets:
    # spec=78, trans=39, bug=39, comp=56 (total 212)
    eval_quota = {
        "spec_to_code": 17,
        "translation": 8,
        "bugfix": 8,
        "composition": 13,
    }

    for fam in eval_quota:
        if fam not in by_family:
            raise ValueError(f"missing family in merged dataset: {fam}")

    eval_ids: Set[str] = set()
    for fam, fam_samples in by_family.items():
        fam_samples.sort(key=lambda r: str(r["id"]))
        picked = spread_indices(len(fam_samples), eval_quota[fam])
        for i, s in enumerate(fam_samples):
            if i in picked:
                eval_ids.add(str(s["id"]))

    id_to_sample: Dict[str, Dict[str, object]] = {str(s["id"]): s for s in samples}
    all_source_functions = sorted({str(s["source_function"]) for s in samples})

    def eval_source_fn_counts(ids_set: Set[str]) -> Counter:
        return Counter(str(id_to_sample[sid]["source_function"]) for sid in ids_set)

    # Repair coverage so every source function appears in eval.
    changed = True
    while changed:
        changed = False
        fn_counts = eval_source_fn_counts(eval_ids)
        missing_fns = [fn for fn in all_source_functions if fn_counts[fn] == 0]
        if not missing_fns:
            break

        for fn in missing_fns:
            candidates = [s for s in samples if str(s["source_function"]) == fn and str(s["id"]) not in eval_ids]
            swapped = False
            for cand in candidates:
                fam = str(cand["family"])
                fam_eval = [id_to_sample[sid] for sid in eval_ids if str(id_to_sample[sid]["family"]) == fam]
                removable = [r for r in fam_eval if fn_counts[str(r["source_function"])] > 1]
                if not removable:
                    continue
                removable.sort(key=lambda r: (fn_counts[str(r["source_function"])] , str(r["id"])), reverse=True)
                out = removable[0]
                eval_ids.remove(str(out["id"]))
                eval_ids.add(str(cand["id"]))
                changed = True
                swapped = True
                break
            if swapped:
                break

    missing_after = [fn for fn in all_source_functions if eval_source_fn_counts(eval_ids)[fn] == 0]
    if missing_after:
        raise ValueError(f"eval split missing source functions after repair: {missing_after}")

    train_rows: List[Dict[str, object]] = []
    eval_rows: List[Dict[str, object]] = []

    for s in samples:
        row = dict(s)
        if str(s["id"]) in eval_ids:
            row["split"] = "eval"
            eval_rows.append(row)
        else:
            row["split"] = "train"
            train_rows.append(row)

    if len(samples) != 212:
        raise ValueError(f"expected 212 samples, got {len(samples)}")
    if len(eval_rows) != 46 or len(train_rows) != 166:
        raise ValueError(f"split mismatch: train={len(train_rows)}, eval={len(eval_rows)}")

    all_rows = [dict(s, split=("eval" if str(s["id"]) in eval_ids else "train")) for s in samples]

    write_jsonl(ALL_PATH, all_rows)
    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(EVAL_PATH, eval_rows)

    summary = {
        "total": len(samples),
        "train": len(train_rows),
        "eval": len(eval_rows),
        "source_datasets": [str(p) for p in INPUT_DATASETS],
        "families": {
            fam: {
                "total": len(fam_samples),
                "eval": sum(1 for s in fam_samples if str(s["id"]) in eval_ids),
                "train": sum(1 for s in fam_samples if str(s["id"]) not in eval_ids),
            }
            for fam, fam_samples in sorted(by_family.items())
        },
        "difficulty": dict(sorted(Counter(str(r["difficulty"]) for r in samples).items())),
        "source_functions": len(all_source_functions),
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote: {ALL_PATH}")
    print(f"Wrote: {TRAIN_PATH}")
    print(f"Wrote: {EVAL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
