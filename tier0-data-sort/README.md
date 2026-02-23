# Tier 0 SFT: Data / Sort

This folder contains a curated Tier-0 SFT dataset for:

- Module: `lattice/data/sort.ss`
- Tests used as behavioral oracle: `lattice/data/test-sort.ss`

## Files

- `generate_sort_sft.py`: deterministic dataset generator
- `validate_sort_sft.py`: schema + executable validator
- `all.jsonl`: full dataset (train + eval)
- `train.jsonl`: train split
- `eval.jsonl`: eval split
- `summary.json`: family/split/difficulty counts
- `validation-report.json`: validator output

## Covered functions (8)

- `merge`
- `split-at`
- `merge-sort-by`
- `partition`
- `quicksort-by`
- `insert-sorted`
- `insertion-sort-by`
- `nth-smallest`

## Dataset shape

- Total: 80 samples
- Split: 66 train / 14 eval
- Families:
  - `spec_to_code`: 16
  - `translation`: 16
  - `bugfix`: 16
  - `composition`: 32
- Difficulty: 20 easy / 31 medium / 29 hard

## Generation

```bash
python3 data/tier0-data-sort/generate_sort_sft.py
```

## Validation

```bash
python3 data/tier0-data-sort/validate_sort_sft.py
```

Validation executes all `verify_expr` checks in Scheme and enforces eval coverage for every source function in this dataset.
