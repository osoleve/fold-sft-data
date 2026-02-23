# Tier 0 SFT: Data / Heap

This folder contains a curated Tier-0 SFT dataset for:

- Module: `lattice/data/heap.ss`
- Tests used as behavioral oracle: `lattice/data/test-heap.ss`

## Files

- `generate_heap_sft.py`: deterministic dataset generator
- `validate_heap_sft.py`: schema + executable validator
- `all.jsonl`: full dataset (train + eval)
- `train.jsonl`: train split
- `eval.jsonl`: eval split
- `summary.json`: family/split/difficulty counts
- `validation-report.json`: validator output

## Covered functions (8)

- `heap-empty?`
- `make-heap-node`
- `heap-merge`
- `heap-insert`
- `heap-delete-min`
- `heap-pop`
- `heap-size`
- `heap->list`

## Dataset shape

- Total: 80 samples
- Split: 66 train / 14 eval
- Families:
  - `spec_to_code`: 16
  - `translation`: 16
  - `bugfix`: 16
  - `composition`: 32

## Generation

```bash
python3 data/tier0-data-heap/generate_heap_sft.py
```

## Validation

```bash
python3 data/tier0-data-heap/validate_heap_sft.py
```

Validation executes all `verify_expr` checks in Scheme and enforces eval coverage for every source function in this dataset.
