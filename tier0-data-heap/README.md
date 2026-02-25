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

## Covered functions (16)

- `heap-empty?`
- `make-heap-node`
- `heap-merge`
- `heap-insert`
- `heap-min`
- `heap-delete-min`
- `heap-pop`
- `heap-size`
- `heap-merge-by`
- `heap-insert-by`
- `heap-delete-top-by`
- `heap-fold`
- `list->heap-by`
- `heap->list`
- `heapsort`
- `heapsort-by`

## Dataset shape

- Total: 128 samples
- Split: 102 train / 26 eval (leakage-aware, source-function coverage enforced)
- Families:
  - `spec_to_code`: 32
  - `translation`: 32
  - `bugfix`: 24
  - `composition`: 40

## Generation

```bash
python3 data/tier0-data-heap/generate_heap_sft.py
```

## Validation

```bash
python3 data/tier0-data-heap/validate_heap_sft.py
```

Validation executes all `verify_expr` checks in Scheme and enforces eval coverage for every source function in this dataset.
Split generation uses `data/sft_split_utils.py` so near-duplicate ground truths/verify blocks stay within one split.
