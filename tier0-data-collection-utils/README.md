# SFT: Data / Collection Utils (Pure Subset)

This folder contains a curated SFT dataset for a pure subset of:

- Module: `lattice/data/collection-utils.ss`
- Tests used as behavioral oracle: `lattice/data/test-collection-utils.ss`

Note: this module itself is documented as higher-tier in the lattice; this dataset targets deterministic, non-FS functions only.

## Files

- `generate_collection_utils_sft.py`: deterministic dataset generator
- `validate_collection_utils_sft.py`: schema + executable validator
- `all.jsonl`: full dataset (train + eval)
- `train.jsonl`: train split
- `eval.jsonl`: eval split
- `summary.json`: family/split/difficulty counts
- `validation-report.json`: validator output

## Covered functions (8)

- `foldr`
- `collection-hashes`
- `collection-size`
- `collection-empty?`
- `make-collection-from-blocks`
- `collection-add`
- `collection-remove`
- `collection-merge`

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
python3 data/tier0-data-collection-utils/generate_collection_utils_sft.py
```

## Validation

```bash
python3 data/tier0-data-collection-utils/validate_collection_utils_sft.py
```

Validation executes all `verify_expr` checks in Scheme and enforces eval coverage for every source function in this dataset.
