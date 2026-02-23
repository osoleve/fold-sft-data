# Tier 0 SFT: Data / Set

This folder contains a curated Tier-0 SFT dataset for:

- Module: `lattice/data/set.ss`
- Tests used as behavioral oracle: `lattice/data/test-data-structures.ss`

## Files

- `generate_set_sft.py`: deterministic dataset generator
- `validate_set_sft.py`: schema + executable validator
- `all.jsonl`: full dataset (train + eval)
- `train.jsonl`: train split
- `eval.jsonl`: eval split
- `summary.json`: family/split counts
- `validation-report.json`: validator output

## Sample families

- `spec_to_code`: write full function definitions from requirements
- `translation`: convert Python/Chez snippets to Fold-native Scheme
- `bugfix`: repair realistic implementation defects
- `composition`: use set APIs to solve concrete tasks

## Generation

```bash
python3 data/tier0-data-set/generate_set_sft.py
```

## Validation

```bash
python3 data/tier0-data-set/validate_set_sft.py
```

Validation includes per-sample executable checks (`verify_expr`) in Scheme and enforces that `eval.jsonl` covers every source function in this module.
