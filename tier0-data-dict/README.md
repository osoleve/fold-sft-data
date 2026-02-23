# Tier 0 SFT: Data / Dict

This folder contains a curated Tier-0 SFT dataset for:

- Module: `lattice/data/dict.ss`
- Tests used as behavioral oracle: `lattice/data/test-data-structures.ss`

## Files

- `generate_dict_sft.py`: deterministic dataset generator
- `validate_dict_sft.py`: schema + executable validator
- `all.jsonl`: full dataset (train + eval)
- `train.jsonl`: train split
- `eval.jsonl`: eval split
- `summary.json`: family/split counts
- `validation-report.json`: validator output

## Sample families

- `spec_to_code`: write full function definitions from requirements
- `translation`: convert Python/Chez snippets to Fold-native Scheme
- `bugfix`: repair realistic implementation defects
- `composition`: use dict APIs to solve concrete tasks

## Generation

```bash
python3 data/tier0-data-dict/generate_dict_sft.py
```

## Validation

```bash
python3 data/tier0-data-dict/validate_dict_sft.py
```

Validation includes per-sample executable checks (`verify_expr`) in Scheme and enforces that `eval.jsonl` covers every source function in this module.
