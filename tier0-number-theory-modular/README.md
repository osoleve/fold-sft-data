# Tier 0 SFT: Number Theory / Modular

This folder contains a curated Tier-0 SFT dataset for:

- Module: `lattice/number-theory/modular.ss`
- Tests used as behavioral oracle: `lattice/number-theory/test-modular.ss`

## Files

- `generate_modular_sft.py`: deterministic dataset generator
- `validate_modular_sft.py`: schema + executable validator
- `all.jsonl`: full dataset (train + eval)
- `train.jsonl`: train split
- `eval.jsonl`: eval split
- `summary.json`: family/split counts
- `validation-report.json`: validator output

## Sample families

- `spec_to_code`: write full function definitions from requirements
- `translation`: convert Python/Chez snippets to Fold-native Scheme
- `bugfix`: repair realistic implementation defects
- `composition`: use modular APIs to solve concrete tasks

## Generation

```bash
python3 data/tier0-number-theory-modular/generate_modular_sft.py
```

## Validation

```bash
python3 data/tier0-number-theory-modular/validate_modular_sft.py
```

Validation includes per-sample executable checks (`verify_expr`) in Scheme.
It also enforces source-function-disjoint train/eval splits to prevent cross-split leakage.
