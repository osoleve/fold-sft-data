# Tier 1 SFT: Random / Probability

This folder contains a curated Tier-1 SFT dataset pipeline for:

- Module: `lattice/random/probability.ss`
- Tests used as behavioral oracle: `lattice/random/test-probability.ss`

## Files

- `generate_probability_sft.py`: deterministic dataset generator
- `validate_probability_sft.py`: schema + executable validator
- `all.jsonl`: full dataset (train + eval)
- `train.jsonl`: train split
- `eval.jsonl`: eval split
- `summary.json`: family/split/difficulty/function counts
- `validation-report.json`: validator output

## Sample families

- `spec_to_code`: implement probability-monad and log-weight utilities from specs/skeletons
- `translation`: translate Python/Chez snippets into canonical Fold Scheme
- `bugfix`: repair realistic single-point implementation defects
- `composition`: compose probability/state/log-weight APIs in executable expressions

## Scope

- Total samples: 80
- Family split: 16 `spec_to_code`, 16 `translation`, 16 `bugfix`, 32 `composition`
- Split target: 66 train / 14 eval
- Source functions:
  - `make-prob`
  - `prob?`
  - `run-prob`
  - `sample-prob`
  - `weight-prob`
  - `prob-bind`
  - `log-sum-exp`
  - `normalize-log-weights`

## Generation

```bash
python3 data/tier1-random-probability/generate_probability_sft.py
```

## Validation

```bash
python3 data/tier1-random-probability/validate_probability_sft.py
```

Validation checks schema/splits/family quotas, verifies `all/train/eval` consistency, runs each `verify_expr` in Scheme, and enforces dataset-specific quality guards for composition and bugfix entries.
