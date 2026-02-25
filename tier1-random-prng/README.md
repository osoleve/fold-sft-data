# Tier 1 SFT: Random / PRNG

This folder contains a curated Tier-1 SFT dataset for:

- Module: `lattice/random/prng.ss`
- Tests used as behavioral oracle: `lattice/random/test-prng.ss`

## Files

- `generate_prng_sft.py`: deterministic dataset generator
- `validate_prng_sft.py`: schema + executable validator
- `all.jsonl`: full dataset (train + eval)
- `train.jsonl`: train split
- `eval.jsonl`: eval split
- `summary.json`: family/split/difficulty/function counts
- `validation-report.json`: validator output

## Sample families

- `spec_to_code`: implement PRNG utility/state-transition functions from requirements
- `translation`: convert Python/Chez snippets into Fold-native Scheme
- `bugfix`: repair realistic PRNG implementation defects with minimal edits
- `composition`: compose PRNG helpers into executable expressions

## Scope

- Total samples: 80
- Source functions: 8
  - `u32`
  - `u64`
  - `rotl32`
  - `splitmix-next`
  - `make-pcg`
  - `pcg-next`
  - `make-xorshift128`
  - `xorshift128-next`

## Generation

```bash
python3 data/tier1-random-prng/generate_prng_sft.py
```

## Validation

```bash
python3 data/tier1-random-prng/validate_prng_sft.py
```

Validation runs schema and split checks, confirms `all/train/eval` consistency, and executes every `verify_expr` in Scheme.
