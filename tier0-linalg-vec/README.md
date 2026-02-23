# tier0-linalg-vec

SFT dataset for `lattice/linalg/vec.ss` and `lattice/linalg/test-vec.ss`.

## Contents

- `generate_vec_sft.py` - deterministic sample generator
- `validate_vec_sft.py` - schema + executable verification
- `all.jsonl` - all samples
- `train.jsonl` - train split
- `eval.jsonl` - eval split
- `summary.json` - counts and distributions

## Dataset shape

- Total: 80
- Families:
  - `spec_to_code`: 16
  - `translation`: 16
  - `bugfix`: 16
  - `composition`: 32
- Difficulty: 23 easy / 37 medium / 20 hard
- Split: 66 train / 14 eval
- Source functions: 8 (`vec-ref`, `vec-first`, `vec-map`, `vec-zip-with`, `vec-slice`, `vec-dot`, `vec-normalize`, `vec-approx-equal?`)

## Run

```bash
python3 data/tier0-linalg-vec/generate_vec_sft.py
python3 data/tier0-linalg-vec/validate_vec_sft.py
```

## Quality notes

- Verify dependency closure includes both implementation dependencies and symbols referenced by `verify_expr`.
- Verification covers both happy paths and error-return behaviors (out-of-bounds, invalid range, dimension mismatch, zero vector).
- Composition tasks include direct calls, properties, and integration across multiple vector operations.
