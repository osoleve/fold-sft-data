# tier0-linalg-integer-matrix

SFT dataset for `lattice/linalg/integer-matrix.ss` and `lattice/linalg/test-integer-matrix.ss`.

## Contents

- `generate_integer_matrix_sft.py` - deterministic sample generator
- `validate_integer_matrix_sft.py` - schema + executable verification
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
- Difficulty: 35 easy / 20 medium / 25 hard
- Split: 66 train / 14 eval
- Source functions: 8 (`int-mat-ref`, `int-mat-set!`, `int-mat-copy`, `swap-rows!`, `swap-cols!`, `make-identity-vec`, `matrix-minor`, `matrix-determinant-int`)

## Run

```bash
python3 data/tier0-linalg-integer-matrix/generate_integer_matrix_sft.py
python3 data/tier0-linalg-integer-matrix/validate_integer_matrix_sft.py
```

## Quality notes

- Verify dependency closure includes both implementation dependencies and symbols referenced by `verify_expr`.
- Determinant verification includes empty matrix and row-swap sign behavior.
- Bugfix cases for row/column swaps use observable loop-bound defects (not purely cosmetic no-op guards).
