# tier0-linalg-matrix

SFT dataset for `lattice/linalg/matrix.ss` and `lattice/linalg/test-matrix.ss`.

## Contents

- `generate_matrix_sft.py` - deterministic sample generator
- `validate_matrix_sft.py` - schema + executable verification
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
- Difficulty: 21 easy / 37 medium / 22 hard
- Split: 66 train / 14 eval
- Source functions: 8 (`matrix-from-lists`, `matrix-ref`, `matrix-transpose`, `matrix-map2`, `matrix-mul`, `matrix-vec-mul`, `matrix-identity`, `matrix-submatrix`)

## Run

```bash
python3 data/tier0-linalg-matrix/generate_matrix_sft.py
python3 data/tier0-linalg-matrix/validate_matrix_sft.py
```

## Quality notes

- Verify dependency closure includes implementation dependencies plus symbols referenced by each `verify_expr`.
- Verification includes both success and error paths (ragged input, out-of-bounds, invalid range, dimension mismatch).
- Composition tasks include direct use, edge cases, and multi-function integration.
