# tier0-linalg-matrix-decomp

SFT dataset for `lattice/linalg/matrix-decomp.ss` and `lattice/linalg/test-matrix-decomp.ss`.

## Contents

- `generate_matrix_decomp_sft.py` - deterministic sample generator
- `validate_matrix_decomp_sft.py` - schema + executable verification
- `all.jsonl` - all samples
- `train.jsonl` - train split
- `eval.jsonl` - eval split
- `summary.json` - counts and distributions
- `validation-report.json` - validator status/report

## Dataset shape

- Total: 80
- Families:
  - `spec_to_code`: 16
  - `translation`: 16
  - `bugfix`: 16
  - `composition`: 32
- Difficulty: 30 easy / 26 medium / 24 hard
- Split: 66 train / 14 eval
- Source functions: 8 (`matrix-copy`, `matrix-set!`, `matrix-column`, `matrix-lu`, `matrix-lu-solve`, `matrix-cholesky`, `permutation-sign`, `matrix-det`)

## Run

```bash
python3 data/tier0-linalg-matrix-decomp/generate_matrix_decomp_sft.py
python3 data/tier0-linalg-matrix-decomp/validate_matrix_decomp_sft.py
```

## Quality notes

- Target definitions are extracted directly from source modules to reduce drift and syntax mismatch.
- Verify dependency closure includes implementation deps plus symbol references found in each `verify_expr`.
- Composition tasks cover matrix mutation, decomposition success/error paths, solve correctness, permutation parity, and determinant properties.
