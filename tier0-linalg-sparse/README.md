# tier0-linalg-sparse

SFT dataset for `lattice/linalg/sparse.ss` and `lattice/linalg/test-sparse.ss`.

## Contents

- `generate_sparse_sft.py` - deterministic sample generator
- `validate_sparse_sft.py` - schema + executable verification
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
- Difficulty: 21 easy / 25 medium / 34 hard
- Split: 66 train / 14 eval
- Source functions: 8 (`sparse-coo-from-triplets`, `sparse-coo-ref`, `coo->csr`, `sparse-csr-ref`, `dense->sparse-coo`, `sparse-csr-vec-mul`, `sparse-coo-add-impl`, `sparse-coo-drop-below`)

## Run

```bash
python3 data/tier0-linalg-sparse/generate_sparse_sft.py
python3 data/tier0-linalg-sparse/validate_sparse_sft.py
```

## Quality notes

- Verify dependency closure includes both implementation dependencies and symbols referenced by `verify_expr`.
- Translation inputs include full-behavior Python and Chez variants; thin alias wrappers were removed.
- Bugfix samples target realistic sparse defects: row/column swaps, CSR row pointer errors, dimension checks, tolerance filter inversions, and duplicate-coordinate accumulation.
