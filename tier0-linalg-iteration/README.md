# tier0-linalg-iteration

SFT dataset for `lattice/linalg/iteration.ss` and `lattice/linalg/test-iteration.ss`.

## Contents

- `generate_iteration_sft.py` - deterministic sample generator
- `validate_iteration_sft.py` - schema + executable verification
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
- Difficulty: 25 easy / 40 medium / 15 hard
- Split: 66 train / 14 eval
- Source functions: 8 (`vec-map-idx`, `vec-fold-idx`, `vec-zip-map-idx`, `matrix-do!`, `range-fold`, `dot-product-loop`, `vec-tabulate`, `vec-scan`)

## Run

```bash
python3 data/tier0-linalg-iteration/generate_iteration_sft.py
python3 data/tier0-linalg-iteration/validate_iteration_sft.py
```

## Quality notes

- Verify dependency closure includes both declared dependencies and symbol refs found in each function's `verify_expr`.
- Verification covers indexing/folding/iteration patterns across vector, range, matrix, dot-product, and scan macros.
- Bugfix samples target realistic defects (off-by-one bounds, wrong accumulator threading, aliased inputs, loop bound errors, and incorrect arithmetic update rules).
