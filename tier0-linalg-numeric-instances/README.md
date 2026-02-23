# tier0-linalg-numeric-instances

SFT dataset for `lattice/linalg/numeric-instances.ss` and `lattice/linalg/test-numeric-instances.ss`.

## Contents

- `generate_numeric_instances_sft.py` - deterministic sample generator
- `validate_numeric_instances_sft.py` - schema + executable verification
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
- Difficulty: 37 easy / 37 medium / 6 hard
- Split: 66 train / 14 eval
- Source functions: 8 (`vec-abs`, `vec-signum`, `vec-recip`, `vec-pow`, `matrix-hadamard`, `matrix/`, `matrix-recip`, `scalar-matrix+`)

## Run

```bash
python3 data/tier0-linalg-numeric-instances/generate_numeric_instances_sft.py
python3 data/tier0-linalg-numeric-instances/validate_numeric_instances_sft.py
```

## Quality notes

- Validation loads `linalg/numeric-instances` to avoid module-name ambiguity.
- Composition verifies were normalized to remove double-wrapped boolean checks.
- `source_function` attribution in composition rows aligns with the actual generated expression.
- Composition prompt/ground-truth pairs were tightened for reciprocal multiplication and matrix-recip/matrix/ composition semantics.
- Bugfix examples were revised to avoid identity-function stubs and emphasize realistic arithmetic defects.
