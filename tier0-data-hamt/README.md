# tier0-data-hamt

SFT dataset for `lattice/data/hamt.ss` and `lattice/data/test-hamt.ss`.

## Contents

- `generate_hamt_sft.py` - deterministic sample generator
- `validate_hamt_sft.py` - schema + executable verification
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
- Difficulty: 22 easy / 25 medium / 33 hard
- Split: 66 train / 14 eval
- Source functions: 8 (`hamt-empty?`, `hamt-lookup`, `hamt-has-key?`, `hamt-assoc`, `hamt-dissoc`, `hamt-size`, `hamt-merge-with`, `alist->hamt`)

## Run

```bash
python3 data/tier0-data-hamt/generate_hamt_sft.py
python3 data/tier0-data-hamt/validate_hamt_sft.py
```

## Quality notes

- Verify dependency closure includes helper definitions and symbols referenced by `verify_expr`.
- `hamt-has-key?` and `hamt-merge-with` checks explicitly cover the `#f`-value edge case.
- Composition tasks include duplicate-key conversion behavior and conflict-resolution argument ordering.
