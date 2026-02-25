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

- Total: 98
- Families:
  - `spec_to_code`: 34
  - `translation`: 16
  - `bugfix`: 16
  - `composition`: 32
- Difficulty: 26 easy / 33 medium / 39 hard
- Split: 62 train / 36 eval (leakage-aware, source-function coverage enforced)
- Source functions: 17
  - Core: `hamt-empty?`, `hamt-lookup`, `hamt-has-key?`, `hamt-assoc`, `hamt-dissoc`, `hamt-size`, `hamt-merge-with`, `alist->hamt`
  - Expanded coverage: `hamt-lookup-or`, `hamt-fold`, `hamt-keys`, `hamt-values`, `hamt-entries`, `hamt-map-values`, `hamt-filter`, `hamt-merge`, `dict->hamt`

## Run

```bash
python3 data/tier0-data-hamt/generate_hamt_sft.py
python3 data/tier0-data-hamt/validate_hamt_sft.py
```

## Quality notes

- Verify dependency closure includes helper definitions and symbols referenced by `verify_expr`.
- Split generation uses `data/sft_split_utils.py` leakage-aware component grouping, with guaranteed eval coverage for every `source_function`.
- `hamt-has-key?` and `hamt-merge-with` checks explicitly cover the `#f`-value edge case.
- Composition tasks include duplicate-key conversion behavior and conflict-resolution argument ordering.
