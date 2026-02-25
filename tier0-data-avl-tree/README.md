# tier0-data-avl-tree

SFT dataset for `lattice/data/avl-tree.ss` and `lattice/data/test-avl-tree.ss`.

## Contents

- `generate_avl_tree_sft.py` - deterministic sample generator
- `validate_avl_tree_sft.py` - schema + executable verification
- `all.jsonl` - all samples
- `train.jsonl` - train split
- `eval.jsonl` - eval split
- `summary.json` - counts and distributions
- `validation-report.json` - validator status/report

## Dataset shape

- Total: 128
- Families:
  - `spec_to_code`: 32
  - `translation`: 32
  - `bugfix`: 24
  - `composition`: 40
- Difficulty: 32 easy / 63 medium / 33 hard
- Split: 102 train / 26 eval (leakage-aware, source-function coverage enforced)
- Source functions: 16
  - Core: `avl-empty?`, `make-avl-node`, `rebalance`, `avl-lookup-by`, `avl-insert-by`, `avl-delete-min-by`, `avl-delete-by`, `avl-range-by`
  - Expanded coverage: `avl-lookup`, `avl-contains?`, `avl-insert`, `avl-delete`, `avl-range`, `avl-keys-between`, `avl-less-than`, `avl-greater-than`

## Run

```bash
python3 data/tier0-data-avl-tree/generate_avl_tree_sft.py
python3 data/tier0-data-avl-tree/validate_avl_tree_sft.py
```

## Quality notes

- Verify dependency closure includes both declared implementation dependencies and symbol refs found inside each `verify_expr`.
- Split generation uses `data/sft_split_utils.py` leakage-aware component grouping, with guaranteed eval coverage for every `source_function`.
- Composition tasks cover lookup/insert/delete/range behavior and invariant checks (`avl-valid?`, `avl-bst-valid?`) with non-trivial trees.
- Bugfix tasks focus on realistic AVL defects (rotation selection, comparator direction, successor replacement, and rebalance omissions).
