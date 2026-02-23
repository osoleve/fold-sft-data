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

- Total: 80
- Families:
  - `spec_to_code`: 16
  - `translation`: 16
  - `bugfix`: 16
  - `composition`: 32
- Difficulty: 14 easy / 33 medium / 33 hard
- Split: 66 train / 14 eval
- Source functions: 8 (`avl-empty?`, `make-avl-node`, `rebalance`, `avl-lookup-by`, `avl-insert-by`, `avl-delete-min-by`, `avl-delete-by`, `avl-range-by`)

## Run

```bash
python3 data/tier0-data-avl-tree/generate_avl_tree_sft.py
python3 data/tier0-data-avl-tree/validate_avl_tree_sft.py
```

## Quality notes

- Verify dependency closure includes both declared implementation dependencies and symbol refs found inside each `verify_expr`.
- Composition tasks cover lookup/insert/delete/range behavior and invariant checks (`avl-valid?`, `avl-bst-valid?`) with non-trivial trees.
- Bugfix tasks focus on realistic AVL defects (rotation selection, comparator direction, successor replacement, and rebalance omissions).
