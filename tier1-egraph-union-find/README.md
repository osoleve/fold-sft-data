# Tier 1 SFT: EGraph / Union-Find

This folder contains a curated Tier-1 SFT dataset for core disjoint-set logic in the e-graph union-find module.

- Module: `lattice/egraph/union-find.ss`
- Tests referenced: `lattice/egraph/test-union-find.ss`

## Files

- `generate_union_find_sft.py`: deterministic dataset generator
- `validate_union_find_sft.py`: schema + executable validator
- `all.jsonl`: full dataset (train + eval)
- `train.jsonl`: train split
- `eval.jsonl`: eval split
- `summary.json`: family/split/difficulty/source-function counts
- `validation-report.json`: validator output

## Covered Functions (8)

- `make-uf`
- `uf?`
- `uf-count`
- `uf-size`
- `uf-make-set!`
- `uf-find`
- `uf-union!`
- `uf-same-set?`

## Dataset Shape

- Total: 80 samples
- Split: 66 train / 14 eval
- Families:
  - `spec_to_code`: 16
  - `translation`: 16
  - `bugfix`: 16
  - `composition`: 32
- Difficulty: 36 easy / 26 medium / 18 hard

## Design Notes

- Prompt diversification is deterministic via `data/sft_prompt_diversity.py`.
- Split assignment is deterministic and enforces eval coverage across all 8 source functions.
- Verification dependency closure includes both static dependencies and `verify_refs` symbol extraction from `verify_expr` to avoid missing helper definitions.
- Bugfix tasks are realistic single-point defects (invariants, rank/count updates, path compression, predicate logic) without stubs.

## Generation

```bash
python3 data/tier1-egraph-union-find/generate_union_find_sft.py
```

## Validation

```bash
python3 data/tier1-egraph-union-find/validate_union_find_sft.py
```

Validation executes all `verify_expr` checks in Scheme and writes `validation-report.json`.
