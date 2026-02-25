# Tier1 EGraph Match SFT Dataset

SFT dataset for `lattice/egraph/match.ss`, focused on pattern matching and rewrite application over e-graphs.

## Coverage

- Source module: `lattice/egraph/match.ss`
- Source tests: `lattice/egraph/test-match.ss`
- Source functions (8):
  - `pattern-var?`
  - `subst-try-extend`
  - `subst-merge`
  - `ematch-pattern`
  - `ematch`
  - `pattern-apply`
  - `apply-rule`
  - `apply-rules`

## Families

- `spec_to_code`: 24
- `translation`: 24
- `bugfix`: 16
- `composition`: 32
- Total: 96
- Split: 79 train / 17 eval

## Files

- `generate_egraph_match_sft.py`: deterministic generator
- `validate_egraph_match_sft.py`: schema + executable validator
- `all.jsonl`: complete dataset
- `train.jsonl`: train split
- `eval.jsonl`: eval split
- `summary.json`: aggregate counts
- `validation-report.json`: validator output

## Build

```bash
python3 data/tier1-egraph-match/generate_egraph_match_sft.py
python3 data/tier1-egraph-match/validate_egraph_match_sft.py
```

## Notes

- Prompt variation is deterministic via `user/sft/generate.ss` (grammar DSL).
- Generation is deterministic and two-stage:
  1. Python generator writes canonical `prompt_body` + labels to `.pre_diversify.jsonl`.
  2. `user/sft/generate.ss` rewrites prompts into stylistic variants and emits final `prompt`.
- Split assignment is leakage-aware via `data/sft_split_utils.py`.
- Translation includes `source-excerpt-to-fold` tasks with doc-free targets and `chez-to-fold` tasks.
- Composition prompts intentionally omit raw verifier snippets to reduce answer leakage from prompt text.
- Validator checks:
  - schema/shape invariants and tautology guards
  - near-duplicate prompt detection
  - composition `ground_truth`/`verify_expr` coherence for wrapper-style checks
  - non-trivial Chez→Fold translation similarity thresholding
  - executable verification of all `verify_expr` checks
  - bugfix negative checks (buggy snippets must fail their `verify_expr`)
