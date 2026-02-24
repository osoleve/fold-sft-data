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
- The generator emits pre-diversification `prompt_body`, then `generate.ss` produces final `prompt`.
- Split assignment is leakage-aware via `data/sft_split_utils.py`.
