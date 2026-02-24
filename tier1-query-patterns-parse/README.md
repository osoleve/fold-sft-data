# Tier 1 SFT: Query / Patterns Parse

This folder contains a curated Tier-1 SFT dataset for:

- Module: `lattice/query/patterns-parse.ss`
- Tests used as behavioral oracle: `lattice/query/test-patterns-parse.ss`

## Files

- `generate_patterns_parse_sft.py`: deterministic dataset generator
- `validate_patterns_parse_sft.py`: schema + quality + executable validator
- `all.jsonl`: full dataset (train + eval)
- `train.jsonl`: train split
- `eval.jsonl`: eval split
- `summary.json`: family/split/function/difficulty counts
- `validation-report.json`: validator output

## Sample families

- `spec_to_code`: implement parser/security functions from contracts or skeletons
- `translation`: translate Python/Chez snippets to canonical Fold Scheme
- `bugfix`: repair realistic single-point defects with minimal edits
- `composition`: compose parser utilities, validation helpers, and sanitization pipeline

## Targeted source functions

- `extract-tags`
- `extract-tag-positions`
- `parse-tag-at`
- `valid-tag-key?`
- `format-tag`
- `tags->string`
- `has-path-traversal?`
- `safe-extract-tags`

## Notes

- Prompt diversification is deterministic via `data/sft_prompt_diversity.py`.
- Split is deterministic, eval-coverage constrained, and targets ~18% eval per family.
- Current generated split is `79 train / 17 eval` over `96` total samples.
- Verification closures are self-contained and include:
  - implementation dependency closure from `DEPENDS`
  - symbol-scanned refs from each `verify_expr` (`verify_refs`) so helper calls used only in verification are still injected.
- Validator includes quality checks for:
  - composition/source-function mismatches
  - tautological `verify_expr == ground_truth`
  - weak bugfix stubs
  - prompt/ground-truth alignment for non-composition families

## Generation

```bash
python3 data/tier1-query-patterns-parse/generate_patterns_parse_sft.py
```

## Validation

```bash
python3 data/tier1-query-patterns-parse/validate_patterns_parse_sft.py
```

Validation executes every sample `verify_expr` in Scheme and writes `validation-report.json`.
