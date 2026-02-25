# Tier 1 SFT: Query / Query DSL

This folder contains a curated Tier-1 SFT dataset for:

- Module: `lattice/query/query-dsl.ss`
- Tests used as behavioral oracle: `lattice/query/test-query-dsl.ss`

## Files

- `generate_query_dsl_sft.py`: deterministic dataset generator
- `validate_query_dsl_sft.py`: schema + executable validator
- `all.jsonl`: full dataset (train + eval)
- `train.jsonl`: train split
- `eval.jsonl`: eval split
- `summary.json`: family/split/function/difficulty counts
- `validation-report.json`: validator output

## Sample families

- `spec_to_code`: implement target query DSL functions from contract/skeleton prompts
- `translation`: convert Python/Chez snippets to canonical Fold Scheme
- `bugfix`: repair realistic single-point behavioral defects
- `composition`: synthesize expressions by composing query predicate utilities

## Targeted functions

- `build-tag-predicate`
- `build-has-refs-predicate`
- `build-refs-count-predicate`
- `build-refs-to-predicate`
- `build-payload-size-predicate`
- `interpret-match`
- `and-all`
- `or-any`

## Notes

- Prompt diversification is deterministic via `data/sft_prompt_diversity.py`.
- Split is deterministic, eval-coverage constrained, and targets ~18% eval per family.
- Current generated split is `79 train / 17 eval` over `96` total samples.
- Verify closure includes symbol-scanned dependency injection via `verify_refs(verify_expr)`.

## Generation

```bash
python3 data/tier1-query-query-dsl/generate_query_dsl_sft.py
```

## Validation

```bash
python3 data/tier1-query-query-dsl/validate_query_dsl_sft.py
```

Validation executes every sample `verify_expr` in Scheme and writes `validation-report.json`.
