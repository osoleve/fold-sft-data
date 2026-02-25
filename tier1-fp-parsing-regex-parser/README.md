# Tier1 FP Parsing Regex Parser SFT Dataset

- Source module: `lattice/fp/parsing/regex.ss`
- Source tests: `lattice/fp/parsing/test-regex.ss`, `lattice/fp/parsing/test-regex-extensions.ss`
- Total samples (current): `96`
- Split (current): `79 train / 17 eval` (deterministic, ~18% eval with source-function coverage)
- Families (current): `24 spec_to_code / 24 translation / 16 bugfix / 32 composition`

## Covered Functions

1. `parse-class-range`
2. `parse-interval`
3. `apply-postfix-op`
4. `parse-seq`
5. `parse-alt`
6. `compile-repeat`
7. `compile-anchor`
8. `compile-lookahead`

## Notes

- Targets parser-combinator and regex-compiler prerequisites for parser combinator pipelines.
- Covers quantifier range parsing, alternation/sequence normalization, postfix rewrites, and assertion-based compilation (`^/$`, lookahead).
- Verify expressions run directly against `(require 'regex)` during validation.

## Generate / Validate

```bash
python3 data/tier1-fp-parsing-regex-parser/generate_fp_parsing_regex_parser_sft.py
python3 data/tier1-fp-parsing-regex-parser/validate_fp_parsing_regex_parser_sft.py
```

Validation should end with `VALIDATION_OK`.
