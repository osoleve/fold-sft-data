# Tier1 FP Parsing Parser-Compile SFT Dataset

- Source module: `lattice/fp/parsing/parser-compile.ss`
- Source tests: `lattice/fp/parsing/test-parser-compile.ss`
- Total samples (current): `96`
- Split (current): `79 train / 17 eval` (deterministic, ~18% eval with source-function coverage)
- Families (current): `24 spec_to_code / 24 translation / 16 bugfix / 32 composition`

## Covered Functions

1. `advance-pos-range`
2. `dfa->parser`
3. `regex-ast->parser`
4. `compile-repeat-to-parser`
5. `regex->parser`
6. `regex->combinator-parser`
7. `compile-regex`
8. `compiled-regex-parse`

## Notes

- Focuses on parser-compile prerequisites: DFA-backed prefix parsing, regex AST combinator lowering, repeat quantifier parser synthesis, and compiled-regex dual-form behavior.
- Includes checks that distinguish DFA longest-prefix semantics (`regex->parser`) from ordered combinator semantics (`regex->combinator-parser`).
- Verify expressions run directly against `(require 'parser-compile)` during validation.

## Generate / Validate

```bash
python3 data/tier1-fp-parsing-parser-compile/generate_fp_parsing_parser_compile_sft.py
python3 data/tier1-fp-parsing-parser-compile/validate_fp_parsing_parser_compile_sft.py
```

Validation should end with `VALIDATION_OK`.
