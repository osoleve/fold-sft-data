# Tier1 FP Meta Combinators SFT Dataset

- Source module: `lattice/fp/meta/combinators.ss`
- Source tests: `lattice/fp/meta/test-combinators.ss`
- Total samples: `80`
- Split: `66 train / 14 eval`
- Families: `16 spec_to_code / 16 translation / 16 bugfix / 32 composition`

## Covered Functions

1. `compose`
2. `pipe`
3. `curry2`
4. `partial`
5. `maybe-bind`
6. `sequence-maybe`
7. `either-bind`
8. `group-by`

## Notes

- Targets parser-combinator prerequisites in the FP base layer: composition/currying plus Maybe/Either composition patterns.
- Composition prompts emphasize directionality (`compose` vs `pipe`), monadic propagation behavior, and consecutive-key grouping semantics.
- Verify expressions run directly against `(require 'combinators)` during validation.

## Generate / Validate

```bash
python3 data/tier1-fp-meta-combinators/generate_fp_meta_combinators_sft.py
python3 data/tier1-fp-meta-combinators/validate_fp_meta_combinators_sft.py
```

Validation should end with `VALIDATION_OK`.
