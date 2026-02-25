# Tier1 Optics Profunctor SFT Dataset

- Source module: `lattice/optics/profunctor-optics.ss`
- Source tests: `lattice/optics/test-profunctor-optics.ss`
- Total samples: `80`
- Split: `66 train / 14 eval`
- Families: `16 spec_to_code / 16 translation / 16 bugfix / 32 composition`

## Covered Functions

1. `make-profunctor`
2. `profunctor?`
3. `profunctor-dimap`
4. `profunctor-lmap`
5. `profunctor-rmap`
6. `dimap`
7. `lmap`
8. `rmap`

## Notes

- Focuses on the core profunctor API and its derived dispatch wrappers.
- Verification closures are self-contained via dependency closure from:
  - implementation dependencies, and
  - symbols referenced directly inside `verify_expr`.
- Composition tasks emphasize law-style consistency checks (`lmap`/`rmap` vs `dimap`) and higher-order function behavior.

## Generate / Validate

```bash
python3 data/tier1-optics-profunctor-optics/generate_profunctor_optics_sft.py
python3 data/tier1-optics-profunctor-optics/validate_profunctor_optics_sft.py
```

Validation should end with `VALIDATION_OK`.
