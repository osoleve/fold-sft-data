# Tier1 Random Distributions SFT Dataset

- Source module: `lattice/random/distributions.ss`
- Source tests: `lattice/random/test-distributions.ss`
- Total samples: `80`
- Split: `66 train / 14 eval`
- Families: `16 spec_to_code / 16 translation / 16 bugfix / 32 composition`

## Covered Functions

1. `horner-eval`
2. `safe-log`
3. `standard-normal-cdf`
4. `standard-normal-quantile`
5. `uniform-cdf`
6. `exponential-quantile`
7. `poisson-pmf`
8. `binomial-pmf`

## Notes

- Focuses on deterministic distribution math and boundary conditions.
- Verification closures are self-contained via dependency closure from:
  - implementation dependencies, and
  - symbols referenced directly inside `verify_expr`.
- Bugfix tasks use single-point defects (formula term, branch behavior, boundary guard).

## Generate / Validate

```bash
python3 data/tier1-random-distributions/generate_distributions_sft.py
python3 data/tier1-random-distributions/validate_distributions_sft.py
```

Validation should end with `VALIDATION_OK`.
