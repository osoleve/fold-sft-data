# Tier1 Info Model Selection SFT Dataset

- Source module: `lattice/info/model-selection.ss`
- Source tests: `lattice/info/test-model-selection.ss`
- Total samples: `80`
- Split: `66 train / 14 eval`
- Families: `16 spec_to_code / 16 translation / 16 bugfix / 32 composition`

## Covered Functions

1. `log-likelihood-gaussian`
2. `log-likelihood-gaussian-vec`
3. `aic`
4. `bic`
5. `aicc`
6. `aic-weights`
7. `evidence-ratio`
8. `residual-entropy-bits`

## Notes

- Focuses on information criteria and likelihood-based model ranking utilities.
- Verification closures are self-contained via dependency closure from:
  - implementation dependencies, and
  - symbols referenced directly inside `verify_expr`.
- Composition samples emphasize formula checks, edge cases, and cross-function consistency.

## Generate / Validate

```bash
python3 data/tier1-info-model-selection/generate_model_selection_sft.py
python3 data/tier1-info-model-selection/validate_model_selection_sft.py
```

Validation should end with `VALIDATION_OK`.
