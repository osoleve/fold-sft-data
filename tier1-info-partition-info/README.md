# Tier1 Info Partition Info SFT Dataset

- Source module: `lattice/info/partition-info.ss`
- Source tests: `lattice/info/test-partition-info.ss`
- Total samples: `80`
- Split: `66 train / 14 eval`
- Families: `16 spec_to_code / 16 translation / 16 bugfix / 32 composition`

## Covered Functions

1. `partition-sizes`
2. `partition-entropy`
3. `partition-mi`
4. `partition-nmi`
5. `partition-vi`
6. `partition-vi-normalized`
7. `unique-labels`
8. `label-index-map`

## Notes

- Focuses on information-theoretic partition quality metrics and core label-table helpers.
- Verification closures are self-contained via dependency closure from:
  - implementation dependencies, and
  - symbols referenced directly inside `verify_expr`.
- Composition samples emphasize metric invariants (bounds, symmetry, identity cases) and helper correctness.

## Generate / Validate

```bash
python3 data/tier1-info-partition-info/generate_partition_info_sft.py
python3 data/tier1-info-partition-info/validate_partition_info_sft.py
```

Validation should end with `VALIDATION_OK`.
