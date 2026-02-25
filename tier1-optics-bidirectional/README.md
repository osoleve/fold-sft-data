# Tier1 Optics Bidirectional SFT Dataset

SFT dataset for `lattice/optics/bidirectional.ss`, focused on versioned migration construction, composition, direction dispatch, reversal, and law verification.

## Coverage

- Source module: `lattice/optics/bidirectional.ss`
- Source tests: `lattice/optics/test-bidirectional.ss`
- Source functions (10):
  - `make-migration`
  - `migrate`
  - `rollback`
  - `migration-apply`
  - `migration-compose`
  - `migration-chain`
  - `migration-flip`
  - `make-migration-from-functions`
  - `make-identity-migration`
  - `verify-migration-laws`

## Families

- `spec_to_code`: 30
- `translation`: 30
- `bugfix`: 20
- `composition`: 50
- Total: 130
- Split: 105 train / 25 eval

## Files

- `generate_bidirectional_sft.py`: deterministic generator
- `validate_bidirectional_sft.py`: schema + executable validator
- `all.jsonl`: complete dataset
- `train.jsonl`: train split
- `eval.jsonl`: eval split
- `summary.json`: aggregate counts
- `validation-report.json`: validator output

## Build

```bash
python3 data/tier1-optics-bidirectional/generate_bidirectional_sft.py
python3 data/tier1-optics-bidirectional/validate_bidirectional_sft.py
```

## Notes

- Prompt variation is generated via `user/sft/generate.ss` (grammar DSL).
- Generation is two-stage and deterministic:
  1. Python emits canonical rows with `prompt_body` to `.pre_diversify.jsonl`.
  2. DSL generation emits final diversified `prompt` while preserving metadata and split.
- Split assignment is leakage-aware via `data/sft_split_utils.py`.
- Translation includes Python->Fold, Chez->Fold, and source-excerpt->Fold tasks.
- Excerpt translation targets use doc-free canonical outputs to preserve executable core definitions.
- Validator checks:
  - schema and family/source invariants
  - tautological verify guards
  - near-duplicate prompt detection
  - composition `ground_truth`/`verify_expr` wrapper coherence
  - non-trivial Chez->Fold similarity thresholding
  - executable verification of all `verify_expr` checks
  - bugfix negative checks (buggy snippet must fail its verifier)
