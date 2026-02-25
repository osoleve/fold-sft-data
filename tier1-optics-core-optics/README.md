# Tier1 Optics Core SFT Dataset

SFT dataset for `lattice/optics/optics.ss`, focused on core optic construction/composition semantics.

## Coverage

- Source module: `lattice/optics/optics.ss`
- Source tests: `lattice/optics/test-optics.ss`
- Source functions (8):
  - `make-iso`
  - `iso-over`
  - `iso-compose`
  - `prism-over`
  - `affine-compose`
  - `traversal-compose`
  - `fold-preview`
  - `optic-compose`

## Families

- `spec_to_code`: 24
- `translation`: 24
- `bugfix`: 16
- `composition`: 40
- Total: 104
- Split: generated leakage-aware (train/eval sizes are deterministic for a fixed dataset content)

## Files

- `generate_optics_core_sft.py`: deterministic generator
- `validate_optics_core_sft.py`: schema + executable validator
- `all.jsonl`: complete dataset
- `train.jsonl`: train split
- `eval.jsonl`: eval split
- `summary.json`: aggregate counts
- `validation-report.json`: validator output

## Build

```bash
python3 data/tier1-optics-core-optics/generate_optics_core_sft.py
python3 data/tier1-optics-core-optics/validate_optics_core_sft.py
```

## Notes

- Prompt variation is deterministic via `user/sft/generate.ss` (grammar DSL).
- Generation is deterministic and two-stage:
  1. Python generator writes canonical `prompt_body` + labels to `.pre_diversify.jsonl`.
  2. `user/sft/generate.ss` rewrites prompts into stylistic variants and emits final `prompt`.
- Split assignment is leakage-aware via `data/sft_split_utils.py`.
- Translation includes `source-excerpt-to-fold` tasks with doc-free targets and `chez-to-fold` tasks.
- Composition prompts intentionally omit raw verifier snippets to reduce answer leakage from prompt text.
- Validator checks:
  - schema/shape invariants and tautology guards
  - near-duplicate prompt detection
  - composition `ground_truth`/`verify_expr` coherence for wrapper-style checks
  - non-trivial Chez→Fold translation similarity thresholding
  - executable verification of all `verify_expr` checks
  - bugfix negative checks (buggy snippets must fail their `verify_expr`)
