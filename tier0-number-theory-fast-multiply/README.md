# Tier 0 SFT: Number Theory / Fast Multiply (Core)

This folder contains a curated Tier-0 SFT dataset for core limb arithmetic in:
- `lattice/number-theory/fast-multiply.ss`

## Files
- `generate_fast_multiply_sft.py`: deterministic dataset generator
- `validate_fast_multiply_sft.py`: schema + executable verifier
- `all.jsonl`: complete dataset with split labels
- `train.jsonl`: training split
- `eval.jsonl`: evaluation split
- `summary.json`: family/split counts
- `validation-report.json`: validator output

## Task Families
- `spec_to_code`: implement function definitions from contracts/skeletons
- `translation`: translate Python/Chez snippets into Fold-native Scheme
- `bugfix`: fix realistic arithmetic/carry/borrow defects
- `composition`: compose limb APIs for conversions and multiplication tasks

## Covered Functions
- `limbs->integer`
- `integer->limbs`
- `limbs-normalize`
- `limbs-add`
- `limbs-sub`
- `limbs-shift`
- `limb-scale`
- `limbs-multiply-schoolbook`

## Generation
```bash
python3 data/tier0-number-theory-fast-multiply/generate_fast_multiply_sft.py
```

## Validation
```bash
python3 data/tier0-number-theory-fast-multiply/validate_fast_multiply_sft.py
```

The validator checks schema, duplicate IDs/prompts, split integrity, and executes every `verify_expr` in Chez Scheme.
