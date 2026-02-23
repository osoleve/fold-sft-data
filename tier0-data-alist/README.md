# Tier 0 SFT: Data / Alist

This folder contains a curated Tier-0 SFT dataset for:
- `lattice/data/alist.ss`

## Files
- `generate_alist_sft.py`: deterministic dataset generator
- `validate_alist_sft.py`: schema + executable verifier
- `all.jsonl`: complete dataset with split labels
- `train.jsonl`: training split
- `eval.jsonl`: evaluation split
- `summary.json`: family/split counts
- `validation-report.json`: validator output

## Task Families
- `spec_to_code`: implement functions from API contracts/skeletons
- `translation`: translate Python/Chez snippets into Fold-native Scheme
- `bugfix`: repair realistic single-function defects
- `composition`: solve usage tasks by composing alist APIs

## Generation
```bash
python3 data/tier0-data-alist/generate_alist_sft.py
```

## Validation
```bash
python3 data/tier0-data-alist/validate_alist_sft.py
```

The validator checks schema, duplicate IDs/prompts, split integrity, and executes every `verify_expr` in Chez Scheme.
