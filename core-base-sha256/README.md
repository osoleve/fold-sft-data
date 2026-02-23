# core/base/sha256 SFT Dataset

Synthetic SFT data for `core/base/sha256.ss`.

## Scope
- Source module: `core/base/sha256.ss`
- Source tests: `core/base/test-sha256.ss`
- Covered functions: 9 (`iota`, `pad-message`, `make-schedule`, `compress`, `sha256`, `sha256-hex`, `hash->hex`, `hex->hash`, `hex-digit`)
- Omitted: `hash-block` (depends on external block address/serialization symbols)
- Families: `spec_to_code`, `translation`, `bugfix`, `composition`
- Total samples: 54

## Split
- Train: 43
- Eval: 11

## Family counts
- `spec_to_code`: 18
- `translation`: 9
- `bugfix`: 9
- `composition`: 18

## Files
- `generate_core_sha256_sft.py`: deterministic generator
- `validate_core_sha256_sft.py`: schema + split + executable verification
- `all.jsonl`: full dataset
- `train.jsonl`: train split
- `eval.jsonl`: eval split
- `summary.json`: generation summary
- `validation-report.json`: latest validation report

## Regenerate
```bash
python3 data/core-base-sha256/generate_core_sha256_sft.py
python3 data/core-base-sha256/validate_core_sha256_sft.py
```
