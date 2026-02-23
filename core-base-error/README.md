# core/base/error SFT Dataset

Synthetic SFT data for `core/base/error.ss`.

## Scope
- Source module: `core/base/error.ss`
- Source tests: `core/base/test-error.ss`
- Families: `spec_to_code`, `translation`, `bugfix`, `composition`
- Total samples: 100

## Split
- Train: 80
- Eval: 20

## Family counts
- `spec_to_code`: 36
- `translation`: 18
- `bugfix`: 18
- `composition`: 28

## Files
- `generate_core_error_sft.py`: deterministic generator
- `validate_core_error_sft.py`: schema + split + executable verification
- `all.jsonl`: full dataset
- `train.jsonl`: train split
- `eval.jsonl`: eval split
- `summary.json`: generation summary
- `validation-report.json`: latest validation report

## Regenerate
```bash
python3 data/core-base-error/generate_core_error_sft.py
python3 data/core-base-error/validate_core_error_sft.py
```
