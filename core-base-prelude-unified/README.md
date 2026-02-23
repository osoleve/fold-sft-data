# core/base/prelude (Unified) SFT Dataset

Consolidated SFT data for `core/base/prelude.ss`.

This merges:
- `data/core-base-prelude` (initial 26-function set)
- `data/core-base-prelude-extended` (additional 13-function set)

into a single dataset with one deterministic split.

## Scope
- Source module: `core/base/prelude.ss`
- Source tests: `core/base/test-prelude.ss`
- Covered functions: 39
- Families: `spec_to_code`, `translation`, `bugfix`, `composition`
- Total samples: 212

## Split
- Train: 166
- Eval: 46

## Family counts
- `spec_to_code`: 78
- `translation`: 39
- `bugfix`: 39
- `composition`: 56

## Difficulty
- Easy: 116
- Medium: 80
- Hard: 16

## Files
- `generate_core_prelude_unified_sft.py`: merges source datasets and creates unified split
- `validate_core_prelude_unified_sft.py`: schema + split + executable verification
- `all.jsonl`: full unified dataset
- `train.jsonl`: train split
- `eval.jsonl`: eval split
- `summary.json`: generation summary
- `validation-report.json`: latest validation report

## Regenerate
```bash
python3 data/core-base-prelude-unified/generate_core_prelude_unified_sft.py
python3 data/core-base-prelude-unified/validate_core_prelude_unified_sft.py
```
