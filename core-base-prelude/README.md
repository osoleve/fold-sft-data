# core/base/prelude SFT Dataset

Synthetic SFT data for a focused subset of `core/base/prelude.ss`.

## Scope
- Source module: `core/base/prelude.ss`
- Source tests: `core/base/test-prelude.ss`
- Covered functions: 26 (`andmap`, `ormap`, `filter`, `filter-map`, `fold-left`, `fold-right`, `zip`, `iota`, `range`, `take`, `drop`, `find`, `last`, `init`, `replicate`, `span`, `break`, `sum`, `product`, `mean`, `identity`, `flatten`, `append-map`, `partition`, `group-by`, `distinct-by`)
- Families: `spec_to_code`, `translation`, `bugfix`, `composition`
- Total samples: 134

## Split
- Train: 104
- Eval: 30

## Family counts
- `spec_to_code`: 52
- `translation`: 26
- `bugfix`: 26
- `composition`: 30

## Files
- `generate_core_prelude_sft.py`: deterministic generator
- `validate_core_prelude_sft.py`: schema + split + executable verification
- `all.jsonl`: full dataset
- `train.jsonl`: train split
- `eval.jsonl`: eval split
- `summary.json`: generation summary
- `validation-report.json`: latest validation report

## Regenerate
```bash
python3 data/core-base-prelude/generate_core_prelude_sft.py
python3 data/core-base-prelude/validate_core_prelude_sft.py
```
