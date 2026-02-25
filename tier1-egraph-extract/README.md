# Tier1 EGraph Extract SFT Dataset

SFT dataset for `lattice/egraph/extract.ss`, focused on cost-based extraction and optimization over e-graphs.

## Coverage

- Source module: `lattice/egraph/extract.ss`
- Source tests: `lattice/egraph/test-extract.ss`
- Source functions (8):
  - `make-extraction-state`
  - `extraction-state?`
  - `extract`
  - `extract-term`
  - `extract-all`
  - `optimize`
  - `optimize-with-config`
  - `compare-extractions`

## Families

- `spec_to_code`: 24
- `translation`: 24
- `bugfix`: 16
- `composition`: 32
- Total: 96
- Split: 79 train / 17 eval

## Files

- `generate_egraph_extract_sft.py`: deterministic generator
- `validate_egraph_extract_sft.py`: schema + executable validator
- `all.jsonl`: complete dataset
- `train.jsonl`: train split
- `eval.jsonl`: eval split
- `summary.json`: aggregate counts
- `validation-report.json`: validator output

## Build

```bash
python3 data/tier1-egraph-extract/generate_egraph_extract_sft.py
python3 data/tier1-egraph-extract/validate_egraph_extract_sft.py
```

## Notes

- Prompt generation is two-stage and deterministic:
  1. Python generator builds canonical rows with `prompt_body` + metadata + deterministic `split`.
  2. `user/sft/generate.ss` consumes those rows and emits diversified `prompt` while preserving `prompt_body` and `split`.
- Split assignment is leakage-aware via `data/sft_split_utils.py`.
- Translation family includes doc-free target variants for source-excerpt translation prompts.
- Composition prompts avoid embedding raw `verify_expr` code to reduce answer leakage.
- Validator checks include:
  - bugfix negatives (buggy snippets must fail their `verify_expr`)
  - near-duplicate prompts
  - composition `ground_truth`/`verify_expr` coherence for wrapper-style checks
  - trivial Chez→Fold translation similarity (doc-stripped threshold)
