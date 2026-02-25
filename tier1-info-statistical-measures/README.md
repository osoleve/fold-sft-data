# tier1-info-statistical-measures

SFT dataset for `lattice/info/statistical-measures.ss` and `lattice/info/test-statistical-measures.ss`.

## Contents

- `generate_statistical_measures_sft.py` - deterministic sample generator
- `validate_statistical_measures_sft.py` - schema + executable verification
- `all.jsonl` - all samples
- `train.jsonl` - train split
- `eval.jsonl` - eval split
- `summary.json` - counts and distributions
- `validation-report.json` - validator status/report

## Dataset shape

- Total: 80
- Families:
  - `spec_to_code`: 16
  - `translation`: 16
  - `bugfix`: 16
  - `composition`: 32
- Difficulty: 18 easy / 38 medium / 24 hard
- Split: 66 train / 14 eval
- Source functions: 8
  - `bhattacharyya-coefficient`
  - `bhattacharyya-distance`
  - `hellinger-distance`
  - `total-variation-distance`
  - `chi-squared-divergence`
  - `symmetric-chi-squared`
  - `jeffreys-divergence`
  - `alpha-divergence`

## Run

```bash
python3 data/tier1-info-statistical-measures/generate_statistical_measures_sft.py
python3 data/tier1-info-statistical-measures/validate_statistical_measures_sft.py
```

## Quality notes

- Deterministic prompt diversification is applied via `data/sft_prompt_diversity.py`.
- Verify expressions are self-contained closures with dependency closure over explicit roots plus symbol-scanned verify references.
- Composition prompts explicitly name the primary `source_function`, and composition expressions are checked for function-token alignment.
- Bugfix tasks focus on realistic single-point defects (sign errors, missing factors, branch-condition mistakes, swapped direction/formula terms).
