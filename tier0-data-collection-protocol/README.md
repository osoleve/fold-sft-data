# tier0-data-collection-protocol

SFT dataset for `lattice/data/collection-protocol.ss` with executable behavior through `lattice/data/collection-impl.ss`.

## Contents

- `generate_collection_protocol_sft.py` - deterministic sample generator
- `validate_collection_protocol_sft.py` - schema + executable verification
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
- Difficulty: 17 easy / 45 medium / 18 hard
- Split: 66 train / 14 eval
- Source functions: 8 (`coll-count`, `coll-any?`, `coll-all?`, `coll-map-list`, `coll-filter-list`, `coll-find`, `coll-partition`, `coll-protocols`)

## Run

```bash
python3 data/tier0-data-collection-protocol/generate_collection_protocol_sft.py
python3 data/tier0-data-collection-protocol/validate_collection_protocol_sft.py
```

## Quality notes

- Verify expressions are self-contained via dependency closure over both function deps and symbols referenced directly in verification checks.
- Tasks exercise protocol-dispatch behavior against concrete implementations (AVL, heap, kdtree, quadtree) through `collection-impl`.
- Bugfix samples target dispatch polarity, predicate composition, and accumulator threading rather than placeholder stubs.
