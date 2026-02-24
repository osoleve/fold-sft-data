# Fold SFT Data

Deterministic SFT dataset generators for teaching language models the [Fold](https://github.com/osoleve/fold) computational substrate — a content-addressed Scheme runtime with a verified mathematical skill lattice.

Each dataset targets a specific lattice module or skill tier. Generators produce four task families per module:

| Family | What It Trains |
|--------|---------------|
| `spec_to_code` | Implement a function from its behavioral spec |
| `translation` | Convert Python or Chez Scheme to Fold-native form |
| `bugfix` | Fix a single-point defect with minimal changes |
| `composition` | Compose module primitives into working expressions |

## Structure

```
sft_prompt_diversity.py          # Shared deterministic prompt diversification
<dataset-name>/
    generate_<name>_sft.py       # Deterministic generator
    validate_<name>_sft.py       # Schema + executable validator (runs Chez Scheme)
    all.jsonl                    # Full dataset
    train.jsonl                  # Train split (~82%)
    eval.jsonl                   # Eval split (~18%)
    summary.json                 # Family/split/difficulty/function counts
    validation-report.json       # Validator output
    README.md                    # Dataset-specific documentation
```

## Methodology

General build/maintenance/expansion playbook for these datasets:

- [`SFT_DATASET_METHODOLOGY.md`](SFT_DATASET_METHODOLOGY.md)

## Datasets

### Core Base
| Dataset | Samples | Functions | Source Module |
|---------|---------|-----------|--------------|
| `core-base-prelude-unified` | 212 | 24 | `core/base/prelude.ss` |
| `core-base-error` | 100 | 12 | `core/base/error.ss` |
| `core-base-sha256` | 54 | 6 | `core/base/sha256.ss` |
| `core-base-span` | 54 | 6 | `core/base/span.ss` |

### Data Structures (Tier 0)
| Dataset | Samples | Functions | Source Module |
|---------|---------|-----------|--------------|
| `tier0-data-alist` | 80 | 8 | `lattice/data/alist.ss` |
| `tier0-data-avl-tree` | 80 | 8 | `lattice/data/avl-tree.ss` |
| `tier0-data-collection-protocol` | 80 | 8 | `lattice/data/collection-protocol.ss` |
| `tier0-data-collection-utils` | 80 | 8 | `lattice/data/collection-utils.ss` |
| `tier0-data-dict` | 118 | 12 | `lattice/data/dict.ss` |
| `tier0-data-hamt` | 80 | 8 | `lattice/data/hamt.ss` |
| `tier0-data-heap` | 80 | 8 | `lattice/data/heap.ss` |
| `tier0-data-queue` | 80 | 8 | `lattice/data/queue.ss` |
| `tier0-data-set` | 96 | 10 | `lattice/data/set.ss` |
| `tier0-data-sort` | 80 | 8 | `lattice/data/sort.ss` |
| `tier0-data-stack` | 72 | 8 | `lattice/data/stack.ss` |

### Linear Algebra (Tier 0)
| Dataset | Samples | Functions | Source Module |
|---------|---------|-----------|--------------|
| `tier0-linalg-integer-matrix` | 80 | 8 | `lattice/linalg/integer-matrix.ss` |
| `tier0-linalg-iteration` | 80 | 8 | `lattice/linalg/iteration.ss` |
| `tier0-linalg-matrix` | 80 | 12 | `lattice/linalg/matrix.ss` |
| `tier0-linalg-matrix-decomp` | 80 | 8 | `lattice/linalg/matrix-decomp.ss` |
| `tier0-linalg-numeric-instances` | 80 | 8 | `lattice/linalg/numeric-instances.ss` |
| `tier0-linalg-sparse` | 80 | 8 | `lattice/linalg/sparse.ss` |
| `tier0-linalg-vec` | 80 | 8 | `lattice/linalg/vec.ss` |

### Number Theory (Tier 0)
| Dataset | Samples | Functions | Source Module |
|---------|---------|-----------|--------------|
| `tier0-number-theory-fast-multiply` | 80 | 8 | `lattice/number-theory/fast-multiply.ss` |
| `tier0-number-theory-modular` | 140 | 14 | `lattice/number-theory/modular.ss` |
| `tier0-number-theory-primality` | 80 | 8 | `lattice/number-theory/primality.ss` |

### Meta / Knowledge Graph (Tier 1)
| Dataset | Samples | Functions | Source Module |
|---------|---------|-----------|--------------|
| `tier1-meta-kg-usage` | 80 | 8 | `lattice/meta/search.ss` |
| `tier1-meta-kg-ingest` | 80 | 8 | `lattice/meta/kg-ingest.ss` |

### Functional Programming (Tier 1)
| Dataset | Samples | Functions | Source Module |
|---------|---------|-----------|--------------|
| `tier1-fp-meta-combinators` | 80 | 8 | `lattice/fp/meta/combinators.ss` |
| `tier1-fp-parsing-fsm` | 80 | 8 | `lattice/fp/parsing/fsm.ss` |
| `tier1-fp-parsing-regex-parser` | 80 | 8 | `lattice/fp/parsing/regex.ss` |
| `tier1-fp-parsing-parser-compile` | 96 | 8 | `lattice/fp/parsing/parser-compile.ss` |

### Query / Info / Random / Egraph (Tier 1)
| Dataset | Samples | Functions | Source Module |
|---------|---------|-----------|--------------|
| `tier1-query-query-dsl` | 80 | 8 | `lattice/query/query-dsl.ss` |
| `tier1-query-patterns-parse` | 96 | 8 | `lattice/query/patterns-parse.ss` |
| `tier1-query-aho-corasick` | 96 | 8 | `lattice/query/aho-corasick.ss` |
| `tier1-random-prng` | 80 | 8 | `lattice/random/prng.ss` |
| `tier1-random-distributions` | 80 | 8 | `lattice/random/distributions.ss` |
| `tier1-random-probability` | 80 | 8 | `lattice/random/probability.ss` |
| `tier1-info-entropy` | 80 | 8 | `lattice/info/entropy.ss` |
| `tier1-info-model-selection` | 80 | 8 | `lattice/info/model-selection.ss` |
| `tier1-info-partition-info` | 80 | 8 | `lattice/info/partition-info.ss` |
| `tier1-info-statistical-measures` | 80 | 8 | `lattice/info/statistical-measures.ss` |
| `tier1-egraph-union-find` | 80 | 8 | `lattice/egraph/union-find.ss` |
| `tier1-egraph-egraph` | 80 | 8 | `lattice/egraph/egraph.ss` |

### Next Uncovered Targets
High-value uncovered modules with strong tests and compact APIs:

| Candidate Module | Why It Is A Good SFT Target |
|------------------|------------------------------|
| `lattice/egraph/match.ss` | Core e-graph reasoning primitive with rich substitution and rewrite behaviors. |
| `lattice/egraph/extract.ss` | Teaches extraction of cheapest terms from equivalence classes, pairing well with existing egraph datasets. |
| `lattice/info/empirical-info.ss` | Compact information-theoretic estimators useful for finite-sample modeling tasks. |
| `lattice/info/channel-capacity.ss` | Well-scoped Shannon-capacity computations with strong mathematical supervision signals. |

## Usage

```bash
# Generate a dataset
python3 <dataset>/generate_<name>_sft.py

# Validate (requires Chez Scheme)
python3 <dataset>/validate_<name>_sft.py

# Regenerate all
for d in */generate_*.py; do python3 "$d"; done

# Validate all
for d in */validate_*.py; do python3 "$d"; done
```

## Prompt Diversity

`sft_prompt_diversity.py` adds deterministic wording variation to prompts via SHA256-stable selection from:
- **Family prefixes**: 3 "Task mode:" openers per family
- **Category hints**: domain-appropriate guidance lines
- **Verify check blocks**: acceptance/regression checks extracted from `verify_expr`
- **Family suffixes**: closing instructions

All selection is deterministic — same inputs always produce the same prompt. ~64% of prompts include verify-derived check blocks; the rest omit them for natural variety.
