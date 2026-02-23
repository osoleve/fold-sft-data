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
| `tier0-linalg-matrix` | 80 | 12 | `lattice/linalg/matrix.ss` |
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
