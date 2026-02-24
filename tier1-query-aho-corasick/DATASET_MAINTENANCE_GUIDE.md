# Tier1 Query Aho-Corasick Dataset: Build and Maintenance Guide

This guide explains how to create, maintain, and expand:

- Dataset: `data/tier1-query-aho-corasick`
- Source module: `lattice/query/aho-corasick.ss`
- Source tests: `lattice/query/test-aho-corasick.ss`

It is written for maintainers who need to update this dataset safely as the source module evolves.

## 1. Dataset Contract

The dataset is deterministic and must preserve these invariants unless intentionally changed:

- Total samples: no fixed cap (current build: `96`)
- Family split:
  - `spec_to_code`: minimum `16` (current: `24`)
  - `translation`: minimum `16` (current: `24`)
  - `bugfix`: `16`
  - `composition`: `32`
- Train/eval split: deterministic, dynamic per-family quota (~18% eval, minimum floors) with source-function coverage enforcement
- Source function coverage: each covered function appears in eval at least once
- Output files:
  - `all.jsonl`
  - `train.jsonl`
  - `eval.jsonl`
  - `summary.json`
  - `validation-report.json`

## 2. Covered Functions

Current function set (8):

1. `build-trie`
2. `insert-pattern-mut!`
3. `compute-failures`
4. `bfs-mut!`
5. `find-fail`
6. `get-next`
7. `make-automaton`
8. `search`

When source module behavior changes, update this function list only if coverage should change.

## 3. Files and Roles

- `generate_query_aho_corasick_sft.py`
  - Declares canonical solutions (`DEFS`)
  - Defines sample families
  - Performs deterministic split logic
  - Writes JSONL artifacts and summary
- `validate_query_aho_corasick_sft.py`
  - Validates schema and split integrity
  - Executes all `verify_expr` checks in Scheme
  - Writes `validation-report.json`
- `README.md`
  - Dataset summary for discoverability
- `DATASET_MAINTENANCE_GUIDE.md` (this file)
  - Operational instructions and QA process

## 4. Prerequisites

From repo root (`/home/osoleve/fold`):

- `python3`
- Chez Scheme executable `scheme`
- Working module/test files under `lattice/query/`

## 5. Standard Build Workflow

Run from repo root:

```bash
python3 data/tier1-query-aho-corasick/generate_query_aho_corasick_sft.py
python3 data/tier1-query-aho-corasick/validate_query_aho_corasick_sft.py
scheme --quiet --script lattice/query/test-aho-corasick.ss
```

Expected outcomes:

- Generator prints totals and writes artifacts
- Validator ends with `VALIDATION_OK`
- Module tests report `[PASS] All tests passed!`

## 6. How to Add or Modify Samples

Use these rules to avoid low-signal data.

### 6.1 `spec_to_code`

- Ground truth must be a complete `(define (...))`.
- Prompt should include function name + concise behavior contract.
- Keep function body identical to source behavior, not stylistic rewrites.

### 6.2 `translation`

- Python/Chez snippets must be semantically equivalent to target function.
- Avoid pseudo-code. Snippets must be executable-looking and precise.
- Ground truth remains the canonical implementation from `DEFS`.

### 6.3 `bugfix`

- Exactly one meaningful defect per sample.
- Bug should be realistic and local (minimal fix possible).
- Avoid syntax errors unless syntax repair is the intended skill.
- Every buggy sample should fail the function's `verify_expr`.

### 6.4 `composition`

- Should exercise API usage rather than re-implementing internals.
- Prefer concise expressions that test behavior intersections:
  - overlap matching
  - failure fallback
  - ordering/index correctness
  - inherited outputs via fail links

## 7. `verify_expr` Design Rules

`verify_expr` is the primary quality gate. Keep it strict and behavior-focused.

- Must evaluate to `#t` for correct implementation.
- Must reject plausible wrong implementations.
- Should test at least one edge case where relevant:
  - empty patterns/text
  - overlapping patterns
  - fallback paths
- Avoid tautologies and overly weak checks.

### Good pattern

- Check multiple independent properties in one expression.
- Include at least one negative assertion (something that must *not* happen).

### Anti-patterns

- Checking only return type.
- Checking a single happy-path input.
- Validating values that buggy implementations can still satisfy.

## 8. Bugfix QA (Mandatory)

After any bugfix sample or verify change, run a semantic bugfix sweep to ensure every buggy variant fails verification.

Use this script from repo root:

```bash
python3 - <<'PY'
import importlib.util
import subprocess
import tempfile
from pathlib import Path

repo = Path('/home/osoleve/fold')
mod_path = repo / 'data/tier1-query-aho-corasick/generate_query_aho_corasick_sft.py'
spec = importlib.util.spec_from_file_location('gen_ac', mod_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

bad = []
for i, case in enumerate(mod.BUGGY_CASES, 1):
    fn = case['fn']
    verify = mod.VERIFY_BY_FUNCTION[fn]
    script = "\n".join([
        '(load "core/lang/module.ss")',
        "(require 'aho-corasick)",
        case['buggy'],
        f'(display (if (equal? {verify} #t) "PASS" "FAIL"))',
        '(newline)',
    ])
    with tempfile.NamedTemporaryFile('w', suffix='.ss', delete=False) as tf:
        tf.write(script)
        path = tf.name
    try:
        proc = subprocess.run(
            ['scheme', '--quiet', '--script', path],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=20,
        )
        status = (proc.stdout or '').strip().splitlines()
        status = status[-1] if status else 'EX'
    except subprocess.TimeoutExpired:
        status = 'TIMEOUT'
    finally:
        Path(path).unlink(missing_ok=True)
    if status != 'FAIL':
        bad.append((i, fn, status))

print("ALL_BUGGY_FAIL" if not bad else f"WEAK_CASES={bad}")
PY
```

Pass criteria:

- `ALL_BUGGY_FAIL`

If not, strengthen `verify_expr` and/or bug snippet semantics.

## 9. Expanding the Dataset

Expansion can mean:

- Adding more functions
- Increasing samples per family
- Adding new family templates per function

### 9.1 Adding a Function

1. Add function to:
   - `DEFS`
   - `FUNCTION_ORDER`
   - `FUNCTION_SPECS`
   - `SKELETONS`
   - `VERIFY_BY_FUNCTION`
   - `PYTHON_SNIPPETS`
   - `CHEZ_SNIPPETS`
2. Add at least two bugfix cases in `BUGGY_CASES`.
3. Add four composition cases for that function.
4. Regenerate and validate.
5. Ensure eval includes the new function (split code should enforce this).

### 9.2 Increasing Total Samples

- Keep family balance intentional.
- Update hard-coded checks:
  - expected composition count
  - expected total count
  - expected train/eval counts
- Adjust `EVAL_QUOTA` to keep evaluation coverage representative.

### 9.3 When Not to Expand

- If source module APIs are unstable.
- If verify expressions are not yet discriminative.
- If module tests are failing.

## 10. Maintenance Triggers

Update this dataset when any of these occur:

- `aho-corasick.ss` behavior changes
- `test-aho-corasick.ss` adds/removes behavior expectations
- supporting data structure APIs (`queue`, `dict`, `set`) change semantics affecting checks

Maintenance steps:

1. Diff source module and tests.
2. Update affected `DEFS`/snippets/verifies.
3. Regenerate artifacts.
4. Run validator and module tests.
5. Run bugfix QA sweep.
6. Update README if function coverage/count changed.

## 11. Troubleshooting

### Validator fails with exceptions

- Usually malformed Scheme in `ground_truth` or `verify_expr`.
- Re-run the failing verify snippet directly in a small script.

### Module tests pass but validator fails

- Usually weak assumptions in dataset verify checks or missing required imports in validator.

### Bugfix QA shows `PASS` on buggy sample

- Verify is too weak for that bug class.
- Add assertions that directly test the broken invariant.

### Split coverage error

- Ensure function appears in at least one eval candidate; if needed add more samples for that function.

## 12. Release Checklist

Before considering updates complete:

1. Generator runs and writes all artifacts.
2. Validator outputs `VALIDATION_OK`.
3. `lattice/query/test-aho-corasick.ss` passes.
4. Bugfix QA confirms all buggy variants fail.
5. README and `data/README.md` are consistent with actual counts/functions.
