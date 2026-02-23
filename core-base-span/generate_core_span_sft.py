#!/usr/bin/env python3
"""Generate SFT samples for core/base/span.ss."""

from __future__ import annotations

import json
import sys
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

OUT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(__file__).resolve().parents[1]
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from sft_prompt_diversity import diversify_prompt
ALL_PATH = OUT_DIR / "all.jsonl"
TRAIN_PATH = OUT_DIR / "train.jsonl"
EVAL_PATH = OUT_DIR / "eval.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.json"

SOURCE_MODULE = "core/base/span.ss"
SOURCE_TEST = "core/base/test-error.ss"

DEFS: Dict[str, str] = {
    "make-span": """(define (make-span file line column end-line end-column)
  (list 'span file line column end-line end-column))""",
    "span?": """(define (span? x)
  (and (pair? x) (eq? (car x) 'span)))""",
    "span-file": """(define (span-file s)
  (list-ref s 1))""",
    "span-line": """(define (span-line s)
  (list-ref s 2))""",
    "span-column": """(define (span-column s)
  (list-ref s 3))""",
    "span-end-line": """(define (span-end-line s)
  (list-ref s 4))""",
    "span-end-column": """(define (span-end-column s)
  (list-ref s 5))""",
    "merge-spans": """(define (merge-spans s1 s2)
  (make-span (span-file s1)
             (span-line s1)
             (span-column s1)
             (span-end-line s2)
             (span-end-column s2)))""",
    "format-span": """(define (format-span s)
  (if (span? s)
      (string-append (span-file s) ":"
                     (number->string (span-line s)) ":"
                     (number->string (span-column s)))
      "<unknown>"))""",
}

FUNCTION_ORDER = [
    "make-span",
    "span?",
    "span-file",
    "span-line",
    "span-column",
    "span-end-line",
    "span-end-column",
    "merge-spans",
    "format-span",
]

DEPENDS: Dict[str, List[str]] = {
    "make-span": [],
    "span?": [],
    "span-file": [],
    "span-line": [],
    "span-column": [],
    "span-end-line": [],
    "span-end-column": [],
    "merge-spans": ["make-span", "span-file", "span-line", "span-column", "span-end-line", "span-end-column"],
    "format-span": ["span?", "span-file", "span-line", "span-column"],
}

FUNCTION_SPECS = {
    "make-span": "Construct a span value as `(span file line column end-line end-column)`.",
    "span?": "Return #t iff value is a span tagged with symbol `span`.",
    "span-file": "Return file field from span.",
    "span-line": "Return line field from span.",
    "span-column": "Return column field from span.",
    "span-end-line": "Return end-line field from span.",
    "span-end-column": "Return end-column field from span.",
    "merge-spans": "Create a new span from start fields of first span and end fields of second span.",
    "format-span": "Format span as `file:line:column`, and `<unknown>` for non-span values.",
}

SKELETONS = {
    "make-span": """(define (make-span file line column end-line end-column)
  ;; TODO: build canonical span record
  <TODO>)""",
    "span?": """(define (span? x)
  ;; TODO: check whether x is tagged as span
  <TODO>)""",
    "span-file": """(define (span-file s)
  ;; TODO: extract file field
  <TODO>)""",
    "span-line": """(define (span-line s)
  ;; TODO: extract line field
  <TODO>)""",
    "span-column": """(define (span-column s)
  ;; TODO: extract column field
  <TODO>)""",
    "span-end-line": """(define (span-end-line s)
  ;; TODO: extract end-line field
  <TODO>)""",
    "span-end-column": """(define (span-end-column s)
  ;; TODO: extract end-column field
  <TODO>)""",
    "merge-spans": """(define (merge-spans s1 s2)
  ;; TODO: combine start from s1 with end from s2
  <TODO>)""",
    "format-span": """(define (format-span s)
  ;; TODO: string format for spans; fallback to <unknown>
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "make-span": "(equal? (make-span \"f.ss\" 2 3 2 8) '(span \"f.ss\" 2 3 2 8))",
    "span?": "(and (span? '(span \"f.ss\" 1 2 1 3)) (not (span? '(not-span x))))",
    "span-file": "(equal? (span-file (make-span \"main.ss\" 1 2 1 4)) \"main.ss\")",
    "span-line": "(equal? (span-line (make-span \"main.ss\" 7 2 7 4)) 7)",
    "span-column": "(equal? (span-column (make-span \"main.ss\" 7 9 7 14)) 9)",
    "span-end-line": "(equal? (span-end-line (make-span \"main.ss\" 7 9 8 3)) 8)",
    "span-end-column": "(equal? (span-end-column (make-span \"main.ss\" 7 9 8 12)) 12)",
    "merge-spans": "(let* ([s1 (make-span \"a.ss\" 2 3 2 7)] [s2 (make-span \"a.ss\" 4 1 4 5)] [m (merge-spans s1 s2)]) (and (equal? (span-file m) \"a.ss\") (= (span-line m) 2) (= (span-column m) 3) (= (span-end-line m) 4) (= (span-end-column m) 5)))",
    "format-span": "(and (equal? (format-span (make-span \"file.ss\" 10 4 10 8)) \"file.ss:10:4\") (equal? (format-span '(oops)) \"<unknown>\"))",
}

PYTHON_SNIPPETS = {
    "make-span": "def make_span(file, line, col, end_line, end_col):\n    return ('span', file, line, col, end_line, end_col)",
    "span?": "def is_span(x):\n    return isinstance(x, (list, tuple)) and len(x) > 0 and x[0] == 'span'",
    "span-file": "def span_file(s):\n    return s[1]",
    "span-line": "def span_line(s):\n    return s[2]",
    "span-column": "def span_column(s):\n    return s[3]",
    "span-end-line": "def span_end_line(s):\n    return s[4]",
    "span-end-column": "def span_end_column(s):\n    return s[5]",
    "merge-spans": "def merge_spans(s1, s2):\n    return make_span(span_file(s1), span_line(s1), span_column(s1), span_end_line(s2), span_end_column(s2))",
    "format-span": "def format_span(s):\n    if is_span(s):\n        return f\"{span_file(s)}:{span_line(s)}:{span_column(s)}\"\n    return '<unknown>'",
}

BUGGY_CASES = [
    {
        "fn": "make-span",
        "buggy": "(define (make-span file line column end-line end-column)\n  (list 'span line column end-line end-column file))",
        "note": "Field order is wrong; file must be second element.",
    },
    {
        "fn": "span?",
        "buggy": "(define (span? x)\n  (pair? x))",
        "note": "Any pair is not necessarily a span.",
    },
    {
        "fn": "span-file",
        "buggy": "(define (span-file s)\n  (list-ref s 2))",
        "note": "File is at index 1.",
    },
    {
        "fn": "span-line",
        "buggy": "(define (span-line s)\n  (list-ref s 1))",
        "note": "Line is at index 2, not index 1.",
    },
    {
        "fn": "span-column",
        "buggy": "(define (span-column s)\n  (list-ref s 4))",
        "note": "Column is index 3.",
    },
    {
        "fn": "span-end-line",
        "buggy": "(define (span-end-line s)\n  (span-line s))",
        "note": "End line must come from index 4.",
    },
    {
        "fn": "span-end-column",
        "buggy": "(define (span-end-column s)\n  (span-column s))",
        "note": "End column must come from index 5.",
    },
    {
        "fn": "merge-spans",
        "buggy": "(define (merge-spans s1 s2)\n  (make-span (span-file s2) (span-line s2) (span-column s2) (span-end-line s1) (span-end-column s1)))",
        "note": "Start fields must come from first span, end fields from second.",
    },
    {
        "fn": "format-span",
        "buggy": "(define (format-span s)\n  (if (span? s)\n      (span-file s)\n      \"\"))",
        "note": "Formatted span must include file, line, and column; non-span fallback is <unknown>.",
    },
]

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")
REQUIRED_KEYS = [
    "id",
    "family",
    "category",
    "difficulty",
    "source_module",
    "source_test",
    "source_function",
    "prompt",
    "ground_truth",
    "verify_expr",
    "tags",
]

samples: List[Dict[str, object]] = []
family_counter: Dict[str, int] = defaultdict(int)


def add_sample(
    family: str,
    category: str,
    difficulty: str,
    source_function: str,
    prompt: str,
    ground_truth: str,
    verify_expr: str,
    tags: List[str],
) -> None:
    family_counter[family] += 1
    sid = f"core_span_{family}_{family_counter[family]:03d}"
    sample = {
        "id": sid,
        "family": family,
        "category": category,
        "difficulty": difficulty,
        "source_module": SOURCE_MODULE,
        "source_test": SOURCE_TEST,
        "source_function": source_function,
        "prompt": diversify_prompt(prompt.strip(), family, source_function, family_counter[family], category, verify_expr),
        "ground_truth": ground_truth.strip(),
        "verify_expr": verify_expr.strip(),
        "tags": tags,
    }
    for k in REQUIRED_KEYS:
        if k not in sample:
            raise ValueError(f"missing key {k}")
    samples.append(sample)


def verify_refs(fn: str) -> List[str]:
    tokens = set(TOKEN_RE.findall(VERIFY_BY_FUNCTION[fn]))
    return [name for name in FUNCTION_ORDER if name != fn and name in tokens]


def dependency_closure(fn: str) -> List[str]:
    ordered: List[str] = []
    seen: Set[str] = set()

    def visit(name: str) -> None:
        for dep in DEPENDS.get(name, []):
            if dep not in seen:
                seen.add(dep)
                visit(dep)
                ordered.append(dep)

    for dep in DEPENDS.get(fn, []):
        if dep not in seen:
            seen.add(dep)
            visit(dep)
            ordered.append(dep)

    for dep in verify_refs(fn):
        if dep not in seen:
            seen.add(dep)
            visit(dep)
            ordered.append(dep)

    return ordered


def def_verify(fn: str) -> str:
    parts = [DEFS[d] for d in dependency_closure(fn)] + [DEFS[fn], VERIFY_BY_FUNCTION[fn]]
    body = "\n  ".join(parts)
    return f"(let ()\n  {body})"


# -----------------------------------------------------------------------------
# Families 1-3
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    diff = "easy" if fn in {"make-span", "span?", "span-file", "span-line", "span-column", "span-end-line", "span-end-column"} else "medium"

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=diff,
        source_function=fn,
        prompt=f"""You are implementing core source-location utilities in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["core", "base", "span", "spec-to-code", fn],
    )

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=diff,
        source_function=fn,
        prompt=f"""Complete this Fold Scheme skeleton.

```scheme
{SKELETONS[fn]}
```

Replace `<TODO>` and return only the completed definition for `{fn}`.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["core", "base", "span", "skeleton-completion", fn],
    )

for fn in FUNCTION_ORDER:
    diff = "easy" if fn in {"make-span", "span?", "span-file", "span-line", "span-column", "span-end-line", "span-end-column"} else "medium"
    add_sample(
        family="translation",
        category="translation",
        difficulty=diff,
        source_function=fn,
        prompt=f"""Translate the following Python function into Fold-native Scheme.
Preserve behavior exactly and use target function name `{fn}`.
Return only the Scheme definition.

```python
{PYTHON_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["core", "base", "span", "python-to-scheme", fn],
    )

for case in BUGGY_CASES:
    fn = str(case["fn"])
    diff = "easy" if fn in {"make-span", "span?", "span-file", "span-line", "span-column", "span-end-line", "span-end-column"} else "medium"
    add_sample(
        family="bugfix",
        category="debugging",
        difficulty=diff,
        source_function=fn,
        prompt=f"""Fix the bug in this Fold Scheme function with minimal semantic changes.
Target: `{fn}` in `{SOURCE_MODULE}`.
Known issue: {case['note']}

```scheme
{case['buggy']}
```

Return only the corrected definition.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["core", "base", "span", "bugfix", fn],
    )


# -----------------------------------------------------------------------------
# Family 4: composition
# -----------------------------------------------------------------------------

def add_composition(source_function: str, prompt: str, ground_truth: str, verify_expr: str, difficulty: str, extra_tags: List[str]) -> None:
    add_sample(
        family="composition",
        category="usage",
        difficulty=difficulty,
        source_function=source_function,
        prompt=prompt,
        ground_truth=ground_truth,
        verify_expr=verify_expr,
        tags=["core", "base", "span", "composition", source_function] + extra_tags,
    )


composition_cases = [
    ("make-span", "Construct span for file `a.ss` at 1:2 to 1:5.", "(make-span \"a.ss\" 1 2 1 5)", "(equal? (make-span \"a.ss\" 1 2 1 5) '(span \"a.ss\" 1 2 1 5))", "easy", ["direct"]),
    ("span?", "Check whether `(make-span \"a.ss\" 1 1 1 1)` is a span.", "(span? (make-span \"a.ss\" 1 1 1 1))", "(equal? (span? (make-span \"a.ss\" 1 1 1 1)) #t)", "easy", ["direct"]),
    ("span?", "Check whether symbol `'x` is a span.", "(span? 'x)", "(equal? (span? 'x) #f)", "easy", ["direct"]),
    ("span-file", "Extract file from `(make-span \"main.ss\" 4 2 4 9)`.", "(span-file (make-span \"main.ss\" 4 2 4 9))", "(equal? (span-file (make-span \"main.ss\" 4 2 4 9)) \"main.ss\")", "easy", ["direct"]),
    ("span-line", "Extract line from `(make-span \"main.ss\" 4 2 4 9)`.", "(span-line (make-span \"main.ss\" 4 2 4 9))", "(equal? (span-line (make-span \"main.ss\" 4 2 4 9)) 4)", "easy", ["direct"]),
    ("span-column", "Extract column from `(make-span \"main.ss\" 4 2 4 9)`.", "(span-column (make-span \"main.ss\" 4 2 4 9))", "(equal? (span-column (make-span \"main.ss\" 4 2 4 9)) 2)", "easy", ["direct"]),
    ("span-end-line", "Extract end line from `(make-span \"main.ss\" 4 2 7 9)`.", "(span-end-line (make-span \"main.ss\" 4 2 7 9))", "(equal? (span-end-line (make-span \"main.ss\" 4 2 7 9)) 7)", "easy", ["direct"]),
    ("span-end-column", "Extract end column from `(make-span \"main.ss\" 4 2 7 9)`.", "(span-end-column (make-span \"main.ss\" 4 2 7 9))", "(equal? (span-end-column (make-span \"main.ss\" 4 2 7 9)) 9)", "easy", ["direct"]),
    ("merge-spans", "Merge spans `(a.ss 2:3-2:8)` and `(a.ss 5:1-5:6)`.", "(merge-spans (make-span \"a.ss\" 2 3 2 8) (make-span \"a.ss\" 5 1 5 6))", "(equal? (merge-spans (make-span \"a.ss\" 2 3 2 8) (make-span \"a.ss\" 5 1 5 6)) '(span \"a.ss\" 2 3 5 6))", "medium", ["direct"]),
    ("format-span", "Format `(make-span \"file.ss\" 10 4 10 8)`.", "(format-span (make-span \"file.ss\" 10 4 10 8))", "(equal? (format-span (make-span \"file.ss\" 10 4 10 8)) \"file.ss:10:4\")", "medium", ["direct"]),
    ("format-span", "Format non-span value `'oops`.", "(format-span 'oops)", "(equal? (format-span 'oops) \"<unknown>\")", "easy", ["edge-case"]),
    ("merge-spans", "Return #t iff merge keeps start line from first span.", "(= (span-line (merge-spans (make-span \"x.ss\" 3 2 3 7) (make-span \"x.ss\" 9 1 9 2))) 3)", "(equal? (= (span-line (merge-spans (make-span \"x.ss\" 3 2 3 7) (make-span \"x.ss\" 9 1 9 2))) 3) #t)", "medium", ["property"]),
    ("merge-spans", "Return #t iff merge keeps end column from second span.", "(= (span-end-column (merge-spans (make-span \"x.ss\" 3 2 3 7) (make-span \"x.ss\" 9 1 9 42))) 42)", "(equal? (= (span-end-column (merge-spans (make-span \"x.ss\" 3 2 3 7) (make-span \"x.ss\" 9 1 9 42))) 42) #t)", "medium", ["property"]),
    ("span-file", "Get merged span file from two same-file spans.", "(span-file (merge-spans (make-span \"k.ss\" 1 1 1 2) (make-span \"k.ss\" 2 1 2 2)))", "(equal? (span-file (merge-spans (make-span \"k.ss\" 1 1 1 2) (make-span \"k.ss\" 2 1 2 2))) \"k.ss\")", "medium", ["integration"]),
    ("span-line", "Build one span and return `(list line col end-col)`.", "(let ([s (make-span \"m.ss\" 8 6 9 1)]) (list (span-line s) (span-column s) (span-end-column s)))", "(equal? (let ([s (make-span \"m.ss\" 8 6 9 1)]) (list (span-line s) (span-column s) (span-end-column s))) '(8 6 1))", "medium", ["integration"]),
    ("span-end-column", "Map end columns over two spans.", "(map span-end-column (list (make-span \"a\" 1 1 1 5) (make-span \"b\" 2 2 2 9)))", "(equal? (map span-end-column (list (make-span \"a\" 1 1 1 5) (make-span \"b\" 2 2 2 9))) '(5 9))", "medium", ["list"]),
    ("format-span", "Map format-span over span and non-span values.", "(map format-span (list (make-span \"f\" 1 2 1 3) 'x))", "(equal? (map format-span (list (make-span \"f\" 1 2 1 3) 'x)) '(\"f:1:2\" \"<unknown>\"))", "medium", ["list"]),
    ("make-span", "Return #t iff all accessors recover fields used at construction.", "(let ([s (make-span \"z.ss\" 11 7 12 9)]) (and (equal? (span-file s) \"z.ss\") (= (span-line s) 11) (= (span-column s) 7) (= (span-end-line s) 12) (= (span-end-column s) 9)))", "(equal? (let ([s (make-span \"z.ss\" 11 7 12 9)]) (and (equal? (span-file s) \"z.ss\") (= (span-line s) 11) (= (span-column s) 7) (= (span-end-line s) 12) (= (span-end-column s) 9))) #t)", "medium", ["property"]),
]

for fn, prompt, gt, verify, diff, tags in composition_cases:
    add_composition(fn, prompt, gt, verify, diff, tags)

if len([s for s in samples if s["family"] == "composition"]) != 18:
    raise ValueError("composition family must contain exactly 18 samples")
if len(samples) != 54:
    raise ValueError(f"expected 54 samples, got {len(samples)}")


# -----------------------------------------------------------------------------
# Split
# -----------------------------------------------------------------------------
by_family: Dict[str, List[Dict[str, object]]] = defaultdict(list)
for s in samples:
    by_family[str(s["family"])].append(s)

EVAL_QUOTA = {
    "spec_to_code": 3,
    "translation": 2,
    "bugfix": 2,
    "composition": 4,
}


def spread_indices(n: int, k: int) -> Set[int]:
    if k <= 0:
        return set()
    if k >= n:
        return set(range(n))
    if k == 1:
        return {n // 2}
    idxs = {round(i * (n - 1) / (k - 1)) for i in range(k)}
    cursor = 0
    while len(idxs) < k:
        if cursor not in idxs:
            idxs.add(cursor)
        cursor += 1
    return idxs


eval_ids: Set[str] = set()
for fam, fam_samples in by_family.items():
    picked = spread_indices(len(fam_samples), EVAL_QUOTA[fam])
    for i, s in enumerate(fam_samples):
        if i in picked:
            eval_ids.add(str(s["id"]))

id_to_sample: Dict[str, Dict[str, object]] = {str(s["id"]): s for s in samples}
all_source_functions = sorted({str(s["source_function"]) for s in samples})


def eval_source_fn_counts(ids: Set[str]) -> Counter:
    return Counter(str(id_to_sample[sid]["source_function"]) for sid in ids)


changed = True
while changed:
    changed = False
    fn_counts = eval_source_fn_counts(eval_ids)
    missing_fns = [fn for fn in all_source_functions if fn_counts[fn] == 0]
    if not missing_fns:
        break

    for fn in missing_fns:
        candidates = [s for s in samples if str(s["source_function"]) == fn and str(s["id"]) not in eval_ids]
        swapped = False
        for cand in candidates:
            fam = str(cand["family"])
            fam_eval = [id_to_sample[sid] for sid in eval_ids if str(id_to_sample[sid]["family"]) == fam]
            removable = [r for r in fam_eval if fn_counts[str(r["source_function"])] > 1]
            if not removable:
                continue
            removable.sort(key=lambda r: (fn_counts[str(r["source_function"])] , str(r["id"])), reverse=True)
            out = removable[0]
            eval_ids.remove(str(out["id"]))
            eval_ids.add(str(cand["id"]))
            changed = True
            swapped = True
            break
        if swapped:
            break

missing_after = [fn for fn in all_source_functions if eval_source_fn_counts(eval_ids)[fn] == 0]
if missing_after:
    raise ValueError(f"eval split is missing source functions: {missing_after}")

train_rows: List[Dict[str, object]] = []
eval_rows: List[Dict[str, object]] = []
for s in samples:
    row = dict(s)
    if s["id"] in eval_ids:
        row["split"] = "eval"
        eval_rows.append(row)
    else:
        row["split"] = "train"
        train_rows.append(row)

if len(train_rows) != 43 or len(eval_rows) != 11:
    raise ValueError(f"split mismatch: train={len(train_rows)}, eval={len(eval_rows)}")


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


write_jsonl(ALL_PATH, [dict(s, split=("eval" if s["id"] in eval_ids else "train")) for s in samples])
write_jsonl(TRAIN_PATH, train_rows)
write_jsonl(EVAL_PATH, eval_rows)

summary = {
    "total": len(samples),
    "train": len(train_rows),
    "eval": len(eval_rows),
    "families": {
        fam: {
            "total": len(fam_samples),
            "eval": sum(1 for s in fam_samples if s["id"] in eval_ids),
            "train": sum(1 for s in fam_samples if s["id"] not in eval_ids),
        }
        for fam, fam_samples in sorted(by_family.items())
    },
}

with SUMMARY_PATH.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print(f"Wrote: {ALL_PATH}")
print(f"Wrote: {TRAIN_PATH}")
print(f"Wrote: {EVAL_PATH}")
