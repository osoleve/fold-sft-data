#!/usr/bin/env python3
"""Generate Tier-0 SFT samples for lattice/data/sort.ss."""

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

SOURCE_MODULE = "lattice/data/sort.ss"
SOURCE_TEST = "lattice/data/test-sort.ss"

DEFS: Dict[str, str] = {
    "merge": """(define (merge xs ys cmp)
  (cond
    [(null? xs) ys]
    [(null? ys) xs]
    [(not (cmp (car ys) (car xs)))
     (cons (car xs) (merge (cdr xs) ys cmp))]
    [else
     (cons (car ys) (merge xs (cdr ys) cmp))]))""",
    "split-at": """(define (split-at lst n)
  (if (or (= n 0) (null? lst))
      (cons '() lst)
      (let ([rest (split-at (cdr lst) (- n 1))])
        (cons (cons (car lst) (car rest))
              (cdr rest)))))""",
    "merge-sort-by": """(define (merge-sort-by cmp lst)
  (let ([len (length lst)])
    (if (<= len 1)
        lst
        (let* ([mid (quotient len 2)]
               [split (split-at lst mid)]
               [left (car split)]
               [right (cdr split)])
          (merge (merge-sort-by cmp left)
                 (merge-sort-by cmp right)
                 cmp)))))""",
    "partition": """(define (partition cmp pivot lst)
  (let loop ([xs lst] [lt '()] [eq '()] [gt '()])
    (if (null? xs)
        (list (reverse lt) (reverse eq) (reverse gt))
        (let ([x (car xs)])
          (cond
            [(cmp x pivot)
             (loop (cdr xs) (cons x lt) eq gt)]
            [(cmp pivot x)
             (loop (cdr xs) lt eq (cons x gt))]
            [else
             (loop (cdr xs) lt (cons x eq) gt)])))))""",
    "quicksort-by": """(define (quicksort-by cmp lst)
  (if (or (null? lst) (null? (cdr lst)))
      lst
      (let* ([pivot (car lst)]
             [parts (partition cmp pivot (cdr lst))]
             [lt (car parts)]
             [eq (cadr parts)]
             [gt (caddr parts)])
        (append (quicksort-by cmp lt)
                (cons pivot eq)
                (quicksort-by cmp gt)))))""",
    "insert-sorted": """(define (insert-sorted cmp x sorted)
  (cond
    [(null? sorted) (list x)]
    [(cmp x (car sorted)) (cons x sorted)]
    [else (cons (car sorted) (insert-sorted cmp x (cdr sorted)))]))""",
    "insertion-sort-by": """(define (insertion-sort-by cmp lst)
  (fold-left (lambda (sorted x) (insert-sorted cmp x sorted))
             '()
             lst))""",
    "nth-smallest": """(define (nth-smallest n lst)
  (if (null? lst)
      (error 'nth-smallest "List is empty")
      (let* ([pivot (car lst)]
             [parts (partition < pivot (cdr lst))]
             [lt (car parts)]
             [eq (cadr parts)]
             [gt (caddr parts)]
             [lt-len (length lt)])
        (cond
          [(< n lt-len)
           (nth-smallest n lt)]
          [(< n (+ lt-len 1 (length eq)))
           pivot]
          [else
           (nth-smallest (- n lt-len 1 (length eq)) gt)]))))""",
}

DEPENDS: Dict[str, List[str]] = {
    "merge": [],
    "split-at": [],
    "merge-sort-by": ["split-at", "merge"],
    "partition": [],
    "quicksort-by": ["partition"],
    "insert-sorted": [],
    "insertion-sort-by": ["insert-sorted"],
    "nth-smallest": ["partition"],
}

FUNCTION_ORDER = [
    "merge",
    "split-at",
    "merge-sort-by",
    "partition",
    "quicksort-by",
    "insert-sorted",
    "insertion-sort-by",
    "nth-smallest",
]

FUNCTION_SPECS = {
    "merge": "Merge two already-sorted lists under comparator cmp, preserving stability when keys are equal.",
    "split-at": "Split lst at index n and return (cons left right) where left has at most n elements.",
    "merge-sort-by": "Stable merge sort with custom comparator cmp.",
    "partition": "Partition lst around pivot into (lt eq gt) buckets according to cmp.",
    "quicksort-by": "Quicksort with custom comparator cmp using three-way partitioning.",
    "insert-sorted": "Insert x into a sorted list while preserving order.",
    "insertion-sort-by": "Sort lst by repeatedly inserting elements with comparator cmp.",
    "nth-smallest": "Return the n-th smallest element (0-indexed); raise an error on empty input.",
}

SKELETONS = {
    "merge": """(define (merge xs ys cmp)
  ;; TODO: merge two sorted lists, taking from xs when keys are equal
  <TODO>)""",
    "split-at": """(define (split-at lst n)
  ;; TODO: return (cons left right) where left contains first n elements
  <TODO>)""",
    "merge-sort-by": """(define (merge-sort-by cmp lst)
  ;; TODO: stable merge sort using split-at and merge
  <TODO>)""",
    "partition": """(define (partition cmp pivot lst)
  ;; TODO: return (list lt eq gt)
  <TODO>)""",
    "quicksort-by": """(define (quicksort-by cmp lst)
  ;; TODO: sort using partition and recursive quicksort
  <TODO>)""",
    "insert-sorted": """(define (insert-sorted cmp x sorted)
  ;; TODO: insert x into sorted position
  <TODO>)""",
    "insertion-sort-by": """(define (insertion-sort-by cmp lst)
  ;; TODO: fold through lst and insert each value into accumulator
  <TODO>)""",
    "nth-smallest": """(define (nth-smallest n lst)
  ;; TODO: quickselect-style n-th order statistic
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "merge": "(let* ([xs '((a . 1) (b . 2) (c . 2))] [ys '((d . 2) (e . 3))] [m (merge xs ys (lambda (u v) (< (cdr u) (cdr v))))]) (equal? (map car m) '(a b c d e)))",
    "split-at": "(and (equal? (split-at '(1 2 3 4) 2) (cons '(1 2) '(3 4))) (equal? (split-at '(1 2) 5) (cons '(1 2) '())))",
    "merge-sort-by": "(and (equal? (merge-sort-by < '(5 2 8 1 9 3)) '(1 2 3 5 8 9)) (equal? (map car (merge-sort-by (lambda (x y) (< (cdr x) (cdr y))) '((a . 2) (b . 1) (c . 2)))) '(b a c)))",
    "partition": "(let ([parts (partition < 3 '(1 3 2 4 3 5))]) (and (equal? (car parts) '(1 2)) (equal? (cadr parts) '(3 3)) (equal? (caddr parts) '(4 5))))",
    "quicksort-by": "(and (equal? (quicksort-by < '(3 1 4 1 5 9)) '(1 1 3 4 5 9)) (equal? (quicksort-by > '(3 1 4 1 5)) '(5 4 3 1 1)))",
    "insert-sorted": "(and (equal? (insert-sorted < 3 '(1 2 4 5)) '(1 2 3 4 5)) (equal? (insert-sorted < 2 '(1 2 2 3)) '(1 2 2 2 3)))",
    "insertion-sort-by": "(and (equal? (insertion-sort-by < '(5 2 8 1 9 3)) '(1 2 3 5 8 9)) (equal? (insertion-sort-by > '(3 1 4 1 5)) '(5 4 3 1 1)))",
    "nth-smallest": "(and (= (nth-smallest 0 '(5 2 8 1 9 3 7)) 1) (= (nth-smallest 3 '(5 2 8 1 9 3 7)) 5) (= (nth-smallest 2 '(3 1 2 1 3)) 2) (guard (ex [else #t]) (begin (nth-smallest 0 '()) #f)))",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")
TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "merge": "def merge(xs, ys, cmp):\n    if not xs:\n        return ys\n    if not ys:\n        return xs\n    if not cmp(ys[0], xs[0]):\n        return [xs[0]] + merge(xs[1:], ys, cmp)\n    return [ys[0]] + merge(xs, ys[1:], cmp)",
    "split-at": "def split_at(lst, n):\n    if n == 0 or not lst:\n        return ([], lst)\n    left, right = split_at(lst[1:], n - 1)\n    return ([lst[0]] + left, right)",
    "merge-sort-by": "def merge_sort_by(cmp, lst):\n    if len(lst) <= 1:\n        return lst\n    mid = len(lst) // 2\n    left, right = split_at(lst, mid)\n    return merge(merge_sort_by(cmp, left), merge_sort_by(cmp, right), cmp)",
    "partition": "def partition(cmp, pivot, lst):\n    lt, eq, gt = [], [], []\n    for x in lst:\n        if cmp(x, pivot):\n            lt.append(x)\n        elif cmp(pivot, x):\n            gt.append(x)\n        else:\n            eq.append(x)\n    return (lt, eq, gt)",
    "quicksort-by": "def quicksort_by(cmp, lst):\n    if len(lst) <= 1:\n        return lst\n    pivot = lst[0]\n    lt, eq, gt = partition(cmp, pivot, lst[1:])\n    return quicksort_by(cmp, lt) + [pivot] + eq + quicksort_by(cmp, gt)",
    "insert-sorted": "def insert_sorted(cmp, x, sorted_lst):\n    if not sorted_lst:\n        return [x]\n    if cmp(x, sorted_lst[0]):\n        return [x] + sorted_lst\n    return [sorted_lst[0]] + insert_sorted(cmp, x, sorted_lst[1:])",
    "insertion-sort-by": "def insertion_sort_by(cmp, lst):\n    out = []\n    for x in lst:\n        out = insert_sorted(cmp, x, out)\n    return out",
    "nth-smallest": "def nth_smallest(n, lst):\n    if not lst:\n        raise ValueError('List is empty')\n    pivot = lst[0]\n    lt, eq, gt = partition(lambda a, b: a < b, pivot, lst[1:])\n    if n < len(lt):\n        return nth_smallest(n, lt)\n    if n < len(lt) + 1 + len(eq):\n        return pivot\n    return nth_smallest(n - len(lt) - 1 - len(eq), gt)",
}

CHEZ_SNIPPETS = {
    "merge": "(define (merge0 xs ys cmp)\n  (cond\n    ((null? xs) ys)\n    ((null? ys) xs)\n    ((not (cmp (car ys) (car xs)))\n     (cons (car xs) (merge0 (cdr xs) ys cmp)))\n    (else\n     (cons (car ys) (merge0 xs (cdr ys) cmp)))))",
    "split-at": "(define (split0 xs n)\n  (if (or (= n 0) (null? xs))\n      (cons '() xs)\n      (let ((rest (split0 (cdr xs) (- n 1))))\n        (cons (cons (car xs) (car rest))\n              (cdr rest)))))",
    "merge-sort-by": "(define (msort cmp xs)\n  (if (<= (length xs) 1)\n      xs\n      (let* ((mid (quotient (length xs) 2))\n             (parts (split-at xs mid))\n             (left (car parts))\n             (right (cdr parts)))\n        (merge (msort cmp left)\n               (msort cmp right)\n               cmp))))",
    "partition": "(define (partition0 cmp p xs)\n  (let loop ((rest xs) (lt '()) (eq '()) (gt '()))\n    (if (null? rest)\n        (list (reverse lt) (reverse eq) (reverse gt))\n        (let ((x (car rest)))\n          (cond\n            ((cmp x p) (loop (cdr rest) (cons x lt) eq gt))\n            ((cmp p x) (loop (cdr rest) lt eq (cons x gt)))\n            (else (loop (cdr rest) lt (cons x eq) gt)))))))",
    "quicksort-by": "(define (qsort cmp xs)\n  (if (or (null? xs) (null? (cdr xs)))\n      xs\n      (let* ((p (car xs))\n             (parts (partition cmp p (cdr xs)))\n             (lt (car parts))\n             (eq (cadr parts))\n             (gt (caddr parts)))\n        (append (qsort cmp lt)\n                (cons p eq)\n                (qsort cmp gt)))))",
    "insert-sorted": "(define (insert0 cmp x xs)\n  (cond\n    ((null? xs) (list x))\n    ((cmp x (car xs)) (cons x xs))\n    (else (cons (car xs) (insert0 cmp x (cdr xs))))))",
    "insertion-sort-by": "(define (isort cmp xs)\n  (fold-left (lambda (acc x) (insert-sorted cmp x acc))\n             '()\n             xs))",
    "nth-smallest": "(define (nth0 n xs)\n  (if (null? xs)\n      (error 'nth-smallest \"List is empty\")\n      (let* ((p (car xs))\n             (parts (partition < p (cdr xs)))\n             (lt (car parts))\n             (eq (cadr parts))\n             (gt (caddr parts))\n             (m (length lt)))\n        (cond\n          ((< n m) (nth0 n lt))\n          ((< n (+ m 1 (length eq))) p)\n          (else (nth0 (- n m 1 (length eq)) gt))))))",
}

BUGGY_CASES = [
    {
        "fn": "merge",
        "buggy": "(define (merge xs ys cmp)\n  (cond\n    [(null? xs) ys]\n    [(null? ys) xs]\n    [(cmp (car xs) (car ys))\n     (cons (car xs) (merge (cdr xs) ys cmp))]\n    [else\n     (cons (car ys) (merge xs (cdr ys) cmp))]))",
        "note": "Equal keys must prefer the left list to preserve stability.",
    },
    {
        "fn": "merge",
        "buggy": "(define (merge xs ys cmp)\n  (if (null? xs)\n      ys\n      xs))",
        "note": "Merging must consume both inputs, not return xs unchanged when non-empty.",
    },
    {
        "fn": "split-at",
        "buggy": "(define (split-at lst n)\n  (if (or (= n 1) (null? lst))\n      (cons '() lst)\n      (let ([rest (split-at (cdr lst) (- n 1))])\n        (cons (cons (car lst) (car rest))\n              (cdr rest)))))",
        "note": "Base case should trigger at n = 0, not n = 1.",
    },
    {
        "fn": "split-at",
        "buggy": "(define (split-at lst n)\n  (if (or (= n 0) (null? lst))\n      (list '() lst)\n      (let ([rest (split-at (cdr lst) (- n 1))])\n        (list (cons (car lst) (car rest))\n              (cdr rest)))))",
        "note": "Return type must be a pair via cons, not a two-element list.",
    },
    {
        "fn": "merge-sort-by",
        "buggy": "(define (merge-sort-by cmp lst)\n  (let ([len (length lst)])\n    (if (<= len 1)\n        lst\n        (let* ([mid (quotient len 2)]\n               [split (split-at lst mid)]\n               [left (car split)]\n               [right (cdr split)])\n          (merge (merge-sort-by cmp left)\n                 right\n                 cmp)))))",
        "note": "Both halves must be recursively sorted before merging.",
    },
    {
        "fn": "merge-sort-by",
        "buggy": "(define (merge-sort-by cmp lst)\n  (let ([len (length lst)])\n    (if (< len 1)\n        lst\n        (let* ([mid (quotient len 2)]\n               [split (split-at lst mid)]\n               [left (car split)]\n               [right (cdr split)])\n          (merge (merge-sort-by cmp left)\n                 (merge-sort-by cmp right)\n                 cmp)))))",
        "note": "Length-1 input must hit the base case to avoid non-terminating recursion.",
    },
    {
        "fn": "partition",
        "buggy": "(define (partition cmp pivot lst)\n  (let loop ([xs lst] [lt '()] [eq '()] [gt '()])\n    (if (null? xs)\n        (list (reverse lt) (reverse eq) (reverse gt))\n        (let ([x (car xs)])\n          (cond\n            [(cmp x pivot)\n             (loop (cdr xs) (cons x lt) eq gt)]\n            [(cmp pivot x)\n             (loop (cdr xs) lt eq (cons x gt))]\n            [else\n             (loop (cdr xs) lt eq (cons x gt))])))))",
        "note": "Elements equal to pivot must go into the eq bucket, not gt.",
    },
    {
        "fn": "partition",
        "buggy": "(define (partition cmp pivot lst)\n  (let loop ([xs lst] [lt '()] [eq '()] [gt '()])\n    (if (null? xs)\n        (list lt eq gt)\n        (let ([x (car xs)])\n          (cond\n            [(cmp x pivot)\n             (loop (cdr xs) (cons x lt) eq gt)]\n            [(cmp pivot x)\n             (loop (cdr xs) lt eq (cons x gt))]\n            [else\n             (loop (cdr xs) lt (cons x eq) gt)])))))",
        "note": "Accumulator lists must be reversed before returning to preserve encounter order.",
    },
    {
        "fn": "quicksort-by",
        "buggy": "(define (quicksort-by cmp lst)\n  (if (or (null? lst) (null? (cdr lst)))\n      lst\n      (let* ([pivot (car lst)]\n             [parts (partition cmp pivot (cdr lst))]\n             [lt (car parts)]\n             [gt (caddr parts)])\n        (append (quicksort-by cmp lt)\n                (list pivot)\n                (quicksort-by cmp gt)))))",
        "note": "Values equal to pivot must be preserved; this version drops duplicates.",
    },
    {
        "fn": "quicksort-by",
        "buggy": "(define (quicksort-by cmp lst)\n  (if (or (null? lst) (null? (cdr lst)))\n      lst\n      (let* ([pivot (car lst)]\n             [parts (partition < pivot (cdr lst))]\n             [lt (car parts)]\n             [eq (cadr parts)]\n             [gt (caddr parts)])\n        (append (quicksort-by cmp lt)\n                (cons pivot eq)\n                (quicksort-by cmp gt)))))",
        "note": "Partitioning must use cmp, not a hardcoded numeric comparator.",
    },
    {
        "fn": "insert-sorted",
        "buggy": "(define (insert-sorted cmp x sorted)\n  (cond\n    [(null? sorted) (list x)]\n    [(not (cmp (car sorted) x)) (cons x sorted)]\n    [else (cons (car sorted) (insert-sorted cmp x (cdr sorted)))]))",
        "note": "For equal keys, existing elements should stay before the inserted element.",
    },
    {
        "fn": "insert-sorted",
        "buggy": "(define (insert-sorted cmp x sorted)\n  (append sorted (list x)))",
        "note": "Insertion must place x at its sorted position, not always append.",
    },
    {
        "fn": "insertion-sort-by",
        "buggy": "(define (insertion-sort-by cmp lst)\n  (fold-left (lambda (sorted x) (insert-sorted cmp x sorted))\n             lst\n             lst))",
        "note": "Accumulator should start empty; starting from lst duplicates data.",
    },
    {
        "fn": "insertion-sort-by",
        "buggy": "(define (insertion-sort-by cmp lst)\n  lst)",
        "note": "Function must actually sort the input list.",
    },
    {
        "fn": "nth-smallest",
        "buggy": "(define (nth-smallest n lst)\n  (if (null? lst)\n      (error 'nth-smallest \"List is empty\")\n      (let* ([pivot (car lst)]\n             [parts (partition < pivot (cdr lst))]\n             [lt (car parts)]\n             [eq (cadr parts)]\n             [gt (caddr parts)]\n             [lt-len (length lt)])\n        (cond\n          [(< n lt-len)\n           (nth-smallest n lt)]\n          [(< n (+ lt-len (length eq)))\n           pivot]\n          [else\n           (nth-smallest (- n lt-len 1 (length eq)) gt)]))))",
        "note": "Pivot range must include the pivot itself (+ 1).",
    },
    {
        "fn": "nth-smallest",
        "buggy": "(define (nth-smallest n lst)\n  (let* ([pivot (car lst)]\n         [parts (partition < pivot (cdr lst))]\n         [lt (car parts)]\n         [eq (cadr parts)]\n         [gt (caddr parts)]\n         [lt-len (length lt)])\n    (cond\n      [(< n lt-len)\n       (nth-smallest n lt)]\n      [(< n (+ lt-len 1 (length eq)))\n       pivot]\n      [else\n       (nth-smallest (- n lt-len 1 (length eq)) gt)])))",
        "note": "Must guard empty input before taking car/cdr.",
    },
]

DIFFICULTY = {
    "merge": "medium",
    "split-at": "easy",
    "merge-sort-by": "hard",
    "partition": "medium",
    "quicksort-by": "hard",
    "insert-sorted": "easy",
    "insertion-sort-by": "medium",
    "nth-smallest": "hard",
}

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
    sid = f"sort_{family}_{family_counter[family]:03d}"
    sample = {
        "id": sid,
        "family": family,
        "category": category,
        "difficulty": difficulty,
        "source_module": SOURCE_MODULE,
        "source_test": SOURCE_TEST,
        "source_function": source_function,
        "prompt_body": prompt.strip(),
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
    parts = [DEFS[dep] for dep in dependency_closure(fn)] + [DEFS[fn], VERIFY_BY_FUNCTION[fn]]
    body = "\n  ".join(parts)
    return f"(let ()\n  {body})"


# -----------------------------------------------------------------------------
# Family 1: spec_to_code (16)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""You are implementing Tier-0 sorting code in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "data", "sort", "spec-to-code", fn],
    )

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Complete this Fold Scheme skeleton.

```scheme
{SKELETONS[fn]}
```

Replace `<TODO>` and return only the completed definition for `{fn}`.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "data", "sort", "skeleton-completion", fn],
    )


# -----------------------------------------------------------------------------
# Family 2: translation (16)
# -----------------------------------------------------------------------------
for fn in TRANSLATION_FUNCTIONS:
    add_sample(
        family="translation",
        category="translation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Translate the following Python function into Fold-native Scheme.
Preserve behavior exactly and use the target function name `{fn}`.
Return only the Scheme definition.

```python
{PYTHON_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "data", "sort", "python-to-scheme", fn],
    )

    add_sample(
        family="translation",
        category="translation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Convert this Chez-style snippet to canonical Fold style.
Target function name must be `{fn}`.
Return only the corrected Fold definition.

```scheme
{CHEZ_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "data", "sort", "chez-to-fold", fn],
    )


# -----------------------------------------------------------------------------
# Family 3: bugfix (16)
# -----------------------------------------------------------------------------
for case in BUGGY_CASES:
    fn = str(case["fn"])
    add_sample(
        family="bugfix",
        category="debugging",
        difficulty=DIFFICULTY[fn],
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
        tags=["tier0", "data", "sort", "bugfix", fn],
    )


# -----------------------------------------------------------------------------
# Family 4: composition/use (32)
# -----------------------------------------------------------------------------


def add_composition(
    source_function: str,
    prompt: str,
    ground_truth: str,
    verify_expr: str,
    difficulty: str,
    extra_tags: List[str],
) -> None:
    add_sample(
        family="composition",
        category="usage",
        difficulty=difficulty,
        source_function=source_function,
        prompt=prompt,
        ground_truth=ground_truth,
        verify_expr=verify_expr,
        tags=["tier0", "data", "sort", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # Direct
    (
        "merge",
        "Merge two ascending lists '(1 3 5) and '(2 4 6) using <.",
        "(merge '(1 3 5) '(2 4 6) <)",
        "(equal? (merge '(1 3 5) '(2 4 6) <) '(1 2 3 4 5 6))",
        "easy",
        ["direct"],
    ),
    (
        "merge",
        "Merge pair lists by cdr and return the element labels in order.",
        "(map car (merge '((a . 2) (b . 2)) '((c . 2) (d . 3)) (lambda (x y) (< (cdr x) (cdr y)))))",
        "(equal? (map car (merge '((a . 2) (b . 2)) '((c . 2) (d . 3)) (lambda (x y) (< (cdr x) (cdr y))))) '(a b c d))",
        "medium",
        ["direct", "stability"],
    ),
    (
        "split-at",
        "Split '(1 2 3 4 5) at index 2 and return the resulting pair.",
        "(split-at '(1 2 3 4 5) 2)",
        "(equal? (split-at '(1 2 3 4 5) 2) (cons '(1 2) '(3 4 5)))",
        "easy",
        ["direct"],
    ),
    (
        "split-at",
        "Split '(1 2) at index 5; the right side should be empty.",
        "(split-at '(1 2) 5)",
        "(equal? (split-at '(1 2) 5) (cons '(1 2) '()))",
        "easy",
        ["edge-case"],
    ),
    (
        "merge-sort-by",
        "Sort '(5 2 8 1 9 3) ascending with merge-sort-by.",
        "(merge-sort-by < '(5 2 8 1 9 3))",
        "(equal? (merge-sort-by < '(5 2 8 1 9 3)) '(1 2 3 5 8 9))",
        "medium",
        ["direct"],
    ),
    (
        "merge-sort-by",
        "Sort '(3 1 4 1 5) descending with comparator >.",
        "(merge-sort-by > '(3 1 4 1 5))",
        "(equal? (merge-sort-by > '(3 1 4 1 5)) '(5 4 3 1 1))",
        "medium",
        ["direct"],
    ),
    (
        "partition",
        "Partition '(1 3 2 4 3 5) around pivot 3 using <.",
        "(partition < 3 '(1 3 2 4 3 5))",
        "(equal? (partition < 3 '(1 3 2 4 3 5)) '((1 2) (3 3) (4 5)))",
        "medium",
        ["direct"],
    ),
    (
        "partition",
        "Partition '(4 4 4) around pivot 4 and return the three buckets.",
        "(partition < 4 '(4 4 4))",
        "(equal? (partition < 4 '(4 4 4)) '(() (4 4 4) ()))",
        "easy",
        ["direct", "duplicates"],
    ),
    (
        "quicksort-by",
        "Sort '(3 1 4 1 5 9) ascending with quicksort-by.",
        "(quicksort-by < '(3 1 4 1 5 9))",
        "(equal? (quicksort-by < '(3 1 4 1 5 9)) '(1 1 3 4 5 9))",
        "medium",
        ["direct"],
    ),
    (
        "quicksort-by",
        "Sort strings '(\"cherry\" \"apple\" \"banana\") lexicographically.",
        "(quicksort-by string<? '(\"cherry\" \"apple\" \"banana\"))",
        "(equal? (quicksort-by string<? '(\"cherry\" \"apple\" \"banana\")) '(\"apple\" \"banana\" \"cherry\"))",
        "medium",
        ["direct"],
    ),
    (
        "insert-sorted",
        "Insert 3 into sorted list '(1 2 4 5).",
        "(insert-sorted < 3 '(1 2 4 5))",
        "(equal? (insert-sorted < 3 '(1 2 4 5)) '(1 2 3 4 5))",
        "easy",
        ["direct"],
    ),
    (
        "insertion-sort-by",
        "Sort '(4 1 3 2) ascending with insertion-sort-by.",
        "(insertion-sort-by < '(4 1 3 2))",
        "(equal? (insertion-sort-by < '(4 1 3 2)) '(1 2 3 4))",
        "medium",
        ["direct"],
    ),
    (
        "nth-smallest",
        "Return the 3rd smallest element (index 2) of '(7 1 5 3 9).",
        "(nth-smallest 2 '(7 1 5 3 9))",
        "(equal? (nth-smallest 2 '(7 1 5 3 9)) 5)",
        "hard",
        ["direct"],
    ),
    (
        "nth-smallest",
        "Return the largest element via nth-smallest on '(7 1 5 3 9).",
        "(nth-smallest 4 '(7 1 5 3 9))",
        "(equal? (nth-smallest 4 '(7 1 5 3 9)) 9)",
        "hard",
        ["direct"],
    ),

    # Properties
    (
        "merge",
        "Return #t iff merge output length equals sum of input lengths.",
        "(= (length (merge '(1 4 7) '(2 3 9 10) <)) (+ (length '(1 4 7)) (length '(2 3 9 10))))",
        "(= (length (merge '(1 4 7) '(2 3 9 10) <)) (+ (length '(1 4 7)) (length '(2 3 9 10))))",
        "medium",
        ["property"],
    ),
    (
        "split-at",
        "Return #t iff split-at preserves total element count.",
        "(let ([p (split-at '(a b c d e) 3)]) (= (+ (length (car p)) (length (cdr p))) 5))",
        "(let ([p (split-at '(a b c d e) 3)]) (= (+ (length (car p)) (length (cdr p))) 5))",
        "medium",
        ["property"],
    ),
    (
        "merge-sort-by",
        "Return #t iff merge-sort-by is idempotent for '(4 1 3 2 3).",
        "(equal? (merge-sort-by < (merge-sort-by < '(4 1 3 2 3))) (merge-sort-by < '(4 1 3 2 3)))",
        "(equal? (merge-sort-by < (merge-sort-by < '(4 1 3 2 3))) (merge-sort-by < '(4 1 3 2 3)))",
        "hard",
        ["property"],
    ),
    (
        "quicksort-by",
        "Return #t iff quicksort-by and merge-sort-by agree on numeric ascending sort.",
        "(equal? (quicksort-by < '(5 2 8 1 9 3 7 4 6)) (merge-sort-by < '(5 2 8 1 9 3 7 4 6)))",
        "(equal? (quicksort-by < '(5 2 8 1 9 3 7 4 6)) (merge-sort-by < '(5 2 8 1 9 3 7 4 6)))",
        "hard",
        ["property"],
    ),
    (
        "partition",
        "Return #t iff partition bucket sizes add up to original list size.",
        "(let ([p (partition < 5 '(7 1 5 3 5 9 0))]) (= (+ (length (car p)) (length (cadr p)) (length (caddr p))) 7))",
        "(let ([p (partition < 5 '(7 1 5 3 5 9 0))]) (= (+ (length (car p)) (length (cadr p)) (length (caddr p))) 7))",
        "medium",
        ["property"],
    ),
    (
        "partition",
        "Return #t iff partition buckets respect pivot ordering constraints.",
        "(let ([p (partition < 5 '(7 1 5 3 5 9 0))]) (and (null? (filter (lambda (x) (not (< x 5))) (car p))) (null? (filter (lambda (x) (not (equal? x 5))) (cadr p))) (null? (filter (lambda (x) (not (> x 5))) (caddr p)))))",
        "(let ([p (partition < 5 '(7 1 5 3 5 9 0))]) (and (null? (filter (lambda (x) (not (< x 5))) (car p))) (null? (filter (lambda (x) (not (equal? x 5))) (cadr p))) (null? (filter (lambda (x) (not (> x 5))) (caddr p)))))",
        "hard",
        ["property"],
    ),
    (
        "quicksort-by",
        "Return #t iff descending quicksort equals reverse of ascending quicksort.",
        "(equal? (quicksort-by > '(5 1 4 2 3)) (reverse (quicksort-by < '(5 1 4 2 3))))",
        "(equal? (quicksort-by > '(5 1 4 2 3)) (reverse (quicksort-by < '(5 1 4 2 3))))",
        "medium",
        ["property"],
    ),
    (
        "insert-sorted",
        "Return #t iff insert-sorted increases list length by one.",
        "(= (length (insert-sorted < 6 '(1 3 5 7 9))) (+ 1 (length '(1 3 5 7 9))))",
        "(= (length (insert-sorted < 6 '(1 3 5 7 9))) (+ 1 (length '(1 3 5 7 9))))",
        "easy",
        ["property"],
    ),
    (
        "insertion-sort-by",
        "Return #t iff insertion-sort-by is idempotent for '(6 1 5 2 4 3).",
        "(equal? (insertion-sort-by < (insertion-sort-by < '(6 1 5 2 4 3))) (insertion-sort-by < '(6 1 5 2 4 3)))",
        "(equal? (insertion-sort-by < (insertion-sort-by < '(6 1 5 2 4 3))) (insertion-sort-by < '(6 1 5 2 4 3)))",
        "medium",
        ["property"],
    ),
    (
        "nth-smallest",
        "Return #t iff nth-smallest 0 equals the head of merge-sort-by output.",
        "(= (nth-smallest 0 '(9 2 5 1 7)) (car (merge-sort-by < '(9 2 5 1 7))))",
        "(= (nth-smallest 0 '(9 2 5 1 7)) (car (merge-sort-by < '(9 2 5 1 7))))",
        "hard",
        ["property"],
    ),
    (
        "nth-smallest",
        "Return #t iff nth-smallest raises on empty input.",
        "(guard (ex [else #t]) (begin (nth-smallest 0 '()) #f))",
        "(guard (ex [else #t]) (begin (nth-smallest 0 '()) #f))",
        "hard",
        ["edge-case", "property"],
    ),

    # Fold/loop/integration
    (
        "insert-sorted",
        "Use fold-left with insert-sorted to build a sorted list from '(5 2 4 1 3).",
        "(fold-left (lambda (acc x) (insert-sorted < x acc)) '() '(5 2 4 1 3))",
        "(equal? (fold-left (lambda (acc x) (insert-sorted < x acc)) '() '(5 2 4 1 3)) '(1 2 3 4 5))",
        "hard",
        ["fold"],
    ),
    (
        "nth-smallest",
        "Return #t iff nth-smallest matches list-ref of quicksort-by at the same index.",
        "(= (nth-smallest 3 '(8 1 6 2 7 3 5 4)) (list-ref (quicksort-by < '(8 1 6 2 7 3 5 4)) 3))",
        "(= (nth-smallest 3 '(8 1 6 2 7 3 5 4)) (list-ref (quicksort-by < '(8 1 6 2 7 3 5 4)) 3))",
        "hard",
        ["integration"],
    ),
    (
        "insertion-sort-by",
        "Sort pairs by descending cdr and return only labels.",
        "(map car (insertion-sort-by (lambda (x y) (> (cdr x) (cdr y))) '((a . 2) (b . 5) (c . 1))))",
        "(equal? (map car (insertion-sort-by (lambda (x y) (> (cdr x) (cdr y))) '((a . 2) (b . 5) (c . 1)))) '(b a c))",
        "hard",
        ["integration"],
    ),
    (
        "merge",
        "Split a sorted list into two halves and merge them back.",
        "(let* ([xs '(1 2 3 4 5 6)] [p (split-at xs 3)]) (merge (car p) (cdr p) <))",
        "(equal? (let* ([xs '(1 2 3 4 5 6)] [p (split-at xs 3)]) (merge (car p) (cdr p) <)) '(1 2 3 4 5 6))",
        "medium",
        ["integration"],
    ),
    (
        "quicksort-by",
        "Sort an empty list with quicksort-by.",
        "(quicksort-by < '())",
        "(equal? (quicksort-by < '()) '())",
        "easy",
        ["edge-case"],
    ),
    (
        "partition",
        "Partition '(1 2 3) around pivot 10 and verify all items are in lt.",
        "(partition < 10 '(1 2 3))",
        "(equal? (partition < 10 '(1 2 3)) '((1 2 3) () ()))",
        "easy",
        ["integration"],
    ),
    (
        "nth-smallest",
        "Return #t iff nth-smallest agrees with merge-sort-by at index 2.",
        "(= (nth-smallest 2 '(8 3 5 1 9)) (list-ref (merge-sort-by < '(8 3 5 1 9)) 2))",
        "(= (nth-smallest 2 '(8 3 5 1 9)) (list-ref (merge-sort-by < '(8 3 5 1 9)) 2))",
        "hard",
        ["integration"],
    ),
]

for fn, prompt, gt, verify, diff, tags in composition_cases:
    add_composition(fn, prompt, gt, verify, diff, tags)

if sum(1 for s in samples if s["family"] == "composition") != 32:
    raise ValueError("composition family must contain exactly 32 samples")

# -----------------------------------------------------------------------------
# Split train/eval
# -----------------------------------------------------------------------------
if len(samples) != 80:
    raise ValueError(f"expected 80 samples, got {len(samples)}")

by_family: Dict[str, List[Dict[str, object]]] = defaultdict(list)
for s in samples:
    by_family[str(s["family"])].append(s)

EVAL_QUOTA = {
    "spec_to_code": 3,
    "translation": 3,
    "bugfix": 3,
    "composition": 5,
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
            removable.sort(key=lambda r: (fn_counts[str(r["source_function"])], str(r["id"])), reverse=True)
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

if len(train_rows) != 66 or len(eval_rows) != 14:
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
    "difficulty": dict(sorted(Counter(str(s["difficulty"]) for s in samples).items())),
    "source_functions": dict(sorted(Counter(str(s["source_function"]) for s in samples).items())),
}

with SUMMARY_PATH.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print(f"Wrote: {ALL_PATH}")
print(f"Wrote: {TRAIN_PATH}")
print(f"Wrote: {EVAL_PATH}")
