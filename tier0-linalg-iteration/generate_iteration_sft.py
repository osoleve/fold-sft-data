#!/usr/bin/env python3
"""Generate SFT samples for lattice/linalg/iteration.ss."""

from __future__ import annotations

import json
import re
import sys
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

SOURCE_MODULE = "lattice/linalg/iteration.ss"
SOURCE_TEST = "lattice/linalg/test-iteration.ss"

DEFS: Dict[str, str] = {
    "vec-map-idx": """(define-syntax vec-map-idx
  (syntax-rules ()
                [(_ idx vec body)
                 (let* ([v vec]
                        [n (vector-length v)]
                        [result (make-vector n 0)])
                       (do ([idx 0 (+ idx 1)])
                           ((= idx n) result)
                           (vector-set! result idx body)))]))""",
    "vec-fold-idx": """(define-syntax vec-fold-idx
  (syntax-rules ()
                [(_ acc init idx vec body)
                 (let* ([v vec]
                        [n (vector-length v)])
                       (do ([idx 0 (+ idx 1)]
                            [acc init body])
                           ((= idx n) acc)))]))""",
    "vec-zip-map-idx": """(define-syntax vec-zip-map-idx
  (syntax-rules ()
                [(_ idx vec1 vec2 body)
                 (let* ([v1 vec1]
                        [v2 vec2]
                        [n (vector-length v1)]
                        [result (make-vector n 0)])
                       (do ([idx 0 (+ idx 1)])
                           ((= idx n) result)
                           (vector-set! result idx body)))]))""",
    "matrix-do!": """(define-syntax matrix-do!
  (syntax-rules ()
                [(_ i j rows cols body ...)
                 (let ([r rows]
                       [c cols])
                      (do ([i 0 (+ i 1)])
                          ((= i r))
                          (do ([j 0 (+ j 1)])
                              ((= j c))
                              body ...)))]))""",
    "range-fold": """(define-syntax range-fold
  (syntax-rules ()
                [(_ acc init var start end body)
                 (do ([var start (+ var 1)]
                      [acc init body])
                     ((= var end) acc))]))""",
    "dot-product-loop": """(define-syntax dot-product-loop
  (syntax-rules ()
                [(_ k len get-a get-b)
                 (do ([k 0 (+ k 1)]
                      [sum 0 (+ sum (* get-a get-b))])
                     ((= k len) sum))]))""",
    "vec-tabulate": """(define-syntax vec-tabulate
  (syntax-rules ()
                [(_ size idx body)
                 (let* ([len size]
                        [result (make-vector len 0)])
                       (do ([idx 0 (+ idx 1)])
                           ((= idx len) result)
                           (vector-set! result idx body)))]))""",
    "vec-scan": """(define-syntax vec-scan
  (syntax-rules ()
                [(_ size init idx acc body)
                 (let* ([len size]
                        [result (make-vector len 0)])
                       (if (= len 0)
                           result
                           (begin
                             (vector-set! result 0 init)
                             (let loop ([idx 1] [acc init])
                                  (if (= idx len)
                                      result
                                      (let ([acc body])
                                           (vector-set! result idx acc)
                                           (loop (+ idx 1) acc)))))))]))""",
}

DEPENDS: Dict[str, List[str]] = {
    "vec-map-idx": [],
    "vec-fold-idx": [],
    "vec-zip-map-idx": [],
    "matrix-do!": [],
    "range-fold": [],
    "dot-product-loop": [],
    "vec-tabulate": [],
    "vec-scan": [],
}

FUNCTION_ORDER = [
    "vec-map-idx",
    "vec-fold-idx",
    "vec-zip-map-idx",
    "matrix-do!",
    "range-fold",
    "dot-product-loop",
    "vec-tabulate",
    "vec-scan",
]

FUNCTION_SPECS = {
    "vec-map-idx": "Map over a vector with index access and return a newly allocated vector.",
    "vec-fold-idx": "Left-fold over vector indices, updating accumulator with each index.",
    "vec-zip-map-idx": "Zip-map two vectors by index into a new vector of the first vector's length.",
    "matrix-do!": "Execute body for each matrix coordinate (i,j) in row-major nested loops.",
    "range-fold": "Fold over integer range [start, end), updating accumulator each step.",
    "dot-product-loop": "Accumulate sum_{k=0..len-1} get-a(k) * get-b(k).",
    "vec-tabulate": "Construct vector of length size where element i is body evaluated with idx=i.",
    "vec-scan": "Construct prefix-accumulation vector with element 0=init and iterative updates from idx=1.",
}

SKELETONS = {
    "vec-map-idx": """(define-syntax vec-map-idx
  (syntax-rules ()
                [(_ idx vec body)
                 ;; TODO: allocate result and map body at each idx
                 <TODO>]))""",
    "vec-fold-idx": """(define-syntax vec-fold-idx
  (syntax-rules ()
                [(_ acc init idx vec body)
                 ;; TODO: iterate indices and thread accumulator
                 <TODO>]))""",
    "vec-zip-map-idx": """(define-syntax vec-zip-map-idx
  (syntax-rules ()
                [(_ idx vec1 vec2 body)
                 ;; TODO: zip two vectors by index into new result
                 <TODO>]))""",
    "matrix-do!": """(define-syntax matrix-do!
  (syntax-rules ()
                [(_ i j rows cols body ...)
                 ;; TODO: nested i/j loops over rows x cols
                 <TODO>]))""",
    "range-fold": """(define-syntax range-fold
  (syntax-rules ()
                [(_ acc init var start end body)
                 ;; TODO: fold over var from start to end-1
                 <TODO>]))""",
    "dot-product-loop": """(define-syntax dot-product-loop
  (syntax-rules ()
                [(_ k len get-a get-b)
                 ;; TODO: accumulate sum of pairwise products
                 <TODO>]))""",
    "vec-tabulate": """(define-syntax vec-tabulate
  (syntax-rules ()
                [(_ size idx body)
                 ;; TODO: allocate vector and fill each element from body
                 <TODO>]))""",
    "vec-scan": """(define-syntax vec-scan
  (syntax-rules ()
                [(_ size init idx acc body)
                 ;; TODO: handle empty case, seed init, iterate idx>=1
                 <TODO>]))""",
}

VERIFY_BY_FUNCTION = {
    "vec-map-idx": "(let* ([src (vector 3 4 5)] [mapped (vec-map-idx i src (+ (vector-ref src i) i))]) (and (equal? mapped (vector 3 5 7)) (= (vec-fold-idx acc 0 k mapped (+ acc (vector-ref mapped k))) 15)))",
    "vec-fold-idx": "(let ([src (vec-tabulate 5 i (+ i 1))]) (and (= (vec-fold-idx acc 0 i src (+ acc (vector-ref src i))) 15) (= (vec-fold-idx acc 0 i src (+ acc (* i (vector-ref src i)))) 40)))",
    "vec-zip-map-idx": "(let* ([a (vector 1 2 3)] [b (vector 10 20 30)] [z (vec-zip-map-idx i a b (+ (vector-ref a i) (vector-ref b i)))]) (and (equal? z (vector 11 22 33)) (= (dot-product-loop k 3 (vector-ref z k) 1) 66)))",
    "matrix-do!": "(let ([rows 2] [cols 3] [data (make-vector 6 0)]) (matrix-do! i j rows cols (vector-set! data (+ (* i cols) j) (+ (* i 10) j))) (and (equal? data (vector 0 1 2 10 11 12)) (= (vec-fold-idx s 0 k data (+ s (vector-ref data k))) 36)))",
    "range-fold": "(and (= (range-fold s 0 i 1 11 (+ s i)) 55) (equal? (vec-tabulate 4 i (range-fold acc 0 j 0 (+ i 1) (+ acc j))) (vector 0 1 3 6)))",
    "dot-product-loop": "(let* ([a (vector 1 2 3)] [b (vector 4 5 6)] [p (vec-zip-map-idx i a b (* (vector-ref a i) (vector-ref b i)))]) (and (= (dot-product-loop k 3 (vector-ref a k) (vector-ref b k)) 32) (= (dot-product-loop k 3 (vector-ref p k) 1) 32)))",
    "vec-tabulate": "(let ([sq (vec-tabulate 5 i (* i i))]) (and (equal? sq (vector 0 1 4 9 16)) (= (range-fold s 0 i 0 5 (+ s (vector-ref sq i))) 30)))",
    "vec-scan": "(let ([scan (vec-scan 5 0 i acc (+ acc i))]) (and (equal? scan (vector 0 1 3 6 10)) (= (dot-product-loop k 5 (vector-ref scan k) 1) 20) (equal? (vec-scan 0 99 i acc (+ acc 1)) (vector))))",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")
TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "vec-map-idx": "def vec_map_idx(v, f):\n    n = len(v)\n    out = [0] * n\n    for i in range(n):\n        out[i] = f(i, v)\n    return out",
    "vec-fold-idx": "def vec_fold_idx(init, v, step):\n    acc = init\n    for i in range(len(v)):\n        acc = step(acc, i, v)\n    return acc",
    "vec-zip-map-idx": "def vec_zip_map_idx(v1, v2, f):\n    n = len(v1)\n    out = [0] * n\n    for i in range(n):\n        out[i] = f(i, v1, v2)\n    return out",
    "matrix-do!": "def matrix_do(rows, cols, body):\n    for i in range(rows):\n        for j in range(cols):\n            body(i, j)",
    "range-fold": "def range_fold(init, start, end, step):\n    acc = init\n    i = start\n    while i < end:\n        acc = step(acc, i)\n        i += 1\n    return acc",
    "dot-product-loop": "def dot_product_loop(length, get_a, get_b):\n    s = 0\n    for k in range(length):\n        s += get_a(k) * get_b(k)\n    return s",
    "vec-tabulate": "def vec_tabulate(size, f):\n    out = [0] * size\n    for i in range(size):\n        out[i] = f(i)\n    return out",
    "vec-scan": "def vec_scan(size, init, step):\n    out = [0] * size\n    if size == 0:\n        return out\n    out[0] = init\n    acc = init\n    for i in range(1, size):\n        acc = step(i, acc)\n        out[i] = acc\n    return out",
}

CHEZ_SNIPPETS = {
    "vec-map-idx": "(define-syntax vmap-idx\n  (syntax-rules ()\n    ((_ i v expr)\n     (let* ((src v)\n            (n (vector-length src))\n            (out (make-vector n 0)))\n       (do ((i 0 (+ i 1)))\n           ((= i n) out)\n         (vector-set! out i expr))))))",
    "vec-fold-idx": "(define-syntax vfold-idx\n  (syntax-rules ()\n    ((_ acc init i v expr)\n     (let* ((src v)\n            (n (vector-length src)))\n       (do ((i 0 (+ i 1))\n            (acc init expr))\n           ((= i n) acc))))))",
    "vec-zip-map-idx": "(define-syntax vzip-map-idx\n  (syntax-rules ()\n    ((_ i a b expr)\n     (let* ((v1 a)\n            (v2 b)\n            (n (vector-length v1))\n            (out (make-vector n 0)))\n       (do ((i 0 (+ i 1)))\n           ((= i n) out)\n         (vector-set! out i expr))))))",
    "matrix-do!": "(define-syntax mat-do!\n  (syntax-rules ()\n    ((_ i j rows cols body ...)\n     (let ((r rows)\n           (c cols))\n       (do ((i 0 (+ i 1)))\n           ((= i r))\n         (do ((j 0 (+ j 1)))\n             ((= j c))\n           body ...))))))",
    "range-fold": "(define-syntax rfold\n  (syntax-rules ()\n    ((_ acc init i start end expr)\n     (do ((i start (+ i 1))\n          (acc init expr))\n         ((= i end) acc)))))",
    "dot-product-loop": "(define-syntax dot-loop\n  (syntax-rules ()\n    ((_ k len get-a get-b)\n     (do ((k 0 (+ k 1))\n          (sum 0 (+ sum (* get-a get-b))))\n         ((= k len) sum)))))",
    "vec-tabulate": "(define-syntax vtabulate\n  (syntax-rules ()\n    ((_ size i expr)\n     (let* ((n size)\n            (out (make-vector n 0)))\n       (do ((i 0 (+ i 1)))\n           ((= i n) out)\n         (vector-set! out i expr))))))",
    "vec-scan": "(define-syntax vscan\n  (syntax-rules ()\n    ((_ size init i acc expr)\n     (let* ((n size)\n            (out (make-vector n 0)))\n       (if (= n 0)\n           out\n           (begin\n             (vector-set! out 0 init)\n             (let loop ((i 1) (acc init))\n               (if (= i n)\n                   out\n                   (let ((acc expr))\n                     (vector-set! out i acc)\n                     (loop (+ i 1) acc))))))))))",
}

BUGGY_CASES = [
    {
        "fn": "vec-map-idx",
        "buggy": "(define-syntax vec-map-idx\n  (syntax-rules ()\n                [(_ idx vec body)\n                 (let* ([v vec]\n                        [n (vector-length v)]\n                        [result (make-vector n 0)])\n                       (do ([idx 0 (+ idx 1)])\n                           ((= idx n) result)\n                           (vector-set! result idx (vector-ref v idx))))]))",
        "note": "Macro ignores body and copies input unchanged.",
    },
    {
        "fn": "vec-map-idx",
        "buggy": "(define-syntax vec-map-idx\n  (syntax-rules ()\n                [(_ idx vec body)\n                 (let* ([v vec]\n                        [n (vector-length v)]\n                        [result (make-vector n 0)])\n                       (do ([idx 0 (+ idx 1)])\n                           ((>= idx (- n 1)) result)\n                           (vector-set! result idx body)))]))",
        "note": "Off-by-one termination skips filling the final element.",
    },
    {
        "fn": "vec-fold-idx",
        "buggy": "(define-syntax vec-fold-idx\n  (syntax-rules ()\n                [(_ acc init idx vec body)\n                 (let* ([v vec]\n                        [n (vector-length v)])\n                       (do ([idx 0 (+ idx 1)]\n                            [acc init body])\n                           ((>= idx (- n 1)) acc)))]))",
        "note": "Fold exits one iteration early and misses the last index.",
    },
    {
        "fn": "vec-fold-idx",
        "buggy": "(define-syntax vec-fold-idx\n  (syntax-rules ()\n                [(_ acc init idx vec body)\n                 (let* ([v vec]\n                        [n (vector-length v)])\n                       (do ([idx 0 (+ idx 1)]\n                            [acc init (let ([acc init]) body)])\n                           ((= idx n) acc)))]))",
        "note": "Accumulator is reset to init on every step instead of threaded forward.",
    },
    {
        "fn": "vec-zip-map-idx",
        "buggy": "(define-syntax vec-zip-map-idx\n  (syntax-rules ()\n                [(_ idx vec1 vec2 body)\n                 (let* ([v1 vec1]\n                        [v2 vec1]\n                        [n (vector-length v1)]\n                        [result (make-vector n 0)])\n                       (do ([idx 0 (+ idx 1)])\n                           ((= idx n) result)\n                           (vector-set! result idx body)))]))",
        "note": "Second input vector is accidentally aliased to the first.",
    },
    {
        "fn": "vec-zip-map-idx",
        "buggy": "(define-syntax vec-zip-map-idx\n  (syntax-rules ()\n                [(_ idx vec1 vec2 body)\n                 (let* ([v1 vec1]\n                        [v2 vec2]\n                        [n (vector-length v1)]\n                        [result (make-vector n 0)])\n                       (do ([idx 0 (+ idx 1)])\n                           ((>= idx (- n 1)) result)\n                           (vector-set! result idx body)))]))",
        "note": "Loop stops before the final zipped element is written.",
    },
    {
        "fn": "matrix-do!",
        "buggy": "(define-syntax matrix-do!\n  (syntax-rules ()\n                [(_ i j rows cols body ...)\n                 (let ([r rows]\n                       [c cols])\n                      (do ([i 0 (+ i 1)])\n                          ((= i r))\n                          (do ([j 0 (+ j 1)])\n                              ((= j r))\n                              body ...)))]))",
        "note": "Inner loop uses row count instead of column count.",
    },
    {
        "fn": "matrix-do!",
        "buggy": "(define-syntax matrix-do!\n  (syntax-rules ()\n                [(_ i j rows cols body ...)\n                 (let ([r rows]\n                       [c cols])\n                      (do ([i 1 (+ i 1)])\n                          ((= i r))\n                          (do ([j 0 (+ j 1)])\n                              ((= j c))\n                              body ...)))]))",
        "note": "Row iteration starts at 1 and skips the first row.",
    },
    {
        "fn": "range-fold",
        "buggy": "(define-syntax range-fold\n  (syntax-rules ()\n                [(_ acc init var start end body)\n                 (do ([var (+ start 1) (+ var 1)]\n                      [acc init body])\n                     ((= var end) acc))]))",
        "note": "Range starts at start+1, dropping the first term.",
    },
    {
        "fn": "range-fold",
        "buggy": "(define-syntax range-fold\n  (syntax-rules ()\n                [(_ acc init var start end body)\n                 (do ([var start (+ var 1)]\n                      [acc 0 body])\n                     ((= var end) acc))]))",
        "note": "Accumulator ignores caller-provided init value.",
    },
    {
        "fn": "dot-product-loop",
        "buggy": "(define-syntax dot-product-loop\n  (syntax-rules ()\n                [(_ k len get-a get-b)\n                 (do ([k 0 (+ k 1)]\n                      [sum 0 (+ sum (+ get-a get-b))])\n                     ((= k len) sum))]))",
        "note": "Update adds terms instead of multiplying pairwise.",
    },
    {
        "fn": "dot-product-loop",
        "buggy": "(define-syntax dot-product-loop\n  (syntax-rules ()\n                [(_ k len get-a get-b)\n                 (do ([k 0 (+ k 1)]\n                      [sum 0 (+ sum (* get-a get-b))])\n                     ((>= k (- len 1)) sum))]))",
        "note": "Termination condition skips the final multiply-accumulate step.",
    },
    {
        "fn": "vec-tabulate",
        "buggy": "(define-syntax vec-tabulate\n  (syntax-rules ()\n                [(_ size idx body)\n                 (let* ([len size]\n                        [result (make-vector len 0)])\n                       (do ([idx 0 (+ idx 1)])\n                           ((= idx len) result)\n                           (vector-set! result idx idx)))]))",
        "note": "Body is ignored and index value is stored directly.",
    },
    {
        "fn": "vec-tabulate",
        "buggy": "(define-syntax vec-tabulate\n  (syntax-rules ()\n                [(_ size idx body)\n                 (let* ([len size]\n                        [result (make-vector len 0)])\n                       (do ([idx 0 (+ idx 1)])\n                           ((>= idx (- len 1)) result)\n                           (vector-set! result idx body)))]))",
        "note": "Final element is never written because of an off-by-one loop bound.",
    },
    {
        "fn": "vec-scan",
        "buggy": "(define-syntax vec-scan\n  (syntax-rules ()\n                [(_ size init idx acc body)\n                 (let* ([len size]\n                        [result (make-vector len 0)])\n                       (if (= len 0)\n                           result\n                           (begin\n                             (vector-set! result 0 init)\n                             (let loop ([idx 0] [acc init])\n                                  (if (= idx len)\n                                      result\n                                      (let ([acc body])\n                                           (vector-set! result idx acc)\n                                           (loop (+ idx 1) acc)))))))]))",
        "note": "Scan loop starts at index 0 and overwrites the seeded initial element.",
    },
    {
        "fn": "vec-scan",
        "buggy": "(define-syntax vec-scan\n  (syntax-rules ()\n                [(_ size init idx acc body)\n                 (let* ([len size]\n                        [result (make-vector len 0)])\n                       (if (= len 0)\n                           result\n                           (begin\n                             (vector-set! result 0 init)\n                             (let loop ([idx 1] [acc init])\n                                  (if (= idx len)\n                                      result\n                                      (let ([acc body])\n                                           (vector-set! result idx acc)\n                                           (loop (+ idx 1) init)))))))]))",
        "note": "Recursive step feeds init instead of the updated accumulator.",
    },
]

DIFFICULTY = {
    "vec-map-idx": "medium",
    "vec-fold-idx": "medium",
    "vec-zip-map-idx": "medium",
    "matrix-do!": "medium",
    "range-fold": "easy",
    "dot-product-loop": "medium",
    "vec-tabulate": "easy",
    "vec-scan": "hard",
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
    sid = f"iteration_{family}_{family_counter[family]:03d}"
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
    return [name for name in DEFS.keys() if name != fn and name in tokens]


def dependency_closure(fn: str) -> List[str]:
    ordered: List[str] = []
    seen: Set[str] = set()

    def visit(name: str) -> None:
        for dep in DEPENDS.get(name, []):
            if dep == fn:
                continue
            if dep in DEFS and dep not in seen:
                seen.add(dep)
                visit(dep)
                ordered.append(dep)

    for dep in DEPENDS.get(fn, []):
        if dep == fn:
            continue
        if dep in DEFS and dep not in seen:
            seen.add(dep)
            visit(dep)
            ordered.append(dep)

    for dep in verify_refs(fn):
        if dep == fn:
            continue
        if dep in DEFS and dep not in seen:
            seen.add(dep)
            visit(dep)
            ordered.append(dep)

    return ordered


def def_verify(fn: str) -> str:
    parts = [DEFS[d] for d in dependency_closure(fn)] + [DEFS[fn], VERIFY_BY_FUNCTION[fn]]
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
        prompt=f"""Implement this macro in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Macro: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one `define-syntax` form for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["linalg", "iteration", "spec-to-code", fn],
    )

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Complete this Fold Scheme macro skeleton.

Module: {SOURCE_MODULE}
Macro target: `{fn}`
Behavior contract: {FUNCTION_SPECS[fn]}

```scheme
{SKELETONS[fn]}
```

Output only the completed macro definition.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["linalg", "iteration", "spec-to-code", "skeleton", fn],
    )


# -----------------------------------------------------------------------------
# Family 2: translation (16)
# -----------------------------------------------------------------------------
for fn in TRANSLATION_FUNCTIONS:
    add_sample(
        family="translation",
        category="transpile",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Translate the following Python helper into Fold-native Scheme macro form.
Preserve behavior exactly.

Target macro name: `{fn}`

```python
{PYTHON_SNIPPETS[fn]}
```

Return only the Scheme macro definition.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["linalg", "iteration", "translation", "python", fn],
    )

    add_sample(
        family="translation",
        category="transpile",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Convert this Chez-style macro snippet to canonical Fold style.
Keep semantics identical.

Target macro: `{fn}`

```scheme
{CHEZ_SNIPPETS[fn]}
```

Return only Fold code.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["linalg", "iteration", "translation", "chez", fn],
    )


# -----------------------------------------------------------------------------
# Family 3: bugfix (16)
# -----------------------------------------------------------------------------
for case in BUGGY_CASES:
    fn = case["fn"]
    add_sample(
        family="bugfix",
        category="repair",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Fix the bug in this Fold Scheme macro with minimal semantic changes.
Target: `{fn}` in `{SOURCE_MODULE}`.
Known issue: {case['note']}

```scheme
{case['buggy']}
```

Return only the corrected macro definition.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["linalg", "iteration", "bugfix", fn],
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
        tags=["linalg", "iteration", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # vec-map-idx
    (
        "vec-map-idx",
        "Map square over #(1 2 3 4) using vec-map-idx.",
        "(let ([v (vector 1 2 3 4)]) (vec-map-idx i v (* (vector-ref v i) (vector-ref v i))))",
        "(equal? (let ([v (vector 1 2 3 4)]) (vec-map-idx i v (* (vector-ref v i) (vector-ref v i)))) (vector 1 4 9 16))",
        "easy",
        ["direct"],
    ),
    (
        "vec-map-idx",
        "Add index to each element of #(10 20 30) with vec-map-idx.",
        "(let ([v (vector 10 20 30)]) (vec-map-idx i v (+ (vector-ref v i) i)))",
        "(equal? (let ([v (vector 10 20 30)]) (vec-map-idx i v (+ (vector-ref v i) i))) (vector 10 21 32))",
        "medium",
        ["direct"],
    ),
    (
        "vec-map-idx",
        "Map (+ x 10) over #(1 2 3) then fold-sum the mapped vector.",
        "(let* ([v (vector 1 2 3)] [mapped (vec-map-idx i v (+ (vector-ref v i) 10))]) (vec-fold-idx acc 0 k mapped (+ acc (vector-ref mapped k))))",
        "(equal? (let* ([v (vector 1 2 3)] [mapped (vec-map-idx i v (+ (vector-ref v i) 10))]) (vec-fold-idx acc 0 k mapped (+ acc (vector-ref mapped k)))) 36)",
        "hard",
        ["integration"],
    ),
    (
        "vec-map-idx",
        "Map over empty vector and return the resulting vector.",
        "(vec-map-idx i (vector) i)",
        "(equal? (vec-map-idx i (vector) i) (vector))",
        "easy",
        ["edge-case"],
    ),

    # vec-fold-idx
    (
        "vec-fold-idx",
        "Fold-sum #(1 2 3 4 5) with vec-fold-idx.",
        "(let ([v (vector 1 2 3 4 5)]) (vec-fold-idx acc 0 i v (+ acc (vector-ref v i))))",
        "(equal? (let ([v (vector 1 2 3 4 5)]) (vec-fold-idx acc 0 i v (+ acc (vector-ref v i)))) 15)",
        "easy",
        ["direct"],
    ),
    (
        "vec-fold-idx",
        "Compute weighted sum sum(i * v[i]) for #(1 2 3 4 5).",
        "(let ([v (vector 1 2 3 4 5)]) (vec-fold-idx acc 0 i v (+ acc (* i (vector-ref v i)))))",
        "(equal? (let ([v (vector 1 2 3 4 5)]) (vec-fold-idx acc 0 i v (+ acc (* i (vector-ref v i))))) 40)",
        "medium",
        ["direct"],
    ),
    (
        "vec-fold-idx",
        "Fold over empty vector and return the initial accumulator 42.",
        "(vec-fold-idx acc 42 i (vector) (+ acc i))",
        "(equal? (vec-fold-idx acc 42 i (vector) (+ acc i)) 42)",
        "easy",
        ["edge-case"],
    ),
    (
        "vec-fold-idx",
        "Tabulate #(1 2 3 4) then fold-product over it.",
        "(let ([v (vec-tabulate 4 i (+ i 1))]) (vec-fold-idx acc 1 i v (* acc (vector-ref v i))))",
        "(equal? (let ([v (vec-tabulate 4 i (+ i 1))]) (vec-fold-idx acc 1 i v (* acc (vector-ref v i)))) 24)",
        "hard",
        ["integration"],
    ),

    # vec-zip-map-idx
    (
        "vec-zip-map-idx",
        "Add #(1 2 3) and #(10 20 30) with vec-zip-map-idx.",
        "(let ([a (vector 1 2 3)] [b (vector 10 20 30)]) (vec-zip-map-idx i a b (+ (vector-ref a i) (vector-ref b i))))",
        "(equal? (let ([a (vector 1 2 3)] [b (vector 10 20 30)]) (vec-zip-map-idx i a b (+ (vector-ref a i) (vector-ref b i)))) (vector 11 22 33))",
        "medium",
        ["direct"],
    ),
    (
        "vec-zip-map-idx",
        "Zip two vectors with expression a[i]*b[i]+i.",
        "(let ([a (vector 2 3 4)] [b (vector 5 6 7)]) (vec-zip-map-idx i a b (+ (* (vector-ref a i) (vector-ref b i)) i)))",
        "(equal? (let ([a (vector 2 3 4)] [b (vector 5 6 7)]) (vec-zip-map-idx i a b (+ (* (vector-ref a i) (vector-ref b i)) i))) (vector 10 19 30))",
        "medium",
        ["direct"],
    ),
    (
        "vec-zip-map-idx",
        "Zip-multiply #(1 2 3) and #(4 5 6), then sum via dot-product-loop with ones.",
        "(let* ([a (vector 1 2 3)] [b (vector 4 5 6)] [p (vec-zip-map-idx i a b (* (vector-ref a i) (vector-ref b i)))]) (dot-product-loop k 3 (vector-ref p k) 1))",
        "(equal? (let* ([a (vector 1 2 3)] [b (vector 4 5 6)] [p (vec-zip-map-idx i a b (* (vector-ref a i) (vector-ref b i)))]) (dot-product-loop k 3 (vector-ref p k) 1)) 32)",
        "hard",
        ["integration"],
    ),
    (
        "vec-zip-map-idx",
        "Zip-map two empty vectors and return the result.",
        "(vec-zip-map-idx i (vector) (vector) i)",
        "(equal? (vec-zip-map-idx i (vector) (vector) i) (vector))",
        "easy",
        ["edge-case"],
    ),

    # matrix-do!
    (
        "matrix-do!",
        "Fill a 2x3 flat data vector with value 10*i + j using matrix-do!.",
        "(let ([rows 2] [cols 3] [data (make-vector 6 0)]) (matrix-do! i j rows cols (vector-set! data (+ (* i cols) j) (+ (* i 10) j))) data)",
        "(equal? (let ([rows 2] [cols 3] [data (make-vector 6 0)]) (matrix-do! i j rows cols (vector-set! data (+ (* i cols) j) (+ (* i 10) j))) data) (vector 0 1 2 10 11 12))",
        "medium",
        ["direct"],
    ),
    (
        "matrix-do!",
        "Use matrix-do! to fill a 3x2 grid with ones, then sum all entries.",
        "(let ([rows 3] [cols 2] [data (make-vector 6 0)]) (matrix-do! i j rows cols (vector-set! data (+ (* i cols) j) 1)) (vec-fold-idx s 0 k data (+ s (vector-ref data k))))",
        "(equal? (let ([rows 3] [cols 2] [data (make-vector 6 0)]) (matrix-do! i j rows cols (vector-set! data (+ (* i cols) j) 1)) (vec-fold-idx s 0 k data (+ s (vector-ref data k)))) 6)",
        "hard",
        ["integration"],
    ),
    (
        "matrix-do!",
        "Build a 3x3 identity-pattern flat vector using matrix-do!.",
        "(let ([n 3] [data (make-vector 9 0)]) (matrix-do! i j n n (vector-set! data (+ (* i n) j) (if (= i j) 1 0))) data)",
        "(equal? (let ([n 3] [data (make-vector 9 0)]) (matrix-do! i j n n (vector-set! data (+ (* i n) j) (if (= i j) 1 0))) data) (vector 1 0 0 0 1 0 0 0 1))",
        "medium",
        ["property"],
    ),
    (
        "matrix-do!",
        "Run matrix-do! with zero rows and return the untouched empty vector.",
        "(let ([data (make-vector 0 0)]) (matrix-do! i j 0 5 (vector-set! data 0 1)) data)",
        "(equal? (let ([data (make-vector 0 0)]) (matrix-do! i j 0 5 (vector-set! data 0 1)) data) (vector))",
        "easy",
        ["edge-case"],
    ),

    # range-fold
    (
        "range-fold",
        "Compute sum of integers from 1 to 10 using range-fold.",
        "(range-fold s 0 i 1 11 (+ s i))",
        "(equal? (range-fold s 0 i 1 11 (+ s i)) 55)",
        "easy",
        ["direct"],
    ),
    (
        "range-fold",
        "Compute 5! using range-fold over [1,6).",
        "(range-fold p 1 i 1 6 (* p i))",
        "(equal? (range-fold p 1 i 1 6 (* p i)) 120)",
        "easy",
        ["direct"],
    ),
    (
        "range-fold",
        "Evaluate an empty range fold and return initial accumulator 99.",
        "(range-fold acc 99 i 5 5 (+ acc i))",
        "(equal? (range-fold acc 99 i 5 5 (+ acc i)) 99)",
        "easy",
        ["edge-case"],
    ),
    (
        "range-fold",
        "Use range-fold inside vec-tabulate to produce triangular numbers for i=0..4.",
        "(vec-tabulate 5 i (range-fold s 0 j 0 (+ i 1) (+ s j)))",
        "(equal? (vec-tabulate 5 i (range-fold s 0 j 0 (+ i 1) (+ s j))) (vector 0 1 3 6 10))",
        "hard",
        ["integration"],
    ),

    # dot-product-loop
    (
        "dot-product-loop",
        "Compute dot product of #(1 2 3) and #(4 5 6).",
        "(let ([a (vector 1 2 3)] [b (vector 4 5 6)]) (dot-product-loop k 3 (vector-ref a k) (vector-ref b k)))",
        "(equal? (let ([a (vector 1 2 3)] [b (vector 4 5 6)]) (dot-product-loop k 3 (vector-ref a k) (vector-ref b k))) 32)",
        "medium",
        ["direct"],
    ),
    (
        "dot-product-loop",
        "Compute dot product of orthogonal vectors #(1 0 0) and #(0 1 0).",
        "(let ([a (vector 1 0 0)] [b (vector 0 1 0)]) (dot-product-loop k 3 (vector-ref a k) (vector-ref b k)))",
        "(equal? (let ([a (vector 1 0 0)] [b (vector 0 1 0)]) (dot-product-loop k 3 (vector-ref a k) (vector-ref b k))) 0)",
        "medium",
        ["property"],
    ),
    (
        "dot-product-loop",
        "Tabulate #(1 2 3 4) and return its self-dot-product.",
        "(let ([v (vec-tabulate 4 i (+ i 1))]) (dot-product-loop k 4 (vector-ref v k) (vector-ref v k)))",
        "(equal? (let ([v (vec-tabulate 4 i (+ i 1))]) (dot-product-loop k 4 (vector-ref v k) (vector-ref v k))) 30)",
        "hard",
        ["integration"],
    ),
    (
        "dot-product-loop",
        "Run dot-product-loop with length 0 and return the neutral sum.",
        "(dot-product-loop k 0 1 2)",
        "(equal? (dot-product-loop k 0 1 2) 0)",
        "easy",
        ["edge-case"],
    ),

    # vec-tabulate
    (
        "vec-tabulate",
        "Build squares vector of length 5 using vec-tabulate.",
        "(vec-tabulate 5 i (* i i))",
        "(equal? (vec-tabulate 5 i (* i i)) (vector 0 1 4 9 16))",
        "easy",
        ["direct"],
    ),
    (
        "vec-tabulate",
        "Copy #(10 20 30) with vec-tabulate.",
        "(let ([v (vector 10 20 30)]) (vec-tabulate 3 i (vector-ref v i)))",
        "(equal? (let ([v (vector 10 20 30)]) (vec-tabulate 3 i (vector-ref v i))) (vector 10 20 30))",
        "medium",
        ["direct"],
    ),
    (
        "vec-tabulate",
        "Create an empty vector with vec-tabulate size 0.",
        "(vec-tabulate 0 i i)",
        "(equal? (vec-tabulate 0 i i) (vector))",
        "easy",
        ["edge-case"],
    ),
    (
        "vec-tabulate",
        "Use vec-tabulate + range-fold to generate triangular numbers #(0 1 3 6).",
        "(vec-tabulate 4 i (range-fold s 0 j 0 (+ i 1) (+ s j)))",
        "(equal? (vec-tabulate 4 i (range-fold s 0 j 0 (+ i 1) (+ s j))) (vector 0 1 3 6))",
        "hard",
        ["integration"],
    ),

    # vec-scan
    (
        "vec-scan",
        "Compute prefix sums with vec-scan where update is acc+i for size 5.",
        "(vec-scan 5 0 i acc (+ acc i))",
        "(equal? (vec-scan 5 0 i acc (+ acc i)) (vector 0 1 3 6 10))",
        "hard",
        ["direct"],
    ),
    (
        "vec-scan",
        "Run vec-scan with size 0 and return empty vector.",
        "(vec-scan 0 999 i acc (+ acc 1))",
        "(equal? (vec-scan 0 999 i acc (+ acc 1)) (vector))",
        "easy",
        ["edge-case"],
    ),
    (
        "vec-scan",
        "Run vec-scan with size 1 and init 42.",
        "(vec-scan 1 42 i acc (+ acc 1))",
        "(equal? (vec-scan 1 42 i acc (+ acc 1)) (vector 42))",
        "medium",
        ["edge-case"],
    ),
    (
        "vec-scan",
        "Use vec-scan to build fibonacci state pairs, then project first component into a vector.",
        "(let ([pairs (vec-scan 6 (cons 1 0) i acc (cons (+ (car acc) (cdr acc)) (car acc)))]) (vec-tabulate 6 j (car (vector-ref pairs j))))",
        "(equal? (let ([pairs (vec-scan 6 (cons 1 0) i acc (cons (+ (car acc) (cdr acc)) (car acc)))]) (vec-tabulate 6 j (car (vector-ref pairs j)))) (vector 1 1 2 3 5 8))",
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
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


write_jsonl(ALL_PATH, train_rows + eval_rows)
write_jsonl(TRAIN_PATH, train_rows)
write_jsonl(EVAL_PATH, eval_rows)

summary = {
    "total": len(samples),
    "train": len(train_rows),
    "eval": len(eval_rows),
    "families": {
        fam: {
            "total": len(group),
            "eval": sum(1 for x in group if x["id"] in eval_ids),
            "train": sum(1 for x in group if x["id"] not in eval_ids),
        }
        for fam, group in sorted(by_family.items())
    },
    "difficulty": dict(sorted(Counter(str(s["difficulty"]) for s in samples).items())),
    "source_functions": dict(sorted(Counter(str(s["source_function"]) for s in samples).items())),
}
SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

print(json.dumps(summary, indent=2))
print(f"Wrote: {ALL_PATH}")
print(f"Wrote: {TRAIN_PATH}")
print(f"Wrote: {EVAL_PATH}")
