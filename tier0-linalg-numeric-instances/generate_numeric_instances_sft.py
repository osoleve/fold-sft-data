#!/usr/bin/env python3
"""Generate SFT samples for lattice/linalg/numeric-instances.ss."""

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

SOURCE_MODULE = "lattice/linalg/numeric-instances.ss"
SOURCE_TEST = "lattice/linalg/test-numeric-instances.ss"

DEFS: Dict[str, str] = {
    "vec-abs": """(define (vec-abs v)
  (vec-map abs v))""",
    "vec-signum": """(define (vec-signum v)
  (vec-map (lambda (x)
                   (cond [(< x 0) -1]
                         [(> x 0) 1]
                         [else 0]))
           v))""",
    "vec-recip": """(define (vec-recip v)
  (vec-map (lambda (x) (/ 1 x)) v))""",
    "vec-pow": """(define (vec-pow v1 v2)
  (vec-zip-with expt v1 v2))""",
    "matrix-hadamard": """(define (matrix-hadamard m1 m2)
  (let ([r1 (matrix-rows m1)]
        [c1 (matrix-cols m1)]
        [r2 (matrix-rows m2)]
        [c2 (matrix-cols m2)])
       (if (or (not (= r1 r2)) (not (= c1 c2)))
           `(error dimension-mismatch (,r1 ,c1) (,r2 ,c2))
           (let ([data1 (matrix-data m1)]
                 [data2 (matrix-data m2)]
                 [result (make-vector (* r1 c1) 0)])
                (do ([i 0 (+ i 1)])
                    ((= i (* r1 c1)))
                    (vector-set! result i (* (vector-ref data1 i)
                                             (vector-ref data2 i))))
                (list 'matrix r1 c1 result)))))""",
    "matrix/": """(define (matrix/ m1 m2)
  (let ([r1 (matrix-rows m1)]
        [c1 (matrix-cols m1)]
        [r2 (matrix-rows m2)]
        [c2 (matrix-cols m2)])
       (if (or (not (= r1 r2)) (not (= c1 c2)))
           `(error dimension-mismatch (,r1 ,c1) (,r2 ,c2))
           (let ([data1 (matrix-data m1)]
                 [data2 (matrix-data m2)]
                 [result (make-vector (* r1 c1) 0)])
                (do ([i 0 (+ i 1)])
                    ((= i (* r1 c1)))
                    (vector-set! result i (/ (vector-ref data1 i)
                                             (vector-ref data2 i))))
                (list 'matrix r1 c1 result)))))""",
    "matrix-recip": """(define (matrix-recip m)
  (matrix-map (lambda (x) (/ 1 x)) m))""",
    "scalar-matrix+": """(define (scalar-matrix+ k m)
  (matrix-map (lambda (x) (+ k x)) m))""",
}

DEPENDS: Dict[str, List[str]] = {
    "vec-abs": [],
    "vec-signum": [],
    "vec-recip": [],
    "vec-pow": [],
    "matrix-hadamard": [],
    "matrix/": [],
    "matrix-recip": [],
    "scalar-matrix+": [],
}

FUNCTION_ORDER = [
    "vec-abs",
    "vec-signum",
    "vec-recip",
    "vec-pow",
    "matrix-hadamard",
    "matrix/",
    "matrix-recip",
    "scalar-matrix+",
]

FUNCTION_SPECS = {
    "vec-abs": "Apply absolute value elementwise to a numeric vector.",
    "vec-signum": "Map each element to -1, 0, or 1 depending on sign.",
    "vec-recip": "Map each element x to reciprocal 1/x.",
    "vec-pow": "Elementwise exponentiation; require equal vector lengths or return dimension mismatch error.",
    "matrix-hadamard": "Elementwise matrix multiplication; shapes must match.",
    "matrix/": "Elementwise matrix division; shapes must match.",
    "matrix-recip": "Elementwise reciprocal of matrix entries.",
    "scalar-matrix+": "Add scalar to every matrix element.",
}

SKELETONS = {
    "vec-abs": """(define (vec-abs v)
  ;; TODO: elementwise absolute value
  <TODO>)""",
    "vec-signum": """(define (vec-signum v)
  ;; TODO: map negatives to -1, positives to 1, zeros to 0
  <TODO>)""",
    "vec-recip": """(define (vec-recip v)
  ;; TODO: elementwise reciprocal
  <TODO>)""",
    "vec-pow": """(define (vec-pow v1 v2)
  ;; TODO: elementwise exponentiation with vec-zip-with
  <TODO>)""",
    "matrix-hadamard": """(define (matrix-hadamard m1 m2)
  ;; TODO: dimension check + elementwise matrix multiply
  <TODO>)""",
    "matrix/": """(define (matrix/ m1 m2)
  ;; TODO: dimension check + elementwise matrix division
  <TODO>)""",
    "matrix-recip": """(define (matrix-recip m)
  ;; TODO: map each element x to 1/x
  <TODO>)""",
    "scalar-matrix+": """(define (scalar-matrix+ k m)
  ;; TODO: add scalar k to each matrix element
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "vec-abs": "(and (equal? (vec-abs (vec -1 2 -3)) '#(1 2 3)) (equal? (vec-abs (vec)) '#()))",
    "vec-signum": "(and (equal? (vec-signum (vec -5 0 7)) '#(-1 0 1)) (equal? (vec-signum (vec 0 0)) '#(0 0)))",
    "vec-recip": "(and (equal? (vec-recip (vec 2 4 5)) '#(1/2 1/4 1/5)) (equal? (vec-recip (vec -2)) '#(-1/2)))",
    "vec-pow": "(and (equal? (vec-pow (vec 2 3 4) (vec 3 2 1)) '#(8 9 4)) (equal? (vec-pow (vec 1 2) (vec 3)) '(error dimension-mismatch 2 1)))",
    "matrix-hadamard": "(and (equal? (matrix->lists (matrix-hadamard (matrix-from-lists '((1 2) (3 4))) (matrix-from-lists '((2 3) (4 5))))) '((2 6) (12 20))) (equal? (matrix-hadamard (matrix-from-lists '((1 2))) (matrix-from-lists '((1) (2)))) '(error dimension-mismatch (1 2) (2 1))))",
    "matrix/": "(and (equal? (matrix->lists (matrix/ (matrix-from-lists '((10 20) (30 40))) (matrix-from-lists '((2 4) (5 8))))) '((5 5) (6 5))) (equal? (matrix/ (matrix-from-lists '((1 2))) (matrix-from-lists '((1) (2)))) '(error dimension-mismatch (1 2) (2 1))))",
    "matrix-recip": "(and (equal? (matrix->lists (matrix-recip (matrix-from-lists '((2 4) (5 10))))) '((1/2 1/4) (1/5 1/10))) (equal? (matrix->lists (matrix-recip (matrix-from-lists '((1))))) '((1))))",
    "scalar-matrix+": "(and (equal? (matrix->lists (scalar-matrix+ 10 (matrix-from-lists '((1 2) (3 4))))) '((11 12) (13 14))) (equal? (matrix->lists (scalar-matrix+ -1 (matrix-from-lists '((1 0))))) '((0 -1))))",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")
TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "vec-abs": "def vec_abs(v):\n    return [abs(x) for x in v]",
    "vec-signum": "def vec_signum(v):\n    out = [0] * len(v)\n    for i, x in enumerate(v):\n        if x < 0:\n            out[i] = -1\n        elif x > 0:\n            out[i] = 1\n        else:\n            out[i] = 0\n    return out",
    "vec-recip": "def vec_recip(v):\n    return [1 / x for x in v]",
    "vec-pow": "def vec_pow(v1, v2):\n    if len(v1) != len(v2):\n        return ['error', 'dimension-mismatch', len(v1), len(v2)]\n    out = [0] * len(v1)\n    for i in range(len(v1)):\n        out[i] = v1[i] ** v2[i]\n    return out",
    "matrix-hadamard": "def matrix_hadamard(m1, m2):\n    r1, c1 = matrix_rows(m1), matrix_cols(m1)\n    r2, c2 = matrix_rows(m2), matrix_cols(m2)\n    if r1 != r2 or c1 != c2:\n        return ['error', 'dimension-mismatch', [r1, c1], [r2, c2]]\n    d1, d2 = matrix_data(m1), matrix_data(m2)\n    out = [0] * (r1 * c1)\n    for i in range(r1 * c1):\n        out[i] = d1[i] * d2[i]\n    return ['matrix', r1, c1, list_to_vector(out)]",
    "matrix/": "def matrix_div(m1, m2):\n    r1, c1 = matrix_rows(m1), matrix_cols(m1)\n    r2, c2 = matrix_rows(m2), matrix_cols(m2)\n    if r1 != r2 or c1 != c2:\n        return ['error', 'dimension-mismatch', [r1, c1], [r2, c2]]\n    d1, d2 = matrix_data(m1), matrix_data(m2)\n    out = [0] * (r1 * c1)\n    for i in range(r1 * c1):\n        out[i] = d1[i] / d2[i]\n    return ['matrix', r1, c1, list_to_vector(out)]",
    "matrix-recip": "def matrix_recip(m):\n    return matrix_map(lambda x: 1 / x, m)",
    "scalar-matrix+": "def scalar_matrix_add(k, m):\n    return matrix_map(lambda x: k + x, m)",
}

CHEZ_SNIPPETS = {
    "vec-abs": "(define (vabs v)\n  (vec-map abs v))",
    "vec-signum": "(define (vsign v)\n  (vec-map (lambda (x)\n             (cond ((< x 0) -1)\n                   ((> x 0) 1)\n                   (else 0)))\n           v))",
    "vec-recip": "(define (vrecip v)\n  (vec-map (lambda (x) (/ 1 x)) v))",
    "vec-pow": "(define (vpow a b)\n  (vec-zip-with expt a b))",
    "matrix-hadamard": "(define (mhadamard a b)\n  (let ((r1 (matrix-rows a)) (c1 (matrix-cols a))\n        (r2 (matrix-rows b)) (c2 (matrix-cols b)))\n    (if (or (not (= r1 r2)) (not (= c1 c2)))\n        `(error dimension-mismatch (,r1 ,c1) (,r2 ,c2))\n        (let ((d1 (matrix-data a)) (d2 (matrix-data b))\n              (out (make-vector (* r1 c1) 0)))\n          (do ((i 0 (+ i 1)))\n              ((= i (* r1 c1)))\n            (vector-set! out i (* (vector-ref d1 i) (vector-ref d2 i))))\n          (list 'matrix r1 c1 out)))))",
    "matrix/": "(define (mdiv a b)\n  (let ((r1 (matrix-rows a)) (c1 (matrix-cols a))\n        (r2 (matrix-rows b)) (c2 (matrix-cols b)))\n    (if (or (not (= r1 r2)) (not (= c1 c2)))\n        `(error dimension-mismatch (,r1 ,c1) (,r2 ,c2))\n        (let ((d1 (matrix-data a)) (d2 (matrix-data b))\n              (out (make-vector (* r1 c1) 0)))\n          (do ((i 0 (+ i 1)))\n              ((= i (* r1 c1)))\n            (vector-set! out i (/ (vector-ref d1 i) (vector-ref d2 i))))\n          (list 'matrix r1 c1 out)))))",
    "matrix-recip": "(define (mrecip m)\n  (matrix-map (lambda (x) (/ 1 x)) m))",
    "scalar-matrix+": "(define (s+m k m)\n  (matrix-map (lambda (x) (+ k x)) m))",
}

BUGGY_CASES = [
    {
        "fn": "vec-abs",
        "buggy": "(define (vec-abs v)\n  (vec-map (lambda (x) (- x)) v))",
        "note": "Absolute value is not simple negation for positive entries.",
    },
    {
        "fn": "vec-abs",
        "buggy": "(define (vec-abs v)\n  (vec-map (lambda (x)\n                   (if (< x 0)\n                       (- x)\n                       (+ x 1)))\n           v))",
        "note": "Positive values should be unchanged by absolute value; they must not be incremented.",
    },
    {
        "fn": "vec-signum",
        "buggy": "(define (vec-signum v)\n  (vec-map (lambda (x)\n                   (cond [(< x 0) -1]\n                         [else 1]))\n           v))",
        "note": "Zero must map to 0, not 1.",
    },
    {
        "fn": "vec-signum",
        "buggy": "(define (vec-signum v)\n  (vec-map (lambda (x)\n                   (cond [(< x 0) 1]\n                         [(> x 0) -1]\n                         [else 0]))\n           v))",
        "note": "Signs for positive/negative cases are swapped.",
    },
    {
        "fn": "vec-recip",
        "buggy": "(define (vec-recip v)\n  (vec-map (lambda (x) (/ 1 (abs x))) v))",
        "note": "Reciprocal must preserve sign; using abs loses negative signs.",
    },
    {
        "fn": "vec-recip",
        "buggy": "(define (vec-recip v)\n  (vec-map (lambda (x) (/ x 1)) v))",
        "note": "Division direction is reversed.",
    },
    {
        "fn": "vec-pow",
        "buggy": "(define (vec-pow v1 v2)\n  (vec-zip-with + v1 v2))",
        "note": "Operation should exponentiate elementwise, not add.",
    },
    {
        "fn": "vec-pow",
        "buggy": "(define (vec-pow v1 v2)\n  (if (= (vector-length v1) (vector-length v2))\n      '(error dimension-mismatch)\n      (vec-zip-with expt v1 v2)))",
        "note": "Dimension mismatch condition is inverted.",
    },
    {
        "fn": "matrix-hadamard",
        "buggy": "(define (matrix-hadamard m1 m2)\n  (matrix-add m1 m2))",
        "note": "Hadamard product is elementwise multiplication, not addition.",
    },
    {
        "fn": "matrix-hadamard",
        "buggy": "(define (matrix-hadamard m1 m2)\n  (let ([r1 (matrix-rows m1)] [c1 (matrix-cols m1)] [r2 (matrix-rows m2)] [c2 (matrix-cols m2)])\n    (if (and (= r1 r2) (= c1 c2))\n        `(error dimension-mismatch (,r1 ,c1) (,r2 ,c2))\n        (list 'matrix r1 c1 (make-vector (* r1 c1) 0)))))",
        "note": "Dimension check should reject mismatched shapes, not matched ones.",
    },
    {
        "fn": "matrix/",
        "buggy": "(define (matrix/ m1 m2)\n  (matrix-hadamard m1 m2))",
        "note": "Elementwise division is required, not multiplication.",
    },
    {
        "fn": "matrix/",
        "buggy": "(define (matrix/ m1 m2)\n  (let ([r1 (matrix-rows m1)] [c1 (matrix-cols m1)] [r2 (matrix-rows m2)] [c2 (matrix-cols m2)])\n    (if (or (not (= r1 r2)) (not (= c1 c2)))\n        `(error dimension-mismatch (,r1 ,c1) (,r2 ,c2))\n        (let ([data1 (matrix-data m1)] [data2 (matrix-data m2)] [result (make-vector (* r1 c1) 0)])\n          (do ([i 0 (+ i 1)])\n              ((= i (* r1 c1)))\n            (vector-set! result i (/ (vector-ref data2 i) (vector-ref data1 i))))\n          (list 'matrix r1 c1 result)))))",
        "note": "Numerator and denominator order is reversed.",
    },
    {
        "fn": "matrix-recip",
        "buggy": "(define (matrix-recip m)\n  (matrix-map (lambda (x) (/ -1 x)) m))",
        "note": "Reciprocals should be 1/x, not negated reciprocals.",
    },
    {
        "fn": "matrix-recip",
        "buggy": "(define (matrix-recip m)\n  (matrix-map (lambda (x) (/ x 1)) m))",
        "note": "Division direction is wrong; this leaves the matrix unchanged.",
    },
    {
        "fn": "scalar-matrix+",
        "buggy": "(define (scalar-matrix+ k m)\n  (matrix-map (lambda (x) (* k x)) m))",
        "note": "Function should add scalar, not scale by multiplication.",
    },
    {
        "fn": "scalar-matrix+",
        "buggy": "(define (scalar-matrix+ k m)\n  (matrix-map (lambda (x) (+ x x)) m))",
        "note": "Mapped expression must use scalar k, not duplicate x.",
    },
]

DIFFICULTY = {
    "vec-abs": "easy",
    "vec-signum": "medium",
    "vec-recip": "easy",
    "vec-pow": "medium",
    "matrix-hadamard": "medium",
    "matrix/": "medium",
    "matrix-recip": "easy",
    "scalar-matrix+": "easy",
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
    sid = f"numeric_instances_{family}_{family_counter[family]:03d}"
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
        prompt=f"""Implement this function in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["linalg", "numeric-instances", "spec-to-code", fn],
    )

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Complete this Fold Scheme skeleton.

Module: {SOURCE_MODULE}
Function target: `{fn}`
Behavior contract: {FUNCTION_SPECS[fn]}

```scheme
{SKELETONS[fn]}
```

Output only the completed function definition.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["linalg", "numeric-instances", "spec-to-code", "skeleton", fn],
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
        prompt=f"""Translate the following Python function into Fold-native Scheme.
Preserve behavior exactly, including dimension mismatch behavior.

Target function name: `{fn}`

```python
{PYTHON_SNIPPETS[fn]}
```

Return only the Scheme definition.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["linalg", "numeric-instances", "translation", "python", fn],
    )

    add_sample(
        family="translation",
        category="transpile",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Convert this Chez-style snippet to canonical Fold style.
Keep semantics identical.

Target function: `{fn}`

```scheme
{CHEZ_SNIPPETS[fn]}
```

Return only Fold code.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["linalg", "numeric-instances", "translation", "chez", fn],
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
        prompt=f"""Fix the bug in this Fold Scheme function with minimal semantic changes.
Target: `{fn}` in `{SOURCE_MODULE}`.
Known issue: {case['note']}

```scheme
{case['buggy']}
```

Return only the corrected definition.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["linalg", "numeric-instances", "bugfix", fn],
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
        tags=["linalg", "numeric-instances", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # vec-abs
    ("vec-abs", "Apply vec-abs to #( -1 2 -3 ).", "(vec-abs (vec -1 2 -3))", "(equal? (vec-abs (vec -1 2 -3)) '#(1 2 3))", "easy", ["direct"]),
    ("vec-abs", "Apply vec-abs to empty vector.", "(vec-abs (vec))", "(equal? (vec-abs (vec)) '#())", "easy", ["edge-case"]),
    ("vec-abs", "Compose vec-abs then vec-recip on #( -2 -4 ).", "(vec-recip (vec-abs (vec -2 -4)))", "(equal? (vec-recip (vec-abs (vec -2 -4))) '#(1/2 1/4))", "medium", ["integration"]),
    ("vec-abs", "Return #t iff sum of abs values for #( -1 2 -3 ) is 6.", "(= (vec-sum (vec-abs (vec -1 2 -3))) 6)", "(equal? (= (vec-sum (vec-abs (vec -1 2 -3))) 6) #t)", "medium", ["property"]),

    # vec-signum
    ("vec-signum", "Apply vec-signum to #( -5 0 7 ).", "(vec-signum (vec -5 0 7))", "(equal? (vec-signum (vec -5 0 7)) '#(-1 0 1))", "medium", ["direct"]),
    ("vec-signum", "Apply vec-signum to all-zero vector.", "(vec-signum (vec 0 0 0))", "(equal? (vec-signum (vec 0 0 0)) '#(0 0 0))", "easy", ["edge-case"]),
    ("vec-signum", "Apply vec-signum after vec-abs on #( -2 0 3 ).", "(vec-signum (vec-abs (vec -2 0 3)))", "(equal? (vec-signum (vec-abs (vec -2 0 3))) '#(1 0 1))", "medium", ["integration"]),
    ("vec-signum", "Return #t iff vec-signum outputs only values in {-1, 0, 1} on sample.", "(let ([v (vec-signum (vec -9 -1 0 4 5))]) (null? (filter (lambda (x) (not (or (= x -1) (= x 0) (= x 1)))) (vector->list v))))", "(equal? (let ([v (vec-signum (vec -9 -1 0 4 5))]) (null? (filter (lambda (x) (not (or (= x -1) (= x 0) (= x 1)))) (vector->list v)))) #t)", "medium", ["property"]),

    # vec-recip
    ("vec-recip", "Apply vec-recip to #(2 4 5).", "(vec-recip (vec 2 4 5))", "(equal? (vec-recip (vec 2 4 5)) '#(1/2 1/4 1/5))", "easy", ["direct"]),
    ("vec-recip", "Apply vec-recip to single negative element.", "(vec-recip (vec -2))", "(equal? (vec-recip (vec -2)) '#(-1/2))", "easy", ["direct"]),
    ("vec-recip", "Compose vec-recip after vec-pow #(2 3)^(#(2 1)).", "(vec-recip (vec-pow (vec 2 3) (vec 2 1)))", "(equal? (vec-recip (vec-pow (vec 2 3) (vec 2 1))) '#(1/4 1/3))", "medium", ["integration"]),
    ("vec-recip", "Return #t iff reciprocals multiplied by originals are ones for sample.", "(let ([v (vec 2 5)]) (vec-zip-with * (vec-recip v) v))", "(equal? (let ([v (vec 2 5)]) (vec-zip-with * (vec-recip v) v)) '#(1 1))", "hard", ["property"]),

    # vec-pow
    ("vec-pow", "Elementwise power #(2 3 4)^(#(3 2 1)).", "(vec-pow (vec 2 3 4) (vec 3 2 1))", "(equal? (vec-pow (vec 2 3 4) (vec 3 2 1)) '#(8 9 4))", "medium", ["direct"]),
    ("vec-pow", "Return dimension mismatch on vec-pow length mismatch.", "(vec-pow (vec 1 2) (vec 3))", "(equal? (vec-pow (vec 1 2) (vec 3)) '(error dimension-mismatch 2 1))", "easy", ["edge-case"]),
    ("vec-pow", "Compose vec-pow then vec-abs with negative bases and odd/even exponents.", "(vec-abs (vec-pow (vec -2 -3) (vec 2 3)))", "(equal? (vec-abs (vec-pow (vec -2 -3) (vec 2 3))) '#(4 27))", "hard", ["integration"]),
    ("vec-pow", "Return #t iff x^1 returns x for sample vector.", "(equal? (vec-pow (vec 5 -2 9) (vec 1 1 1)) (vec 5 -2 9))", "(equal? (vec-pow (vec 5 -2 9) (vec 1 1 1)) (vec 5 -2 9))", "medium", ["property"]),

    # matrix-hadamard
    ("matrix-hadamard", "Hadamard multiply two 2x2 matrices.", "(matrix->lists (matrix-hadamard (matrix-from-lists '((1 2) (3 4))) (matrix-from-lists '((2 3) (4 5)))))", "(equal? (matrix->lists (matrix-hadamard (matrix-from-lists '((1 2) (3 4))) (matrix-from-lists '((2 3) (4 5))))) '((2 6) (12 20)))", "medium", ["direct"]),
    ("matrix-hadamard", "Return dimension mismatch for incompatible hadamard inputs.", "(matrix-hadamard (matrix-from-lists '((1 2))) (matrix-from-lists '((1) (2))))", "(equal? (matrix-hadamard (matrix-from-lists '((1 2))) (matrix-from-lists '((1) (2)))) '(error dimension-mismatch (1 2) (2 1)))", "easy", ["edge-case"]),
    ("matrix-hadamard", "Compose scalar-matrix+ then matrix-hadamard.", "(matrix->lists (matrix-hadamard (scalar-matrix+ 1 (matrix-from-lists '((1 2) (3 4)))) (matrix-from-lists '((2 2) (2 2)))))", "(equal? (matrix->lists (matrix-hadamard (scalar-matrix+ 1 (matrix-from-lists '((1 2) (3 4)))) (matrix-from-lists '((2 2) (2 2))))) '((4 6) (8 10)))", "hard", ["integration"]),
    ("matrix-hadamard", "Return #t iff hadamard with all-ones matrix preserves input.", "(equal? (matrix->lists (matrix-hadamard (matrix-from-lists '((5 6) (7 8))) (matrix-from-lists '((1 1) (1 1)))) ) '((5 6) (7 8)))", "(equal? (matrix->lists (matrix-hadamard (matrix-from-lists '((5 6) (7 8))) (matrix-from-lists '((1 1) (1 1)))) ) '((5 6) (7 8)))", "medium", ["property"]),

    # matrix/
    ("matrix/", "Elementwise divide ((10 20)(30 40)) by ((2 4)(5 8)).", "(matrix->lists (matrix/ (matrix-from-lists '((10 20) (30 40))) (matrix-from-lists '((2 4) (5 8)))))", "(equal? (matrix->lists (matrix/ (matrix-from-lists '((10 20) (30 40))) (matrix-from-lists '((2 4) (5 8))))) '((5 5) (6 5)))", "medium", ["direct"]),
    ("matrix/", "Return dimension mismatch on matrix/ shape mismatch.", "(matrix/ (matrix-from-lists '((1 2))) (matrix-from-lists '((1) (2))))", "(equal? (matrix/ (matrix-from-lists '((1 2))) (matrix-from-lists '((1) (2)))) '(error dimension-mismatch (1 2) (2 1)))", "easy", ["edge-case"]),
    ("matrix/", "Compose matrix-hadamard with matrix/ to recover original matrix.", "(matrix->lists (matrix/ (matrix-hadamard (matrix-from-lists '((2 3) (4 5))) (matrix-from-lists '((7 11) (13 17)))) (matrix-from-lists '((7 11) (13 17)))))", "(equal? (matrix->lists (matrix/ (matrix-hadamard (matrix-from-lists '((2 3) (4 5))) (matrix-from-lists '((7 11) (13 17)))) (matrix-from-lists '((7 11) (13 17))))) '((2 3) (4 5)))", "hard", ["integration"]),
    ("matrix/", "Return #t iff dividing matrix by itself yields all ones.", "(equal? (matrix->lists (matrix/ (matrix-from-lists '((2 4) (5 10))) (matrix-from-lists '((2 4) (5 10))))) '((1 1) (1 1)))", "(equal? (matrix->lists (matrix/ (matrix-from-lists '((2 4) (5 10))) (matrix-from-lists '((2 4) (5 10))))) '((1 1) (1 1)))", "medium", ["property"]),

    # matrix-recip
    ("matrix-recip", "Apply matrix-recip to ((2 4)(5 10)).", "(matrix->lists (matrix-recip (matrix-from-lists '((2 4) (5 10)))))", "(equal? (matrix->lists (matrix-recip (matrix-from-lists '((2 4) (5 10))))) '((1/2 1/4) (1/5 1/10)))", "easy", ["direct"]),
    ("matrix-recip", "Apply matrix-recip to singleton ((1)).", "(matrix->lists (matrix-recip (matrix-from-lists '((1)))))", "(equal? (matrix->lists (matrix-recip (matrix-from-lists '((1))))) '((1)))", "easy", ["edge-case"]),
    ("matrix-recip", "Compose matrix-recip with matrix/ by dividing ones matrix by source.", "(equal? (matrix->lists (matrix-recip (matrix-from-lists '((2 4) (5 10))))) (matrix->lists (matrix/ (matrix-from-lists '((1 1) (1 1))) (matrix-from-lists '((2 4) (5 10))))))", "(equal? (matrix->lists (matrix-recip (matrix-from-lists '((2 4) (5 10))))) (matrix->lists (matrix/ (matrix-from-lists '((1 1) (1 1))) (matrix-from-lists '((2 4) (5 10))))))", "medium", ["integration"]),
    ("matrix-recip", "Return #t iff reciprocal of reciprocal recovers original for sample.", "(equal? (matrix->lists (matrix-recip (matrix-recip (matrix-from-lists '((2 4) (5 10)))))) '((2 4) (5 10)))", "(equal? (matrix->lists (matrix-recip (matrix-recip (matrix-from-lists '((2 4) (5 10)))))) '((2 4) (5 10)))", "hard", ["property"]),

    # scalar-matrix+
    ("scalar-matrix+", "Add scalar 10 to matrix ((1 2)(3 4)).", "(matrix->lists (scalar-matrix+ 10 (matrix-from-lists '((1 2) (3 4)))))", "(equal? (matrix->lists (scalar-matrix+ 10 (matrix-from-lists '((1 2) (3 4))))) '((11 12) (13 14)))", "easy", ["direct"]),
    ("scalar-matrix+", "Add scalar -1 to matrix ((1 0)).", "(matrix->lists (scalar-matrix+ -1 (matrix-from-lists '((1 0)))))", "(equal? (matrix->lists (scalar-matrix+ -1 (matrix-from-lists '((1 0))))) '((0 -1)))", "easy", ["direct"]),
    ("scalar-matrix+", "Compose scalar-matrix+ then matrix-hadamard with ones.", "(matrix->lists (matrix-hadamard (scalar-matrix+ 3 (matrix-from-lists '((1 2) (3 4)))) (matrix-from-lists '((1 1) (1 1)))))", "(equal? (matrix->lists (matrix-hadamard (scalar-matrix+ 3 (matrix-from-lists '((1 2) (3 4)))) (matrix-from-lists '((1 1) (1 1))))) '((4 5) (6 7)))", "hard", ["integration"]),
    ("scalar-matrix+", "Return #t iff adding 0 leaves matrix unchanged.", "(equal? (matrix->lists (scalar-matrix+ 0 (matrix-from-lists '((9 8) (7 6))))) '((9 8) (7 6)))", "(equal? (matrix->lists (scalar-matrix+ 0 (matrix-from-lists '((9 8) (7 6))))) '((9 8) (7 6)))", "easy", ["property"]),
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
