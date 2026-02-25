#!/usr/bin/env python3
"""Generate SFT samples for lattice/linalg/vec.ss."""

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

SOURCE_MODULE = "lattice/linalg/vec.ss"
SOURCE_TEST = "lattice/linalg/test-vec.ss"

DEFS: Dict[str, str] = {
    "vec-ref": """(define (vec-ref v i)
  (if (and (>= i 0) (< i (vector-length v)))
      (vector-ref v i)
      `(error out-of-bounds ,i ,(vector-length v))))""",
    "vec-first": """(define (vec-first v)
  (if (> (vector-length v) 0)
      (vector-ref v 0)
      '(error empty-vector)))""",
    "vec-map": """(define (vec-map f v)
  (vec-map-idx i v (f (vector-ref v i))))""",
    "vec-zip-with": """(define (vec-zip-with f v1 v2)
  (let ([n1 (vector-length v1)]
        [n2 (vector-length v2)])
       (if (not (= n1 n2))
           `(error dimension-mismatch ,n1 ,n2)
           (vec-zip-map-idx i v1 v2 (f (vector-ref v1 i) (vector-ref v2 i))))))""",
    "vec-slice": """(define (vec-slice v start end)
  (let ([len (vector-length v)])
       (cond
        [(< start 0) `(error out-of-bounds ,start ,len)]
        [(> end len) `(error out-of-bounds ,end ,len)]
        [(> start end) `(error invalid-range ,start ,end)]
        [else
         (let ([new-len (- end start)]
               [src v]
               [offset start])
              (vec-tabulate new-len i (vector-ref src (+ offset i))))])))""",
    "vec-dot": """(define (vec-dot v1 v2)
  (let ([n1 (vector-length v1)]
        [n2 (vector-length v2)])
       (if (not (= n1 n2))
           `(error dimension-mismatch ,n1 ,n2)
           (dot-product-loop i n1 (vector-ref v1 i) (vector-ref v2 i)))))""",
    "vec-norm-squared": """(define (vec-norm-squared v)
  (vec-dot v v))""",
    "vec-norm": """(define (vec-norm v)
  (sqrt (vec-norm-squared v)))""",
    "vec-scale": """(define (vec-scale k v)
  (vec-map (lambda (x) (* k x)) v))""",
    "vec-normalize": """(define (vec-normalize v)
  (let ([n (vec-norm v)])
       (if (= n 0)
           '(error zero-vector)
           (vec-scale (/ 1 n) v))))""",
    "vec-approx-equal?": """(define (vec-approx-equal? v1 v2 . epsilon-arg)
  (let ([epsilon (if (null? epsilon-arg) 1e-10 (car epsilon-arg))])
       (and (= (vector-length v1) (vector-length v2))
            (let ([n (vector-length v1)])
                 (do ([i 0 (+ i 1)]
                      [eq #t (and eq (< (abs (- (vector-ref v1 i)
                                                (vector-ref v2 i)))
                                        epsilon))])
                     ((or (= i n) (not eq)) eq))))))""",
}

DEPENDS: Dict[str, List[str]] = {
    "vec-ref": [],
    "vec-first": [],
    "vec-map": [],
    "vec-zip-with": [],
    "vec-slice": [],
    "vec-dot": [],
    "vec-norm-squared": ["vec-dot"],
    "vec-norm": ["vec-norm-squared"],
    "vec-scale": ["vec-map"],
    "vec-normalize": ["vec-norm", "vec-scale"],
    "vec-approx-equal?": [],
}

FUNCTION_ORDER = [
    "vec-ref",
    "vec-first",
    "vec-map",
    "vec-zip-with",
    "vec-slice",
    "vec-dot",
    "vec-normalize",
    "vec-approx-equal?",
]

FUNCTION_SPECS = {
    "vec-ref": "Return element at index i, or `(error out-of-bounds i len)` when i is invalid.",
    "vec-first": "Return first element of vector, or `(error empty-vector)` for empty vectors.",
    "vec-map": "Return a new vector by applying function f to each element in v.",
    "vec-zip-with": "Zip two equal-length vectors with f, otherwise return dimension-mismatch error.",
    "vec-slice": "Return subvector from start (inclusive) to end (exclusive) with bounds checks.",
    "vec-dot": "Compute dot product for equal-length numeric vectors, else return dimension-mismatch error.",
    "vec-normalize": "Return unit-length vector in same direction, or `(error zero-vector)` if norm is zero.",
    "vec-approx-equal?": "Approximate vector equality with optional epsilon argument (default 1e-10).",
}

SKELETONS = {
    "vec-ref": """(define (vec-ref v i)
  ;; TODO: bounds-check and return element or error
  <TODO>)""",
    "vec-first": """(define (vec-first v)
  ;; TODO: handle empty vector case
  <TODO>)""",
    "vec-map": """(define (vec-map f v)
  ;; TODO: map f across all elements
  <TODO>)""",
    "vec-zip-with": """(define (vec-zip-with f v1 v2)
  ;; TODO: enforce equal lengths, then zip with f
  <TODO>)""",
    "vec-slice": """(define (vec-slice v start end)
  ;; TODO: validate range and return subvector
  <TODO>)""",
    "vec-dot": """(define (vec-dot v1 v2)
  ;; TODO: dimension check and dot-product accumulation
  <TODO>)""",
    "vec-normalize": """(define (vec-normalize v)
  ;; TODO: normalize unless norm is zero
  <TODO>)""",
    "vec-approx-equal?": """(define (vec-approx-equal? v1 v2 . epsilon-arg)
  ;; TODO: optional epsilon + elementwise tolerance check
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "vec-ref": "(and (= (vec-ref (vector 10 20 30) 1) 20) (equal? (vec-ref (vector 10 20 30) 3) '(error out-of-bounds 3 3)))",
    "vec-first": "(and (= (vec-first (vector 5 6 7)) 5) (equal? (vec-first (vector)) '(error empty-vector)))",
    "vec-map": "(and (equal? (vec-map (lambda (x) (* 2 x)) (vector 1 2 3)) (vector 2 4 6)) (equal? (vec-map (lambda (x) (+ x 1)) (vector)) (vector)))",
    "vec-zip-with": "(and (equal? (vec-zip-with + (vector 1 2 3) (vector 4 5 6)) (vector 5 7 9)) (equal? (vec-zip-with + (vector 1 2) (vector 9)) '(error dimension-mismatch 2 1)))",
    "vec-slice": "(and (equal? (vec-slice (vector 1 2 3 4 5) 1 4) (vector 2 3 4)) (equal? (vec-slice (vector 1 2 3) 2 1) '(error invalid-range 2 1)) (equal? (vec-slice (vector 1 2 3) 0 3) (vector 1 2 3)))",
    "vec-dot": "(and (= (vec-dot (vector 1 2 3) (vector 4 5 6)) 32) (equal? (vec-dot (vector 1 2) (vector 9)) '(error dimension-mismatch 2 1)))",
    "vec-normalize": "(and (equal? (vec-normalize (vector 1 0 0)) (vector 1 0 0)) (equal? (vec-normalize (vector 0 0 0)) '(error zero-vector)))",
    "vec-approx-equal?": "(and (vec-approx-equal? (vector 1.0 2.0) (vector 1.00000000001 2.0)) (not (vec-approx-equal? (vector 1.0 2.0) (vector 1.1 2.0))) (vec-approx-equal? (vector 1.0) (vector 1.05) 0.1))",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")
TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "vec-ref": "def vec_ref(v, i):\n    if 0 <= i < len(v):\n        return v[i]\n    return ['error', 'out-of-bounds', i, len(v)]",
    "vec-first": "def vec_first(v):\n    if len(v) > 0:\n        return v[0]\n    return ['error', 'empty-vector']",
    "vec-map": "def vec_map(f, v):\n    out = [0] * len(v)\n    for i in range(len(v)):\n        out[i] = f(v[i])\n    return out",
    "vec-zip-with": "def vec_zip_with(f, v1, v2):\n    if len(v1) != len(v2):\n        return ['error', 'dimension-mismatch', len(v1), len(v2)]\n    out = [0] * len(v1)\n    for i in range(len(v1)):\n        out[i] = f(v1[i], v2[i])\n    return out",
    "vec-slice": "def vec_slice(v, start, end):\n    n = len(v)\n    if start < 0:\n        return ['error', 'out-of-bounds', start, n]\n    if end > n:\n        return ['error', 'out-of-bounds', end, n]\n    if start > end:\n        return ['error', 'invalid-range', start, end]\n    out = [0] * (end - start)\n    for i in range(end - start):\n        out[i] = v[start + i]\n    return out",
    "vec-dot": "def vec_dot(v1, v2):\n    if len(v1) != len(v2):\n        return ['error', 'dimension-mismatch', len(v1), len(v2)]\n    s = 0\n    for i in range(len(v1)):\n        s += v1[i] * v2[i]\n    return s",
    "vec-normalize": "def vec_normalize(v):\n    n2 = 0\n    for x in v:\n        n2 += x * x\n    n = n2 ** 0.5\n    if n == 0:\n        return ['error', 'zero-vector']\n    return [(1 / n) * x for x in v]",
    "vec-approx-equal?": "def vec_approx_equal(v1, v2, epsilon=1e-10):\n    if len(v1) != len(v2):\n        return False\n    for i in range(len(v1)):\n        if abs(v1[i] - v2[i]) >= epsilon:\n            return False\n    return True",
}

CHEZ_SNIPPETS = {
    "vec-ref": "(define (vref v i)\n  (if (and (>= i 0) (< i (vector-length v)))\n      (vector-ref v i)\n      `(error out-of-bounds ,i ,(vector-length v))))",
    "vec-first": "(define (vfirst v)\n  (if (> (vector-length v) 0)\n      (vector-ref v 0)\n      '(error empty-vector)))",
    "vec-map": "(define (vmap f v)\n  (vec-map-idx i v (f (vector-ref v i))))",
    "vec-zip-with": "(define (vzip f a b)\n  (let ((n1 (vector-length a))\n        (n2 (vector-length b)))\n    (if (not (= n1 n2))\n        `(error dimension-mismatch ,n1 ,n2)\n        (vec-zip-map-idx i a b (f (vector-ref a i) (vector-ref b i))))))",
    "vec-slice": "(define (vslice v s e)\n  (let ((len (vector-length v)))\n    (cond ((< s 0) `(error out-of-bounds ,s ,len))\n          ((> e len) `(error out-of-bounds ,e ,len))\n          ((> s e) `(error invalid-range ,s ,e))\n          (else\n           (let ((n (- e s))\n                 (src v)\n                 (off s))\n             (vec-tabulate n i (vector-ref src (+ off i))))))))",
    "vec-dot": "(define (vdot a b)\n  (let ((n1 (vector-length a))\n        (n2 (vector-length b)))\n    (if (not (= n1 n2))\n        `(error dimension-mismatch ,n1 ,n2)\n        (dot-product-loop i n1 (vector-ref a i) (vector-ref b i)))))",
    "vec-normalize": "(define (vnormalize v)\n  (let ((n (sqrt (vec-dot v v))))\n    (if (= n 0)\n        '(error zero-vector)\n        (vec-map (lambda (x) (* (/ 1 n) x)) v))))",
    "vec-approx-equal?": "(define (vapprox? a b . eps-arg)\n  (let ((eps (if (null? eps-arg) 1e-10 (car eps-arg))))\n    (and (= (vector-length a) (vector-length b))\n         (let ((n (vector-length a)))\n           (do ((i 0 (+ i 1))\n                (ok #t (and ok (< (abs (- (vector-ref a i) (vector-ref b i))) eps))))\n               ((or (= i n) (not ok)) ok))))))",
}

BUGGY_CASES = [
    {
        "fn": "vec-ref",
        "buggy": "(define (vec-ref v i)\n  (if (and (>= i 0) (< i (- (vector-length v) 1)))\n      (vector-ref v i)\n      `(error out-of-bounds ,i ,(vector-length v))))",
        "note": "Upper-bound check is off by one; last valid index should be accepted.",
    },
    {
        "fn": "vec-ref",
        "buggy": "(define (vec-ref v i)\n  (if (and (>= i 0) (< i (vector-length v)))\n      (vector-ref v i)\n      `(error out-of-bounds ,(vector-length v) ,i)))",
        "note": "Error payload order is wrong; keep `(error out-of-bounds i len)`.",
    },
    {
        "fn": "vec-first",
        "buggy": "(define (vec-first v)\n  (if (> (vector-length v) 0)\n      (vector-ref v (- (vector-length v) 1))\n      '(error empty-vector)))",
        "note": "Function should return first element, not last.",
    },
    {
        "fn": "vec-first",
        "buggy": "(define (vec-first v)\n  (if (> (vector-length v) 0)\n      (vector-ref v 0)\n      0))",
        "note": "Empty vectors must return `(error empty-vector)`.",
    },
    {
        "fn": "vec-map",
        "buggy": "(define (vec-map f v)\n  (vec-map-idx i v (vector-ref v i)))",
        "note": "Mapped function `f` is ignored; each output element must be transformed by `f`.",
    },
    {
        "fn": "vec-map",
        "buggy": "(define (vec-map f v)\n  (vec-map-idx i v (f i)))",
        "note": "Mapping input should be vector values, not indices.",
    },
    {
        "fn": "vec-zip-with",
        "buggy": "(define (vec-zip-with f v1 v2)\n  (let ([n1 (vector-length v1)] [n2 (vector-length v2)])\n    (if (= n1 n2)\n        `(error dimension-mismatch ,n1 ,n2)\n        (vec-zip-map-idx i v1 v2 (f (vector-ref v1 i) (vector-ref v2 i))))))",
        "note": "Dimension check is inverted.",
    },
    {
        "fn": "vec-zip-with",
        "buggy": "(define (vec-zip-with f v1 v2)\n  (let ([n1 (vector-length v1)] [n2 (vector-length v2)])\n    (if (not (= n1 n2))\n        `(error dimension-mismatch ,n1 ,n2)\n        (vec-zip-map-idx i v1 v2 (f (vector-ref v1 i) (vector-ref v1 i))))))",
        "note": "Zipping should use one element from each vector.",
    },
    {
        "fn": "vec-slice",
        "buggy": "(define (vec-slice v start end)\n  (let ([len (vector-length v)])\n    (cond\n      [(< start 0) `(error out-of-bounds ,start ,len)]\n      [(> end len) `(error out-of-bounds ,end ,len)]\n      [else\n       (let ([new-len (+ 1 (- end start))] [src v] [offset start])\n         (vec-tabulate new-len i (vector-ref src (+ offset i))))])))",
        "note": "Slice length should be `end - start`, not inclusive of end.",
    },
    {
        "fn": "vec-slice",
        "buggy": "(define (vec-slice v start end)\n  (let ([len (vector-length v)])\n    (cond\n      [(< start 0) `(error out-of-bounds ,start ,len)]\n      [(> end len) `(error out-of-bounds ,end ,len)]\n      [(> start end) `(error invalid-range ,end ,start)]\n      [else\n       (let ([new-len (- end start)] [src v] [offset start])\n         (vec-tabulate new-len i (vector-ref src (+ offset i))))])))",
        "note": "Invalid-range error payload should preserve `(start end)` order.",
    },
    {
        "fn": "vec-dot",
        "buggy": "(define (vec-dot v1 v2)\n  (let ([n1 (vector-length v1)] [n2 (vector-length v2)])\n    (if (not (= n1 n2))\n        0\n        (dot-product-loop i n1 (vector-ref v1 i) (vector-ref v2 i)))))",
        "note": "Dimension mismatch must return an error value, not 0.",
    },
    {
        "fn": "vec-dot",
        "buggy": "(define (vec-dot v1 v2)\n  (let ([n1 (vector-length v1)] [n2 (vector-length v2)])\n    (if (not (= n1 n2))\n        `(error dimension-mismatch ,n1 ,n2)\n        (dot-product-loop i n1 (vector-ref v1 i) (+ (vector-ref v2 i) 1)))))",
        "note": "Dot product must multiply corresponding entries directly.",
    },
    {
        "fn": "vec-normalize",
        "buggy": "(define (vec-normalize v)\n  (let ([n (vec-norm v)])\n    (if (= n 1)\n        '(error zero-vector)\n        (vec-scale (/ 1 n) v))))",
        "note": "Zero-vector guard must check `n = 0`, not `n = 1`.",
    },
    {
        "fn": "vec-normalize",
        "buggy": "(define (vec-normalize v)\n  (let ([n (vec-norm v)])\n    (if (= n 0)\n        '(error zero-vector)\n        (vec-scale n v))))",
        "note": "Normalization must scale by reciprocal norm.",
    },
    {
        "fn": "vec-approx-equal?",
        "buggy": "(define (vec-approx-equal? v1 v2 . epsilon-arg)\n  (let ([epsilon (if (null? epsilon-arg) 1e-10 (car epsilon-arg))])\n    (and (= (vector-length v1) (vector-length v2))\n         (let ([n (vector-length v1)])\n           (do ([i 0 (+ i 1)]\n                [eq #t (and eq (> (abs (- (vector-ref v1 i) (vector-ref v2 i))) epsilon))])\n               ((or (= i n) (not eq)) eq))))))",
        "note": "Tolerance comparison direction is inverted.",
    },
    {
        "fn": "vec-approx-equal?",
        "buggy": "(define (vec-approx-equal? v1 v2 . epsilon-arg)\n  (let ([epsilon (if (null? epsilon-arg) 1e-2 (car epsilon-arg))])\n    (and (= (vector-length v1) (vector-length v2))\n         (let ([n (vector-length v1)])\n           (do ([i 0 (+ i 1)]\n                [eq #t (and eq (< (abs (- (vector-ref v1 i) (vector-ref v2 i))) epsilon))])\n               ((or (= i n) (not eq)) eq))))))",
        "note": "Default epsilon should remain strict at 1e-10.",
    },
]

DIFFICULTY = {
    "vec-ref": "easy",
    "vec-first": "easy",
    "vec-map": "medium",
    "vec-zip-with": "medium",
    "vec-slice": "medium",
    "vec-dot": "medium",
    "vec-normalize": "hard",
    "vec-approx-equal?": "hard",
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
    sid = f"vec_{family}_{family_counter[family]:03d}"
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
            if dep not in seen:
                seen.add(dep)
                visit(dep)
                ordered.append(dep)

    for dep in DEPENDS.get(fn, []):
        if dep == fn:
            continue
        if dep not in seen:
            seen.add(dep)
            visit(dep)
            ordered.append(dep)

    for dep in verify_refs(fn):
        if dep == fn:
            continue
        if dep not in seen:
            seen.add(dep)
            visit(dep)
            ordered.append(dep)

    return ordered


def def_verify(fn: str) -> str:
    parts = [DEFS[d] for d in dependency_closure(fn)] + [VERIFY_BY_FUNCTION[fn]]
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
        tags=["linalg", "vec", "spec-to-code", fn],
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
        tags=["linalg", "vec", "spec-to-code", "skeleton", fn],
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
Preserve behavior exactly, including error values.

Target function name: `{fn}`

```python
{PYTHON_SNIPPETS[fn]}
```

Return only the Scheme definition.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["linalg", "vec", "translation", "python", fn],
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
        tags=["linalg", "vec", "translation", "chez", fn],
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
        tags=["linalg", "vec", "bugfix", fn],
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
        tags=["linalg", "vec", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # vec-ref
    ("vec-ref", "Read index 2 from vector #(4 5 6 7).", "(vec-ref (vector 4 5 6 7) 2)", "(equal? (vec-ref (vector 4 5 6 7) 2) 6)", "easy", ["direct"]),
    ("vec-ref", "Return out-of-bounds error for index 3 on a length-3 vector.", "(vec-ref (vector 1 2 3) 3)", "(equal? (vec-ref (vector 1 2 3) 3) '(error out-of-bounds 3 3))", "easy", ["edge-case"]),
    ("vec-ref", "Slice #(9 8 7 6) from 1 to 3 and read index 1 of that slice.", "(vec-ref (vec-slice (vector 9 8 7 6) 1 3) 1)", "(equal? (vec-ref (vec-slice (vector 9 8 7 6) 1 3) 1) 7)", "medium", ["integration"]),
    ("vec-ref", "Return #t iff vec-ref at index 0 equals vec-first on non-empty vector.", "(= (vec-ref (vector 11 12 13) 0) (vec-first (vector 11 12 13)))", "(equal? (= (vec-ref (vector 11 12 13) 0) (vec-first (vector 11 12 13))) #t)", "medium", ["property"]),

    # vec-first
    ("vec-first", "Return first element of #(3 4 5).", "(vec-first (vector 3 4 5))", "(equal? (vec-first (vector 3 4 5)) 3)", "easy", ["direct"]),
    ("vec-first", "Return empty-vector error for empty input.", "(vec-first (vector))", "(equal? (vec-first (vector)) '(error empty-vector))", "easy", ["edge-case"]),
    ("vec-first", "Map (+ x 10) over #(1 2 3), then take vec-first.", "(vec-first (vec-map (lambda (x) (+ x 10)) (vector 1 2 3)))", "(equal? (vec-first (vec-map (lambda (x) (+ x 10)) (vector 1 2 3))) 11)", "medium", ["integration"]),
    ("vec-first", "Take slice [2,5) of #(0 1 2 3 4 5) and return vec-first.", "(vec-first (vec-slice (vector 0 1 2 3 4 5) 2 5))", "(equal? (vec-first (vec-slice (vector 0 1 2 3 4 5) 2 5)) 2)", "medium", ["integration"]),

    # vec-map
    ("vec-map", "Double every element of #(1 2 3).", "(vec-map (lambda (x) (* 2 x)) (vector 1 2 3))", "(equal? (vec-map (lambda (x) (* 2 x)) (vector 1 2 3)) (vector 2 4 6))", "medium", ["direct"]),
    ("vec-map", "Map (+ x 1) over empty vector.", "(vec-map (lambda (x) (+ x 1)) (vector))", "(equal? (vec-map (lambda (x) (+ x 1)) (vector)) (vector))", "easy", ["edge-case"]),
    ("vec-map", "Map square over #(1 2 3), then dot with #(1 1 1).", "(vec-dot (vec-map (lambda (x) (* x x)) (vector 1 2 3)) (vector 1 1 1))", "(equal? (vec-dot (vec-map (lambda (x) (* x x)) (vector 1 2 3)) (vector 1 1 1)) 14)", "hard", ["integration"]),
    ("vec-map", "Return #t iff vec-map preserves length.", "(= (vector-length (vec-map (lambda (x) (- x 1)) (vector 3 4 5 6))) 4)", "(equal? (= (vector-length (vec-map (lambda (x) (- x 1)) (vector 3 4 5 6))) 4) #t)", "medium", ["property"]),

    # vec-zip-with
    ("vec-zip-with", "Add #(1 2 3) and #(4 5 6) with vec-zip-with.", "(vec-zip-with + (vector 1 2 3) (vector 4 5 6))", "(equal? (vec-zip-with + (vector 1 2 3) (vector 4 5 6)) (vector 5 7 9))", "medium", ["direct"]),
    ("vec-zip-with", "Return dimension mismatch error for lengths 2 and 3.", "(vec-zip-with + (vector 1 2) (vector 9 8 7))", "(equal? (vec-zip-with + (vector 1 2) (vector 9 8 7)) '(error dimension-mismatch 2 3))", "easy", ["edge-case"]),
    ("vec-zip-with", "Subtract #(2 3 4) from #(10 20 30).", "(vec-zip-with - (vector 10 20 30) (vector 2 3 4))", "(equal? (vec-zip-with - (vector 10 20 30) (vector 2 3 4)) (vector 8 17 26))", "medium", ["direct"]),
    ("vec-zip-with", "Zip-multiply two vectors and sum via vec-dot with ones.", "(vec-dot (vec-zip-with * (vector 1 2 3) (vector 4 5 6)) (vector 1 1 1))", "(equal? (vec-dot (vec-zip-with * (vector 1 2 3) (vector 4 5 6)) (vector 1 1 1)) 32)", "hard", ["integration"]),

    # vec-slice
    ("vec-slice", "Take slice [1,4) from #(1 2 3 4 5).", "(vec-slice (vector 1 2 3 4 5) 1 4)", "(equal? (vec-slice (vector 1 2 3 4 5) 1 4) (vector 2 3 4))", "medium", ["direct"]),
    ("vec-slice", "Return invalid-range error for start=3 end=1.", "(vec-slice (vector 1 2 3 4) 3 1)", "(equal? (vec-slice (vector 1 2 3 4) 3 1) '(error invalid-range 3 1))", "easy", ["edge-case"]),
    ("vec-slice", "Take full slice [0,3) from #(8 9 10).", "(vec-slice (vector 8 9 10) 0 3)", "(equal? (vec-slice (vector 8 9 10) 0 3) (vector 8 9 10))", "easy", ["direct"]),
    ("vec-slice", "Slice [1,3) from #(5 6 7 8), then normalize that result.", "(vec-normalize (vec-slice (vector 5 6 7 8) 1 3))", "(equal? (vec-normalize (vec-slice (vector 5 6 7 8) 1 3)) (vec-normalize (vector 6 7)))", "hard", ["integration"]),

    # vec-dot
    ("vec-dot", "Compute dot product of #(1 2 3) and #(4 5 6).", "(vec-dot (vector 1 2 3) (vector 4 5 6))", "(equal? (vec-dot (vector 1 2 3) (vector 4 5 6)) 32)", "medium", ["direct"]),
    ("vec-dot", "Return dimension mismatch error for dot of length-2 and length-1 vectors.", "(vec-dot (vector 1 2) (vector 9))", "(equal? (vec-dot (vector 1 2) (vector 9)) '(error dimension-mismatch 2 1))", "easy", ["edge-case"]),
    ("vec-dot", "Compute dot of orthogonal vectors #(1 0 0) and #(0 1 0).", "(vec-dot (vector 1 0 0) (vector 0 1 0))", "(equal? (vec-dot (vector 1 0 0) (vector 0 1 0)) 0)", "medium", ["property"]),
    ("vec-dot", "Map (* 2 x) over #(1 2 3) and dot with #(1 1 1).", "(vec-dot (vec-map (lambda (x) (* 2 x)) (vector 1 2 3)) (vector 1 1 1))", "(equal? (vec-dot (vec-map (lambda (x) (* 2 x)) (vector 1 2 3)) (vector 1 1 1)) 12)", "hard", ["integration"]),

    # vec-normalize
    ("vec-normalize", "Normalize unit vector #(1 0 0).", "(vec-normalize (vector 1 0 0))", "(equal? (vec-normalize (vector 1 0 0)) (vector 1 0 0))", "hard", ["direct"]),
    ("vec-normalize", "Normalize zero vector and return the error.", "(vec-normalize (vector 0 0 0))", "(equal? (vec-normalize (vector 0 0 0)) '(error zero-vector))", "easy", ["edge-case"]),
    ("vec-normalize", "Return #t iff normalized #(3 4) is approximately #(0.6 0.8).", "(vec-approx-equal? (vec-normalize (vector 3 4)) (vector 0.6 0.8) 1e-12)", "(equal? (vec-approx-equal? (vec-normalize (vector 3 4)) (vector 0.6 0.8) 1e-12) #t)", "hard", ["property"]),
    ("vec-normalize", "Return #t iff norm of normalized #(3 4) is approximately 1.", "(< (abs (- (vec-dot (vec-normalize (vector 3 4)) (vec-normalize (vector 3 4))) 1.0)) 1e-12)", "(equal? (< (abs (- (vec-dot (vec-normalize (vector 3 4)) (vec-normalize (vector 3 4))) 1.0)) 1e-12) #t)", "hard", ["property"]),

    # vec-approx-equal?
    ("vec-approx-equal?", "Check default-epsilon approximate equality for two close vectors.", "(vec-approx-equal? (vector 1.0 2.0) (vector 1.00000000001 2.0))", "(equal? (vec-approx-equal? (vector 1.0 2.0) (vector 1.00000000001 2.0)) #t)", "hard", ["direct"]),
    ("vec-approx-equal?", "Check that visibly different vectors are not approximately equal.", "(vec-approx-equal? (vector 1.0 2.0) (vector 1.1 2.0))", "(equal? (vec-approx-equal? (vector 1.0 2.0) (vector 1.1 2.0)) #f)", "medium", ["direct"]),
    ("vec-approx-equal?", "Use custom epsilon=0.1 to compare #(1.0) and #(1.05).", "(vec-approx-equal? (vector 1.0) (vector 1.05) 0.1)", "(equal? (vec-approx-equal? (vector 1.0) (vector 1.05) 0.1) #t)", "medium", ["direct"]),
    ("vec-approx-equal?", "Length mismatch should return #f.", "(vec-approx-equal? (vector 1.0 2.0) (vector 1.0))", "(equal? (vec-approx-equal? (vector 1.0 2.0) (vector 1.0)) #f)", "easy", ["edge-case"]),
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
