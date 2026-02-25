#!/usr/bin/env python3
"""Generate Tier-0 SFT samples for lattice/linalg/matrix-decomp.ss."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

OUT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[2]
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from sft_prompt_diversity import diversify_prompt

ALL_PATH = OUT_DIR / "all.jsonl"
TRAIN_PATH = OUT_DIR / "train.jsonl"
EVAL_PATH = OUT_DIR / "eval.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.json"

SOURCE_MODULE = "lattice/linalg/matrix-decomp.ss"
SOURCE_TEST = "lattice/linalg/test-matrix-decomp.ss"

MATRIX_DECOMP_PATH = ROOT / "lattice/linalg/matrix-decomp.ss"
MATRIX_PATH = ROOT / "lattice/linalg/matrix.ss"


def extract_balanced_form(text: str, start: int) -> str:
    depth = 0
    in_string = False
    in_comment = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_comment:
            if ch == "\n":
                in_comment = False
            continue

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == ";":
            in_comment = True
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError("unbalanced form while extracting definition")


def extract_define(path: Path, fn_name: str) -> str:
    text = path.read_text(encoding="utf-8")
    marker = f"(define ({fn_name}"
    idx = text.find(marker)
    if idx == -1:
        raise ValueError(f"Could not find function definition for {fn_name} in {path}")
    return extract_balanced_form(text, idx)


def extract_value_define(path: Path, name: str) -> str:
    text = path.read_text(encoding="utf-8")
    marker = f"(define {name}"
    idx = text.find(marker)
    if idx == -1:
        raise ValueError(f"Could not find value definition for {name} in {path}")
    return extract_balanced_form(text, idx)


TARGET_FUNCTIONS = [
    "matrix-copy",
    "matrix-set!",
    "matrix-column",
    "matrix-lu",
    "matrix-lu-solve",
    "matrix-cholesky",
    "permutation-sign",
    "matrix-det",
]

DEFS: Dict[str, str] = {
    fn: extract_define(MATRIX_DECOMP_PATH, fn)
    for fn in TARGET_FUNCTIONS
}

SUPPORT_DEFS: Dict[str, str] = {
    "*matrix-tolerance*": extract_value_define(MATRIX_DECOMP_PATH, "*matrix-tolerance*"),
    "matrix-rows": extract_define(MATRIX_PATH, "matrix-rows"),
    "matrix-cols": extract_define(MATRIX_PATH, "matrix-cols"),
    "matrix-data": extract_define(MATRIX_PATH, "matrix-data"),
    "make-matrix": extract_define(MATRIX_PATH, "make-matrix"),
    "matrix-from-lists": extract_define(MATRIX_PATH, "matrix-from-lists"),
    "matrix-ref": extract_define(MATRIX_PATH, "matrix-ref"),
    "matrix-transpose": extract_define(MATRIX_PATH, "matrix-transpose"),
    "matrix-mul": extract_define(MATRIX_PATH, "matrix-mul"),
    "matrix-identity": extract_define(MATRIX_PATH, "matrix-identity"),
    "approx=?": """(define (approx=? a b tol)
  (< (abs (- a b)) tol))""",
}

ALL_DEFS: Dict[str, str] = {**SUPPORT_DEFS, **DEFS}

DEPENDS: Dict[str, List[str]] = {
    "*matrix-tolerance*": [],
    "matrix-rows": [],
    "matrix-cols": [],
    "matrix-data": [],
    "make-matrix": [],
    "matrix-from-lists": ["make-matrix", "matrix-set!"],
    "matrix-ref": ["matrix-rows", "matrix-data", "matrix-cols"],
    "matrix-transpose": ["matrix-rows", "matrix-cols", "make-matrix", "matrix-ref", "matrix-set!"],
    "matrix-mul": ["matrix-rows", "matrix-cols", "make-matrix", "matrix-ref", "matrix-set!"],
    "matrix-identity": ["make-matrix", "matrix-set!"],
    "approx=?": [],
    "matrix-copy": ["matrix-rows", "matrix-cols", "matrix-data"],
    "matrix-set!": ["matrix-cols", "matrix-data"],
    "matrix-column": ["matrix-rows", "matrix-ref"],
    "matrix-lu": [
        "*matrix-tolerance*",
        "matrix-rows",
        "matrix-cols",
        "matrix-copy",
        "make-matrix",
        "matrix-set!",
        "matrix-ref",
        "matrix-data",
    ],
    "matrix-lu-solve": ["matrix-ref"],
    "matrix-cholesky": ["matrix-rows", "matrix-cols", "make-matrix", "matrix-ref", "matrix-set!"],
    "permutation-sign": [],
    "matrix-det": ["matrix-lu", "matrix-rows", "matrix-ref", "permutation-sign"],
}

FUNCTION_ORDER = TARGET_FUNCTIONS

FUNCTION_SPECS = {
    "matrix-copy": "Deep copy a matrix into a new matrix structure with independent backing storage.",
    "matrix-set!": "Mutate matrix element at row i, column j to val using row-major indexing.",
    "matrix-column": "Extract column j as a dense vector of length matrix-rows.",
    "matrix-lu": "Compute LU decomposition with partial pivoting, returning `(list L U P)` or `(error ...)`.",
    "matrix-lu-solve": "Solve `Ax=b` given LU decomposition result `(L U P)` using permutation + forward/back substitution.",
    "matrix-cholesky": "Compute Cholesky factor L for positive-definite square matrix A such that A = L*L^T.",
    "permutation-sign": "Return +1 for even permutations and -1 for odd permutations.",
    "matrix-det": "Compute determinant via LU decomposition, including permutation sign correction.",
}

SKELETONS = {
    "matrix-copy": """(define (matrix-copy m)
  ;; TODO: allocate new data vector and copy all entries
  <TODO>)""",
    "matrix-set!": """(define (matrix-set! m i j val)
  ;; TODO: mutate underlying row-major data vector at (i,j)
  <TODO>)""",
    "matrix-column": """(define (matrix-column m j)
  ;; TODO: extract column j into a vector of length (matrix-rows m)
  <TODO>)""",
    "matrix-lu": """(define (matrix-lu a)
  ;; TODO: LU decomposition with partial pivoting
  <TODO>)""",
    "matrix-lu-solve": """(define (matrix-lu-solve lu-result b)
  ;; TODO: apply permutation, then forward/back substitute
  <TODO>)""",
    "matrix-cholesky": """(define (matrix-cholesky a)
  ;; TODO: lower-triangular Cholesky decomposition with PD checks
  <TODO>)""",
    "permutation-sign": """(define (permutation-sign p)
  ;; TODO: count inversions and return +/-1
  <TODO>)""",
    "matrix-det": """(define (matrix-det a)
  ;; TODO: determinant via LU + permutation sign
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "matrix-copy": """(let* ([m (matrix-from-lists '((1 2) (3 4)))]
       [c (matrix-copy m)])
  (matrix-set! c 0 0 99)
  (and (= (matrix-ref m 0 0) 1)
       (= (matrix-ref c 0 0) 99)
       (= (matrix-ref c 1 1) 4)))""",
    "matrix-set!": """(let ([m (make-matrix 2 2 0)])
  (matrix-set! m 1 0 7)
  (and (= (matrix-ref m 1 0) 7)
       (= (matrix-ref m 0 1) 0)))""",
    "matrix-column": """(equal? (matrix-column (matrix-from-lists '((1 2 3) (4 5 6) (7 8 9))) 1)
        '#(2 5 8))""",
    "matrix-lu": """(let* ([a (matrix-from-lists '((4 3) (6 3)))]
       [res (matrix-lu a)])
  (and (pair? res)
       (not (eq? (car res) 'error))
       (let ([l (car res)]
             [u (cadr res)]
             [p (caddr res)])
         (and (= (vector-length p) 2)
              (= (matrix-ref l 0 0) 1)
              (= (matrix-ref l 1 1) 1)
              (approx=? (matrix-ref u 1 0) 0 1e-10)))))""",
    "matrix-lu-solve": """(let* ([a (matrix-from-lists '((2 1) (1 3)))]
       [b '#(5 6)]
       [lu (matrix-lu a)]
       [x (matrix-lu-solve lu b)]
       [ax0 (+ (* (matrix-ref a 0 0) (vector-ref x 0))
               (* (matrix-ref a 0 1) (vector-ref x 1)))]
       [ax1 (+ (* (matrix-ref a 1 0) (vector-ref x 0))
               (* (matrix-ref a 1 1) (vector-ref x 1)))])
  (and (approx=? ax0 5 1e-8)
       (approx=? ax1 6 1e-8)))""",
    "matrix-cholesky": """(let* ([a (matrix-from-lists '((4 12) (12 37)))]
       [res (matrix-cholesky a)])
  (and (pair? res)
       (not (eq? (car res) 'error))
       (let* ([l (car res)]
              [lt (matrix-transpose l)]
              [llt (matrix-mul l lt)])
         (and (approx=? (matrix-ref llt 0 0) 4 1e-6)
              (approx=? (matrix-ref llt 0 1) 12 1e-6)
              (approx=? (matrix-ref llt 1 1) 37 1e-6)))))""",
    "permutation-sign": """(and (= (permutation-sign '#(0 1 2 3)) 1)
     (= (permutation-sign '#(1 0 2 3)) -1)
     (= (permutation-sign '#(1 2 0 3)) 1))""",
    "matrix-det": """(and (= (matrix-det (matrix-from-lists '((1 2) (3 4)))) -2)
     (= (matrix-det (matrix-identity 3)) 1)
     (= (matrix-det (matrix-from-lists '((2 1 0) (0 3 4) (0 0 5)))) 30))""",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")
TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "matrix-copy": """def matrix_copy(m):
    rows, cols, data = m.rows, m.cols, m.data
    out = [0] * (rows * cols)
    for i in range(rows * cols):
        out[i] = data[i]
    return Matrix(rows, cols, out)""",
    "matrix-set!": """def matrix_set(m, i, j, val):
    idx = i * m.cols + j
    m.data[idx] = val""",
    "matrix-column": """def matrix_column(m, j):
    out = [0] * m.rows
    for i in range(m.rows):
        out[i] = m.ref(i, j)
    return out""",
    "matrix-lu": """def matrix_lu(a):
    m, n = a.rows, a.cols
    if m != n:
        return ("error", "not-square", m, n)
    lu = matrix_copy(a)
    p = list(range(n))

    for k in range(n):
        max_row = k
        max_val = abs(lu.ref(k, k))
        for i in range(k + 1, n):
            v = abs(lu.ref(i, k))
            if v > max_val:
                max_val = v
                max_row = i
        if max_val < MATRIX_TOLERANCE:
            return ("error", "singular-matrix", k)

        if max_row != k:
            for col in range(n):
                lu.swap(k, col, max_row, col)
            p[k], p[max_row] = p[max_row], p[k]

        for i in range(k + 1, n):
            factor = lu.ref(i, k) / lu.ref(k, k)
            lu.set(i, k, factor)
            for j in range(k + 1, n):
                lu.set(i, j, lu.ref(i, j) - factor * lu.ref(k, j))

    L = make_matrix(n, n, 0)
    U = make_matrix(n, n, 0)
    for i in range(n):
        L.set(i, i, 1)
        for j in range(i):
            L.set(i, j, lu.ref(i, j))
        for j in range(i, n):
            U.set(i, j, lu.ref(i, j))
    return (L, U, p)""",
    "matrix-lu-solve": """def matrix_lu_solve(lu_result, b):
    L, U, p = lu_result
    n = len(b)
    pb = [b[p[i]] for i in range(n)]

    y = [0] * n
    for i in range(n):
        s = 0
        for j in range(i):
            s += L.ref(i, j) * y[j]
        y[i] = pb[i] - s

    x = [0] * n
    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(i + 1, n):
            s += U.ref(i, j) * x[j]
        x[i] = (y[i] - s) / U.ref(i, i)
    return x""",
    "matrix-cholesky": """def matrix_cholesky(a):
    n = a.rows
    if n != a.cols:
        return ("error", "not-square", a.rows, a.cols)
    L = make_matrix(n, n, 0)

    for i in range(n):
        for j in range(i + 1):
            s = 0
            for k in range(j):
                s += L.ref(i, k) * L.ref(j, k)
            if i == j:
                val = a.ref(i, i) - s
                if val <= 0:
                    return ("error", "not-positive-definite", i, val)
                L.set(i, i, sqrt(val))
            else:
                L.set(i, j, (a.ref(i, j) - s) / L.ref(j, j))
    return (L,)""",
    "permutation-sign": """def permutation_sign(p):
    inv = 0
    n = len(p)
    for i in range(n):
        for j in range(i + 1, n):
            if p[i] > p[j]:
                inv += 1
    return 1 if (inv % 2 == 0) else -1""",
    "matrix-det": """def matrix_det(a):
    res = matrix_lu(a)
    if isinstance(res, tuple) and len(res) >= 2 and res[0] == "error":
        if res[1] == "singular-matrix":
            return 0
        return res
    L, U, p = res
    prod = 1
    for i in range(U.rows):
        prod *= U.ref(i, i)
    return prod * permutation_sign(p)""",
}

CHEZ_SNIPPETS = {
    "matrix-copy": """(define (matrix-copy0 m)
  (let* ((rows (matrix-rows m))
         (cols (matrix-cols m))
         (data (matrix-data m))
         (out (make-vector (* rows cols) 0)))
    (do ((i 0 (+ i 1)))
        ((= i (* rows cols)) (list 'matrix rows cols out))
      (vector-set! out i (vector-ref data i)))))""",
    "matrix-set!": """(define (matrix-set!0 m i j v)
  (let ((cols (matrix-cols m))
        (data (matrix-data m)))
    (vector-set! data (+ (* i cols) j) v)))""",
    "matrix-column": """(define (matrix-column0 m j)
  (let* ((rows (matrix-rows m))
         (out (make-vector rows 0)))
    (do ((i 0 (+ i 1)))
        ((= i rows) out)
      (vector-set! out i (matrix-ref m i j)))))""",
    "matrix-lu": DEFS["matrix-lu"].replace("(define (matrix-lu ", "(define (matrix-lu0 ", 1),
    "matrix-lu-solve": DEFS["matrix-lu-solve"].replace("(define (matrix-lu-solve ", "(define (matrix-lu-solve0 ", 1),
    "matrix-cholesky": DEFS["matrix-cholesky"].replace("(define (matrix-cholesky ", "(define (matrix-cholesky0 ", 1),
    "permutation-sign": """(define (perm-sign0 p)
  (let ((n (vector-length p)))
    (let outer ((i 0) (swaps 0))
      (if (= i n)
          (if (even? swaps) 1 -1)
          (let inner ((j (+ i 1)) (s swaps))
            (if (= j n)
                (outer (+ i 1) s)
                (inner (+ j 1)
                       (if (> (vector-ref p i) (vector-ref p j))
                           (+ s 1)
                           s))))))))""",
    "matrix-det": """(define (matrix-det0 a)
  (let ((res (matrix-lu a)))
    (if (and (pair? res) (eq? (car res) 'error))
        (if (eq? (cadr res) 'singular-matrix)
            0
            res)
        (let ((u (cadr res))
              (p (caddr res)))
          (let ((n (matrix-rows u)))
            (let loop ((i 0) (prod 1))
              (if (= i n)
                  (* prod (permutation-sign p))
                  (loop (+ i 1) (* prod (matrix-ref u i i))))))))))""",
}


def mutate(fn: str, old: str, new: str) -> str:
    src = DEFS[fn]
    if old not in src:
        raise ValueError(f"Could not mutate {fn}: pattern not found")
    return src.replace(old, new, 1)


BUGGY_CASES = [
    {
        "fn": "matrix-copy",
        "buggy": """(define (matrix-copy m)
  m)""",
        "note": "Returns original matrix instead of deep copy; mutations to copy alias original.",
    },
    {
        "fn": "matrix-copy",
        "buggy": mutate("matrix-copy", "[(= i (* rows cols))", "[(= i rows)"),
        "note": "Copy loop stops after only `rows` entries instead of `rows*cols`.",
    },
    {
        "fn": "matrix-set!",
        "buggy": mutate("matrix-set!", "(+ (* i cols) j)", "(+ (* j cols) i)"),
        "note": "Row/column indexing is transposed, writing to wrong location.",
    },
    {
        "fn": "matrix-set!",
        "buggy": """(define (matrix-set! m i j val)
  (let ([cols (matrix-cols m)]
        [data (matrix-data m)])
       (vector-set! (make-vector (vector-length data) 0)
                    (+ (* i cols) j)
                    val)))""",
        "note": "Writes into a temporary vector, leaving matrix data unchanged.",
    },
    {
        "fn": "matrix-column",
        "buggy": mutate("matrix-column", "[rows (matrix-rows m)]", "[rows (matrix-cols m)]"),
        "note": "Uses column count for output length instead of row count.",
    },
    {
        "fn": "matrix-column",
        "buggy": mutate("matrix-column", "(matrix-ref m i j)", "(matrix-ref m i 0)"),
        "note": "Ignores requested column index and always extracts column 0.",
    },
    {
        "fn": "matrix-lu",
        "buggy": mutate("matrix-lu", "(if (not (= m n))", "(if (= m n)"),
        "note": "Square-matrix guard is inverted.",
    },
    {
        "fn": "matrix-lu",
        "buggy": mutate("matrix-lu", "(if (> val max-val)", "(if (< val max-val)"),
        "note": "Pivot selection chooses smaller pivots, breaking numerical stability.",
    },
    {
        "fn": "matrix-lu-solve",
        "buggy": mutate("matrix-lu-solve", "(vector-ref b (vector-ref p i))", "(vector-ref b i)"),
        "note": "Fails to apply permutation vector to RHS before substitution.",
    },
    {
        "fn": "matrix-lu-solve",
        "buggy": mutate("matrix-lu-solve", "(matrix-ref u i i)", "(matrix-ref l i i)"),
        "note": "Back-substitution divides by L diagonal instead of U diagonal.",
    },
    {
        "fn": "matrix-cholesky",
        "buggy": mutate("matrix-cholesky", "(if (not (= n (matrix-cols a)))", "(if (= n (matrix-cols a))"),
        "note": "Square-matrix precondition check is inverted.",
    },
    {
        "fn": "matrix-cholesky",
        "buggy": mutate("matrix-cholesky", "(if (<= val 0)", "(if (< val 0)"),
        "note": "Allows zero pivots, incorrectly accepting non-positive-definite matrices.",
    },
    {
        "fn": "permutation-sign",
        "buggy": mutate("permutation-sign", "(if (> (vector-ref p i) (vector-ref p j))", "(if (< (vector-ref p i) (vector-ref p j))"),
        "note": "Counts non-inversions rather than inversions.",
    },
    {
        "fn": "permutation-sign",
        "buggy": mutate("permutation-sign", "[i 0] [swaps 0]", "[i 0] [swaps 1]"),
        "note": "Initial inversion count is off by one, flipping parity.",
    },
    {
        "fn": "matrix-det",
        "buggy": mutate("matrix-det", "(matrix-ref u i i)", "(matrix-ref l i i)"),
        "note": "Uses L diagonal product instead of U diagonal product.",
    },
    {
        "fn": "matrix-det",
        "buggy": mutate("matrix-det", "(* prod (permutation-sign p))", "prod"),
        "note": "Drops permutation sign correction term.",
    },
]

DIFFICULTY = {
    "matrix-copy": "easy",
    "matrix-set!": "easy",
    "matrix-column": "medium",
    "matrix-lu": "hard",
    "matrix-lu-solve": "hard",
    "matrix-cholesky": "hard",
    "permutation-sign": "easy",
    "matrix-det": "medium",
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
    sid = f"matrix_decomp_{family}_{family_counter[family]:03d}"
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


def refs_in_text(text: str, exclude: str | None = None) -> List[str]:
    tokens = set(TOKEN_RE.findall(text))
    refs: List[str] = []
    for name in ALL_DEFS.keys():
        if exclude and name == exclude:
            continue
        if name in tokens:
            refs.append(name)
    return refs


def verify_refs(fn: str) -> List[str]:
    return refs_in_text(VERIFY_BY_FUNCTION[fn], exclude=fn)


def dependency_closure(fn: str) -> List[str]:
    ordered: List[str] = []
    seen: Set[str] = set()

    def visit(name: str) -> None:
        for dep in DEPENDS.get(name, []):
            if dep == fn:
                continue
            if dep in ALL_DEFS and dep not in seen:
                seen.add(dep)
                visit(dep)
                ordered.append(dep)

    for dep in DEPENDS.get(fn, []):
        if dep == fn:
            continue
        if dep in ALL_DEFS and dep not in seen:
            seen.add(dep)
            visit(dep)
            ordered.append(dep)

    for dep in verify_refs(fn):
        if dep == fn:
            continue
        if dep in ALL_DEFS and dep not in seen:
            seen.add(dep)
            visit(dep)
            ordered.append(dep)

    return ordered


def def_verify(fn: str) -> str:
    parts = [ALL_DEFS[d] for d in dependency_closure(fn)] + [DEFS[fn], VERIFY_BY_FUNCTION[fn]]
    body = "\n  ".join(parts)
    return f"(let ()\n  {body})"


for fn in FUNCTION_ORDER:
    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Implement this Fold Scheme function.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one `define` for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "linalg", "matrix-decomp", "spec-to-code", fn],
    )

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Complete this Fold Scheme skeleton.

Module: {SOURCE_MODULE}
Target function: `{fn}`
Behavior contract: {FUNCTION_SPECS[fn]}

```scheme
{SKELETONS[fn]}
```

Return only the completed function definition.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "linalg", "matrix-decomp", "skeleton", fn],
    )


for fn in TRANSLATION_FUNCTIONS:
    add_sample(
        family="translation",
        category="transpile",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Translate this Python function into Fold-native Scheme.
Preserve behavior exactly and name the function `{fn}`.

```python
{PYTHON_SNIPPETS[fn]}
```

Return only the Fold definition.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "linalg", "matrix-decomp", "translation", "python", fn],
    )

    add_sample(
        family="translation",
        category="transpile",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Convert this Chez-style definition to canonical Fold style.
Keep semantics identical.

Target function: `{fn}`

```scheme
{CHEZ_SNIPPETS[fn]}
```

Return only Fold code.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "linalg", "matrix-decomp", "translation", "chez", fn],
    )


for case in BUGGY_CASES:
    fn = case["fn"]
    add_sample(
        family="bugfix",
        category="repair",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Fix the bug in this Fold Scheme function with minimal changes.
Target: `{fn}` in `{SOURCE_MODULE}`.
Known issue: {case['note']}

```scheme
{case['buggy']}
```

Return only the corrected function definition.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "linalg", "matrix-decomp", "bugfix", fn],
    )


def wrap_verify_all(verify_expr: str) -> str:
    parts = list(ALL_DEFS.values()) + [verify_expr]
    body = "\n  ".join(parts)
    return f"(let ()\n  {body})"


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
        tags=["tier0", "linalg", "matrix-decomp", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # matrix-copy
    (
        "matrix-copy",
        "Copy a 2x2 matrix and return the copied bottom-right value.",
        "(let* ([m (matrix-from-lists '((1 2) (3 4)))] [c (matrix-copy m)]) (matrix-ref c 1 1))",
        "(equal? (let* ([m (matrix-from-lists '((1 2) (3 4)))] [c (matrix-copy m)]) (matrix-ref c 1 1)) 4)",
        "easy",
        ["direct"],
    ),
    (
        "matrix-copy",
        "Mutate a copied matrix and return original (0,0) value.",
        "(let* ([m (matrix-from-lists '((1 2) (3 4)))] [c (matrix-copy m)]) (matrix-set! c 0 0 99) (matrix-ref m 0 0))",
        "(equal? (let* ([m (matrix-from-lists '((1 2) (3 4)))] [c (matrix-copy m)]) (matrix-set! c 0 0 99) (matrix-ref m 0 0)) 1)",
        "medium",
        ["property"],
    ),
    (
        "matrix-copy",
        "Return determinant of a copied 2x2 matrix.",
        "(let ([m (matrix-from-lists '((1 2) (3 4)))]) (matrix-det (matrix-copy m)))",
        "(equal? (let ([m (matrix-from-lists '((1 2) (3 4)))]) (matrix-det (matrix-copy m))) -2)",
        "medium",
        ["integration"],
    ),
    (
        "matrix-copy",
        "Copy a 0x0 matrix and return `(rows cols)`.",
        "(let* ([m (make-matrix 0 0 0)] [c (matrix-copy m)]) (list (matrix-rows c) (matrix-cols c)))",
        "(equal? (let* ([m (make-matrix 0 0 0)] [c (matrix-copy m)]) (list (matrix-rows c) (matrix-cols c))) '(0 0))",
        "easy",
        ["edge-case"],
    ),

    # matrix-set!
    (
        "matrix-set!",
        "Set value 7 at (1,0) in a fresh 2x2 matrix and read it back.",
        "(let ([m (make-matrix 2 2 0)]) (matrix-set! m 1 0 7) (matrix-ref m 1 0))",
        "(equal? (let ([m (make-matrix 2 2 0)]) (matrix-set! m 1 0 7) (matrix-ref m 1 0)) 7)",
        "easy",
        ["direct"],
    ),
    (
        "matrix-set!",
        "Set diagonal values in a 2x2 matrix and return determinant.",
        "(let ([m (make-matrix 2 2 0)]) (matrix-set! m 0 0 3) (matrix-set! m 1 1 5) (matrix-det m))",
        "(equal? (let ([m (make-matrix 2 2 0)]) (matrix-set! m 0 0 3) (matrix-set! m 1 1 5) (matrix-det m)) 15)",
        "medium",
        ["integration"],
    ),
    (
        "matrix-set!",
        "Write one off-diagonal entry and return `(changed untouched)`.",
        "(let ([m (make-matrix 2 2 0)]) (matrix-set! m 0 1 9) (list (matrix-ref m 0 1) (matrix-ref m 1 0)))",
        "(equal? (let ([m (make-matrix 2 2 0)]) (matrix-set! m 0 1 9) (list (matrix-ref m 0 1) (matrix-ref m 1 0))) '(9 0))",
        "easy",
        ["property"],
    ),
    (
        "matrix-set!",
        "Set only element of 1x1 matrix and return determinant.",
        "(let ([m (make-matrix 1 1 0)]) (matrix-set! m 0 0 11) (matrix-det m))",
        "(equal? (let ([m (make-matrix 1 1 0)]) (matrix-set! m 0 0 11) (matrix-det m)) 11)",
        "easy",
        ["edge-case"],
    ),

    # matrix-column
    (
        "matrix-column",
        "Return column 1 of a 3x3 matrix.",
        "(matrix-column (matrix-from-lists '((1 2 3) (4 5 6) (7 8 9))) 1)",
        "(equal? (matrix-column (matrix-from-lists '((1 2 3) (4 5 6) (7 8 9))) 1) '#(2 5 8))",
        "easy",
        ["direct"],
    ),
    (
        "matrix-column",
        "Extract column 0 from a 3x1 matrix.",
        "(matrix-column (matrix-from-lists '((10) (20) (30))) 0)",
        "(equal? (matrix-column (matrix-from-lists '((10) (20) (30))) 0) '#(10 20 30))",
        "easy",
        ["edge-case"],
    ),
    (
        "matrix-column",
        "Return sum of entries in column 2 of a 3x3 matrix.",
        "(let ([c (matrix-column (matrix-from-lists '((1 2 3) (4 5 6) (7 8 9))) 2)]) (+ (vector-ref c 0) (vector-ref c 1) (vector-ref c 2)))",
        "(equal? (let ([c (matrix-column (matrix-from-lists '((1 2 3) (4 5 6) (7 8 9))) 2)]) (+ (vector-ref c 0) (vector-ref c 1) (vector-ref c 2))) 18)",
        "medium",
        ["property"],
    ),
    (
        "matrix-column",
        "Set `(0,1)=9` then return column 1.",
        "(let ([m (matrix-from-lists '((1 2) (3 4)))]) (matrix-set! m 0 1 9) (matrix-column m 1))",
        "(equal? (let ([m (matrix-from-lists '((1 2) (3 4)))]) (matrix-set! m 0 1 9) (matrix-column m 1)) '#(9 4))",
        "medium",
        ["integration"],
    ),

    # matrix-lu
    (
        "matrix-lu",
        "Return #t iff LU decomposition of a 2x2 matrix succeeds.",
        "(let ([r (matrix-lu (matrix-from-lists '((4 3) (6 3))))]) (and (pair? r) (not (eq? (car r) 'error))))",
        "(equal? (let ([r (matrix-lu (matrix-from-lists '((4 3) (6 3))))]) (and (pair? r) (not (eq? (car r) 'error)))) #t)",
        "medium",
        ["direct"],
    ),
    (
        "matrix-lu",
        "Return error code from LU on non-square matrix.",
        "(cadr (matrix-lu (matrix-from-lists '((1 2 3) (4 5 6)))))",
        "(equal? (cadr (matrix-lu (matrix-from-lists '((1 2 3) (4 5 6))))) 'not-square)",
        "medium",
        ["edge-case"],
    ),
    (
        "matrix-lu",
        "Return error code from LU on singular matrix.",
        "(cadr (matrix-lu (matrix-from-lists '((1 2) (2 4)))))",
        "(equal? (cadr (matrix-lu (matrix-from-lists '((1 2) (2 4))))) 'singular-matrix)",
        "hard",
        ["edge-case"],
    ),
    (
        "matrix-lu",
        "Return permutation vector length from LU of a 3x3 matrix.",
        "(let* ([r (matrix-lu (matrix-from-lists '((2 1 1) (4 -6 0) (-2 7 2))))] [p (caddr r)]) (vector-length p))",
        "(equal? (let* ([r (matrix-lu (matrix-from-lists '((2 1 1) (4 -6 0) (-2 7 2))))] [p (caddr r)]) (vector-length p)) 3)",
        "hard",
        ["property"],
    ),

    # matrix-lu-solve
    (
        "matrix-lu-solve",
        "Solve a 2x2 linear system and return x[0].",
        "(let* ([a (matrix-from-lists '((2 1) (1 3)))] [b '#(5 6)] [x (matrix-lu-solve (matrix-lu a) b)]) (vector-ref x 0))",
        "(let ([x0 (let* ([a (matrix-from-lists '((2 1) (1 3)))] [b '#(5 6)] [x (matrix-lu-solve (matrix-lu a) b)]) (vector-ref x 0))]) (< (abs (- x0 1.8)) 1e-8))",
        "hard",
        ["direct"],
    ),
    (
        "matrix-lu-solve",
        "Solve identity system and return x as vector.",
        "(let* ([a (matrix-identity 3)] [b '#(7 8 9)] [x (matrix-lu-solve (matrix-lu a) b)]) x)",
        "(equal? (let* ([a (matrix-identity 3)] [b '#(7 8 9)] [x (matrix-lu-solve (matrix-lu a) b)]) x) '#(7 8 9))",
        "medium",
        ["property"],
    ),
    (
        "matrix-lu-solve",
        "Solve system and return first reconstructed Ax component.",
        "(let* ([a (matrix-from-lists '((2 1) (1 3)))] [b '#(5 6)] [x (matrix-lu-solve (matrix-lu a) b)]) (+ (* (matrix-ref a 0 0) (vector-ref x 0)) (* (matrix-ref a 0 1) (vector-ref x 1))))",
        "(let ([ax0 (let* ([a (matrix-from-lists '((2 1) (1 3)))] [b '#(5 6)] [x (matrix-lu-solve (matrix-lu a) b)]) (+ (* (matrix-ref a 0 0) (vector-ref x 0)) (* (matrix-ref a 0 1) (vector-ref x 1))))]) (< (abs (- ax0 5)) 1e-8))",
        "hard",
        ["integration"],
    ),
    (
        "matrix-lu-solve",
        "Solve a 3x3 system with known solution and return x[2].",
        "(let* ([a (matrix-from-lists '((1 2 3) (4 5 6) (7 8 10)))] [b '#(14 32 53)] [x (matrix-lu-solve (matrix-lu a) b)]) (vector-ref x 2))",
        "(let ([x2 (let* ([a (matrix-from-lists '((1 2 3) (4 5 6) (7 8 10)))] [b '#(14 32 53)] [x (matrix-lu-solve (matrix-lu a) b)]) (vector-ref x 2))]) (< (abs (- x2 3.0)) 1e-8))",
        "hard",
        ["integration"],
    ),

    # matrix-cholesky
    (
        "matrix-cholesky",
        "Return error code for Cholesky on non-square matrix.",
        "(cadr (matrix-cholesky (matrix-from-lists '((1 2 3) (4 5 6)))))",
        "(equal? (cadr (matrix-cholesky (matrix-from-lists '((1 2 3) (4 5 6))))) 'not-square)",
        "easy",
        ["edge-case"],
    ),
    (
        "matrix-cholesky",
        "Return error code for Cholesky on non-positive-definite matrix.",
        "(cadr (matrix-cholesky (matrix-from-lists '((1 2) (2 1)))))",
        "(equal? (cadr (matrix-cholesky (matrix-from-lists '((1 2) (2 1))))) 'not-positive-definite)",
        "medium",
        ["edge-case"],
    ),
    (
        "matrix-cholesky",
        "Run Cholesky on identity and return L(1,1).",
        "(let* ([res (matrix-cholesky (matrix-identity 3))] [l (car res)]) (matrix-ref l 1 1))",
        "(equal? (let* ([res (matrix-cholesky (matrix-identity 3))] [l (car res)]) (matrix-ref l 1 1)) 1)",
        "medium",
        ["direct"],
    ),
    (
        "matrix-cholesky",
        "Compute reconstructed entry (1,0) from L*L^T for a 2x2 PD matrix.",
        "(let* ([a (matrix-from-lists '((4 12) (12 37)))] [l (car (matrix-cholesky a))] [llt (matrix-mul l (matrix-transpose l))]) (matrix-ref llt 1 0))",
        "(let ([v (let* ([a (matrix-from-lists '((4 12) (12 37)))] [l (car (matrix-cholesky a))] [llt (matrix-mul l (matrix-transpose l))]) (matrix-ref llt 1 0))]) (< (abs (- v 12)) 1e-6))",
        "hard",
        ["integration"],
    ),

    # permutation-sign
    (
        "permutation-sign",
        "Return sign of identity permutation #(0 1 2 3).",
        "(permutation-sign '#(0 1 2 3))",
        "(equal? (permutation-sign '#(0 1 2 3)) 1)",
        "easy",
        ["direct"],
    ),
    (
        "permutation-sign",
        "Return sign of single transposition permutation #(1 0 2 3).",
        "(permutation-sign '#(1 0 2 3))",
        "(equal? (permutation-sign '#(1 0 2 3)) -1)",
        "easy",
        ["direct"],
    ),
    (
        "permutation-sign",
        "Return sign of 3-cycle permutation #(1 2 0).",
        "(permutation-sign '#(1 2 0))",
        "(equal? (permutation-sign '#(1 2 0)) 1)",
        "medium",
        ["property"],
    ),
    (
        "permutation-sign",
        "Return sign of permutation #(2 1 0).",
        "(permutation-sign '#(2 1 0))",
        "(equal? (permutation-sign '#(2 1 0)) -1)",
        "medium",
        ["property"],
    ),

    # matrix-det
    (
        "matrix-det",
        "Return determinant of 2x2 matrix ((1 2) (3 4)).",
        "(matrix-det (matrix-from-lists '((1 2) (3 4))))",
        "(equal? (matrix-det (matrix-from-lists '((1 2) (3 4)))) -2)",
        "easy",
        ["direct"],
    ),
    (
        "matrix-det",
        "Return determinant of identity 3x3 matrix.",
        "(matrix-det (matrix-identity 3))",
        "(equal? (matrix-det (matrix-identity 3)) 1)",
        "easy",
        ["property"],
    ),
    (
        "matrix-det",
        "Return determinant of singular matrix ((1 2) (2 4)).",
        "(matrix-det (matrix-from-lists '((1 2) (2 4))))",
        "(equal? (matrix-det (matrix-from-lists '((1 2) (2 4)))) 0)",
        "medium",
        ["edge-case"],
    ),
    (
        "matrix-det",
        "Return determinant of upper-triangular 3x3 matrix.",
        "(matrix-det (matrix-from-lists '((2 1 0) (0 3 4) (0 0 5))))",
        "(equal? (matrix-det (matrix-from-lists '((2 1 0) (0 3 4) (0 0 5)))) 30)",
        "medium",
        ["property"],
    ),
]

for fn, prompt, gt, verify, diff, tags in composition_cases:
    add_composition(fn, prompt, gt, wrap_verify_all(verify), diff, tags)

if sum(1 for s in samples if s["family"] == "composition") != 32:
    raise ValueError("composition family must contain exactly 32 samples")

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
