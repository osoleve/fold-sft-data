#!/usr/bin/env python3
"""Generate SFT samples for lattice/linalg/integer-matrix.ss."""

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

SOURCE_MODULE = "lattice/linalg/integer-matrix.ss"
SOURCE_TEST = "lattice/linalg/test-integer-matrix.ss"

DEFS: Dict[str, str] = {
    "int-mat-ref": """(define (int-mat-ref data cols i j)
  (vector-ref data (+ (* i cols) j)))""",
    "int-mat-set!": """(define (int-mat-set! data cols i j val)
  (vector-set! data (+ (* i cols) j) val))""",
    "int-mat-copy": """(define (int-mat-copy v)
  (let* ([n (vector-length v)]
         [result (make-vector n 0)])
    (do ([i 0 (+ i 1)])
        ((= i n) result)
      (vector-set! result i (vector-ref v i)))))""",
    "swap-rows!": """(define (swap-rows! data rows cols i1 i2)
  (when (not (= i1 i2))
    (do ([j 0 (+ j 1)])
        ((= j cols))
      (let ([tmp (int-mat-ref data cols i1 j)])
        (int-mat-set! data cols i1 j (int-mat-ref data cols i2 j))
        (int-mat-set! data cols i2 j tmp)))))""",
    "swap-cols!": """(define (swap-cols! data rows cols j1 j2)
  (when (not (= j1 j2))
    (do ([i 0 (+ i 1)])
        ((= i rows))
      (let ([tmp (int-mat-ref data cols i j1)])
        (int-mat-set! data cols i j1 (int-mat-ref data cols i j2))
        (int-mat-set! data cols i j2 tmp)))))""",
    "make-identity-vec": """(define (make-identity-vec n)
  (let ([data (make-vector (* n n) 0)])
    (do ([i 0 (+ i 1)])
        ((= i n) data)
      (vector-set! data (+ (* i n) i) 1))))""",
    "matrix-minor": """(define (matrix-minor mat i j)
  (let* ([n (matrix-rows mat)]
         [data (matrix-data mat)]
         [new-n (- n 1)]
         [result (make-vector (* new-n new-n) 0)])
    (do ([row 0 (+ row 1)]
         [new-row 0 (if (= row i) new-row (+ new-row 1))])
        ((= row n) (list 'matrix new-n new-n result))
      (when (not (= row i))
        (do ([col 0 (+ col 1)]
             [new-col 0 (if (= col j) new-col (+ new-col 1))])
            ((= col n))
          (when (not (= col j))
            (vector-set! result (+ (* new-row new-n) new-col)
                        (vector-ref data (+ (* row n) col)))))))))""",
    "matrix-determinant-int": """(define (matrix-determinant-int mat)
  (let* ([n (matrix-rows mat)])
    (if (= n 0)
        1
        (let ([data (int-mat-copy (matrix-data mat))]
              [sign 1])
          (let loop ([k 0] [prev-pivot 1])
            (if (>= k (- n 1))
                (* sign (int-mat-ref data n (- n 1) (- n 1)))
                (let ([pivot-idx (let search ([i k])
                                  (cond
                                    [(>= i n) #f]
                                    [(not (= 0 (int-mat-ref data n i k))) i]
                                    [else (search (+ i 1))]))])
                  (if (not pivot-idx)
                      0
                      (begin
                        (when (not (= pivot-idx k))
                          (swap-rows! data n n k pivot-idx)
                          (set! sign (- sign)))
                        (let ([pivot (int-mat-ref data n k k)])
                          (do ([i (+ k 1) (+ i 1)])
                              ((= i n))
                            (do ([j (+ k 1) (+ j 1)])
                                ((= j n))
                              (let ([new-val (quotient
                                              (- (* pivot (int-mat-ref data n i j))
                                                 (* (int-mat-ref data n i k)
                                                    (int-mat-ref data n k j)))
                                              prev-pivot)])
                                (int-mat-set! data n i j new-val))))
                          (loop (+ k 1) pivot)))))))))))""",
}

DEPENDS: Dict[str, List[str]] = {
    "int-mat-ref": [],
    "int-mat-set!": [],
    "int-mat-copy": [],
    "swap-rows!": ["int-mat-ref", "int-mat-set!"],
    "swap-cols!": ["int-mat-ref", "int-mat-set!"],
    "make-identity-vec": [],
    "matrix-minor": [],
    "matrix-determinant-int": ["int-mat-copy", "int-mat-ref", "int-mat-set!", "swap-rows!"],
}

FUNCTION_ORDER = [
    "int-mat-ref",
    "int-mat-set!",
    "int-mat-copy",
    "swap-rows!",
    "swap-cols!",
    "make-identity-vec",
    "matrix-minor",
    "matrix-determinant-int",
]

FUNCTION_SPECS = {
    "int-mat-ref": "Read matrix element at (i,j) from row-major flat vector with `cols` columns.",
    "int-mat-set!": "Mutate row-major flat vector entry at (i,j) to val using `cols` stride.",
    "int-mat-copy": "Return a fresh copy of vector v with identical contents.",
    "swap-rows!": "In-place swap of rows i1 and i2 in flat matrix data.",
    "swap-cols!": "In-place swap of columns j1 and j2 in flat matrix data.",
    "make-identity-vec": "Create n x n identity matrix as flat row-major vector.",
    "matrix-minor": "Return matrix with row i and column j removed.",
    "matrix-determinant-int": "Compute exact integer determinant using Bareiss elimination.",
}

SKELETONS = {
    "int-mat-ref": """(define (int-mat-ref data cols i j)
  ;; TODO: index row-major flat storage
  <TODO>)""",
    "int-mat-set!": """(define (int-mat-set! data cols i j val)
  ;; TODO: mutate row-major flat storage
  <TODO>)""",
    "int-mat-copy": """(define (int-mat-copy v)
  ;; TODO: allocate and copy vector contents
  <TODO>)""",
    "swap-rows!": """(define (swap-rows! data rows cols i1 i2)
  ;; TODO: swap two rows in-place
  <TODO>)""",
    "swap-cols!": """(define (swap-cols! data rows cols j1 j2)
  ;; TODO: swap two columns in-place
  <TODO>)""",
    "make-identity-vec": """(define (make-identity-vec n)
  ;; TODO: create flat identity matrix vector
  <TODO>)""",
    "matrix-minor": """(define (matrix-minor mat i j)
  ;; TODO: remove row i and column j
  <TODO>)""",
    "matrix-determinant-int": """(define (matrix-determinant-int mat)
  ;; TODO: Bareiss determinant algorithm for integer matrices
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "int-mat-ref": "(and (= (int-mat-ref (vector 1 2 3 4 5 6) 3 0 2) 3) (= (int-mat-ref (vector 1 2 3 4 5 6) 3 1 1) 5))",
    "int-mat-set!": "(let ([d (vector 1 2 3 4)]) (int-mat-set! d 2 1 0 99) (and (= (int-mat-ref d 2 1 0) 99) (equal? d (vector 1 2 99 4))))",
    "int-mat-copy": "(let* ([v (vector 1 2 3)] [c (int-mat-copy v)]) (vector-set! v 0 9) (and (equal? c (vector 1 2 3)) (not (eq? v c))))",
    "swap-rows!": "(let ([d (vector 1 2 3 4 5 6)]) (swap-rows! d 2 3 0 1) (equal? d (vector 4 5 6 1 2 3)))",
    "swap-cols!": "(let ([d (vector 1 2 3 4 5 6)]) (swap-cols! d 2 3 0 2) (equal? d (vector 3 2 1 6 5 4)))",
    "make-identity-vec": "(and (equal? (make-identity-vec 1) (vector 1)) (equal? (make-identity-vec 3) (vector 1 0 0 0 1 0 0 0 1)))",
    "matrix-minor": "(and (equal? (matrix->lists (matrix-minor (matrix-from-lists '((1 2 3) (4 5 6) (7 8 9))) 1 1)) '((1 3) (7 9))) (equal? (matrix->lists (matrix-minor (matrix-from-lists '((3 4) (1 2))) 0 1)) '((1))))",
    "matrix-determinant-int": "(and (= (matrix-determinant-int (matrix-from-lists '())) 1) (= (matrix-determinant-int (matrix-from-lists '((3 4) (1 2)))) 2) (= (matrix-determinant-int (matrix-from-lists '((1 2 3) (4 5 6) (7 8 10)))) -3) (= (matrix-determinant-int (matrix-identity 4)) 1) (= (matrix-determinant-int (matrix-from-lists '((1 2) (2 4)))) 0) (= (matrix-determinant-int (matrix-from-lists '((0 1) (1 0)))) -1))",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")
TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "int-mat-ref": "def int_mat_ref(data, cols, i, j):\n    return data[i * cols + j]",
    "int-mat-set!": "def int_mat_set(data, cols, i, j, val):\n    data[i * cols + j] = val",
    "int-mat-copy": "def int_mat_copy(v):\n    out = [0] * len(v)\n    for i in range(len(v)):\n        out[i] = v[i]\n    return out",
    "swap-rows!": "def swap_rows(data, rows, cols, i1, i2):\n    if i1 != i2:\n        for j in range(cols):\n            idx1 = i1 * cols + j\n            idx2 = i2 * cols + j\n            data[idx1], data[idx2] = data[idx2], data[idx1]",
    "swap-cols!": "def swap_cols(data, rows, cols, j1, j2):\n    if j1 != j2:\n        for i in range(rows):\n            idx1 = i * cols + j1\n            idx2 = i * cols + j2\n            data[idx1], data[idx2] = data[idx2], data[idx1]",
    "make-identity-vec": "def make_identity_vec(n):\n    data = [0] * (n * n)\n    for i in range(n):\n        data[i * n + i] = 1\n    return data",
    "matrix-minor": "def matrix_minor(mat, i, j):\n    n = mat_rows(mat)\n    data = mat_data(mat)\n    new_n = n - 1\n    out = [0] * (new_n * new_n)\n    new_row = 0\n    for row in range(n):\n        if row == i:\n            continue\n        new_col = 0\n        for col in range(n):\n            if col == j:\n                continue\n            out[new_row * new_n + new_col] = data[row * n + col]\n            new_col += 1\n        new_row += 1\n    return ['matrix', new_n, new_n, list_to_vector(out)]",
    "matrix-determinant-int": "def matrix_determinant_int(mat):\n    n = mat_rows(mat)\n    if n == 0:\n        return 1\n    data = int_mat_copy(mat_data(mat))\n    sign = 1\n    k = 0\n    prev_pivot = 1\n    while k < n - 1:\n        pivot_idx = None\n        i = k\n        while i < n:\n            if int_mat_ref(data, n, i, k) != 0:\n                pivot_idx = i\n                break\n            i += 1\n        if pivot_idx is None:\n            return 0\n        if pivot_idx != k:\n            swap_rows(data, n, n, k, pivot_idx)\n            sign = -sign\n        pivot = int_mat_ref(data, n, k, k)\n        for i in range(k + 1, n):\n            for j in range(k + 1, n):\n                new_val = ((pivot * int_mat_ref(data, n, i, j)) - (int_mat_ref(data, n, i, k) * int_mat_ref(data, n, k, j))) // prev_pivot\n                int_mat_set(data, n, i, j, new_val)\n        prev_pivot = pivot\n        k += 1\n    return sign * int_mat_ref(data, n, n - 1, n - 1)",
}

CHEZ_SNIPPETS = {
    "int-mat-ref": "(define (imat-ref d c i j)\n  (vector-ref d (+ (* i c) j)))",
    "int-mat-set!": "(define (imat-set! d c i j v)\n  (vector-set! d (+ (* i c) j) v))",
    "int-mat-copy": "(define (imat-copy v)\n  (let* ((n (vector-length v))\n         (out (make-vector n 0)))\n    (do ((i 0 (+ i 1)))\n        ((= i n) out)\n      (vector-set! out i (vector-ref v i)))))",
    "swap-rows!": "(define (swap-rows0! d r c a b)\n  (unless (= a b)\n    (do ((j 0 (+ j 1)))\n        ((= j c))\n      (let ((tmp (int-mat-ref d c a j)))\n        (int-mat-set! d c a j (int-mat-ref d c b j))\n        (int-mat-set! d c b j tmp)))))",
    "swap-cols!": "(define (swap-cols0! d r c a b)\n  (unless (= a b)\n    (do ((i 0 (+ i 1)))\n        ((= i r))\n      (let ((tmp (int-mat-ref d c i a)))\n        (int-mat-set! d c i a (int-mat-ref d c i b))\n        (int-mat-set! d c i b tmp)))))",
    "make-identity-vec": "(define (identity-flat n)\n  (let ((data (make-vector (* n n) 0)))\n    (do ((i 0 (+ i 1)))\n        ((= i n) data)\n      (vector-set! data (+ (* i n) i) 1))))",
    "matrix-minor": "(define (minor0 m i j)\n  (let* ((n (matrix-rows m))\n         (data (matrix-data m))\n         (new-n (- n 1))\n         (out (make-vector (* new-n new-n) 0)))\n    (do ((row 0 (+ row 1))\n         (new-row 0 (if (= row i) new-row (+ new-row 1))))\n        ((= row n) (list 'matrix new-n new-n out))\n      (unless (= row i)\n        (do ((col 0 (+ col 1))\n             (new-col 0 (if (= col j) new-col (+ new-col 1))))\n            ((= col n))\n          (unless (= col j)\n            (vector-set! out (+ (* new-row new-n) new-col)\n                         (vector-ref data (+ (* row n) col)))))))))",
    "matrix-determinant-int": "(define (det-int0 mat)\n  (let* ((n (matrix-rows mat)))\n    (if (= n 0)\n        1\n        (let ((data (int-mat-copy (matrix-data mat)))\n              (sign 1))\n          (let loop ((k 0) (prev 1))\n            (if (>= k (- n 1))\n                (* sign (int-mat-ref data n (- n 1) (- n 1)))\n                (let ((pidx (let search ((i k))\n                             (cond ((>= i n) #f)\n                                   ((not (= 0 (int-mat-ref data n i k))) i)\n                                   (else (search (+ i 1)))))))\n                  (if (not pidx)\n                      0\n                      (begin\n                        (when (not (= pidx k))\n                          (swap-rows! data n n k pidx)\n                          (set! sign (- sign)))\n                        (let ((pivot (int-mat-ref data n k k)))\n                          (do ((i (+ k 1) (+ i 1)))\n                              ((= i n))\n                            (do ((j (+ k 1) (+ j 1)))\n                                ((= j n))\n                              (let ((new-val (quotient (- (* pivot (int-mat-ref data n i j))\n                                                          (* (int-mat-ref data n i k)\n                                                             (int-mat-ref data n k j)))\n                                                      prev)))\n                                (int-mat-set! data n i j new-val))))\n                          (loop (+ k 1) pivot))))))))))",
}

BUGGY_CASES = [
    {
        "fn": "int-mat-ref",
        "buggy": "(define (int-mat-ref data cols i j)\n  (vector-ref data (+ (* j cols) i)))",
        "note": "Row/column indexing is transposed; use i*cols + j.",
    },
    {
        "fn": "int-mat-ref",
        "buggy": "(define (int-mat-ref data cols i j)\n  (vector-ref data (+ (* i (+ cols 1)) j)))",
        "note": "Stride must be exactly cols, not cols+1.",
    },
    {
        "fn": "int-mat-set!",
        "buggy": "(define (int-mat-set! data cols i j val)\n  (vector-set! data (+ (* j cols) i) val))",
        "note": "Target index is wrong; mutation must use row-major indexing.",
    },
    {
        "fn": "int-mat-set!",
        "buggy": "(define (int-mat-set! data cols i j val)\n  data)",
        "note": "Function must mutate data in-place.",
    },
    {
        "fn": "int-mat-copy",
        "buggy": "(define (int-mat-copy v)\n  v)",
        "note": "Must return a fresh vector copy, not alias original.",
    },
    {
        "fn": "int-mat-copy",
        "buggy": "(define (int-mat-copy v)\n  (let* ([n (vector-length v)] [result (make-vector n 0)])\n    (do ([i 0 (+ i 1)])\n        ((= i (- n 1)) result)\n      (vector-set! result i (vector-ref v i)))))",
        "note": "Loop termination skips copying the last element.",
    },
    {
        "fn": "swap-rows!",
        "buggy": "(define (swap-rows! data rows cols i1 i2)\n  (when (not (= i1 i2))\n    (let ([tmp (int-mat-ref data cols i1 0)])\n      (int-mat-set! data cols i1 0 (int-mat-ref data cols i2 0))\n      (int-mat-set! data cols i2 0 tmp))))",
        "note": "Must swap every column, not just column 0.",
    },
    {
        "fn": "swap-rows!",
        "buggy": "(define (swap-rows! data rows cols i1 i2)\n  (when (not (= i1 i2))\n    (do ([j 0 (+ j 1)])\n        ((= j rows))\n      (let ([tmp (int-mat-ref data cols i1 j)])\n        (int-mat-set! data cols i1 j (int-mat-ref data cols i2 j))\n        (int-mat-set! data cols i2 j tmp)))))",
        "note": "Loop bound must be cols, not rows; all columns must be swapped.",
    },
    {
        "fn": "swap-cols!",
        "buggy": "(define (swap-cols! data rows cols j1 j2)\n  (when (not (= j1 j2))\n    (let ([tmp (int-mat-ref data cols 0 j1)])\n      (int-mat-set! data cols 0 j1 (int-mat-ref data cols 0 j2))\n      (int-mat-set! data cols 0 j2 tmp))))",
        "note": "Must swap every row, not just row 0.",
    },
    {
        "fn": "swap-cols!",
        "buggy": "(define (swap-cols! data rows cols j1 j2)\n  (when (not (= j1 j2))\n    (do ([i 0 (+ i 1)])\n        ((= i cols))\n      (let ([tmp (int-mat-ref data cols i j1)])\n        (int-mat-set! data cols i j1 (int-mat-ref data cols i j2))\n        (int-mat-set! data cols i j2 tmp)))))",
        "note": "Loop bound must be rows, not cols; all rows must be swapped.",
    },
    {
        "fn": "make-identity-vec",
        "buggy": "(define (make-identity-vec n)\n  (let ([data (make-vector (* n n) 0)])\n    (do ([i 0 (+ i 1)])\n        ((= i n) data)\n      (vector-set! data (+ (* i n) (+ i 1)) 1))))",
        "note": "Diagonal index should be i*n+i, not shifted to the superdiagonal.",
    },
    {
        "fn": "make-identity-vec",
        "buggy": "(define (make-identity-vec n)\n  (make-vector (* n n) 1))",
        "note": "Identity requires zeros off-diagonal.",
    },
    {
        "fn": "matrix-minor",
        "buggy": "(define (matrix-minor mat i j)\n  mat)",
        "note": "Minor must remove one row and one column.",
    },
    {
        "fn": "matrix-minor",
        "buggy": "(define (matrix-minor mat i j)\n  (let* ([n (matrix-rows mat)] [data (matrix-data mat)] [new-n (- n 1)] [result (make-vector (* new-n new-n) 0)])\n    (do ([row 0 (+ row 1)] [new-row 0 (if (= row i) new-row (+ new-row 1))])\n        ((= row n) (list 'matrix new-n new-n result))\n      (when (not (= row i))\n        (do ([col 0 (+ col 1)] [new-col 0 (if (= col j) new-col (+ new-col 1))])\n            ((= col n))\n          (when (not (= col j))\n            (vector-set! result (+ (* new-col new-n) new-row)\n                        (vector-ref data (+ (* row n) col)))))))))",
        "note": "Result write index is transposed; preserve row-major minor layout.",
    },
    {
        "fn": "matrix-determinant-int",
        "buggy": "(define (matrix-determinant-int mat)\n  0)",
        "note": "Determinant must be computed, not constant.",
    },
    {
        "fn": "matrix-determinant-int",
        "buggy": "(define (matrix-determinant-int mat)\n  (let* ([n (matrix-rows mat)])\n    (if (= n 0)\n        0\n        (let ([data (int-mat-copy (matrix-data mat))] [sign 1])\n          (let loop ([k 0] [prev-pivot 1])\n            (if (>= k (- n 1))\n                (int-mat-ref data n (- n 1) (- n 1))\n                (let ([pivot-idx (let search ([i k])\n                                  (cond [(>= i n) #f] [(not (= 0 (int-mat-ref data n i k))) i] [else (search (+ i 1))]))])\n                  (if (not pivot-idx)\n                      0\n                      (begin\n                        (when (not (= pivot-idx k)) (swap-rows! data n n k pivot-idx))\n                        (let ([pivot (int-mat-ref data n k k)])\n                          (do ([i (+ k 1) (+ i 1)])\n                              ((= i n))\n                            (do ([j (+ k 1) (+ j 1)])\n                                ((= j n))\n                              (int-mat-set! data n i j (quotient (- (* pivot (int-mat-ref data n i j)) (* (int-mat-ref data n i k) (int-mat-ref data n k j))) prev-pivot))))\n                          (loop (+ k 1) pivot))))))))))",
        "note": "Empty matrix determinant should be 1, and row swaps must flip sign.",
    },
]

DIFFICULTY = {
    "int-mat-ref": "easy",
    "int-mat-set!": "easy",
    "int-mat-copy": "easy",
    "swap-rows!": "medium",
    "swap-cols!": "medium",
    "make-identity-vec": "easy",
    "matrix-minor": "hard",
    "matrix-determinant-int": "hard",
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
    sid = f"integer_matrix_{family}_{family_counter[family]:03d}"
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
        tags=["linalg", "integer-matrix", "spec-to-code", fn],
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
        tags=["linalg", "integer-matrix", "skeleton-completion", fn],
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
Preserve behavior exactly and use target function name `{fn}`.
Return only the Scheme definition.

```python
{PYTHON_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["linalg", "integer-matrix", "python-to-scheme", fn],
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
        tags=["linalg", "integer-matrix", "chez-to-fold", fn],
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
        tags=["linalg", "integer-matrix", "bugfix", fn],
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
        tags=["linalg", "integer-matrix", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # Direct
    ("int-mat-ref", "Read entry (1,2) from flat 2x3 data vector.", "(int-mat-ref (vector 1 2 3 4 5 6) 3 1 2)", "(equal? (int-mat-ref (vector 1 2 3 4 5 6) 3 1 2) 6)", "easy", ["direct"]),
    ("int-mat-ref", "Read entry (0,0) from flat 2x2 vector.", "(int-mat-ref (vector 9 8 7 6) 2 0 0)", "(equal? (int-mat-ref (vector 9 8 7 6) 2 0 0) 9)", "easy", ["direct"]),
    ("int-mat-set!", "Set (1,0)=42 in 2x2 data and return resulting vector.", "(let ([d (vector 1 2 3 4)]) (int-mat-set! d 2 1 0 42) d)", "(equal? (let ([d (vector 1 2 3 4)]) (int-mat-set! d 2 1 0 42) d) (vector 1 2 42 4))", "easy", ["direct"]),
    ("int-mat-set!", "Set an entry then read it back with int-mat-ref.", "(let ([d (vector 1 2 3 4 5 6)]) (int-mat-set! d 3 0 1 77) (int-mat-ref d 3 0 1))", "(equal? (let ([d (vector 1 2 3 4 5 6)]) (int-mat-set! d 3 0 1 77) (int-mat-ref d 3 0 1)) 77)", "easy", ["direct"]),
    ("int-mat-copy", "Copy vector '(1 2 3) and return the copy.", "(int-mat-copy (vector 1 2 3))", "(equal? (int-mat-copy (vector 1 2 3)) (vector 1 2 3))", "easy", ["direct"]),
    ("int-mat-copy", "Mutate source after int-mat-copy and return copied first entry.", "(let* ([v (vector 1 2 3)] [c (int-mat-copy v)]) (vector-set! v 0 9) (vector-ref c 0))", "(equal? (let* ([v (vector 1 2 3)] [c (int-mat-copy v)]) (vector-set! v 0 9) (vector-ref c 0)) 1)", "medium", ["direct"]),
    ("swap-rows!", "Swap rows 0 and 1 in 2x3 data and return vector.", "(let ([d (vector 1 2 3 4 5 6)]) (swap-rows! d 2 3 0 1) d)", "(equal? (let ([d (vector 1 2 3 4 5 6)]) (swap-rows! d 2 3 0 1) d) (vector 4 5 6 1 2 3))", "medium", ["direct"]),
    ("swap-rows!", "Call swap-rows! with identical row indices and return data.", "(let ([d (vector 1 2 3 4)]) (swap-rows! d 2 2 1 1) d)", "(equal? (let ([d (vector 1 2 3 4)]) (swap-rows! d 2 2 1 1) d) (vector 1 2 3 4))", "easy", ["edge-case"]),
    ("swap-cols!", "Swap columns 0 and 2 in 2x3 data and return vector.", "(let ([d (vector 1 2 3 4 5 6)]) (swap-cols! d 2 3 0 2) d)", "(equal? (let ([d (vector 1 2 3 4 5 6)]) (swap-cols! d 2 3 0 2) d) (vector 3 2 1 6 5 4))", "medium", ["direct"]),
    ("swap-cols!", "Call swap-cols! with identical column indices and return data.", "(let ([d (vector 1 2 3 4)]) (swap-cols! d 2 2 0 0) d)", "(equal? (let ([d (vector 1 2 3 4)]) (swap-cols! d 2 2 0 0) d) (vector 1 2 3 4))", "easy", ["edge-case"]),
    ("make-identity-vec", "Build identity flat vector for n=3.", "(make-identity-vec 3)", "(equal? (make-identity-vec 3) (vector 1 0 0 0 1 0 0 0 1))", "easy", ["direct"]),
    ("make-identity-vec", "Build identity flat vector for n=1.", "(make-identity-vec 1)", "(equal? (make-identity-vec 1) (vector 1))", "easy", ["direct"]),
    ("matrix-minor", "Take minor removing row 1 column 1 from 3x3 matrix.", "(matrix->lists (matrix-minor (matrix-from-lists '((1 2 3) (4 5 6) (7 8 9))) 1 1))", "(equal? (matrix->lists (matrix-minor (matrix-from-lists '((1 2 3) (4 5 6) (7 8 9))) 1 1)) '((1 3) (7 9)))", "hard", ["direct"]),
    ("matrix-minor", "Take minor removing row 0 column 1 from 2x2 matrix.", "(matrix->lists (matrix-minor (matrix-from-lists '((3 4) (1 2))) 0 1))", "(equal? (matrix->lists (matrix-minor (matrix-from-lists '((3 4) (1 2))) 0 1)) '((1)))", "hard", ["direct"]),
    ("matrix-determinant-int", "Compute determinant of 2x2 integer matrix ((3 4) (1 2)).", "(matrix-determinant-int (matrix-from-lists '((3 4) (1 2))))", "(equal? (matrix-determinant-int (matrix-from-lists '((3 4) (1 2)))) 2)", "hard", ["direct"]),
    ("matrix-determinant-int", "Compute determinant of 3x3 matrix ((1 2 3) (4 5 6) (7 8 10)).", "(matrix-determinant-int (matrix-from-lists '((1 2 3) (4 5 6) (7 8 10))))", "(equal? (matrix-determinant-int (matrix-from-lists '((1 2 3) (4 5 6) (7 8 10)))) -3)", "hard", ["direct"]),

    # Properties
    ("int-mat-set!", "Return #t iff setting then reading same index recovers written value.", "(let ([d (vector 0 1 2 3 4 5)]) (int-mat-set! d 3 1 1 88) (= (int-mat-ref d 3 1 1) 88))", "(equal? (let ([d (vector 0 1 2 3 4 5)]) (int-mat-set! d 3 1 1 88) (= (int-mat-ref d 3 1 1) 88)) #t)", "easy", ["property"]),
    ("int-mat-copy", "Return #t iff copy length equals source length.", "(= (vector-length (int-mat-copy (vector 5 6 7 8))) 4)", "(equal? (= (vector-length (int-mat-copy (vector 5 6 7 8))) 4) #t)", "easy", ["property"]),
    ("swap-rows!", "Return #t iff swap-rows! preserves sum of all elements.", "(let* ([d (vector 1 2 3 4 5 6)] [before (+ 1 2 3 4 5 6)]) (swap-rows! d 2 3 0 1) (= (+ (vector-ref d 0) (vector-ref d 1) (vector-ref d 2) (vector-ref d 3) (vector-ref d 4) (vector-ref d 5)) before))", "(equal? (let* ([d (vector 1 2 3 4 5 6)] [before (+ 1 2 3 4 5 6)]) (swap-rows! d 2 3 0 1) (= (+ (vector-ref d 0) (vector-ref d 1) (vector-ref d 2) (vector-ref d 3) (vector-ref d 4) (vector-ref d 5)) before)) #t)", "medium", ["property"]),
    ("swap-cols!", "Return #t iff swap-cols! preserves sum of all elements.", "(let* ([d (vector 1 2 3 4 5 6)] [before (+ 1 2 3 4 5 6)]) (swap-cols! d 2 3 0 2) (= (+ (vector-ref d 0) (vector-ref d 1) (vector-ref d 2) (vector-ref d 3) (vector-ref d 4) (vector-ref d 5)) before))", "(equal? (let* ([d (vector 1 2 3 4 5 6)] [before (+ 1 2 3 4 5 6)]) (swap-cols! d 2 3 0 2) (= (+ (vector-ref d 0) (vector-ref d 1) (vector-ref d 2) (vector-ref d 3) (vector-ref d 4) (vector-ref d 5)) before)) #t)", "medium", ["property"]),
    ("make-identity-vec", "Return #t iff identity vector for n=4 has exactly 4 ones.", "(let ([d (make-identity-vec 4)]) (= (let loop ([i 0] [acc 0]) (if (= i (vector-length d)) acc (loop (+ i 1) (+ acc (if (= (vector-ref d i) 1) 1 0))))) 4))", "(equal? (let ([d (make-identity-vec 4)]) (= (let loop ([i 0] [acc 0]) (if (= i (vector-length d)) acc (loop (+ i 1) (+ acc (if (= (vector-ref d i) 1) 1 0))))) 4)) #t)", "medium", ["property"]),
    ("matrix-minor", "Return #t iff minor of 4x4 has dimension 3x3.", "(let ([m (matrix-minor (matrix-from-lists '((1 2 3 4) (5 6 7 8) (9 10 11 12) (13 14 15 16))) 2 1)]) (and (= (matrix-rows m) 3) (= (matrix-cols m) 3)))", "(equal? (let ([m (matrix-minor (matrix-from-lists '((1 2 3 4) (5 6 7 8) (9 10 11 12) (13 14 15 16))) 2 1)]) (and (= (matrix-rows m) 3) (= (matrix-cols m) 3))) #t)", "hard", ["property"]),
    ("matrix-determinant-int", "Return #t iff determinant of identity 4x4 is 1.", "(= (matrix-determinant-int (matrix-identity 4)) 1)", "(equal? (= (matrix-determinant-int (matrix-identity 4)) 1) #t)", "hard", ["property"]),
    ("matrix-determinant-int", "Return #t iff determinant of singular matrix is 0.", "(= (matrix-determinant-int (matrix-from-lists '((1 2) (2 4)))) 0)", "(equal? (= (matrix-determinant-int (matrix-from-lists '((1 2) (2 4)))) 0) #t)", "hard", ["property"]),

    # Integration/loop
    ("swap-rows!", "Apply two row swaps and return final vector.", "(let ([d (vector 1 2 3 4 5 6)]) (swap-rows! d 2 3 0 1) (swap-rows! d 2 3 0 1) d)", "(equal? (let ([d (vector 1 2 3 4 5 6)]) (swap-rows! d 2 3 0 1) (swap-rows! d 2 3 0 1) d) (vector 1 2 3 4 5 6))", "medium", ["integration"]),
    ("swap-cols!", "Apply a column swap then read the new value at (1,0).", "(let ([d (vector 1 2 3 4 5 6)]) (swap-cols! d 2 3 0 2) (int-mat-ref d 3 1 0))", "(equal? (let ([d (vector 1 2 3 4 5 6)]) (swap-cols! d 2 3 0 2) (int-mat-ref d 3 1 0)) 6)", "medium", ["integration"]),
    ("make-identity-vec", "Build matrix from make-identity-vec 3 and compute determinant.", "(matrix-determinant-int (matrix-from-vec 3 3 (make-identity-vec 3)))", "(equal? (matrix-determinant-int (matrix-from-vec 3 3 (make-identity-vec 3))) 1)", "hard", ["integration"]),
    ("matrix-minor", "Compute determinant of minor(0,0) for 3x3 matrix.", "(matrix-determinant-int (matrix-minor (matrix-from-lists '((2 3 1) (4 5 6) (7 8 9))) 0 0))", "(equal? (matrix-determinant-int (matrix-minor (matrix-from-lists '((2 3 1) (4 5 6) (7 8 9))) 0 0)) -3)", "hard", ["integration"]),
    ("matrix-determinant-int", "Return #t iff determinant is invariant under transpose.", "(let ([m (matrix-from-lists '((2 5 7) (1 3 4) (6 8 9)) )]) (= (matrix-determinant-int m) (matrix-determinant-int (matrix-transpose m))))", "(equal? (let ([m (matrix-from-lists '((2 5 7) (1 3 4) (6 8 9)) )]) (= (matrix-determinant-int m) (matrix-determinant-int (matrix-transpose m)))) #t)", "hard", ["integration"]),
    ("int-mat-set!", "Use named-let to fill the diagonal of 3x3 flat data with 1s, then return data.", "(let ([d (make-vector 9 0)]) (let loop ([i 0]) (if (= i 3) d (begin (int-mat-set! d 3 i i 1) (loop (+ i 1))))) )", "(equal? (let ([d (make-vector 9 0)]) (let loop ([i 0]) (if (= i 3) d (begin (int-mat-set! d 3 i i 1) (loop (+ i 1))))) ) (vector 1 0 0 0 1 0 0 0 1))", "hard", ["loop"]),
    ("matrix-determinant-int", "Compute determinant after swapping two rows in flat data and rebuilding matrix.", "(let ([d (vector 1 2 3 4)]) (swap-rows! d 2 2 0 1) (matrix-determinant-int (matrix-from-vec 2 2 d)))", "(equal? (let ([d (vector 1 2 3 4)]) (swap-rows! d 2 2 0 1) (matrix-determinant-int (matrix-from-vec 2 2 d))) 2)", "hard", ["integration"]),
    ("matrix-minor", "Take minor of 2x2 matrix and compute determinant of the resulting 1x1.", "(matrix-determinant-int (matrix-minor (matrix-from-lists '((11 12) (13 14))) 0 1))", "(equal? (matrix-determinant-int (matrix-minor (matrix-from-lists '((11 12) (13 14))) 0 1)) 13)", "hard", ["integration"]),
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
