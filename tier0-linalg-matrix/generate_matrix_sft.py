#!/usr/bin/env python3
"""Generate SFT samples for lattice/linalg/matrix.ss."""

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

SOURCE_MODULE = "lattice/linalg/matrix.ss"
SOURCE_TEST = "lattice/linalg/test-matrix.ss"

DEFS: Dict[str, str] = {
    "matrix-rows": """(define (matrix-rows m)
  (cadr m))""",
    "matrix-cols": """(define (matrix-cols m)
  (caddr m))""",
    "matrix-data": """(define (matrix-data m)
  (cadddr m))""",
    "matrix-from-lists": """(define (matrix-from-lists rows)
  (if (null? rows)
      (list 'matrix 0 0 (vector))
      (let* ([m (length rows)]
             [n (length (car rows))])
            (let check-rows ([rs rows] [i 0])
                 (cond
                  [(null? rs)
                   (let ([data (make-vector (* m n) 0)])
                        (do ([i 0 (+ i 1)]
                             [rs rows (cdr rs)])
                            ((= i m))
                            (do ([j 0 (+ j 1)]
                                 [cs (car rs) (cdr cs)])
                                ((= j n))
                                (vector-set! data (+ (* i n) j) (car cs))))
                        (list 'matrix m n data))]
                  [(not (= (length (car rs)) n))
                   `(error ragged-input row ,i expected-length ,n actual-length ,(length (car rs)))]
                  [else
                   (check-rows (cdr rs) (+ i 1))])))))""",
    "matrix-ref": """(define (matrix-ref m i j)
  (let ([rows (matrix-rows m)]
        [cols (matrix-cols m)]
        [data (matrix-data m)])
       (if (and (>= i 0) (< i rows) (>= j 0) (< j cols))
           (vector-ref data (+ (* i cols) j))
           `(error out-of-bounds (,i ,j) (,rows ,cols)))))""",
    "matrix-transpose": """(define (matrix-transpose m)
  (let* ([rows (matrix-rows m)]
         [cols (matrix-cols m)]
         [data (matrix-data m)]
         [result (make-vector (* rows cols) 0)])
        (do ([i 0 (+ i 1)])
            ((= i rows))
            (do ([j 0 (+ j 1)])
                ((= j cols))
                (vector-set! result (+ (* j rows) i)
                             (vector-ref data (+ (* i cols) j)))))
        (list 'matrix cols rows result)))""",
    "matrix-map2": """(define (matrix-map2 f m1 m2)
  (let ([r1 (matrix-rows m1)] [c1 (matrix-cols m1)]
        [r2 (matrix-rows m2)] [c2 (matrix-cols m2)])
       (if (not (and (= r1 r2) (= c1 c2)))
           `(error dimension-mismatch (,r1 ,c1) (,r2 ,c2))
           (let* ([data1 (matrix-data m1)]
                  [data2 (matrix-data m2)]
                  [n (* r1 c1)])
                 (list 'matrix r1 c1
                       (vec-tabulate n i
                         (f (vector-ref data1 i)
                            (vector-ref data2 i))))))))""",
    "matrix-mul": """(define (matrix-mul m1 m2)
  (let ([r1 (matrix-rows m1)] [c1 (matrix-cols m1)]
        [r2 (matrix-rows m2)] [c2 (matrix-cols m2)])
       (if (not (= c1 r2))
           `(error dimension-mismatch (,r1 ,c1) (,r2 ,c2))
           (let* ([data1 (matrix-data m1)]
                  [data2 (matrix-data m2)]
                  [result (make-vector (* r1 c2) 0)])
                 (do ([i 0 (+ i 1)])
                     ((= i r1))
                     (do ([k 0 (+ k 1)])
                         ((= k c1))
                         (let ([a-ik (vector-ref data1 (+ (* i c1) k))]
                               [b-row-k (* k c2)]
                               [c-row-i (* i c2)])
                              (do ([j 0 (+ j 1)])
                                  ((= j c2))
                                  (let ([idx (+ c-row-i j)])
                                       (vector-set! result idx
                                                    (+ (vector-ref result idx)
                                                       (* a-ik (vector-ref data2 (+ b-row-k j))))))))))
                 (list 'matrix r1 c2 result)))))""",
    "matrix-vec-mul": """(define (matrix-vec-mul m v)
  (let ([rows (matrix-rows m)]
        [cols (matrix-cols m)]
        [n (vector-length v)])
       (if (not (= cols n))
           `(error dimension-mismatch ,cols ,n)
           (let ([data (matrix-data m)]
                 [result (make-vector rows 0)])
                (do ([i 0 (+ i 1)])
                    ((= i rows) result)
                    (do ([j 0 (+ j 1)]
                         [sum 0 (+ sum (* (vector-ref data (+ (* i cols) j))
                                          (vector-ref v j)))])
                        ((= j cols)
                         (vector-set! result i sum))))))))""",
    "matrix-identity": """(define (matrix-identity n)
  (let ([data (make-vector (* n n) 0)])
       (do ([i 0 (+ i 1)])
           ((= i n) (list 'matrix n n data))
           (vector-set! data (+ (* i n) i) 1))))""",
    "matrix-submatrix": """(define (matrix-submatrix m r1 c1 r2 c2)
  (let ([rows (matrix-rows m)]
        [cols (matrix-cols m)]
        [data (matrix-data m)])
       (cond
        [(or (< r1 0) (< c1 0)) `(error out-of-bounds (,r1 ,c1))]
        [(or (> r2 rows) (> c2 cols)) `(error out-of-bounds (,r2 ,c2))]
        [(or (> r1 r2) (> c1 c2)) `(error invalid-range)]
        [else
         (let* ([new-rows (- r2 r1)]
                [new-cols (- c2 c1)]
                [result (make-vector (* new-rows new-cols) 0)])
               (do ([i 0 (+ i 1)])
                   ((= i new-rows) (list 'matrix new-rows new-cols result))
                   (do ([j 0 (+ j 1)])
                       ((= j new-cols))
                       (vector-set! result (+ (* i new-cols) j)
                                    (vector-ref data (+ (* (+ r1 i) cols) (+ c1 j)))))))])))""",
    "matrix->lists": """(define (matrix->lists m)
  (let ([rows (matrix-rows m)]
        [cols (matrix-cols m)]
        [data (matrix-data m)])
       (do ([i (- rows 1) (- i 1)]
            [result '() (cons (do ([j (- cols 1) (- j 1)]
                                   [row '() (cons (vector-ref data (+ (* i cols) j)) row)])
                                  ((< j 0) row))
                              result)])
           ((< i 0) result))))""",
}

DEPENDS: Dict[str, List[str]] = {
    "matrix-rows": [],
    "matrix-cols": [],
    "matrix-data": [],
    "matrix-from-lists": [],
    "matrix-ref": ["matrix-rows", "matrix-cols", "matrix-data"],
    "matrix-transpose": ["matrix-rows", "matrix-cols", "matrix-data"],
    "matrix-map2": ["matrix-rows", "matrix-cols", "matrix-data"],
    "matrix-mul": ["matrix-rows", "matrix-cols", "matrix-data"],
    "matrix-vec-mul": ["matrix-rows", "matrix-cols", "matrix-data"],
    "matrix-identity": [],
    "matrix-submatrix": ["matrix-rows", "matrix-cols", "matrix-data"],
    "matrix->lists": ["matrix-rows", "matrix-cols", "matrix-data"],
}

FUNCTION_ORDER = [
    "matrix-from-lists",
    "matrix-ref",
    "matrix-transpose",
    "matrix-map2",
    "matrix-mul",
    "matrix-vec-mul",
    "matrix-identity",
    "matrix-submatrix",
]

FUNCTION_SPECS = {
    "matrix-from-lists": "Build a row-major matrix from rectangular row lists; return ragged-input error on mismatched row lengths.",
    "matrix-ref": "Return element at (i,j) when in bounds, else `(error out-of-bounds (i j) (rows cols))`.",
    "matrix-transpose": "Return a new matrix with rows/cols swapped and data transposed.",
    "matrix-map2": "Apply binary function elementwise on two same-shape matrices, else dimension-mismatch error.",
    "matrix-mul": "Matrix multiplication A(r1 x c1) * B(r2 x c2), with dimension check `c1 = r2`.",
    "matrix-vec-mul": "Multiply matrix by vector with column/length check; return result vector or dimension-mismatch error.",
    "matrix-identity": "Construct n x n identity matrix with ones on diagonal and zeros elsewhere.",
    "matrix-submatrix": "Extract half-open slice [r1,r2) x [c1,c2); validate bounds and range.",
}

SKELETONS = {
    "matrix-from-lists": """(define (matrix-from-lists rows)
  ;; TODO: validate rectangular rows and pack row-major vector
  <TODO>)""",
    "matrix-ref": """(define (matrix-ref m i j)
  ;; TODO: bounds check then index row-major data
  <TODO>)""",
    "matrix-transpose": """(define (matrix-transpose m)
  ;; TODO: swap dimensions and transpose entries
  <TODO>)""",
    "matrix-map2": """(define (matrix-map2 f m1 m2)
  ;; TODO: enforce same shape and map f pairwise
  <TODO>)""",
    "matrix-mul": """(define (matrix-mul m1 m2)
  ;; TODO: implement matrix multiply with dimension check
  <TODO>)""",
    "matrix-vec-mul": """(define (matrix-vec-mul m v)
  ;; TODO: matrix-vector multiply with dimension check
  <TODO>)""",
    "matrix-identity": """(define (matrix-identity n)
  ;; TODO: allocate n*n and set diagonal to 1
  <TODO>)""",
    "matrix-submatrix": """(define (matrix-submatrix m r1 c1 r2 c2)
  ;; TODO: validate range and copy sub-block
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "matrix-from-lists": "(and (equal? (matrix->lists (matrix-from-lists '((1 2 3) (4 5 6)))) '((1 2 3) (4 5 6))) (equal? (matrix->lists (matrix-from-lists '())) '()) (equal? (matrix-from-lists '((1 2 3) (4 5))) '(error ragged-input row 1 expected-length 3 actual-length 2)))",
    "matrix-ref": "(and (= (matrix-ref (matrix-from-lists '((1 2 3) (4 5 6))) 1 2) 6) (equal? (matrix-ref (matrix-from-lists '((1 2) (3 4))) 2 0) '(error out-of-bounds (2 0) (2 2))))",
    "matrix-transpose": "(and (equal? (matrix->lists (matrix-transpose (matrix-from-lists '((1 2 3) (4 5 6))))) '((1 4) (2 5) (3 6))) (equal? (matrix->lists (matrix-transpose (matrix-from-lists '()))) '()))",
    "matrix-map2": "(and (equal? (matrix->lists (matrix-map2 + (matrix-from-lists '((1 2) (3 4))) (matrix-from-lists '((10 20) (30 40))))) '((11 22) (33 44))) (equal? (matrix-map2 + (matrix-from-lists '((1 2))) (matrix-from-lists '((1) (2))) ) '(error dimension-mismatch (1 2) (2 1))))",
    "matrix-mul": "(and (equal? (matrix->lists (matrix-mul (matrix-from-lists '((1 2) (3 4))) (matrix-from-lists '((5 6) (7 8))))) '((19 22) (43 50))) (equal? (matrix-mul (matrix-from-lists '((1 2 3))) (matrix-from-lists '((1 2) (3 4)))) '(error dimension-mismatch (1 3) (2 2))))",
    "matrix-vec-mul": "(and (equal? (matrix-vec-mul (matrix-from-lists '((1 2 3) (4 5 6))) (vector 1 2 3)) (vector 14 32)) (equal? (matrix-vec-mul (matrix-from-lists '((1 2) (3 4))) (vector 1 2 3)) '(error dimension-mismatch 2 3)))",
    "matrix-identity": "(and (equal? (matrix->lists (matrix-identity 3)) '((1 0 0) (0 1 0) (0 0 1))) (equal? (matrix->lists (matrix-identity 0)) '()))",
    "matrix-submatrix": "(and (equal? (matrix->lists (matrix-submatrix (matrix-from-lists '((1 2 3 4) (5 6 7 8) (9 10 11 12))) 1 1 3 3)) '((6 7) (10 11))) (equal? (matrix-submatrix (matrix-from-lists '((1 2) (3 4))) 2 0 1 1) '(error invalid-range)) (equal? (matrix-submatrix (matrix-from-lists '((1 2) (3 4))) -1 0 1 1) '(error out-of-bounds (-1 0))))",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")
TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "matrix-from-lists": "def matrix_from_lists(rows):\n    if not rows:\n        return ['matrix', 0, 0, []]\n    m = len(rows)\n    n = len(rows[0])\n    i = 0\n    for r in rows:\n        if len(r) != n:\n            return ['error', 'ragged-input', 'row', i, 'expected-length', n, 'actual-length', len(r)]\n        i += 1\n    data = [0] * (m * n)\n    for i in range(m):\n        for j in range(n):\n            data[i * n + j] = rows[i][j]\n    return ['matrix', m, n, list_to_vector(data)]",
    "matrix-ref": "def matrix_ref(m, i, j):\n    rows = matrix_rows(m)\n    cols = matrix_cols(m)\n    data = matrix_data(m)\n    if 0 <= i < rows and 0 <= j < cols:\n        return data[i * cols + j]\n    return ['error', 'out-of-bounds', [i, j], [rows, cols]]",
    "matrix-transpose": "def matrix_transpose(m):\n    rows = matrix_rows(m)\n    cols = matrix_cols(m)\n    data = matrix_data(m)\n    out = [0] * (rows * cols)\n    for i in range(rows):\n        for j in range(cols):\n            out[j * rows + i] = data[i * cols + j]\n    return ['matrix', cols, rows, list_to_vector(out)]",
    "matrix-map2": "def matrix_map2(f, m1, m2):\n    r1, c1 = matrix_rows(m1), matrix_cols(m1)\n    r2, c2 = matrix_rows(m2), matrix_cols(m2)\n    if not (r1 == r2 and c1 == c2):\n        return ['error', 'dimension-mismatch', [r1, c1], [r2, c2]]\n    d1, d2 = matrix_data(m1), matrix_data(m2)\n    out = [0] * (r1 * c1)\n    for i in range(r1 * c1):\n        out[i] = f(d1[i], d2[i])\n    return ['matrix', r1, c1, list_to_vector(out)]",
    "matrix-mul": "def matrix_mul(m1, m2):\n    r1, c1 = matrix_rows(m1), matrix_cols(m1)\n    r2, c2 = matrix_rows(m2), matrix_cols(m2)\n    if c1 != r2:\n        return ['error', 'dimension-mismatch', [r1, c1], [r2, c2]]\n    d1, d2 = matrix_data(m1), matrix_data(m2)\n    out = [0] * (r1 * c2)\n    for i in range(r1):\n        for k in range(c1):\n            a_ik = d1[i * c1 + k]\n            b_row_k = k * c2\n            c_row_i = i * c2\n            for j in range(c2):\n                idx = c_row_i + j\n                out[idx] = out[idx] + a_ik * d2[b_row_k + j]\n    return ['matrix', r1, c2, list_to_vector(out)]",
    "matrix-vec-mul": "def matrix_vec_mul(m, v):\n    rows = matrix_rows(m)\n    cols = matrix_cols(m)\n    if cols != len(v):\n        return ['error', 'dimension-mismatch', cols, len(v)]\n    data = matrix_data(m)\n    out = [0] * rows\n    for i in range(rows):\n        s = 0\n        for j in range(cols):\n            s += data[i * cols + j] * v[j]\n        out[i] = s\n    return list_to_vector(out)",
    "matrix-identity": "def matrix_identity(n):\n    data = [0] * (n * n)\n    for i in range(n):\n        data[i * n + i] = 1\n    return ['matrix', n, n, list_to_vector(data)]",
    "matrix-submatrix": "def matrix_submatrix(m, r1, c1, r2, c2):\n    rows, cols = matrix_rows(m), matrix_cols(m)\n    data = matrix_data(m)\n    if r1 < 0 or c1 < 0:\n        return ['error', 'out-of-bounds', [r1, c1]]\n    if r2 > rows or c2 > cols:\n        return ['error', 'out-of-bounds', [r2, c2]]\n    if r1 > r2 or c1 > c2:\n        return ['error', 'invalid-range']\n    nr, nc = r2 - r1, c2 - c1\n    out = [0] * (nr * nc)\n    for i in range(nr):\n        for j in range(nc):\n            out[i * nc + j] = data[(r1 + i) * cols + (c1 + j)]\n    return ['matrix', nr, nc, list_to_vector(out)]",
}

CHEZ_SNIPPETS = {
    "matrix-from-lists": "(define (mat-from-rows rows)\n  (if (null? rows)\n      (list 'matrix 0 0 (vector))\n      (let* ((m (length rows))\n             (n (length (car rows))))\n        (let check ((rs rows) (i 0))\n          (cond ((null? rs)\n                 (let ((data (make-vector (* m n) 0)))\n                   (do ((i 0 (+ i 1)) (rs rows (cdr rs)))\n                       ((= i m))\n                     (do ((j 0 (+ j 1)) (cs (car rs) (cdr cs)))\n                         ((= j n))\n                       (vector-set! data (+ (* i n) j) (car cs))))\n                   (list 'matrix m n data)))\n                ((not (= (length (car rs)) n))\n                 `(error ragged-input row ,i expected-length ,n actual-length ,(length (car rs))))\n                (else (check (cdr rs) (+ i 1))))))))",
    "matrix-ref": "(define (mat-ref m i j)\n  (let ((r (matrix-rows m)) (c (matrix-cols m)) (d (matrix-data m)))\n    (if (and (>= i 0) (< i r) (>= j 0) (< j c))\n        (vector-ref d (+ (* i c) j))\n        `(error out-of-bounds (,i ,j) (,r ,c)))))",
    "matrix-transpose": "(define (mat-transpose m)\n  (let* ((r (matrix-rows m))\n         (c (matrix-cols m))\n         (d (matrix-data m))\n         (out (make-vector (* r c) 0)))\n    (do ((i 0 (+ i 1)))\n        ((= i r))\n      (do ((j 0 (+ j 1)))\n          ((= j c))\n        (vector-set! out (+ (* j r) i)\n                     (vector-ref d (+ (* i c) j)))))\n    (list 'matrix c r out)))",
    "matrix-map2": "(define (mat-map2 f a b)\n  (let ((r1 (matrix-rows a)) (c1 (matrix-cols a))\n        (r2 (matrix-rows b)) (c2 (matrix-cols b)))\n    (if (not (and (= r1 r2) (= c1 c2)))\n        `(error dimension-mismatch (,r1 ,c1) (,r2 ,c2))\n        (let* ((d1 (matrix-data a)) (d2 (matrix-data b)) (n (* r1 c1)))\n          (list 'matrix r1 c1\n                (vec-tabulate n i (f (vector-ref d1 i) (vector-ref d2 i))))))))",
    "matrix-mul": "(define (mat-mul a b)\n  (let ((r1 (matrix-rows a)) (c1 (matrix-cols a))\n        (r2 (matrix-rows b)) (c2 (matrix-cols b)))\n    (if (not (= c1 r2))\n        `(error dimension-mismatch (,r1 ,c1) (,r2 ,c2))\n        (let* ((d1 (matrix-data a)) (d2 (matrix-data b)) (out (make-vector (* r1 c2) 0)))\n          (do ((i 0 (+ i 1)))\n              ((= i r1))\n            (do ((k 0 (+ k 1)))\n                ((= k c1))\n              (let ((aik (vector-ref d1 (+ (* i c1) k)))\n                    (bk (* k c2))\n                    (ci (* i c2)))\n                (do ((j 0 (+ j 1)))\n                    ((= j c2))\n                  (let ((idx (+ ci j)))\n                    (vector-set! out idx (+ (vector-ref out idx) (* aik (vector-ref d2 (+ bk j))))))))))\n          (list 'matrix r1 c2 out))))",
    "matrix-vec-mul": "(define (mat-vec-mul m v)\n  (let ((r (matrix-rows m)) (c (matrix-cols m)) (n (vector-length v)))\n    (if (not (= c n))\n        `(error dimension-mismatch ,c ,n)\n        (let ((d (matrix-data m)) (out (make-vector r 0)))\n          (do ((i 0 (+ i 1)))\n              ((= i r) out)\n            (do ((j 0 (+ j 1))\n                 (sum 0 (+ sum (* (vector-ref d (+ (* i c) j)) (vector-ref v j)))))\n                ((= j c) (vector-set! out i sum))))))))",
    "matrix-identity": "(define (eye n)\n  (let ((data (make-vector (* n n) 0)))\n    (do ((i 0 (+ i 1)))\n        ((= i n) (list 'matrix n n data))\n      (vector-set! data (+ (* i n) i) 1))))",
    "matrix-submatrix": "(define (mat-sub m r1 c1 r2 c2)\n  (let ((rows (matrix-rows m))\n        (cols (matrix-cols m))\n        (data (matrix-data m)))\n    (cond ((or (< r1 0) (< c1 0)) `(error out-of-bounds (,r1 ,c1)))\n          ((or (> r2 rows) (> c2 cols)) `(error out-of-bounds (,r2 ,c2)))\n          ((or (> r1 r2) (> c1 c2)) `(error invalid-range))\n          (else\n           (let* ((nr (- r2 r1)) (nc (- c2 c1)) (out (make-vector (* nr nc) 0)))\n             (do ((i 0 (+ i 1)))\n                 ((= i nr) (list 'matrix nr nc out))\n               (do ((j 0 (+ j 1)))\n                   ((= j nc))\n                 (vector-set! out (+ (* i nc) j)\n                              (vector-ref data (+ (* (+ r1 i) cols) (+ c1 j))))))))))",
}

BUGGY_CASES = [
    {
        "fn": "matrix-from-lists",
        "buggy": "(define (matrix-from-lists rows)\n  (if (null? rows)\n      (list 'matrix 0 0 (vector))\n      (let* ([m (length rows)] [n (length (car rows))] [data (make-vector (* m n) 0)])\n        (do ([i 0 (+ i 1)] [rs rows (cdr rs)])\n            ((= i m) (list 'matrix m n data))\n          (do ([j 0 (+ j 1)] [cs (car rs) (cdr cs)])\n              ((= j n))\n            (vector-set! data (+ (* i n) j) (car cs)))))))",
        "note": "Must validate ragged rows and return ragged-input error.",
    },
    {
        "fn": "matrix-from-lists",
        "buggy": "(define (matrix-from-lists rows)\n  (if (null? rows)\n      '(error empty-matrix)\n      (let* ([m (length rows)] [n (length (car rows))] [data (make-vector (* m n) 0)])\n        (do ([i 0 (+ i 1)] [rs rows (cdr rs)])\n            ((= i m) (list 'matrix m n data))\n          (do ([j 0 (+ j 1)] [cs (car rs) (cdr cs)])\n              ((= j n))\n            (vector-set! data (+ (* i n) j) (car cs)))))))",
        "note": "Empty input should produce a 0x0 matrix, not an error.",
    },
    {
        "fn": "matrix-ref",
        "buggy": "(define (matrix-ref m i j)\n  (let ([rows (matrix-rows m)] [cols (matrix-cols m)] [data (matrix-data m)])\n    (if (and (>= i 0) (< i rows) (>= j 0) (< j cols))\n        (vector-ref data (+ (* i rows) j))\n        `(error out-of-bounds (,i ,j) (,rows ,cols)))))",
        "note": "Row-major indexing must multiply by column count, not row count.",
    },
    {
        "fn": "matrix-ref",
        "buggy": "(define (matrix-ref m i j)\n  (vector-ref (matrix-data m) (+ (* i (matrix-cols m)) j)))",
        "note": "Must check bounds and return structured out-of-bounds error.",
    },
    {
        "fn": "matrix-transpose",
        "buggy": "(define (matrix-transpose m)\n  (let* ([rows (matrix-rows m)] [cols (matrix-cols m)] [data (matrix-data m)] [result (make-vector (* rows cols) 0)])\n    (do ([i 0 (+ i 1)])\n        ((= i rows))\n      (do ([j 0 (+ j 1)])\n          ((= j cols))\n        (vector-set! result (+ (* i cols) j) (vector-ref data (+ (* i cols) j)))))\n    (list 'matrix rows cols result)))",
        "note": "Transpose must swap dimensions and move elements across diagonal.",
    },
    {
        "fn": "matrix-transpose",
        "buggy": "(define (matrix-transpose m)\n  (let* ([rows (matrix-rows m)] [cols (matrix-cols m)] [data (matrix-data m)] [result (make-vector (* rows cols) 0)])\n    (do ([i 0 (+ i 1)])\n        ((= i rows))\n      (do ([j 0 (+ j 1)])\n          ((= j cols))\n        (vector-set! result (+ (* j rows) i) (vector-ref data (+ (* j cols) i)))))\n    (list 'matrix cols rows result)))",
        "note": "Read index must be original (i,j), not swapped source coordinates.",
    },
    {
        "fn": "matrix-map2",
        "buggy": "(define (matrix-map2 f m1 m2)\n  (let ([r1 (matrix-rows m1)] [c1 (matrix-cols m1)] [r2 (matrix-rows m2)] [c2 (matrix-cols m2)])\n    (if (and (= r1 r2) (= c1 c2))\n        `(error dimension-mismatch (,r1 ,c1) (,r2 ,c2))\n        (let* ([data1 (matrix-data m1)] [data2 (matrix-data m2)] [n (* r1 c1)])\n          (list 'matrix r1 c1 (vec-tabulate n i (f (vector-ref data1 i) (vector-ref data2 i))))))))",
        "note": "Dimension guard is inverted.",
    },
    {
        "fn": "matrix-map2",
        "buggy": "(define (matrix-map2 f m1 m2)\n  (let ([r1 (matrix-rows m1)] [c1 (matrix-cols m1)] [r2 (matrix-rows m2)] [c2 (matrix-cols m2)])\n    (if (not (and (= r1 r2) (= c1 c2)))\n        `(error dimension-mismatch (,r1 ,c1) (,r2 ,c2))\n        (let* ([data1 (matrix-data m1)] [data2 (matrix-data m2)] [n (* r1 c1)])\n          (list 'matrix r1 c1 (vec-tabulate n i (f (vector-ref data1 i) (vector-ref data1 i))))))))",
        "note": "Elementwise combine must use one value from each matrix.",
    },
    {
        "fn": "matrix-mul",
        "buggy": "(define (matrix-mul m1 m2)\n  (let ([r1 (matrix-rows m1)] [c1 (matrix-cols m1)] [r2 (matrix-rows m2)] [c2 (matrix-cols m2)])\n    (if (not (= r1 c2))\n        `(error dimension-mismatch (,r1 ,c1) (,r2 ,c2))\n        (let* ([data1 (matrix-data m1)] [data2 (matrix-data m2)] [result (make-vector (* r1 c2) 0)])\n          (do ([i 0 (+ i 1)])\n              ((= i r1))\n            (do ([k 0 (+ k 1)])\n                ((= k c1))\n              (let ([a-ik (vector-ref data1 (+ (* i c1) k))] [b-row-k (* k c2)] [c-row-i (* i c2)])\n                (do ([j 0 (+ j 1)])\n                    ((= j c2))\n                  (let ([idx (+ c-row-i j)])\n                    (vector-set! result idx (+ (vector-ref result idx) (* a-ik (vector-ref data2 (+ b-row-k j))))))))))\n          (list 'matrix r1 c2 result)))))",
        "note": "Compatibility check must be `c1 = r2`.",
    },
    {
        "fn": "matrix-mul",
        "buggy": "(define (matrix-mul m1 m2)\n  (let ([r1 (matrix-rows m1)] [c1 (matrix-cols m1)] [r2 (matrix-rows m2)] [c2 (matrix-cols m2)])\n    (if (not (= c1 r2))\n        `(error dimension-mismatch (,r1 ,c1) (,r2 ,c2))\n        (let* ([data1 (matrix-data m1)] [data2 (matrix-data m2)] [result (make-vector (* r1 c2) 0)])\n          (do ([i 0 (+ i 1)])\n              ((= i r1))\n            (do ([k 0 (+ k 1)])\n                ((= k c1))\n              (let ([a-ik (vector-ref data1 (+ (* i c1) k))] [b-row-k (* k c2)] [c-row-i (* i c2)])\n                (do ([j 0 (+ j 1)])\n                    ((= j c2))\n                  (let ([idx (+ c-row-i j)])\n                    (vector-set! result idx (+ (vector-ref result idx) (* a-ik (vector-ref data2 (+ j b-row-k 1))))))))))\n          (list 'matrix r1 c2 result)))))",
        "note": "Inner multiply must use matching column index j without shifting.",
    },
    {
        "fn": "matrix-vec-mul",
        "buggy": "(define (matrix-vec-mul m v)\n  (let ([rows (matrix-rows m)] [cols (matrix-cols m)] [n (vector-length v)])\n    (if (not (= rows n))\n        `(error dimension-mismatch ,cols ,n)\n        (let ([data (matrix-data m)] [result (make-vector rows 0)])\n          (do ([i 0 (+ i 1)])\n              ((= i rows) result)\n            (do ([j 0 (+ j 1)] [sum 0 (+ sum (* (vector-ref data (+ (* i cols) j)) (vector-ref v j)))])\n                ((= j cols) (vector-set! result i sum))))))))",
        "note": "Vector length should match matrix column count.",
    },
    {
        "fn": "matrix-vec-mul",
        "buggy": "(define (matrix-vec-mul m v)\n  (let ([rows (matrix-rows m)] [cols (matrix-cols m)] [n (vector-length v)])\n    (if (not (= cols n))\n        `(error dimension-mismatch ,cols ,n)\n        (let ([data (matrix-data m)] [result (make-vector rows 0)])\n          (do ([i 0 (+ i 1)])\n              ((= i rows) result)\n            (do ([j 0 (+ j 1)] [sum 0 (+ sum (* (vector-ref data (+ (* j cols) i)) (vector-ref v j)))])\n                ((= j cols) (vector-set! result i sum))))))))",
        "note": "Accumulation index into matrix data must stay row-major at (i,j).",
    },
    {
        "fn": "matrix-identity",
        "buggy": "(define (matrix-identity n)\n  (let ([data (make-vector (* n n) 0)])\n    (do ([i 0 (+ i 1)])\n        ((= i n) (list 'matrix n n data))\n      (vector-set! data (+ (* i n) (+ i 1)) 1))))",
        "note": "Diagonal index should be i*n+i, not superdiagonal.",
    },
    {
        "fn": "matrix-identity",
        "buggy": "(define (matrix-identity n)\n  (list 'matrix n n (make-vector (* n n) 1)))",
        "note": "Identity matrix must contain zeros off the diagonal.",
    },
    {
        "fn": "matrix-submatrix",
        "buggy": "(define (matrix-submatrix m r1 c1 r2 c2)\n  (let ([rows (matrix-rows m)] [cols (matrix-cols m)] [data (matrix-data m)])\n    (cond\n      [(or (< r1 0) (< c1 0)) `(error out-of-bounds (,r1 ,c1))]\n      [(or (> r2 rows) (> c2 cols)) `(error out-of-bounds (,r2 ,c2))]\n      [(or (> r1 r2) (> c1 c2)) `(error invalid-range)]\n      [else\n       (let* ([new-rows (+ 1 (- r2 r1))] [new-cols (+ 1 (- c2 c1))] [result (make-vector (* new-rows new-cols) 0)])\n         (do ([i 0 (+ i 1)])\n             ((= i new-rows) (list 'matrix new-rows new-cols result))\n           (do ([j 0 (+ j 1)])\n               ((= j new-cols))\n             (vector-set! result (+ (* i new-cols) j)\n                          (vector-ref data (+ (* (+ r1 i) cols) (+ c1 j)))))))])))",
        "note": "Submatrix bounds are half-open; dimensions are (r2-r1) x (c2-c1).",
    },
    {
        "fn": "matrix-submatrix",
        "buggy": "(define (matrix-submatrix m r1 c1 r2 c2)\n  (let ([rows (matrix-rows m)] [cols (matrix-cols m)] [data (matrix-data m)])\n    (cond\n      [(or (< r1 0) (< c1 0)) `(error out-of-bounds (,r1 ,c1))]\n      [(or (>= r2 rows) (>= c2 cols)) `(error out-of-bounds (,r2 ,c2))]\n      [(or (> r1 r2) (> c1 c2)) `(error invalid-range)]\n      [else\n       (let* ([new-rows (- r2 r1)] [new-cols (- c2 c1)] [result (make-vector (* new-rows new-cols) 0)])\n         (do ([i 0 (+ i 1)])\n             ((= i new-rows) (list 'matrix new-rows new-cols result))\n           (do ([j 0 (+ j 1)])\n               ((= j new-cols))\n             (vector-set! result (+ (* i new-cols) j)\n                          (vector-ref data (+ (* (+ r1 i) cols) (+ c1 j)))))))])))",
        "note": "Upper bounds should allow r2=rows and c2=cols for full-edge slices.",
    },
]

DIFFICULTY = {
    "matrix-from-lists": "medium",
    "matrix-ref": "easy",
    "matrix-transpose": "medium",
    "matrix-map2": "medium",
    "matrix-mul": "hard",
    "matrix-vec-mul": "medium",
    "matrix-identity": "easy",
    "matrix-submatrix": "hard",
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
    sid = f"matrix_{family}_{family_counter[family]:03d}"
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
        tags=["linalg", "matrix", "spec-to-code", fn],
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
        tags=["linalg", "matrix", "spec-to-code", "skeleton", fn],
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
        tags=["linalg", "matrix", "translation", "python", fn],
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
        tags=["linalg", "matrix", "translation", "chez", fn],
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
        tags=["linalg", "matrix", "bugfix", fn],
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
        tags=["linalg", "matrix", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # matrix-from-lists
    ("matrix-from-lists", "Build matrix from '((1 2) (3 4)) and return as lists.", "(matrix->lists (matrix-from-lists '((1 2) (3 4))))", "(equal? (matrix->lists (matrix-from-lists '((1 2) (3 4)))) '((1 2) (3 4)))", "easy", ["direct"]),
    ("matrix-from-lists", "Return ragged-input error for '((1 2 3) (4 5)).", "(matrix-from-lists '((1 2 3) (4 5)))", "(equal? (matrix-from-lists '((1 2 3) (4 5))) '(error ragged-input row 1 expected-length 3 actual-length 2))", "medium", ["edge-case"]),
    ("matrix-from-lists", "Build from empty list and return shape pair.", "(cons (matrix-rows (matrix-from-lists '())) (matrix-cols (matrix-from-lists '())))", "(equal? (cons (matrix-rows (matrix-from-lists '())) (matrix-cols (matrix-from-lists '()))) '(0 . 0))", "easy", ["edge-case"]),
    ("matrix-from-lists", "Build two matrices then matrix-mul them and return lists.", "(matrix->lists (matrix-mul (matrix-from-lists '((1 2) (3 4))) (matrix-from-lists '((5 6) (7 8)))))", "(equal? (matrix->lists (matrix-mul (matrix-from-lists '((1 2) (3 4))) (matrix-from-lists '((5 6) (7 8))))) '((19 22) (43 50)))", "hard", ["integration"]),

    # matrix-ref
    ("matrix-ref", "Read element (1,2) from matrix ((1 2 3) (4 5 6)).", "(matrix-ref (matrix-from-lists '((1 2 3) (4 5 6))) 1 2)", "(equal? (matrix-ref (matrix-from-lists '((1 2 3) (4 5 6))) 1 2) 6)", "easy", ["direct"]),
    ("matrix-ref", "Access out-of-bounds row and return error.", "(matrix-ref (matrix-from-lists '((1 2) (3 4))) 3 0)", "(equal? (matrix-ref (matrix-from-lists '((1 2) (3 4))) 3 0) '(error out-of-bounds (3 0) (2 2)))", "easy", ["edge-case"]),
    ("matrix-ref", "Take submatrix then read its (0,1) entry.", "(matrix-ref (matrix-submatrix (matrix-from-lists '((1 2 3) (4 5 6) (7 8 9))) 1 0 3 2) 0 1)", "(equal? (matrix-ref (matrix-submatrix (matrix-from-lists '((1 2 3) (4 5 6) (7 8 9))) 1 0 3 2) 0 1) 5)", "medium", ["integration"]),
    ("matrix-ref", "Transpose matrix then read element (2,0).", "(matrix-ref (matrix-transpose (matrix-from-lists '((1 2 3) (4 5 6)))) 2 0)", "(equal? (matrix-ref (matrix-transpose (matrix-from-lists '((1 2 3) (4 5 6)))) 2 0) 3)", "medium", ["integration"]),

    # matrix-transpose
    ("matrix-transpose", "Transpose ((1 2 3) (4 5 6)).", "(matrix->lists (matrix-transpose (matrix-from-lists '((1 2 3) (4 5 6)))))", "(equal? (matrix->lists (matrix-transpose (matrix-from-lists '((1 2 3) (4 5 6))))) '((1 4) (2 5) (3 6)))", "medium", ["direct"]),
    ("matrix-transpose", "Double transpose should recover original matrix lists.", "(matrix->lists (matrix-transpose (matrix-transpose (matrix-from-lists '((9 8) (7 6))))))", "(equal? (matrix->lists (matrix-transpose (matrix-transpose (matrix-from-lists '((9 8) (7 6)))))) '((9 8) (7 6)))", "medium", ["property"]),
    ("matrix-transpose", "Transpose empty matrix and return lists.", "(matrix->lists (matrix-transpose (matrix-from-lists '())))", "(equal? (matrix->lists (matrix-transpose (matrix-from-lists '()))) '())", "easy", ["edge-case"]),
    ("matrix-transpose", "Multiply A by transpose(B) where A=((1 2)(3 4)) and B=((5 6)(7 8)).", "(matrix->lists (matrix-mul (matrix-from-lists '((1 2) (3 4))) (matrix-transpose (matrix-from-lists '((5 6) (7 8))))))", "(equal? (matrix->lists (matrix-mul (matrix-from-lists '((1 2) (3 4))) (matrix-transpose (matrix-from-lists '((5 6) (7 8)))))) '((17 23) (39 53)))", "hard", ["integration"]),

    # matrix-map2
    ("matrix-map2", "Add two 2x2 matrices with matrix-map2.", "(matrix->lists (matrix-map2 + (matrix-from-lists '((1 2) (3 4))) (matrix-from-lists '((10 20) (30 40)))))", "(equal? (matrix->lists (matrix-map2 + (matrix-from-lists '((1 2) (3 4))) (matrix-from-lists '((10 20) (30 40))))) '((11 22) (33 44)))", "medium", ["direct"]),
    ("matrix-map2", "Return dimension mismatch when shapes differ.", "(matrix-map2 + (matrix-from-lists '((1 2))) (matrix-from-lists '((3) (4))))", "(equal? (matrix-map2 + (matrix-from-lists '((1 2))) (matrix-from-lists '((3) (4)))) '(error dimension-mismatch (1 2) (2 1)))", "easy", ["edge-case"]),
    ("matrix-map2", "Subtract two matrices with matrix-map2 and return lists.", "(matrix->lists (matrix-map2 - (matrix-from-lists '((7 8) (9 10))) (matrix-from-lists '((1 2) (3 4)))))", "(equal? (matrix->lists (matrix-map2 - (matrix-from-lists '((7 8) (9 10))) (matrix-from-lists '((1 2) (3 4))))) '((6 6) (6 6)))", "medium", ["direct"]),
    ("matrix-map2", "Use matrix-map2(*) then sum first row via matrix-ref.", "(+ (matrix-ref (matrix-map2 * (matrix-from-lists '((1 2) (3 4))) (matrix-from-lists '((5 6) (7 8)))) 0 0) (matrix-ref (matrix-map2 * (matrix-from-lists '((1 2) (3 4))) (matrix-from-lists '((5 6) (7 8)))) 0 1))", "(equal? (+ (matrix-ref (matrix-map2 * (matrix-from-lists '((1 2) (3 4))) (matrix-from-lists '((5 6) (7 8)))) 0 0) (matrix-ref (matrix-map2 * (matrix-from-lists '((1 2) (3 4))) (matrix-from-lists '((5 6) (7 8)))) 0 1)) 17)", "hard", ["integration"]),

    # matrix-mul
    ("matrix-mul", "Multiply ((1 2)(3 4)) by ((5 6)(7 8)).", "(matrix->lists (matrix-mul (matrix-from-lists '((1 2) (3 4))) (matrix-from-lists '((5 6) (7 8)))))", "(equal? (matrix->lists (matrix-mul (matrix-from-lists '((1 2) (3 4))) (matrix-from-lists '((5 6) (7 8))))) '((19 22) (43 50)))", "hard", ["direct"]),
    ("matrix-mul", "Return dimension mismatch for (1x3)*(2x2).", "(matrix-mul (matrix-from-lists '((1 2 3))) (matrix-from-lists '((1 2) (3 4))))", "(equal? (matrix-mul (matrix-from-lists '((1 2 3))) (matrix-from-lists '((1 2) (3 4)))) '(error dimension-mismatch (1 3) (2 2)))", "medium", ["edge-case"]),
    ("matrix-mul", "Multiply by identity matrix and return unchanged lists.", "(matrix->lists (matrix-mul (matrix-from-lists '((2 3) (4 5))) (matrix-identity 2)))", "(equal? (matrix->lists (matrix-mul (matrix-from-lists '((2 3) (4 5))) (matrix-identity 2))) '((2 3) (4 5)))", "hard", ["property"]),
    ("matrix-mul", "Compute (A*B)(1,1) for A=((1 2)(3 4)), B=((5 6)(7 8)).", "(matrix-ref (matrix-mul (matrix-from-lists '((1 2) (3 4))) (matrix-from-lists '((5 6) (7 8)))) 1 1)", "(equal? (matrix-ref (matrix-mul (matrix-from-lists '((1 2) (3 4))) (matrix-from-lists '((5 6) (7 8)))) 1 1) 50)", "hard", ["integration"]),

    # matrix-vec-mul
    ("matrix-vec-mul", "Multiply ((1 2 3)(4 5 6)) by vector #(1 2 3).", "(matrix-vec-mul (matrix-from-lists '((1 2 3) (4 5 6))) (vector 1 2 3))", "(equal? (matrix-vec-mul (matrix-from-lists '((1 2 3) (4 5 6))) (vector 1 2 3)) (vector 14 32))", "medium", ["direct"]),
    ("matrix-vec-mul", "Return dimension mismatch for 2x2 matrix and length-3 vector.", "(matrix-vec-mul (matrix-from-lists '((1 2) (3 4))) (vector 1 2 3))", "(equal? (matrix-vec-mul (matrix-from-lists '((1 2) (3 4))) (vector 1 2 3)) '(error dimension-mismatch 2 3))", "easy", ["edge-case"]),
    ("matrix-vec-mul", "Multiply identity 3x3 by vector #(7 8 9).", "(matrix-vec-mul (matrix-identity 3) (vector 7 8 9))", "(equal? (matrix-vec-mul (matrix-identity 3) (vector 7 8 9)) (vector 7 8 9))", "medium", ["property"]),
    ("matrix-vec-mul", "Multiply transpose(((1 2 3)(4 5 6))) by #(1 1) and return vector.", "(matrix-vec-mul (matrix-transpose (matrix-from-lists '((1 2 3) (4 5 6)))) (vector 1 1))", "(equal? (matrix-vec-mul (matrix-transpose (matrix-from-lists '((1 2 3) (4 5 6)))) (vector 1 1)) (vector 5 7 9))", "hard", ["integration"]),

    # matrix-identity
    ("matrix-identity", "Return 3x3 identity as lists.", "(matrix->lists (matrix-identity 3))", "(equal? (matrix->lists (matrix-identity 3)) '((1 0 0) (0 1 0) (0 0 1)))", "easy", ["direct"]),
    ("matrix-identity", "Return 0x0 identity as empty list of rows.", "(matrix->lists (matrix-identity 0))", "(equal? (matrix->lists (matrix-identity 0)) '())", "easy", ["edge-case"]),
    ("matrix-identity", "Check that identity(4) has ones on all diagonal entries.", "(and (= (matrix-ref (matrix-identity 4) 0 0) 1) (= (matrix-ref (matrix-identity 4) 1 1) 1) (= (matrix-ref (matrix-identity 4) 2 2) 1) (= (matrix-ref (matrix-identity 4) 3 3) 1))", "(equal? (and (= (matrix-ref (matrix-identity 4) 0 0) 1) (= (matrix-ref (matrix-identity 4) 1 1) 1) (= (matrix-ref (matrix-identity 4) 2 2) 1) (= (matrix-ref (matrix-identity 4) 3 3) 1)) #t)", "medium", ["property"]),
    ("matrix-identity", "Multiply identity(2) by ((9 8)(7 6)).", "(matrix->lists (matrix-mul (matrix-identity 2) (matrix-from-lists '((9 8) (7 6)))))", "(equal? (matrix->lists (matrix-mul (matrix-identity 2) (matrix-from-lists '((9 8) (7 6))))) '((9 8) (7 6)))", "hard", ["integration"]),

    # matrix-submatrix
    ("matrix-submatrix", "Extract rows [1,3), cols [1,3) from 3x4 matrix and return lists.", "(matrix->lists (matrix-submatrix (matrix-from-lists '((1 2 3 4) (5 6 7 8) (9 10 11 12))) 1 1 3 3))", "(equal? (matrix->lists (matrix-submatrix (matrix-from-lists '((1 2 3 4) (5 6 7 8) (9 10 11 12))) 1 1 3 3)) '((6 7) (10 11)))", "hard", ["direct"]),
    ("matrix-submatrix", "Return invalid-range when start row exceeds end row.", "(matrix-submatrix (matrix-from-lists '((1 2) (3 4))) 2 0 1 1)", "(equal? (matrix-submatrix (matrix-from-lists '((1 2) (3 4))) 2 0 1 1) '(error invalid-range))", "medium", ["edge-case"]),
    ("matrix-submatrix", "Return out-of-bounds error for negative start index.", "(matrix-submatrix (matrix-from-lists '((1 2) (3 4))) -1 0 1 1)", "(equal? (matrix-submatrix (matrix-from-lists '((1 2) (3 4))) -1 0 1 1) '(error out-of-bounds (-1 0)))", "medium", ["edge-case"]),
    ("matrix-submatrix", "Take full-range submatrix and compare to original lists.", "(matrix->lists (matrix-submatrix (matrix-from-lists '((2 3) (4 5))) 0 0 2 2))", "(equal? (matrix->lists (matrix-submatrix (matrix-from-lists '((2 3) (4 5))) 0 0 2 2)) '((2 3) (4 5)))", "hard", ["property"]),
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
