#!/usr/bin/env python3
"""Generate SFT samples for lattice/linalg/sparse.ss."""

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

SOURCE_MODULE = "lattice/linalg/sparse.ss"
SOURCE_TEST = "lattice/linalg/test-sparse.ss"

GLOBAL_DEFS = [
    "(define *sparse-epsilon* 1e-15)",
]

DEFS: Dict[str, str] = {
    # Helpers
    "make-sparse-coo": """(define (make-sparse-coo rows cols row-indices col-indices values)
  (list 'sparse-coo rows cols row-indices col-indices values))""",
    "sparse-coo-rows": """(define (sparse-coo-rows m)
  (list-ref m 1))""",
    "sparse-coo-cols": """(define (sparse-coo-cols m)
  (list-ref m 2))""",
    "sparse-coo-row-indices": """(define (sparse-coo-row-indices m)
  (list-ref m 3))""",
    "sparse-coo-col-indices": """(define (sparse-coo-col-indices m)
  (list-ref m 4))""",
    "sparse-coo-values": """(define (sparse-coo-values m)
  (list-ref m 5))""",
    "sparse-coo-nnz": """(define (sparse-coo-nnz m)
  (vector-length (sparse-coo-values m)))""",
    "make-sparse-csr": """(define (make-sparse-csr rows cols row-ptrs col-indices values)
  (list 'sparse-csr rows cols row-ptrs col-indices values))""",
    "sparse-csr-rows": """(define (sparse-csr-rows m)
  (list-ref m 1))""",
    "sparse-csr-cols": """(define (sparse-csr-cols m)
  (list-ref m 2))""",
    "sparse-csr-row-ptrs": """(define (sparse-csr-row-ptrs m)
  (list-ref m 3))""",
    "sparse-csr-col-indices": """(define (sparse-csr-col-indices m)
  (list-ref m 4))""",
    "sparse-csr-values": """(define (sparse-csr-values m)
  (list-ref m 5))""",
    "sparse-csr-nnz": """(define (sparse-csr-nnz m)
  (vector-length (sparse-csr-values m)))""",

    # Targets
    "sparse-coo-from-triplets": """(define (sparse-coo-from-triplets rows cols triplets)
  (let* ([n (length triplets)]
         [row-idx (make-vector n 0)]
         [col-idx (make-vector n 0)]
         [vals (make-vector n 0)])
        (do ([i 0 (+ i 1)]
             [ts triplets (cdr ts)])
            ((= i n) (make-sparse-coo rows cols row-idx col-idx vals))
            (let ([t (car ts)])
                 (vector-set! row-idx i (car t))
                 (vector-set! col-idx i (cadr t))
                 (vector-set! vals i (caddr t))))))""",
    "sparse-coo-ref": """(define (sparse-coo-ref m i j)
  (let ([row-idx (sparse-coo-row-indices m)]
        [col-idx (sparse-coo-col-indices m)]
        [vals (sparse-coo-values m)]
        [nnz (sparse-coo-nnz m)])
       (let loop ([k 0])
            (cond
             [(= k nnz) 0]
             [(and (= (vector-ref row-idx k) i)
                   (= (vector-ref col-idx k) j))
              (vector-ref vals k)]
             [else (loop (+ k 1))]))))""",
    "coo->csr": """(define (coo->csr coo)
  (let* ([rows (sparse-coo-rows coo)]
         [cols (sparse-coo-cols coo)]
         [row-idx (sparse-coo-row-indices coo)]
         [col-idx (sparse-coo-col-indices coo)]
         [vals (sparse-coo-values coo)]
         [nnz (sparse-coo-nnz coo)])
        (if (= nnz 0)
            (let ([row-ptrs (make-vector (+ rows 1) 0)])
                 (make-sparse-csr rows cols row-ptrs (make-vector 0 0) (make-vector 0 0)))
            (let* ([indices (make-vector nnz 0)])
                  (do ([k 0 (+ k 1)])
                      ((= k nnz))
                      (vector-set! indices k k))
                  (vector-sort-by!
                   (lambda (a b)
                           (let ([ra (vector-ref row-idx a)]
                                 [rb (vector-ref row-idx b)])
                                (or (< ra rb)
                                    (and (= ra rb)
                                         (< (vector-ref col-idx a)
                                            (vector-ref col-idx b))))))
                   indices)
                  (let ([row-ptrs (make-vector (+ rows 1) 0)]
                        [out-cols (make-vector nnz 0)]
                        [out-vals (make-vector nnz 0)])
                       (do ([k 0 (+ k 1)]
                            [current-row 0 (vector-ref row-idx (vector-ref indices k))])
                           ((= k nnz)
                            (do ([r (+ current-row 1) (+ r 1)])
                                ((> r rows))
                                (vector-set! row-ptrs r nnz)))
                           (let* ([idx (vector-ref indices k)]
                                  [r (vector-ref row-idx idx)]
                                  [c (vector-ref col-idx idx)]
                                  [v (vector-ref vals idx)])
                                 (when (> r current-row)
                                       (do ([prev-row (+ current-row 1) (+ prev-row 1)])
                                           ((> prev-row r))
                                           (vector-set! row-ptrs prev-row k)))
                                 (vector-set! out-cols k c)
                                 (vector-set! out-vals k v)))
                       (make-sparse-csr rows cols row-ptrs out-cols out-vals))))))""",
    "sparse-csr-ref": """(define (sparse-csr-ref m i j)
  (let* ([row-ptrs (sparse-csr-row-ptrs m)]
         [col-idx (sparse-csr-col-indices m)]
         [vals (sparse-csr-values m)]
         [start (vector-ref row-ptrs i)]
         [end (vector-ref row-ptrs (+ i 1))])
        (let loop ([k start])
             (cond
              [(= k end) 0]
              [(= (vector-ref col-idx k) j) (vector-ref vals k)]
              [else (loop (+ k 1))]))))""",
    "dense->sparse-coo": """(define (dense->sparse-coo m . tol-arg)
  (let* ([tol (if (null? tol-arg) 0 (car tol-arg))]
         [rows (matrix-rows m)]
         [cols (matrix-cols m)]
         [data (matrix-data m)]
         [nnz (let loop ([i 0] [count 0])
                   (if (= i (vector-length data))
                       count
                       (loop (+ i 1)
                             (if (> (abs (vector-ref data i)) tol)
                                 (+ count 1)
                                 count))))]
         [row-idx (make-vector nnz 0)]
         [col-idx (make-vector nnz 0)]
         [vals (make-vector nnz 0)])
        (let loop ([i 0] [k 0])
             (if (= i (vector-length data))
                 (make-sparse-coo rows cols row-idx col-idx vals)
                 (let ([v (vector-ref data i)])
                      (if (> (abs v) tol)
                          (begin
                           (vector-set! row-idx k (quotient i cols))
                           (vector-set! col-idx k (remainder i cols))
                           (vector-set! vals k v)
                           (loop (+ i 1) (+ k 1)))
                          (loop (+ i 1) k)))))))""",
    "sparse-csr-vec-mul": """(define (sparse-csr-vec-mul m v)
  (let* ([rows (sparse-csr-rows m)]
         [cols (sparse-csr-cols m)]
         [n (vector-length v)])
        (if (not (= cols n))
            `(error dimension-mismatch ,cols ,n)
            (let ([row-ptrs (sparse-csr-row-ptrs m)]
                  [col-idx (sparse-csr-col-indices m)]
                  [vals (sparse-csr-values m)])
                 (vec-tabulate rows i
                   (let ([start (vector-ref row-ptrs i)]
                         [end (vector-ref row-ptrs (+ i 1))])
                        (range-fold sum 0 k start end
                          (+ sum (* (vector-ref vals k)
                                    (vector-ref v (vector-ref col-idx k)))))))))))""",
    "sparse-coo-add-impl": """(define (sparse-coo-add-impl a b . eps-arg)
  (let* ([eps (if (null? eps-arg) *sparse-epsilon* (car eps-arg))]
         [rows (sparse-coo-rows a)]
         [cols (sparse-coo-cols a)]
         [acc-a (let ([row-idx (sparse-coo-row-indices a)]
                      [col-idx (sparse-coo-col-indices a)]
                      [vals (sparse-coo-values a)]
                      [nnz-a (sparse-coo-nnz a)])
                  (let loop ([k 0] [acc hamt-empty])
                    (if (= k nnz-a) acc
                        (let* ([i (vector-ref row-idx k)]
                               [j (vector-ref col-idx k)]
                               [key (cons i j)]
                               [old (hamt-lookup-or key acc 0)])
                          (loop (+ k 1) (hamt-assoc key (+ old (vector-ref vals k)) acc))))))]
         [acc (let ([row-idx (sparse-coo-row-indices b)]
                    [col-idx (sparse-coo-col-indices b)]
                    [vals (sparse-coo-values b)]
                    [nnz-b (sparse-coo-nnz b)])
                (let loop ([k 0] [h acc-a])
                  (if (= k nnz-b) h
                      (let* ([i (vector-ref row-idx k)]
                             [j (vector-ref col-idx k)]
                             [key (cons i j)]
                             [old (hamt-lookup-or key h 0)])
                        (loop (+ k 1) (hamt-assoc key (+ old (vector-ref vals k)) h))))))])
        (let* ([entries (hamt-entries acc)]
               [triplets (filter-map
                          (lambda (entry)
                            (let ([v (cdr entry)])
                              (if (< (abs v) eps)
                                  #f
                                  (let ([key (car entry)])
                                    (list (car key) (cdr key) v)))))
                          entries)]
               [sorted (sort-by (lambda (a b)
                                          (or (< (car a) (car b))
                                              (and (= (car a) (car b))
                                                   (< (cadr a) (cadr b)))))
                                triplets)]
               [nnz (length sorted)]
               [out-rows (make-vector nnz 0)]
               [out-cols (make-vector nnz 0)]
               [out-vals (make-vector nnz 0)])
              (do ([k 0 (+ k 1)]
                   [ts sorted (cdr ts)])
                  ((= k nnz) (make-sparse-coo rows cols out-rows out-cols out-vals))
                  (let ([t (car ts)])
                       (vector-set! out-rows k (car t))
                       (vector-set! out-cols k (cadr t))
                       (vector-set! out-vals k (caddr t)))))))""",
    "sparse-coo-drop-below": """(define (sparse-coo-drop-below tol coo)
  (let* ([rows (sparse-coo-rows coo)]
         [cols (sparse-coo-cols coo)]
         [row-idx (sparse-coo-row-indices coo)]
         [col-idx (sparse-coo-col-indices coo)]
         [vals (sparse-coo-values coo)]
         [nnz (sparse-coo-nnz coo)]
         [keep-count (let loop ([k 0] [count 0])
                          (if (= k nnz)
                              count
                              (loop (+ k 1)
                                    (if (>= (abs (vector-ref vals k)) tol)
                                        (+ count 1)
                                        count))))]
         [new-rows (make-vector keep-count 0)]
         [new-cols (make-vector keep-count 0)]
         [new-vals (make-vector keep-count 0)])
        (let loop ([k 0] [j 0])
             (if (= k nnz)
                 (make-sparse-coo rows cols new-rows new-cols new-vals)
                 (let ([v (vector-ref vals k)])
                      (if (>= (abs v) tol)
                          (begin
                           (vector-set! new-rows j (vector-ref row-idx k))
                           (vector-set! new-cols j (vector-ref col-idx k))
                           (vector-set! new-vals j v)
                           (loop (+ k 1) (+ j 1)))
                          (loop (+ k 1) j)))))))""",
}

DEPENDS: Dict[str, List[str]] = {
    "sparse-coo-from-triplets": ["make-sparse-coo"],
    "sparse-coo-ref": [
        "sparse-coo-row-indices",
        "sparse-coo-col-indices",
        "sparse-coo-values",
        "sparse-coo-nnz",
    ],
    "coo->csr": [
        "sparse-coo-rows",
        "sparse-coo-cols",
        "sparse-coo-row-indices",
        "sparse-coo-col-indices",
        "sparse-coo-values",
        "sparse-coo-nnz",
        "make-sparse-csr",
    ],
    "sparse-csr-ref": ["sparse-csr-row-ptrs", "sparse-csr-col-indices", "sparse-csr-values"],
    "dense->sparse-coo": ["make-sparse-coo"],
    "sparse-csr-vec-mul": ["sparse-csr-rows", "sparse-csr-cols", "sparse-csr-row-ptrs", "sparse-csr-col-indices", "sparse-csr-values"],
    "sparse-coo-add-impl": [
        "sparse-coo-rows",
        "sparse-coo-cols",
        "sparse-coo-row-indices",
        "sparse-coo-col-indices",
        "sparse-coo-values",
        "sparse-coo-nnz",
        "make-sparse-coo",
    ],
    "sparse-coo-drop-below": [
        "sparse-coo-rows",
        "sparse-coo-cols",
        "sparse-coo-row-indices",
        "sparse-coo-col-indices",
        "sparse-coo-values",
        "sparse-coo-nnz",
        "make-sparse-coo",
    ],
}

FUNCTION_ORDER = [
    "sparse-coo-from-triplets",
    "sparse-coo-ref",
    "coo->csr",
    "sparse-csr-ref",
    "dense->sparse-coo",
    "sparse-csr-vec-mul",
    "sparse-coo-add-impl",
    "sparse-coo-drop-below",
]

FUNCTION_SPECS = {
    "sparse-coo-from-triplets": "Build COO sparse matrix from list of (row col value) triplets.",
    "sparse-coo-ref": "Lookup COO element at (i,j); return 0 if coordinate is absent.",
    "coo->csr": "Convert COO to CSR, sorting by (row,col) and building row pointers.",
    "sparse-csr-ref": "Lookup CSR element at (i,j) by scanning the row segment.",
    "dense->sparse-coo": "Convert dense matrix to COO, dropping values with |v| <= tolerance (default 0).",
    "sparse-csr-vec-mul": "Multiply CSR matrix by vector; return dimension mismatch error when incompatible.",
    "sparse-coo-add-impl": "Add two COO matrices via HAMT accumulation and tolerance-based near-zero dropping.",
    "sparse-coo-drop-below": "Remove COO entries with magnitude below tolerance.",
}

SKELETONS = {
    "sparse-coo-from-triplets": """(define (sparse-coo-from-triplets rows cols triplets)
  ;; TODO: fill row/col/value vectors from triplets
  <TODO>)""",
    "sparse-coo-ref": """(define (sparse-coo-ref m i j)
  ;; TODO: scan COO entries for coordinate, else return 0
  <TODO>)""",
    "coo->csr": """(define (coo->csr coo)
  ;; TODO: sort COO indices and build CSR row pointers and payload arrays
  <TODO>)""",
    "sparse-csr-ref": """(define (sparse-csr-ref m i j)
  ;; TODO: search row segment in CSR
  <TODO>)""",
    "dense->sparse-coo": """(define (dense->sparse-coo m . tol-arg)
  ;; TODO: two-pass dense->COO conversion with tolerance
  <TODO>)""",
    "sparse-csr-vec-mul": """(define (sparse-csr-vec-mul m v)
  ;; TODO: CSR matrix-vector multiply with dimension check
  <TODO>)""",
    "sparse-coo-add-impl": """(define (sparse-coo-add-impl a b . eps-arg)
  ;; TODO: accumulate sums by coordinate and drop near-zeros
  <TODO>)""",
    "sparse-coo-drop-below": """(define (sparse-coo-drop-below tol coo)
  ;; TODO: filter COO entries by magnitude threshold
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "sparse-coo-from-triplets": "(let ([m (sparse-coo-from-triplets 3 3 '((0 0 5) (1 2 7) (2 1 9)))]) (and (= (sparse-coo-nnz m) 3) (= (sparse-coo-ref m 1 2) 7) (= (sparse-coo-ref m 0 1) 0)))",
    "sparse-coo-ref": "(let ([m (sparse-coo-from-triplets 3 3 '((0 0 5) (1 2 7) (2 1 9)))]) (and (= (sparse-coo-ref m 0 0) 5) (= (sparse-coo-ref m 2 1) 9) (= (sparse-coo-ref m 1 1) 0)))",
    "coo->csr": "(let* ([coo (sparse-coo-from-triplets 3 3 '((0 2 2) (0 0 1) (1 1 3) (2 2 5) (2 0 4)))] [csr (coo->csr coo)]) (and (= (sparse-csr-ref csr 0 0) 1) (= (sparse-csr-ref csr 0 2) 2) (= (sparse-csr-ref csr 2 0) 4) (equal? (vector->list (sparse-csr-row-ptrs csr)) '(0 2 3 5))))",
    "sparse-csr-ref": "(let* ([coo (sparse-coo-from-triplets 3 3 '((0 0 5) (0 2 3) (1 1 7) (2 0 2) (2 2 9)))] [csr (coo->csr coo)]) (and (= (sparse-csr-ref csr 0 2) 3) (= (sparse-csr-ref csr 1 1) 7) (= (sparse-csr-ref csr 0 1) 0)))",
    "dense->sparse-coo": "(let* ([dense (matrix-from-lists '((1 0 2) (0 3 0) (4 0 5)))] [coo (dense->sparse-coo dense)]) (and (= (sparse-coo-nnz coo) 5) (= (sparse-coo-ref coo 0 0) 1) (= (sparse-coo-ref coo 1 1) 3) (= (sparse-coo-ref coo 0 1) 0)))",
    "sparse-csr-vec-mul": "(let* ([coo (sparse-coo-from-triplets 3 3 '((0 0 1) (0 1 2) (1 1 3) (2 0 4) (2 2 5)))] [csr (coo->csr coo)]) (and (equal? (sparse-csr-vec-mul csr (vec 1 2 3)) (vec 5 6 19)) (equal? (sparse-csr-vec-mul csr (vec 1 2)) '(error dimension-mismatch 3 2))))",
    "sparse-coo-add-impl": "(let* ([a (sparse-coo-from-triplets 2 2 '((0 0 1.0) (1 1 2.0)))] [b (sparse-coo-from-triplets 2 2 '((0 0 3.0) (0 1 4.0)))] [c (sparse-coo-add-impl a b)] [z (sparse-coo-add-impl a (sparse-coo-from-triplets 2 2 '((0 0 -1.0) (1 1 -2.0))))]) (and (= (sparse-coo-ref c 0 0) 4.0) (= (sparse-coo-ref c 0 1) 4.0) (= (sparse-coo-ref c 1 1) 2.0) (= (sparse-coo-nnz z) 0)))",
    "sparse-coo-drop-below": "(let* ([coo (sparse-coo-from-triplets 3 3 '((0 0 1.0) (0 1 1e-16) (1 1 2.0) (2 2 1e-20)))] [cleaned (sparse-coo-drop-below 1e-14 coo)]) (and (= (sparse-coo-nnz cleaned) 2) (= (sparse-coo-ref cleaned 0 0) 1.0) (= (sparse-coo-ref cleaned 1 1) 2.0) (= (sparse-coo-ref cleaned 0 1) 0)))",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")
TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "sparse-coo-from-triplets": "def sparse_coo_from_triplets(rows, cols, triplets):\n    n = len(triplets)\n    r = [0] * n\n    c = [0] * n\n    v = [0] * n\n    for i, t in enumerate(triplets):\n        r[i] = t[0]\n        c[i] = t[1]\n        v[i] = t[2]\n    return make_sparse_coo(rows, cols, list_to_vector(r), list_to_vector(c), list_to_vector(v))",
    "sparse-coo-ref": "def sparse_coo_ref(m, i, j):\n    row_idx = sparse_coo_row_indices(m)\n    col_idx = sparse_coo_col_indices(m)\n    vals = sparse_coo_values(m)\n    nnz = sparse_coo_nnz(m)\n    k = 0\n    while k < nnz:\n        if row_idx[k] == i and col_idx[k] == j:\n            return vals[k]\n        k += 1\n    return 0",
    "coo->csr": "def coo_to_csr(coo):\n    rows = sparse_coo_rows(coo)\n    cols = sparse_coo_cols(coo)\n    row_idx = sparse_coo_row_indices(coo)\n    col_idx = sparse_coo_col_indices(coo)\n    vals = sparse_coo_values(coo)\n    nnz = sparse_coo_nnz(coo)\n    if nnz == 0:\n        return make_sparse_csr(rows, cols, list_to_vector([0] * (rows + 1)), list_to_vector([]), list_to_vector([]))\n    idxs = list(range(nnz))\n    idxs.sort(key=lambda k: (row_idx[k], col_idx[k]))\n    row_ptrs = [0] * (rows + 1)\n    out_cols = [0] * nnz\n    out_vals = [0] * nnz\n    current_row = 0\n    for k, idx in enumerate(idxs):\n        r = row_idx[idx]\n        c = col_idx[idx]\n        v = vals[idx]\n        while current_row < r:\n            current_row += 1\n            row_ptrs[current_row] = k\n        out_cols[k] = c\n        out_vals[k] = v\n    for r in range(current_row + 1, rows + 1):\n        row_ptrs[r] = nnz\n    return make_sparse_csr(rows, cols, list_to_vector(row_ptrs), list_to_vector(out_cols), list_to_vector(out_vals))",
    "sparse-csr-ref": "def sparse_csr_ref(m, i, j):\n    row_ptrs = sparse_csr_row_ptrs(m)\n    col_idx = sparse_csr_col_indices(m)\n    vals = sparse_csr_values(m)\n    start = row_ptrs[i]\n    end = row_ptrs[i + 1]\n    k = start\n    while k < end:\n        if col_idx[k] == j:\n            return vals[k]\n        k += 1\n    return 0",
    "dense->sparse-coo": "def dense_to_sparse_coo(m, tol=0):\n    rows = matrix_rows(m)\n    cols = matrix_cols(m)\n    data = matrix_data(m)\n    r, c, v = [], [], []\n    for i, x in enumerate(data):\n        if abs(x) > tol:\n            r.append(i // cols)\n            c.append(i % cols)\n            v.append(x)\n    return make_sparse_coo(rows, cols, list_to_vector(r), list_to_vector(c), list_to_vector(v))",
    "sparse-csr-vec-mul": "def sparse_csr_vec_mul(m, v):\n    rows = sparse_csr_rows(m)\n    cols = sparse_csr_cols(m)\n    if cols != len(v):\n        return ['error', 'dimension-mismatch', cols, len(v)]\n    row_ptrs = sparse_csr_row_ptrs(m)\n    col_idx = sparse_csr_col_indices(m)\n    vals = sparse_csr_values(m)\n    out = [0] * rows\n    for i in range(rows):\n        s = 0\n        for k in range(row_ptrs[i], row_ptrs[i + 1]):\n            s += vals[k] * v[col_idx[k]]\n        out[i] = s\n    return list_to_vector(out)",
    "sparse-coo-add-impl": "def sparse_coo_add_impl(a, b, eps=1e-15):\n    rows = sparse_coo_rows(a)\n    cols = sparse_coo_cols(a)\n    acc = {}\n    for k in range(sparse_coo_nnz(a)):\n        key = (sparse_coo_row_indices(a)[k], sparse_coo_col_indices(a)[k])\n        acc[key] = acc.get(key, 0) + sparse_coo_values(a)[k]\n    for k in range(sparse_coo_nnz(b)):\n        key = (sparse_coo_row_indices(b)[k], sparse_coo_col_indices(b)[k])\n        acc[key] = acc.get(key, 0) + sparse_coo_values(b)[k]\n    triplets = [(i, j, v) for (i, j), v in acc.items() if abs(v) >= eps]\n    triplets.sort(key=lambda t: (t[0], t[1]))\n    return sparse_coo_from_triplets(rows, cols, triplets)",
    "sparse-coo-drop-below": "def sparse_coo_drop_below(tol, coo):\n    rows = sparse_coo_rows(coo)\n    cols = sparse_coo_cols(coo)\n    triplets = []\n    for k in range(sparse_coo_nnz(coo)):\n        v = sparse_coo_values(coo)[k]\n        if abs(v) >= tol:\n            triplets.append((sparse_coo_row_indices(coo)[k], sparse_coo_col_indices(coo)[k], v))\n    return sparse_coo_from_triplets(rows, cols, triplets)",
}

CHEZ_SNIPPETS = {
    "sparse-coo-from-triplets": "(define (coo-from-triplets r c ts)\n  (let* ((n (length ts))\n         (ri (make-vector n 0))\n         (ci (make-vector n 0))\n         (vs (make-vector n 0)))\n    (do ((i 0 (+ i 1))\n         (xs ts (cdr xs)))\n        ((= i n) (make-sparse-coo r c ri ci vs))\n      (let ((t (car xs)))\n        (vector-set! ri i (car t))\n        (vector-set! ci i (cadr t))\n        (vector-set! vs i (caddr t))))))",
    "sparse-coo-ref": "(define (coo-ref m i j)\n  (let ((ri (sparse-coo-row-indices m))\n        (ci (sparse-coo-col-indices m))\n        (vs (sparse-coo-values m))\n        (n (sparse-coo-nnz m)))\n    (let loop ((k 0))\n      (cond ((= k n) 0)\n            ((and (= (vector-ref ri k) i) (= (vector-ref ci k) j)) (vector-ref vs k))\n            (else (loop (+ k 1)))))))",
    "coo->csr": """(define (coo->csr0 coo)
  (let* ((rows (sparse-coo-rows coo))
         (cols (sparse-coo-cols coo))
         (row-idx (sparse-coo-row-indices coo))
         (col-idx (sparse-coo-col-indices coo))
         (vals (sparse-coo-values coo))
         (nnz (sparse-coo-nnz coo)))
    (if (= nnz 0)
        (let ((row-ptrs (make-vector (+ rows 1) 0)))
          (make-sparse-csr rows cols row-ptrs (make-vector 0 0) (make-vector 0 0)))
        (let* ((indices (make-vector nnz 0)))
          (do ((k 0 (+ k 1)))
              ((= k nnz))
            (vector-set! indices k k))
          (vector-sort-by!
           (lambda (a b)
             (let ((ra (vector-ref row-idx a))
                   (rb (vector-ref row-idx b)))
               (or (< ra rb)
                   (and (= ra rb)
                        (< (vector-ref col-idx a)
                           (vector-ref col-idx b))))))
           indices)
          (let ((row-ptrs (make-vector (+ rows 1) 0))
                (out-cols (make-vector nnz 0))
                (out-vals (make-vector nnz 0)))
            (do ((k 0 (+ k 1))
                 (current-row 0 (vector-ref row-idx (vector-ref indices k))))
                ((= k nnz)
                 (do ((r (+ current-row 1) (+ r 1)))
                     ((> r rows))
                   (vector-set! row-ptrs r nnz))
                 (make-sparse-csr rows cols row-ptrs out-cols out-vals))
              (let* ((idx (vector-ref indices k))
                     (r (vector-ref row-idx idx))
                     (c (vector-ref col-idx idx))
                     (v (vector-ref vals idx)))
                (when (> r current-row)
                  (do ((prev-row (+ current-row 1) (+ prev-row 1)))
                      ((> prev-row r))
                    (vector-set! row-ptrs prev-row k)))
                (vector-set! out-cols k c)
                (vector-set! out-vals k v))))))))""",
    "sparse-csr-ref": "(define (csr-ref m i j)\n  (let* ((rp (sparse-csr-row-ptrs m))\n         (ci (sparse-csr-col-indices m))\n         (vs (sparse-csr-values m))\n         (start (vector-ref rp i))\n         (end (vector-ref rp (+ i 1))))\n    (let loop ((k start))\n      (cond ((= k end) 0)\n            ((= (vector-ref ci k) j) (vector-ref vs k))\n            (else (loop (+ k 1)))))))",
    "dense->sparse-coo": "(define (dense->coo m . tol-arg)\n  (let* ((tol (if (null? tol-arg) 0 (car tol-arg)))\n         (rows (matrix-rows m))\n         (cols (matrix-cols m))\n         (data (matrix-data m))\n         (nnz (let loop ((i 0) (count 0))\n                (if (= i (vector-length data))\n                    count\n                    (loop (+ i 1)\n                          (if (> (abs (vector-ref data i)) tol) (+ count 1) count)))))\n         (ri (make-vector nnz 0))\n         (ci (make-vector nnz 0))\n         (vs (make-vector nnz 0)))\n    (let loop ((i 0) (k 0))\n      (if (= i (vector-length data))\n          (make-sparse-coo rows cols ri ci vs)\n          (let ((v (vector-ref data i)))\n            (if (> (abs v) tol)\n                (begin\n                  (vector-set! ri k (quotient i cols))\n                  (vector-set! ci k (remainder i cols))\n                  (vector-set! vs k v)\n                  (loop (+ i 1) (+ k 1)))\n                (loop (+ i 1) k)))))))",
    "sparse-csr-vec-mul": "(define (csr-mv m v)\n  (let* ((rows (sparse-csr-rows m))\n         (cols (sparse-csr-cols m))\n         (n (vector-length v)))\n    (if (not (= cols n))\n        `(error dimension-mismatch ,cols ,n)\n        (let ((rp (sparse-csr-row-ptrs m))\n              (ci (sparse-csr-col-indices m))\n              (vs (sparse-csr-values m)))\n          (vec-tabulate rows i\n            (let ((start (vector-ref rp i))\n                  (end (vector-ref rp (+ i 1))))\n              (range-fold sum 0 k start end\n                (+ sum (* (vector-ref vs k)\n                          (vector-ref v (vector-ref ci k)))))))))))",
    "sparse-coo-add-impl": """(define (coo-add-impl0 a b . eps-arg)
  (let* ((eps (if (null? eps-arg) *sparse-epsilon* (car eps-arg)))
         (rows (sparse-coo-rows a))
         (cols (sparse-coo-cols a))
         (acc-a (let ((row-idx (sparse-coo-row-indices a))
                      (col-idx (sparse-coo-col-indices a))
                      (vals (sparse-coo-values a))
                      (nnz-a (sparse-coo-nnz a)))
                  (let loop ((k 0) (acc hamt-empty))
                    (if (= k nnz-a)
                        acc
                        (let* ((i (vector-ref row-idx k))
                               (j (vector-ref col-idx k))
                               (key (cons i j))
                               (old (hamt-lookup-or key acc 0)))
                          (loop (+ k 1) (hamt-assoc key (+ old (vector-ref vals k)) acc)))))))
         (acc (let ((row-idx (sparse-coo-row-indices b))
                    (col-idx (sparse-coo-col-indices b))
                    (vals (sparse-coo-values b))
                    (nnz-b (sparse-coo-nnz b)))
                (let loop ((k 0) (h acc-a))
                  (if (= k nnz-b)
                      h
                      (let* ((i (vector-ref row-idx k))
                             (j (vector-ref col-idx k))
                             (key (cons i j))
                             (old (hamt-lookup-or key h 0)))
                        (loop (+ k 1) (hamt-assoc key (+ old (vector-ref vals k)) h))))))))
    (let* ((entries (hamt-entries acc))
           (triplets (filter-map
                      (lambda (entry)
                        (let ((v (cdr entry)))
                          (if (< (abs v) eps)
                              #f
                              (let ((key (car entry)))
                                (list (car key) (cdr key) v)))))
                      entries))
           (sorted (sort-by (lambda (x y)
                              (or (< (car x) (car y))
                                  (and (= (car x) (car y))
                                       (< (cadr x) (cadr y)))))
                            triplets))
           (nnz (length sorted))
           (out-rows (make-vector nnz 0))
           (out-cols (make-vector nnz 0))
           (out-vals (make-vector nnz 0)))
      (do ((k 0 (+ k 1))
           (ts sorted (cdr ts)))
          ((= k nnz) (make-sparse-coo rows cols out-rows out-cols out-vals))
        (let ((t (car ts)))
          (vector-set! out-rows k (car t))
          (vector-set! out-cols k (cadr t))
          (vector-set! out-vals k (caddr t))))))""",
    "sparse-coo-drop-below": """(define (coo-drop tol coo)
  (let* ((rows (sparse-coo-rows coo))
         (cols (sparse-coo-cols coo))
         (row-idx (sparse-coo-row-indices coo))
         (col-idx (sparse-coo-col-indices coo))
         (vals (sparse-coo-values coo))
         (nnz (sparse-coo-nnz coo))
         (keep-count (let loop ((k 0) (count 0))
                       (if (= k nnz)
                           count
                           (loop (+ k 1)
                                 (if (>= (abs (vector-ref vals k)) tol)
                                     (+ count 1)
                                     count)))))
         (new-rows (make-vector keep-count 0))
         (new-cols (make-vector keep-count 0))
         (new-vals (make-vector keep-count 0)))
    (let loop ((k 0) (j 0))
      (if (= k nnz)
          (make-sparse-coo rows cols new-rows new-cols new-vals)
          (let ((v (vector-ref vals k)))
            (if (>= (abs v) tol)
                (begin
                  (vector-set! new-rows j (vector-ref row-idx k))
                  (vector-set! new-cols j (vector-ref col-idx k))
                  (vector-set! new-vals j v)
                  (loop (+ k 1) (+ j 1)))
                (loop (+ k 1) j))))))""",
}

BUGGY_CASES = [
    {
        "fn": "sparse-coo-from-triplets",
        "buggy": "(define (sparse-coo-from-triplets rows cols triplets)\n  (let* ([n (length triplets)] [row-idx (make-vector n 0)] [col-idx (make-vector n 0)] [vals (make-vector n 0)])\n    (do ([i 0 (+ i 1)] [ts triplets (cdr ts)])\n        ((= i n) (make-sparse-coo rows cols row-idx col-idx vals))\n      (let ([t (car ts)])\n        (vector-set! row-idx i (cadr t))\n        (vector-set! col-idx i (car t))\n        (vector-set! vals i (caddr t))))))",
        "note": "Row and column indices are swapped while loading triplets.",
    },
    {
        "fn": "sparse-coo-from-triplets",
        "buggy": "(define (sparse-coo-from-triplets rows cols triplets)\n  (let* ([n (length triplets)] [row-idx (make-vector n 0)] [col-idx (make-vector n 0)] [vals (make-vector n 0)])\n    (do ([i 0 (+ i 1)] [ts triplets (cdr ts)])\n        ((= i n) (make-sparse-coo rows cols row-idx col-idx vals))\n      (let ([t (car ts)])\n        (vector-set! row-idx i (car t))\n        (vector-set! col-idx i (cadr t))\n        (vector-set! vals i 0)))))",
        "note": "Values vector is not populated from triplets.",
    },
    {
        "fn": "sparse-coo-ref",
        "buggy": "(define (sparse-coo-ref m i j)\n  (let ([row-idx (sparse-coo-row-indices m)] [col-idx (sparse-coo-col-indices m)] [vals (sparse-coo-values m)] [nnz (sparse-coo-nnz m)])\n    (let loop ([k 0])\n      (cond [(= k nnz) 0]\n            [(and (= (vector-ref row-idx k) j) (= (vector-ref col-idx k) i)) (vector-ref vals k)]\n            [else (loop (+ k 1))]))))",
        "note": "Lookup compares coordinates with i/j swapped.",
    },
    {
        "fn": "sparse-coo-ref",
        "buggy": "(define (sparse-coo-ref m i j)\n  (let ([row-idx (sparse-coo-row-indices m)] [col-idx (sparse-coo-col-indices m)] [vals (sparse-coo-values m)] [nnz (sparse-coo-nnz m)])\n    (let loop ([k 0])\n      (cond [(= k nnz) #f]\n            [(and (= (vector-ref row-idx k) i) (= (vector-ref col-idx k) j)) (vector-ref vals k)]\n            [else (loop (+ k 1))]))))",
        "note": "Absent entries should return numeric zero, not #f.",
    },
    {
        "fn": "coo->csr",
        "buggy": "(define (coo->csr coo)\n  (let* ([rows (sparse-coo-rows coo)] [cols (sparse-coo-cols coo)] [row-idx (sparse-coo-row-indices coo)] [col-idx (sparse-coo-col-indices coo)] [vals (sparse-coo-values coo)] [nnz (sparse-coo-nnz coo)])\n    (if (= nnz 0)\n        (make-sparse-csr rows cols (make-vector rows 0) (make-vector 0 0) (make-vector 0 0))\n        (let* ([indices (make-vector nnz 0)])\n          (do ([k 0 (+ k 1)]) ((= k nnz)) (vector-set! indices k k))\n          (vector-sort-by! (lambda (a b) (< (vector-ref col-idx a) (vector-ref col-idx b))) indices)\n          (let ([row-ptrs (make-vector (+ rows 1) 0)] [out-cols (make-vector nnz 0)] [out-vals (make-vector nnz 0)])\n            (do ([k 0 (+ k 1)] [current-row 0 (vector-ref row-idx (vector-ref indices k))])\n                ((= k nnz) (make-sparse-csr rows cols row-ptrs out-cols out-vals))\n              (let* ([idx (vector-ref indices k)] [r (vector-ref row-idx idx)] [c (vector-ref col-idx idx)] [v (vector-ref vals idx)])\n                (when (> r current-row)\n                  (do ([prev-row (+ current-row 1) (+ prev-row 1)]) ((> prev-row r)) (vector-set! row-ptrs prev-row k)))\n                (vector-set! out-cols k c)\n                (vector-set! out-vals k v))))))))",
        "note": "Sorting for CSR must be row-major (row then col), and empty row-ptrs needs length rows+1.",
    },
    {
        "fn": "coo->csr",
        "buggy": "(define (coo->csr coo)\n  (let* ([rows (sparse-coo-rows coo)] [cols (sparse-coo-cols coo)] [row-idx (sparse-coo-row-indices coo)] [col-idx (sparse-coo-col-indices coo)] [vals (sparse-coo-values coo)] [nnz (sparse-coo-nnz coo)])\n    (if (= nnz 0)\n        (let ([row-ptrs (make-vector (+ rows 1) 0)]) (make-sparse-csr rows cols row-ptrs (make-vector 0 0) (make-vector 0 0)))\n        (let* ([indices (make-vector nnz 0)])\n          (do ([k 0 (+ k 1)]) ((= k nnz)) (vector-set! indices k k))\n          (vector-sort-by! (lambda (a b) (let ([ra (vector-ref row-idx a)] [rb (vector-ref row-idx b)]) (or (< ra rb) (and (= ra rb) (< (vector-ref col-idx a) (vector-ref col-idx b)))))) indices)\n          (let ([row-ptrs (make-vector (+ rows 1) 0)] [out-cols (make-vector nnz 0)] [out-vals (make-vector nnz 0)])\n            (do ([k 0 (+ k 1)] [current-row 0 (vector-ref row-idx (vector-ref indices k))])\n                ((= k nnz) (make-sparse-csr rows cols row-ptrs out-cols out-vals))\n              (let* ([idx (vector-ref indices k)] [r (vector-ref row-idx idx)] [c (vector-ref col-idx idx)] [v (vector-ref vals idx)])\n                (when (> r current-row)\n                  (do ([prev-row (+ current-row 1) (+ prev-row 1)]) ((> prev-row r)) (vector-set! row-ptrs prev-row (+ k 1))))\n                (vector-set! out-cols k c)\n                (vector-set! out-vals k v))))))))",
        "note": "Row pointer updates should record current write index k, not k+1.",
    },
    {
        "fn": "sparse-csr-ref",
        "buggy": "(define (sparse-csr-ref m i j)\n  (let* ([row-ptrs (sparse-csr-row-ptrs m)] [col-idx (sparse-csr-col-indices m)] [vals (sparse-csr-values m)] [start 0] [end (vector-ref row-ptrs (+ i 1))])\n    (let loop ([k start])\n      (cond [(= k end) 0] [(= (vector-ref col-idx k) j) (vector-ref vals k)] [else (loop (+ k 1))]))))",
        "note": "Search must start at row-specific start pointer, not always 0.",
    },
    {
        "fn": "sparse-csr-ref",
        "buggy": "(define (sparse-csr-ref m i j)\n  (let* ([row-ptrs (sparse-csr-row-ptrs m)] [col-idx (sparse-csr-col-indices m)] [vals (sparse-csr-values m)] [start (vector-ref row-ptrs i)] [end (vector-ref row-ptrs (+ i 1))])\n    (let loop ([k start])\n      (cond [(= k end) 0] [(= (vector-ref col-idx k) i) (vector-ref vals k)] [else (loop (+ k 1))]))))",
        "note": "Inner comparison must match requested column j, not row i.",
    },
    {
        "fn": "dense->sparse-coo",
        "buggy": "(define (dense->sparse-coo m . tol-arg)\n  (let* ([tol (if (null? tol-arg) 0 (car tol-arg))] [rows (matrix-rows m)] [cols (matrix-cols m)] [data (matrix-data m)] [nnz (vector-length data)] [row-idx (make-vector nnz 0)] [col-idx (make-vector nnz 0)] [vals (make-vector nnz 0)])\n    (let loop ([i 0] [k 0])\n      (if (= i (vector-length data))\n          (make-sparse-coo rows cols row-idx col-idx vals)\n          (let ([v (vector-ref data i)])\n            (vector-set! row-idx k (quotient i cols))\n            (vector-set! col-idx k (remainder i cols))\n            (vector-set! vals k v)\n            (loop (+ i 1) (+ k 1)))))))",
        "note": "Conversion must skip entries below tolerance and size arrays to true nnz.",
    },
    {
        "fn": "dense->sparse-coo",
        "buggy": "(define (dense->sparse-coo m . tol-arg)\n  (let* ([tol (if (null? tol-arg) 0 (car tol-arg))] [rows (matrix-rows m)] [cols (matrix-cols m)] [data (matrix-data m)] [nnz (let loop ([i 0] [count 0]) (if (= i (vector-length data)) count (loop (+ i 1) (if (> (abs (vector-ref data i)) tol) (+ count 1) count))))] [row-idx (make-vector nnz 0)] [col-idx (make-vector nnz 0)] [vals (make-vector nnz 0)])\n    (let loop ([i 0] [k 0])\n      (if (= i (vector-length data))\n          (make-sparse-coo rows cols row-idx col-idx vals)\n          (let ([v (vector-ref data i)])\n            (if (> (abs v) tol)\n                (begin\n                  (vector-set! row-idx k (remainder i cols))\n                  (vector-set! col-idx k (quotient i cols))\n                  (vector-set! vals k v)\n                  (loop (+ i 1) (+ k 1)))\n                (loop (+ i 1) k)))))))",
        "note": "Dense linear index decoding uses row=quotient, col=remainder (not swapped).",
    },
    {
        "fn": "sparse-csr-vec-mul",
        "buggy": "(define (sparse-csr-vec-mul m v)\n  (let* ([rows (sparse-csr-rows m)] [cols (sparse-csr-cols m)] [n (vector-length v)])\n    (if (not (= rows n))\n        `(error dimension-mismatch ,cols ,n)\n        (let ([row-ptrs (sparse-csr-row-ptrs m)] [col-idx (sparse-csr-col-indices m)] [vals (sparse-csr-values m)])\n          (vec-tabulate rows i\n            (let ([start (vector-ref row-ptrs i)] [end (vector-ref row-ptrs (+ i 1))])\n              (range-fold sum 0 k start end\n                (+ sum (* (vector-ref vals k) (vector-ref v (vector-ref col-idx k)))))))))))",
        "note": "Dimension check must compare matrix cols to vector length.",
    },
    {
        "fn": "sparse-csr-vec-mul",
        "buggy": "(define (sparse-csr-vec-mul m v)\n  (let* ([rows (sparse-csr-rows m)] [cols (sparse-csr-cols m)] [n (vector-length v)])\n    (if (not (= cols n))\n        `(error dimension-mismatch ,cols ,n)\n        (let ([row-ptrs (sparse-csr-row-ptrs m)] [col-idx (sparse-csr-col-indices m)] [vals (sparse-csr-values m)])\n          (vec-tabulate rows i\n            (let ([start (vector-ref row-ptrs i)] [end (vector-ref row-ptrs (+ i 1))])\n              (range-fold sum 0 k start end\n                (+ sum (* (vector-ref vals k) (vector-ref v i))))))))))",
        "note": "Each term must index vector by sparse column index, not current row.",
    },
    {
        "fn": "sparse-coo-add-impl",
        "buggy": "(define (sparse-coo-add-impl a b . eps-arg)\n  (let* ([eps (if (null? eps-arg) *sparse-epsilon* (car eps-arg))] [rows (sparse-coo-rows a)] [cols (sparse-coo-cols a)] [acc-a (let ([row-idx (sparse-coo-row-indices a)] [col-idx (sparse-coo-col-indices a)] [vals (sparse-coo-values a)] [nnz-a (sparse-coo-nnz a)]) (let loop ([k 0] [acc hamt-empty]) (if (= k nnz-a) acc (let* ([i (vector-ref row-idx k)] [j (vector-ref col-idx k)] [key (cons i j)] [old (hamt-lookup-or key acc 0)]) (loop (+ k 1) (hamt-assoc key (+ old (vector-ref vals k)) acc))))))] [acc (let ([row-idx (sparse-coo-row-indices b)] [col-idx (sparse-coo-col-indices b)] [vals (sparse-coo-values b)] [nnz-b (sparse-coo-nnz b)]) (let loop ([k 0] [h acc-a]) (if (= k nnz-b) h (let* ([i (vector-ref row-idx k)] [j (vector-ref col-idx k)] [key (cons i j)]) (loop (+ k 1) (hamt-assoc key (vector-ref vals k) h))))))])\n    (let* ([entries (hamt-entries acc)] [triplets (filter-map (lambda (entry) (let ([v (cdr entry)]) (if (< (abs v) eps) #f (let ([key (car entry)]) (list (car key) (cdr key) v))))) entries)] [sorted (sort-by (lambda (a b) (or (< (car a) (car b)) (and (= (car a) (car b)) (< (cadr a) (cadr b))))) triplets)] [nnz (length sorted)] [out-rows (make-vector nnz 0)] [out-cols (make-vector nnz 0)] [out-vals (make-vector nnz 0)])\n      (do ([k 0 (+ k 1)] [ts sorted (cdr ts)]) ((= k nnz) (make-sparse-coo rows cols out-rows out-cols out-vals)) (let ([t (car ts)]) (vector-set! out-rows k (car t)) (vector-set! out-cols k (cadr t)) (vector-set! out-vals k (caddr t)))))))",
        "note": "Accumulator must add into existing coordinate value when B has duplicates/overlaps.",
    },
    {
        "fn": "sparse-coo-add-impl",
        "buggy": "(define (sparse-coo-add-impl a b . eps-arg)\n  (let* ([eps (if (null? eps-arg) *sparse-epsilon* (car eps-arg))] [rows (sparse-coo-rows a)] [cols (sparse-coo-cols a)] [acc-a (let ([row-idx (sparse-coo-row-indices a)] [col-idx (sparse-coo-col-indices a)] [vals (sparse-coo-values a)] [nnz-a (sparse-coo-nnz a)]) (let loop ([k 0] [acc hamt-empty]) (if (= k nnz-a) acc (let* ([i (vector-ref row-idx k)] [j (vector-ref col-idx k)] [key (cons i j)] [old (hamt-lookup-or key acc 0)]) (loop (+ k 1) (hamt-assoc key (+ old (vector-ref vals k)) acc))))))] [acc acc-a])\n    (let* ([entries (hamt-entries acc)] [triplets (filter-map (lambda (entry) (let ([v (cdr entry)]) (if (> (abs v) eps) #f (let ([key (car entry)]) (list (car key) (cdr key) v))))) entries)] [sorted (sort-by (lambda (a b) (or (< (car a) (car b)) (and (= (car a) (car b)) (< (cadr a) (cadr b))))) triplets)] [nnz (length sorted)] [out-rows (make-vector nnz 0)] [out-cols (make-vector nnz 0)] [out-vals (make-vector nnz 0)])\n      (do ([k 0 (+ k 1)] [ts sorted (cdr ts)]) ((= k nnz) (make-sparse-coo rows cols out-rows out-cols out-vals)) (let ([t (car ts)]) (vector-set! out-rows k (car t)) (vector-set! out-cols k (cadr t)) (vector-set! out-vals k (caddr t)))))))",
        "note": "B-side accumulation is missing and the tolerance filter is reversed; both break addition semantics.",
    },
    {
        "fn": "sparse-coo-drop-below",
        "buggy": "(define (sparse-coo-drop-below tol coo)\n  (let* ([rows (sparse-coo-rows coo)] [cols (sparse-coo-cols coo)] [row-idx (sparse-coo-row-indices coo)] [col-idx (sparse-coo-col-indices coo)] [vals (sparse-coo-values coo)] [nnz (sparse-coo-nnz coo)] [keep-count (let loop ([k 0] [count 0]) (if (= k nnz) count (loop (+ k 1) (if (>= (abs (vector-ref vals k)) tol) (+ count 1) count))))] [new-rows (make-vector keep-count 0)] [new-cols (make-vector keep-count 0)] [new-vals (make-vector keep-count 0)])\n    (let loop ([k 0] [j 0])\n      (if (= k nnz)\n          (make-sparse-coo rows cols new-rows new-cols new-vals)\n          (let ([v (vector-ref vals k)])\n            (if (>= (abs v) tol)\n                (begin (vector-set! new-rows j (vector-ref row-idx k)) (vector-set! new-cols j (vector-ref col-idx k)) (vector-set! new-vals j 0) (loop (+ k 1) (+ j 1)))\n                (loop (+ k 1) j)))))))",
        "note": "Copied entries should preserve original values; zeroing output values corrupts kept coordinates.",
    },
    {
        "fn": "sparse-coo-drop-below",
        "buggy": "(define (sparse-coo-drop-below tol coo)\n  (let* ([rows (sparse-coo-rows coo)] [cols (sparse-coo-cols coo)] [row-idx (sparse-coo-row-indices coo)] [col-idx (sparse-coo-col-indices coo)] [vals (sparse-coo-values coo)] [nnz (sparse-coo-nnz coo)] [keep-count (let loop ([k 0] [count 0]) (if (= k nnz) count (loop (+ k 1) (if (< (abs (vector-ref vals k)) tol) (+ count 1) count))))] [new-rows (make-vector keep-count 0)] [new-cols (make-vector keep-count 0)] [new-vals (make-vector keep-count 0)])\n    (let loop ([k 0] [j 0])\n      (if (= k nnz)\n          (make-sparse-coo rows cols new-rows new-cols new-vals)\n          (let ([v (vector-ref vals k)])\n            (if (< (abs v) tol)\n                (begin (vector-set! new-rows j (vector-ref row-idx k)) (vector-set! new-cols j (vector-ref col-idx k)) (vector-set! new-vals j v) (loop (+ k 1) (+ j 1)))\n                (loop (+ k 1) j)))))))",
        "note": "Keep condition is inverted; entries with magnitude >= tol should be retained.",
    },
]

DIFFICULTY = {
    "sparse-coo-from-triplets": "medium",
    "sparse-coo-ref": "easy",
    "coo->csr": "hard",
    "sparse-csr-ref": "easy",
    "dense->sparse-coo": "hard",
    "sparse-csr-vec-mul": "hard",
    "sparse-coo-add-impl": "hard",
    "sparse-coo-drop-below": "medium",
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
    sid = f"sparse_{family}_{family_counter[family]:03d}"
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
    parts = GLOBAL_DEFS + [DEFS[d] for d in dependency_closure(fn)] + [DEFS[fn], VERIFY_BY_FUNCTION[fn]]
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
        tags=["linalg", "sparse", "spec-to-code", fn],
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
        tags=["linalg", "sparse", "spec-to-code", "skeleton", fn],
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
Preserve behavior exactly, including sparse zero/default semantics.

Target function name: `{fn}`

```python
{PYTHON_SNIPPETS[fn]}
```

Return only the Scheme definition.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["linalg", "sparse", "translation", "python", fn],
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
        tags=["linalg", "sparse", "translation", "chez", fn],
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
        tags=["linalg", "sparse", "bugfix", fn],
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
        tags=["linalg", "sparse", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # sparse-coo-from-triplets
    ("sparse-coo-from-triplets", "Build COO from three triplets and return nnz.", "(sparse-coo-nnz (sparse-coo-from-triplets 3 3 '((0 0 5) (1 2 7) (2 1 9))))", "(equal? (sparse-coo-nnz (sparse-coo-from-triplets 3 3 '((0 0 5) (1 2 7) (2 1 9)))) 3)", "easy", ["direct"]),
    ("sparse-coo-from-triplets", "Build COO and read value at (1,2).", "(sparse-coo-ref (sparse-coo-from-triplets 3 3 '((0 0 5) (1 2 7) (2 1 9))) 1 2)", "(equal? (sparse-coo-ref (sparse-coo-from-triplets 3 3 '((0 0 5) (1 2 7) (2 1 9))) 1 2) 7)", "medium", ["integration"]),
    ("sparse-coo-from-triplets", "Create empty COO from empty triplet list and return nnz.", "(sparse-coo-nnz (sparse-coo-from-triplets 4 5 '()))", "(equal? (sparse-coo-nnz (sparse-coo-from-triplets 4 5 '())) 0)", "easy", ["edge-case"]),
    ("coo->csr", "Build COO then convert to CSR and query (2,1).", "(sparse-csr-ref (coo->csr (sparse-coo-from-triplets 3 3 '((0 0 5) (1 2 7) (2 1 9)))) 2 1)", "(equal? (sparse-csr-ref (coo->csr (sparse-coo-from-triplets 3 3 '((0 0 5) (1 2 7) (2 1 9)))) 2 1) 9)", "hard", ["integration"]),

    # sparse-coo-ref
    ("sparse-coo-ref", "Read present entry (0,0) from COO.", "(sparse-coo-ref (sparse-coo-from-triplets 2 2 '((0 0 4) (1 1 6))) 0 0)", "(equal? (sparse-coo-ref (sparse-coo-from-triplets 2 2 '((0 0 4) (1 1 6))) 0 0) 4)", "easy", ["direct"]),
    ("sparse-coo-ref", "Read missing entry (0,1) from COO and get zero.", "(sparse-coo-ref (sparse-coo-from-triplets 2 2 '((0 0 4) (1 1 6))) 0 1)", "(equal? (sparse-coo-ref (sparse-coo-from-triplets 2 2 '((0 0 4) (1 1 6))) 0 1) 0)", "easy", ["edge-case"]),
    ("sparse-coo-ref", "Read from COO after dense->sparse conversion.", "(sparse-coo-ref (dense->sparse-coo (matrix-from-lists '((1 0) (0 2)))) 1 1)", "(equal? (sparse-coo-ref (dense->sparse-coo (matrix-from-lists '((1 0) (0 2)))) 1 1) 2)", "medium", ["integration"]),
    ("sparse-coo-ref", "Return #t iff ref returns zero for guaranteed absent coordinate.", "(= (sparse-coo-ref (sparse-coo-from-triplets 3 3 '((0 0 1))) 2 2) 0)", "(equal? (= (sparse-coo-ref (sparse-coo-from-triplets 3 3 '((0 0 1))) 2 2) 0) #t)", "easy", ["property"]),

    # coo->csr
    ("coo->csr", "Convert unsorted COO to CSR and read (0,2).", "(sparse-csr-ref (coo->csr (sparse-coo-from-triplets 3 3 '((0 2 2) (0 0 1) (1 1 3) (2 2 5) (2 0 4)))) 0 2)", "(equal? (sparse-csr-ref (coo->csr (sparse-coo-from-triplets 3 3 '((0 2 2) (0 0 1) (1 1 3) (2 2 5) (2 0 4)))) 0 2) 2)", "hard", ["direct"]),
    ("coo->csr", "Convert empty COO to CSR and return row-ptrs list.", "(vector->list (sparse-csr-row-ptrs (coo->csr (sparse-coo-from-triplets 3 4 '()))))", "(equal? (vector->list (sparse-csr-row-ptrs (coo->csr (sparse-coo-from-triplets 3 4 '())))) '(0 0 0 0))", "medium", ["edge-case"]),
    ("sparse-coo-from-triplets", "Build single-triplet COO, convert to CSR, and read (2,0).", "(sparse-csr-ref (coo->csr (sparse-coo-from-triplets 3 3 '((2 0 9)))) 2 0)", "(equal? (sparse-csr-ref (coo->csr (sparse-coo-from-triplets 3 3 '((2 0 9)))) 2 0) 9)", "medium", ["integration"]),
    ("coo->csr", "Return #t iff CSR nnz equals COO nnz after conversion.", "(let ([coo (sparse-coo-from-triplets 3 3 '((0 0 1) (1 1 2) (2 2 3)))]) (= (sparse-csr-nnz (coo->csr coo)) (sparse-coo-nnz coo)))", "(equal? (let ([coo (sparse-coo-from-triplets 3 3 '((0 0 1) (1 1 2) (2 2 3)))]) (= (sparse-csr-nnz (coo->csr coo)) (sparse-coo-nnz coo))) #t)", "medium", ["property"]),

    # sparse-csr-ref
    ("sparse-csr-ref", "Read existing (1,1) from CSR matrix.", "(let* ([csr (coo->csr (sparse-coo-from-triplets 3 3 '((1 1 7))))]) (sparse-csr-ref csr 1 1))", "(equal? (let* ([csr (coo->csr (sparse-coo-from-triplets 3 3 '((1 1 7))))]) (sparse-csr-ref csr 1 1)) 7)", "easy", ["direct"]),
    ("sparse-csr-ref", "Read missing CSR coordinate and get zero.", "(let* ([csr (coo->csr (sparse-coo-from-triplets 3 3 '((1 1 7))))]) (sparse-csr-ref csr 0 2))", "(equal? (let* ([csr (coo->csr (sparse-coo-from-triplets 3 3 '((1 1 7))))]) (sparse-csr-ref csr 0 2)) 0)", "easy", ["edge-case"]),
    ("sparse-csr-ref", "Read from CSR produced by dense->sparse conversion.", "(sparse-csr-ref (coo->csr (dense->sparse-coo (matrix-from-lists '((0 2) (3 0))))) 1 0)", "(equal? (sparse-csr-ref (coo->csr (dense->sparse-coo (matrix-from-lists '((0 2) (3 0))))) 1 0) 3)", "medium", ["integration"]),
    ("sparse-csr-ref", "Return #t iff csr-ref equals coo-ref after conversion.", "(let* ([coo (sparse-coo-from-triplets 2 3 '((0 1 4) (1 2 5)))] [csr (coo->csr coo)]) (= (sparse-csr-ref csr 1 2) (sparse-coo-ref coo 1 2)))", "(equal? (let* ([coo (sparse-coo-from-triplets 2 3 '((0 1 4) (1 2 5)))] [csr (coo->csr coo)]) (= (sparse-csr-ref csr 1 2) (sparse-coo-ref coo 1 2))) #t)", "medium", ["property"]),

    # dense->sparse-coo
    ("dense->sparse-coo", "Convert dense matrix ((1 0 2)(0 3 0)) to COO and return nnz.", "(sparse-coo-nnz (dense->sparse-coo (matrix-from-lists '((1 0 2) (0 3 0)))))", "(equal? (sparse-coo-nnz (dense->sparse-coo (matrix-from-lists '((1 0 2) (0 3 0))))) 3)", "medium", ["direct"]),
    ("dense->sparse-coo", "Convert all-zero dense matrix to COO and return nnz.", "(sparse-coo-nnz (dense->sparse-coo (matrix-from-lists '((0 0) (0 0)))))", "(equal? (sparse-coo-nnz (dense->sparse-coo (matrix-from-lists '((0 0) (0 0))))) 0)", "easy", ["edge-case"]),
    ("dense->sparse-coo", "Use tolerance 1e-14 to drop tiny value 1e-16.", "(sparse-coo-nnz (dense->sparse-coo (matrix-from-lists '((1.0 1e-16) (0.0 2.0))) 1e-14))", "(equal? (sparse-coo-nnz (dense->sparse-coo (matrix-from-lists '((1.0 1e-16) (0.0 2.0))) 1e-14)) 2)", "hard", ["edge-case"]),
    ("dense->sparse-coo", "Convert dense to COO then CSR and read (0,2).", "(sparse-csr-ref (coo->csr (dense->sparse-coo (matrix-from-lists '((1 0 2) (0 3 0))))) 0 2)", "(equal? (sparse-csr-ref (coo->csr (dense->sparse-coo (matrix-from-lists '((1 0 2) (0 3 0))))) 0 2) 2)", "hard", ["integration"]),

    # sparse-csr-vec-mul
    ("sparse-csr-vec-mul", "Multiply CSR by vector #(1 2 3) and return result.", "(sparse-csr-vec-mul (coo->csr (sparse-coo-from-triplets 3 3 '((0 0 1) (0 1 2) (1 1 3) (2 0 4) (2 2 5)))) (vec 1 2 3))", "(equal? (sparse-csr-vec-mul (coo->csr (sparse-coo-from-triplets 3 3 '((0 0 1) (0 1 2) (1 1 3) (2 0 4) (2 2 5)))) (vec 1 2 3)) (vec 5 6 19))", "hard", ["direct"]),
    ("sparse-csr-vec-mul", "Return dimension mismatch for wrong vector length.", "(sparse-csr-vec-mul (coo->csr (sparse-coo-from-triplets 2 3 '((0 0 1)))) (vec 1 2))", "(equal? (sparse-csr-vec-mul (coo->csr (sparse-coo-from-triplets 2 3 '((0 0 1)))) (vec 1 2)) '(error dimension-mismatch 3 2))", "medium", ["edge-case"]),
    ("sparse-csr-vec-mul", "Compare sparse matvec with dense matrix-vec-mul on same data.", "(let* ([dense (matrix-from-lists '((2 0 1) (0 3 0) (4 0 2)))] [csr (coo->csr (dense->sparse-coo dense))] [v (vec 1 2 3)]) (equal? (sparse-csr-vec-mul csr v) (matrix-vec-mul dense v)))", "(equal? (let* ([dense (matrix-from-lists '((2 0 1) (0 3 0) (4 0 2)))] [csr (coo->csr (dense->sparse-coo dense))] [v (vec 1 2 3)]) (equal? (sparse-csr-vec-mul csr v) (matrix-vec-mul dense v))) #t)", "hard", ["integration"]),
    ("sparse-csr-vec-mul", "Matvec with zero vector should return zero vector.", "(sparse-csr-vec-mul (coo->csr (sparse-coo-from-triplets 2 2 '((0 0 3) (1 1 4)))) (vec 0 0))", "(equal? (sparse-csr-vec-mul (coo->csr (sparse-coo-from-triplets 2 2 '((0 0 3) (1 1 4)))) (vec 0 0)) (vec 0 0))", "medium", ["property"]),

    # sparse-coo-add-impl
    ("sparse-coo-add-impl", "Add two COO matrices and read (0,0).", "(sparse-coo-ref (sparse-coo-add-impl (sparse-coo-from-triplets 2 2 '((0 0 1.0) (1 1 2.0))) (sparse-coo-from-triplets 2 2 '((0 0 3.0) (0 1 4.0)))) 0 0)", "(equal? (sparse-coo-ref (sparse-coo-add-impl (sparse-coo-from-triplets 2 2 '((0 0 1.0) (1 1 2.0))) (sparse-coo-from-triplets 2 2 '((0 0 3.0) (0 1 4.0)))) 0 0) 4.0)", "hard", ["direct"]),
    ("sparse-coo-add-impl", "Cancellation should produce nnz 0 under default epsilon.", "(sparse-coo-nnz (sparse-coo-add-impl (sparse-coo-from-triplets 2 2 '((0 0 1.0))) (sparse-coo-from-triplets 2 2 '((0 0 -1.0)))))", "(equal? (sparse-coo-nnz (sparse-coo-add-impl (sparse-coo-from-triplets 2 2 '((0 0 1.0))) (sparse-coo-from-triplets 2 2 '((0 0 -1.0))))) 0)", "medium", ["edge-case"]),
    ("sparse-coo-add-impl", "Add with duplicate coordinates across inputs and read sum.", "(sparse-coo-ref (sparse-coo-add-impl (sparse-coo-from-triplets 2 2 '((0 1 2.0) (0 1 3.0))) (sparse-coo-from-triplets 2 2 '((0 1 4.0)))) 0 1)", "(equal? (sparse-coo-ref (sparse-coo-add-impl (sparse-coo-from-triplets 2 2 '((0 1 2.0) (0 1 3.0))) (sparse-coo-from-triplets 2 2 '((0 1 4.0)))) 0 1) 9.0)", "hard", ["property"]),
    ("sparse-coo-add-impl", "Add A and B, convert result to CSR, and read (1,1).", "(sparse-csr-ref (coo->csr (sparse-coo-add-impl (sparse-coo-from-triplets 2 2 '((1 1 2.0))) (sparse-coo-from-triplets 2 2 '((1 1 5.0))))) 1 1)", "(equal? (sparse-csr-ref (coo->csr (sparse-coo-add-impl (sparse-coo-from-triplets 2 2 '((1 1 2.0))) (sparse-coo-from-triplets 2 2 '((1 1 5.0))))) 1 1) 7.0)", "hard", ["integration"]),

    # sparse-coo-drop-below
    ("sparse-coo-drop-below", "Drop entries below 1e-14 and return nnz.", "(sparse-coo-nnz (sparse-coo-drop-below 1e-14 (sparse-coo-from-triplets 3 3 '((0 0 1.0) (0 1 1e-16) (1 1 2.0) (2 2 1e-20)))))", "(equal? (sparse-coo-nnz (sparse-coo-drop-below 1e-14 (sparse-coo-from-triplets 3 3 '((0 0 1.0) (0 1 1e-16) (1 1 2.0) (2 2 1e-20))))) 2)", "medium", ["direct"]),
    ("sparse-coo-drop-below", "With tol=0, keep all nonzero values from COO.", "(sparse-coo-nnz (sparse-coo-drop-below 0 (sparse-coo-from-triplets 2 2 '((0 0 1.0) (1 1 -2.0)))))", "(equal? (sparse-coo-nnz (sparse-coo-drop-below 0 (sparse-coo-from-triplets 2 2 '((0 0 1.0) (1 1 -2.0))))) 2)", "easy", ["edge-case"]),
    ("sparse-coo-drop-below", "Drop tiny entries then read surviving (1,1).", "(sparse-coo-ref (sparse-coo-drop-below 1e-10 (sparse-coo-from-triplets 2 2 '((0 0 1e-12) (1 1 3.0)))) 1 1)", "(equal? (sparse-coo-ref (sparse-coo-drop-below 1e-10 (sparse-coo-from-triplets 2 2 '((0 0 1e-12) (1 1 3.0)))) 1 1) 3.0)", "medium", ["integration"]),
    ("sparse-coo-drop-below", "Dropping after add should match direct cancellation behavior.", "(sparse-coo-nnz (sparse-coo-drop-below 1e-14 (sparse-coo-add-impl (sparse-coo-from-triplets 1 1 '((0 0 0.1))) (sparse-coo-from-triplets 1 1 '((0 0 -0.1))))))", "(equal? (sparse-coo-nnz (sparse-coo-drop-below 1e-14 (sparse-coo-add-impl (sparse-coo-from-triplets 1 1 '((0 0 0.1))) (sparse-coo-from-triplets 1 1 '((0 0 -0.1)))))) 0)", "hard", ["integration"]),
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
