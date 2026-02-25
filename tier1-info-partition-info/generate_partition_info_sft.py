#!/usr/bin/env python3
"""Generate Tier-1 partition-info SFT samples for lattice/info/partition-info.ss."""

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

SOURCE_MODULE = "lattice/info/partition-info.ss"
SOURCE_TEST = "lattice/info/test-partition-info.ss"

DEFS: Dict[str, str] = {
    "partition-sizes": """(define (partition-sizes labels)
  (let loop ([i 0] [counts hamt-empty])
    (if (= i (vector-length labels))
        (let* ([keys (hamt-keys counts)]
               [sorted-keys (sort-labels keys)])
          (map (lambda (k) (hamt-lookup-or k counts 0)) sorted-keys))
        (let ([lbl (vector-ref labels i)])
          (loop (+ i 1)
                (hamt-assoc lbl (+ 1 (hamt-lookup-or lbl counts 0)) counts))))))""",
    "partition-entropy": """(define (partition-entropy labels)
  (let* ([n (vector-length labels)]
         [sizes (partition-sizes labels)]
         [probs (map (lambda (s) (/ s n)) sizes)])
    (entropy probs)))""",
    "partition-mi": """(define (partition-mi labels-a labels-b)
  (let* ([n (vector-length labels-a)]
         [joint (build-contingency-table labels-a labels-b)]
         [joint-probs (map (lambda (row)
                             (map (lambda (c) (/ c n)) row))
                           joint)]
         [marginal-a (map (lambda (row) (fold-left + 0 row)) joint-probs)]
         [num-cols (if (null? joint-probs) 0 (length (car joint-probs)))]
         [marginal-b (if (= num-cols 0)
                         '()
                         (map (lambda (j)
                                (fold-left + 0
                                           (map (lambda (row) (list-ref row j))
                                                joint-probs)))
                              (iota num-cols)))])
    (mutual-information joint-probs marginal-a marginal-b)))""",
    "partition-nmi": """(define (partition-nmi labels-a labels-b)
  (let ([ha (partition-entropy labels-a)]
        [hb (partition-entropy labels-b)]
        [mi (partition-mi labels-a labels-b)])
    (if (or (<= ha 0) (<= hb 0))
        (if (and (<= ha 0) (<= hb 0)) 1.0 0.0)
        (/ mi (sqrt (* ha hb))))))""",
    "partition-vi": """(define (partition-vi labels-a labels-b)
  (let ([ha (partition-entropy labels-a)]
        [hb (partition-entropy labels-b)]
        [mi (partition-mi labels-a labels-b)])
    (max 0 (+ ha hb (* -2 mi)))))""",
    "partition-vi-normalized": """(define (partition-vi-normalized labels-a labels-b)
  (let* ([n (vector-length labels-a)]
         [vi (partition-vi labels-a labels-b)])
    (if (<= n 1)
        0
        (/ vi (log2 n)))))""",
    "unique-labels": """(define (unique-labels labels)
  (let loop ([i 0] [seen hamt-empty] [result '()])
    (if (= i (vector-length labels))
        (sort-labels result)
        (let ([lbl (vector-ref labels i)])
          (if (hamt-lookup lbl seen)
              (loop (+ i 1) seen result)
              (loop (+ i 1) (hamt-assoc lbl #t seen) (cons lbl result)))))))""",
    "label-index-map": """(define (label-index-map labels-list)
  (let loop ([ls labels-list] [i 0] [ht hamt-empty])
    (if (null? ls)
        ht
        (loop (cdr ls) (+ i 1) (hamt-assoc (car ls) i ht)))))""",
}

SUPPORT_DEFS: Dict[str, str] = {
    "approx=?": """(define (approx=? expected actual tol)
  (< (abs (- expected actual)) tol))""",
    "insert-sorted": """(define (insert-sorted x sorted)
  (cond
    [(null? sorted) (list x)]
    [(< x (car sorted)) (cons x sorted)]
    [else (cons (car sorted) (insert-sorted x (cdr sorted)))]))""",
    "sort-labels": """(define (sort-labels keys)
  (let loop ([remaining keys] [sorted '()])
    (if (null? remaining)
        sorted
        (loop (cdr remaining)
              (insert-sorted (car remaining) sorted)))))""",
    "build-contingency-table": """(define (build-contingency-table labels-a labels-b)
  (let* ([n (vector-length labels-a)]
         [labels-a-unique (unique-labels labels-a)]
         [labels-b-unique (unique-labels labels-b)]
         [na (length labels-a-unique)]
         [nb (length labels-b-unique)]
         [a-index (label-index-map labels-a-unique)]
         [b-index (label-index-map labels-b-unique)]
         [counts (make-vector (* na nb) 0)])
    (let loop ([i 0])
      (if (= i n)
          (let row-loop ([r 0] [rows '()])
            (if (= r na)
                (reverse rows)
                (let col-loop ([c 0] [cols '()])
                  (if (= c nb)
                      (row-loop (+ r 1) (cons (reverse cols) rows))
                      (col-loop (+ c 1)
                                (cons (vector-ref counts (+ (* r nb) c)) cols))))))
          (let ([ai (hamt-lookup-or (vector-ref labels-a i) a-index 0)]
                [bi (hamt-lookup-or (vector-ref labels-b i) b-index 0)])
            (let ([idx (+ (* ai nb) bi)])
              (vector-set! counts idx (+ 1 (vector-ref counts idx))))
            (loop (+ i 1)))))))""",
}

ALL_DEFS: Dict[str, str] = {**SUPPORT_DEFS, **DEFS}

DEPENDS: Dict[str, List[str]] = {
    "approx=?": [],
    "insert-sorted": [],
    "sort-labels": ["insert-sorted"],
    "label-index-map": [],
    "unique-labels": ["sort-labels"],
    "build-contingency-table": ["unique-labels", "label-index-map"],
    "partition-sizes": ["sort-labels"],
    "partition-entropy": ["partition-sizes"],
    "partition-mi": ["build-contingency-table"],
    "partition-nmi": ["partition-entropy", "partition-mi"],
    "partition-vi": ["partition-entropy", "partition-mi"],
    "partition-vi-normalized": ["partition-vi"],
}

FUNCTION_ORDER = [
    "partition-sizes",
    "partition-entropy",
    "partition-mi",
    "partition-nmi",
    "partition-vi",
    "partition-vi-normalized",
    "unique-labels",
    "label-index-map",
]

SUPPORT_ORDER = [
    "approx=?",
    "insert-sorted",
    "sort-labels",
    "build-contingency-table",
]

FUNCTION_SPECS = {
    "partition-sizes": "Count community sizes from a label vector and return counts ordered by sorted label value.",
    "partition-entropy": "Compute Shannon entropy of partition proportions in bits.",
    "partition-mi": "Compute mutual information between two partitions via contingency table probabilities.",
    "partition-nmi": "Compute normalized mutual information in [0,1] with explicit zero-entropy conventions.",
    "partition-vi": "Compute variation of information VI = H(A)+H(B)-2*I(A;B), clamped at zero.",
    "partition-vi-normalized": "Normalize VI by log2(n), returning 0 for n<=1.",
    "unique-labels": "Extract and sort unique labels from a label vector.",
    "label-index-map": "Build a HAMT mapping each label in a list to its positional index.",
}

SKELETONS = {
    "partition-sizes": """(define (partition-sizes labels)
  ;; TODO: count labels and return counts ordered by sorted label keys
  <TODO>)""",
    "partition-entropy": """(define (partition-entropy labels)
  ;; TODO: convert partition sizes to probabilities and compute entropy
  <TODO>)""",
    "partition-mi": """(define (partition-mi labels-a labels-b)
  ;; TODO: build contingency table and compute mutual information
  <TODO>)""",
    "partition-nmi": """(define (partition-nmi labels-a labels-b)
  ;; TODO: normalized mutual information with zero-entropy conventions
  <TODO>)""",
    "partition-vi": """(define (partition-vi labels-a labels-b)
  ;; TODO: variation of information with non-negativity clamp
  <TODO>)""",
    "partition-vi-normalized": """(define (partition-vi-normalized labels-a labels-b)
  ;; TODO: divide VI by log2(n), returning 0 when n<=1
  <TODO>)""",
    "unique-labels": """(define (unique-labels labels)
  ;; TODO: deduplicate labels and return sorted unique label list
  <TODO>)""",
    "label-index-map": """(define (label-index-map labels-list)
  ;; TODO: map each label to its zero-based index in labels-list
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "partition-sizes": "(and (equal? (partition-sizes '#(0 0 0 1 1 2)) '(3 2 1)) (equal? (partition-sizes '#(0 0 0 0)) '(4)))",
    "partition-entropy": "(and (< (abs (partition-entropy '#(0 0 0 0))) 1e-10) (< (abs (- (partition-entropy '#(0 0 1 1)) 1.0)) 1e-10))",
    "partition-mi": "(let* ([labels '#(0 0 1 1 2 2)] [mi (partition-mi labels labels)] [h (partition-entropy labels)] [a '#(0 0 1 1 0 0 1 1)] [b '#(0 1 0 1 0 1 0 1)]) (and (< (abs (- mi h)) 1e-10) (< (abs (partition-mi a b)) 1e-10)))",
    "partition-nmi": "(and (< (abs (- (partition-nmi '#(0 0 1 1 2 2) '#(0 0 1 1 2 2)) 1.0)) 1e-10) (< (abs (partition-nmi '#(0 0 1 1 0 0 1 1) '#(0 1 0 1 0 1 0 1))) 1e-10))",
    "partition-vi": "(let* ([a '#(0 0 1 1 2)] [b '#(0 1 1 2 2)] [vi-ab (partition-vi a b)] [vi-ba (partition-vi b a)]) (and (< (abs (partition-vi '#(0 0 1 1) '#(0 0 1 1))) 1e-10) (>= vi-ab -1e-10) (< (abs (- vi-ab vi-ba)) 1e-10)))",
    "partition-vi-normalized": "(let ([nvi (partition-vi-normalized '#(0 0 1 1 2) '#(0 1 1 2 2))]) (and (>= nvi -1e-10) (<= nvi (+ 1.0 1e-10)) (< (abs (partition-vi-normalized '#(0 0 1 1) '#(0 0 1 1))) 1e-10)))",
    "unique-labels": "(equal? (unique-labels '#(3 1 3 2 1 2)) '(1 2 3))",
    "label-index-map": "(let ([m (label-index-map '(10 20 30))]) (and (= (hamt-lookup-or 10 m -1) 0) (= (hamt-lookup-or 20 m -1) 1) (= (hamt-lookup-or 30 m -1) 2)))",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")

TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "partition-sizes": """def partition_sizes(labels):
    counts = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    return [counts[k] for k in sorted(counts.keys())]""",
    "partition-entropy": """def partition_entropy(labels):
    n = len(labels)
    sizes = partition_sizes(labels)
    probs = [s / n for s in sizes]
    return entropy(probs)""",
    "partition-mi": """def partition_mi(labels_a, labels_b):
    n = len(labels_a)
    joint = build_contingency_table(labels_a, labels_b)
    joint_probs = [[c / n for c in row] for row in joint]
    marginal_a = [sum(row) for row in joint_probs]
    num_cols = 0 if not joint_probs else len(joint_probs[0])
    marginal_b = [] if num_cols == 0 else [sum(row[j] for row in joint_probs) for j in range(num_cols)]
    return mutual_information(joint_probs, marginal_a, marginal_b)""",
    "partition-nmi": """def partition_nmi(labels_a, labels_b):
    ha = partition_entropy(labels_a)
    hb = partition_entropy(labels_b)
    mi = partition_mi(labels_a, labels_b)
    if ha <= 0 or hb <= 0:
        return 1.0 if ha <= 0 and hb <= 0 else 0.0
    return mi / sqrt(ha * hb)""",
    "partition-vi": """def partition_vi(labels_a, labels_b):
    ha = partition_entropy(labels_a)
    hb = partition_entropy(labels_b)
    mi = partition_mi(labels_a, labels_b)
    return max(0, ha + hb - 2 * mi)""",
    "partition-vi-normalized": """def partition_vi_normalized(labels_a, labels_b):
    n = len(labels_a)
    vi = partition_vi(labels_a, labels_b)
    if n <= 1:
        return 0
    return vi / log2(n)""",
    "unique-labels": """def unique_labels(labels):
    return sorted(set(labels))""",
    "label-index-map": """def label_index_map(labels_list):
    out = {}
    for i, label in enumerate(labels_list):
        out[label] = i
    return out""",
}

CHEZ_SNIPPETS = {
    "partition-sizes": """(define (partition-sizes0 labels)
  (let loop ([i 0] [counts hamt-empty])
    (if (= i (vector-length labels))
        (let* ([keys (hamt-keys counts)]
               [sorted-keys (sort-labels keys)])
          (map (lambda (k) (hamt-lookup-or k counts 0)) sorted-keys))
        (let ([lbl (vector-ref labels i)])
          (loop (+ i 1)
                (hamt-assoc lbl (+ 1 (hamt-lookup-or lbl counts 0)) counts))))))""",
    "partition-entropy": """(define (partition-entropy0 labels)
  (let* ([n (vector-length labels)]
         [sizes (partition-sizes labels)]
         [probs (map (lambda (s) (/ s n)) sizes)])
    (entropy probs)))""",
    "partition-mi": """(define (partition-mi0 labels-a labels-b)
  (let* ([n (vector-length labels-a)]
         [joint (build-contingency-table labels-a labels-b)]
         [joint-probs (map (lambda (row) (map (lambda (c) (/ c n)) row)) joint)]
         [marginal-a (map (lambda (row) (fold-left + 0 row)) joint-probs)]
         [num-cols (if (null? joint-probs) 0 (length (car joint-probs)))]
         [marginal-b (if (= num-cols 0)
                         '()
                         (map (lambda (j)
                                (fold-left + 0 (map (lambda (row) (list-ref row j)) joint-probs)))
                              (iota num-cols)))])
    (mutual-information joint-probs marginal-a marginal-b)))""",
    "partition-nmi": """(define (partition-nmi0 labels-a labels-b)
  (let ([ha (partition-entropy labels-a)]
        [hb (partition-entropy labels-b)]
        [mi (partition-mi labels-a labels-b)])
    (if (or (<= ha 0) (<= hb 0))
        (if (and (<= ha 0) (<= hb 0)) 1.0 0.0)
        (/ mi (sqrt (* ha hb))))))""",
    "partition-vi": """(define (partition-vi0 labels-a labels-b)
  (let ([ha (partition-entropy labels-a)]
        [hb (partition-entropy labels-b)]
        [mi (partition-mi labels-a labels-b)])
    (max 0 (+ ha hb (* -2 mi)))))""",
    "partition-vi-normalized": """(define (partition-vi-normalized0 labels-a labels-b)
  (let* ([n (vector-length labels-a)]
         [vi (partition-vi labels-a labels-b)])
    (if (<= n 1)
        0
        (/ vi (log2 n)))))""",
    "unique-labels": """(define (unique-labels0 labels)
  (let loop ([i 0] [seen hamt-empty] [result '()])
    (if (= i (vector-length labels))
        (sort-labels result)
        (let ([lbl (vector-ref labels i)])
          (if (hamt-lookup lbl seen)
              (loop (+ i 1) seen result)
              (loop (+ i 1) (hamt-assoc lbl #t seen) (cons lbl result)))))))""",
    "label-index-map": """(define (label-index-map0 labels-list)
  (let loop ([ls labels-list] [i 0] [ht hamt-empty])
    (if (null? ls)
        ht
        (loop (cdr ls) (+ i 1) (hamt-assoc (car ls) i ht)))))""",
}

BUGGY_CASES = [
    {
        "fn": "partition-sizes",
        "buggy": """(define (partition-sizes labels)
  (let loop ([i 0] [counts hamt-empty])
    (if (= i (vector-length labels))
        (hamt-values counts)
        (let ([lbl (vector-ref labels i)])
          (loop (+ i 1)
                (hamt-assoc lbl (+ 1 (hamt-lookup-or lbl counts 0)) counts))))))""",
        "note": "Output must be deterministic and sorted by label keys, not unsorted HAMT values.",
    },
    {
        "fn": "partition-sizes",
        "buggy": """(define (partition-sizes labels)
  (let loop ([i 0] [counts hamt-empty])
    (if (= i (vector-length labels))
        (let* ([keys (hamt-keys counts)]
               [sorted-keys (sort-labels keys)])
          (map (lambda (k) (hamt-lookup-or k counts 1)) sorted-keys))
        (let ([lbl (vector-ref labels i)])
          (loop (+ i 1)
                (hamt-assoc lbl (+ 1 (hamt-lookup-or lbl counts 0)) counts))))))""",
        "note": "Missing labels should default to 0 during lookup, not 1.",
    },
    {
        "fn": "partition-entropy",
        "buggy": """(define (partition-entropy labels)
  (let* ([sizes (partition-sizes labels)])
    (entropy sizes)))""",
        "note": "Entropy expects probabilities, not raw counts.",
    },
    {
        "fn": "partition-entropy",
        "buggy": """(define (partition-entropy labels)
  (let* ([n (vector-length labels)]
         [sizes (partition-sizes labels)]
         [probs (map (lambda (s) (/ n s)) sizes)])
    (entropy probs)))""",
        "note": "Probability conversion should be s/n, not n/s.",
    },
    {
        "fn": "partition-mi",
        "buggy": """(define (partition-mi labels-a labels-b)
  (let* ([n (vector-length labels-a)]
         [joint (build-contingency-table labels-a labels-b)]
         [joint-probs (map (lambda (row) (map (lambda (c) (/ c n)) row)) joint)]
         [marginal-a (map (lambda (row) (fold-left + 0 row)) joint-probs)]
         [num-cols (if (null? joint-probs) 0 (length (car joint-probs)))]
         [marginal-b (if (= num-cols 0)
                         '()
                         (map (lambda (j)
                                (fold-left + 0 (map (lambda (row) (list-ref row j)) joint-probs)))
                              (iota num-cols)))])
    (+ (entropy marginal-a) (entropy marginal-b) (joint-entropy joint-probs))))""",
        "note": "Mutual information subtracts joint entropy; it does not add it.",
    },
    {
        "fn": "partition-mi",
        "buggy": """(define (partition-mi labels-a labels-b)
  (let* ([n (vector-length labels-a)]
         [joint (build-contingency-table labels-a labels-b)]
         [joint-probs (map (lambda (row) (map (lambda (c) (/ c n)) row)) joint)]
         [marginal-a (map (lambda (row) (fold-left + 0 row)) joint-probs)]
         [num-cols (if (null? joint-probs) 0 (length (car joint-probs)))]
         [marginal-b (if (= num-cols 0)
                         '()
                         (map (lambda (j)
                                (fold-left + 0 (map (lambda (row) (list-ref row j)) joint-probs)))
                              (iota num-cols)))])
    (mutual-information joint-probs marginal-b marginal-a)))""",
        "note": "Argument order to mutual-information should be (joint, marginal-a, marginal-b).",
    },
    {
        "fn": "partition-nmi",
        "buggy": """(define (partition-nmi labels-a labels-b)
  (let ([ha (partition-entropy labels-a)]
        [hb (partition-entropy labels-b)]
        [mi (partition-mi labels-a labels-b)])
    (if (or (<= ha 0) (<= hb 0))
        0.0
        (/ mi (+ ha hb)))))""",
        "note": "NMI denominator should be sqrt(Ha*Hb) with special zero-entropy convention.",
    },
    {
        "fn": "partition-nmi",
        "buggy": """(define (partition-nmi labels-a labels-b)
  (let ([ha (partition-entropy labels-a)]
        [hb (partition-entropy labels-b)]
        [mi (partition-mi labels-a labels-b)])
    (if (or (<= ha 0) (<= hb 0))
        (if (and (<= ha 0) (<= hb 0)) 0.0 1.0)
        (/ mi (sqrt (* ha hb))))))""",
        "note": "Zero-entropy convention is inverted: both-zero => 1.0, one-zero => 0.0.",
    },
    {
        "fn": "partition-vi",
        "buggy": """(define (partition-vi labels-a labels-b)
  (let ([ha (partition-entropy labels-a)]
        [hb (partition-entropy labels-b)]
        [mi (partition-mi labels-a labels-b)])
    (+ ha hb (* 2 mi))))""",
        "note": "VI subtracts 2*MI, not adds it.",
    },
    {
        "fn": "partition-vi",
        "buggy": """(define (partition-vi labels-a labels-b)
  (let ([ha (partition-entropy labels-a)]
        [hb (partition-entropy labels-b)]
        [mi (partition-mi labels-a labels-b)])
    (+ ha hb (* -2 mi))))""",
        "note": "VI should be clamped to non-negative values with max 0.",
    },
    {
        "fn": "partition-vi-normalized",
        "buggy": """(define (partition-vi-normalized labels-a labels-b)
  (let* ([n (vector-length labels-a)]
         [vi (partition-vi labels-a labels-b)])
    (if (<= n 1)
        0
        (/ vi n))))""",
        "note": "Normalization divisor should be log2(n), not n.",
    },
    {
        "fn": "partition-vi-normalized",
        "buggy": """(define (partition-vi-normalized labels-a labels-b)
  (let* ([n (vector-length labels-a)]
         [vi (partition-vi labels-a labels-b)])
    (if (< n 1)
        0
        (/ vi (log2 n)))))""",
        "note": "Guard should be n<=1 to avoid division by log2(1)=0.",
    },
    {
        "fn": "unique-labels",
        "buggy": """(define (unique-labels labels)
  (let loop ([i 0] [seen hamt-empty] [result '()])
    (if (= i (vector-length labels))
        result
        (let ([lbl (vector-ref labels i)])
          (if (hamt-lookup lbl seen)
              (loop (+ i 1) seen result)
              (loop (+ i 1) (hamt-assoc lbl #t seen) (cons lbl result)))))))""",
        "note": "Unique labels should be returned in sorted order.",
    },
    {
        "fn": "unique-labels",
        "buggy": """(define (unique-labels labels)
  (let loop ([i 0] [seen hamt-empty] [result '()])
    (if (= i (vector-length labels))
        (sort-labels result)
        (let ([lbl (vector-ref labels i)])
          (loop (+ i 1) (hamt-assoc lbl #t seen) (cons lbl result))))))""",
        "note": "Function must skip labels already seen; this version duplicates them.",
    },
    {
        "fn": "label-index-map",
        "buggy": """(define (label-index-map labels-list)
  (let loop ([ls labels-list] [i 1] [ht hamt-empty])
    (if (null? ls)
        ht
        (loop (cdr ls) (+ i 1) (hamt-assoc (car ls) i ht)))))""",
        "note": "Indices should start at 0, not 1.",
    },
    {
        "fn": "label-index-map",
        "buggy": """(define (label-index-map labels-list)
  (let loop ([ls labels-list] [i 0] [ht hamt-empty])
    (if (null? ls)
        ht
        (loop (cdr ls) (+ i 1) (hamt-assoc i (car ls) ht)))))""",
        "note": "Mapping direction is reversed; keys must be labels and values indices.",
    },
]

DIFFICULTY = {
    "partition-sizes": "easy",
    "partition-entropy": "easy",
    "partition-mi": "hard",
    "partition-nmi": "hard",
    "partition-vi": "medium",
    "partition-vi-normalized": "medium",
    "unique-labels": "easy",
    "label-index-map": "easy",
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
    sid = f"info_partition_{family}_{family_counter[family]:03d}"
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


def verify_refs(verify_expr: str) -> List[str]:
    tokens = set(TOKEN_RE.findall(verify_expr))
    names = FUNCTION_ORDER + SUPPORT_ORDER
    return [name for name in names if name in tokens]


def dependency_closure(roots: List[str]) -> List[str]:
    ordered: List[str] = []
    seen: Set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        for dep in DEPENDS.get(name, []):
            visit(dep)
        if name in ALL_DEFS:
            ordered.append(name)

    for root in roots:
        visit(root)

    return ordered


def build_verify(verify_check: str, roots: List[str] | None = None) -> str:
    wanted: List[str] = []
    for root in roots or []:
        if root not in wanted:
            wanted.append(root)
    for ref in verify_refs(verify_check):
        if ref not in wanted:
            wanted.append(ref)

    defs_needed = dependency_closure(wanted)
    parts = [ALL_DEFS[name] for name in defs_needed] + [verify_check.strip()]
    body = "\n  ".join(parts)
    return f"(let ()\n  {body})"


def def_verify(fn: str) -> str:
    return build_verify(VERIFY_BY_FUNCTION[fn], [fn])


# -----------------------------------------------------------------------------
# Family 1: spec_to_code (16)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Implement this partition-information utility in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "info", "partition-info", "spec-to-code", fn],
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
        tags=["tier1", "info", "partition-info", "skeleton-completion", fn],
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
        tags=["tier1", "info", "partition-info", "python-to-scheme", fn],
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
        tags=["tier1", "info", "partition-info", "chez-to-fold", fn],
    )


# -----------------------------------------------------------------------------
# Family 3: bugfix (16)
# -----------------------------------------------------------------------------
for case in BUGGY_CASES:
    fn = case["fn"]
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
        tags=["tier1", "info", "partition-info", "bugfix", fn],
    )


# -----------------------------------------------------------------------------
# Family 4: composition/use (32)
# -----------------------------------------------------------------------------
def add_composition(
    source_function: str,
    prompt: str,
    ground_truth: str,
    verify_check: str,
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
        verify_expr=build_verify(verify_check, [source_function]),
        tags=["tier1", "info", "partition-info", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # partition-sizes
    {
        "fn": "partition-sizes",
        "prompt": "Compute partition sizes for labels '#(0 0 0 1 1 2).",
        "gt": "(partition-sizes '#(0 0 0 1 1 2))",
        "verify": "(equal? (partition-sizes '#(0 0 0 1 1 2)) '(3 2 1))",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "partition-sizes",
        "prompt": "Compute partition sizes for single-community labels '#(0 0 0 0).",
        "gt": "(partition-sizes '#(0 0 0 0))",
        "verify": "(equal? (partition-sizes '#(0 0 0 0)) '(4))",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },
    {
        "fn": "partition-sizes",
        "prompt": "Return whether partition-size output is sorted by label for labels '#(2 2 1 1 1).",
        "gt": "(equal? (partition-sizes '#(2 2 1 1 1)) '(3 2))",
        "verify": "(equal? (equal? (partition-sizes '#(2 2 1 1 1)) '(3 2)) #t)",
        "difficulty": "easy",
        "tags": ["ordering"],
    },
    {
        "fn": "partition-sizes",
        "prompt": "Return whether partition sizes preserve total count for labels '#(0 1 1 2 2 2).",
        "gt": "(= (fold-left + 0 (partition-sizes '#(0 1 1 2 2 2))) 6)",
        "verify": "(equal? (= (fold-left + 0 (partition-sizes '#(0 1 1 2 2 2))) 6) #t)",
        "difficulty": "easy",
        "tags": ["property"],
    },

    # partition-entropy
    {
        "fn": "partition-entropy",
        "prompt": "Compute partition entropy for uniform binary labels '#(0 0 1 1).",
        "gt": "(partition-entropy '#(0 0 1 1))",
        "verify": "(approx=? 1.0 (partition-entropy '#(0 0 1 1)) 1e-10)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "partition-entropy",
        "prompt": "Compute partition entropy for single-community labels '#(0 0 0 0).",
        "gt": "(partition-entropy '#(0 0 0 0))",
        "verify": "(< (abs (partition-entropy '#(0 0 0 0))) 1e-10)",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },
    {
        "fn": "partition-entropy",
        "prompt": "Return whether a uniform four-way partition has entropy 2 bits.",
        "gt": "(partition-entropy '#(0 1 2 3))",
        "verify": "(approx=? 2.0 (partition-entropy '#(0 1 2 3)) 1e-10)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "partition-entropy",
        "prompt": "Return whether non-uniform partition entropy lies strictly between 0 and 1 bit for '#(0 0 0 1).",
        "gt": "(let ([h (partition-entropy '#(0 0 0 1))]) (and (> h 0) (< h 1.0)))",
        "verify": "(equal? (let ([h (partition-entropy '#(0 0 0 1))]) (and (> h 0) (< h 1.0))) #t)",
        "difficulty": "medium",
        "tags": ["range"],
    },

    # partition-mi
    {
        "fn": "partition-mi",
        "prompt": "Compute MI for identical partitions '#(0 0 1 1 2 2) and '#(0 0 1 1 2 2).",
        "gt": "(partition-mi '#(0 0 1 1 2 2) '#(0 0 1 1 2 2))",
        "verify": "(let* ([labels '#(0 0 1 1 2 2)] [mi (partition-mi labels labels)] [h (partition-entropy labels)]) (< (abs (- mi h)) 1e-10))",
        "difficulty": "hard",
        "tags": ["identity"],
    },
    {
        "fn": "partition-mi",
        "prompt": "Compute MI for independent-style partitions '#(0 0 1 1 0 0 1 1) and '#(0 1 0 1 0 1 0 1).",
        "gt": "(partition-mi '#(0 0 1 1 0 0 1 1) '#(0 1 0 1 0 1 0 1))",
        "verify": "(< (abs (partition-mi '#(0 0 1 1 0 0 1 1) '#(0 1 0 1 0 1 0 1))) 1e-10)",
        "difficulty": "hard",
        "tags": ["independence"],
    },
    {
        "fn": "partition-mi",
        "prompt": "Return whether MI is non-negative for partitions '#(0 0 1 1 2) and '#(0 1 1 2 2).",
        "gt": "(>= (partition-mi '#(0 0 1 1 2) '#(0 1 1 2 2)) -1e-10)",
        "verify": "(equal? (>= (partition-mi '#(0 0 1 1 2) '#(0 1 1 2 2)) -1e-10) #t)",
        "difficulty": "medium",
        "tags": ["property"],
    },
    {
        "fn": "partition-mi",
        "prompt": "Return whether MI is symmetric for two partitions.",
        "gt": "(let ([a '#(0 0 1 1 2)] [b '#(0 1 1 2 2)]) (approx=? (partition-mi a b) (partition-mi b a) 1e-10))",
        "verify": "(equal? (let ([a '#(0 0 1 1 2)] [b '#(0 1 1 2 2)]) (approx=? (partition-mi a b) (partition-mi b a) 1e-10)) #t)",
        "difficulty": "hard",
        "tags": ["symmetry"],
    },

    # partition-nmi
    {
        "fn": "partition-nmi",
        "prompt": "Compute NMI for identical partitions '#(0 0 1 1 2 2).",
        "gt": "(partition-nmi '#(0 0 1 1 2 2) '#(0 0 1 1 2 2))",
        "verify": "(approx=? 1.0 (partition-nmi '#(0 0 1 1 2 2) '#(0 0 1 1 2 2)) 1e-10)",
        "difficulty": "easy",
        "tags": ["identity"],
    },
    {
        "fn": "partition-nmi",
        "prompt": "Compute NMI for independent-style partitions '#(0 0 1 1 0 0 1 1) and '#(0 1 0 1 0 1 0 1).",
        "gt": "(partition-nmi '#(0 0 1 1 0 0 1 1) '#(0 1 0 1 0 1 0 1))",
        "verify": "(< (abs (partition-nmi '#(0 0 1 1 0 0 1 1) '#(0 1 0 1 0 1 0 1))) 1e-10)",
        "difficulty": "hard",
        "tags": ["independence"],
    },
    {
        "fn": "partition-nmi",
        "prompt": "Return whether NMI is bounded in [0,1] for partitions '#(0 0 0 1 1 2 2 2) and '#(0 1 1 1 2 2 0 0).",
        "gt": "(let ([nmi (partition-nmi '#(0 0 0 1 1 2 2 2) '#(0 1 1 1 2 2 0 0))]) (and (>= nmi -1e-10) (<= nmi (+ 1.0 1e-10))))",
        "verify": "(equal? (let ([nmi (partition-nmi '#(0 0 0 1 1 2 2 2) '#(0 1 1 1 2 2 0 0))]) (and (>= nmi -1e-10) (<= nmi (+ 1.0 1e-10)))) #t)",
        "difficulty": "hard",
        "tags": ["range"],
    },
    {
        "fn": "partition-nmi",
        "prompt": "Return NMI when both partitions are single-community labels '#(0 0 0 0).",
        "gt": "(partition-nmi '#(0 0 0 0) '#(0 0 0 0))",
        "verify": "(approx=? 1.0 (partition-nmi '#(0 0 0 0) '#(0 0 0 0)) 1e-10)",
        "difficulty": "medium",
        "tags": ["zero-entropy"],
    },

    # partition-vi
    {
        "fn": "partition-vi",
        "prompt": "Compute VI for identical partitions '#(0 0 1 1).",
        "gt": "(partition-vi '#(0 0 1 1) '#(0 0 1 1))",
        "verify": "(< (abs (partition-vi '#(0 0 1 1) '#(0 0 1 1))) 1e-10)",
        "difficulty": "easy",
        "tags": ["identity"],
    },
    {
        "fn": "partition-vi",
        "prompt": "Return whether VI is symmetric for partitions '#(0 0 1 1 2) and '#(0 1 1 2 2).",
        "gt": "(let ([a '#(0 0 1 1 2)] [b '#(0 1 1 2 2)]) (approx=? (partition-vi a b) (partition-vi b a) 1e-10))",
        "verify": "(equal? (let ([a '#(0 0 1 1 2)] [b '#(0 1 1 2 2)]) (approx=? (partition-vi a b) (partition-vi b a) 1e-10)) #t)",
        "difficulty": "medium",
        "tags": ["symmetry"],
    },
    {
        "fn": "partition-vi",
        "prompt": "Return whether VI is non-negative for partitions '#(0 0 1 1 2) and '#(0 1 1 2 2).",
        "gt": "(>= (partition-vi '#(0 0 1 1 2) '#(0 1 1 2 2)) -1e-10)",
        "verify": "(equal? (>= (partition-vi '#(0 0 1 1 2) '#(0 1 1 2 2)) -1e-10) #t)",
        "difficulty": "medium",
        "tags": ["property"],
    },
    {
        "fn": "partition-vi",
        "prompt": "Return whether VI and NMI move in opposite directions between identical and independent partitions.",
        "gt": "(let ([id-nmi (partition-nmi '#(0 0 1 1) '#(0 0 1 1))] [id-vi (partition-vi '#(0 0 1 1) '#(0 0 1 1))] [ind-nmi (partition-nmi '#(0 0 1 1 0 0 1 1) '#(0 1 0 1 0 1 0 1))] [ind-vi (partition-vi '#(0 0 1 1 0 0 1 1) '#(0 1 0 1 0 1 0 1))]) (and (> id-nmi ind-nmi) (< id-vi ind-vi)))",
        "verify": "(equal? (let ([id-nmi (partition-nmi '#(0 0 1 1) '#(0 0 1 1))] [id-vi (partition-vi '#(0 0 1 1) '#(0 0 1 1))] [ind-nmi (partition-nmi '#(0 0 1 1 0 0 1 1) '#(0 1 0 1 0 1 0 1))] [ind-vi (partition-vi '#(0 0 1 1 0 0 1 1) '#(0 1 0 1 0 1 0 1))]) (and (> id-nmi ind-nmi) (< id-vi ind-vi))) #t)",
        "difficulty": "hard",
        "tags": ["integration"],
    },

    # partition-vi-normalized
    {
        "fn": "partition-vi-normalized",
        "prompt": "Compute normalized VI for partitions '#(0 0 1 1 2) and '#(0 1 1 2 2).",
        "gt": "(partition-vi-normalized '#(0 0 1 1 2) '#(0 1 1 2 2))",
        "verify": "(let ([nvi (partition-vi-normalized '#(0 0 1 1 2) '#(0 1 1 2 2))]) (and (>= nvi -1e-10) (<= nvi (+ 1.0 1e-10))))",
        "difficulty": "medium",
        "tags": ["direct"],
    },
    {
        "fn": "partition-vi-normalized",
        "prompt": "Compute normalized VI for identical partitions '#(0 0 1 1).",
        "gt": "(partition-vi-normalized '#(0 0 1 1) '#(0 0 1 1))",
        "verify": "(< (abs (partition-vi-normalized '#(0 0 1 1) '#(0 0 1 1))) 1e-10)",
        "difficulty": "easy",
        "tags": ["identity"],
    },
    {
        "fn": "partition-vi-normalized",
        "prompt": "Return normalized VI when n<=1 using '#(0) and '#(0).",
        "gt": "(partition-vi-normalized '#(0) '#(0))",
        "verify": "(= (partition-vi-normalized '#(0) '#(0)) 0)",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },
    {
        "fn": "partition-vi-normalized",
        "prompt": "Return whether normalized VI preserves ordering against raw VI for the same n.",
        "gt": "(let ([a '#(0 0 1 1 2)] [b '#(0 1 1 2 2)] [c '#(0 0 0 1 1)]) (and (> (partition-vi a b) (partition-vi a c)) (> (partition-vi-normalized a b) (partition-vi-normalized a c))))",
        "verify": "(equal? (let ([a '#(0 0 1 1 2)] [b '#(0 1 1 2 2)] [c '#(0 0 0 1 1)]) (and (> (partition-vi a b) (partition-vi a c)) (> (partition-vi-normalized a b) (partition-vi-normalized a c)))) #t)",
        "difficulty": "hard",
        "tags": ["ordering"],
    },

    # unique-labels
    {
        "fn": "unique-labels",
        "prompt": "Compute unique sorted labels from '#(3 1 3 2 1 2).",
        "gt": "(unique-labels '#(3 1 3 2 1 2))",
        "verify": "(equal? (unique-labels '#(3 1 3 2 1 2)) '(1 2 3))",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "unique-labels",
        "prompt": "Compute unique labels from single-valued vector '#(5 5 5).",
        "gt": "(unique-labels '#(5 5 5))",
        "verify": "(equal? (unique-labels '#(5 5 5)) '(5))",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },
    {
        "fn": "unique-labels",
        "prompt": "Return whether unique-label output is sorted ascending for '#(9 2 5 2 9 1).",
        "gt": "(equal? (unique-labels '#(9 2 5 2 9 1)) '(1 2 5 9))",
        "verify": "(equal? (equal? (unique-labels '#(9 2 5 2 9 1)) '(1 2 5 9)) #t)",
        "difficulty": "easy",
        "tags": ["ordering"],
    },
    {
        "fn": "unique-labels",
        "prompt": "Return whether unique-label count for '#(1 1 2 2 3 3 4) is 4.",
        "gt": "(= (length (unique-labels '#(1 1 2 2 3 3 4))) 4)",
        "verify": "(equal? (= (length (unique-labels '#(1 1 2 2 3 3 4))) 4) #t)",
        "difficulty": "easy",
        "tags": ["count"],
    },

    # label-index-map
    {
        "fn": "label-index-map",
        "prompt": "Build label index map for '(10 20 30) and return index for label 20.",
        "gt": "(let ([m (label-index-map '(10 20 30))]) (hamt-lookup-or 20 m -1))",
        "verify": "(= (let ([m (label-index-map '(10 20 30))]) (hamt-lookup-or 20 m -1)) 1)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "label-index-map",
        "prompt": "Build label index map for '(7) and return index for label 7.",
        "gt": "(let ([m (label-index-map '(7))]) (hamt-lookup-or 7 m -1))",
        "verify": "(= (let ([m (label-index-map '(7))]) (hamt-lookup-or 7 m -1)) 0)",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },
    {
        "fn": "label-index-map",
        "prompt": "Return whether label-index-map assigns increasing indices following list order for '(3 9 4).",
        "gt": "(let ([m (label-index-map '(3 9 4))]) (and (= (hamt-lookup-or 3 m -1) 0) (= (hamt-lookup-or 9 m -1) 1) (= (hamt-lookup-or 4 m -1) 2)))",
        "verify": "(equal? (let ([m (label-index-map '(3 9 4))]) (and (= (hamt-lookup-or 3 m -1) 0) (= (hamt-lookup-or 9 m -1) 1) (= (hamt-lookup-or 4 m -1) 2))) #t)",
        "difficulty": "easy",
        "tags": ["ordering"],
    },
    {
        "fn": "label-index-map",
        "prompt": "Return whether composing unique-labels with label-index-map gives index range 0..k-1.",
        "gt": "(let* ([labels (unique-labels '#(5 3 5 2 3 1))] [m (label-index-map labels)]) (and (= (hamt-lookup-or 1 m -1) 0) (= (hamt-lookup-or 2 m -1) 1) (= (hamt-lookup-or 3 m -1) 2) (= (hamt-lookup-or 5 m -1) 3)))",
        "verify": "(equal? (let* ([labels (unique-labels '#(5 3 5 2 3 1))] [m (label-index-map labels)]) (and (= (hamt-lookup-or 1 m -1) 0) (= (hamt-lookup-or 2 m -1) 1) (= (hamt-lookup-or 3 m -1) 2) (= (hamt-lookup-or 5 m -1) 3))) #t)",
        "difficulty": "medium",
        "tags": ["integration"],
    },
]

for case in composition_cases:
    add_composition(
        source_function=str(case["fn"]),
        prompt=str(case["prompt"]),
        ground_truth=str(case["gt"]),
        verify_check=str(case["verify"]),
        difficulty=str(case["difficulty"]),
        extra_tags=list(case["tags"]),
    )

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
