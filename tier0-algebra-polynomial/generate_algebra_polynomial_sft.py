#!/usr/bin/env python3
"""Generate Tier-0 algebra polynomial SFT samples for lattice/algebra/polynomial.ss."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

OUT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DATA_ROOT.parent
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from sft_split_utils import compute_leakage_aware_eval_ids

ALL_PATH = OUT_DIR / "all.jsonl"
TRAIN_PATH = OUT_DIR / "train.jsonl"
EVAL_PATH = OUT_DIR / "eval.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.json"
PRE_DIVERSIFY_PATH = OUT_DIR / ".pre_diversify.jsonl"
SFT_GENERATOR_PATH = REPO_ROOT / "user" / "sft" / "generate.ss"

SOURCE_MODULE = "lattice/algebra/polynomial.ss"
SOURCE_TEST = "lattice/algebra/test-polynomial.ss"

# =============================================================================
# DEFS: Exact canonical implementations from polynomial.ss
# =============================================================================

DEFS: Dict[str, str] = {
    # ==== Predicates and Accessors ====
    "polynomial?": """(define (polynomial? p)
  (doc 'export #t)
  (and (pair? p)
       (eq? (car p) 'polynomial)))""",

    "poly-field": """(define (poly-field p)
  (doc 'export #t)
  (cadr p))""",

    "poly-coeffs": """(define (poly-coeffs p)
  (doc 'export #t)
  (caddr p))""",

    "poly-degree": """(define (poly-degree p)
  (doc 'export #t)
  (let ([coeffs (poly-coeffs p)])
    (- (length coeffs) 1)))""",

    "poly-leading-coeff": """(define (poly-leading-coeff p)
  (doc 'export #t)
  (let ([coeffs (poly-coeffs p)])
    (if (null? coeffs)
        (field-zero (poly-field p))
        (list-ref coeffs (- (length coeffs) 1)))))""",

    "poly-coeff-at": """(define (poly-coeff-at p k)
  (doc 'export #t)
  (let ([coeffs (poly-coeffs p)]
        [F (poly-field p)])
    (if (>= k (length coeffs))
        (field-zero F)
        (list-ref coeffs k))))""",

    "poly-zero?": """(define (poly-zero? p)
  (doc 'export #t)
  (let ([F (poly-field p)]
        [coeffs (poly-coeffs p)])
    (and (= (length coeffs) 1)
         ((field-equal-fn F) (car coeffs) (field-zero F)))))""",

    # ==== Constructors ====
    "make-polynomial": """(define (make-polynomial field coeffs)
  (doc 'export #t)
  (doc 'type '(-> Field (List Coeff) Polynomial))
  (doc 'description "Create a polynomial over field F with given coefficients")
  (doc 'description "Automatically normalizes (strips trailing zeros)")
  (list 'polynomial field (poly-normalize-coeffs field coeffs)))""",

    "poly-zero-over": """(define (poly-zero-over field)
  (doc 'export #t)
  (make-polynomial field (list (field-zero field))))""",

    "poly-one-over": """(define (poly-one-over field)
  (doc 'export #t)
  (make-polynomial field (list (field-one field))))""",

    "poly-constant": """(define (poly-constant field c)
  (doc 'export #t)
  (make-polynomial field (list c)))""",

    "poly-monomial": """(define (poly-monomial field coeff degree)
  (doc 'export #t)
  (let ([zero (field-zero field)])
    (if ((field-equal-fn field) coeff zero)
        (poly-zero-over field)
        (make-polynomial field
          (append (make-list degree zero) (list coeff))))))""",

    "poly-x": """(define (poly-x field)
  (doc 'export #t)
  (poly-monomial field (field-one field) 1))""",

    # ==== Arithmetic ====
    "poly-add": """(define (poly-add p1 p2)
  (doc 'export #t)
  (let* ([F (poly-field p1)]
         [c1 (poly-coeffs p1)]
         [c2 (poly-coeffs p2)]
         [add (field-add-op F)])
    (make-polynomial F (poly-add-coeffs add c1 c2 (field-zero F)))))""",

    "poly-neg": """(define (poly-neg p)
  (doc 'export #t)
  (let* ([F (poly-field p)]
         [neg (field-neg-fn F)])
    (make-polynomial F (map neg (poly-coeffs p)))))""",

    "poly-sub": """(define (poly-sub p1 p2)
  (doc 'export #t)
  (poly-add p1 (poly-neg p2)))""",

    "poly-scale": """(define (poly-scale p c)
  (doc 'export #t)
  (let* ([F (poly-field p)]
         [mul (field-mul-op F)])
    (make-polynomial F (map (lambda (a) (mul c a)) (poly-coeffs p)))))""",

    "poly-mul": """(define (poly-mul p1 p2)
  (doc 'export #t)
  (let* ([F (poly-field p1)]
         [c1 (poly-coeffs p1)]
         [c2 (poly-coeffs p2)]
         [zero (field-zero F)]
         [add (field-add-op F)]
         [mul (field-mul-op F)]
         [n1 (length c1)]
         [n2 (length c2)]
         [n (+ n1 n2 -1)])
    (if (or (poly-zero? p1) (poly-zero? p2))
        (poly-zero-over F)
        (make-polynomial F
          (let loop ([k 0] [result '()])
            (if (= k n)
                (reverse result)
                (loop (+ k 1)
                      (cons (poly-mul-coeff-at add mul c1 c2 k n1 n2 zero)
                            result))))))))""",

    "poly-power": """(define (poly-power p n)
  (doc 'export #t)
  (cond
    [(= n 0) (poly-one-over (poly-field p))]
    [(= n 1) p]
    [(even? n)
     (let ([half (poly-power p (/ n 2))])
       (poly-mul half half))]
    [else
     (poly-mul p (poly-power p (- n 1)))]))""",

    # ==== Equality ====
    "poly-equal?": """(define (poly-equal? p1 p2)
  (doc 'export #t)
  (let* ([F (poly-field p1)]
         [eq-fn (field-equal-fn F)]
         [c1 (poly-coeffs p1)]
         [c2 (poly-coeffs p2)])
    (and (= (length c1) (length c2))
         (let loop ([l1 c1] [l2 c2])
           (or (null? l1)
               (and (eq-fn (car l1) (car l2))
                    (loop (cdr l1) (cdr l2))))))))""",

    # ==== Division ====
    "poly-divmod": """(define (poly-divmod p1 p2)
  (doc 'export #t)
  (let* ([F (poly-field p1)]
         [d2 (poly-degree p2)])
    (if (poly-zero? p2)
        (error 'poly-divmod "division by zero polynomial")
        (poly-divmod-loop p1 p2 (poly-zero-over F) F))))""",

    "poly-div": """(define (poly-div p1 p2)
  (doc 'export #t)
  (car (poly-divmod p1 p2)))""",

    "poly-mod": """(define (poly-mod p1 p2)
  (doc 'export #t)
  (cdr (poly-divmod p1 p2)))""",

    "poly-divides?": """(define (poly-divides? p1 p2)
  (doc 'export #t)
  (poly-zero? (poly-mod p2 p1)))""",

    # ==== GCD and Extended Euclidean ====
    "poly-gcd": """(define (poly-gcd p1 p2)
  (doc 'export #t)
  (let ([F (poly-field p1)])
    (if (poly-zero? p2)
        (poly-make-monic p1)
        (poly-gcd p2 (poly-mod p1 p2)))))""",

    "poly-make-monic": """(define (poly-make-monic p)
  (doc 'export #t)
  (if (poly-zero? p)
      p
      (let* ([F (poly-field p)]
             [lc (poly-leading-coeff p)]
             [inv-lc (field-inv F lc)])
        (poly-scale p inv-lc))))""",

    "poly-extended-gcd": """(define (poly-extended-gcd p1 p2)
  (doc 'export #t)
  (let ([F (poly-field p1)])
    (poly-ext-gcd-loop p1 p2
                       (poly-one-over F) (poly-zero-over F)
                       (poly-zero-over F) (poly-one-over F)
                       F)))""",

    "poly-lcm": """(define (poly-lcm p1 p2)
  (doc 'export #t)
  (if (or (poly-zero? p1) (poly-zero? p2))
      (poly-zero-over (poly-field p1))
      (poly-div (poly-mul p1 p2) (poly-gcd p1 p2))))""",

    # ==== Evaluation ====
    "poly-eval": """(define (poly-eval p x)
  (doc 'export #t)
  (let* ([F (poly-field p)]
         [coeffs (reverse (poly-coeffs p))]
         [add (field-add-op F)]
         [mul (field-mul-op F)])
    (if (null? coeffs)
        (field-zero F)
        (let loop ([cs (cdr coeffs)] [acc (car coeffs)])
          (if (null? cs)
              acc
              (loop (cdr cs) (add (mul acc x) (car cs))))))))""",

    # ==== Derivative ====
    "poly-derivative": """(define (poly-derivative p)
  (doc 'export #t)
  (let* ([F (poly-field p)]
         [coeffs (poly-coeffs p)]
         [add (field-add-op F)])
    (if (<= (length coeffs) 1)
        (poly-zero-over F)
        (make-polynomial F
          (let loop ([cs (cdr coeffs)] [k 1] [result '()])
            (if (null? cs)
                (reverse result)
                (loop (cdr cs) (+ k 1)
                      (cons (poly-scalar-mul-int F (car cs) k) result))))))))""",

    # ==== Factorization ====
    "poly-square-free": """(define (poly-square-free p)
  (doc 'export #t)
  (let ([p-prime (poly-derivative p)])
    (if (poly-zero? p-prime)
        p
        (poly-div p (poly-gcd p p-prime)))))""",

    "poly-square-free-factorization": """(define (poly-square-free-factorization p)
  (doc 'export #t)
  (let ([F (poly-field p)])
    (if (poly-zero? p)
        '()
        (let* ([p-prime (poly-derivative p)]
               [a0 (poly-gcd p p-prime)]
               [b0 (poly-div p a0)]
               [c0 (poly-div p-prime a0)])
          (poly-sqf-loop b0 (poly-sub c0 (poly-derivative b0)) 1 '() F)))))""",

    # ==== Interpolation ====
    "poly-lagrange-interpolate": """(define (poly-lagrange-interpolate F points)
  (doc 'export #t)
  (if (null? points)
      (poly-zero-over F)
      (let ([n (length points)]
            [xs (map car points)]
            [ys (map cdr points)])
        (poly-lagrange-sum F xs ys 0 (poly-zero-over F)))))""",

    "poly-newton-interpolate": """(define (poly-newton-interpolate F points)
  (doc 'export #t)
  (if (null? points)
      (poly-zero-over F)
      (let* ([xs (map car points)]
             [ys (map cdr points)]
             [n (length points)]
             [coeffs (poly-divided-differences F xs ys)])
        (poly-newton-form F xs coeffs))))""",

    # ==== Display ====
    "poly->string": """(define (poly->string p . opts)
  (doc 'export #t)
  (let* ([var (if (null? opts) 'x (car opts))]
         [F (poly-field p)]
         [coeffs (poly-coeffs p)]
         [n (length coeffs)])
    (if (and (= n 1) ((field-equal-fn F) (car coeffs) (field-zero F)))
        "0"
        (let loop ([i (- n 1)] [first? #t] [result ""])
          (if (< i 0)
              result
              (let* ([c (list-ref coeffs i)]
                     [term (poly-term->string c i var first? F)])
                (loop (- i 1)
                      (and first? (string=? term ""))
                      (string-append result term))))))))""",

    # ==== Ring Construction ====
    "make-polynomial-ring": """(define (make-polynomial-ring F)
  (doc 'export #t)
  (make-ring
   '()
   (lambda (p1 p2) (poly-add p1 p2))
   (lambda (p1 p2) (poly-mul p1 p2))
   (poly-zero-over F)
   (poly-one-over F)
   (lambda (p) (poly-neg p))
   (lambda (p1 p2) (poly-equal? p1 p2))))""",

    # ==== Backwards Compatibility ====
    "poly-ring": """(doc poly-ring 'export #t)
(define poly-ring poly-field)""",
}


def strip_doc_forms(defn: str) -> str:
    lines = [line for line in defn.splitlines() if not line.strip().startswith("(doc ")]
    return "\n".join(lines)


DOC_FREE_DEFS: Dict[str, str] = {fn: strip_doc_forms(code) for fn, code in DEFS.items()}


# =============================================================================
# Function Order and Specs
# =============================================================================

FUNCTION_ORDER = [
    # Predicates and Accessors (Easy)
    "polynomial?",
    "poly-field",
    "poly-coeffs",
    "poly-degree",
    "poly-leading-coeff",
    "poly-coeff-at",
    "poly-zero?",
    # Constructors (Easy/Medium)
    "make-polynomial",
    "poly-zero-over",
    "poly-one-over",
    "poly-constant",
    "poly-monomial",
    "poly-x",
    # Arithmetic (Medium/Hard)
    "poly-add",
    "poly-neg",
    "poly-sub",
    "poly-scale",
    "poly-mul",
    "poly-power",
    # Equality (Medium)
    "poly-equal?",
    # Division (Hard)
    "poly-divmod",
    "poly-div",
    "poly-mod",
    "poly-divides?",
    # GCD (Hard)
    "poly-gcd",
    "poly-make-monic",
    "poly-extended-gcd",
    "poly-lcm",
    # Evaluation (Medium)
    "poly-eval",
    # Derivative (Medium)
    "poly-derivative",
    # Factorization (Hard)
    "poly-square-free",
    "poly-square-free-factorization",
    # Interpolation (Hard)
    "poly-lagrange-interpolate",
    "poly-newton-interpolate",
    # Display (Medium)
    "poly->string",
    # Ring Construction (Medium)
    "make-polynomial-ring",
    # Backwards Compatibility (Easy)
    "poly-ring",
]

FUNCTION_SPECS = {
    "polynomial?": {
        "signature": "(polynomial? p) -> Boolean",
        "description": "Check if p is a polynomial (tagged list)",
        "params": {"p": "Any value to check"},
        "returns": "#t if p is a polynomial, #f otherwise",
    },
    "poly-field": {
        "signature": "(poly-field p) -> Field",
        "description": "Get the coefficient field of a polynomial",
        "params": {"p": "Polynomial"},
        "returns": "The Field over which p is defined",
    },
    "poly-coeffs": {
        "signature": "(poly-coeffs p) -> (List Coeff)",
        "description": "Get coefficient list in ascending power order",
        "params": {"p": "Polynomial"},
        "returns": "List of coefficients [a_0, a_1, ..., a_n]",
    },
    "poly-degree": {
        "signature": "(poly-degree p) -> Nat",
        "description": "Degree of polynomial (length-1). Zero polynomial has degree -1.",
        "params": {"p": "Polynomial"},
        "returns": "Degree as integer",
    },
    "poly-leading-coeff": {
        "signature": "(poly-leading-coeff p) -> Coeff",
        "description": "Get leading (highest-degree) coefficient",
        "params": {"p": "Polynomial"},
        "returns": "Coefficient of highest power term",
    },
    "poly-coeff-at": {
        "signature": "(poly-coeff-at p k) -> Coeff",
        "description": "Get coefficient of x^k",
        "params": {"p": "Polynomial", "k": "Power index"},
        "returns": "Coefficient at position k, or field-zero if out of bounds",
    },
    "poly-zero?": {
        "signature": "(poly-zero? p) -> Boolean",
        "description": "Check if polynomial is the zero polynomial",
        "params": {"p": "Polynomial"},
        "returns": "#t if p equals zero",
    },
    "make-polynomial": {
        "signature": "(make-polynomial field coeffs) -> Polynomial",
        "description": "Create a polynomial over field F with given coefficients. Automatically normalizes.",
        "params": {"field": "Field structure", "coeffs": "List of coefficients"},
        "returns": "Normalized polynomial",
    },
    "poly-zero-over": {
        "signature": "(poly-zero-over field) -> Polynomial",
        "description": "The zero polynomial over field F",
        "params": {"field": "Field structure"},
        "returns": "Zero polynomial",
    },
    "poly-one-over": {
        "signature": "(poly-one-over field) -> Polynomial",
        "description": "The constant polynomial 1 over field F",
        "params": {"field": "Field structure"},
        "returns": "Constant polynomial equal to 1",
    },
    "poly-constant": {
        "signature": "(poly-constant field c) -> Polynomial",
        "description": "Create constant polynomial from coefficient c",
        "params": {"field": "Field structure", "c": "Constant value"},
        "returns": "Constant polynomial",
    },
    "poly-monomial": {
        "signature": "(poly-monomial field coeff degree) -> Polynomial",
        "description": "Create monomial c*x^n",
        "params": {"field": "Field", "coeff": "Coefficient c", "degree": "Power n"},
        "returns": "Monomial polynomial",
    },
    "poly-x": {
        "signature": "(poly-x field) -> Polynomial",
        "description": "The polynomial x (the indeterminate) over field F",
        "params": {"field": "Field structure"},
        "returns": "Polynomial equal to x",
    },
    "poly-add": {
        "signature": "(poly-add p1 p2) -> Polynomial",
        "description": "Add two polynomials over the same field",
        "params": {"p1": "First polynomial", "p2": "Second polynomial"},
        "returns": "Sum polynomial",
    },
    "poly-neg": {
        "signature": "(poly-neg p) -> Polynomial",
        "description": "Negate a polynomial",
        "params": {"p": "Polynomial to negate"},
        "returns": "Negated polynomial",
    },
    "poly-sub": {
        "signature": "(poly-sub p1 p2) -> Polynomial",
        "description": "Subtract polynomials (p1 - p2)",
        "params": {"p1": "Minuend", "p2": "Subtrahend"},
        "returns": "Difference polynomial",
    },
    "poly-scale": {
        "signature": "(poly-scale p c) -> Polynomial",
        "description": "Multiply polynomial by scalar c",
        "params": {"p": "Polynomial", "c": "Scalar from the field"},
        "returns": "Scaled polynomial",
    },
    "poly-mul": {
        "signature": "(poly-mul p1 p2) -> Polynomial",
        "description": "Multiply two polynomials (convolution)",
        "params": {"p1": "First polynomial", "p2": "Second polynomial"},
        "returns": "Product polynomial",
    },
    "poly-power": {
        "signature": "(poly-power p n) -> Polynomial",
        "description": "Raise polynomial to power n using repeated squaring",
        "params": {"p": "Base polynomial", "n": "Non-negative integer exponent"},
        "returns": "p^n",
    },
    "poly-equal?": {
        "signature": "(poly-equal? p1 p2) -> Boolean",
        "description": "Check if two polynomials are equal",
        "params": {"p1": "First polynomial", "p2": "Second polynomial"},
        "returns": "#t if polynomials are equal",
    },
    "poly-divmod": {
        "signature": "(poly-divmod p1 p2) -> (Polynomial . Polynomial)",
        "description": "Division with remainder: p1 = q * p2 + r where deg(r) < deg(p2). Returns (quotient . remainder)",
        "params": {"p1": "Dividend", "p2": "Divisor"},
        "returns": "Cons pair (quotient . remainder)",
    },
    "poly-div": {
        "signature": "(poly-div p1 p2) -> Polynomial",
        "description": "Get quotient of polynomial division",
        "params": {"p1": "Dividend", "p2": "Divisor"},
        "returns": "Quotient polynomial",
    },
    "poly-mod": {
        "signature": "(poly-mod p1 p2) -> Polynomial",
        "description": "Get remainder of polynomial division",
        "params": {"p1": "Dividend", "p2": "Divisor"},
        "returns": "Remainder polynomial (degree < deg(p2))",
    },
    "poly-divides?": {
        "signature": "(poly-divides? p1 p2) -> Boolean",
        "description": "Check if p1 divides p2 (p2 = q * p1 for some q)",
        "params": {"p1": "Potential divisor", "p2": "Polynomial to divide"},
        "returns": "#t if p1 divides p2 evenly",
    },
    "poly-gcd": {
        "signature": "(poly-gcd p1 p2) -> Polynomial",
        "description": "Compute GCD using Euclidean algorithm. Returns monic GCD.",
        "params": {"p1": "First polynomial", "p2": "Second polynomial"},
        "returns": "Monic GCD polynomial",
    },
    "poly-make-monic": {
        "signature": "(poly-make-monic p) -> Polynomial",
        "description": "Scale polynomial so leading coefficient is 1",
        "params": {"p": "Polynomial"},
        "returns": "Monic polynomial (leading coeff = 1)",
    },
    "poly-extended-gcd": {
        "signature": "(poly-extended-gcd p1 p2) -> (Polynomial Polynomial Polynomial)",
        "description": "Extended Euclidean algorithm. Returns (gcd, s, t) such that gcd = s*p1 + t*p2",
        "params": {"p1": "First polynomial", "p2": "Second polynomial"},
        "returns": "List (gcd s t) of Bezout coefficients",
    },
    "poly-lcm": {
        "signature": "(poly-lcm p1 p2) -> Polynomial",
        "description": "Least common multiple of polynomials",
        "params": {"p1": "First polynomial", "p2": "Second polynomial"},
        "returns": "LCM polynomial",
    },
    "poly-eval": {
        "signature": "(poly-eval p x) -> Coeff",
        "description": "Evaluate polynomial at point using Horner's method",
        "params": {"p": "Polynomial", "x": "Point to evaluate at"},
        "returns": "Value p(x)",
    },
    "poly-derivative": {
        "signature": "(poly-derivative p) -> Polynomial",
        "description": "Formal derivative: d/dx (sum a_k x^k) = sum k*a_k x^{k-1}",
        "params": {"p": "Polynomial"},
        "returns": "Derivative polynomial",
    },
    "poly-square-free": {
        "signature": "(poly-square-free p) -> Polynomial",
        "description": "Square-free part: p / gcd(p, p'). Removes repeated roots.",
        "params": {"p": "Polynomial"},
        "returns": "Square-free polynomial",
    },
    "poly-square-free-factorization": {
        "signature": "(poly-square-free-factorization p) -> (List (Polynomial . Nat))",
        "description": "Yun's algorithm for square-free factorization. Returns list of (factor . multiplicity) pairs.",
        "params": {"p": "Polynomial"},
        "returns": "List of (factor . multiplicity) pairs",
    },
    "poly-lagrange-interpolate": {
        "signature": "(poly-lagrange-interpolate F points) -> Polynomial",
        "description": "Lagrange interpolation through points [(x_0, y_0), ..., (x_n, y_n)]",
        "params": {"F": "Field", "points": "List of (x . y) pairs"},
        "returns": "Interpolating polynomial p with p(x_i) = y_i",
    },
    "poly-newton-interpolate": {
        "signature": "(poly-newton-interpolate F points) -> Polynomial",
        "description": "Newton interpolation using divided differences",
        "params": {"F": "Field", "points": "List of (x . y) pairs"},
        "returns": "Interpolating polynomial",
    },
    "poly->string": {
        "signature": "(poly->string p [var]) -> String",
        "description": "Pretty-print polynomial. Optional variable name (default 'x).",
        "params": {"p": "Polynomial", "var": "Optional variable symbol"},
        "returns": "String representation",
    },
    "make-polynomial-ring": {
        "signature": "(make-polynomial-ring F) -> Ring",
        "description": "Construct polynomial ring F[x] over coefficient field F",
        "params": {"F": "Field"},
        "returns": "Ring structure for F[x]",
    },
    "poly-ring": {
        "signature": "(poly-ring p) -> Field",
        "description": "Alias for poly-field for backwards compatibility",
        "params": {"p": "Polynomial"},
        "returns": "Coefficient field",
    },
}


# =============================================================================
# Difficulty Classification
# =============================================================================

BASE_DIFFICULTY = {
    # Easy: simple accessors/predicates
    "polynomial?": "easy",
    "poly-field": "easy",
    "poly-coeffs": "easy",
    "poly-degree": "easy",
    "poly-leading-coeff": "easy",
    "poly-coeff-at": "easy",
    "poly-zero?": "easy",
    "poly-zero-over": "easy",
    "poly-one-over": "easy",
    "poly-constant": "easy",
    "poly-x": "easy",
    "poly-ring": "easy",
    # Medium: arithmetic, equality, evaluation, display, constructors with logic
    "make-polynomial": "medium",
    "poly-monomial": "medium",
    "poly-add": "medium",
    "poly-neg": "medium",
    "poly-sub": "medium",
    "poly-scale": "medium",
    "poly-equal?": "medium",
    "poly-eval": "medium",
    "poly-derivative": "medium",
    "poly->string": "medium",
    "poly-power": "medium",
    "make-polynomial-ring": "medium",
    # Hard: division, GCD, factorization, interpolation
    "poly-mul": "hard",
    "poly-divmod": "hard",
    "poly-div": "hard",
    "poly-mod": "hard",
    "poly-divides?": "hard",
    "poly-gcd": "hard",
    "poly-make-monic": "hard",
    "poly-extended-gcd": "hard",
    "poly-lcm": "hard",
    "poly-square-free": "hard",
    "poly-square-free-factorization": "hard",
    "poly-lagrange-interpolate": "hard",
    "poly-newton-interpolate": "hard",
}

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
DIFFICULTY_INDEX = {name: idx for idx, name in enumerate(DIFFICULTY_LEVELS)}

# =============================================================================
# Skeletons for completion tasks
# =============================================================================

SKELETONS = {
    "polynomial?": """(define (polynomial? p)
  ;; TODO: Check if p is tagged as 'polynomial
  <TODO>)""",
    "poly-field": """(define (poly-field p)
  ;; TODO: Extract field from polynomial structure
  <TODO>)""",
    "poly-coeffs": """(define (poly-coeffs p)
  ;; TODO: Extract coefficient list from polynomial
  <TODO>)""",
    "poly-degree": """(define (poly-degree p)
  ;; TODO: Compute degree from coefficient list
  <TODO>)""",
    "poly-leading-coeff": """(define (poly-leading-coeff p)
  ;; TODO: Get last coefficient from list
  <TODO>)""",
    "poly-coeff-at": """(define (poly-coeff-at p k)
  ;; TODO: Get k-th coefficient or zero if out of bounds
  <TODO>)""",
    "poly-zero?": """(define (poly-zero? p)
  ;; TODO: Check if polynomial has single zero coefficient
  <TODO>)""",
    "make-polynomial": """(define (make-polynomial field coeffs)
  ;; TODO: Create polynomial tagged list with normalized coeffs
  <TODO>)""",
    "poly-zero-over": """(define (poly-zero-over field)
  ;; TODO: Create zero polynomial over field
  <TODO>)""",
    "poly-one-over": """(define (poly-one-over field)
  ;; TODO: Create constant 1 polynomial over field
  <TODO>)""",
    "poly-constant": """(define (poly-constant field c)
  ;; TODO: Create constant polynomial from value c
  <TODO>)""",
    "poly-monomial": """(define (poly-monomial field coeff degree)
  ;; TODO: Create c*x^n with leading zeros
  <TODO>)""",
    "poly-x": """(define (poly-x field)
  ;; TODO: Create polynomial representing the indeterminate x
  <TODO>)""",
    "poly-add": """(define (poly-add p1 p2)
  ;; TODO: Add coefficients pairwise using field addition
  <TODO>)""",
    "poly-neg": """(define (poly-neg p)
  ;; TODO: Negate each coefficient using field negation
  <TODO>)""",
    "poly-sub": """(define (poly-sub p1 p2)
  ;; TODO: Subtract by adding negation
  <TODO>)""",
    "poly-scale": """(define (poly-scale p c)
  ;; TODO: Multiply each coefficient by scalar c
  <TODO>)""",
    "poly-mul": """(define (poly-mul p1 p2)
  ;; TODO: Convolution of coefficient lists
  <TODO>)""",
    "poly-power": """(define (poly-power p n)
  ;; TODO: Exponentiation by repeated squaring
  <TODO>)""",
    "poly-equal?": """(define (poly-equal? p1 p2)
  ;; TODO: Compare coefficients using field equality
  <TODO>)""",
    "poly-divmod": """(define (poly-divmod p1 p2)
  ;; TODO: Polynomial long division, return (quotient . remainder)
  <TODO>)""",
    "poly-div": """(define (poly-div p1 p2)
  ;; TODO: Return quotient from divmod
  <TODO>)""",
    "poly-mod": """(define (poly-mod p1 p2)
  ;; TODO: Return remainder from divmod
  <TODO>)""",
    "poly-divides?": """(define (poly-divides? p1 p2)
  ;; TODO: Check if remainder of p2/p1 is zero
  <TODO>)""",
    "poly-gcd": """(define (poly-gcd p1 p2)
  ;; TODO: Euclidean algorithm, return monic GCD
  <TODO>)""",
    "poly-make-monic": """(define (poly-make-monic p)
  ;; TODO: Scale by inverse of leading coefficient
  <TODO>)""",
    "poly-extended-gcd": """(define (poly-extended-gcd p1 p2)
  ;; TODO: Extended Euclidean algorithm for Bezout coefficients
  <TODO>)""",
    "poly-lcm": """(define (poly-lcm p1 p2)
  ;; TODO: LCM = (p1 * p2) / GCD(p1, p2)
  <TODO>)""",
    "poly-eval": """(define (poly-eval p x)
  ;; TODO: Evaluate using Horner's method
  <TODO>)""",
    "poly-derivative": """(define (poly-derivative p)
  ;; TODO: Formal derivative: multiply each coeff by its power
  <TODO>)""",
    "poly-square-free": """(define (poly-square-free p)
  ;; TODO: Square-free part: p / GCD(p, p')
  <TODO>)""",
    "poly-square-free-factorization": """(define (poly-square-free-factorization p)
  ;; TODO: Yun's algorithm for square-free factorization
  <TODO>)""",
    "poly-lagrange-interpolate": """(define (poly-lagrange-interpolate F points)
  ;; TODO: Lagrange interpolation through given points
  <TODO>)""",
    "poly-newton-interpolate": """(define (poly-newton-interpolate F points)
  ;; TODO: Newton interpolation using divided differences
  <TODO>)""",
    "poly->string": """(define (poly->string p . opts)
  ;; TODO: Pretty-print polynomial with variable name
  <TODO>)""",
    "make-polynomial-ring": """(define (make-polynomial-ring F)
  ;; TODO: Create Ring structure with poly operations
  <TODO>)""",
    "poly-ring": """(define (poly-ring p)
  ;; TODO: Alias for poly-field
  <TODO>)""",
}


# =============================================================================
# Verification Expressions
# =============================================================================

VERIFY_BY_FUNCTION = {
    # Easy
    "polynomial?": "(let ([p (make-polynomial Q-field '(1 2 3))]) (and (polynomial? p) (not (polynomial? 'not-a-poly)) (not (polynomial? 42))))",
    "poly-field": "(let ([p (make-polynomial Q-field '(1 2))]) (eq? (poly-field p) Q-field))",
    "poly-coeffs": "(let ([p (make-polynomial Q-field '(1 2 3))]) (equal? (poly-coeffs p) '(1 2 3)))",
    "poly-degree": "(let ([p1 (make-polynomial Q-field '(1))] [p2 (make-polynomial Q-field '(1 2))] [p3 (make-polynomial Q-field '(1 2 3 4))]) (and (= (poly-degree p1) 0) (= (poly-degree p2) 1) (= (poly-degree p3) 3)))",
    "poly-leading-coeff": "(let ([p (make-polynomial Q-field '(3 2 5))]) (= (poly-leading-coeff p) 5))",
    "poly-coeff-at": "(let ([p (make-polynomial Q-field '(1 2 3))]) (and (= (poly-coeff-at p 0) 1) (= (poly-coeff-at p 1) 2) (= (poly-coeff-at p 2) 3) (= (poly-coeff-at p 3) 0) (= (poly-coeff-at p 5) 0)))",
    "poly-zero?": "(let ([zero (poly-zero-over Q-field)] [nonzero (make-polynomial Q-field '(1 2))] [tricky (make-polynomial Q-field '(0 2))]) (and (poly-zero? zero) (not (poly-zero? nonzero)) (not (poly-zero? tricky))))",
    "make-polynomial": "(let ([p (make-polynomial Q-field '(1 0 2 0))]) (and (polynomial? p) (equal? (poly-coeffs p) '(1 0 2)) (= (poly-degree p) 2)))",
    "poly-zero-over": "(let ([z (poly-zero-over Q-field)]) (and (polynomial? z) (poly-zero? z) (= (poly-degree z) 0)))",
    "poly-one-over": "(let ([one (poly-one-over Q-field)]) (and (polynomial? one) (= (poly-degree one) 0) (= (poly-leading-coeff one) 1)))",
    "poly-constant": "(let ([c (poly-constant Q-field 5)]) (and (= (poly-degree c) 0) (= (poly-leading-coeff c) 5)))",
    "poly-monomial": "(let ([m (poly-monomial Q-field 3 4)]) (and (= (poly-degree m) 4) (= (poly-leading-coeff m) 3) (= (poly-coeff-at m 0) 0)))",
    "poly-x": "(let ([x (poly-x Q-field)]) (and (= (poly-degree x) 1) (= (poly-coeff-at x 0) 0) (= (poly-coeff-at x 1) 1)))",
    "poly-ring": "(let ([p (make-polynomial Q-field '(1 2))]) (eq? (poly-ring p) Q-field))",
    # Medium
    "poly-add": "(let* ([p1 (make-polynomial Q-field '(1 2))] [p2 (make-polynomial Q-field '(3 0 1))] [sum (poly-add p1 p2)]) (and (= (poly-degree sum) 2) (= (poly-coeff-at sum 0) 4) (= (poly-coeff-at sum 1) 2) (= (poly-coeff-at sum 2) 1)))",
    "poly-neg": "(let* ([p (make-polynomial Q-field '(1 -2 3))] [np (poly-neg p)]) (and (= (poly-coeff-at np 0) -1) (= (poly-coeff-at np 1) 2) (= (poly-coeff-at np 2) -3)))",
    "poly-sub": "(let* ([p1 (make-polynomial Q-field '(3 0 1))] [p2 (make-polynomial Q-field '(1 2))] [diff (poly-sub p1 p2)]) (and (= (poly-degree diff) 2) (= (poly-coeff-at diff 0) 2) (= (poly-coeff-at diff 1) -2) (= (poly-coeff-at diff 2) 1)))",
    "poly-scale": "(let* ([p (make-polynomial Q-field '(1 2))] [sp (poly-scale p 3)]) (and (= (poly-coeff-at sp 0) 3) (= (poly-coeff-at sp 1) 6)))",
    "poly-equal?": "(let* ([p1 (make-polynomial Q-field '(1 2 3))] [p2 (make-polynomial Q-field '(1 2 3))] [p3 (make-polynomial Q-field '(1 2 4))] [p4 (make-polynomial Q-field '(1 2))]) (and (poly-equal? p1 p2) (not (poly-equal? p1 p3)) (not (poly-equal? p1 p4))))",
    "poly-eval": "(let ([p1 (make-polynomial Q-field '(2 3))] [p2 (make-polynomial Q-field '(1 1 1))]) (and (= (poly-eval p1 4) 14) (= (poly-eval p2 2) 7)))",
    "poly-derivative": "(let* ([p (make-polynomial Q-field '(0 1 2 1))] [dp (poly-derivative p)]) (and (= (poly-degree dp) 2) (= (poly-coeff-at dp 0) 1) (= (poly-coeff-at dp 1) 4) (= (poly-coeff-at dp 2) 3)))",
    "poly-power": "(let* ([p (make-polynomial Q-field '(1 1))] [p3 (poly-power p 3)]) (and (= (poly-degree p3) 3) (= (poly-coeff-at p3 0) 1) (= (poly-coeff-at p3 1) 3) (= (poly-coeff-at p3 2) 3) (= (poly-coeff-at p3 3) 1)))",
    "make-polynomial-ring": "(let* ([R (make-polynomial-ring Q-field)] [x (poly-x Q-field)] [one (poly-one-over Q-field)]) (and (ring? R) (poly-equal? (ring-add R x (ring-neg R x)) (ring-zero R)) (poly-equal? (ring-mul R x one) x)))",
    "poly->string": "(let ([p (make-polynomial Q-field '(1 2 1))]) (string=? (poly->string p) \"x^2 + 2x + 1\"))",
    # Hard
    "poly-mul": "(let* ([p1 (make-polynomial Q-field '(1 1))] [p2 (make-polynomial Q-field '(1 -1))] [prod (poly-mul p1 p2)]) (and (= (poly-degree prod) 2) (= (poly-coeff-at prod 0) 1) (= (poly-coeff-at prod 1) 0) (= (poly-coeff-at prod 2) -1)))",
    "poly-divmod": "(let* ([dividend (make-polynomial Q-field '(-1 0 1))] [divisor (make-polynomial Q-field '(-1 1))] [result (poly-divmod dividend divisor)] [q (car result)] [r (cdr result)]) (and (poly-zero? r) (= (poly-coeff-at q 0) 1) (= (poly-coeff-at q 1) 1)))",
    "poly-div": "(let* ([p1 (make-polynomial Q-field '(0 0 1))] [p2 (make-polynomial Q-field '(0 1))] [q (poly-div p1 p2)]) (and (= (poly-degree q) 1) (= (poly-coeff-at q 0) 0) (= (poly-coeff-at q 1) 1)))",
    "poly-mod": "(let* ([p1 (make-polynomial Q-field '(1 0 1))] [p2 (make-polynomial Q-field '(0 1))] [r (poly-mod p1 p2)]) (and (= (poly-degree r) 0) (= (poly-leading-coeff r) 1)))",
    "poly-divides?": "(let* ([p1 (make-polynomial Q-field '(-1 1))] [p2 (make-polynomial Q-field '(-1 0 1))]) (and (poly-divides? p1 p2) (not (poly-divides? p2 p1))))",
    "poly-gcd": "(let* ([xp1 (make-polynomial Q-field '(1 1))] [xm1 (make-polynomial Q-field '(-1 1))] [p1 (poly-mul xp1 xp1)] [p2 (poly-mul xp1 xm1)] [g (poly-gcd p1 p2)]) (and (= (poly-degree g) 1) (= (poly-leading-coeff g) 1) (= (poly-coeff-at g 0) 1) (= (poly-coeff-at g 1) 1)))",
    "poly-make-monic": "(let* ([p (make-polynomial Q-field '(2 4 2))] [m (poly-make-monic p)]) (and (= (poly-leading-coeff m) 1) (= (poly-coeff-at m 0) 1) (= (poly-coeff-at m 1) 2) (= (poly-coeff-at m 2) 1)))",
    "poly-extended-gcd": "(let* ([p1 (make-polynomial Q-field '(1 1))] [p2 (make-polynomial Q-field '(-1 1))] [result (poly-extended-gcd p1 p2)] [g (car result)] [s (cadr result)] [t (caddr result)] [lhs (poly-add (poly-mul s p1) (poly-mul t p2))]) (and (poly-equal? g lhs) (= (poly-leading-coeff g) 1)))",
    "poly-lcm": "(let* ([xp1 (make-polynomial Q-field '(1 1))] [xm1 (make-polynomial Q-field '(-1 1))] [p1 (poly-mul xp1 xp1)] [p2 (poly-mul xp1 xm1)] [l (poly-lcm p1 p2)] [prod (poly-mul p1 p2)] [g (poly-gcd p1 p2)]) (poly-equal? l (poly-div prod g)))",
    "poly-square-free": "(let* ([xp1 (make-polynomial Q-field '(1 1))] [xm1 (make-polynomial Q-field '(-1 1))] [p (poly-mul (poly-mul xp1 xp1) xm1)] [sf (poly-square-free p)]) (= (poly-degree sf) 2))",
    "poly-square-free-factorization": "(let* ([xp1 (make-polynomial Q-field '(1 1))] [xm1 (make-polynomial Q-field '(-1 1))] [p (poly-mul (poly-mul xp1 xp1) xm1)] [factors (poly-square-free-factorization p)]) (= (length factors) 2))",
    "poly-lagrange-interpolate": "(let* ([points '((0 . 1) (1 . 2) (2 . 5))] [p (poly-lagrange-interpolate Q-field points)]) (and (= (poly-eval p 0) 1) (= (poly-eval p 1) 2) (= (poly-eval p 2) 5)))",
    "poly-newton-interpolate": "(let* ([points '((0 . 1) (1 . 3) (3 . 10))] [p (poly-newton-interpolate Q-field points)]) (and (= (poly-eval p 0) 1) (= (poly-eval p 1) 3) (= (poly-eval p 3) 10)))",
}


# =============================================================================
# Python Pseudocode Snippets
# =============================================================================

PYTHON_SNIPPETS = {
    "polynomial?": """def is_polynomial(p):
    return isinstance(p, list) and len(p) > 0 and p[0] == 'polynomial'""",
    "poly-field": """def polynomial_field(p):
    return p[1]""",
    "poly-coeffs": """def polynomial_coeffs(p):
    return p[2]""",
    "poly-degree": """def polynomial_degree(p):
    coeffs = polynomial_coeffs(p)
    return len(coeffs) - 1""",
    "poly-leading-coeff": """def polynomial_leading_coeff(p):
    coeffs = polynomial_coeffs(p)
    return coeffs[-1] if coeffs else field_zero(polynomial_field(p))""",
    "poly-coeff-at": """def polynomial_coeff_at(p, k):
    coeffs = polynomial_coeffs(p)
    F = polynomial_field(p)
    return coeffs[k] if k < len(coeffs) else field_zero(F)""",
    "poly-zero?": """def is_zero_polynomial(p):
    coeffs = polynomial_coeffs(p)
    F = polynomial_field(p)
    return len(coeffs) == 1 and field_equal(F, coeffs[0], field_zero(F))""",
    "make-polynomial": """def make_polynomial(field, coeffs):
    # Normalize by stripping trailing zeros
    while len(coeffs) > 1 and field_equal(field, coeffs[-1], field_zero(field)):
        coeffs = coeffs[:-1]
    return ['polynomial', field, coeffs]""",
    "poly-zero-over": """def zero_polynomial_over(field):
    return make_polynomial(field, [field_zero(field)])""",
    "poly-one-over": """def one_polynomial_over(field):
    return make_polynomial(field, [field_one(field)])""",
    "poly-constant": """def constant_polynomial(field, c):
    return make_polynomial(field, [c])""",
    "poly-monomial": """def monomial(field, coeff, degree):
    zero = field_zero(field)
    if field_equal(field, coeff, zero):
        return zero_polynomial_over(field)
    coeffs = [zero] * degree + [coeff]
    return make_polynomial(field, coeffs)""",
    "poly-x": """def polynomial_x(field):
    return monomial(field, field_one(field), 1)""",
    "poly-add": """def polynomial_add(p1, p2):
    F = polynomial_field(p1)
    c1 = polynomial_coeffs(p1)
    c2 = polynomial_coeffs(p2)
    result = []
    for i in range(max(len(c1), len(c2))):
        a = c1[i] if i < len(c1) else field_zero(F)
        b = c2[i] if i < len(c2) else field_zero(F)
        result.append(field_add(F, a, b))
    return make_polynomial(F, result)""",
    "poly-neg": """def polynomial_neg(p):
    F = polynomial_field(p)
    coeffs = polynomial_coeffs(p)
    return make_polynomial(F, [field_neg(F, c) for c in coeffs])""",
    "poly-sub": """def polynomial_sub(p1, p2):
    return polynomial_add(p1, polynomial_neg(p2))""",
    "poly-scale": """def polynomial_scale(p, scalar):
    F = polynomial_field(p)
    coeffs = polynomial_coeffs(p)
    return make_polynomial(F, [field_mul(F, scalar, c) for c in coeffs])""",
    "poly-mul": """def polynomial_mul(p1, p2):
    F = polynomial_field(p1)
    c1 = polynomial_coeffs(p1)
    c2 = polynomial_coeffs(p2)
    if is_zero_polynomial(p1) or is_zero_polynomial(p2):
        return zero_polynomial_over(F)
    n = len(c1) + len(c2) - 1
    result = [field_zero(F)] * n
    for k in range(n):
        s = field_zero(F)
        for i in range(min(k + 1, len(c1))):
            j = k - i
            if j < len(c2):
                s = field_add(F, s, field_mul(F, c1[i], c2[j]))
        result[k] = s
    return make_polynomial(F, result)""",
    "poly-power": """def polynomial_power(p, n):
    if n == 0:
        return one_polynomial_over(polynomial_field(p))
    if n == 1:
        return p
    if n % 2 == 0:
        half = polynomial_power(p, n // 2)
        return polynomial_mul(half, half)
    return polynomial_mul(p, polynomial_power(p, n - 1))""",
    "poly-equal?": """def polynomial_equal(p1, p2):
    F = polynomial_field(p1)
    c1 = polynomial_coeffs(p1)
    c2 = polynomial_coeffs(p2)
    if len(c1) != len(c2):
        return False
    return all(field_equal(F, a, b) for a, b in zip(c1, c2))""",
    "poly-divmod": """def polynomial_divmod(p1, p2):
    F = polynomial_field(p1)
    if is_zero_polynomial(p2):
        raise ValueError("Division by zero")
    quotient = zero_polynomial_over(F)
    remainder = p1
    d2 = polynomial_degree(p2)
    lc2 = polynomial_leading_coeff(p2)
    while not is_zero_polynomial(remainder) and polynomial_degree(remainder) >= d2:
        d_diff = polynomial_degree(remainder) - d2
        coeff = field_div(F, polynomial_leading_coeff(remainder), lc2)
        term = monomial(F, coeff, d_diff)
        quotient = polynomial_add(quotient, term)
        subtrahend = polynomial_mul(term, p2)
        remainder = polynomial_sub(remainder, subtrahend)
    return (quotient, remainder)""",
    "poly-div": """def polynomial_div(p1, p2):
    return polynomial_divmod(p1, p2)[0]""",
    "poly-mod": """def polynomial_mod(p1, p2):
    return polynomial_divmod(p1, p2)[1]""",
    "poly-divides?": """def polynomial_divides(p1, p2):
    return is_zero_polynomial(polynomial_mod(p2, p1))""",
    "poly-gcd": """def polynomial_gcd(p1, p2):
    F = polynomial_field(p1)
    if is_zero_polynomial(p2):
        return make_monic(p1)
    return polynomial_gcd(p2, polynomial_mod(p1, p2))""",
    "poly-make-monic": """def make_monic(p):
    if is_zero_polynomial(p):
        return p
    F = polynomial_field(p)
    lc = polynomial_leading_coeff(p)
    inv_lc = field_inv(F, lc)
    return polynomial_scale(p, inv_lc)""",
    "poly-extended-gcd": """def polynomial_extended_gcd(p1, p2):
    F = polynomial_field(p1)
    def loop(r0, r1, s0, s1, t0, t1):
        if is_zero_polynomial(r1):
            if is_zero_polynomial(r0):
                return (r0, s0, t0)
            lc = polynomial_leading_coeff(r0)
            inv_lc = field_inv(F, lc)
            return (polynomial_scale(r0, inv_lc), polynomial_scale(s0, inv_lc), polynomial_scale(t0, inv_lc))
        q, r2 = polynomial_divmod(r0, r1)
        s2 = polynomial_sub(s0, polynomial_mul(q, s1))
        t2 = polynomial_sub(t0, polynomial_mul(q, t1))
        return loop(r1, r2, s1, s2, t1, t2)
    return loop(p1, p2, one_polynomial_over(F), zero_polynomial_over(F), zero_polynomial_over(F), one_polynomial_over(F))""",
    "poly-lcm": """def polynomial_lcm(p1, p2):
    if is_zero_polynomial(p1) or is_zero_polynomial(p2):
        return zero_polynomial_over(polynomial_field(p1))
    return polynomial_div(polynomial_mul(p1, p2), polynomial_gcd(p1, p2))""",
    "poly-eval": """def polynomial_eval(p, x):
    F = polynomial_field(p)
    coeffs = list(reversed(polynomial_coeffs(p)))
    if not coeffs:
        return field_zero(F)
    result = coeffs[0]
    for c in coeffs[1:]:
        result = field_add(F, field_mul(F, result, x), c)
    return result""",
    "poly-derivative": """def polynomial_derivative(p):
    F = polynomial_field(p)
    coeffs = polynomial_coeffs(p)
    if len(coeffs) <= 1:
        return zero_polynomial_over(F)
    result = []
    for i, c in enumerate(coeffs[1:], 1):
        result.append(field_mul_int(F, c, i))
    return make_polynomial(F, result)""",
    "poly-square-free": """def polynomial_square_free(p):
    p_prime = polynomial_derivative(p)
    if is_zero_polynomial(p_prime):
        return p
    return polynomial_div(p, polynomial_gcd(p, p_prime))""",
    "poly-square-free-factorization": """def polynomial_square_free_factorization(p):
    F = polynomial_field(p)
    if is_zero_polynomial(p):
        return []
    p_prime = polynomial_derivative(p)
    a0 = polynomial_gcd(p, p_prime)
    b0 = polynomial_div(p, a0)
    c0 = polynomial_div(p_prime, a0)
    # Yun's algorithm loop would continue here
    result = []
    if not polynomial_equal(a0, one_polynomial_over(F)):
        result.append((a0, 1))
    return result""",
    "poly-lagrange-interpolate": """def lagrange_interpolate(F, points):
    if not points:
        return zero_polynomial_over(F)
    result = zero_polynomial_over(F)
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        # Compute Lagrange basis polynomial L_i
        Li = one_polynomial_over(F)
        denom = field_one(F)
        for j, xj in enumerate(xs):
            if i != j:
                # Li *= (x - xj)
                factor = make_polynomial(F, [field_neg(F, xj), field_one(F)])
                Li = polynomial_mul(Li, factor)
                denom = field_mul(F, denom, field_sub(F, xi, xj))
        Li = polynomial_scale(Li, field_div(F, yi, denom))
        result = polynomial_add(result, Li)
    return result""",
    "poly-newton-interpolate": """def newton_interpolate(F, points):
    if not points:
        return zero_polynomial_over(F)
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    # Compute divided differences
    n = len(points)
    table = list(ys)
    coeffs = [table[0]]
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            table[i] = field_div(F, field_sub(F, table[i], table[i-1]), field_sub(F, xs[i], xs[i-j]))
        coeffs.append(table[j])
    # Build Newton form
    result = constant_polynomial(F, coeffs[-1])
    for i in range(n - 2, -1, -1):
        factor = make_polynomial(F, [field_neg(F, xs[i]), field_one(F)])
        result = polynomial_add(constant_polynomial(F, coeffs[i]), polynomial_mul(factor, result))
    return result""",
    "poly->string": """def polynomial_to_string(p, var='x'):
    coeffs = polynomial_coeffs(p)
    F = polynomial_field(p)
    if len(coeffs) == 1 and field_equal(F, coeffs[0], field_zero(F)):
        return "0"
    parts = []
    for i in range(len(coeffs) - 1, -1, -1):
        c = coeffs[i]
        if field_equal(F, c, field_zero(F)):
            continue
        term = ""
        if i == 0:
            term = str(c)
        elif i == 1:
            term = f"{c}{var}" if c != 1 else var
        else:
            term = f"{c}{var}^{i}" if c != 1 else f"{var}^{i}"
        parts.append(term)
    return " + ".join(parts)""",
    "make-polynomial-ring": """def make_polynomial_ring(F):
    return make_ring(
        elements=[],  # Not enumerable
        add_op=lambda p1, p2: polynomial_add(p1, p2),
        mul_op=lambda p1, p2: polynomial_mul(p1, p2),
        zero=zero_polynomial_over(F),
        one=one_polynomial_over(F),
        neg_fn=lambda p: polynomial_neg(p),
        equal_fn=lambda p1, p2: polynomial_equal(p1, p2)
    )""",
    "poly-ring": """def polynomial_ring(p):
    return polynomial_field(p)""",
}


# =============================================================================
# Chez-style Scheme Snippets (alternative implementations)
# =============================================================================

CHEZ_SNIPPETS = {
    "polynomial?": """(define polynomial?
  (lambda (p)
    (if (pair? p)
        (eq? (car p) 'polynomial)
        #f)))""",
    "poly-field": """(define poly-field
  (lambda (p)
    (cadr p)))""",
    "poly-coeffs": """(define poly-coeffs
  (lambda (p)
    (caddr p)))""",
    "poly-degree": """(define poly-degree
  (lambda (p)
    (let ([cs (poly-coeffs p)])
      (- (length cs) 1))))""",
    "poly-leading-coeff": """(define poly-leading-coeff
  (lambda (p)
    (let ([cs (poly-coeffs p)])
      (cond
        [(null? cs) (field-zero (poly-field p))]
        [else (list-ref cs (- (length cs) 1))]))))""",
    "poly-coeff-at": """(define poly-coeff-at
  (lambda (p k)
    (let ([cs (poly-coeffs p)] [F (poly-field p)])
      (cond
        [(>= k (length cs)) (field-zero F)]
        [else (list-ref cs k)])))))""",
    "poly-zero?": """(define poly-zero?
  (lambda (p)
    (let ([F (poly-field p)] [cs (poly-coeffs p)])
      (and (= (length cs) 1)
           ((field-equal-fn F) (car cs) (field-zero F))))))""",
    "make-polynomial": """(define make-polynomial
  (lambda (fld coeffs)
    (list 'polynomial fld (poly-normalize-coeffs fld coeffs))))""",
    "poly-zero-over": """(define poly-zero-over
  (lambda (fld)
    (make-polynomial fld (list (field-zero fld)))))""",
    "poly-one-over": """(define poly-one-over
  (lambda (fld)
    (make-polynomial fld (list (field-one fld)))))""",
    "poly-constant": """(define poly-constant
  (lambda (fld c)
    (make-polynomial fld (list c))))""",
    "poly-monomial": """(define poly-monomial
  (lambda (fld coeff deg)
    (let ([z (field-zero fld)])
      (if ((field-equal-fn fld) coeff z)
          (poly-zero-over fld)
          (make-polynomial fld
            (append (make-list deg z) (list coeff)))))))""",
    "poly-x": """(define poly-x
  (lambda (fld)
    (poly-monomial fld (field-one fld) 1)))""",
    "poly-add": """(define poly-add
  (lambda (a b)
    (let* ([F (poly-field a)]
           [as (poly-coeffs a)]
           [bs (poly-coeffs b)]
           [plus (field-add-op F)])
      (make-polynomial F (poly-add-coeffs plus as bs (field-zero F))))))""",
    "poly-neg": """(define poly-neg
  (lambda (p)
    (let* ([F (poly-field p)]
           [negate (field-neg-fn F)])
      (make-polynomial F (map negate (poly-coeffs p))))))""",
    "poly-sub": """(define poly-sub
  (lambda (a b)
    (poly-add a (poly-neg b))))""",
    "poly-scale": """(define poly-scale
  (lambda (p scalar)
    (let* ([F (poly-field p)]
           [times (field-mul-op F)])
      (make-polynomial F
        (map (lambda (x) (times scalar x)) (poly-coeffs p))))))""",
    "poly-mul": """(define poly-mul
  (lambda (a b)
    (let* ([F (poly-field a)]
           [as (poly-coeffs a)]
           [bs (poly-coeffs b)]
           [z (field-zero F)]
           [plus (field-add-op F)]
           [times (field-mul-op F)]
           [n1 (length as)]
           [n2 (length bs)]
           [n (+ n1 n2 -1)])
      (if (or (poly-zero? a) (poly-zero? b))
          (poly-zero-over F)
          (make-polynomial F
            (let iterate ([k 0] [acc '()])
              (if (= k n)
                  (reverse acc)
                  (iterate (+ k 1)
                           (cons (poly-mul-coeff-at plus times as bs k n1 n2 z)
                                 acc))))))))))""",
    "poly-power": """(define poly-power
  (lambda (p n)
    (cond
      [(= n 0) (poly-one-over (poly-field p))]
      [(= n 1) p]
      [(even? n)
       (let ([h (poly-power p (quotient n 2))])
         (poly-mul h h))]
      [else (poly-mul p (poly-power p (- n 1)))]))))""",
    "poly-equal?": """(define poly-equal?
  (lambda (a b)
    (let* ([F (poly-field a)]
           [eq (field-equal-fn F)]
           [as (poly-coeffs a)]
           [bs (poly-coeffs b)])
      (and (= (length as) (length bs))
           (let check ([xs as] [ys bs])
             (or (null? xs)
                 (and (eq (car xs) (car ys))
                      (check (cdr xs) (cdr ys))))))))""",
    "poly-divmod": """(define poly-divmod
  (lambda (n d)
    (let* ([F (poly-field n)]
           [dd (poly-degree d)])
      (if (poly-zero? d)
          (error 'poly-divmod "division by zero")
          (poly-divmod-loop n d (poly-zero-over F) F)))))""",
    "poly-div": """(define poly-div
  (lambda (n d)
    (car (poly-divmod n d))))""",
    "poly-mod": """(define poly-mod
  (lambda (n d)
    (cdr (poly-divmod n d))))""",
    "poly-divides?": """(define poly-divides?
  (lambda (d n)
    (poly-zero? (poly-mod n d))))""",
    "poly-gcd": """(define poly-gcd
  (lambda (a b)
    (let ([F (poly-field a)])
      (if (poly-zero? b)
          (poly-make-monic a)
          (poly-gcd b (poly-mod a b))))))""",
    "poly-make-monic": """(define poly-make-monic
  (lambda (p)
    (if (poly-zero? p)
        p
        (let* ([F (poly-field p)]
               [lc (poly-leading-coeff p)]
               [inv (field-inv F lc)])
          (poly-scale p inv)))))""",
    "poly-extended-gcd": """(define poly-extended-gcd
  (lambda (a b)
    (let ([F (poly-field a)])
      (poly-ext-gcd-loop a b
                         (poly-one-over F) (poly-zero-over F)
                         (poly-zero-over F) (poly-one-over F)
                         F))))""",
    "poly-lcm": """(define poly-lcm
  (lambda (a b)
    (if (or (poly-zero? a) (poly-zero? b))
        (poly-zero-over (poly-field a))
        (poly-div (poly-mul a b) (poly-gcd a b)))))""",
    "poly-eval": """(define poly-eval
  (lambda (p x)
    (let* ([F (poly-field p)]
           [cs (reverse (poly-coeffs p))]
           [plus (field-add-op F)]
           [times (field-mul-op F)])
      (if (null? cs)
          (field-zero F)
          (let eval-loop ([rest (cdr cs)] [acc (car cs)])
            (if (null? rest)
                acc
                (eval-loop (cdr rest) (plus (times acc x) (car rest))))))))))""",
    "poly-derivative": """(define poly-derivative
  (lambda (p)
    (let* ([F (poly-field p)]
           [cs (poly-coeffs p)])
      (if (<= (length cs) 1)
          (poly-zero-over F)
          (make-polynomial F
            (let deriv-loop ([xs (cdr cs)] [k 1] [acc '()])
              (if (null? xs)
                  (reverse acc)
                  (deriv-loop (cdr xs) (+ k 1)
                              (cons (poly-scalar-mul-int F (car xs) k) acc))))))))))""",
    "poly-square-free": """(define poly-square-free
  (lambda (p)
    (let ([dp (poly-derivative p)])
      (if (poly-zero? dp)
          p
          (poly-div p (poly-gcd p dp))))))""",
    "poly-square-free-factorization": """(define poly-square-free-factorization
  (lambda (p)
    (let ([F (poly-field p)])
      (if (poly-zero? p)
          '()
          (let* ([dp (poly-derivative p)]
                 [a0 (poly-gcd p dp)]
                 [b0 (poly-div p a0)]
                 [c0 (poly-div dp a0)])
            (poly-sqf-loop b0 (poly-sub c0 (poly-derivative b0)) 1 '() F))))))""",
    "poly-lagrange-interpolate": """(define poly-lagrange-interpolate
  (lambda (F pts)
    (if (null? pts)
        (poly-zero-over F)
        (let ([xs (map car pts)] [ys (map cdr pts)])
          (poly-lagrange-sum F xs ys 0 (poly-zero-over F))))))""",
    "poly-newton-interpolate": """(define poly-newton-interpolate
  (lambda (F pts)
    (if (null? pts)
        (poly-zero-over F)
        (let* ([xs (map car pts)]
               [ys (map cdr pts)]
               [coeffs (poly-divided-differences F xs ys)])
          (poly-newton-form F xs coeffs))))))""",
    "poly->string": """(define poly->string
  (lambda (p . args)
    (let* ([v (if (null? args) 'x (car args))]
           [F (poly-field p)]
           [cs (poly-coeffs p)]
           [n (length cs)])
      (if (and (= n 1) ((field-equal-fn F) (car cs) (field-zero F)))
          "0"
          (let str-loop ([i (- n 1)] [first #t] [res ""])
            (if (< i 0)
                res
                (let* ([c (list-ref cs i)]
                       [term (poly-term->string c i v first F)])
                  (str-loop (- i 1)
                            (and first (string=? term ""))
                            (string-append res term))))))))))""",
    "make-polynomial-ring": """(define make-polynomial-ring
  (lambda (F)
    (make-ring
      '()
      (lambda (x y) (poly-add x y))
      (lambda (x y) (poly-mul x y))
      (poly-zero-over F)
      (poly-one-over F)
      (lambda (x) (poly-neg x))
      (lambda (x y) (poly-equal? x y)))))""",
    "poly-ring": """(define poly-ring poly-field)""",
}


# =============================================================================
# Buggy Cases for bugfix family
# =============================================================================

BUGGY_CASES = [
    # ==== Easy functions (1 bug each) ====
    {
        "fn": "polynomial?",
        "buggy": """(define (polynomial? p)
  (and (pair? p)
       (eq? (car p) 'poly)))""",
        "note": "Wrong tag symbol - checks for 'poly instead of 'polynomial.",
    },
    {
        "fn": "poly-field",
        "buggy": """(define (poly-field p)
  (car p))""",
        "note": "Field is the second element (cadr), not the first (car).",
    },
    {
        "fn": "poly-degree",
        "buggy": """(define (poly-degree p)
  (length (poly-coeffs p)))""",
        "note": "Degree is length-1, not length. Off-by-one error.",
    },
    {
        "fn": "poly-leading-coeff",
        "buggy": """(define (poly-leading-coeff p)
  (let ([coeffs (poly-coeffs p)])
    (if (null? coeffs)
        (field-zero (poly-field p))
        (car coeffs))))""",
        "note": "Should get last element (list-ref (- len 1)), not first (car).",
    },
    {
        "fn": "poly-coeff-at",
        "buggy": """(define (poly-coeff-at p k)
  (let ([coeffs (poly-coeffs p)]
        [F (poly-field p)])
    (if (> k (length coeffs))
        (field-zero F)
        (list-ref coeffs k))))""",
        "note": "Comparison should be >= not >. When k equals length, should return zero.",
    },
    {
        "fn": "poly-zero?",
        "buggy": """(define (poly-zero? p)
  (let ([F (poly-field p)]
        [coeffs (poly-coeffs p)])
    ((field-equal-fn F) (car coeffs) (field-zero F))))""",
        "note": "Missing length check. A polynomial with multiple zeros is still zero, but this only checks first coeff.",
    },
    {
        "fn": "make-polynomial",
        "buggy": """(define (make-polynomial field coeffs)
  (list 'polynomial field coeffs))""",
        "note": "Missing normalization - must call poly-normalize-coeffs to strip trailing zeros.",
    },
    {
        "fn": "make-polynomial",
        "buggy": """(define (make-polynomial field coeffs)
  (doc 'export #t)
  (list 'polynomial (poly-normalize-coeffs field coeffs) field))""",
        "note": "Arguments to list are in wrong order - should be ('polynomial field coeffs).",
    },
    {
        "fn": "poly-monomial",
        "buggy": """(define (poly-monomial field coeff degree)
  (let ([zero (field-zero field)])
    (if ((field-equal-fn field) coeff zero)
        (poly-zero-over field)
        (make-polynomial field
          (append (make-list (+ degree 1) zero) (list coeff))))))""",
        "note": "Off-by-one in degree padding - uses (+ degree 1) zeros instead of degree zeros, making degree one too high.",
    },
    {
        "fn": "poly-x",
        "buggy": """(define (poly-x field)
  (poly-monomial field (field-one field) 0))""",
        "note": "Degree should be 1 for x, not 0 (which would be constant 1).",
    },
    # ==== Medium functions (2 bugs each) ====
    {
        "fn": "poly-add",
        "buggy": """(define (poly-add p1 p2)
  (let* ([F (poly-field p1)]
         [c1 (poly-coeffs p1)]
         [c2 (poly-coeffs p2)]
         [add (field-add-op F)])
    (make-polynomial F (poly-add-coeffs add c1 c2 (field-zero F)))
    (make-polynomial F c1)))""",
        "note": "Second make-polynomial shadows the first - returns p1 instead of sum.",
    },
    {
        "fn": "poly-add",
        "buggy": """(define (poly-add p1 p2)
  (let* ([F (poly-field p1)]
         [c1 (poly-coeffs p1)]
         [c2 (poly-coeffs p2)]
         [add (field-mul-op F)])
    (make-polynomial F (poly-add-coeffs add c1 c2 (field-zero F)))))""",
        "note": "Using field-mul-op instead of field-add-op - multiplies coefficients pairwise instead of adding.",
    },
    {
        "fn": "poly-neg",
        "buggy": """(define (poly-neg p)
  (let* ([F (poly-field p)]
         [neg (field-neg-fn F)])
    (map neg (poly-coeffs p))))""",
        "note": "Returns raw list instead of polynomial - missing make-polynomial wrapper.",
    },
    {
        "fn": "poly-sub",
        "buggy": """(define (poly-sub p1 p2)
  (poly-add p2 (poly-neg p1)))""",
        "note": "Arguments swapped - should be (poly-add p1 (poly-neg p2)) for p1 - p2.",
    },
    {
        "fn": "poly-scale",
        "buggy": """(define (poly-scale p c)
  (let* ([F (poly-field p)]
         [add (field-add-op F)])
    (make-polynomial F (map (lambda (a) (add c a)) (poly-coeffs p)))))""",
        "note": "Uses field-add-op instead of field-mul-op - adds scalar to each coefficient instead of multiplying.",
    },
    {
        "fn": "poly-scale",
        "buggy": """(define (poly-scale p c)
  (let* ([F (poly-field p)]
         [mul (field-mul-op F)])
    (make-polynomial F (map (lambda (a) (mul c c)) (poly-coeffs p)))))""",
        "note": "Multiplies scalar by itself instead of by each coefficient - all coefficients become c*c.",
    },
    {
        "fn": "poly-equal?",
        "buggy": """(define (poly-equal? p1 p2)
  (let* ([F (poly-field p1)]
         [eq-fn (field-equal-fn F)]
         [c1 (poly-coeffs p1)]
         [c2 (poly-coeffs p2)])
    (let loop ([l1 c1] [l2 c2])
      (or (null? l1)
          (and (eq-fn (car l1) (car l2))
               (loop (cdr l1) (cdr l2)))))))""",
        "note": "Missing length check - polynomials of different lengths can pass if shorter is prefix.",
    },
    {
        "fn": "poly-equal?",
        "buggy": """(define (poly-equal? p1 p2)
  (let* ([F (poly-field p1)]
         [eq-fn (field-equal-fn F)]
         [c1 (poly-coeffs p1)]
         [c2 (poly-coeffs p2)])
    (= (length c1) (length c2))))""",
        "note": "Only checks coefficient list lengths, not actual values - any two same-degree polynomials would compare equal.",
    },
    {
        "fn": "poly-eval",
        "buggy": """(define (poly-eval p x)
  (let* ([F (poly-field p)]
         [coeffs (poly-coeffs p)]
         [add (field-add-op F)]
         [mul (field-mul-op F)])
    (if (null? coeffs)
        (field-zero F)
        (let loop ([cs (cdr coeffs)] [acc (car coeffs)])
          (if (null? cs)
              acc
              (loop (cdr cs) (add (mul acc x) (car cs))))))))""",
        "note": "Missing reverse - Horner's method needs descending order but coeffs are ascending.",
    },
    {
        "fn": "poly-eval",
        "buggy": """(define (poly-eval p x)
  (let* ([F (poly-field p)]
         [coeffs (reverse (poly-coeffs p))]
         [add (field-add-op F)]
         [mul (field-mul-op F)])
    (if (null? coeffs)
        (field-zero F)
        (let loop ([cs (cdr coeffs)] [acc (car coeffs)])
          (if (null? cs)
              acc
              (loop (cdr cs) (mul (add acc x) (car cs))))))))""",
        "note": "Swapped add and mul in Horner step - should be (add (mul acc x) c), not (mul (add acc x) c).",
    },
    {
        "fn": "poly-derivative",
        "buggy": """(define (poly-derivative p)
  (let* ([F (poly-field p)]
         [coeffs (poly-coeffs p)])
    (if (<= (length coeffs) 1)
        (poly-zero-over F)
        (make-polynomial F
          (let loop ([cs coeffs] [k 0] [result '()])
            (if (null? cs)
                (reverse result)
                (loop (cdr cs) (+ k 1)
                      (cons (poly-scalar-mul-int F (car cs) k) result))))))))""",
        "note": "Starting k at 0 instead of 1 - derivative of constant should be 0, but this includes it.",
    },
    {
        "fn": "poly-power",
        "buggy": """(define (poly-power p n)
  (cond
    [(= n 0) (poly-one-over (poly-field p))]
    [(= n 1) p]
    [(even? n)
     (let ([half (poly-power p (/ n 2))])
       (poly-mul half half))]
    [else
     (poly-mul p (poly-power p (+ n 1)))]))""",
        "note": "Wrong recursion in odd case - should be (- n 1) not (+ n 1), causes infinite loop.",
    },
    {
        "fn": "poly->string",
        "buggy": """(define (poly->string p . opts)
  (let* ([var (if (null? opts) 'x (car opts))]
         [F (poly-field p)]
         [coeffs (poly-coeffs p)]
         [n (length coeffs)])
    (if (and (= n 1) ((field-equal-fn F) (car coeffs) (field-zero F)))
        "0"
        (let loop ([i 0] [first? #t] [result ""])
          (if (>= i n)
              result
              (let* ([c (list-ref coeffs i)]
                     [term (poly-term->string c i var first? F)])
                (loop (+ i 1)
                      (and first? (string=? term ""))
                      (string-append result term))))))))""",
        "note": "Iterating in wrong direction - should go from high degree to low for proper formatting.",
    },
    # ==== Hard functions (2 bugs each) ====
    {
        "fn": "poly-mul",
        "buggy": """(define (poly-mul p1 p2)
  (let* ([F (poly-field p1)]
         [c1 (poly-coeffs p1)]
         [c2 (poly-coeffs p2)]
         [zero (field-zero F)]
         [add (field-add-op F)]
         [mul (field-mul-op F)]
         [n1 (length c1)]
         [n2 (length c2)]
         [n (+ n1 n2 -2)])
    (if (or (poly-zero? p1) (poly-zero? p2))
        (poly-zero-over F)
        (make-polynomial F
          (let loop ([k 0] [result '()])
            (if (= k n)
                (reverse result)
                (loop (+ k 1)
                      (cons (poly-mul-coeff-at add mul c1 c2 k n1 n2 zero)
                            result))))))))""",
        "note": "Off-by-one in degree: uses (+ n1 n2 -2) instead of (+ n1 n2 -1), truncating the highest-degree term.",
    },
    {
        "fn": "poly-mul",
        "buggy": """(define (poly-mul p1 p2)
  (let* ([F (poly-field p1)]
         [c1 (poly-coeffs p1)]
         [c2 (poly-coeffs p2)]
         [zero (field-zero F)]
         [add (field-add-op F)]
         [mul (field-mul-op F)]
         [n1 (length c1)])
    (if (or (poly-zero? p1) (poly-zero? p2))
        (poly-zero-over F)
        (make-polynomial F
          (let loop ([k 0] [result '()])
            (if (= k (+ n1 n1 -1))
                (reverse result)
                (loop (+ k 1)
                      (cons (poly-mul-coeff-at add mul c1 c1 k n1 n1 zero)
                            result))))))))""",
        "note": "Uses c1 and n1 for both operands - squares p1 instead of multiplying p1*p2.",
    },
    {
        "fn": "poly-divmod",
        "buggy": """(define (poly-divmod p1 p2)
  (let* ([F (poly-field p1)]
         [d2 (poly-degree p2)])
    (if (poly-zero? p2)
        (error 'poly-divmod "division by zero polynomial")
        (poly-divmod-loop p1 p2 (poly-one-over F) F))))""",
        "note": "Initializes quotient accumulator to 1 instead of 0 - adds spurious constant to quotient.",
    },
    {
        "fn": "poly-divmod",
        "buggy": """(define (poly-divmod p1 p2)
  (let* ([F (poly-field p1)]
         [d2 (poly-degree p2)])
    (if (poly-zero? p2)
        (error 'poly-divmod "division by zero polynomial")
        (poly-divmod-loop p2 p1 (poly-zero-over F) F))))""",
        "note": "Arguments swapped in loop call - should be (poly-divmod-loop p1 p2 ...) not (p2 p1).",
    },
    {
        "fn": "poly-gcd",
        "buggy": """(define (poly-gcd p1 p2)
  (let ([F (poly-field p1)])
    (if (poly-zero? p2)
        p1
        (poly-gcd p2 (poly-mod p1 p2)))))""",
        "note": "Not making result monic - should call (poly-make-monic p1) in base case.",
    },
    {
        "fn": "poly-gcd",
        "buggy": """(define (poly-gcd p1 p2)
  (let ([F (poly-field p1)])
    (if (poly-zero? p1)
        (poly-make-monic p2)
        (poly-gcd p2 (poly-mod p1 p2)))))""",
        "note": "Wrong base case - should check p2 not p1, and return monic of p1 not p2.",
    },
    {
        "fn": "poly-make-monic",
        "buggy": """(define (poly-make-monic p)
  (if (poly-zero? p)
      p
      (let* ([F (poly-field p)]
             [lc (poly-leading-coeff p)])
        (poly-scale p lc))))""",
        "note": "Scales by leading coefficient instead of its inverse - amplifies instead of normalizing.",
    },
    {
        "fn": "poly-make-monic",
        "buggy": """(define (poly-make-monic p)
  (if (poly-zero? p)
      p
      (poly-neg p)))""",
        "note": "Negates the polynomial instead of scaling by inverse of leading coefficient.",
    },
    {
        "fn": "poly-extended-gcd",
        "buggy": """(define (poly-extended-gcd p1 p2)
  (let ([F (poly-field p1)])
    (poly-ext-gcd-loop p1 p2
                       (poly-one-over F) (poly-zero-over F)
                       (poly-zero-over F) (poly-one-over F)
                       F)))

(define (poly-ext-gcd-loop r0 r1 s0 s1 t0 t1 F)
  (if (poly-zero? r1)
      (list r0 s0 t0)
      (let* ([qr (poly-divmod r0 r1)]
             [q (car qr)]
             [r2 (cdr qr)]
             [s2 (poly-sub s0 (poly-mul q s1))]
             [t2 (poly-sub t0 (poly-mul q t1))])
        (poly-ext-gcd-loop r1 r2 s1 s2 t1 t2 F))))""",
        "note": "Not making gcd monic - should scale r0, s0, t0 by inverse of leading coeff.",
    },
    {
        "fn": "poly-extended-gcd",
        "buggy": """(define (poly-extended-gcd p1 p2)
  (let ([F (poly-field p1)])
    (poly-ext-gcd-loop p1 p2
                       (poly-zero-over F) (poly-one-over F)
                       (poly-one-over F) (poly-zero-over F)
                       F)))""",
        "note": "Initial coefficients swapped - s0 should start at 1 and t0 at 0, not vice versa.",
    },
    {
        "fn": "poly-divides?",
        "buggy": """(define (poly-divides? p1 p2)
  (poly-zero? (poly-mod p1 p2)))""",
        "note": "Arguments swapped - should be (poly-mod p2 p1) for checking if p1 divides p2.",
    },
    {
        "fn": "poly-lcm",
        "buggy": """(define (poly-lcm p1 p2)
  (if (or (poly-zero? p1) (poly-zero? p2))
      (poly-zero-over (poly-field p1))
      (poly-mul p1 p2)))""",
        "note": "Missing division by GCD - LCM should be (p1*p2)/gcd(p1,p2), not just p1*p2.",
    },
    {
        "fn": "poly-square-free",
        "buggy": """(define (poly-square-free p)
  (let ([p-prime (poly-derivative p)])
    (if (poly-zero? p-prime)
        p
        (poly-div p p-prime))))""",
        "note": "Divides by derivative instead of by gcd(p, p') - gives wrong square-free decomposition.",
    },
    {
        "fn": "poly-square-free",
        "buggy": """(define (poly-square-free p)
  (let ([p-prime (poly-derivative p)])
    (if (poly-zero? p-prime)
        p
        (poly-gcd p p-prime))))""",
        "note": "Returning GCD instead of quotient - should be (poly-div p (poly-gcd p p-prime)).",
    },
    {
        "fn": "poly-lagrange-interpolate",
        "buggy": """(define (poly-lagrange-interpolate F points)
  (if (null? points)
      (poly-zero-over F)
      (let ([xs (map car points)]
            [ys (map cdr points)])
        (poly-lagrange-sum F xs ys 1 (poly-zero-over F)))))""",
        "note": "Starting index at 1 instead of 0 - misses first point in interpolation.",
    },
    {
        "fn": "poly-lagrange-interpolate",
        "buggy": """(define (poly-lagrange-interpolate F points)
  (if (null? points)
      (poly-zero-over F)
      (poly-lagrange-sum F 
        (map car points) 
        (map car points)
        0 
        (poly-zero-over F))))""",
        "note": "Using car instead of cdr for ys - takes x values instead of y values.",
    },
    {
        "fn": "poly-newton-interpolate",
        "buggy": """(define (poly-newton-interpolate F points)
  (if (null? points)
      (poly-zero-over F)
      (let* ([xs (map car points)]
             [ys (map cdr points)]
             [coeffs (poly-divided-differences F xs ys)])
        (poly-newton-form F (reverse xs) coeffs))))""",
        "note": "Reversing xs breaks Newton form - should keep original order.",
    },
    {
        "fn": "poly-newton-interpolate",
        "buggy": """(define (poly-newton-interpolate F points)
  (if (null? points)
      (poly-zero-over F)
      (let* ([xs (map car points)]
             [ys (map cdr points)]
             [coeffs (reverse (poly-divided-differences F xs ys))])
        (poly-newton-form F xs coeffs))))""",
        "note": "Reversing coeffs is wrong - Newton form expects specific order from divided differences.",
    },
    {
        "fn": "make-polynomial-ring",
        "buggy": """(define (make-polynomial-ring F)
  (make-ring
   '()
   (lambda (x y) (poly-add x y))
   (lambda (x y) (poly-mul x y))
   (poly-one-over F)
   (poly-zero-over F)
   (lambda (x) (poly-neg x))
   (lambda (x y) (poly-equal? x y))))""",
        "note": "Zero and one elements are swapped - ring-zero is actually 1 and ring-one is actually 0.",
    },
]


# =============================================================================
# Composition Cases
# =============================================================================

COMPOSITION_CASES = [
    # ==== make-polynomial compositions ====
    {
        "fn": "make-polynomial",
        "prompt": "Create a polynomial 2 + 3x + x^2 over Q, verify its degree and leading coefficient.",
        "gt": "(let* ([p (make-polynomial Q-field '(2 3 1))]) (and (= (poly-degree p) 2) (= (poly-leading-coeff p) 1)))",
        "verify": "(equal? (let* ([p (make-polynomial Q-field '(2 3 1))]) (and (= (poly-degree p) 2) (= (poly-leading-coeff p) 1))) #t)",
        "difficulty": "medium",
    },
    {
        "fn": "make-polynomial",
        "prompt": "Create a polynomial from '(1 0 2 0 0) and verify normalization stripped trailing zeros.",
        "gt": "(let ([p (make-polynomial Q-field '(1 0 2 0 0))]) (equal? (poly-coeffs p) '(1 0 2)))",
        "verify": "(equal? (let ([p (make-polynomial Q-field '(1 0 2 0 0))]) (equal? (poly-coeffs p) '(1 0 2))) #t)",
        "difficulty": "medium",
    },
    # ==== polynomial? compositions ====
    {
        "fn": "polynomial?",
        "prompt": "Create polynomials using different constructors and verify all satisfy polynomial?.",
        "gt": "(let* ([p1 (make-polynomial Q-field '(1 2))] [p2 (poly-x Q-field)] [p3 (poly-constant Q-field 5)]) (and (polynomial? p1) (polynomial? p2) (polynomial? p3)))",
        "verify": "(equal? (let* ([p1 (make-polynomial Q-field '(1 2))] [p2 (poly-x Q-field)] [p3 (poly-constant Q-field 5)]) (and (polynomial? p1) (polynomial? p2) (polynomial? p3))) #t)",
        "difficulty": "easy",
    },
    # ==== poly-field compositions ====
    {
        "fn": "poly-field",
        "prompt": "Create polynomials over Q and verify they all have Q-field as their coefficient field.",
        "gt": "(let* ([p1 (make-polynomial Q-field '(1))] [p2 (poly-monomial Q-field 3 5)] [p3 (poly-add p1 p2)]) (and (eq? (poly-field p1) Q-field) (eq? (poly-field p2) Q-field) (eq? (poly-field p3) Q-field)))",
        "verify": "(equal? (let* ([p1 (make-polynomial Q-field '(1))] [p2 (poly-monomial Q-field 3 5)] [p3 (poly-add p1 p2)]) (and (eq? (poly-field p1) Q-field) (eq? (poly-field p2) Q-field) (eq? (poly-field p3) Q-field))) #t)",
        "difficulty": "easy",
    },
    # ==== poly-degree compositions ====
    {
        "fn": "poly-degree",
        "prompt": "Create (x+1)^3 by multiplication and verify degree is 3.",
        "gt": "(let* ([x+1 (make-polynomial Q-field '(1 1))] [cubed (poly-mul (poly-mul x+1 x+1) x+1)]) (= (poly-degree cubed) 3))",
        "verify": "(equal? (let* ([x+1 (make-polynomial Q-field '(1 1))] [cubed (poly-mul (poly-mul x+1 x+1) x+1)]) (= (poly-degree cubed) 3)) #t)",
        "difficulty": "medium",
    },
    {
        "fn": "poly-degree",
        "prompt": "Compute degree of derivative of x^4 + 2x^2 + 1.",
        "gt": "(let* ([p (make-polynomial Q-field '(1 0 2 0 1))] [dp (poly-derivative p)]) (= (poly-degree dp) 3))",
        "verify": "(equal? (let* ([p (make-polynomial Q-field '(1 0 2 0 1))] [dp (poly-derivative p)]) (= (poly-degree dp) 3)) #t)",
        "difficulty": "medium",
    },
    # ==== poly-leading-coeff compositions ====
    {
        "fn": "poly-leading-coeff",
        "prompt": "Compute leading coefficient of (2x+1)*(3x-1) and verify it equals 6.",
        "gt": "(let* ([p1 (make-polynomial Q-field '(1 2))] [p2 (make-polynomial Q-field '(-1 3))] [prod (poly-mul p1 p2)]) (= (poly-leading-coeff prod) 6))",
        "verify": "(equal? (let* ([p1 (make-polynomial Q-field '(1 2))] [p2 (make-polynomial Q-field '(-1 3))] [prod (poly-mul p1 p2)]) (= (poly-leading-coeff prod) 6)) #t)",
        "difficulty": "medium",
    },
    # ==== poly-zero? compositions ====
    {
        "fn": "poly-zero?",
        "prompt": "Verify that p - p = zero for any polynomial p.",
        "gt": "(let* ([p (make-polynomial Q-field '(1 2 3 4))]) (poly-zero? (poly-sub p p)))",
        "verify": "(equal? (let* ([p (make-polynomial Q-field '(1 2 3 4))]) (poly-zero? (poly-sub p p))) #t)",
        "difficulty": "medium",
    },
    # ==== poly-x compositions ====
    {
        "fn": "poly-x",
        "prompt": "Verify that evaluating x at point a returns a.",
        "gt": "(let* ([x (poly-x Q-field)]) (= (poly-eval x 5) 5))",
        "verify": "(equal? (let* ([x (poly-x Q-field)]) (= (poly-eval x 5) 5)) #t)",
        "difficulty": "easy",
    },
    # ==== poly-add compositions ====
    {
        "fn": "poly-add",
        "prompt": "Verify (x^2 + 1) + (x + 2) = x^2 + x + 3 by checking coefficients.",
        "gt": "(let* ([p1 (make-polynomial Q-field '(1 0 1))] [p2 (make-polynomial Q-field '(2 1))] [sum (poly-add p1 p2)]) (and (= (poly-coeff-at sum 0) 3) (= (poly-coeff-at sum 1) 1) (= (poly-coeff-at sum 2) 1)))",
        "verify": "(equal? (let* ([p1 (make-polynomial Q-field '(1 0 1))] [p2 (make-polynomial Q-field '(2 1))] [sum (poly-add p1 p2)]) (and (= (poly-coeff-at sum 0) 3) (= (poly-coeff-at sum 1) 1) (= (poly-coeff-at sum 2) 1))) #t)",
        "difficulty": "medium",
    },
    {
        "fn": "poly-add",
        "prompt": "Verify addition is commutative: p1 + p2 = p2 + p1 for random polynomials.",
        "gt": "(let* ([p1 (make-polynomial Q-field '(1 2 3))] [p2 (make-polynomial Q-field '(4 5))]) (poly-equal? (poly-add p1 p2) (poly-add p2 p1)))",
        "verify": "(equal? (let* ([p1 (make-polynomial Q-field '(1 2 3))] [p2 (make-polynomial Q-field '(4 5))]) (poly-equal? (poly-add p1 p2) (poly-add p2 p1))) #t)",
        "difficulty": "medium",
    },
    # ==== poly-mul compositions ====
    {
        "fn": "poly-mul",
        "prompt": "Verify (x-1)(x+1) = x^2 - 1 by checking all coefficients.",
        "gt": "(let* ([xm1 (make-polynomial Q-field '(-1 1))] [xp1 (make-polynomial Q-field '(1 1))] [prod (poly-mul xm1 xp1)]) (and (= (poly-coeff-at prod 0) -1) (= (poly-coeff-at prod 1) 0) (= (poly-coeff-at prod 2) 1)))",
        "verify": "(equal? (let* ([xm1 (make-polynomial Q-field '(-1 1))] [xp1 (make-polynomial Q-field '(1 1))] [prod (poly-mul xm1 xp1)]) (and (= (poly-coeff-at prod 0) -1) (= (poly-coeff-at prod 1) 0) (= (poly-coeff-at prod 2) 1))) #t)",
        "difficulty": "medium",
    },
    {
        "fn": "poly-mul",
        "prompt": "Verify distributive property: a*(b+c) = a*b + a*c.",
        "gt": "(let* ([a (make-polynomial Q-field '(1 1))] [b (make-polynomial Q-field '(2 3))] [c (make-polynomial Q-field '(4 5))]) (poly-equal? (poly-mul a (poly-add b c)) (poly-add (poly-mul a b) (poly-mul a c))))",
        "verify": "(equal? (let* ([a (make-polynomial Q-field '(1 1))] [b (make-polynomial Q-field '(2 3))] [c (make-polynomial Q-field '(4 5))]) (poly-equal? (poly-mul a (poly-add b c)) (poly-add (poly-mul a b) (poly-mul a c)))) #t)",
        "difficulty": "hard",
    },
    # ==== poly-power compositions ====
    {
        "fn": "poly-power",
        "prompt": "Verify (x+1)^4 has correct coefficients using binomial theorem.",
        "gt": "(let* ([xp1 (make-polynomial Q-field '(1 1))] [p4 (poly-power xp1 4)]) (and (= (poly-coeff-at p4 0) 1) (= (poly-coeff-at p4 1) 4) (= (poly-coeff-at p4 2) 6) (= (poly-coeff-at p4 3) 4) (= (poly-coeff-at p4 4) 1)))",
        "verify": "(equal? (let* ([xp1 (make-polynomial Q-field '(1 1))] [p4 (poly-power xp1 4)]) (and (= (poly-coeff-at p4 0) 1) (= (poly-coeff-at p4 1) 4) (= (poly-coeff-at p4 2) 6) (= (poly-coeff-at p4 3) 4) (= (poly-coeff-at p4 4) 1))) #t)",
        "difficulty": "hard",
    },
    # ==== poly-divmod compositions ====
    {
        "fn": "poly-divmod",
        "prompt": "Divide x^3 - 1 by x - 1 and verify remainder is zero and quotient is x^2 + x + 1.",
        "gt": "(let* ([p1 (make-polynomial Q-field '(-1 0 0 1))] [p2 (make-polynomial Q-field '(-1 1))] [result (poly-divmod p1 p2)] [q (car result)] [r (cdr result)]) (and (poly-zero? r) (= (poly-degree q) 2) (= (poly-coeff-at q 0) 1) (= (poly-coeff-at q 1) 1) (= (poly-coeff-at q 2) 1)))",
        "verify": "(equal? (let* ([p1 (make-polynomial Q-field '(-1 0 0 1))] [p2 (make-polynomial Q-field '(-1 1))] [result (poly-divmod p1 p2)] [q (car result)] [r (cdr result)]) (and (poly-zero? r) (= (poly-degree q) 2) (= (poly-coeff-at q 0) 1) (= (poly-coeff-at q 1) 1) (= (poly-coeff-at q 2) 1))) #t)",
        "difficulty": "hard",
    },
    {
        "fn": "poly-divmod",
        "prompt": "Verify division algorithm: for random p1, p2, we have p1 = q*p2 + r with deg(r) < deg(p2).",
        "gt": "(let* ([p1 (make-polynomial Q-field '(1 2 3 4))] [p2 (make-polynomial Q-field '(1 1))] [result (poly-divmod p1 p2)] [q (car result)] [r (cdr result)] [reconstructed (poly-add (poly-mul q p2) r)]) (and (poly-equal? p1 reconstructed) (< (poly-degree r) (poly-degree p2))))",
        "verify": "(equal? (let* ([p1 (make-polynomial Q-field '(1 2 3 4))] [p2 (make-polynomial Q-field '(1 1))] [result (poly-divmod p1 p2)] [q (car result)] [r (cdr result)] [reconstructed (poly-add (poly-mul q p2) r)]) (and (poly-equal? p1 reconstructed) (< (poly-degree r) (poly-degree p2)))) #t)",
        "difficulty": "hard",
    },
    # ==== poly-gcd compositions ====
    {
        "fn": "poly-gcd",
        "prompt": "Verify GCD of (x+1)^2*(x+2) and (x+1)*(x+2)^2 is (x+1)*(x+2) up to monic scaling.",
        "gt": "(let* ([xp1 (make-polynomial Q-field '(1 1))] [xp2 (make-polynomial Q-field '(2 1))] [p1 (poly-mul (poly-mul xp1 xp1) xp2)] [p2 (poly-mul xp1 (poly-mul xp2 xp2))] [g (poly-gcd p1 p2)] [expected (poly-mul xp1 xp2)]) (poly-equal? g (poly-make-monic expected)))",
        "verify": "(equal? (let* ([xp1 (make-polynomial Q-field '(1 1))] [xp2 (make-polynomial Q-field '(2 1))] [p1 (poly-mul (poly-mul xp1 xp1) xp2)] [p2 (poly-mul xp1 (poly-mul xp2 xp2))] [g (poly-gcd p1 p2)] [expected (poly-mul xp1 xp2)]) (poly-equal? g (poly-make-monic expected))) #t)",
        "difficulty": "hard",
    },
    {
        "fn": "poly-gcd",
        "prompt": "Verify GCD of any polynomial with itself is the monic version of itself.",
        "gt": "(let* ([p (make-polynomial Q-field '(2 4 2))] [g (poly-gcd p p)]) (poly-equal? g (poly-make-monic p)))",
        "verify": "(equal? (let* ([p (make-polynomial Q-field '(2 4 2))] [g (poly-gcd p p)]) (poly-equal? g (poly-make-monic p))) #t)",
        "difficulty": "medium",
    },
    # ==== poly-extended-gcd compositions ====
    {
        "fn": "poly-extended-gcd",
        "prompt": "Verify Bezout identity: for GCD result (g,s,t), check g = s*p1 + t*p2.",
        "gt": "(let* ([p1 (make-polynomial Q-field '(1 2 1))] [p2 (make-polynomial Q-field '(1 1))] [result (poly-extended-gcd p1 p2)] [g (car result)] [s (cadr result)] [t (caddr result)] [lhs (poly-add (poly-mul s p1) (poly-mul t p2))]) (poly-equal? g lhs))",
        "verify": "(equal? (let* ([p1 (make-polynomial Q-field '(1 2 1))] [p2 (make-polynomial Q-field '(1 1))] [result (poly-extended-gcd p1 p2)] [g (car result)] [s (cadr result)] [t (caddr result)] [lhs (poly-add (poly-mul s p1) (poly-mul t p2))]) (poly-equal? g lhs)) #t)",
        "difficulty": "hard",
    },
    # ==== poly-eval compositions ====
    {
        "fn": "poly-eval",
        "prompt": "Verify evaluation of (x+1)^3 at x=2 equals 27.",
        "gt": "(let* ([xp1 (make-polynomial Q-field '(1 1))] [cubed (poly-power xp1 3)]) (= (poly-eval cubed 2) 27))",
        "verify": "(equal? (let* ([xp1 (make-polynomial Q-field '(1 1))] [cubed (poly-power xp1 3)]) (= (poly-eval cubed 2) 27)) #t)",
        "difficulty": "medium",
    },
    {
        "fn": "poly-eval",
        "prompt": "Verify polynomial evaluation is homomorphism: eval(p1*p2, a) = eval(p1,a) * eval(p2,a).",
        "gt": "(let* ([p1 (make-polynomial Q-field '(1 1))] [p2 (make-polynomial Q-field '(2 3))] [prod (poly-mul p1 p2)] [a 5]) (= (poly-eval prod a) (* (poly-eval p1 a) (poly-eval p2 a))))",
        "verify": "(equal? (let* ([p1 (make-polynomial Q-field '(1 1))] [p2 (make-polynomial Q-field '(2 3))] [prod (poly-mul p1 p2)] [a 5]) (= (poly-eval prod a) (* (poly-eval p1 a) (poly-eval p2 a)))) #t)",
        "difficulty": "hard",
    },
    # ==== poly-derivative compositions ====
    {
        "fn": "poly-derivative",
        "prompt": "Verify product rule: d/dx(p1*p2) = p1'*p2 + p1*p2'.",
        "gt": "(let* ([p1 (make-polynomial Q-field '(1 1))] [p2 (make-polynomial Q-field '(2 3))] [prod (poly-mul p1 p2)] [d-prod (poly-derivative prod)] [lhs (poly-add (poly-mul (poly-derivative p1) p2) (poly-mul p1 (poly-derivative p2)))]) (poly-equal? d-prod lhs))",
        "verify": "(equal? (let* ([p1 (make-polynomial Q-field '(1 1))] [p2 (make-polynomial Q-field '(2 3))] [prod (poly-mul p1 p2)] [d-prod (poly-derivative prod)] [lhs (poly-add (poly-mul (poly-derivative p1) p2) (poly-mul p1 (poly-derivative p2)))]) (poly-equal? d-prod lhs)) #t)",
        "difficulty": "hard",
    },
    {
        "fn": "poly-derivative",
        "prompt": "Verify derivative of constant is zero.",
        "gt": "(let* ([c (poly-constant Q-field 42)]) (poly-zero? (poly-derivative c)))",
        "verify": "(equal? (let* ([c (poly-constant Q-field 42)]) (poly-zero? (poly-derivative c))) #t)",
        "difficulty": "easy",
    },
    # ==== poly-square-free compositions ====
    {
        "fn": "poly-square-free",
        "prompt": "Verify square-free part of (x+1)^3 is (x+1).",
        "gt": "(let* ([xp1 (make-polynomial Q-field '(1 1))] [cubed (poly-power xp1 3)] [sf (poly-square-free cubed)]) (poly-equal? sf (poly-make-monic xp1)))",
        "verify": "(equal? (let* ([xp1 (make-polynomial Q-field '(1 1))] [cubed (poly-power xp1 3)] [sf (poly-square-free cubed)]) (poly-equal? sf (poly-make-monic xp1))) #t)",
        "difficulty": "hard",
    },
    # ==== poly-lagrange-interpolate compositions ====
    {
        "fn": "poly-lagrange-interpolate",
        "prompt": "Interpolate through (0,1), (1,0), (2,1) and verify all three points.",
        "gt": "(let* ([points '((0 . 1) (1 . 0) (2 . 1))] [p (poly-lagrange-interpolate Q-field points)]) (and (= (poly-eval p 0) 1) (= (poly-eval p 1) 0) (= (poly-eval p 2) 1)))",
        "verify": "(equal? (let* ([points '((0 . 1) (1 . 0) (2 . 1))] [p (poly-lagrange-interpolate Q-field points)]) (and (= (poly-eval p 0) 1) (= (poly-eval p 1) 0) (= (poly-eval p 2) 1))) #t)",
        "difficulty": "medium",
    },
    {
        "fn": "poly-lagrange-interpolate",
        "prompt": "Interpolate linear function through two points and verify linearity.",
        "gt": "(let* ([points '((0 . 3) (1 . 5))] [p (poly-lagrange-interpolate Q-field points)]) (and (= (poly-degree p) 1) (= (poly-eval p 0) 3) (= (poly-eval p 1) 5) (= (poly-eval p 2) 7)))",
        "verify": "(equal? (let* ([points '((0 . 3) (1 . 5))] [p (poly-lagrange-interpolate Q-field points)]) (and (= (poly-degree p) 1) (= (poly-eval p 0) 3) (= (poly-eval p 1) 5) (= (poly-eval p 2) 7))) #t)",
        "difficulty": "medium",
    },
    # ==== poly-newton-interpolate compositions ====
    {
        "fn": "poly-newton-interpolate",
        "prompt": "Newton interpolate through (0,1), (1,2), (2,3), (3,5) and verify all points.",
        "gt": "(let* ([points '((0 . 1) (1 . 2) (2 . 3) (3 . 5))] [p (poly-newton-interpolate Q-field points)]) (and (= (poly-eval p 0) 1) (= (poly-eval p 1) 2) (= (poly-eval p 2) 3) (= (poly-eval p 3) 5)))",
        "verify": "(equal? (let* ([points '((0 . 1) (1 . 2) (2 . 3) (3 . 5))] [p (poly-newton-interpolate Q-field points)]) (and (= (poly-eval p 0) 1) (= (poly-eval p 1) 2) (= (poly-eval p 2) 3) (= (poly-eval p 3) 5))) #t)",
        "difficulty": "hard",
    },
    # ==== poly->string compositions ====
    {
        "fn": "poly->string",
        "prompt": "Convert x^2 + 2x + 1 to string and verify format.",
        "gt": "(let* ([p (make-polynomial Q-field '(1 2 1))]) (string=? (poly->string p) \"x^2 + 2x + 1\"))",
        "verify": "(equal? (let* ([p (make-polynomial Q-field '(1 2 1))]) (string=? (poly->string p) \"x^2 + 2x + 1\")) #t)",
        "difficulty": "easy",
    },
    # ==== make-polynomial-ring compositions ====
    {
        "fn": "make-polynomial-ring",
        "prompt": "Create Q[x] polynomial ring and verify ring axioms: a + 0 = a, a * 1 = a.",
        "gt": "(let* ([R (make-polynomial-ring Q-field)] [a (make-polynomial Q-field '(2 3 1))] [zero (ring-zero R)] [one (ring-one R)]) (and (poly-equal? (ring-add R a zero) a) (poly-equal? (ring-mul R a one) a)))",
        "verify": "(equal? (let* ([R (make-polynomial-ring Q-field)] [a (make-polynomial Q-field '(2 3 1))] [zero (ring-zero R)] [one (ring-one R)]) (and (poly-equal? (ring-add R a zero) a) (poly-equal? (ring-mul R a one) a))) #t)",
        "difficulty": "medium",
    },
    {
        "fn": "make-polynomial-ring",
        "prompt": "Verify polynomial ring has correct additive inverses.",
        "gt": "(let* ([R (make-polynomial-ring Q-field)] [a (make-polynomial Q-field '(1 2 3))]) (poly-zero? (ring-add R a (ring-neg R a))))",
        "verify": "(equal? (let* ([R (make-polynomial-ring Q-field)] [a (make-polynomial Q-field '(1 2 3))]) (poly-zero? (ring-add R a (ring-neg R a)))) #t)",
        "difficulty": "medium",
    },
    # ==== poly-ring (alias) compositions ====
    {
        "fn": "poly-ring",
        "prompt": "Verify poly-ring is alias for poly-field by comparing results.",
        "gt": "(let* ([p (make-polynomial Q-field '(1 2 3))]) (eq? (poly-ring p) (poly-field p)))",
        "verify": "(equal? (let* ([p (make-polynomial Q-field '(1 2 3))]) (eq? (poly-ring p) (poly-field p))) #t)",
        "difficulty": "easy",
    },
]


# =============================================================================
# Main Generation Code
# =============================================================================

REQUIRED_KEYS = [
    "id",
    "family",
    "category",
    "difficulty",
    "source_module",
    "source_test",
    "source_function",
    "prompt_body",
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
    sid = f"algebra_poly_{family}_{family_counter[family]:03d}"
    sample = {
        "id": sid,
        "family": family,
        "category": category,
        "difficulty": difficulty,
        "source_module": SOURCE_MODULE,
        "source_test": SOURCE_TEST,
        "source_function": source_function,
        "prompt_body": prompt.strip(),
        "ground_truth": ground_truth.strip(),
        "verify_expr": verify_expr.strip(),
        "tags": tags,
    }
    for key in REQUIRED_KEYS:
        if key not in sample:
            raise ValueError(f"missing key {key}")
    samples.append(sample)


def def_verify(fn: str) -> str:
    return VERIFY_BY_FUNCTION[fn].strip()


def bump_difficulty(level: str, delta: int) -> str:
    idx = DIFFICULTY_INDEX[level] + delta
    idx = max(0, min(idx, len(DIFFICULTY_LEVELS) - 1))
    return DIFFICULTY_LEVELS[idx]


def task_difficulty(fn: str, family: str, task_kind: str, override: str | None = None) -> str:
    if override:
        return override

    base = BASE_DIFFICULTY[fn]
    if family == "spec_to_code":
        if task_kind == "skeleton":
            return bump_difficulty(base, -1)
        if task_kind == "contract":
            return bump_difficulty(base, +1)
        return base

    if family == "translation":
        if task_kind == "chez":
            return bump_difficulty(base, -1)
        if task_kind == "excerpt":
            return base
        return base

    if family == "bugfix":
        return base

    return base


FUNCTION_SECTION = {
    "polynomial?": "predicates",
    "poly-field": "accessors",
    "poly-coeffs": "accessors",
    "poly-degree": "accessors",
    "poly-leading-coeff": "accessors",
    "poly-coeff-at": "accessors",
    "poly-zero?": "predicates",
    "make-polynomial": "constructors",
    "poly-zero-over": "constructors",
    "poly-one-over": "constructors",
    "poly-constant": "constructors",
    "poly-monomial": "constructors",
    "poly-x": "constructors",
    "poly-add": "arithmetic",
    "poly-neg": "arithmetic",
    "poly-sub": "arithmetic",
    "poly-scale": "arithmetic",
    "poly-mul": "arithmetic",
    "poly-power": "arithmetic",
    "poly-equal?": "equality",
    "poly-divmod": "division",
    "poly-div": "division",
    "poly-mod": "division",
    "poly-divides?": "division",
    "poly-gcd": "gcd",
    "poly-make-monic": "gcd",
    "poly-extended-gcd": "gcd",
    "poly-lcm": "gcd",
    "poly-eval": "evaluation",
    "poly-derivative": "calculus",
    "poly-square-free": "factorization",
    "poly-square-free-factorization": "factorization",
    "poly-lagrange-interpolate": "interpolation",
    "poly-newton-interpolate": "interpolation",
    "poly->string": "display",
    "make-polynomial-ring": "ring",
    "poly-ring": "compatibility",
}


def make_source_excerpt(fn: str, snippet: str) -> str:
    section = FUNCTION_SECTION[fn]
    indented = "\n".join(f"  {line}" for line in snippet.splitlines())
    return (
        ";;; lattice/algebra/polynomial.ss excerpt\n"
        "(require 'prelude)\n"
        "(require 'field)\n"
        "\n"
        "(doc 'module 'polynomial)\n"
        f"(doc 'section '{section})\n"
        "\n"
        "(define (poly-add-coeffs add c1 c2 zero) ...)\n"
        "\n"
        f"{indented}\n"
    )


# -----------------------------------------------------------------------------
# Family 1: spec_to_code (3 variants × 36 functions = 108 samples)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    spec = FUNCTION_SPECS[fn]
    if isinstance(spec, dict):
        spec_text = f"{spec['signature']} - {spec['description']}"
    else:
        spec_text = str(spec)

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=task_difficulty(fn, "spec_to_code", "direct"),
        source_function=fn,
        prompt=f"""Implement this polynomial algebra function in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {spec_text}

Write exactly one Scheme definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "algebra", "polynomial", "spec-to-code", fn],
    )

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=task_difficulty(fn, "spec_to_code", "skeleton"),
        source_function=fn,
        prompt=f"""Complete this Fold Scheme skeleton.

```scheme
{SKELETONS[fn]}
```

Replace `<TODO>` and return only the completed definition for `{fn}`.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "algebra", "polynomial", "skeleton-completion", fn],
    )

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=task_difficulty(fn, "spec_to_code", "contract"),
        source_function=fn,
        prompt=f"""Implement `{fn}` from this contract.

Module: `{SOURCE_MODULE}`
Contract focus: {spec_text}

Requirements:
1. Keep the exact function name/signature.
2. Preserve polynomial algebra semantics and edge behavior.
3. Return only one production-ready definition.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "algebra", "polynomial", "contract-implementation", fn],
    )


# -----------------------------------------------------------------------------
# Family 2: translation (3 variants × 36 functions = 108 samples)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    add_sample(
        family="translation",
        category="translation",
        difficulty=task_difficulty(fn, "translation", "python"),
        source_function=fn,
        prompt=f"""Translate this Python function into Fold-native Scheme.
Preserve behavior and keep the function name `{fn}`.
Return only the Scheme definition.

```python
{PYTHON_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "algebra", "polynomial", "python-to-scheme", fn],
    )

    add_sample(
        family="translation",
        category="translation",
        difficulty=task_difficulty(fn, "translation", "chez"),
        source_function=fn,
        prompt=f"""Convert this Chez-style snippet to canonical Fold style.
The target function must be named `{fn}`.
Return only the final Fold definition.

```scheme
{CHEZ_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "algebra", "polynomial", "chez-to-fold", fn],
    )

    add_sample(
        family="translation",
        category="translation",
        difficulty=task_difficulty(fn, "translation", "excerpt"),
        source_function=fn,
        prompt=f"""Extract and translate the target function from this source-style module excerpt.
Return only a single Fold definition for `{fn}`.
Drop metadata doc forms from the output and keep executable behavior unchanged.

```scheme
{make_source_excerpt(fn, CHEZ_SNIPPETS[fn])}
```""",
        ground_truth=DOC_FREE_DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "algebra", "polynomial", "source-excerpt-to-fold", "doc-free-target", fn],
    )


# -----------------------------------------------------------------------------
# Family 3: bugfix (52 buggy cases)
# -----------------------------------------------------------------------------
for case in BUGGY_CASES:
    fn = case["fn"]
    add_sample(
        family="bugfix",
        category="debugging",
        difficulty=task_difficulty(fn, "bugfix", "bugfix", str(case.get("difficulty", "")) or None),
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
        tags=["tier0", "algebra", "polynomial", "bugfix", fn],
    )


# -----------------------------------------------------------------------------
# Family 4: composition/use (47 composition cases)
# -----------------------------------------------------------------------------
for case in COMPOSITION_CASES:
    fn = case["fn"]
    composition_prompt = (
        f"{case['prompt'].rstrip()}\n\n"
        f"Ensure `{fn}` is part of the composed solution.\n"
        "Return only the final Fold expression."
    )
    add_sample(
        family="composition",
        category="usage",
        difficulty=case["difficulty"],
        source_function=fn,
        prompt=composition_prompt,
        ground_truth=case["gt"],
        verify_expr=case["verify"],
        tags=["tier0", "algebra", "polynomial", "composition", fn],
    )


# -----------------------------------------------------------------------------
# Diversification and Output
# -----------------------------------------------------------------------------


def diversify_prompts(samples: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Call the DSL diversification tool if available."""
    if not SFT_GENERATOR_PATH.exists():
        return samples

    # Write pre-diversify samples
    with PRE_DIVERSIFY_PATH.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Call diversification
    try:
        proc = subprocess.run(
            ["scheme", "--quiet", "--script", str(SFT_GENERATOR_PATH),
             "--input", str(PRE_DIVERSIFY_PATH),
             "--output", str(ALL_PATH),
             "--diversify-prompts"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode == 0 and ALL_PATH.exists():
            diversified = []
            with ALL_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        diversified.append(json.loads(line))
            return diversified
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"Diversification failed or timed out: {e}", file=sys.stderr)

    return samples


def assign_splits(samples: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Assign train/eval splits using leakage-aware algorithm."""
    eval_min_by_family = {
        "spec_to_code": 3,
        "translation": 3,
        "bugfix": 2,
        "composition": 5,
    }

    # NOTE: enforce_source_function_coverage=False because this dataset's
    # leakage components are large (6-8 samples each, since all families for
    # a given source function share ground_truth/verify_expr). Enforcing
    # coverage forces one component per source function into eval, which
    # overshoots the 18% target by 3-4x.
    eval_ids = compute_leakage_aware_eval_ids(
        samples,
        eval_ratio=0.18,
        eval_min_by_family=eval_min_by_family,
        enforce_source_function_coverage=False,
    )

    for s in samples:
        s["split"] = "eval" if s["id"] in eval_ids else "train"
        # Generate final prompt with split info
        s["prompt"] = s["prompt_body"]

    return samples


def main() -> int:
    print(f"Generated {len(samples)} samples before diversification")

    # Diversify prompts
    diversified = diversify_prompts(samples)
    print(f"After diversification: {len(diversified)} samples")

    # Assign splits
    final_samples = assign_splits(diversified)

    # Write outputs
    train_samples = [s for s in final_samples if s["split"] == "train"]
    eval_samples = [s for s in final_samples if s["split"] == "eval"]

    with ALL_PATH.open("w", encoding="utf-8") as f:
        for s in final_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with TRAIN_PATH.open("w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with EVAL_PATH.open("w", encoding="utf-8") as f:
        for s in eval_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Write summary
    summary = {
        "total": len(final_samples),
        "train": len(train_samples),
        "eval": len(eval_samples),
        "families": dict(sorted(Counter(str(s["family"]) for s in final_samples).items())),
        "difficulty": dict(sorted(Counter(str(s["difficulty"]) for s in final_samples).items())),
        "source_functions": len({str(s["source_function"]) for s in final_samples}),
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {len(final_samples)} samples to {OUT_DIR}")
    print(f"  Train: {len(train_samples)}")
    print(f"  Eval: {len(eval_samples)}")
    print(f"Summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
