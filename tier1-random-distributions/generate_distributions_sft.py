#!/usr/bin/env python3
"""Generate Tier-1 random distributions SFT samples for lattice/random/distributions.ss."""

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

SOURCE_MODULE = "lattice/random/distributions.ss"
SOURCE_TEST = "lattice/random/test-distributions.ss"

SOURCE_DEFS: Dict[str, str] = {
    "horner-eval": """(define (horner-eval coeffs x)
  (fold-right (lambda (c acc) (+ c (* acc x))) 0 coeffs))""",
    "safe-log": """(define (safe-log u)
  (log-num (max u 1e-300)))""",
    "standard-normal-cdf": """(define (standard-normal-cdf x)
  (let* ([sign (if (< x 0) -1 1)]
         [x (abs x)]
         [t (/ 1 (+ 1 (* *erfc-p* (/ x (sqrt 2)))))]
         [y (- 1 (* (horner-eval *erfc-coeffs* t) t (exp-num (* -0.5 x x))))])
    (* 0.5 (+ 1 (* sign y)))))""",
    "standard-normal-quantile": """(define (standard-normal-quantile p)
  (if (or (<= p 0) (>= p 1))
      (error 'standard-normal-quantile "p must be in (0,1)" p)
      (let* ([sign (if (< p 0.5) -1 1)]
             [p* (if (< p 0.5) p (- 1 p))]
             [t (sqrt (* -2 (log-num p*)))])
        (* sign (- t (/ (horner-eval *quantile-num-coeffs* t)
                        (horner-eval *quantile-den-coeffs* t)))))))""",
    "uniform-cdf": """(define (uniform-cdf x a b)
  (cond
    [(< x a) 0]
    [(> x b) 1]
    [else (/ (- x a) (- b a))]))""",
    "exponential-quantile": """(define (exponential-quantile p rate)
  (if (or (< p 0) (> p 1))
      (error 'exponential-quantile "p must be in [0,1]" p)
      (if (= p 1) +inf.0 (/ (- (log-num (- 1 p))) rate))))""",
    "poisson-pmf": """(define (poisson-pmf k rate)
  (if (< k 0)
      0
      (/ (* (expt rate k) (exp-num (- rate)))
         (factorial k))))""",
    "binomial-pmf": """(define (binomial-pmf k n p)
  (if (or (< k 0) (> k n))
      0
      (* (binomial-coeff n k)
         (expt p k)
         (expt (- 1 p) (- n k)))))""",
}

SUPPORT_DEFS: Dict[str, str] = {
    "log-num": """(define (log-num x)
  (log x))""",
    "exp-num": """(define (exp-num x)
  (exp x))""",
    "pi-value": """(define (pi-value)
  (acos -1))""",
    "approx=?": """(define (approx=? a b tol)
  (< (abs (- a b)) tol))""",
    "factorial": """(define (factorial n)
  (if (<= n 1) 1 (* n (factorial (- n 1)))))""",
    "binomial-coeff": """(define (binomial-coeff n k)
  (if (or (< k 0) (> k n))
      0
      (/ (factorial n) (* (factorial k) (factorial (- n k))))))""",
    "uniform-quantile": """(define (uniform-quantile p a b)
  (if (or (< p 0) (> p 1))
      (error 'uniform-quantile "p must be in [0,1]" p)
      (+ a (* p (- b a)))))""",
    "exponential-cdf": """(define (exponential-cdf x rate)
  (if (< x 0) 0 (- 1 (exp-num (* (- rate) x)))))""",
    "*erfc-coeffs*": """(define *erfc-coeffs* '(0.254829592 -0.284496736 1.421413741 -1.453152027 1.061405429))""",
    "*erfc-p*": """(define *erfc-p* 0.3275911)""",
    "*quantile-num-coeffs*": """(define *quantile-num-coeffs* '(2.515517 0.802853 0.010328))""",
    "*quantile-den-coeffs*": """(define *quantile-den-coeffs* '(1 1.432788 0.189269 0.001308))""",
}

ALL_DEFS: Dict[str, str] = {**SUPPORT_DEFS, **SOURCE_DEFS}

FUNCTION_ORDER = [
    "horner-eval",
    "safe-log",
    "standard-normal-cdf",
    "standard-normal-quantile",
    "uniform-cdf",
    "exponential-quantile",
    "poisson-pmf",
    "binomial-pmf",
]

SUPPORT_ORDER = [
    "log-num",
    "exp-num",
    "pi-value",
    "approx=?",
    "factorial",
    "binomial-coeff",
    "uniform-quantile",
    "exponential-cdf",
    "*erfc-coeffs*",
    "*erfc-p*",
    "*quantile-num-coeffs*",
    "*quantile-den-coeffs*",
]

DEPENDS: Dict[str, List[str]] = {
    "log-num": [],
    "exp-num": [],
    "pi-value": [],
    "approx=?": [],
    "factorial": [],
    "binomial-coeff": ["factorial"],
    "uniform-quantile": [],
    "exponential-cdf": ["exp-num"],
    "*erfc-coeffs*": [],
    "*erfc-p*": [],
    "*quantile-num-coeffs*": [],
    "*quantile-den-coeffs*": [],
    "horner-eval": [],
    "safe-log": ["log-num"],
    "standard-normal-cdf": ["horner-eval", "*erfc-coeffs*", "*erfc-p*", "exp-num"],
    "standard-normal-quantile": [
        "horner-eval",
        "*quantile-num-coeffs*",
        "*quantile-den-coeffs*",
        "log-num",
    ],
    "uniform-cdf": [],
    "exponential-quantile": ["log-num"],
    "poisson-pmf": ["factorial", "exp-num"],
    "binomial-pmf": ["binomial-coeff"],
}

FUNCTION_SPECS = {
    "horner-eval": "Evaluate polynomial coefficients a0..an at x using Horner's method.",
    "safe-log": "Compute natural log while clamping input at 1e-300 to avoid log(0).",
    "standard-normal-cdf": "Approximate the standard normal CDF using erfc-style polynomial approximation.",
    "standard-normal-quantile": "Approximate inverse standard normal CDF; reject p outside (0,1).",
    "uniform-cdf": "Compute Uniform(a,b) CDF with correct piecewise behavior below/inside/above interval.",
    "exponential-quantile": "Compute Exp(rate) quantile with domain checks and p=1 -> +inf.0.",
    "poisson-pmf": "Compute Poisson(rate) PMF for nonnegative k; return 0 for k<0.",
    "binomial-pmf": "Compute Binomial(n,p) PMF and return 0 for out-of-range k.",
}

SKELETONS = {
    "horner-eval": """(define (horner-eval coeffs x)
  ;; TODO: use Horner's method over coeffs
  <TODO>)""",
    "safe-log": """(define (safe-log u)
  ;; TODO: guard against zero before applying log
  <TODO>)""",
    "standard-normal-cdf": """(define (standard-normal-cdf x)
  ;; TODO: compute CDF with sign symmetry and erfc-style approximation
  <TODO>)""",
    "standard-normal-quantile": """(define (standard-normal-quantile p)
  ;; TODO: enforce p in (0,1), then apply rational approximation
  <TODO>)""",
    "uniform-cdf": """(define (uniform-cdf x a b)
  ;; TODO: piecewise CDF on [a,b]
  <TODO>)""",
    "exponential-quantile": """(define (exponential-quantile p rate)
  ;; TODO: validate p in [0,1], handle p=1, compute inverse CDF
  <TODO>)""",
    "poisson-pmf": """(define (poisson-pmf k rate)
  ;; TODO: return 0 for k<0; otherwise Poisson PMF formula
  <TODO>)""",
    "binomial-pmf": """(define (binomial-pmf k n p)
  ;; TODO: handle out-of-range k and compute binomial PMF
  <TODO>)""",
}

DIFFICULTY = {
    "horner-eval": "easy",
    "safe-log": "easy",
    "standard-normal-cdf": "hard",
    "standard-normal-quantile": "hard",
    "uniform-cdf": "easy",
    "exponential-quantile": "medium",
    "poisson-pmf": "medium",
    "binomial-pmf": "medium",
}

VERIFY_BY_FUNCTION = {
    "horner-eval": """(and
  (= (horner-eval '(1 2 3) 2) 17)
  (= (horner-eval '(5) 9) 5)
  (= (horner-eval '() 10) 0)
  (= (horner-eval '(3 -1 2) -2) 13))""",
    "safe-log": """(and
  (approx=? (safe-log 1.0) 0.0 1e-12)
  (approx=? (safe-log 0.0) (log-num 1e-300) 1e-12)
  (approx=? (safe-log 2.5) (log-num 2.5) 1e-12)
  (<= (safe-log 1e-400) (safe-log 1e-300)))""",
    "standard-normal-cdf": """(let ([c0 (standard-normal-cdf 0.0)]
       [c1 (standard-normal-cdf 1.96)]
       [cm (standard-normal-cdf -1.5)]
       [cp (standard-normal-cdf 1.5)])
  (and (approx=? c0 0.5 0.001)
       (approx=? c1 0.975 0.02)
       (approx=? (+ cm cp) 1.0 0.02)
       (< cm c0)
       (< c0 cp)))""",
    "standard-normal-quantile": """(let* ([q50 (standard-normal-quantile 0.5)]
       [q975 (standard-normal-quantile 0.975)]
       [q20 (standard-normal-quantile 0.2)]
       [err? (guard (ex [else #t]) (begin (standard-normal-quantile 1.0) #f))])
  (and (approx=? q50 0.0 0.03)
       (approx=? q975 1.96 0.12)
       (approx=? (standard-normal-cdf q20) 0.2 0.03)
       err?))""",
    "uniform-cdf": """(let ([a 2.0] [b 6.0])
  (and (= (uniform-cdf 1.0 a b) 0)
       (= (uniform-cdf 7.0 a b) 1)
       (approx=? (uniform-cdf 4.0 a b) 0.5 1e-12)
       (approx=? (uniform-cdf a a b) 0.0 1e-12)
       (approx=? (uniform-cdf b a b) 1.0 1e-12)))""",
    "exponential-quantile": """(let* ([q (exponential-quantile 0.5 2.0)]
       [q0 (exponential-quantile 0.0 3.0)]
       [q1 (exponential-quantile 1.0 4.0)]
       [back (exponential-cdf q 2.0)]
       [err? (guard (ex [else #t]) (begin (exponential-quantile 1.1 2.0) #f))])
  (and (approx=? q (/ (log-num 2.0) 2.0) 1e-9)
       (approx=? q0 0.0 1e-12)
       (eqv? q1 +inf.0)
       (approx=? back 0.5 1e-9)
       err?))""",
    "poisson-pmf": """(let ([p0 (poisson-pmf 0 2.0)]
       [p1 (poisson-pmf 1 2.0)]
       [p2 (poisson-pmf 2 2.0)])
  (and (approx=? p0 (exp-num -2.0) 1e-12)
       (approx=? (/ p2 p1) 1.0 1e-12)
       (= (poisson-pmf -1 2.0) 0)
       (> p1 p0)))""",
    "binomial-pmf": """(let* ([n 5]
       [p 0.3]
       [s (+ (binomial-pmf 0 n p)
             (binomial-pmf 1 n p)
             (binomial-pmf 2 n p)
             (binomial-pmf 3 n p)
             (binomial-pmf 4 n p)
             (binomial-pmf 5 n p))])
  (and (= (binomial-pmf -1 n p) 0)
       (= (binomial-pmf 6 n p) 0)
       (approx=? s 1.0 1e-9)
       (approx=? (binomial-pmf 2 5 0.4)
                 (* 10 (expt 0.4 2) (expt 0.6 3))
                 1e-12)))""",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")

TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "horner-eval": """def horner_eval(coeffs, x):
    acc = 0
    for c in reversed(coeffs):
        acc = c + acc * x
    return acc""",
    "safe-log": """def safe_log(u):
    return log_num(max(u, 1e-300))""",
    "standard-normal-cdf": """def standard_normal_cdf(x):
    sign = -1 if x < 0 else 1
    x = abs(x)
    t = 1 / (1 + ERFC_P * (x / (2 ** 0.5)))
    y = 1 - (horner_eval(ERFC_COEFFS, t) * t * exp_num(-0.5 * x * x))
    return 0.5 * (1 + sign * y)""",
    "standard-normal-quantile": """def standard_normal_quantile(p):
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0,1)")
    sign = -1 if p < 0.5 else 1
    p_star = p if p < 0.5 else (1 - p)
    t = (-2 * log_num(p_star)) ** 0.5
    return sign * (t - (horner_eval(QUANTILE_NUM_COEFFS, t) / horner_eval(QUANTILE_DEN_COEFFS, t)))""",
    "uniform-cdf": """def uniform_cdf(x, a, b):
    if x < a:
        return 0
    if x > b:
        return 1
    return (x - a) / (b - a)""",
    "exponential-quantile": """def exponential_quantile(p, rate):
    if p < 0 or p > 1:
        raise ValueError("p must be in [0,1]")
    if p == 1:
        return float('inf')
    return -log_num(1 - p) / rate""",
    "poisson-pmf": """def poisson_pmf(k, rate):
    if k < 0:
        return 0
    return (rate ** k) * exp_num(-rate) / factorial(k)""",
    "binomial-pmf": """def binomial_pmf(k, n, p):
    if k < 0 or k > n:
        return 0
    return binomial_coeff(n, k) * (p ** k) * ((1 - p) ** (n - k))""",
}

CHEZ_SNIPPETS = {
    "horner-eval": """(define (poly-eval cs x)
  (fold-right (lambda (c acc)
                (+ c (* acc x)))
              0
              cs))""",
    "safe-log": """(define (guarded-log u)
  (log-num (max u 1e-300)))""",
    "standard-normal-cdf": """(define (stdnorm-cdf x)
  (let* ([sgn (if (< x 0) -1 1)]
         [ax (abs x)]
         [t (/ 1 (+ 1 (* *erfc-p* (/ ax (sqrt 2)))))]
         [y (- 1 (* (horner-eval *erfc-coeffs* t)
                    t
                    (exp-num (* -0.5 ax ax))))])
    (* 0.5 (+ 1 (* sgn y)))))""",
    "standard-normal-quantile": """(define (stdnorm-quantile p)
  (if (or (<= p 0) (>= p 1))
      (error 'stdnorm-quantile "p out of range" p)
      (let* ([sgn (if (< p 0.5) -1 1)]
             [pp (if (< p 0.5) p (- 1 p))]
             [t (sqrt (* -2 (log-num pp)))])
        (* sgn
           (- t
              (/ (horner-eval *quantile-num-coeffs* t)
                 (horner-eval *quantile-den-coeffs* t)))))))""",
    "uniform-cdf": """(define (u-cdf x a b)
  (cond
    [(< x a) 0]
    [(> x b) 1]
    [else (/ (- x a) (- b a))]))""",
    "exponential-quantile": """(define (exp-q p rate)
  (if (or (< p 0) (> p 1))
      (error 'exp-q "p out of range" p)
      (if (= p 1)
          +inf.0
          (/ (- (log-num (- 1 p))) rate))))""",
    "poisson-pmf": """(define (pois-pmf k lam)
  (if (< k 0)
      0
      (/ (* (expt lam k) (exp-num (- lam)))
         (factorial k))))""",
    "binomial-pmf": """(define (binom-pmf k n p)
  (if (or (< k 0) (> k n))
      0
      (* (binomial-coeff n k)
         (expt p k)
         (expt (- 1 p) (- n k)))))""",
}

BUGGY_CASES = [
    {
        "fn": "horner-eval",
        "buggy": """(define (horner-eval coeffs x)
  (fold-left (lambda (acc c) (+ c (* acc x))) 0 coeffs))""",
        "note": "Horner recurrence must consume coefficients from highest degree side (right fold here), not left fold.",
    },
    {
        "fn": "horner-eval",
        "buggy": """(define (horner-eval coeffs x)
  (fold-right (lambda (c acc) (+ c (* acc x))) 1 coeffs))""",
        "note": "Accumulator seed must be 0; seeding with 1 shifts every polynomial result.",
    },
    {
        "fn": "safe-log",
        "buggy": """(define (safe-log u)
  (log-num (min u 1e-300)))""",
        "note": "Clamp must use max to enforce a lower bound, not min.",
    },
    {
        "fn": "safe-log",
        "buggy": """(define (safe-log u)
  (log-num (max u 1e-30)))""",
        "note": "Clamp threshold changed by 270 orders of magnitude; keep 1e-300.",
    },
    {
        "fn": "standard-normal-cdf",
        "buggy": """(define (standard-normal-cdf x)
  (let* ([sign (if (< x 0) -1 1)]
         [x (abs x)]
         [t (/ 1 (+ 1 (* *erfc-p* (/ x (sqrt 2)))))]
         [y (- 1 (* (horner-eval *erfc-coeffs* t) t (exp-num (* -0.5 x x))))])
    (* 0.5 (+ 1 y))))""",
        "note": "Negative-side symmetry is broken: sign factor must be applied to y.",
    },
    {
        "fn": "standard-normal-cdf",
        "buggy": """(define (standard-normal-cdf x)
  (let* ([sign (if (< x 0) -1 1)]
         [x (abs x)]
         [t (/ 1 (+ 1 (* *erfc-p* (/ x (sqrt 2)))))]
         [y (- 1 (* (horner-eval *erfc-coeffs* t) t (exp-num (* -0.5 x))))])
    (* 0.5 (+ 1 (* sign y)))))""",
        "note": "Gaussian exponent must use x squared; dropping one x distorts the tail.",
    },
    {
        "fn": "standard-normal-quantile",
        "buggy": """(define (standard-normal-quantile p)
  (if (or (<= p 0) (>= p 1))
      (error 'standard-normal-quantile "p must be in (0,1)" p)
      (let* ([sign 1]
             [p* (if (< p 0.5) p (- 1 p))]
             [t (sqrt (* -2 (log-num p*)))])
        (* sign (- t (/ (horner-eval *quantile-num-coeffs* t)
                        (horner-eval *quantile-den-coeffs* t)))))))""",
        "note": "Quantiles below 0.5 must be negative; forcing sign=1 breaks symmetry.",
    },
    {
        "fn": "standard-normal-quantile",
        "buggy": """(define (standard-normal-quantile p)
  (if (or (< p 0) (> p 1))
      (error 'standard-normal-quantile "p must be in (0,1)" p)
      (let* ([sign (if (< p 0.5) -1 1)]
             [p* (if (< p 0.5) p (- 1 p))]
             [t (sqrt (* -2 (log-num p*)))])
        (* sign (- t (/ (horner-eval *quantile-num-coeffs* t)
                        (horner-eval *quantile-den-coeffs* t)))))))""",
        "note": "Domain guard must reject p=0 and p=1 as well, not only strict out-of-range values.",
    },
    {
        "fn": "uniform-cdf",
        "buggy": """(define (uniform-cdf x a b)
  (cond
    [(< x a) 0]
    [(> x b) 0]
    [else (/ (- x a) (- b a))]))""",
        "note": "Upper tail should be 1 for x>b, not 0.",
    },
    {
        "fn": "uniform-cdf",
        "buggy": """(define (uniform-cdf x a b)
  (cond
    [(< x a) 0]
    [(> x b) 1]
    [else (/ (- b x) (- b a))]))""",
        "note": "Interior branch is reversed; CDF must increase with x.",
    },
    {
        "fn": "exponential-quantile",
        "buggy": """(define (exponential-quantile p rate)
  (if (or (< p 0) (> p 1))
      (error 'exponential-quantile "p must be in [0,1]" p)
      (if (= p 1) +inf.0 (/ (- (log-num p)) rate))))""",
        "note": "Inverse CDF uses log(1-p), not log(p).",
    },
    {
        "fn": "exponential-quantile",
        "buggy": """(define (exponential-quantile p rate)
  (if (or (< p 0) (> p 1))
      (error 'exponential-quantile "p must be in [0,1]" p)
      (if (= p 1) 0 (/ (- (log-num (- 1 p))) rate))))""",
        "note": "At p=1, exponential quantile diverges to +inf.0, not 0.",
    },
    {
        "fn": "poisson-pmf",
        "buggy": """(define (poisson-pmf k rate)
  (if (< k 0)
      0
      (/ (expt rate k)
         (factorial k))))""",
        "note": "Poisson PMF must include the exp(-rate) factor.",
    },
    {
        "fn": "poisson-pmf",
        "buggy": """(define (poisson-pmf k rate)
  (if (< k 0)
      0
      (/ (* (expt rate k) (exp-num (- rate)))
         (factorial (+ k 1)))))""",
        "note": "Denominator should be k!, not (k+1)!.",
    },
    {
        "fn": "binomial-pmf",
        "buggy": """(define (binomial-pmf k n p)
  (if (or (< k 0) (> k n))
      0
      (* (binomial-coeff n k)
         (expt p k)
         (expt (- 1 p) (+ n k)))))""",
        "note": "Failure term exponent must be (n-k), not (n+k).",
    },
    {
        "fn": "binomial-pmf",
        "buggy": """(define (binomial-pmf k n p)
  (if (< k 0)
      0
      (* (binomial-coeff n k)
         (expt p k)
         (expt (- 1 p) k))))""",
        "note": "Failure term must use remaining trials `(n-k)`, not `k`.",
    },
]

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
    *,
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
    sample_id = f"random_distributions_{family}_{family_counter[family]:03d}"
    sample = {
        "id": sample_id,
        "family": family,
        "category": category,
        "difficulty": difficulty,
        "source_module": SOURCE_MODULE,
        "source_test": SOURCE_TEST,
        "source_function": source_function,
        "prompt_body": prompt.strip(),
        "prompt": diversify_prompt(
            prompt.strip(),
            family,
            source_function,
            family_counter[family],
            category,
            verify_expr,
        ),
        "ground_truth": ground_truth.strip(),
        "verify_expr": verify_expr.strip(),
        "tags": tags,
    }
    for key in REQUIRED_KEYS:
        if key not in sample:
            raise ValueError(f"missing key {key}")
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
        prompt=f"""Implement this function in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=SOURCE_DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "random", "distributions", "spec-to-code", fn],
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
        ground_truth=SOURCE_DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "random", "distributions", "skeleton", fn],
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
Preserve behavior exactly.

Target function name: `{fn}`

```python
{PYTHON_SNIPPETS[fn]}
```

Return only the Scheme definition.""",
        ground_truth=SOURCE_DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "random", "distributions", "translation", "python", fn],
    )

    add_sample(
        family="translation",
        category="translation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Convert this Chez-style snippet to canonical Fold style.
Keep semantics identical.

Target function: `{fn}`

```scheme
{CHEZ_SNIPPETS[fn]}
```

Return only Fold code.""",
        ground_truth=SOURCE_DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "random", "distributions", "translation", "chez", fn],
    )


# -----------------------------------------------------------------------------
# Family 3: bugfix (16)
# -----------------------------------------------------------------------------
if len(BUGGY_CASES) != 16:
    raise ValueError(f"expected 16 bugfix cases, got {len(BUGGY_CASES)}")

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
        ground_truth=SOURCE_DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "random", "distributions", "bugfix", fn],
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
        tags=["tier1", "random", "distributions", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # horner-eval
    {
        "fn": "horner-eval",
        "prompt": "Evaluate polynomial `1 + 0*x - 2*x^2 + x^3` at x=2 using `horner-eval`.",
        "gt": "(horner-eval '(1 0 -2 1) 2)",
        "verify": "(equal? (horner-eval '(1 0 -2 1) 2) 1)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "horner-eval",
        "prompt": "Return whether `horner-eval` matches explicit polynomial evaluation for coefficients `(2 -1 4)` at x=3.",
        "gt": "(= (horner-eval '(2 -1 4) 3) (+ 2 (* -1 3) (* 4 3 3)))",
        "verify": "(equal? (= (horner-eval '(2 -1 4) 3) (+ 2 (* -1 3) (* 4 3 3))) #t)",
        "difficulty": "medium",
        "tags": ["equivalence"],
    },
    {
        "fn": "horner-eval",
        "prompt": "Map `horner-eval` with coefficients `(1 2)` over x in `(0 1 2)`.",
        "gt": "(map (lambda (x) (horner-eval '(1 2) x)) '(0 1 2))",
        "verify": "(equal? (map (lambda (x) (horner-eval '(1 2) x)) '(0 1 2)) '(1 3 5))",
        "difficulty": "easy",
        "tags": ["map"],
    },
    {
        "fn": "horner-eval",
        "prompt": "Evaluate `horner-eval` on `*quantile-den-coeffs*` at t=1.5 and return whether result is positive.",
        "gt": "(> (horner-eval *quantile-den-coeffs* 1.5) 0)",
        "verify": "(equal? (> (horner-eval *quantile-den-coeffs* 1.5) 0) #t)",
        "difficulty": "hard",
        "tags": ["integration"],
    },

    # safe-log
    {
        "fn": "safe-log",
        "prompt": "Return whether `safe-log` at zero equals `log-num 1e-300`.",
        "gt": "(approx=? (safe-log 0.0) (log-num 1e-300) 1e-12)",
        "verify": "(equal? (approx=? (safe-log 0.0) (log-num 1e-300) 1e-12) #t)",
        "difficulty": "easy",
        "tags": ["guard"],
    },
    {
        "fn": "safe-log",
        "prompt": "Map `safe-log` over `(1.0 0.5 0.0)` and return the resulting list.",
        "gt": "(map safe-log '(1.0 0.5 0.0))",
        "verify": "(let ([xs (map safe-log '(1.0 0.5 0.0))]) (and (approx=? (car xs) 0.0 1e-12) (< (cadr xs) 0) (< (caddr xs) (cadr xs))))",
        "difficulty": "medium",
        "tags": ["map"],
    },
    {
        "fn": "safe-log",
        "prompt": "Use `safe-log` to compute `log(4)-log(2)` and compare against `log(2)`.",
        "gt": "(approx=? (- (safe-log 4.0) (safe-log 2.0)) (log-num 2.0) 1e-12)",
        "verify": "(equal? (approx=? (- (safe-log 4.0) (safe-log 2.0)) (log-num 2.0) 1e-12) #t)",
        "difficulty": "medium",
        "tags": ["identity"],
    },
    {
        "fn": "safe-log",
        "prompt": "Express the exponential median formula with `safe-log` and verify it matches `exponential-quantile` at p=0.25, rate=2.",
        "gt": "(approx=? (/ (- (safe-log (- 1 0.25))) 2.0) (exponential-quantile 0.25 2.0) 1e-12)",
        "verify": "(equal? (approx=? (/ (- (safe-log (- 1 0.25))) 2.0) (exponential-quantile 0.25 2.0) 1e-12) #t)",
        "difficulty": "hard",
        "tags": ["integration"],
    },

    # standard-normal-cdf
    {
        "fn": "standard-normal-cdf",
        "prompt": "Evaluate `standard-normal-cdf` at 0.",
        "gt": "(standard-normal-cdf 0.0)",
        "verify": "(approx=? (standard-normal-cdf 0.0) 0.5 0.001)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "standard-normal-cdf",
        "prompt": "Return whether `standard-normal-cdf` satisfies symmetry at x=1.2: CDF(-x)+CDF(x)=1.",
        "gt": "(approx=? (+ (standard-normal-cdf -1.2) (standard-normal-cdf 1.2)) 1.0 0.02)",
        "verify": "(equal? (approx=? (+ (standard-normal-cdf -1.2) (standard-normal-cdf 1.2)) 1.0 0.02) #t)",
        "difficulty": "medium",
        "tags": ["symmetry"],
    },
    {
        "fn": "standard-normal-cdf",
        "prompt": "Compose `standard-normal-cdf` with `standard-normal-quantile` at p=0.8.",
        "gt": "(standard-normal-cdf (standard-normal-quantile 0.8))",
        "verify": "(approx=? (standard-normal-cdf (standard-normal-quantile 0.8)) 0.8 0.03)",
        "difficulty": "hard",
        "tags": ["inverse"],
    },
    {
        "fn": "standard-normal-cdf",
        "prompt": "Check monotonicity of `standard-normal-cdf` at x=-1,0,1.",
        "gt": "(let ([a (standard-normal-cdf -1.0)] [b (standard-normal-cdf 0.0)] [c (standard-normal-cdf 1.0)]) (and (< a b) (< b c)))",
        "verify": "(equal? (let ([a (standard-normal-cdf -1.0)] [b (standard-normal-cdf 0.0)] [c (standard-normal-cdf 1.0)]) (and (< a b) (< b c))) #t)",
        "difficulty": "medium",
        "tags": ["ordering"],
    },

    # standard-normal-quantile
    {
        "fn": "standard-normal-quantile",
        "prompt": "Evaluate `standard-normal-quantile` at p=0.5.",
        "gt": "(standard-normal-quantile 0.5)",
        "verify": "(approx=? (standard-normal-quantile 0.5) 0.0 0.03)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "standard-normal-quantile",
        "prompt": "Compute `standard-normal-quantile` at p=0.95 and check it is near 1.64.",
        "gt": "(standard-normal-quantile 0.95)",
        "verify": "(approx=? (standard-normal-quantile 0.95) 1.64 0.12)",
        "difficulty": "medium",
        "tags": ["tail"],
    },
    {
        "fn": "standard-normal-quantile",
        "prompt": "Return whether `standard-normal-quantile 0.2` is less than `standard-normal-quantile 0.8`.",
        "gt": "(< (standard-normal-quantile 0.2) (standard-normal-quantile 0.8))",
        "verify": "(equal? (< (standard-normal-quantile 0.2) (standard-normal-quantile 0.8)) #t)",
        "difficulty": "medium",
        "tags": ["ordering"],
    },
    {
        "fn": "standard-normal-quantile",
        "prompt": "Round-trip p=0.33 through `standard-normal-quantile` then `standard-normal-cdf`.",
        "gt": "(standard-normal-cdf (standard-normal-quantile 0.33))",
        "verify": "(approx=? (standard-normal-cdf (standard-normal-quantile 0.33)) 0.33 0.03)",
        "difficulty": "hard",
        "tags": ["inverse"],
    },

    # uniform-cdf
    {
        "fn": "uniform-cdf",
        "prompt": "Evaluate `uniform-cdf` at x=4 on interval [2,6].",
        "gt": "(uniform-cdf 4.0 2.0 6.0)",
        "verify": "(approx=? (uniform-cdf 4.0 2.0 6.0) 0.5 1e-12)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "uniform-cdf",
        "prompt": "Return the three-way piecewise outputs of `uniform-cdf` for x in (1,3,8) on [2,6].",
        "gt": "(list (uniform-cdf 1.0 2.0 6.0) (uniform-cdf 3.0 2.0 6.0) (uniform-cdf 8.0 2.0 6.0))",
        "verify": "(equal? (list (uniform-cdf 1.0 2.0 6.0) (uniform-cdf 3.0 2.0 6.0) (uniform-cdf 8.0 2.0 6.0)) '(0 0.25 1))",
        "difficulty": "easy",
        "tags": ["piecewise"],
    },
    {
        "fn": "uniform-cdf",
        "prompt": "Compose `uniform-cdf` with `uniform-quantile` at p=0.3 on [10,20].",
        "gt": "(uniform-cdf (uniform-quantile 0.3 10.0 20.0) 10.0 20.0)",
        "verify": "(approx=? (uniform-cdf (uniform-quantile 0.3 10.0 20.0) 10.0 20.0) 0.3 1e-12)",
        "difficulty": "medium",
        "tags": ["inverse"],
    },
    {
        "fn": "uniform-cdf",
        "prompt": "Check monotonicity of `uniform-cdf` on [0,5] for x=1,2,4.",
        "gt": "(let ([a (uniform-cdf 1.0 0.0 5.0)] [b (uniform-cdf 2.0 0.0 5.0)] [c (uniform-cdf 4.0 0.0 5.0)]) (and (< a b) (< b c)))",
        "verify": "(equal? (let ([a (uniform-cdf 1.0 0.0 5.0)] [b (uniform-cdf 2.0 0.0 5.0)] [c (uniform-cdf 4.0 0.0 5.0)]) (and (< a b) (< b c))) #t)",
        "difficulty": "medium",
        "tags": ["ordering"],
    },

    # exponential-quantile
    {
        "fn": "exponential-quantile",
        "prompt": "Evaluate `exponential-quantile` at p=0.5 with rate=1.",
        "gt": "(exponential-quantile 0.5 1.0)",
        "verify": "(approx=? (exponential-quantile 0.5 1.0) (log-num 2.0) 1e-9)",
        "difficulty": "easy",
        "tags": ["median"],
    },
    {
        "fn": "exponential-quantile",
        "prompt": "Round-trip p=0.7 through `exponential-quantile` then `exponential-cdf` at rate=2.",
        "gt": "(exponential-cdf (exponential-quantile 0.7 2.0) 2.0)",
        "verify": "(approx=? (exponential-cdf (exponential-quantile 0.7 2.0) 2.0) 0.7 1e-9)",
        "difficulty": "medium",
        "tags": ["inverse"],
    },
    {
        "fn": "exponential-quantile",
        "prompt": "Return whether `exponential-quantile` maps p=1 to `+inf.0`.",
        "gt": "(eqv? (exponential-quantile 1.0 3.0) +inf.0)",
        "verify": "(equal? (eqv? (exponential-quantile 1.0 3.0) +inf.0) #t)",
        "difficulty": "medium",
        "tags": ["edge-case"],
    },
    {
        "fn": "exponential-quantile",
        "prompt": "Compare `exponential-quantile` at p=0.5 for rates 1 and 3.",
        "gt": "(> (exponential-quantile 0.5 1.0) (exponential-quantile 0.5 3.0))",
        "verify": "(equal? (> (exponential-quantile 0.5 1.0) (exponential-quantile 0.5 3.0)) #t)",
        "difficulty": "hard",
        "tags": ["rate-effect"],
    },

    # poisson-pmf
    {
        "fn": "poisson-pmf",
        "prompt": "Evaluate `poisson-pmf` at k=0, rate=2.",
        "gt": "(poisson-pmf 0 2.0)",
        "verify": "(approx=? (poisson-pmf 0 2.0) (exp-num -2.0) 1e-12)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "poisson-pmf",
        "prompt": "Return whether ratio `poisson-pmf(3,4)/poisson-pmf(2,4)` equals 4/3.",
        "gt": "(/ (poisson-pmf 3 4.0) (poisson-pmf 2 4.0))",
        "verify": "(approx=? (/ (poisson-pmf 3 4.0) (poisson-pmf 2 4.0)) (/ 4.0 3.0) 1e-12)",
        "difficulty": "medium",
        "tags": ["ratio"],
    },
    {
        "fn": "poisson-pmf",
        "prompt": "Sum `poisson-pmf` from k=0..12 for rate=3 and return whether total is near 1.",
        "gt": "(let loop ([k 0] [acc 0.0]) (if (> k 12) acc (loop (+ k 1) (+ acc (poisson-pmf k 3.0)))))",
        "verify": "(approx=? (let loop ([k 0] [acc 0.0]) (if (> k 12) acc (loop (+ k 1) (+ acc (poisson-pmf k 3.0))))) 1.0 0.01)",
        "difficulty": "hard",
        "tags": ["normalization"],
    },
    {
        "fn": "poisson-pmf",
        "prompt": "For rate=4, check that `poisson-pmf` at k=4 is at least as large as k=3 and k=5.",
        "gt": "(let ([p3 (poisson-pmf 3 4.0)] [p4 (poisson-pmf 4 4.0)] [p5 (poisson-pmf 5 4.0)]) (and (>= p4 p3) (>= p4 p5)))",
        "verify": "(equal? (let ([p3 (poisson-pmf 3 4.0)] [p4 (poisson-pmf 4 4.0)] [p5 (poisson-pmf 5 4.0)]) (and (>= p4 p3) (>= p4 p5))) #t)",
        "difficulty": "hard",
        "tags": ["mode"],
    },

    # binomial-pmf
    {
        "fn": "binomial-pmf",
        "prompt": "Return out-of-range checks for `binomial-pmf` with n=5, p=0.3 at k=-1 and k=6.",
        "gt": "(list (binomial-pmf -1 5 0.3) (binomial-pmf 6 5 0.3))",
        "verify": "(equal? (list (binomial-pmf -1 5 0.3) (binomial-pmf 6 5 0.3)) '(0 0))",
        "difficulty": "easy",
        "tags": ["bounds"],
    },
    {
        "fn": "binomial-pmf",
        "prompt": "Sum `binomial-pmf` over k=0..5 for n=5, p=0.3 and check normalization.",
        "gt": "(+ (binomial-pmf 0 5 0.3) (binomial-pmf 1 5 0.3) (binomial-pmf 2 5 0.3) (binomial-pmf 3 5 0.3) (binomial-pmf 4 5 0.3) (binomial-pmf 5 5 0.3))",
        "verify": "(approx=? (+ (binomial-pmf 0 5 0.3) (binomial-pmf 1 5 0.3) (binomial-pmf 2 5 0.3) (binomial-pmf 3 5 0.3) (binomial-pmf 4 5 0.3) (binomial-pmf 5 5 0.3)) 1.0 1e-9)",
        "difficulty": "medium",
        "tags": ["normalization"],
    },
    {
        "fn": "binomial-pmf",
        "prompt": "Check symmetry of `binomial-pmf` for n=6, p=0.5 at k=2 and k=4.",
        "gt": "(approx=? (binomial-pmf 2 6 0.5) (binomial-pmf 4 6 0.5) 1e-12)",
        "verify": "(equal? (approx=? (binomial-pmf 2 6 0.5) (binomial-pmf 4 6 0.5) 1e-12) #t)",
        "difficulty": "medium",
        "tags": ["symmetry"],
    },
    {
        "fn": "binomial-pmf",
        "prompt": "Verify explicit coefficient formula for `binomial-pmf` at k=2, n=5, p=0.4.",
        "gt": "(approx=? (binomial-pmf 2 5 0.4) (* 10 (expt 0.4 2) (expt 0.6 3)) 1e-12)",
        "verify": "(equal? (approx=? (binomial-pmf 2 5 0.4) (* 10 (expt 0.4 2) (expt 0.6 3)) 1e-12) #t)",
        "difficulty": "hard",
        "tags": ["formula"],
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
for family, family_samples in by_family.items():
    picked = spread_indices(len(family_samples), EVAL_QUOTA[family])
    for i, s in enumerate(family_samples):
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
    missing_functions = [fn for fn in all_source_functions if fn_counts[fn] == 0]
    if not missing_functions:
        break

    for fn in missing_functions:
        candidates = [s for s in samples if str(s["source_function"]) == fn and str(s["id"]) not in eval_ids]
        swapped = False
        for cand in candidates:
            fam = str(cand["family"])
            fam_eval = [id_to_sample[sid] for sid in eval_ids if str(id_to_sample[sid]["family"]) == fam]
            removable = [r for r in fam_eval if fn_counts[str(r["source_function"])] > 1]
            if not removable:
                continue
            removable.sort(
                key=lambda r: (fn_counts[str(r["source_function"])], str(r["id"])),
                reverse=True,
            )
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
all_rows: List[Dict[str, object]] = []
for s in samples:
    row = dict(s)
    row["split"] = "eval" if s["id"] in eval_ids else "train"
    all_rows.append(row)
    if row["split"] == "eval":
        eval_rows.append(row)
    else:
        train_rows.append(row)

if len(train_rows) != 66 or len(eval_rows) != 14:
    raise ValueError(f"split mismatch: train={len(train_rows)}, eval={len(eval_rows)}")


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


write_jsonl(ALL_PATH, all_rows)
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
