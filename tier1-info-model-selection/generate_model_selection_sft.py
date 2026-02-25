#!/usr/bin/env python3
"""Generate Tier-1 model-selection SFT samples for lattice/info/model-selection.ss."""

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

SOURCE_MODULE = "lattice/info/model-selection.ss"
SOURCE_TEST = "lattice/info/test-model-selection.ss"

DEFS: Dict[str, str] = {
    "log-likelihood-gaussian": """(define (log-likelihood-gaussian residuals)
  (let* ([n (length residuals)]
         [ss (fold-left + 0 (map (lambda (r) (* r r)) residuals))]
         [sigma2 (/ ss n)])
    (if (<= sigma2 0)
        0
        (* -0.5 n (+ 1 (log-num (* 2 (pi-value) sigma2)))))))""",
    "log-likelihood-gaussian-vec": """(define (log-likelihood-gaussian-vec residuals-vec)
  (let* ([n (vector-length residuals-vec)]
         [ss (let loop ([i 0] [s 0])
               (if (= i n)
                   s
                   (let ([r (vector-ref residuals-vec i)])
                     (loop (+ i 1) (+ s (* r r))))))]
         [sigma2 (/ ss n)])
    (if (<= sigma2 0)
        0
        (* -0.5 n (+ 1 (log-num (* 2 (pi-value) sigma2)))))))""",
    "aic": """(define (aic log-lik k)
  (+ (* -2 log-lik) (* 2 k)))""",
    "bic": """(define (bic log-lik k n)
  (+ (* -2 log-lik) (* k (log-num n))))""",
    "aicc": """(define (aicc log-lik k n)
  (let ([base (aic log-lik k)])
    (if (<= (- n k 1) 0)
        +inf.0
        (+ base (/ (* 2 k (+ k 1)) (- n k 1))))))""",
    "aic-weights": """(define (aic-weights aic-values)
  (if (null? aic-values)
      '()
      (let* ([aic-min (apply min aic-values)]
             [deltas (map (lambda (a) (- a aic-min)) aic-values)]
             [raw-weights (map (lambda (d) (exp-num (* -0.5 d))) deltas)]
             [total (fold-left + 0 raw-weights)])
        (if (<= total 0)
            (map (lambda (_) (/ 1.0 (length aic-values))) aic-values)
            (map (lambda (w) (/ w total)) raw-weights)))))""",
    "evidence-ratio": """(define (evidence-ratio w-i w-j)
  (if (<= w-j 0)
      +inf.0
      (/ w-i w-j)))""",
    "residual-entropy-bits": """(define (residual-entropy-bits residuals)
  (let* ([n (length residuals)]
         [ss (fold-left + 0 (map (lambda (r) (* r r)) residuals))]
         [sigma2 (/ ss n)])
    (if (<= sigma2 0)
        -inf.0
        (gaussian-entropy (sqrt sigma2)))))""",
}

SUPPORT_DEFS: Dict[str, str] = {
    "approx=?": """(define (approx=? expected actual tol)
  (< (abs (- expected actual)) tol))""",
}

ALL_DEFS: Dict[str, str] = {**SUPPORT_DEFS, **DEFS}

DEPENDS: Dict[str, List[str]] = {
    "approx=?": [],
    "log-likelihood-gaussian": [],
    "log-likelihood-gaussian-vec": [],
    "aic": [],
    "bic": [],
    "aicc": ["aic"],
    "aic-weights": [],
    "evidence-ratio": [],
    "residual-entropy-bits": [],
}

FUNCTION_ORDER = [
    "log-likelihood-gaussian",
    "log-likelihood-gaussian-vec",
    "aic",
    "bic",
    "aicc",
    "aic-weights",
    "evidence-ratio",
    "residual-entropy-bits",
]

SUPPORT_ORDER = ["approx=?"]

FUNCTION_SPECS = {
    "log-likelihood-gaussian": "Compute maximized Gaussian log-likelihood from a residual list via sigma^2 = mean squared residual.",
    "log-likelihood-gaussian-vec": "Vector variant of Gaussian log-likelihood; must match list behavior for equivalent residuals.",
    "aic": "Compute Akaike Information Criterion: AIC = -2*logL + 2*k.",
    "bic": "Compute Bayesian Information Criterion: BIC = -2*logL + k*log(n).",
    "aicc": "Compute corrected AIC for small samples; return +inf.0 when n <= k+1.",
    "aic-weights": "Convert a list of AIC scores to normalized Akaike weights that sum to 1.",
    "evidence-ratio": "Compute relative support w_i / w_j with +inf.0 when denominator weight is non-positive.",
    "residual-entropy-bits": "Estimate Gaussian differential entropy in bits from residual variance; return -inf.0 for zero variance.",
}

SKELETONS = {
    "log-likelihood-gaussian": """(define (log-likelihood-gaussian residuals)
  ;; TODO: compute maximized Gaussian log-likelihood from list residuals
  <TODO>)""",
    "log-likelihood-gaussian-vec": """(define (log-likelihood-gaussian-vec residuals-vec)
  ;; TODO: compute maximized Gaussian log-likelihood from vector residuals
  <TODO>)""",
    "aic": """(define (aic log-lik k)
  ;; TODO: AIC = -2*logL + 2*k
  <TODO>)""",
    "bic": """(define (bic log-lik k n)
  ;; TODO: BIC = -2*logL + k*log(n)
  <TODO>)""",
    "aicc": """(define (aicc log-lik k n)
  ;; TODO: corrected AIC with +inf.0 when n <= k+1
  <TODO>)""",
    "aic-weights": """(define (aic-weights aic-values)
  ;; TODO: delta-shift by min AIC, exponentiate, and normalize to sum 1
  <TODO>)""",
    "evidence-ratio": """(define (evidence-ratio w-i w-j)
  ;; TODO: return w-i / w-j with +inf.0 guard for non-positive denominator
  <TODO>)""",
    "residual-entropy-bits": """(define (residual-entropy-bits residuals)
  ;; TODO: convert residual variance to Gaussian entropy in bits
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "log-likelihood-gaussian": "(and (= (log-likelihood-gaussian '(0 0 0 0)) 0) (let* ([r '(1 -1)] [ll (log-likelihood-gaussian r)] [expected (* -0.5 2 (+ 1 (log-num (* 2 (pi-value)))))] ) (approx=? expected ll 1e-6)))",
    "log-likelihood-gaussian-vec": "(let* ([r '(0.5 -0.3 0.8 -0.2 0.1)] [rv (list->vector r)] [ll-list (log-likelihood-gaussian r)] [ll-vec (log-likelihood-gaussian-vec rv)]) (approx=? ll-list ll-vec 1e-10))",
    "aic": "(and (= (aic 5 2) -6) (= (aic -50 3) 106))",
    "bic": "(let ([result (bic 5 2 100)]) (approx=? (+ -10 (* 2 (log-num 100))) result 1e-10))",
    "aicc": "(and (= (aicc -50 5 6) +inf.0) (let* ([ll -50] [k 3] [n 10000] [a (aic ll k)] [ac (aicc ll k n)]) (< (abs (- a ac)) 0.01)) (let* ([ll -50] [k 5] [n 10] [a (aic ll k)] [ac (aicc ll k n)]) (and (> ac a) (approx=? 15 (- ac a) 1e-10))))",
    "aic-weights": "(let* ([ws (aic-weights '(100 102 105 110))] [same (aic-weights '(100 100 100))] [total (fold-left + 0 ws)]) (and (approx=? 1.0 total 1e-10) (> (car ws) (cadr ws)) (> (cadr ws) (caddr ws)) (approx=? (/ 1.0 3) (car same) 1e-10) (approx=? (/ 1.0 3) (cadr same) 1e-10)))",
    "evidence-ratio": "(and (approx=? 2.0 (evidence-ratio 0.6 0.3) 1e-10) (= (evidence-ratio 0.5 0.0) +inf.0))",
    "residual-entropy-bits": "(let ([h1 (residual-entropy-bits '(1.0 -0.5 0.3 -0.8 0.2))] [h-tight (residual-entropy-bits '(0.01 -0.01 0.005 -0.008))] [h-wide (residual-entropy-bits '(5.0 -3.0 4.0 -2.0))]) (and (> h1 0) (< h-tight h-wide)))",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")

TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "log-likelihood-gaussian": """def log_likelihood_gaussian(residuals):
    n = len(residuals)
    ss = sum(r * r for r in residuals)
    sigma2 = ss / n
    if sigma2 <= 0:
        return 0
    return -0.5 * n * (1 + log(2 * pi * sigma2))""",
    "log-likelihood-gaussian-vec": """def log_likelihood_gaussian_vec(residuals_vec):
    n = len(residuals_vec)
    ss = 0.0
    for r in residuals_vec:
        ss += r * r
    sigma2 = ss / n
    if sigma2 <= 0:
        return 0
    return -0.5 * n * (1 + log(2 * pi * sigma2))""",
    "aic": """def aic(log_lik, k):
    return -2 * log_lik + 2 * k""",
    "bic": """def bic(log_lik, k, n):
    return -2 * log_lik + k * log(n)""",
    "aicc": """def aicc(log_lik, k, n):
    base = aic(log_lik, k)
    if n - k - 1 <= 0:
        return float('inf')
    return base + (2 * k * (k + 1)) / (n - k - 1)""",
    "aic-weights": """def aic_weights(aic_values):
    if not aic_values:
        return []
    a_min = min(aic_values)
    deltas = [a - a_min for a in aic_values]
    raw = [exp(-0.5 * d) for d in deltas]
    total = sum(raw)
    if total <= 0:
        return [1.0 / len(aic_values) for _ in aic_values]
    return [w / total for w in raw]""",
    "evidence-ratio": """def evidence_ratio(w_i, w_j):
    if w_j <= 0:
        return float('inf')
    return w_i / w_j""",
    "residual-entropy-bits": """def residual_entropy_bits(residuals):
    n = len(residuals)
    ss = sum(r * r for r in residuals)
    sigma2 = ss / n
    if sigma2 <= 0:
        return float('-inf')
    sigma = sqrt(sigma2)
    return gaussian_entropy(sigma)""",
}

CHEZ_SNIPPETS = {
    "log-likelihood-gaussian": """(define (gaussian-ll residuals)
  (let* ([n (length residuals)]
         [ss (fold-left + 0 (map (lambda (r) (* r r)) residuals))]
         [sigma2 (/ ss n)])
    (if (<= sigma2 0)
        0
        (* -0.5 n (+ 1 (log-num (* 2 (pi-value) sigma2)))))))""",
    "log-likelihood-gaussian-vec": """(define (gaussian-ll-vec residuals)
  (let* ([n (vector-length residuals)]
         [ss (let loop ([i 0] [s 0])
               (if (= i n)
                   s
                   (let ([r (vector-ref residuals i)])
                     (loop (+ i 1) (+ s (* r r))))))]
         [sigma2 (/ ss n)])
    (if (<= sigma2 0)
        0
        (* -0.5 n (+ 1 (log-num (* 2 (pi-value) sigma2)))))))""",
    "aic": """(define (aic0 log-lik k)
  (+ (* -2 log-lik) (* 2 k)))""",
    "bic": """(define (bic0 log-lik k n)
  (+ (* -2 log-lik) (* k (log-num n))))""",
    "aicc": """(define (aicc0 log-lik k n)
  (let ([base (aic log-lik k)])
    (if (<= (- n k 1) 0)
        +inf.0
        (+ base (/ (* 2 k (+ k 1)) (- n k 1))))))""",
    "aic-weights": """(define (aic-weights0 aic-values)
  (if (null? aic-values)
      '()
      (let* ([a-min (apply min aic-values)]
             [deltas (map (lambda (a) (- a a-min)) aic-values)]
             [raw (map (lambda (d) (exp-num (* -0.5 d))) deltas)]
             [total (fold-left + 0 raw)])
        (if (<= total 0)
            (map (lambda (_) (/ 1.0 (length aic-values))) aic-values)
            (map (lambda (w) (/ w total)) raw)))))""",
    "evidence-ratio": """(define (evidence-ratio0 w-i w-j)
  (if (<= w-j 0)
      +inf.0
      (/ w-i w-j)))""",
    "residual-entropy-bits": """(define (residual-entropy0 residuals)
  (let* ([n (length residuals)]
         [ss (fold-left + 0 (map (lambda (r) (* r r)) residuals))]
         [sigma2 (/ ss n)])
    (if (<= sigma2 0)
        -inf.0
        (gaussian-entropy (sqrt sigma2)))))""",
}

BUGGY_CASES = [
    {
        "fn": "log-likelihood-gaussian",
        "buggy": """(define (log-likelihood-gaussian residuals)
  (let* ([n (length residuals)]
         [ss (fold-left + 0 (map (lambda (r) (* r r)) residuals))]
         [sigma2 (/ ss n)])
    (if (<= sigma2 0)
        0
        (* 0.5 n (+ 1 (log-num (* 2 (pi-value) sigma2)))))))""",
        "note": "Log-likelihood should have a negative leading coefficient (-0.5*n).",
    },
    {
        "fn": "log-likelihood-gaussian",
        "buggy": """(define (log-likelihood-gaussian residuals)
  (let* ([n (length residuals)]
         [ss (fold-left + 0 (map abs residuals))]
         [sigma2 (/ ss n)])
    (if (<= sigma2 0)
        0
        (* -0.5 n (+ 1 (log-num (* 2 (pi-value) sigma2)))))))""",
        "note": "Residual variance must use squared residuals, not absolute values.",
    },
    {
        "fn": "log-likelihood-gaussian-vec",
        "buggy": """(define (log-likelihood-gaussian-vec residuals-vec)
  (let* ([n (vector-length residuals-vec)]
         [ss (let loop ([i 0] [s 0])
               (if (= i n)
                   s
                   (let ([r (vector-ref residuals-vec i)])
                     (loop (+ i 1) (+ s r)))))]
         [sigma2 (/ ss n)])
    (if (<= sigma2 0)
        0
        (* -0.5 n (+ 1 (log-num (* 2 (pi-value) sigma2)))))))""",
        "note": "Vector residual accumulation must sum squared residuals.",
    },
    {
        "fn": "log-likelihood-gaussian-vec",
        "buggy": """(define (log-likelihood-gaussian-vec residuals-vec)
  (let* ([n (vector-length residuals-vec)]
         [ss 0]
         [sigma2 (/ ss n)])
    (if (<= sigma2 0)
        -inf.0
        (* -0.5 n (+ 1 (log-num (* 2 (pi-value) sigma2)))))))""",
        "note": "Degenerate sigma^2<=0 branch should return 0 to match module semantics.",
    },
    {
        "fn": "aic",
        "buggy": """(define (aic log-lik k)
  (+ (* 2 log-lik) (* 2 k)))""",
        "note": "AIC uses -2*logL, not +2*logL.",
    },
    {
        "fn": "aic",
        "buggy": """(define (aic log-lik k)
  (+ (* -2 log-lik) k))""",
        "note": "Parameter penalty should be 2*k, not k.",
    },
    {
        "fn": "bic",
        "buggy": """(define (bic log-lik k n)
  (+ (* -2 log-lik) (* k n)))""",
        "note": "BIC uses log(n) penalty, not linear n.",
    },
    {
        "fn": "bic",
        "buggy": """(define (bic log-lik k n)
  (+ (* 2 log-lik) (* k (log-num n))))""",
        "note": "BIC needs -2*logL with negative sign.",
    },
    {
        "fn": "aicc",
        "buggy": """(define (aicc log-lik k n)
  (aic log-lik k))""",
        "note": "AICc must add the finite-sample correction term.",
    },
    {
        "fn": "aicc",
        "buggy": """(define (aicc log-lik k n)
  (let ([base (aic log-lik k)])
    (if (<= (- n k 1) 0)
        0
        (+ base (/ (* 2 k (+ k 1)) (- n k 1))))))""",
        "note": "Undefined AICc region (n<=k+1) should return +inf.0.",
    },
    {
        "fn": "aic-weights",
        "buggy": """(define (aic-weights aic-values)
  (if (null? aic-values)
      '()
      (let* ([aic-min (apply min aic-values)]
             [deltas (map (lambda (a) (- a aic-min)) aic-values)]
             [raw-weights deltas]
             [total (fold-left + 0 raw-weights)])
        (map (lambda (w) (/ w total)) raw-weights))))""",
        "note": "Weights must exponentiate -0.5*delta before normalization.",
    },
    {
        "fn": "aic-weights",
        "buggy": """(define (aic-weights aic-values)
  (if (null? aic-values)
      '()
      (let* ([aic-min (apply min aic-values)]
             [deltas (map (lambda (a) (- a aic-min)) aic-values)]
             [raw-weights (map (lambda (d) (exp-num (* -0.5 d))) deltas)]
             [total (fold-left + 0 raw-weights)])
        raw-weights)))""",
        "note": "Returned weights must be normalized by the total.",
    },
    {
        "fn": "evidence-ratio",
        "buggy": """(define (evidence-ratio w-i w-j)
  (/ w-j w-i))""",
        "note": "Evidence ratio is w_i / w_j, not inverted.",
    },
    {
        "fn": "evidence-ratio",
        "buggy": """(define (evidence-ratio w-i w-j)
  (if (<= w-j 0)
      0
      (/ w-i w-j)))""",
        "note": "Non-positive denominator should yield +inf.0, not 0.",
    },
    {
        "fn": "residual-entropy-bits",
        "buggy": """(define (residual-entropy-bits residuals)
  (let* ([n (length residuals)]
         [ss (fold-left + 0 (map (lambda (r) (* r r)) residuals))]
         [sigma2 (/ ss n)])
    (if (<= sigma2 0)
        -inf.0
        (gaussian-entropy sigma2))))""",
        "note": "Gaussian entropy helper expects standard deviation sqrt(sigma^2).",
    },
    {
        "fn": "residual-entropy-bits",
        "buggy": """(define (residual-entropy-bits residuals)
  (let* ([n (length residuals)]
         [ss (fold-left + 0 (map (lambda (r) (* r r)) residuals))]
         [sigma2 (/ ss n)])
    (if (<= sigma2 0)
        0
        (gaussian-entropy (sqrt sigma2)))))""",
        "note": "Zero-variance branch should return -inf.0 for differential entropy limit.",
    },
]

DIFFICULTY = {
    "log-likelihood-gaussian": "easy",
    "log-likelihood-gaussian-vec": "medium",
    "aic": "easy",
    "bic": "medium",
    "aicc": "medium",
    "aic-weights": "hard",
    "evidence-ratio": "easy",
    "residual-entropy-bits": "medium",
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
    sid = f"info_model_selection_{family}_{family_counter[family]:03d}"
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
        prompt=f"""Implement this model-selection utility in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "info", "model-selection", "spec-to-code", fn],
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
        tags=["tier1", "info", "model-selection", "skeleton-completion", fn],
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
        tags=["tier1", "info", "model-selection", "python-to-scheme", fn],
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
        tags=["tier1", "info", "model-selection", "chez-to-fold", fn],
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
        tags=["tier1", "info", "model-selection", "bugfix", fn],
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
        tags=["tier1", "info", "model-selection", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # log-likelihood-gaussian
    {
        "fn": "log-likelihood-gaussian",
        "prompt": "Compute Gaussian log-likelihood for residuals '(1 -1).",
        "gt": "(log-likelihood-gaussian '(1 -1))",
        "verify": "(approx=? (* -0.5 2 (+ 1 (log-num (* 2 (pi-value))))) (log-likelihood-gaussian '(1 -1)) 1e-6)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "log-likelihood-gaussian",
        "prompt": "Return Gaussian log-likelihood for perfect residuals '(0 0 0 0).",
        "gt": "(log-likelihood-gaussian '(0 0 0 0))",
        "verify": "(= (log-likelihood-gaussian '(0 0 0 0)) 0)",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },
    {
        "fn": "log-likelihood-gaussian",
        "prompt": "Return whether tight residuals yield higher (less negative) log-likelihood than wide residuals.",
        "gt": "(let ([ll-tight (log-likelihood-gaussian '(0.1 -0.1 0.05 -0.05))] [ll-wide (log-likelihood-gaussian '(3 -2 4 -1))]) (> ll-tight ll-wide))",
        "verify": "(equal? (let ([ll-tight (log-likelihood-gaussian '(0.1 -0.1 0.05 -0.05))] [ll-wide (log-likelihood-gaussian '(3 -2 4 -1))]) (> ll-tight ll-wide)) #t)",
        "difficulty": "medium",
        "tags": ["property"],
    },
    {
        "fn": "log-likelihood-gaussian",
        "prompt": "Return whether list and vector Gaussian log-likelihood implementations agree on the same residual data.",
        "gt": "(let ([r '(0.5 -0.3 0.8 -0.2 0.1)]) (approx=? (log-likelihood-gaussian r) (log-likelihood-gaussian-vec (list->vector r)) 1e-10))",
        "verify": "(equal? (let ([r '(0.5 -0.3 0.8 -0.2 0.1)]) (approx=? (log-likelihood-gaussian r) (log-likelihood-gaussian-vec (list->vector r)) 1e-10)) #t)",
        "difficulty": "medium",
        "tags": ["consistency"],
    },

    # log-likelihood-gaussian-vec
    {
        "fn": "log-likelihood-gaussian-vec",
        "prompt": "Compute vector Gaussian log-likelihood for #(1 -1).",
        "gt": "(log-likelihood-gaussian-vec '#(1 -1))",
        "verify": "(approx=? (* -0.5 2 (+ 1 (log-num (* 2 (pi-value))))) (log-likelihood-gaussian-vec '#(1 -1)) 1e-6)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "log-likelihood-gaussian-vec",
        "prompt": "Return vector Gaussian log-likelihood for perfect residual vector '#(0 0 0).",
        "gt": "(log-likelihood-gaussian-vec '#(0 0 0))",
        "verify": "(= (log-likelihood-gaussian-vec '#(0 0 0)) 0)",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },
    {
        "fn": "log-likelihood-gaussian-vec",
        "prompt": "Return whether small vector residual variance leads to higher log-likelihood than large variance.",
        "gt": "(let ([ll-small (log-likelihood-gaussian-vec '#(0.2 -0.2 0.1 -0.1))] [ll-large (log-likelihood-gaussian-vec '#(5 -4 3 -2))]) (> ll-small ll-large))",
        "verify": "(equal? (let ([ll-small (log-likelihood-gaussian-vec '#(0.2 -0.2 0.1 -0.1))] [ll-large (log-likelihood-gaussian-vec '#(5 -4 3 -2))]) (> ll-small ll-large)) #t)",
        "difficulty": "medium",
        "tags": ["property"],
    },
    {
        "fn": "log-likelihood-gaussian-vec",
        "prompt": "Return whether vector and list log-likelihood versions are numerically equal on the same sample.",
        "gt": "(let ([r '(0.5 -0.3 0.8 -0.2 0.1)]) (approx=? (log-likelihood-gaussian-vec (list->vector r)) (log-likelihood-gaussian r) 1e-10))",
        "verify": "(equal? (let ([r '(0.5 -0.3 0.8 -0.2 0.1)]) (approx=? (log-likelihood-gaussian-vec (list->vector r)) (log-likelihood-gaussian r) 1e-10)) #t)",
        "difficulty": "medium",
        "tags": ["consistency"],
    },

    # aic
    {
        "fn": "aic",
        "prompt": "Compute AIC for log-likelihood -50 with k=3.",
        "gt": "(aic -50 3)",
        "verify": "(= (aic -50 3) 106)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "aic",
        "prompt": "Return whether improved log-likelihood (same k) lowers AIC.",
        "gt": "(< (aic -40 3) (aic -50 3))",
        "verify": "(equal? (< (aic -40 3) (aic -50 3)) #t)",
        "difficulty": "easy",
        "tags": ["property"],
    },
    {
        "fn": "aic",
        "prompt": "Return whether increasing parameter count raises AIC when log-likelihood is fixed.",
        "gt": "(< (aic -50 2) (aic -50 5))",
        "verify": "(equal? (< (aic -50 2) (aic -50 5)) #t)",
        "difficulty": "easy",
        "tags": ["property"],
    },
    {
        "fn": "aic",
        "prompt": "Compute the AIC difference between k=5 and k=2 at the same log-likelihood -50.",
        "gt": "(- (aic -50 5) (aic -50 2))",
        "verify": "(= (- (aic -50 5) (aic -50 2)) 6)",
        "difficulty": "easy",
        "tags": ["difference"],
    },

    # bic
    {
        "fn": "bic",
        "prompt": "Compute BIC for log-likelihood 5, k=2, n=100.",
        "gt": "(bic 5 2 100)",
        "verify": "(approx=? (+ -10 (* 2 (log-num 100))) (bic 5 2 100) 1e-10)",
        "difficulty": "medium",
        "tags": ["direct"],
    },
    {
        "fn": "bic",
        "prompt": "Return whether BIC penalty increases with sample size when log-likelihood and k are fixed.",
        "gt": "(< (bic -50 3 10) (bic -50 3 100))",
        "verify": "(equal? (< (bic -50 3 10) (bic -50 3 100)) #t)",
        "difficulty": "medium",
        "tags": ["property"],
    },
    {
        "fn": "bic",
        "prompt": "Return whether BIC is greater than AIC at n=100 and k=3 for log-likelihood -50.",
        "gt": "(> (bic -50 3 100) (aic -50 3))",
        "verify": "(equal? (> (bic -50 3 100) (aic -50 3)) #t)",
        "difficulty": "medium",
        "tags": ["comparison"],
    },
    {
        "fn": "bic",
        "prompt": "Compute BIC when k=0 (it should reduce to -2*log-likelihood).",
        "gt": "(bic -33 0 500)",
        "verify": "(= (bic -33 0 500) 66)",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },

    # aicc
    {
        "fn": "aicc",
        "prompt": "Compute AICc for log-likelihood -50, k=5, n=10.",
        "gt": "(aicc -50 5 10)",
        "verify": "(approx=? 125 (aicc -50 5 10) 1e-10)",
        "difficulty": "medium",
        "tags": ["direct"],
    },
    {
        "fn": "aicc",
        "prompt": "Return AICc for undefined finite-sample region n<=k+1 (log-likelihood -50, k=5, n=6).",
        "gt": "(aicc -50 5 6)",
        "verify": "(= (aicc -50 5 6) +inf.0)",
        "difficulty": "medium",
        "tags": ["edge-case"],
    },
    {
        "fn": "aicc",
        "prompt": "Return whether AICc converges to AIC for large n (n=10000, k=3, log-likelihood -50).",
        "gt": "(let ([a (aic -50 3)] [ac (aicc -50 3 10000)]) (< (abs (- a ac)) 0.01))",
        "verify": "(equal? (let ([a (aic -50 3)] [ac (aicc -50 3 10000)]) (< (abs (- a ac)) 0.01)) #t)",
        "difficulty": "medium",
        "tags": ["limit"],
    },
    {
        "fn": "aicc",
        "prompt": "Return whether smaller n implies larger AICc correction for fixed log-likelihood and k.",
        "gt": "(> (- (aicc -50 3 20) (aic -50 3)) (- (aicc -50 3 100) (aic -50 3)))",
        "verify": "(equal? (> (- (aicc -50 3 20) (aic -50 3)) (- (aicc -50 3 100) (aic -50 3))) #t)",
        "difficulty": "hard",
        "tags": ["property"],
    },

    # aic-weights
    {
        "fn": "aic-weights",
        "prompt": "Compute Akaike weights for AIC values '(100 102 105 110).",
        "gt": "(aic-weights '(100 102 105 110))",
        "verify": "(let* ([ws (aic-weights '(100 102 105 110))] [total (fold-left + 0 ws)]) (approx=? 1.0 total 1e-10))",
        "difficulty": "hard",
        "tags": ["direct"],
    },
    {
        "fn": "aic-weights",
        "prompt": "Return whether the minimum-AIC model receives the highest Akaike weight.",
        "gt": "(let ([ws (aic-weights '(100 102 105 110))]) (and (> (car ws) (cadr ws)) (> (cadr ws) (caddr ws))))",
        "verify": "(equal? (let ([ws (aic-weights '(100 102 105 110))]) (and (> (car ws) (cadr ws)) (> (cadr ws) (caddr ws)))) #t)",
        "difficulty": "hard",
        "tags": ["ordering"],
    },
    {
        "fn": "aic-weights",
        "prompt": "Return whether identical AIC scores produce uniform weights.",
        "gt": "(let ([ws (aic-weights '(100 100 100))]) (and (approx=? (/ 1.0 3) (car ws) 1e-10) (approx=? (/ 1.0 3) (cadr ws) 1e-10) (approx=? (/ 1.0 3) (caddr ws) 1e-10)))",
        "verify": "(equal? (let ([ws (aic-weights '(100 100 100))]) (and (approx=? (/ 1.0 3) (car ws) 1e-10) (approx=? (/ 1.0 3) (cadr ws) 1e-10) (approx=? (/ 1.0 3) (caddr ws) 1e-10))) #t)",
        "difficulty": "medium",
        "tags": ["edge-case"],
    },
    {
        "fn": "aic-weights",
        "prompt": "Return whether evidence ratio from first two Akaike weights exceeds 1 for AIC values '(100 102 105 110).",
        "gt": "(let* ([ws (aic-weights '(100 102 105 110))] [w0 (car ws)] [w1 (cadr ws)]) (> (evidence-ratio w0 w1) 1.0))",
        "verify": "(equal? (let* ([ws (aic-weights '(100 102 105 110))] [w0 (car ws)] [w1 (cadr ws)]) (> (evidence-ratio w0 w1) 1.0)) #t)",
        "difficulty": "hard",
        "tags": ["integration"],
    },

    # evidence-ratio
    {
        "fn": "evidence-ratio",
        "prompt": "Compute evidence ratio for weights 0.6 and 0.3.",
        "gt": "(evidence-ratio 0.6 0.3)",
        "verify": "(approx=? 2.0 (evidence-ratio 0.6 0.3) 1e-10)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "evidence-ratio",
        "prompt": "Compute evidence ratio when denominator is 0.",
        "gt": "(evidence-ratio 0.5 0.0)",
        "verify": "(= (evidence-ratio 0.5 0.0) +inf.0)",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },
    {
        "fn": "evidence-ratio",
        "prompt": "Return whether evidence ratio is greater than 1 when numerator weight exceeds denominator weight.",
        "gt": "(> (evidence-ratio 0.7 0.2) 1.0)",
        "verify": "(equal? (> (evidence-ratio 0.7 0.2) 1.0) #t)",
        "difficulty": "easy",
        "tags": ["property"],
    },
    {
        "fn": "evidence-ratio",
        "prompt": "Return whether forward and reverse evidence ratios are multiplicative inverses for positive weights.",
        "gt": "(approx=? 1.0 (* (evidence-ratio 0.8 0.2) (evidence-ratio 0.2 0.8)) 1e-10)",
        "verify": "(equal? (approx=? 1.0 (* (evidence-ratio 0.8 0.2) (evidence-ratio 0.2 0.8)) 1e-10) #t)",
        "difficulty": "medium",
        "tags": ["property"],
    },

    # residual-entropy-bits
    {
        "fn": "residual-entropy-bits",
        "prompt": "Compute residual entropy bits for residuals '(1.0 -0.5 0.3 -0.8 0.2).",
        "gt": "(residual-entropy-bits '(1.0 -0.5 0.3 -0.8 0.2))",
        "verify": "(> (residual-entropy-bits '(1.0 -0.5 0.3 -0.8 0.2)) 0)",
        "difficulty": "medium",
        "tags": ["direct"],
    },
    {
        "fn": "residual-entropy-bits",
        "prompt": "Return whether tighter residuals have lower entropy than wide residuals.",
        "gt": "(< (residual-entropy-bits '(0.01 -0.01 0.005 -0.008)) (residual-entropy-bits '(5.0 -3.0 4.0 -2.0)))",
        "verify": "(equal? (< (residual-entropy-bits '(0.01 -0.01 0.005 -0.008)) (residual-entropy-bits '(5.0 -3.0 4.0 -2.0))) #t)",
        "difficulty": "medium",
        "tags": ["comparison"],
    },
    {
        "fn": "residual-entropy-bits",
        "prompt": "Compute residual entropy bits for perfect residuals '(0 0 0 0).",
        "gt": "(residual-entropy-bits '(0 0 0 0))",
        "verify": "(= (residual-entropy-bits '(0 0 0 0)) -inf.0)",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },
    {
        "fn": "residual-entropy-bits",
        "prompt": "Return whether scaling residuals by 10 increases residual entropy.",
        "gt": "(let ([base '(0.4 -0.2 0.1 -0.3)] [scaled '(4.0 -2.0 1.0 -3.0)]) (> (residual-entropy-bits scaled) (residual-entropy-bits base)))",
        "verify": "(equal? (let ([base '(0.4 -0.2 0.1 -0.3)] [scaled '(4.0 -2.0 1.0 -3.0)]) (> (residual-entropy-bits scaled) (residual-entropy-bits base))) #t)",
        "difficulty": "medium",
        "tags": ["scaling"],
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
