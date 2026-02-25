#!/usr/bin/env python3
"""Generate Tier-1 entropy SFT samples for lattice/info/entropy.ss."""

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

SOURCE_MODULE = "lattice/info/entropy.ss"
SOURCE_TEST = "lattice/info/test-entropy.ss"

DEFS: Dict[str, str] = {
    "entropy": """(define (entropy probs)
  (- (fold-left + 0 (map plogp probs))))""",
    "entropy-normalized": """(define (entropy-normalized weights)
  (let* ([total (fold-left + 0 weights)]
         [probs (if (= total 0)
                    weights
                    (map (lambda (w) (/ w total)) weights))])
        (entropy probs)))""",
    "binary-entropy": """(define (binary-entropy p)
  (cond
   [(<= p 0) 0]
   [(>= p 1) 0]
   [else (- (+ (plogp p) (plogp (- 1 p))))]))""",
    "mutual-information": """(define (mutual-information joint-probs marginal-x marginal-y)
  (let ([h-x (entropy marginal-x)]
        [h-y (entropy marginal-y)]
        [h-joint (joint-entropy joint-probs)])
       (+ h-x (- h-y h-joint))))""",
    "cross-entropy": """(define (cross-entropy p q)
  (if (null? p)
      0
      (if (any2 (lambda (pi qi) (and (> pi 0) (<= qi 0))) p q)
          +inf.0
          (- (fold-left + 0
                        (map (lambda (pi qi)
                                     (if (<= pi 0)
                                         0
                                         (* pi (entropy-log2 qi))))
                             p q))))))""",
    "kl-divergence": """(define (kl-divergence p q)
  (if (null? p)
      0
      (fold-left + 0
                 (map (lambda (pi qi)
                              (cond
                               [(<= pi 0) 0]
                               [(<= qi 0) +inf.0]
                               [else (* pi (log2 (/ pi qi)))]))
                      p q))))""",
    "jensen-shannon-divergence": """(define (jensen-shannon-divergence p q)
  (let ([m (map (lambda (pi qi) (/ (+ pi qi) 2)) p q)])
       (/ (+ (kl-divergence p m) (kl-divergence q m)) 2)))""",
    "renyi-entropy": """(define (renyi-entropy alpha probs)
  (cond
   [(< alpha 0) +inf.0]
   [(= alpha 1) (entropy probs)]
   [(= alpha 0) (log2 (length (filter (lambda (p) (> p 0)) probs)))]
   [else
    (let ([sum-p-alpha (fold-left + 0 (map (lambda (p) (expt p alpha)) probs))])
         (* (/ 1 (- 1 alpha)) (log2 sum-p-alpha)))]))""",
}

SUPPORT_DEFS: Dict[str, str] = {
    "approx=?": """(define (approx=? expected actual tol)
  (< (abs (- expected actual)) tol))""",
    "entropy-log2": """(define (entropy-log2 x)
  (if (<= x 0)
      0
      (log2 x)))""",
    "any2": """(define (any2 pred xs ys)
  (cond
   [(null? xs) #f]
   [(null? ys) #f]
   [(pred (car xs) (car ys)) #t]
   [else (any2 pred (cdr xs) (cdr ys))]))""",
    "plogp": """(define (plogp p)
  (if (<= p 0)
      0
      (* p (entropy-log2 p))))""",
    "joint-entropy": """(define (joint-entropy joint-probs)
  (let ([flat (flatten joint-probs)])
       (entropy flat)))""",
}

ALL_DEFS: Dict[str, str] = {**SUPPORT_DEFS, **DEFS}

DEPENDS: Dict[str, List[str]] = {
    "approx=?": [],
    "entropy-log2": [],
    "any2": [],
    "plogp": ["entropy-log2"],
    "joint-entropy": ["entropy"],
    "entropy": ["plogp"],
    "entropy-normalized": ["entropy"],
    "binary-entropy": ["plogp"],
    "mutual-information": ["entropy", "joint-entropy"],
    "cross-entropy": ["any2", "entropy-log2"],
    "kl-divergence": [],
    "jensen-shannon-divergence": ["kl-divergence"],
    "renyi-entropy": ["entropy"],
}

FUNCTION_ORDER = [
    "entropy",
    "entropy-normalized",
    "binary-entropy",
    "mutual-information",
    "cross-entropy",
    "kl-divergence",
    "jensen-shannon-divergence",
    "renyi-entropy",
]

SUPPORT_ORDER = [
    "approx=?",
    "entropy-log2",
    "any2",
    "plogp",
    "joint-entropy",
]

FUNCTION_SPECS = {
    "entropy": "Compute Shannon entropy H(X) = -sum(p_i * log2(p_i)) with 0*log2(0) handled as 0.",
    "entropy-normalized": "Normalize raw non-negative weights to probabilities (when total != 0), then compute Shannon entropy.",
    "binary-entropy": "Compute binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p) with boundary values p<=0 or p>=1 returning 0.",
    "mutual-information": "Compute I(X;Y) = H(X) + H(Y) - H(X,Y) from joint distribution and marginals.",
    "cross-entropy": "Compute cross entropy H(P,Q) with support-mismatch behavior returning +inf.0 when P has mass where Q is zero.",
    "kl-divergence": "Compute D_KL(P||Q) with 0-probability convention and +inf.0 when Q has zero where P is positive.",
    "jensen-shannon-divergence": "Compute JSD(P,Q) using midpoint distribution M=(P+Q)/2 and average KL divergence.",
    "renyi-entropy": "Compute Renyi entropy H_alpha including alpha<0 => +inf.0, alpha=1 => Shannon entropy, alpha=0 => Hartley entropy.",
}

SKELETONS = {
    "entropy": """(define (entropy probs)
  ;; TODO: Shannon entropy from probability list probs
  <TODO>)""",
    "entropy-normalized": """(define (entropy-normalized weights)
  ;; TODO: normalize weights when sum is non-zero, then call entropy
  <TODO>)""",
    "binary-entropy": """(define (binary-entropy p)
  ;; TODO: implement boundary-safe binary entropy
  <TODO>)""",
    "mutual-information": """(define (mutual-information joint-probs marginal-x marginal-y)
  ;; TODO: combine H(X), H(Y), and H(X,Y)
  <TODO>)""",
    "cross-entropy": """(define (cross-entropy p q)
  ;; TODO: return +inf.0 on support mismatch; otherwise compute -sum p_i*log2(q_i)
  <TODO>)""",
    "kl-divergence": """(define (kl-divergence p q)
  ;; TODO: implement D_KL(P||Q) with qi<=0 inf handling
  <TODO>)""",
    "jensen-shannon-divergence": """(define (jensen-shannon-divergence p q)
  ;; TODO: compute midpoint distribution and average two KL terms
  <TODO>)""",
    "renyi-entropy": """(define (renyi-entropy alpha probs)
  ;; TODO: handle alpha special cases and generic Renyi formula
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "entropy": "(and (approx=? 1.0 (entropy '(0.5 0.5)) 0.000001) (approx=? 2.0 (entropy '(0.25 0.25 0.25 0.25)) 0.000001) (approx=? 0.0 (entropy '(1 0 0 0)) 0.000001))",
    "entropy-normalized": "(and (approx=? 1.0 (entropy-normalized '(2 2)) 0.000001) (approx=? 0.811278 (entropy-normalized '(9 3)) 0.001) (approx=? 0.0 (entropy-normalized '(0 0 0)) 0.000001))",
    "binary-entropy": "(and (approx=? 1.0 (binary-entropy 0.5) 0.000001) (approx=? 0.0 (binary-entropy 0.0) 0.000001) (approx=? (binary-entropy 0.1) (binary-entropy 0.9) 0.000001))",
    "mutual-information": "(let ([independent '((0.25 0.25) (0.25 0.25))] [correlated '((0.5 0) (0 0.5))] [m '(0.5 0.5)]) (and (approx=? 0.0 (mutual-information independent m m) 0.000001) (approx=? 1.0 (mutual-information correlated m m) 0.000001)))",
    "cross-entropy": "(and (approx=? 1.0 (cross-entropy '(0.5 0.5) '(0.5 0.5)) 0.000001) (= (cross-entropy '(0.5 0.5) '(1.0 0.0)) +inf.0) (approx=? 0.0 (cross-entropy '() '()) 0.000001))",
    "kl-divergence": "(and (approx=? 0.0 (kl-divergence '(0.5 0.5) '(0.5 0.5)) 0.000001) (>= (kl-divergence '(0.9 0.1) '(0.5 0.5)) 0) (= (kl-divergence '(0.5 0.5) '(1.0 0.0)) +inf.0))",
    "jensen-shannon-divergence": "(let* ([p '(0.9 0.1)] [q '(0.5 0.5)] [same (jensen-shannon-divergence '(0.5 0.5) '(0.5 0.5))] [d1 (jensen-shannon-divergence p q)] [d2 (jensen-shannon-divergence q p)]) (and (approx=? 0.0 same 0.000001) (approx=? d1 d2 0.000001) (<= d1 1.0) (>= d1 0.0)))",
    "renyi-entropy": "(let ([p '(0.5 0.5)] [q '(0.7 0.2 0.1)]) (and (approx=? (entropy p) (renyi-entropy 1 p) 0.000001) (approx=? 1.0 (renyi-entropy 0 p) 0.000001) (approx=? 1.0 (renyi-entropy 2 p) 0.000001) (= (renyi-entropy -1 q) +inf.0)))",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")

TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "entropy": """def entropy(probs):
    return -sum((0 if p <= 0 else p * log2(p)) for p in probs)""",
    "entropy-normalized": """def entropy_normalized(weights):
    total = sum(weights)
    probs = weights if total == 0 else [w / total for w in weights]
    return entropy(probs)""",
    "binary-entropy": """def binary_entropy(p):
    if p <= 0 or p >= 1:
        return 0
    return -(p * log2(p) + (1 - p) * log2(1 - p))""",
    "mutual-information": """def mutual_information(joint_probs, marginal_x, marginal_y):
    h_x = entropy(marginal_x)
    h_y = entropy(marginal_y)
    h_joint = entropy([v for row in joint_probs for v in row])
    return h_x + h_y - h_joint""",
    "cross-entropy": """def cross_entropy(p, q):
    if len(p) == 0:
        return 0
    if any((pi > 0 and qi <= 0) for pi, qi in zip(p, q)):
        return float('inf')
    return -sum((0 if pi <= 0 else pi * log2(qi)) for pi, qi in zip(p, q))""",
    "kl-divergence": """def kl_divergence(p, q):
    if len(p) == 0:
        return 0
    out = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0:
            continue
        if qi <= 0:
            return float('inf')
        out += pi * log2(pi / qi)
    return out""",
    "jensen-shannon-divergence": """def jensen_shannon_divergence(p, q):
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    return (kl_divergence(p, m) + kl_divergence(q, m)) / 2""",
    "renyi-entropy": """def renyi_entropy(alpha, probs):
    if alpha < 0:
        return float('inf')
    if alpha == 1:
        return entropy(probs)
    if alpha == 0:
        return log2(sum(1 for p in probs if p > 0))
    return (1 / (1 - alpha)) * log2(sum((p ** alpha) for p in probs))""",
}

CHEZ_SNIPPETS = {
    "entropy": """(define (entropy0 probs)
  (let loop ([xs probs] [acc 0])
    (if (null? xs)
        (- acc)
        (loop (cdr xs) (+ acc (plogp (car xs)))))))""",
    "entropy-normalized": """(define (entropy-normalized0 weights)
  (let ([total (fold-left + 0 weights)])
    (if (= total 0)
        (entropy weights)
        (entropy (map (lambda (w) (/ w total)) weights)))))""",
    "binary-entropy": """(define (binary-entropy0 p)
  (if (or (<= p 0) (>= p 1))
      0
      (let ([q (- 1 p)])
        (- (+ (plogp p) (plogp q))))))""",
    "mutual-information": """(define (mutual-information0 joint-probs marginal-x marginal-y)
  (let* ([h-x (entropy marginal-x)]
         [h-y (entropy marginal-y)]
         [flat (flatten joint-probs)]
         [h-joint (entropy flat)])
    (- (+ h-x h-y) h-joint)))""",
    "cross-entropy": """(define (cross-entropy0 p q)
  (if (null? p)
      0
      (let loop ([ps p] [qs q] [acc 0])
        (cond
          [(null? ps) (- acc)]
          [(and (> (car ps) 0) (<= (car qs) 0)) +inf.0]
          [(<= (car ps) 0) (loop (cdr ps) (cdr qs) acc)]
          [else
           (loop (cdr ps)
                 (cdr qs)
                 (+ acc (* (car ps) (entropy-log2 (car qs)))))]))))""",
    "kl-divergence": """(define (kl-divergence0 p q)
  (if (null? p)
      0
      (let loop ([ps p] [qs q] [acc 0])
        (cond
          [(null? ps) acc]
          [(<= (car ps) 0) (loop (cdr ps) (cdr qs) acc)]
          [(<= (car qs) 0) +inf.0]
          [else
           (loop (cdr ps)
                 (cdr qs)
                 (+ acc (* (car ps) (log2 (/ (car ps) (car qs))))))]))))""",
    "jensen-shannon-divergence": """(define (jensen-shannon0 p q)
  (let* ([mid (map (lambda (pi qi) (/ (+ pi qi) 2)) p q)]
         [dp (kl-divergence p mid)]
         [dq (kl-divergence q mid)])
    (/ (+ dp dq) 2)))""",
    "renyi-entropy": """(define (renyi-entropy0 alpha probs)
  (cond
    [(< alpha 0) +inf.0]
    [(= alpha 1) (entropy probs)]
    [(= alpha 0)
     (let loop ([xs probs] [n 0])
       (if (null? xs)
           (log2 n)
           (loop (cdr xs) (if (> (car xs) 0) (+ n 1) n))))]
    [else
      (let loop ([xs probs] [sum 0])
        (if (null? xs)
            (* (/ 1 (- 1 alpha)) (log2 sum))
            (loop (cdr xs) (+ sum (expt (car xs) alpha)))))]))""",
}

BUGGY_CASES = [
    {
        "fn": "entropy",
        "buggy": """(define (entropy probs)
  (fold-left + 0 (map plogp probs)))""",
        "note": "Entropy must negate the accumulated p*log2(p) sum.",
    },
    {
        "fn": "entropy",
        "buggy": """(define (entropy probs)
  (- (fold-left + 0 (map entropy-log2 probs))))""",
        "note": "Each term must be weighted by p_i, not just log2(p_i).",
    },
    {
        "fn": "entropy-normalized",
        "buggy": """(define (entropy-normalized weights)
  (let* ([total (length weights)]
         [probs (if (= total 0)
                    weights
                    (map (lambda (w) (/ w total)) weights))])
        (entropy probs)))""",
        "note": "Normalization denominator must be sum(weights), not length(weights).",
    },
    {
        "fn": "entropy-normalized",
        "buggy": """(define (entropy-normalized weights)
  (entropy weights))""",
        "note": "Function must normalize non-zero weights before calling entropy.",
    },
    {
        "fn": "binary-entropy",
        "buggy": """(define (binary-entropy p)
  (cond
   [(<= p 0) 0]
   [(>= p 1) 0]
   [else (+ (plogp p) (plogp (- 1 p)))]))""",
        "note": "Binary entropy requires a leading negation of both plogp terms.",
    },
    {
        "fn": "binary-entropy",
        "buggy": """(define (binary-entropy p)
  (cond
   [(<= p 0) 0]
   [(>= p 1) 0]
   [else (- (+ (plogp p) (plogp p)))]))""",
        "note": "Second term must use (1-p), not p again.",
    },
    {
        "fn": "mutual-information",
        "buggy": """(define (mutual-information joint-probs marginal-x marginal-y)
  (let ([h-x (entropy marginal-x)]
        [h-y (entropy marginal-y)]
        [h-joint (joint-entropy joint-probs)])
       (+ h-x h-y h-joint)))""",
        "note": "Mutual information subtracts joint entropy; it does not add it.",
    },
    {
        "fn": "mutual-information",
        "buggy": """(define (mutual-information joint-probs marginal-x marginal-y)
  (let ([h-x (entropy marginal-x)]
        [h-joint (joint-entropy joint-probs)])
       (- h-x h-joint)))""",
        "note": "Formula must include both marginals: H(X)+H(Y)-H(X,Y).",
    },
    {
        "fn": "cross-entropy",
        "buggy": """(define (cross-entropy p q)
  (if (null? p)
      0
      (- (fold-left + 0
                    (map (lambda (pi qi)
                                 (if (<= pi 0)
                                     0
                                     (* pi (entropy-log2 qi))))
                         p q)))))""",
        "note": "Support mismatch (pi>0 with qi<=0) must return +inf.0.",
    },
    {
        "fn": "cross-entropy",
        "buggy": """(define (cross-entropy p q)
  (if (null? p)
      0
      (fold-left + 0
                 (map (lambda (pi qi)
                              (if (or (<= pi 0) (<= qi 0))
                                  0
                                  (* pi (log2 (/ pi qi)))))
                      p q))))""",
        "note": "This computes KL-like terms; cross entropy must use -pi*log2(qi).",
    },
    {
        "fn": "kl-divergence",
        "buggy": """(define (kl-divergence p q)
  (if (null? p)
      0
      (fold-left + 0
                 (map (lambda (pi qi)
                              (cond
                               [(<= pi 0) 0]
                               [(<= qi 0) +inf.0]
                               [else (* pi (log2 (/ qi pi)))]))
                      p q))))""",
        "note": "Ratio direction is reversed; must be log2(pi/qi).",
    },
    {
        "fn": "kl-divergence",
        "buggy": """(define (kl-divergence p q)
  (if (null? p)
      0
      (fold-left + 0
                 (map (lambda (pi qi)
                              (cond
                               [(<= pi 0) 0]
                               [(<= qi 0) 0]
                               [else (* pi (log2 (/ pi qi)))]))
                      p q))))""",
        "note": "When qi<=0 and pi>0, KL divergence must be +inf.0, not 0.",
    },
    {
        "fn": "jensen-shannon-divergence",
        "buggy": """(define (jensen-shannon-divergence p q)
  (let ([m (map (lambda (pi qi) (+ pi qi)) p q)])
       (/ (+ (kl-divergence p m) (kl-divergence q m)) 2)))""",
        "note": "Midpoint distribution must average each coordinate: (pi+qi)/2.",
    },
    {
        "fn": "jensen-shannon-divergence",
        "buggy": """(define (jensen-shannon-divergence p q)
  (let ([m (map (lambda (pi qi) (/ (+ pi qi) 2)) p q)])
       (+ (kl-divergence p m) (kl-divergence q m))))""",
        "note": "JSD is the average of two KL terms, so divide by 2.",
    },
    {
        "fn": "renyi-entropy",
        "buggy": """(define (renyi-entropy alpha probs)
  (cond
   [(< alpha 0) +inf.0]
   [(= alpha 0) (log2 (length (filter (lambda (p) (> p 0)) probs)))]
   [else
    (let ([sum-p-alpha (fold-left + 0 (map (lambda (p) (expt p alpha)) probs))])
         (* (/ 1 (- 1 alpha)) (log2 sum-p-alpha)))]))""",
        "note": "alpha=1 must dispatch to Shannon entropy to avoid division-by-zero behavior.",
    },
    {
        "fn": "renyi-entropy",
        "buggy": """(define (renyi-entropy alpha probs)
  (cond
   [(< alpha 0) +inf.0]
   [(= alpha 1) (entropy probs)]
   [(= alpha 0) (log2 (length probs))]
   [else
    (let ([sum-p-alpha (fold-left + 0 (map (lambda (p) (expt p alpha)) probs))])
         (* (/ 1 (- 1 alpha)) (log2 sum-p-alpha)))]))""",
        "note": "Hartley case (alpha=0) must count positive-support outcomes only.",
    },
]

DIFFICULTY = {
    "entropy": "easy",
    "entropy-normalized": "medium",
    "binary-entropy": "easy",
    "mutual-information": "hard",
    "cross-entropy": "hard",
    "kl-divergence": "hard",
    "jensen-shannon-divergence": "hard",
    "renyi-entropy": "hard",
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
    sid = f"info_entropy_{family}_{family_counter[family]:03d}"
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


# Dependency-closure fix: scan verify expression symbols, then include refs.
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
    verify_check = VERIFY_BY_FUNCTION[fn]
    wanted: List[str] = []

    for dep in DEPENDS.get(fn, []):
        if dep in ALL_DEFS and dep not in wanted:
            wanted.append(dep)

    for ref in verify_refs(verify_check):
        if ref in ALL_DEFS and ref != fn and ref not in wanted:
            wanted.append(ref)

    defs_needed = dependency_closure(wanted)
    parts = [ALL_DEFS[name] for name in defs_needed] + [verify_check.strip()]
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
        prompt=f"""Implement this entropy utility in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "info", "entropy", "spec-to-code", fn],
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
        tags=["tier1", "info", "entropy", "skeleton-completion", fn],
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
        tags=["tier1", "info", "entropy", "python-to-scheme", fn],
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
        tags=["tier1", "info", "entropy", "chez-to-fold", fn],
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
        tags=["tier1", "info", "entropy", "bugfix", fn],
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
        tags=["tier1", "info", "entropy", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # entropy
    {
        "fn": "entropy",
        "prompt": "Compute entropy of a fair die distribution.",
        "gt": "(entropy '(1/6 1/6 1/6 1/6 1/6 1/6))",
        "verify": "(approx=? 2.584962 (entropy '(1/6 1/6 1/6 1/6 1/6 1/6)) 0.0001)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "entropy",
        "prompt": "Compute entropy for a deterministic outcome '(1 0 0 0).",
        "gt": "(entropy '(1 0 0 0))",
        "verify": "(approx=? 0.0 (entropy '(1 0 0 0)) 0.000001)",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },
    {
        "fn": "entropy",
        "prompt": "Return whether entropy of a 4-way uniform distribution equals 2 bits.",
        "gt": "(approx=? 2.0 (entropy '(0.25 0.25 0.25 0.25)) 0.000001)",
        "verify": "(equal? (approx=? 2.0 (entropy '(0.25 0.25 0.25 0.25)) 0.000001) #t)",
        "difficulty": "medium",
        "tags": ["property"],
    },
    {
        "fn": "entropy",
        "prompt": "Return whether Renyi entropy at alpha=1 matches entropy for '(0.6 0.4).",
        "gt": "(approx=? (entropy '(0.6 0.4)) (renyi-entropy 1 '(0.6 0.4)) 0.000001)",
        "verify": "(equal? (approx=? (entropy '(0.6 0.4)) (renyi-entropy 1 '(0.6 0.4)) 0.000001) #t)",
        "difficulty": "hard",
        "tags": ["integration"],
    },

    # entropy-normalized
    {
        "fn": "entropy-normalized",
        "prompt": "Compute normalized entropy for weights '(2 2 2 2).",
        "gt": "(entropy-normalized '(2 2 2 2))",
        "verify": "(approx=? 2.0 (entropy-normalized '(2 2 2 2)) 0.000001)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "entropy-normalized",
        "prompt": "Return whether scaling all weights leaves entropy-normalized unchanged.",
        "gt": "(approx=? (entropy-normalized '(2 1 1)) (entropy-normalized '(20 10 10)) 0.000001)",
        "verify": "(equal? (approx=? (entropy-normalized '(2 1 1)) (entropy-normalized '(20 10 10)) 0.000001) #t)",
        "difficulty": "medium",
        "tags": ["invariance"],
    },
    {
        "fn": "entropy-normalized",
        "prompt": "Compute entropy-normalized for all-zero weights '(0 0 0).",
        "gt": "(entropy-normalized '(0 0 0))",
        "verify": "(approx=? 0.0 (entropy-normalized '(0 0 0)) 0.000001)",
        "difficulty": "medium",
        "tags": ["edge-case"],
    },
    {
        "fn": "entropy-normalized",
        "prompt": "Return whether entropy-normalized '(9 3) matches entropy of '(0.75 0.25).",
        "gt": "(approx=? (entropy-normalized '(9 3)) (entropy '(0.75 0.25)) 0.000001)",
        "verify": "(equal? (approx=? (entropy-normalized '(9 3)) (entropy '(0.75 0.25)) 0.000001) #t)",
        "difficulty": "medium",
        "tags": ["integration"],
    },

    # binary-entropy
    {
        "fn": "binary-entropy",
        "prompt": "Compute binary entropy at p=0.25.",
        "gt": "(binary-entropy 0.25)",
        "verify": "(approx=? 0.811278 (binary-entropy 0.25) 0.001)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "binary-entropy",
        "prompt": "Return whether binary entropy is symmetric around 0.5 for p=0.2.",
        "gt": "(approx=? (binary-entropy 0.2) (binary-entropy 0.8) 0.000001)",
        "verify": "(equal? (approx=? (binary-entropy 0.2) (binary-entropy 0.8) 0.000001) #t)",
        "difficulty": "medium",
        "tags": ["property"],
    },
    {
        "fn": "binary-entropy",
        "prompt": "Return whether H(0.5) is at least H(0.3).",
        "gt": "(>= (binary-entropy 0.5) (binary-entropy 0.3))",
        "verify": "(equal? (>= (binary-entropy 0.5) (binary-entropy 0.3)) #t)",
        "difficulty": "medium",
        "tags": ["ordering"],
    },
    {
        "fn": "binary-entropy",
        "prompt": "Return the pair (H(0) H(1)).",
        "gt": "(list (binary-entropy 0) (binary-entropy 1))",
        "verify": "(equal? (list (binary-entropy 0) (binary-entropy 1)) '(0 0))",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },

    # mutual-information
    {
        "fn": "mutual-information",
        "prompt": "Compute I(X;Y) for independent fair binary variables.",
        "gt": "(mutual-information '((0.25 0.25) (0.25 0.25)) '(0.5 0.5) '(0.5 0.5))",
        "verify": "(approx=? 0.0 (mutual-information '((0.25 0.25) (0.25 0.25)) '(0.5 0.5) '(0.5 0.5)) 0.000001)",
        "difficulty": "medium",
        "tags": ["direct"],
    },
    {
        "fn": "mutual-information",
        "prompt": "Compute I(X;Y) when X=Y for a fair binary source.",
        "gt": "(mutual-information '((0.5 0) (0 0.5)) '(0.5 0.5) '(0.5 0.5))",
        "verify": "(approx=? 1.0 (mutual-information '((0.5 0) (0 0.5)) '(0.5 0.5) '(0.5 0.5)) 0.000001)",
        "difficulty": "hard",
        "tags": ["direct"],
    },
    {
        "fn": "mutual-information",
        "prompt": "Return whether mutual information is non-negative on a valid 2x2 joint distribution.",
        "gt": "(>= (mutual-information '((0.4 0.1) (0.1 0.4)) '(0.5 0.5) '(0.5 0.5)) 0)",
        "verify": "(equal? (>= (mutual-information '((0.4 0.1) (0.1 0.4)) '(0.5 0.5) '(0.5 0.5)) 0) #t)",
        "difficulty": "hard",
        "tags": ["property"],
    },
    {
        "fn": "mutual-information",
        "prompt": "Return whether I(X;Y) equals H(X) under perfect correlation for fair binary marginals.",
        "gt": "(approx=? (mutual-information '((0.5 0) (0 0.5)) '(0.5 0.5) '(0.5 0.5)) (entropy '(0.5 0.5)) 0.000001)",
        "verify": "(equal? (approx=? (mutual-information '((0.5 0) (0 0.5)) '(0.5 0.5) '(0.5 0.5)) (entropy '(0.5 0.5)) 0.000001) #t)",
        "difficulty": "hard",
        "tags": ["integration"],
    },

    # cross-entropy
    {
        "fn": "cross-entropy",
        "prompt": "Compute cross-entropy of identical fair-coin distributions.",
        "gt": "(cross-entropy '(0.5 0.5) '(0.5 0.5))",
        "verify": "(approx=? 1.0 (cross-entropy '(0.5 0.5) '(0.5 0.5)) 0.000001)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "cross-entropy",
        "prompt": "Compute cross-entropy under support mismatch where Q has a zero-probability event with positive mass in P.",
        "gt": "(cross-entropy '(0.5 0.5) '(1.0 0.0))",
        "verify": "(equal? (cross-entropy '(0.5 0.5) '(1.0 0.0)) +inf.0)",
        "difficulty": "hard",
        "tags": ["edge-case"],
    },
    {
        "fn": "cross-entropy",
        "prompt": "Compute cross-entropy for empty distributions.",
        "gt": "(cross-entropy '() '())",
        "verify": "(approx=? 0.0 (cross-entropy '() '()) 0.000001)",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },
    {
        "fn": "cross-entropy",
        "prompt": "Return whether H(P,Q) is at least H(P) for P='(0.8 0.2) and Q='(0.5 0.5).",
        "gt": "(let ([p '(0.8 0.2)] [q '(0.5 0.5)]) (>= (cross-entropy p q) (entropy p)))",
        "verify": "(equal? (let ([p '(0.8 0.2)] [q '(0.5 0.5)]) (>= (cross-entropy p q) (entropy p))) #t)",
        "difficulty": "hard",
        "tags": ["property"],
    },

    # kl-divergence
    {
        "fn": "kl-divergence",
        "prompt": "Compute D_KL(P||P) for P='(0.5 0.5).",
        "gt": "(kl-divergence '(0.5 0.5) '(0.5 0.5))",
        "verify": "(approx=? 0.0 (kl-divergence '(0.5 0.5) '(0.5 0.5)) 0.000001)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "kl-divergence",
        "prompt": "Return whether KL divergence is asymmetric for P='(0.9 0.1) and Q='(0.5 0.5).",
        "gt": "(let ([d1 (kl-divergence '(0.9 0.1) '(0.5 0.5))] [d2 (kl-divergence '(0.5 0.5) '(0.9 0.1))]) (not (approx=? d1 d2 0.000001)))",
        "verify": "(equal? (let ([d1 (kl-divergence '(0.9 0.1) '(0.5 0.5))] [d2 (kl-divergence '(0.5 0.5) '(0.9 0.1))]) (not (approx=? d1 d2 0.000001))) #t)",
        "difficulty": "hard",
        "tags": ["property"],
    },
    {
        "fn": "kl-divergence",
        "prompt": "Compute KL divergence under support mismatch where Q has a zero-probability event used by P.",
        "gt": "(kl-divergence '(0.5 0.5) '(1.0 0.0))",
        "verify": "(equal? (kl-divergence '(0.5 0.5) '(1.0 0.0)) +inf.0)",
        "difficulty": "hard",
        "tags": ["edge-case"],
    },
    {
        "fn": "kl-divergence",
        "prompt": "Return whether cross-entropy equals entropy plus KL divergence for P='(0.8 0.2), Q='(0.5 0.5).",
        "gt": "(let ([p '(0.8 0.2)] [q '(0.5 0.5)]) (approx=? (cross-entropy p q) (+ (entropy p) (kl-divergence p q)) 0.000001))",
        "verify": "(equal? (let ([p '(0.8 0.2)] [q '(0.5 0.5)]) (approx=? (cross-entropy p q) (+ (entropy p) (kl-divergence p q)) 0.000001)) #t)",
        "difficulty": "hard",
        "tags": ["integration"],
    },

    # jensen-shannon-divergence
    {
        "fn": "jensen-shannon-divergence",
        "prompt": "Compute JSD(P,P) for P='(0.5 0.5).",
        "gt": "(jensen-shannon-divergence '(0.5 0.5) '(0.5 0.5))",
        "verify": "(approx=? 0.0 (jensen-shannon-divergence '(0.5 0.5) '(0.5 0.5)) 0.000001)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "jensen-shannon-divergence",
        "prompt": "Return whether Jensen-Shannon divergence is symmetric for P='(0.9 0.1), Q='(0.5 0.5).",
        "gt": "(let ([d1 (jensen-shannon-divergence '(0.9 0.1) '(0.5 0.5))] [d2 (jensen-shannon-divergence '(0.5 0.5) '(0.9 0.1))]) (approx=? d1 d2 0.000001))",
        "verify": "(equal? (let ([d1 (jensen-shannon-divergence '(0.9 0.1) '(0.5 0.5))] [d2 (jensen-shannon-divergence '(0.5 0.5) '(0.9 0.1))]) (approx=? d1 d2 0.000001)) #t)",
        "difficulty": "medium",
        "tags": ["property"],
    },
    {
        "fn": "jensen-shannon-divergence",
        "prompt": "Compute JSD between opposite deterministic binary distributions.",
        "gt": "(jensen-shannon-divergence '(1 0) '(0 1))",
        "verify": "(approx=? 1.0 (jensen-shannon-divergence '(1 0) '(0 1)) 0.000001)",
        "difficulty": "hard",
        "tags": ["boundary"],
    },
    {
        "fn": "jensen-shannon-divergence",
        "prompt": "Return whether JSD is non-negative for P='(0.7 0.3), Q='(0.4 0.6).",
        "gt": "(>= (jensen-shannon-divergence '(0.7 0.3) '(0.4 0.6)) 0)",
        "verify": "(equal? (>= (jensen-shannon-divergence '(0.7 0.3) '(0.4 0.6)) 0) #t)",
        "difficulty": "medium",
        "tags": ["property"],
    },

    # renyi-entropy
    {
        "fn": "renyi-entropy",
        "prompt": "Compute Renyi entropy of order 0 for '(0.5 0.5).",
        "gt": "(renyi-entropy 0 '(0.5 0.5))",
        "verify": "(approx=? 1.0 (renyi-entropy 0 '(0.5 0.5)) 0.000001)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "renyi-entropy",
        "prompt": "Return whether Renyi entropy at alpha=1 equals Shannon entropy for '(0.7 0.3).",
        "gt": "(approx=? (renyi-entropy 1 '(0.7 0.3)) (entropy '(0.7 0.3)) 0.000001)",
        "verify": "(equal? (approx=? (renyi-entropy 1 '(0.7 0.3)) (entropy '(0.7 0.3)) 0.000001) #t)",
        "difficulty": "hard",
        "tags": ["integration"],
    },
    {
        "fn": "renyi-entropy",
        "prompt": "Compute Renyi entropy of order 2 for '(0.5 0.5).",
        "gt": "(renyi-entropy 2 '(0.5 0.5))",
        "verify": "(approx=? 1.0 (renyi-entropy 2 '(0.5 0.5)) 0.000001)",
        "difficulty": "medium",
        "tags": ["direct"],
    },
    {
        "fn": "renyi-entropy",
        "prompt": "Return whether H_2 <= H_1 for probabilities '(0.7 0.2 0.1).",
        "gt": "(<= (renyi-entropy 2 '(0.7 0.2 0.1)) (renyi-entropy 1 '(0.7 0.2 0.1)))",
        "verify": "(equal? (<= (renyi-entropy 2 '(0.7 0.2 0.1)) (renyi-entropy 1 '(0.7 0.2 0.1))) #t)",
        "difficulty": "hard",
        "tags": ["ordering"],
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
