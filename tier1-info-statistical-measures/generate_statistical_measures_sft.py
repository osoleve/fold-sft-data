#!/usr/bin/env python3
"""Generate Tier-1 statistical-measures SFT samples for lattice/info/statistical-measures.ss."""

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

SOURCE_MODULE = "lattice/info/statistical-measures.ss"
SOURCE_TEST = "lattice/info/test-statistical-measures.ss"

DEFS: Dict[str, str] = {
    "bhattacharyya-coefficient": """(define (bhattacharyya-coefficient p q)
  (if (or (null? p) (null? q))
      0
      (fold-left + 0
                 (map (lambda (pi qi)
                              (if (or (< pi 0) (< qi 0))
                                  0
                                  (sqrt (* pi qi))))
                      p q))))""",
    "bhattacharyya-distance": """(define (bhattacharyya-distance p q)
  (let ([bc (bhattacharyya-coefficient p q)])
       (if (<= bc 0)
           +inf.0
           (- (log-num bc)))))""",
    "hellinger-distance": """(define (hellinger-distance p q)
  (if (or (null? p) (null? q))
      1
      (let ([sum-sq (fold-left + 0
                               (map (lambda (pi qi)
                                            (let ([diff (- (sqrt (max 0 pi))
                                                           (sqrt (max 0 qi)))])
                                                 (* diff diff)))
                                    p q))])
           (sqrt (* 0.5 sum-sq)))))""",
    "total-variation-distance": """(define (total-variation-distance p q)
  (if (or (null? p) (null? q))
      1
      (* 0.5 (fold-left + 0
                        (map (lambda (pi qi)
                                     (abs (- pi qi)))
                             p q)))))""",
    "chi-squared-divergence": """(define (chi-squared-divergence p q)
  (if (or (null? p) (null? q))
      0
      (fold-left + 0
                 (map (lambda (pi qi)
                              (cond
                               [(<= qi 0)
                                (if (> pi 0) +inf.0 0)]
                               [else
                                (let ([diff (- pi qi)])
                                     (/ (* diff diff) qi))]))
                      p q))))""",
    "symmetric-chi-squared": """(define (symmetric-chi-squared p q)
  (let ([pq (chi-squared-divergence p q)]
        [qp (chi-squared-divergence q p)])
       (if (or (= pq +inf.0) (= qp +inf.0))
           +inf.0
           (/ (+ pq qp) 2))))""",
    "jeffreys-divergence": """(define (jeffreys-divergence p q)
  (let ([kl-pq (kl-divergence p q)]
        [kl-qp (kl-divergence q p)])
       (+ kl-pq kl-qp)))""",
    "alpha-divergence": """(define (alpha-divergence alpha p q)
  (cond
   [(or (null? p) (null? q)) 0]
   [(= alpha 1)
    (kl-divergence p q)]
   [(= alpha 0)
    (- (log-num (fold-left + 0
                           (map (lambda (pi qi)
                                        (if (and (> pi 0) (> qi 0))
                                            qi
                                            0))
                                p q))))]
   [else
    (let ([sum-term (fold-left + 0
                               (map (lambda (pi qi)
                                            (cond
                                             [(and (<= pi 0) (<= qi 0)) 0]
                                             [(or (<= pi 0) (<= qi 0)) 0]
                                             [else
                                              (* (expt pi alpha)
                                                 (expt qi (- 1 alpha)))]))
                                    p q))])
         (if (<= sum-term 0)
             +inf.0
             (/ (log-num sum-term)
                (* alpha (- alpha 1)))))]))""",
}

FUNCTION_ORDER = [
    "bhattacharyya-coefficient",
    "bhattacharyya-distance",
    "hellinger-distance",
    "total-variation-distance",
    "chi-squared-divergence",
    "symmetric-chi-squared",
    "jeffreys-divergence",
    "alpha-divergence",
]

SUPPORT_DEFS: Dict[str, str] = {
    "approx=?": """(define (approx=? expected actual tol)
  (< (abs (- expected actual)) tol))""",
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
    "hellinger-distance-from-bc": """(define (hellinger-distance-from-bc bc)
  (sqrt (- 1 (max 0 (min 1 bc)))))""",
    "squared-hellinger-distance": """(define (squared-hellinger-distance p q)
  (let ([bc (bhattacharyya-coefficient p q)])
       (- 1 bc)))""",
    "triangular-discrimination": """(define (triangular-discrimination p q)
  (if (or (null? p) (null? q))
      0
      (fold-left + 0
                 (map (lambda (pi qi)
                              (let ([sum-pq (+ pi qi)])
                                   (if (<= sum-pq 0)
                                       0
                                       (let ([diff (- pi qi)])
                                            (/ (* diff diff) sum-pq)))))
                      p q))))""",
    "squared-loss": """(define (squared-loss p q)
  (if (or (null? p) (null? q))
      0
      (fold-left + 0
                 (map (lambda (pi qi)
                              (let ([diff (- pi qi)])
                                   (* diff diff)))
                      p q))))""",
    "euclidean-distance": """(define (euclidean-distance p q)
  (sqrt (squared-loss p q)))""",
    "matusita-distance": """(define (matusita-distance p q)
  (if (or (null? p) (null? q))
      0
      (let ([sum-sq (fold-left + 0
                               (map (lambda (pi qi)
                                            (let ([diff (- (sqrt (max 0 pi))
                                                           (sqrt (max 0 qi)))])
                                                 (* diff diff)))
                                    p q))])
           (sqrt sum-sq))))""",
    "valid-distribution?": """(define (valid-distribution? p)
  (and (not (null? p))
       (andmap (lambda (pi) (>= pi 0)) p)
       (let ([sum (fold-left + 0 p)])
            (< (abs (- sum 1.0)) 1e-6))))""",
    "normalize-distribution": """(define (normalize-distribution weights)
  (let ([total (fold-left + 0 weights)])
       (if (<= total 0)
           (let ([n (length weights)])
                (map (lambda (_) (/ 1.0 n)) weights))
           (map (lambda (w) (/ w total)) weights))))""",
}

SUPPORT_ORDER = [
    "approx=?",
    "kl-divergence",
    "hellinger-distance-from-bc",
    "squared-hellinger-distance",
    "triangular-discrimination",
    "squared-loss",
    "euclidean-distance",
    "matusita-distance",
    "valid-distribution?",
    "normalize-distribution",
]

ALL_DEFS: Dict[str, str] = {**SUPPORT_DEFS, **DEFS}

FUNCTION_SPECS = {
    "bhattacharyya-coefficient": "Compute BC(P,Q)=sum(sqrt(p_i*q_i)) with empty inputs returning 0 and negative coordinates contributing 0 per term.",
    "bhattacharyya-distance": "Compute D_B(P,Q)=-ln(BC(P,Q)) and return +inf.0 when BC<=0.",
    "hellinger-distance": "Compute Hellinger distance H(P,Q)=sqrt(0.5 * sum((sqrt(max(0,p_i))-sqrt(max(0,q_i)))^2)) with empty inputs returning 1.",
    "total-variation-distance": "Compute TV(P,Q)=0.5*sum(|p_i-q_i|) with empty inputs returning 1.",
    "chi-squared-divergence": "Compute Pearson chi-squared divergence sum((p_i-q_i)^2/q_i) with qi<=0 and pi>0 yielding +inf.0.",
    "symmetric-chi-squared": "Compute symmetric chi-squared as (chi2(P||Q)+chi2(Q||P))/2, propagating +inf.0 if either direction is infinite.",
    "jeffreys-divergence": "Compute Jeffreys divergence J(P,Q)=KL(P||Q)+KL(Q||P).",
    "alpha-divergence": "Compute alpha-divergence with branches alpha=1 -> KL, alpha=0 -> -ln(overlap mass in Q over P support), otherwise log-sum term divided by alpha(alpha-1).",
}

SKELETONS = {
    "bhattacharyya-coefficient": """(define (bhattacharyya-coefficient p q)
  ;; TODO: sum sqrt(p_i * q_i) with empty-list and negative-value handling
  <TODO>)""",
    "bhattacharyya-distance": """(define (bhattacharyya-distance p q)
  ;; TODO: compute -ln(BC), and return +inf.0 when BC <= 0
  <TODO>)""",
    "hellinger-distance": """(define (hellinger-distance p q)
  ;; TODO: implement Hellinger distance using sqrt(max 0 x) per coordinate
  <TODO>)""",
    "total-variation-distance": """(define (total-variation-distance p q)
  ;; TODO: implement TV = 1/2 * L1 distance with empty-list behavior
  <TODO>)""",
    "chi-squared-divergence": """(define (chi-squared-divergence p q)
  ;; TODO: implement chi-squared divergence with qi<=0 edge handling
  <TODO>)""",
    "symmetric-chi-squared": """(define (symmetric-chi-squared p q)
  ;; TODO: average forward and reverse chi-squared divergence
  <TODO>)""",
    "jeffreys-divergence": """(define (jeffreys-divergence p q)
  ;; TODO: sum forward and reverse KL divergence
  <TODO>)""",
    "alpha-divergence": """(define (alpha-divergence alpha p q)
  ;; TODO: handle alpha=1, alpha=0, and generic alpha branches
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "bhattacharyya-coefficient": """(let ([same (bhattacharyya-coefficient '(0.5 0.3 0.2) '(0.5 0.3 0.2))]
      [ab (bhattacharyya-coefficient '(0.5 0.3 0.2) '(0.4 0.35 0.25))]
      [ba (bhattacharyya-coefficient '(0.4 0.35 0.25) '(0.5 0.3 0.2))]
      [disjoint (bhattacharyya-coefficient '(1 0) '(0 1))])
  (and (approx=? 1.0 same 0.000001)
       (approx=? ab ba 0.000001)
       (approx=? 0.0 disjoint 0.000001)
       (<= ab 1.0)
       (>= ab 0.0)))""",
    "bhattacharyya-distance": """(let ([same (bhattacharyya-distance '(0.5 0.3 0.2) '(0.5 0.3 0.2))]
      [disjoint (bhattacharyya-distance '(1 0) '(0 1))]
      [p '(0.6 0.3 0.1)]
      [q '(0.4 0.4 0.2)])
  (let ([bc (bhattacharyya-coefficient p q)]
        [bd (bhattacharyya-distance p q)])
    (and (approx=? 0.0 same 0.000001)
         (= disjoint +inf.0)
         (approx=? (- (log-num bc)) bd 0.000001))))""",
    "hellinger-distance": """(let ([same (hellinger-distance '(0.5 0.5) '(0.5 0.5))]
      [disjoint (hellinger-distance '(1 0 0) '(0 1 0))]
      [p '(0.5 0.3 0.2)]
      [q '(0.4 0.35 0.25)])
  (let ([h (hellinger-distance p q)]
        [bc (bhattacharyya-coefficient p q)])
    (and (approx=? 0.0 same 0.000001)
         (approx=? 1.0 disjoint 0.000001)
         (approx=? (* h h) (- 1 bc) 0.000001)
         (<= h 1.0)
         (>= h 0.0))))""",
    "total-variation-distance": """(let ([tv-same (total-variation-distance '(0.5 0.5) '(0.5 0.5))]
      [tv-known (total-variation-distance '(0.5 0.5) '(0.8 0.2))]
      [p '(0.5 0.3 0.2)]
      [q '(0.4 0.35 0.25)])
  (let ([tv1 (total-variation-distance p q)]
        [tv2 (total-variation-distance q p)])
    (and (approx=? 0.0 tv-same 0.000001)
         (approx=? 0.3 tv-known 0.000001)
         (approx=? tv1 tv2 0.000001)
         (<= tv1 1.0)
         (>= tv1 0.0))))""",
    "chi-squared-divergence": """(let ([same (chi-squared-divergence '(0.5 0.3 0.2) '(0.5 0.3 0.2))]
      [inf-case (chi-squared-divergence '(0.5 0.5) '(1.0 0.0))]
      [p '(0.5 0.3 0.2)]
      [q '(0.4 0.35 0.25)])
  (let ([chi (chi-squared-divergence p q)])
    (and (approx=? 0.0 same 0.000001)
         (= inf-case +inf.0)
         (>= chi 0))))""",
    "symmetric-chi-squared": """(let ([same (symmetric-chi-squared '(0.5 0.3 0.2) '(0.5 0.3 0.2))]
      [p '(0.5 0.3 0.2)]
      [q '(0.4 0.35 0.25)])
  (let ([s1 (symmetric-chi-squared p q)]
        [s2 (symmetric-chi-squared q p)]
        [avg (/ (+ (chi-squared-divergence p q)
                   (chi-squared-divergence q p))
                2)])
    (and (approx=? 0.0 same 0.000001)
         (approx=? s1 s2 0.000001)
         (approx=? s1 avg 0.000001)
         (>= s1 0))))""",
    "jeffreys-divergence": """(let ([same (jeffreys-divergence '(0.5 0.3 0.2) '(0.5 0.3 0.2))]
      [p '(0.5 0.3 0.2)]
      [q '(0.4 0.35 0.25)])
  (let ([j1 (jeffreys-divergence p q)]
        [j2 (jeffreys-divergence q p)]
        [sum-kl (+ (kl-divergence p q) (kl-divergence q p))])
    (and (approx=? 0.0 same 0.000001)
         (approx=? j1 j2 0.000001)
         (approx=? j1 sum-kl 0.000001)
         (>= j1 0))))""",
    "alpha-divergence": """(let ([p '(0.5 0.3 0.2)]
      [q '(0.4 0.35 0.25)]
      [same '(0.333333 0.333333 0.333334)])
  (let ([alpha1 (alpha-divergence 1 p q)]
        [kl (kl-divergence p q)]
        [alpha-close (alpha-divergence 0.9999 p q)]
        [alpha-half (alpha-divergence 0.5 p q)]
        [alpha-same (alpha-divergence 0.5 same same)])
    (and (approx=? alpha1 kl 0.000001)
         (approx=? alpha-close kl 0.1)
         (>= alpha-half 0)
         (approx=? 0.0 alpha-same 0.000001))))""",
}

DIFFICULTY = {
    "bhattacharyya-coefficient": "easy",
    "bhattacharyya-distance": "medium",
    "hellinger-distance": "medium",
    "total-variation-distance": "easy",
    "chi-squared-divergence": "hard",
    "symmetric-chi-squared": "medium",
    "jeffreys-divergence": "medium",
    "alpha-divergence": "hard",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")

TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "bhattacharyya-coefficient": """def bhattacharyya_coefficient(p, q):
    if len(p) == 0 or len(q) == 0:
        return 0
    return sum(0 if (pi < 0 or qi < 0) else (pi * qi) ** 0.5 for pi, qi in zip(p, q))""",
    "bhattacharyya-distance": """def bhattacharyya_distance(p, q):
    bc = bhattacharyya_coefficient(p, q)
    if bc <= 0:
        return float('inf')
    return -log(bc)""",
    "hellinger-distance": """def hellinger_distance(p, q):
    if len(p) == 0 or len(q) == 0:
        return 1
    sq = sum((max(0.0, pi) ** 0.5 - max(0.0, qi) ** 0.5) ** 2 for pi, qi in zip(p, q))
    return (0.5 * sq) ** 0.5""",
    "total-variation-distance": """def total_variation_distance(p, q):
    if len(p) == 0 or len(q) == 0:
        return 1
    return 0.5 * sum(abs(pi - qi) for pi, qi in zip(p, q))""",
    "chi-squared-divergence": """def chi_squared_divergence(p, q):
    if len(p) == 0 or len(q) == 0:
        return 0
    total = 0.0
    for pi, qi in zip(p, q):
        if qi <= 0:
            if pi > 0:
                return float('inf')
            continue
        diff = pi - qi
        total += (diff * diff) / qi
    return total""",
    "symmetric-chi-squared": """def symmetric_chi_squared(p, q):
    pq = chi_squared_divergence(p, q)
    qp = chi_squared_divergence(q, p)
    if pq == float('inf') or qp == float('inf'):
        return float('inf')
    return (pq + qp) / 2""",
    "jeffreys-divergence": """def jeffreys_divergence(p, q):
    return kl_divergence(p, q) + kl_divergence(q, p)""",
    "alpha-divergence": """def alpha_divergence(alpha, p, q):
    if len(p) == 0 or len(q) == 0:
        return 0
    if alpha == 1:
        return kl_divergence(p, q)
    if alpha == 0:
        overlap = sum(qi for pi, qi in zip(p, q) if pi > 0 and qi > 0)
        return -log(overlap)
    total = sum(
        0 if (pi <= 0 or qi <= 0) else (pi ** alpha) * (qi ** (1 - alpha))
        for pi, qi in zip(p, q)
    )
    if total <= 0:
        return float('inf')
    return log(total) / (alpha * (alpha - 1))""",
}

CHEZ_SNIPPETS = {
    "bhattacharyya-coefficient": """(define (bc p q)
  (if (or (null? p) (null? q))
      0
      (fold-left + 0
                 (map (lambda (pi qi)
                        (if (or (< pi 0) (< qi 0))
                            0
                            (sqrt (* pi qi))))
                      p q))))""",
    "bhattacharyya-distance": """(define (bd p q)
  (let ([bc (bhattacharyya-coefficient p q)])
    (if (<= bc 0)
        +inf.0
        (- (log-num bc)))))""",
    "hellinger-distance": """(define (hellinger p q)
  (if (or (null? p) (null? q))
      1
      (let ([sum-sq (fold-left + 0
                               (map (lambda (pi qi)
                                      (let ([diff (- (sqrt (max 0 pi))
                                                     (sqrt (max 0 qi)))])
                                        (* diff diff)))
                                    p q))])
        (sqrt (* 0.5 sum-sq)))))""",
    "total-variation-distance": """(define (tv-distance p q)
  (if (or (null? p) (null? q))
      1
      (* 0.5 (fold-left + 0
                        (map (lambda (pi qi)
                               (abs (- pi qi)))
                             p q)))))""",
    "chi-squared-divergence": """(define (chi2-div p q)
  (if (or (null? p) (null? q))
      0
      (fold-left + 0
                 (map (lambda (pi qi)
                        (cond
                          [(<= qi 0)
                           (if (> pi 0) +inf.0 0)]
                          [else
                           (let ([diff (- pi qi)])
                             (/ (* diff diff) qi))]))
                      p q))))""",
    "symmetric-chi-squared": """(define (sym-chi2 p q)
  (let ([pq (chi-squared-divergence p q)]
        [qp (chi-squared-divergence q p)])
    (if (or (= pq +inf.0) (= qp +inf.0))
        +inf.0
        (/ (+ pq qp) 2))))""",
    "jeffreys-divergence": """(define (jeffreys p q)
  (let ([fwd (kl-divergence p q)]
        [rev (kl-divergence q p)])
    (+ fwd rev)))""",
    "alpha-divergence": """(define (alpha-div a p q)
  (cond
    [(or (null? p) (null? q)) 0]
    [(= a 1)
     (kl-divergence p q)]
    [(= a 0)
     (- (log-num (fold-left + 0
                            (map (lambda (pi qi)
                                   (if (and (> pi 0) (> qi 0)) qi 0))
                                 p q))))]
    [else
     (let ([s (fold-left + 0
                         (map (lambda (pi qi)
                                (if (or (<= pi 0) (<= qi 0))
                                    0
                                    (* (expt pi a) (expt qi (- 1 a)))))
                              p q))])
       (if (<= s 0)
           +inf.0
           (/ (log-num s)
              (* a (- a 1)))))]))""",
}

BUGGY_CASES = [
    {
        "fn": "bhattacharyya-coefficient",
        "buggy": """(define (bhattacharyya-coefficient p q)
  (if (or (null? p) (null? q))
      0
      (fold-left + 0
                 (map (lambda (pi qi)
                              (if (or (< pi 0) (< qi 0))
                                  0
                                  (* pi qi)))
                      p q))))""",
        "note": "Each term must be sqrt(p_i*q_i), not p_i*q_i.",
    },
    {
        "fn": "bhattacharyya-coefficient",
        "buggy": """(define (bhattacharyya-coefficient p q)
  (if (or (null? p) (null? q))
      1
      (fold-left + 0
                 (map (lambda (pi qi)
                              (if (or (< pi 0) (< qi 0))
                                  0
                                  (sqrt (* pi qi))))
                      p q))))""",
        "note": "Empty-list behavior must return 0, not 1.",
    },
    {
        "fn": "bhattacharyya-distance",
        "buggy": """(define (bhattacharyya-distance p q)
  (let ([bc (bhattacharyya-coefficient p q)])
       (if (<= bc 0)
           +inf.0
           (log-num bc))))""",
        "note": "Distance is -ln(BC), so the log term must be negated.",
    },
    {
        "fn": "bhattacharyya-distance",
        "buggy": """(define (bhattacharyya-distance p q)
  (let ([bc (bhattacharyya-coefficient p q)])
       (if (<= bc 0)
           0
           (- (log-num bc)))))""",
        "note": "When BC<=0, the function must return +inf.0.",
    },
    {
        "fn": "hellinger-distance",
        "buggy": """(define (hellinger-distance p q)
  (if (or (null? p) (null? q))
      1
      (let ([sum-sq (fold-left + 0
                               (map (lambda (pi qi)
                                            (let ([diff (- (sqrt (max 0 pi))
                                                           (sqrt (max 0 qi)))])
                                                 (* diff diff)))
                                    p q))])
           (sqrt sum-sq))))""",
        "note": "Hellinger needs the 0.5 factor inside the square root.",
    },
    {
        "fn": "hellinger-distance",
        "buggy": """(define (hellinger-distance p q)
  (if (or (null? p) (null? q))
      0
      (let ([sum-sq (fold-left + 0
                               (map (lambda (pi qi)
                                            (let ([diff (- (sqrt (max 0 pi))
                                                           (sqrt (max 0 qi)))])
                                                 (* diff diff)))
                                    p q))])
           (sqrt (* 0.5 sum-sq)))))""",
        "note": "Empty-list behavior should be 1 for this API, not 0.",
    },
    {
        "fn": "total-variation-distance",
        "buggy": """(define (total-variation-distance p q)
  (if (or (null? p) (null? q))
      1
      (* 0.5 (fold-left + 0
                        (map (lambda (pi qi)
                                     (- pi qi))
                             p q)))))""",
        "note": "TV requires absolute differences before summation.",
    },
    {
        "fn": "total-variation-distance",
        "buggy": """(define (total-variation-distance p q)
  (if (or (null? p) (null? q))
      1
      (fold-left + 0
                 (map (lambda (pi qi)
                              (abs (- pi qi)))
                      p q))))""",
        "note": "The total variation formula includes a leading factor of 0.5.",
    },
    {
        "fn": "chi-squared-divergence",
        "buggy": """(define (chi-squared-divergence p q)
  (if (or (null? p) (null? q))
      0
      (fold-left + 0
                 (map (lambda (pi qi)
                              (cond
                               [(<= qi 0)
                                (if (> pi 0) +inf.0 0)]
                               [else
                                (let ([diff (- pi qi)])
                                     (/ (* diff diff) pi))]))
                      p q))))""",
        "note": "Denominator must be q_i, not p_i.",
    },
    {
        "fn": "chi-squared-divergence",
        "buggy": """(define (chi-squared-divergence p q)
  (if (or (null? p) (null? q))
      0
      (fold-left + 0
                 (map (lambda (pi qi)
                              (cond
                               [(<= qi 0) 0]
                               [else
                                (let ([diff (- pi qi)])
                                     (/ (* diff diff) qi))]))
                      p q))))""",
        "note": "If q_i<=0 while p_i>0, the divergence must be +inf.0.",
    },
    {
        "fn": "symmetric-chi-squared",
        "buggy": """(define (symmetric-chi-squared p q)
  (let ([pq (chi-squared-divergence p q)]
        [qp (chi-squared-divergence q p)])
       (if (or (= pq +inf.0) (= qp +inf.0))
           +inf.0
           (+ pq qp))))""",
        "note": "Symmetric chi-squared is the average, so divide by 2.",
    },
    {
        "fn": "symmetric-chi-squared",
        "buggy": """(define (symmetric-chi-squared p q)
  (chi-squared-divergence p q))""",
        "note": "This loses symmetry by dropping the reverse-direction term.",
    },
    {
        "fn": "jeffreys-divergence",
        "buggy": """(define (jeffreys-divergence p q)
  (let ([kl-pq (kl-divergence p q)]
        [kl-qp (kl-divergence q p)])
       (- kl-pq kl-qp)))""",
        "note": "Jeffreys divergence sums forward and reverse KL; it does not subtract.",
    },
    {
        "fn": "jeffreys-divergence",
        "buggy": """(define (jeffreys-divergence p q)
  (kl-divergence p q))""",
        "note": "The reverse KL term is missing.",
    },
    {
        "fn": "alpha-divergence",
        "buggy": """(define (alpha-divergence alpha p q)
  (cond
   [(or (null? p) (null? q)) 0]
   [(= alpha 1)
    (kl-divergence q p)]
   [(= alpha 0)
    (- (log-num (fold-left + 0
                           (map (lambda (pi qi)
                                        (if (and (> pi 0) (> qi 0))
                                            qi
                                            0))
                                p q))))]
   [else
    (let ([sum-term (fold-left + 0
                               (map (lambda (pi qi)
                                            (cond
                                             [(and (<= pi 0) (<= qi 0)) 0]
                                             [(or (<= pi 0) (<= qi 0)) 0]
                                             [else
                                              (* (expt pi alpha)
                                                 (expt qi (- 1 alpha)))]))
                                    p q))])
         (if (<= sum-term 0)
             +inf.0
             (/ (log-num sum-term)
                (* alpha (- alpha 1)))))]))""",
        "note": "At alpha=1 this branch must reduce to KL(P||Q), not KL(Q||P).",
    },
    {
        "fn": "alpha-divergence",
        "buggy": """(define (alpha-divergence alpha p q)
  (cond
   [(or (null? p) (null? q)) 0]
   [(= alpha 1)
    (kl-divergence p q)]
   [(= alpha 0)
    (- (log-num (fold-left + 0
                           (map (lambda (pi qi)
                                        (if (and (> pi 0) (> qi 0))
                                            qi
                                            0))
                                p q))))]
   [else
    (let ([sum-term (fold-left + 0
                               (map (lambda (pi qi)
                                            (cond
                                             [(and (<= pi 0) (<= qi 0)) 0]
                                             [(or (<= pi 0) (<= qi 0)) 0]
                                             [else
                                              (* (expt pi alpha)
                                                 (expt qi (- 1 alpha)))]))
                                    p q))])
         (if (<= sum-term 0)
             +inf.0
             (/ (log-num sum-term)
                (* alpha (- 1 alpha)))))]))""",
        "note": "The general-case denominator should be alpha*(alpha-1), not alpha*(1-alpha).",
    },
]

DEPENDS: Dict[str, List[str]] = {
    "approx=?": [],
    "kl-divergence": [],
    "hellinger-distance-from-bc": [],
    "squared-hellinger-distance": ["bhattacharyya-coefficient"],
    "triangular-discrimination": [],
    "squared-loss": [],
    "euclidean-distance": ["squared-loss"],
    "matusita-distance": [],
    "valid-distribution?": [],
    "normalize-distribution": [],
    "bhattacharyya-coefficient": [],
    "bhattacharyya-distance": ["bhattacharyya-coefficient"],
    "hellinger-distance": [],
    "total-variation-distance": [],
    "chi-squared-divergence": [],
    "symmetric-chi-squared": ["chi-squared-divergence"],
    "jeffreys-divergence": ["kl-divergence"],
    "alpha-divergence": ["kl-divergence"],
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
    sid = f"info_statistical_measures_{family}_{family_counter[family]:03d}"
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


# Dependency closure for verify_expr includes explicit roots and symbol references in checks.
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
        prompt=f"""Implement this statistical measure in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "info", "statistical-measures", "spec-to-code", fn],
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
        tags=["tier1", "info", "statistical-measures", "skeleton-completion", fn],
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
        tags=["tier1", "info", "statistical-measures", "python-to-scheme", fn],
    )

    add_sample(
        family="translation",
        category="translation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""The following Chez-style helper captures the desired behavior.
Rewrite it as Fold-native Scheme and name the function `{fn}`.
Return only the function definition.

```scheme
{CHEZ_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "info", "statistical-measures", "chez-to-fold", fn],
    )


# -----------------------------------------------------------------------------
# Family 3: bugfix (16)
# -----------------------------------------------------------------------------
for case in BUGGY_CASES:
    fn = case["fn"]
    buggy = case["buggy"]
    note = case["note"]

    add_sample(
        family="bugfix",
        category="debugging",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Repair this Fold function with a minimal, focused patch.

Function: `{fn}`
Known issue: {note}

Buggy implementation:
```scheme
{buggy}
```

Return only the corrected definition for `{fn}`.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "info", "statistical-measures", "bugfix", fn],
    )


# -----------------------------------------------------------------------------
# Family 4: composition (32)
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
        prompt=f"""Write one Fold expression (no helper definitions) for this task.

Primary function: `{source_function}`
Task: {prompt}

Return only the expression.""",
        ground_truth=ground_truth,
        verify_expr=build_verify(verify_check, [source_function]),
        tags=["tier1", "info", "statistical-measures", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # bhattacharyya-coefficient
    {
        "fn": "bhattacharyya-coefficient",
        "prompt": "Normalize two non-negative weight vectors and compute their Bhattacharyya coefficient.",
        "gt": "(let* ([p (normalize-distribution '(2 3 5))] [q (normalize-distribution '(3 3 4))]) (bhattacharyya-coefficient p q))",
        "verify": "(let* ([p (normalize-distribution '(2 3 5))] [q (normalize-distribution '(3 3 4))] [bc (bhattacharyya-coefficient p q)]) (and (>= bc 0) (<= bc 1) (approx=? bc (bhattacharyya-coefficient q p) 0.000001)))",
        "difficulty": "medium",
        "tags": ["normalization", "range"],
    },
    {
        "fn": "bhattacharyya-coefficient",
        "prompt": "Use BC to reconstruct squared Hellinger distance through H = sqrt(1 - BC). Return a Boolean assertion.",
        "gt": "(let* ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)] [bc (bhattacharyya-coefficient p q)] [h (hellinger-distance-from-bc bc)]) (approx=? (* h h) (- 1 bc) 0.000001))",
        "verify": "(equal? (let* ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)] [bc (bhattacharyya-coefficient p q)] [h (hellinger-distance-from-bc bc)]) (approx=? (* h h) (- 1 bc) 0.000001)) #t)",
        "difficulty": "hard",
        "tags": ["cross-measure", "hellinger"],
    },
    {
        "fn": "bhattacharyya-coefficient",
        "prompt": "For disjoint support distributions, return whether BC is zero and Bhattacharyya distance is infinite.",
        "gt": "(let* ([p '(1 0 0 0)] [q '(0 0 1 0)] [bc (bhattacharyya-coefficient p q)] [bd (bhattacharyya-distance p q)]) (and (approx=? bc 0.0 0.000001) (= bd +inf.0)))",
        "verify": "(equal? (let* ([p '(1 0 0 0)] [q '(0 0 1 0)] [bc (bhattacharyya-coefficient p q)] [bd (bhattacharyya-distance p q)]) (and (approx=? bc 0.0 0.000001) (= bd +inf.0))) #t)",
        "difficulty": "easy",
        "tags": ["edge-case", "distance-link"],
    },
    {
        "fn": "bhattacharyya-coefficient",
        "prompt": "Compute the symmetry residual |BC(P,Q) - BC(Q,P)| for two ternary distributions.",
        "gt": "(let ([p '(0.6 0.25 0.15)] [q '(0.45 0.35 0.20)]) (abs (- (bhattacharyya-coefficient p q) (bhattacharyya-coefficient q p))))",
        "verify": "(approx=? 0.0 (let ([p '(0.6 0.25 0.15)] [q '(0.45 0.35 0.20)]) (abs (- (bhattacharyya-coefficient p q) (bhattacharyya-coefficient q p)))) 0.000001)",
        "difficulty": "medium",
        "tags": ["symmetry", "residual"],
    },

    # bhattacharyya-distance
    {
        "fn": "bhattacharyya-distance",
        "prompt": "Normalize two weight vectors and compute Bhattacharyya distance.",
        "gt": "(let* ([p (normalize-distribution '(1 4 5))] [q (normalize-distribution '(2 3 5))]) (bhattacharyya-distance p q))",
        "verify": "(let* ([p (normalize-distribution '(1 4 5))] [q (normalize-distribution '(2 3 5))] [bc (bhattacharyya-coefficient p q)] [d (bhattacharyya-distance p q)]) (and (approx=? d (- (log-num bc)) 0.000001) (>= d 0)))",
        "difficulty": "medium",
        "tags": ["normalization", "formula"],
    },
    {
        "fn": "bhattacharyya-distance",
        "prompt": "Return whether a highly mismatched pair has larger Bhattacharyya distance than a nearby pair.",
        "gt": "(let* ([near-p '(0.5 0.3 0.2)] [near-q '(0.45 0.3 0.25)] [far-p '(0.9 0.05 0.05)] [far-q '(0.333333 0.333333 0.333334)] [d-near (bhattacharyya-distance near-p near-q)] [d-far (bhattacharyya-distance far-p far-q)]) (> d-far d-near))",
        "verify": "(equal? (let* ([near-p '(0.5 0.3 0.2)] [near-q '(0.45 0.3 0.25)] [far-p '(0.9 0.05 0.05)] [far-q '(0.333333 0.333333 0.333334)] [d-near (bhattacharyya-distance near-p near-q)] [d-far (bhattacharyya-distance far-p far-q)]) (> d-far d-near)) #t)",
        "difficulty": "hard",
        "tags": ["ordering", "contrast"],
    },
    {
        "fn": "bhattacharyya-distance",
        "prompt": "Return the residual between D_B(P,Q) and -ln(BC(P,Q)) for one pair.",
        "gt": "(let ([p '(0.7 0.2 0.1)] [q '(0.5 0.3 0.2)]) (- (bhattacharyya-distance p q) (- (log-num (bhattacharyya-coefficient p q)))))",
        "verify": "(approx=? 0.0 (let ([p '(0.7 0.2 0.1)] [q '(0.5 0.3 0.2)]) (- (bhattacharyya-distance p q) (- (log-num (bhattacharyya-coefficient p q))))) 0.000001)",
        "difficulty": "medium",
        "tags": ["consistency", "residual"],
    },
    {
        "fn": "bhattacharyya-distance",
        "prompt": "For identical distributions, assert that distance is zero and coefficient is one.",
        "gt": "(let ([p '(0.4 0.35 0.25)]) (and (approx=? 0.0 (bhattacharyya-distance p p) 0.000001) (approx=? 1.0 (bhattacharyya-coefficient p p) 0.000001)))",
        "verify": "(equal? (let ([p '(0.4 0.35 0.25)]) (and (approx=? 0.0 (bhattacharyya-distance p p) 0.000001) (approx=? 1.0 (bhattacharyya-coefficient p p) 0.000001))) #t)",
        "difficulty": "easy",
        "tags": ["identity", "cross-check"],
    },

    # hellinger-distance
    {
        "fn": "hellinger-distance",
        "prompt": "Normalize two vectors and compute Hellinger distance.",
        "gt": "(let* ([p (normalize-distribution '(2 1 1))] [q (normalize-distribution '(1 2 1))]) (hellinger-distance p q))",
        "verify": "(let* ([p (normalize-distribution '(2 1 1))] [q (normalize-distribution '(1 2 1))] [h (hellinger-distance p q)]) (and (>= h 0) (<= h 1) (approx=? h (hellinger-distance q p) 0.000001)))",
        "difficulty": "medium",
        "tags": ["normalization", "range"],
    },
    {
        "fn": "hellinger-distance",
        "prompt": "Return whether direct Hellinger distance matches the BC-derived form sqrt(1 - BC).",
        "gt": "(let* ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)] [h (hellinger-distance p q)] [h2 (hellinger-distance-from-bc (bhattacharyya-coefficient p q))]) (approx=? h h2 0.000001))",
        "verify": "(equal? (let* ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)] [h (hellinger-distance p q)] [h2 (hellinger-distance-from-bc (bhattacharyya-coefficient p q))]) (approx=? h h2 0.000001)) #t)",
        "difficulty": "hard",
        "tags": ["relation", "bhattacharyya"],
    },
    {
        "fn": "hellinger-distance",
        "prompt": "Check the triangle inequality H(P,R) <= H(P,Q) + H(Q,R) for three distributions.",
        "gt": "(let ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)] [r '(0.9 0.05 0.05)]) (<= (hellinger-distance p r) (+ (hellinger-distance p q) (hellinger-distance q r) 0.001)))",
        "verify": "(equal? (let ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)] [r '(0.9 0.05 0.05)]) (<= (hellinger-distance p r) (+ (hellinger-distance p q) (hellinger-distance q r) 0.001))) #t)",
        "difficulty": "hard",
        "tags": ["triangle-inequality", "property"],
    },
    {
        "fn": "hellinger-distance",
        "prompt": "Compute matusita-distance/sqrt(2) for one pair (it should match Hellinger distance).",
        "gt": "(let ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)]) (- (hellinger-distance p q) (/ (matusita-distance p q) (sqrt 2))))",
        "verify": "(approx=? 0.0 (let ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)]) (- (hellinger-distance p q) (/ (matusita-distance p q) (sqrt 2)))) 0.000001)",
        "difficulty": "medium",
        "tags": ["matusita", "conversion"],
    },

    # total-variation-distance
    {
        "fn": "total-variation-distance",
        "prompt": "Normalize two weight vectors and compute total variation distance.",
        "gt": "(let* ([p (normalize-distribution '(4 3 3))] [q (normalize-distribution '(3 4 3))]) (total-variation-distance p q))",
        "verify": "(let* ([p (normalize-distribution '(4 3 3))] [q (normalize-distribution '(3 4 3))] [tv (total-variation-distance p q)]) (and (>= tv 0) (<= tv 1) (approx=? tv (total-variation-distance q p) 0.000001)))",
        "difficulty": "easy",
        "tags": ["normalization", "range"],
    },
    {
        "fn": "total-variation-distance",
        "prompt": "Return the residual between TV(P,Q) and its manual 0.5*L1 formula for a binary pair.",
        "gt": "(let* ([p '(0.5 0.5)] [q '(0.8 0.2)] [tv (total-variation-distance p q)] [manual (* 0.5 (+ (abs (- 0.5 0.8)) (abs (- 0.5 0.2))))]) (- tv manual))",
        "verify": "(approx=? 0.0 (let* ([p '(0.5 0.5)] [q '(0.8 0.2)] [tv (total-variation-distance p q)] [manual (* 0.5 (+ (abs (- 0.5 0.8)) (abs (- 0.5 0.2))))]) (- tv manual)) 0.000001)",
        "difficulty": "medium",
        "tags": ["formula", "residual"],
    },
    {
        "fn": "total-variation-distance",
        "prompt": "Compute |TV(P,Q) - TV(Q,P)| for two ternary distributions.",
        "gt": "(let ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)]) (abs (- (total-variation-distance p q) (total-variation-distance q p))))",
        "verify": "(approx=? 0.0 (let ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)]) (abs (- (total-variation-distance p q) (total-variation-distance q p)))) 0.000001)",
        "difficulty": "easy",
        "tags": ["symmetry", "residual"],
    },
    {
        "fn": "total-variation-distance",
        "prompt": "Check the TV triangle inequality TV(P,R) <= TV(P,Q) + TV(Q,R) for three distributions.",
        "gt": "(let ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)] [r '(0.9 0.05 0.05)]) (<= (total-variation-distance p r) (+ (total-variation-distance p q) (total-variation-distance q r) 0.001)))",
        "verify": "(equal? (let ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)] [r '(0.9 0.05 0.05)]) (<= (total-variation-distance p r) (+ (total-variation-distance p q) (total-variation-distance q r) 0.001))) #t)",
        "difficulty": "hard",
        "tags": ["triangle-inequality", "property"],
    },

    # chi-squared-divergence
    {
        "fn": "chi-squared-divergence",
        "prompt": "Normalize two vectors (strictly positive target) and compute forward chi-squared divergence.",
        "gt": "(let* ([p (normalize-distribution '(5 3 2))] [q (normalize-distribution '(4 4 2))]) (chi-squared-divergence p q))",
        "verify": "(let* ([p (normalize-distribution '(5 3 2))] [q (normalize-distribution '(4 4 2))] [v (chi-squared-divergence p q)]) (and (>= v 0) (not (= v +inf.0))))",
        "difficulty": "medium",
        "tags": ["normalization", "finite"],
    },
    {
        "fn": "chi-squared-divergence",
        "prompt": "For one support-mismatch pair, assert forward chi-squared is infinite while reverse remains finite/non-negative.",
        "gt": "(let ([p '(0.5 0.5)] [q '(1.0 0.0)]) (and (= (chi-squared-divergence p q) +inf.0) (>= (chi-squared-divergence q p) 0)))",
        "verify": "(equal? (let ([p '(0.5 0.5)] [q '(1.0 0.0)]) (and (= (chi-squared-divergence p q) +inf.0) (>= (chi-squared-divergence q p) 0))) #t)",
        "difficulty": "hard",
        "tags": ["support-mismatch", "asymmetry"],
    },
    {
        "fn": "chi-squared-divergence",
        "prompt": "Return whether chi-squared is asymmetric on two close ternary distributions.",
        "gt": "(let ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)]) (not (approx=? (chi-squared-divergence p q) (chi-squared-divergence q p) 0.000001)))",
        "verify": "(equal? (let ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)]) (not (approx=? (chi-squared-divergence p q) (chi-squared-divergence q p) 0.000001))) #t)",
        "difficulty": "medium",
        "tags": ["asymmetry", "property"],
    },
    {
        "fn": "chi-squared-divergence",
        "prompt": "Return residual between manual average of forward/reverse chi-squared and symmetric-chi-squared.",
        "gt": "(let ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)]) (- (/ (+ (chi-squared-divergence p q) (chi-squared-divergence q p)) 2) (symmetric-chi-squared p q)))",
        "verify": "(approx=? 0.0 (let ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)]) (- (/ (+ (chi-squared-divergence p q) (chi-squared-divergence q p)) 2) (symmetric-chi-squared p q))) 0.000001)",
        "difficulty": "hard",
        "tags": ["consistency", "symmetrization"],
    },

    # symmetric-chi-squared
    {
        "fn": "symmetric-chi-squared",
        "prompt": "Normalize two vectors and compute symmetric chi-squared divergence.",
        "gt": "(let* ([p (normalize-distribution '(5 3 2))] [q (normalize-distribution '(4 4 2))]) (symmetric-chi-squared p q))",
        "verify": "(let* ([p (normalize-distribution '(5 3 2))] [q (normalize-distribution '(4 4 2))] [v (symmetric-chi-squared p q)]) (and (>= v 0) (approx=? v (symmetric-chi-squared q p) 0.000001)))",
        "difficulty": "medium",
        "tags": ["normalization", "symmetry"],
    },
    {
        "fn": "symmetric-chi-squared",
        "prompt": "Return whether symmetric chi-squared equals the arithmetic mean of forward and reverse chi-squared.",
        "gt": "(let ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)]) (approx=? (symmetric-chi-squared p q) (/ (+ (chi-squared-divergence p q) (chi-squared-divergence q p)) 2) 0.000001))",
        "verify": "(equal? (let ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)]) (approx=? (symmetric-chi-squared p q) (/ (+ (chi-squared-divergence p q) (chi-squared-divergence q p)) 2) 0.000001)) #t)",
        "difficulty": "hard",
        "tags": ["mean-identity", "formula"],
    },
    {
        "fn": "symmetric-chi-squared",
        "prompt": "Compute symmetric chi-squared on disjoint support distributions.",
        "gt": "(symmetric-chi-squared '(0.5 0.5 0 0) '(0 0 0.5 0.5))",
        "verify": "(equal? (symmetric-chi-squared '(0.5 0.5 0 0) '(0 0 0.5 0.5)) +inf.0)",
        "difficulty": "hard",
        "tags": ["edge-case", "infinite"],
    },
    {
        "fn": "symmetric-chi-squared",
        "prompt": "For identical distributions, assert both symmetric chi-squared and Jeffreys divergence are zero.",
        "gt": "(let ([p '(0.5 0.3 0.2)]) (and (approx=? 0.0 (symmetric-chi-squared p p) 0.000001) (approx=? 0.0 (jeffreys-divergence p p) 0.000001)))",
        "verify": "(equal? (let ([p '(0.5 0.3 0.2)]) (and (approx=? 0.0 (symmetric-chi-squared p p) 0.000001) (approx=? 0.0 (jeffreys-divergence p p) 0.000001))) #t)",
        "difficulty": "easy",
        "tags": ["identity", "cross-measure"],
    },

    # jeffreys-divergence
    {
        "fn": "jeffreys-divergence",
        "prompt": "Normalize two vectors and compute Jeffreys divergence.",
        "gt": "(let* ([p (normalize-distribution '(6 3 1))] [q (normalize-distribution '(4 4 2))]) (jeffreys-divergence p q))",
        "verify": "(let* ([p (normalize-distribution '(6 3 1))] [q (normalize-distribution '(4 4 2))] [j (jeffreys-divergence p q)]) (approx=? j (+ (kl-divergence p q) (kl-divergence q p)) 0.000001))",
        "difficulty": "medium",
        "tags": ["normalization", "kl-relation"],
    },
    {
        "fn": "jeffreys-divergence",
        "prompt": "Compute symmetry residual |J(P,Q) - J(Q,P)| for two distributions.",
        "gt": "(let ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)]) (abs (- (jeffreys-divergence p q) (jeffreys-divergence q p))))",
        "verify": "(approx=? 0.0 (let ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)]) (abs (- (jeffreys-divergence p q) (jeffreys-divergence q p)))) 0.000001)",
        "difficulty": "easy",
        "tags": ["symmetry", "residual"],
    },
    {
        "fn": "jeffreys-divergence",
        "prompt": "Return whether Jeffreys divergence is at least as large as forward KL on one pair.",
        "gt": "(let ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)]) (>= (jeffreys-divergence p q) (kl-divergence p q)))",
        "verify": "(equal? (let ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)]) (>= (jeffreys-divergence p q) (kl-divergence p q))) #t)",
        "difficulty": "medium",
        "tags": ["bound", "kl"],
    },
    {
        "fn": "jeffreys-divergence",
        "prompt": "Return the residual between Jeffreys divergence and alpha-divergence at alpha=1 in both directions.",
        "gt": "(let ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)]) (- (jeffreys-divergence p q) (+ (alpha-divergence 1 p q) (alpha-divergence 1 q p))))",
        "verify": "(approx=? 0.0 (let ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)]) (- (jeffreys-divergence p q) (+ (alpha-divergence 1 p q) (alpha-divergence 1 q p)))) 0.000001)",
        "difficulty": "hard",
        "tags": ["alpha-link", "consistency"],
    },

    # alpha-divergence
    {
        "fn": "alpha-divergence",
        "prompt": "Normalize two vectors and compute alpha-divergence for alpha=0.5.",
        "gt": "(let* ([p (normalize-distribution '(6 3 1))] [q (normalize-distribution '(4 4 2))]) (alpha-divergence 0.5 p q))",
        "verify": "(let* ([p (normalize-distribution '(6 3 1))] [q (normalize-distribution '(4 4 2))] [a (alpha-divergence 0.5 p q)]) (and (>= a 0) (not (= a +inf.0))))",
        "difficulty": "medium",
        "tags": ["normalization", "finite"],
    },
    {
        "fn": "alpha-divergence",
        "prompt": "Return residual between alpha-divergence at alpha=1 and KL divergence.",
        "gt": "(let ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)]) (- (alpha-divergence 1 p q) (kl-divergence p q)))",
        "verify": "(approx=? 0.0 (let ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)]) (- (alpha-divergence 1 p q) (kl-divergence p q))) 0.000001)",
        "difficulty": "hard",
        "tags": ["limit-case", "kl"],
    },
    {
        "fn": "alpha-divergence",
        "prompt": "Check whether alpha-divergence at alpha=0.9999 is close to KL divergence for one pair.",
        "gt": "(let ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)]) (approx=? (alpha-divergence 0.9999 p q) (kl-divergence p q) 0.1))",
        "verify": "(equal? (let ([p '(0.5 0.3 0.2)] [q '(0.4 0.35 0.25)]) (approx=? (alpha-divergence 0.9999 p q) (kl-divergence p q) 0.1)) #t)",
        "difficulty": "hard",
        "tags": ["continuity", "limit"],
    },
    {
        "fn": "alpha-divergence",
        "prompt": "For an identical distribution pair, assert alpha-divergence is zero for alpha=0.5 and alpha=2.",
        "gt": "(let ([p '(0.333333 0.333333 0.333334)]) (and (approx=? 0.0 (alpha-divergence 0.5 p p) 0.000001) (approx=? 0.0 (alpha-divergence 2 p p) 0.000001)))",
        "verify": "(equal? (let ([p '(0.333333 0.333333 0.333334)]) (and (approx=? 0.0 (alpha-divergence 0.5 p p) 0.000001) (approx=? 0.0 (alpha-divergence 2 p p) 0.000001))) #t)",
        "difficulty": "medium",
        "tags": ["identity", "multi-alpha"],
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
