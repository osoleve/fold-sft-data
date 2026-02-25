#!/usr/bin/env python3
"""Generate Tier-0 SFT samples for lattice/number-theory/fast-multiply.ss."""

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

SOURCE_MODULE = "lattice/number-theory/fast-multiply.ss"
SOURCE_TEST = "lattice/number-theory/test-fast-multiply.ss"

DEFS: Dict[str, str] = {
    "limbs->integer": """(define (limbs->integer limbs base)
  (let loop ([ls limbs] [multiplier 1] [result 0])
    (if (null? ls)
        result
        (loop (cdr ls)
              (* multiplier base)
              (+ result (* (car ls) multiplier))))))""",
    "integer->limbs": """(define (integer->limbs n base)
  (if (= n 0)
      '(0)
      (let loop ([n n] [limbs '()])
        (if (= n 0)
            (reverse limbs)
            (loop (quotient n base)
                  (cons (modulo n base) limbs))))))""",
    "limbs-normalize": """(define (limbs-normalize limbs)
  (let loop ([ls (reverse limbs)])
    (cond
      [(null? ls) '(0)]
      [(and (= (car ls) 0) (not (null? (cdr ls))))
       (loop (cdr ls))]
      [else (reverse ls)])))""",
    "limbs-add": """(define (limbs-add a b base)
  (let loop ([a a] [b b] [carry 0] [result '()])
    (cond
      [(and (null? a) (null? b))
       (limbs-normalize
        (reverse (if (> carry 0) (cons carry result) result)))]
      [(null? a)
       (loop '() (cdr b)
             (quotient (+ (car b) carry) base)
             (cons (modulo (+ (car b) carry) base) result))]
      [(null? b)
       (loop (cdr a) '()
             (quotient (+ (car a) carry) base)
             (cons (modulo (+ (car a) carry) base) result))]
      [else
       (let ([sum (+ (car a) (car b) carry)])
         (loop (cdr a) (cdr b)
               (quotient sum base)
               (cons (modulo sum base) result)))])))""",
    "limbs-sub": """(define (limbs-sub a b base)
  (let loop ([a a] [b b] [borrow 0] [result '()])
    (cond
      [(and (null? a) (null? b))
       (limbs-normalize (reverse result))]
      [(null? b)
       (let ([diff (- (car a) borrow)])
         (if (< diff 0)
             (loop (cdr a) '() 1 (cons (+ diff base) result))
             (loop (cdr a) '() 0 (cons diff result))))]
      [else
       (let* ([av (if (null? a) 0 (car a))]
              [bv (car b)]
              [diff (- av bv borrow)])
         (if (< diff 0)
             (loop (if (null? a) '() (cdr a))
                   (cdr b)
                   1
                   (cons (+ diff base) result))
             (loop (if (null? a) '() (cdr a))
                   (cdr b)
                   0
                   (cons diff result))))])))""",
    "limbs-shift": """(define (limbs-shift limbs k)
  (if (and (= 1 (length limbs)) (= 0 (car limbs)))
      limbs
      (append (make-list k 0) limbs)))""",
    "limb-scale": """(define (limb-scale limbs scalar base)
  (let loop ([ls limbs] [carry 0] [result '()])
    (if (null? ls)
        (limbs-normalize
         (reverse (if (> carry 0) (cons carry result) result)))
        (let ([prod (+ (* (car ls) scalar) carry)])
          (loop (cdr ls)
                (quotient prod base)
                (cons (modulo prod base) result))))))""",
    "limbs-multiply-schoolbook": """(define (limbs-multiply-schoolbook a b base)
  (let loop ([b b] [shift 0] [result '(0)])
    (if (null? b)
        (limbs-normalize result)
        (let ([partial (limb-scale a (car b) base)])
          (loop (cdr b)
                (+ shift 1)
                (limbs-add result (limbs-shift partial shift) base))))))""",
}

FUNCTION_ORDER = [
    "limbs->integer",
    "integer->limbs",
    "limbs-normalize",
    "limbs-add",
    "limbs-sub",
    "limbs-shift",
    "limb-scale",
    "limbs-multiply-schoolbook",
]

FUNCTION_SPECS = {
    "limbs->integer": "Convert least-significant-first limb list into integer under given base.",
    "integer->limbs": "Convert integer to least-significant-first limb list; return '(0) for zero.",
    "limbs-normalize": "Remove high-order trailing zeros while preserving canonical zero representation '(0).",
    "limbs-add": "Add two limb lists with carry propagation in the given base.",
    "limbs-sub": "Subtract b from a (a>=b) with borrow propagation and normalized output.",
    "limbs-shift": "Multiply by base^k by prepending k zero limbs; keep zero canonical.",
    "limb-scale": "Multiply a limb list by one scalar limb with carry propagation.",
    "limbs-multiply-schoolbook": "Compute O(n^2) schoolbook multiplication of two limb lists.",
}

SKELETONS = {
    "limbs->integer": """(define (limbs->integer limbs base)
  ;; TODO: fold least-significant-first limbs into integer
  <TODO>)""",
    "integer->limbs": """(define (integer->limbs n base)
  ;; TODO: emit least-significant-first limbs, canonical zero is '(0)
  <TODO>)""",
    "limbs-normalize": """(define (limbs-normalize limbs)
  ;; TODO: drop high-order trailing zeros while keeping '(0) for zero
  <TODO>)""",
    "limbs-add": """(define (limbs-add a b base)
  ;; TODO: implement limb-wise addition with carry
  <TODO>)""",
    "limbs-sub": """(define (limbs-sub a b base)
  ;; TODO: implement limb-wise subtraction with borrow (assume a>=b)
  <TODO>)""",
    "limbs-shift": """(define (limbs-shift limbs k)
  ;; TODO: multiply by base^k in limb representation
  <TODO>)""",
    "limb-scale": """(define (limb-scale limbs scalar base)
  ;; TODO: scale limb list by one scalar limb with carry
  <TODO>)""",
    "limbs-multiply-schoolbook": """(define (limbs-multiply-schoolbook a b base)
  ;; TODO: accumulate shifted partial products
  <TODO>)""",
}

DEPENDS: Dict[str, List[str]] = {
    "limbs->integer": [],
    "integer->limbs": [],
    "limbs-normalize": [],
    "limbs-add": ["limbs-normalize"],
    "limbs-sub": ["limbs-normalize"],
    "limbs-shift": [],
    "limb-scale": ["limbs-normalize"],
    "limbs-multiply-schoolbook": ["limb-scale", "limbs-add", "limbs-shift", "limbs-normalize"],
}

VERIFY_BY_FUNCTION = {
    "limbs->integer": """(and
  (= (limbs->integer '(0) 10) 0)
  (= (limbs->integer '(5) 10) 5)
  (= (limbs->integer '(3 2 1) 10) 123)
  (= (limbs->integer '(45 23 1) 100) 12345))""",
    "integer->limbs": """(and
  (equal? (integer->limbs 0 10) '(0))
  (equal? (integer->limbs 5 10) '(5))
  (equal? (integer->limbs 123 10) '(3 2 1))
  (equal? (integer->limbs 12345 100) '(45 23 1)))""",
    "limbs-normalize": """(and
  (equal? (limbs-normalize '(1 2 3 0 0)) '(1 2 3))
  (equal? (limbs-normalize '(0 0 0)) '(0))
  (equal? (limbs-normalize '(5 0)) '(5))
  (equal? (limbs-normalize '(7 8 9)) '(7 8 9)))""",
    "limbs-add": """(and
  (equal? (limbs-add '(2) '(3) 10) '(5))
  (equal? (limbs-add '(7) '(3) 10) '(0 1))
  (equal? (limbs-add '(9 9) '(1 1) 10) '(0 1 1))
  (equal? (limbs-add '(1 2 3) '(4 5 6) 10) '(5 7 9)))""",
    "limbs-sub": """(and
  (equal? (limbs-sub '(5) '(3) 10) '(2))
  (equal? (limbs-sub '(0 0 1) '(1) 10) '(9 9))
  (equal? (limbs-sub '(5 7 9) '(4 5 6) 10) '(1 2 3))
  (equal? (limbs-sub '(0 0 2) '(0 0 1) 10) '(0 0 1)))""",
    "limbs-shift": """(and
  (equal? (limbs-shift '(1 2 3) 2) '(0 0 1 2 3))
  (equal? (limbs-shift '(5) 0) '(5))
  (equal? (limbs-shift '(0) 3) '(0)))""",
    "limb-scale": """(and
  (equal? (limb-scale '(5) 2 10) '(0 1))
  (equal? (limb-scale '(2 7) 8 10) '(6 7 5))
  (equal? (limb-scale '(9 9) 9 10) '(1 9 8))
  (equal? (limb-scale '(1 2 3) 0 10) '(0)))""",
    "limbs-multiply-schoolbook": """(and
  (equal? (limbs-multiply-schoolbook '(2 1) '(4 3) 10) '(8 0 4))
  (equal? (limbs-multiply-schoolbook '(0) '(5 6 7) 10) '(0))
  (equal? (limbs->integer (limbs-multiply-schoolbook (integer->limbs 12345 1000)
                                                      (integer->limbs 67890 1000)
                                                      1000)
                         1000)
          (* 12345 67890)))""",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")

TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "limbs->integer": """def limbs_to_integer(limbs, base):
    multiplier = 1
    result = 0
    for limb in limbs:
        result += limb * multiplier
        multiplier *= base
    return result""",
    "integer->limbs": """def integer_to_limbs(n, base):
    if n == 0:
        return [0]
    out = []
    while n > 0:
        out.append(n % base)
        n //= base
    return out""",
    "limbs-normalize": """def limbs_normalize(limbs):
    i = len(limbs) - 1
    while i > 0 and limbs[i] == 0:
        i -= 1
    return limbs[: i + 1]""",
    "limbs-add": """def limbs_add(a, b, base):
    out = []
    carry = 0
    i = 0
    while i < len(a) or i < len(b) or carry:
        av = a[i] if i < len(a) else 0
        bv = b[i] if i < len(b) else 0
        s = av + bv + carry
        out.append(s % base)
        carry = s // base
        i += 1
    return limbs_normalize(out)""",
    "limbs-sub": """def limbs_sub(a, b, base):
    out = []
    borrow = 0
    i = 0
    while i < len(a) or i < len(b):
        av = a[i] if i < len(a) else 0
        bv = b[i] if i < len(b) else 0
        d = av - bv - borrow
        if d < 0:
            d += base
            borrow = 1
        else:
            borrow = 0
        out.append(d)
        i += 1
    return limbs_normalize(out)""",
    "limbs-shift": """def limbs_shift(limbs, k):
    if len(limbs) == 1 and limbs[0] == 0:
        return limbs
    return ([0] * k) + limbs""",
    "limb-scale": """def limb_scale(limbs, scalar, base):
    out = []
    carry = 0
    for limb in limbs:
        p = limb * scalar + carry
        out.append(p % base)
        carry = p // base
    if carry:
        out.append(carry)
    return limbs_normalize(out)""",
    "limbs-multiply-schoolbook": """def limbs_multiply_schoolbook(a, b, base):
    result = [0]
    shift = 0
    for limb in b:
        partial = limb_scale(a, limb, base)
        result = limbs_add(result, limbs_shift(partial, shift), base)
        shift += 1
    return limbs_normalize(result)""",
}

CHEZ_SNIPPETS = {
    "limbs->integer": """(define (to-int limbs base)
  (let loop ([ls limbs] [m 1] [acc 0])
    (if (null? ls)
        acc
        (loop (cdr ls) (* m base) (+ acc (* (car ls) m))))))""",
    "integer->limbs": """(define (to-limbs n base)
  (if (= n 0)
      '(0)
      (let loop ([n n] [acc '()])
        (if (= n 0)
            (reverse acc)
            (loop (quotient n base) (cons (modulo n base) acc))))))""",
    "limbs-normalize": """(define (normalize limbs)
  (let loop ([ls (reverse limbs)])
    (cond
      [(null? ls) '(0)]
      [(and (= (car ls) 0) (not (null? (cdr ls))))
       (loop (cdr ls))]
      [else (reverse ls)])))""",
    "limbs-add": """(define (add-limbs a b base)
  (let loop ([a a] [b b] [carry 0] [out '()])
    (cond
      [(and (null? a) (null? b))
       (limbs-normalize (reverse (if (> carry 0) (cons carry out) out)))]
      [else
       (let* ([av (if (null? a) 0 (car a))]
              [bv (if (null? b) 0 (car b))]
              [s (+ av bv carry)])
         (loop (if (null? a) '() (cdr a))
               (if (null? b) '() (cdr b))
               (quotient s base)
               (cons (modulo s base) out)))])))""",
    "limbs-sub": """(define (sub-limbs a b base)
  (let loop ([a a] [b b] [borrow 0] [out '()])
    (if (and (null? a) (null? b))
        (limbs-normalize (reverse out))
        (let* ([av (if (null? a) 0 (car a))]
               [bv (if (null? b) 0 (car b))]
               [d (- av bv borrow)])
          (if (< d 0)
              (loop (if (null? a) '() (cdr a))
                    (if (null? b) '() (cdr b))
                    1
                    (cons (+ d base) out))
              (loop (if (null? a) '() (cdr a))
                    (if (null? b) '() (cdr b))
                    0
                    (cons d out)))))))""",
    "limbs-shift": """(define (shift-limbs limbs k)
  (if (and (= 1 (length limbs)) (= 0 (car limbs)))
      limbs
      (append (make-list k 0) limbs)))""",
    "limb-scale": """(define (scale-limb limbs scalar base)
  (let loop ([ls limbs] [carry 0] [out '()])
    (if (null? ls)
        (limbs-normalize (reverse (if (> carry 0) (cons carry out) out)))
        (let ([p (+ (* (car ls) scalar) carry)])
          (loop (cdr ls)
                (quotient p base)
                (cons (modulo p base) out))))))""",
    "limbs-multiply-schoolbook": """(define (mul-schoolbook a b base)
  (let loop ([bs b] [shift 0] [res '(0)])
    (if (null? bs)
        (limbs-normalize res)
        (let ([partial (limb-scale a (car bs) base)])
          (loop (cdr bs)
                (+ shift 1)
                (limbs-add res (limbs-shift partial shift) base))))))""",
}

BUGGY_CASES = [
    {
        "fn": "limbs->integer",
        "buggy": """(define (limbs->integer limbs base)
  (let loop ([ls limbs] [multiplier base] [result 0])
    (if (null? ls)
        result
        (loop (cdr ls)
              (* multiplier base)
              (+ result (* (car ls) multiplier))))))""",
        "note": "Multiplier must start at 1 for the least-significant limb.",
    },
    {
        "fn": "limbs->integer",
        "buggy": """(define (limbs->integer limbs base)
  (let loop ([ls limbs] [result 0])
    (if (null? ls)
        result
        (loop (cdr ls)
              (+ (* result base) (car ls))))))""",
        "note": "This treats limbs as most-significant-first; representation is least-significant-first.",
    },
    {
        "fn": "integer->limbs",
        "buggy": """(define (integer->limbs n base)
  (if (= n 0)
      '()
      (let loop ([n n] [limbs '()])
        (if (= n 0)
            (reverse limbs)
            (loop (quotient n base)
                  (cons (modulo n base) limbs))))))""",
        "note": "Zero must map to canonical '(0), not empty list.",
    },
    {
        "fn": "integer->limbs",
        "buggy": """(define (integer->limbs n base)
  (if (= n 0)
      '(0)
      (let loop ([n n] [limbs '()])
        (if (= n 0)
            limbs
            (loop (quotient n base)
                  (cons (modulo n base) limbs))))))""",
        "note": "Output must be least-significant-first; this returns reversed order.",
    },
    {
        "fn": "limbs-normalize",
        "buggy": """(define (limbs-normalize limbs)
  (if (null? limbs)
      '(0)
      (if (= (car limbs) 0)
          (limbs-normalize (cdr limbs))
          limbs)))""",
        "note": "Must remove high-order trailing zeros, not low-order leading zeros.",
    },
    {
        "fn": "limbs-normalize",
        "buggy": """(define (limbs-normalize limbs)
  (let loop ([ls (reverse limbs)])
    (cond
      [(null? ls) '()]
      [(= (car ls) 0) (loop (cdr ls))]
      [else (reverse ls)])))""",
        "note": "Canonical zero representation must be '(0), and a single zero limb should be preserved.",
    },
    {
        "fn": "limbs-add",
        "buggy": """(define (limbs-add a b base)
  (let loop ([a a] [b b] [carry 0] [result '()])
    (cond
      [(and (null? a) (null? b))
       (limbs-normalize (reverse result))]
      [else
       (let* ([av (if (null? a) 0 (car a))]
              [bv (if (null? b) 0 (car b))]
              [sum (+ av bv carry)])
         (loop (if (null? a) '() (cdr a))
               (if (null? b) '() (cdr b))
               (quotient sum base)
               (cons (modulo sum base) result)))])))""",
        "note": "Final carry must be appended when both inputs are exhausted.",
    },
    {
        "fn": "limbs-add",
        "buggy": """(define (limbs-add a b base)
  (let loop ([a a] [b b] [result '()])
    (cond
      [(and (null? a) (null? b)) (reverse result)]
      [else
       (let* ([av (if (null? a) 0 (car a))]
              [bv (if (null? b) 0 (car b))]
              [sum (+ av bv)])
         (loop (if (null? a) '() (cdr a))
               (if (null? b) '() (cdr b))
               (cons (modulo sum base) result)))])))""",
        "note": "Carry propagation is missing; sums above base are handled incorrectly.",
    },
    {
        "fn": "limbs-sub",
        "buggy": """(define (limbs-sub a b base)
  (let loop ([a a] [b b] [result '()])
    (if (and (null? a) (null? b))
        (limbs-normalize (reverse result))
        (let* ([av (if (null? a) 0 (car a))]
               [bv (if (null? b) 0 (car b))]
               [diff (- av bv)])
          (loop (if (null? a) '() (cdr a))
                (if (null? b) '() (cdr b))
                (cons diff result))))))""",
        "note": "Borrow propagation is required when a limb subtraction goes negative.",
    },
    {
        "fn": "limbs-sub",
        "buggy": """(define (limbs-sub a b base)
  (let loop ([a a] [b b] [borrow 0] [result '()])
    (if (and (null? a) (null? b))
        (reverse result)
        (let* ([av (if (null? a) 0 (car a))]
               [bv (if (null? b) 0 (car b))]
               [diff (- av bv borrow)])
          (if (< diff 0)
              (loop (if (null? a) '() (cdr a))
                    (if (null? b) '() (cdr b))
                    1
                    (cons (+ diff base) result))
              (loop (if (null? a) '() (cdr a))
                    (if (null? b) '() (cdr b))
                    0
                    (cons diff result)))))))""",
        "note": "Result must be normalized to remove high-order trailing zeros.",
    },
    {
        "fn": "limbs-shift",
        "buggy": """(define (limbs-shift limbs k)
  (append limbs (make-list k 0)))""",
        "note": "Least-significant-first representation requires prepending zeros, not appending.",
    },
    {
        "fn": "limbs-shift",
        "buggy": """(define (limbs-shift limbs k)
  (append (make-list k 0) limbs))""",
        "note": "Canonical zero should stay '(0) after any shift, not gain extra zero limbs.",
    },
    {
        "fn": "limb-scale",
        "buggy": """(define (limb-scale limbs scalar base)
  (let loop ([ls limbs] [carry 0] [result '()])
    (if (null? ls)
        (limbs-normalize (reverse result))
        (let ([prod (+ (* (car ls) scalar) carry)])
          (loop (cdr ls)
                (quotient prod base)
                (cons (modulo prod base) result))))))""",
        "note": "Final carry must be emitted when scaling completes.",
    },
    {
        "fn": "limb-scale",
        "buggy": """(define (limb-scale limbs scalar base)
  (map (lambda (x) (* x scalar)) limbs))""",
        "note": "Scaling must propagate carries and stay in-base for each limb.",
    },
    {
        "fn": "limbs-multiply-schoolbook",
        "buggy": """(define (limbs-multiply-schoolbook a b base)
  (let loop ([b b] [result '(0)])
    (if (null? b)
        (limbs-normalize result)
        (let ([partial (limb-scale a (car b) base)])
          (loop (cdr b)
                (limbs-add result partial base)))))""",
        "note": "Each partial product must be shifted by its limb position.",
    },
    {
        "fn": "limbs-multiply-schoolbook",
        "buggy": """(define (limbs-multiply-schoolbook a b base)
  (let loop ([b b] [shift 0] [result '(0)])
    (if (null? b)
        result
        (let ([partial (limb-scale a (car b) base)])
          (loop (cdr b)
                (+ shift 1)
                (limbs-shift partial shift)))))""",
        "note": "Partial products must accumulate with addition; current code discards previous result.",
    },
]

DIFFICULTY = {
    "limbs->integer": "easy",
    "integer->limbs": "easy",
    "limbs-normalize": "easy",
    "limbs-add": "medium",
    "limbs-sub": "medium",
    "limbs-shift": "easy",
    "limb-scale": "medium",
    "limbs-multiply-schoolbook": "hard",
}

REQUIRED_KEYS = {
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
}

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
    sid = f"fast_multiply_{family}_{family_counter[family]:03d}"
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
    parts = [DEFS[d] for d in dependency_closure(fn)] + [DEFS[fn], VERIFY_BY_FUNCTION[fn]]
    return "(let ()\n  " + "\n  ".join(parts) + ")"


def wrap_verify_expr(expr: str) -> str:
    parts = [DEFS[name] for name in FUNCTION_ORDER] + [expr]
    return "(let ()\n  " + "\n  ".join(parts) + ")"


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
        tags=["tier0", "number-theory", "fast-multiply", "spec-to-code", fn],
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
        tags=["tier0", "number-theory", "fast-multiply", "spec-to-code", "skeleton", fn],
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
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "number-theory", "fast-multiply", "translation", "python", fn],
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
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "number-theory", "fast-multiply", "translation", "chez", fn],
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
        tags=["tier0", "number-theory", "fast-multiply", "bugfix", fn],
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
        verify_expr=wrap_verify_expr(verify_expr),
        tags=["tier0", "number-theory", "fast-multiply", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # limbs->integer
    (
        "limbs->integer",
        "Convert '(3 2 1) in base 10 to an integer.",
        "(limbs->integer '(3 2 1) 10)",
        "(equal? (limbs->integer '(3 2 1) 10) 123)",
        "easy",
        ["direct"],
    ),
    (
        "limbs->integer",
        "Convert '(0 1) in base 10 to an integer.",
        "(limbs->integer '(0 1) 10)",
        "(equal? (limbs->integer '(0 1) 10) 10)",
        "easy",
        ["direct"],
    ),
    (
        "limbs->integer",
        "Roundtrip 12345 through integer->limbs and back in base 100.",
        "(limbs->integer (integer->limbs 12345 100) 100)",
        "(equal? (limbs->integer (integer->limbs 12345 100) 100) 12345)",
        "medium",
        ["roundtrip"],
    ),
    (
        "limbs->integer",
        "Convert schoolbook product limbs for 12*34 in base 10 to integer.",
        "(limbs->integer (limbs-multiply-schoolbook '(2 1) '(4 3) 10) 10)",
        "(equal? (limbs->integer (limbs-multiply-schoolbook '(2 1) '(4 3) 10) 10) 408)",
        "medium",
        ["integration"],
    ),

    # integer->limbs
    (
        "integer->limbs",
        "Convert integer 123 in base 10 to limbs.",
        "(integer->limbs 123 10)",
        "(equal? (integer->limbs 123 10) '(3 2 1))",
        "easy",
        ["direct"],
    ),
    (
        "integer->limbs",
        "Convert integer 0 in base 10 to canonical zero limbs.",
        "(integer->limbs 0 10)",
        "(equal? (integer->limbs 0 10) '(0))",
        "easy",
        ["edge-case"],
    ),
    (
        "integer->limbs",
        "Convert 12345 in base 100.",
        "(integer->limbs 12345 100)",
        "(equal? (integer->limbs 12345 100) '(45 23 1))",
        "easy",
        ["direct"],
    ),
    (
        "integer->limbs",
        "Return #t iff converting then reconverting 67890 in base 1000 is identity.",
        "(= (limbs->integer (integer->limbs 67890 1000) 1000) 67890)",
        "(equal? (= (limbs->integer (integer->limbs 67890 1000) 1000) 67890) #t)",
        "medium",
        ["property"],
    ),

    # limbs-normalize
    (
        "limbs-normalize",
        "Normalize '(1 2 3 0 0).",
        "(limbs-normalize '(1 2 3 0 0))",
        "(equal? (limbs-normalize '(1 2 3 0 0)) '(1 2 3))",
        "easy",
        ["direct"],
    ),
    (
        "limbs-normalize",
        "Normalize all-zero limbs '(0 0 0).",
        "(limbs-normalize '(0 0 0))",
        "(equal? (limbs-normalize '(0 0 0)) '(0))",
        "easy",
        ["edge-case"],
    ),
    (
        "limbs-normalize",
        "Normalize already-canonical limbs '(7 8 9).",
        "(limbs-normalize '(7 8 9))",
        "(equal? (limbs-normalize '(7 8 9)) '(7 8 9))",
        "easy",
        ["direct"],
    ),
    (
        "limbs-normalize",
        "Normalize the result of subtracting '(0 0 2) - '(0 0 1) in base 10.",
        "(limbs-normalize (limbs-sub '(0 0 2) '(0 0 1) 10))",
        "(equal? (limbs-normalize (limbs-sub '(0 0 2) '(0 0 1) 10)) '(0 0 1))",
        "medium",
        ["integration"],
    ),

    # limbs-add
    (
        "limbs-add",
        "Add '(2) and '(3) in base 10.",
        "(limbs-add '(2) '(3) 10)",
        "(equal? (limbs-add '(2) '(3) 10) '(5))",
        "easy",
        ["direct"],
    ),
    (
        "limbs-add",
        "Add '(9 9) and '(1 1) in base 10 (carry propagation).",
        "(limbs-add '(9 9) '(1 1) 10)",
        "(equal? (limbs-add '(9 9) '(1 1) 10) '(0 1 1))",
        "medium",
        ["carry"],
    ),
    (
        "limbs-add",
        "Return #t iff addition is commutative for '(1 2 3) and '(4 5 6) in base 10.",
        "(equal? (limbs-add '(1 2 3) '(4 5 6) 10) (limbs-add '(4 5 6) '(1 2 3) 10))",
        "(equal? (limbs-add '(1 2 3) '(4 5 6) 10) (limbs-add '(4 5 6) '(1 2 3) 10))",
        "medium",
        ["property"],
    ),
    (
        "limbs-add",
        "Add scaled limbs: (scale '(2 7) by 8) + '(4) in base 10.",
        "(limbs-add (limb-scale '(2 7) 8 10) '(4) 10)",
        "(equal? (limbs-add (limb-scale '(2 7) 8 10) '(4) 10) '(0 8 5))",
        "hard",
        ["integration"],
    ),

    # limbs-sub
    (
        "limbs-sub",
        "Subtract '(3) from '(5) in base 10.",
        "(limbs-sub '(5) '(3) 10)",
        "(equal? (limbs-sub '(5) '(3) 10) '(2))",
        "easy",
        ["direct"],
    ),
    (
        "limbs-sub",
        "Subtract '(1) from '(0 0 1) in base 10 (borrow case).",
        "(limbs-sub '(0 0 1) '(1) 10)",
        "(equal? (limbs-sub '(0 0 1) '(1) 10) '(9 9))",
        "medium",
        ["borrow"],
    ),
    (
        "limbs-sub",
        "Compute '(5 7 9) - '(4 5 6) in base 10.",
        "(limbs-sub '(5 7 9) '(4 5 6) 10)",
        "(equal? (limbs-sub '(5 7 9) '(4 5 6) 10) '(1 2 3))",
        "medium",
        ["direct"],
    ),
    (
        "limbs-sub",
        "Return #t iff (a+b)-b = a for a='(3 2 1), b='(5 4) in base 10.",
        "(equal? (limbs-sub (limbs-add '(3 2 1) '(5 4) 10) '(5 4) 10) '(3 2 1))",
        "(equal? (limbs-sub (limbs-add '(3 2 1) '(5 4) 10) '(5 4) 10) '(3 2 1))",
        "hard",
        ["property"],
    ),

    # limbs-shift
    (
        "limbs-shift",
        "Shift '(1 2 3) left by 2 limbs.",
        "(limbs-shift '(1 2 3) 2)",
        "(equal? (limbs-shift '(1 2 3) 2) '(0 0 1 2 3))",
        "easy",
        ["direct"],
    ),
    (
        "limbs-shift",
        "Shift '(0) left by 3 limbs and keep canonical zero.",
        "(limbs-shift '(0) 3)",
        "(equal? (limbs-shift '(0) 3) '(0))",
        "easy",
        ["edge-case"],
    ),
    (
        "limbs-shift",
        "Shift '(5) by 0 limbs.",
        "(limbs-shift '(5) 0)",
        "(equal? (limbs-shift '(5) 0) '(5))",
        "easy",
        ["direct"],
    ),
    (
        "limbs-shift",
        "Convert shifted limbs '(3 2 1) by 2 (base 10) to integer.",
        "(limbs->integer (limbs-shift '(3 2 1) 2) 10)",
        "(equal? (limbs->integer (limbs-shift '(3 2 1) 2) 10) 12300)",
        "medium",
        ["integration"],
    ),

    # limb-scale
    (
        "limb-scale",
        "Scale '(5) by scalar 2 in base 10.",
        "(limb-scale '(5) 2 10)",
        "(equal? (limb-scale '(5) 2 10) '(0 1))",
        "easy",
        ["direct"],
    ),
    (
        "limb-scale",
        "Scale '(2 7) by scalar 8 in base 10.",
        "(limb-scale '(2 7) 8 10)",
        "(equal? (limb-scale '(2 7) 8 10) '(6 7 5))",
        "medium",
        ["direct"],
    ),
    (
        "limb-scale",
        "Scale any limbs by 0 and return canonical zero.",
        "(limb-scale '(9 8 7) 0 10)",
        "(equal? (limb-scale '(9 8 7) 0 10) '(0))",
        "easy",
        ["edge-case"],
    ),
    (
        "limb-scale",
        "Return #t iff limb-scale matches integer multiplication for '(2 7), scalar 8, base 10.",
        "(= (limbs->integer (limb-scale '(2 7) 8 10) 10) (* (limbs->integer '(2 7) 10) 8))",
        "(equal? (= (limbs->integer (limb-scale '(2 7) 8 10) 10) (* (limbs->integer '(2 7) 10) 8)) #t)",
        "hard",
        ["property"],
    ),

    # limbs-multiply-schoolbook
    (
        "limbs-multiply-schoolbook",
        "Multiply 12 by 34 in base-10 limb form.",
        "(limbs-multiply-schoolbook '(2 1) '(4 3) 10)",
        "(equal? (limbs-multiply-schoolbook '(2 1) '(4 3) 10) '(8 0 4))",
        "medium",
        ["direct"],
    ),
    (
        "limbs-multiply-schoolbook",
        "Multiply by zero in limb form.",
        "(limbs-multiply-schoolbook '(1 2 3) '(0) 10)",
        "(equal? (limbs-multiply-schoolbook '(1 2 3) '(0) 10) '(0))",
        "easy",
        ["edge-case"],
    ),
    (
        "limbs-multiply-schoolbook",
        "Compute integer product 12345*67890 via schoolbook limbs at base 1000.",
        "(limbs->integer (limbs-multiply-schoolbook (integer->limbs 12345 1000) (integer->limbs 67890 1000) 1000) 1000)",
        "(equal? (limbs->integer (limbs-multiply-schoolbook (integer->limbs 12345 1000) (integer->limbs 67890 1000) 1000) 1000) (* 12345 67890))",
        "hard",
        ["integration"],
    ),
    (
        "limbs-multiply-schoolbook",
        "Return #t iff schoolbook multiplication is commutative for (123,456) at base 10.",
        "(equal? (limbs-multiply-schoolbook (integer->limbs 123 10) (integer->limbs 456 10) 10) (limbs-multiply-schoolbook (integer->limbs 456 10) (integer->limbs 123 10) 10))",
        "(equal? (limbs-multiply-schoolbook (integer->limbs 123 10) (integer->limbs 456 10) 10) (limbs-multiply-schoolbook (integer->limbs 456 10) (integer->limbs 123 10) 10))",
        "hard",
        ["property"],
    ),
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
        if not swapped:
            continue

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
    "difficulty": dict(sorted(Counter(str(r["difficulty"]) for r in samples).items())),
    "source_functions": len(all_source_functions),
}

with SUMMARY_PATH.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print(f"Wrote: {ALL_PATH}")
print(f"Wrote: {TRAIN_PATH}")
print(f"Wrote: {EVAL_PATH}")
