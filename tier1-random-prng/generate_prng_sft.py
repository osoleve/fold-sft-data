#!/usr/bin/env python3
"""Generate Tier-1 random PRNG SFT samples for lattice/random/prng.ss."""

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

SOURCE_MODULE = "lattice/random/prng.ss"
SOURCE_TEST = "lattice/random/test-prng.ss"

GLOBAL_DEFS = [
    """(define mask-32 (- (expt 2 32) 1))""",
    """(define mask-64 (- (expt 2 64) 1))""",
    """(define (make-splitmix seed)
  (list 'splitmix (u64 seed)))""",
    """(define (splitmix-state sm)
  (cadr sm))""",
    """(define (rotr32 x k)
  (rotl32 x (- 32 k)))""",
    """(define (pcg-state p)
  (cadr p))""",
    """(define (pcg-inc p)
  (caddr p))""",
    """(define (rotl64 x k)
  (let ([x (u64 x)]
        [k (modulo k 64)])
    (u64 (bitwise-ior (ash x k)
                      (ash x (- k 64))))))""",
    """(define (xorshift128-s0 xs)
  (cadr xs))""",
    """(define (xorshift128-s1 xs)
  (caddr xs))""",
]

DEFS: Dict[str, str] = {
    "u32": """(define (u32 n)
  (bitwise-and n mask-32))""",
    "u64": """(define (u64 n)
  (bitwise-and n mask-64))""",
    "rotl32": """(define (rotl32 x k)
  (let ([x (u32 x)]
        [k (modulo k 32)])
    (u32 (bitwise-ior (ash x k)
                      (ash x (- k 32))))))""",
    "splitmix-next": """(define (splitmix-next sm)
  (let* ([s (u64 (+ (splitmix-state sm) #x9e3779b97f4a7c15))]
         [z (u64 (* (bitwise-xor s (ash s -30))
                    #xbf58476d1ce4e5b9))]
         [z (u64 (* (bitwise-xor z (ash z -27))
                    #x94d049bb133111eb))]
         [z (bitwise-xor z (ash z -31))])
    (cons z (make-splitmix s))))""",
    "make-pcg": """(define (make-pcg seed stream)
  (let* ([inc (u64 (bitwise-ior (ash stream 1) 1))]
         [state0 0]
         [state1 (u64 (+ (* state0 #x5851f42d4c957f2d) inc))]
         [state2 (u64 (+ state1 (u64 seed)))]
         [state3 (u64 (+ (* state2 #x5851f42d4c957f2d) inc))])
    (list 'pcg state3 inc)))""",
    "pcg-next": """(define (pcg-next p)
  (let* ([state (pcg-state p)]
         [inc (pcg-inc p)]
         [xorshifted (u32 (ash
                           (bitwise-xor (ash state -18) state)
                           -27))]
         [rot (ash state -59)]
         [output (rotr32 xorshifted rot)]
         [new-state (u64 (+ (* state #x5851f42d4c957f2d) inc))])
    (cons output (list 'pcg new-state inc))))""",
    "make-xorshift128": """(define (make-xorshift128 seed)
  (let* ([sm0 (make-splitmix seed)]
         [r1 (splitmix-next sm0)]
         [s0 (car r1)]
         [sm1 (cdr r1)]
         [r2 (splitmix-next sm1)]
         [s1 (car r2)])
    (list 'xorshift128
          (if (= s0 0) 1 s0)
          (if (= s1 0) 1 s1))))""",
    "xorshift128-next": """(define (xorshift128-next xs)
  (let* ([s0 (xorshift128-s0 xs)]
         [s1 (xorshift128-s1 xs)]
         [result (u64 (+ s0 s1))]
         [s1-new (bitwise-xor s0 s1)]
         [new-s0 (u64 (bitwise-xor
                       (bitwise-xor (rotl64 s0 24) s1-new)
                       (ash s1-new 16)))]
         [new-s1 (rotl64 s1-new 37)])
    (cons result (list 'xorshift128 new-s0 new-s1))))""",
}

FUNCTION_ORDER = [
    "u32",
    "u64",
    "rotl32",
    "splitmix-next",
    "make-pcg",
    "pcg-next",
    "make-xorshift128",
    "xorshift128-next",
]

FUNCTION_SPECS = {
    "u32": "Truncate an integer to unsigned 32-bit by masking with (2^32 - 1).",
    "u64": "Truncate an integer to unsigned 64-bit by masking with (2^64 - 1).",
    "rotl32": "Rotate a 32-bit integer left by k bits with k reduced modulo 32.",
    "splitmix-next": "Advance splitmix64 state once and return (output . next-generator).",
    "make-pcg": "Construct PCG state (pcg state inc) with odd increment derived from stream and warmed state progression.",
    "pcg-next": "Run one PCG-XSH-RR step: 32-bit output plus updated 64-bit state.",
    "make-xorshift128": "Seed xorshift128+ from splitmix outputs and force both state words non-zero.",
    "xorshift128-next": "Run one xorshift128+ step and return (output . next-state).",
}

SKELETONS = {
    "u32": """(define (u32 n)
  ;; TODO: mask to unsigned 32-bit range
  <TODO>)""",
    "u64": """(define (u64 n)
  ;; TODO: mask to unsigned 64-bit range
  <TODO>)""",
    "rotl32": """(define (rotl32 x k)
  ;; TODO: perform 32-bit rotate-left with modulo-32 shift
  <TODO>)""",
    "splitmix-next": """(define (splitmix-next sm)
  ;; TODO: apply splitmix64 mixing pipeline and return (value . next-state)
  <TODO>)""",
    "make-pcg": """(define (make-pcg seed stream)
  ;; TODO: derive odd increment, warm state, return '(pcg state inc)
  <TODO>)""",
    "pcg-next": """(define (pcg-next p)
  ;; TODO: compute XSH-RR output and LCG state transition
  <TODO>)""",
    "make-xorshift128": """(define (make-xorshift128 seed)
  ;; TODO: seed from splitmix and enforce non-zero state words
  <TODO>)""",
    "xorshift128-next": """(define (xorshift128-next xs)
  ;; TODO: execute xorshift128+ transition and return (value . next-state)
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "u32": """(and
  (= (u32 #xFFFFFFFF) #xFFFFFFFF)
  (= (u32 #x100000000) 0)
  (= (u32 -1) #xFFFFFFFF)
  (= (u32 0) 0))""",
    "u64": """(and
  (= (u64 #xFFFFFFFFFFFFFFFF) #xFFFFFFFFFFFFFFFF)
  (= (u64 #x10000000000000000) 0)
  (= (u64 -1) #xFFFFFFFFFFFFFFFF)
  (= (u64 0) 0))""",
    "rotl32": """(and
  (= (rotl32 1 1) 2)
  (= (rotl32 #x80000000 1) 1)
  (= (rotl32 #x12345678 4) #x23456781)
  (= (rotl32 1 33) 2))""",
    "splitmix-next": """(let* ([sm0 (make-splitmix 42)]
       [r1 (splitmix-next sm0)]
       [v1 (car r1)]
       [sm1 (cdr r1)]
       [r2 (splitmix-next sm1)]
       [v2 (car r2)])
  (and (<= 0 v1)
       (< v1 (expt 2 64))
       (<= 0 v2)
       (< v2 (expt 2 64))
       (not (= v1 v2))
       (= (splitmix-state sm1)
          (u64 (+ 42 #x9e3779b97f4a7c15)))))""",
    "make-pcg": """(let* ([p1 (make-pcg 42 7)]
       [p2 (make-pcg 42 7)]
       [p3 (make-pcg 42 8)])
  (and (equal? p1 p2)
       (not (equal? p1 p3))
       (eq? (car p1) 'pcg)
       (odd? (pcg-inc p1))
       (< (pcg-state p1) (expt 2 64))
       (< (pcg-inc p1) (expt 2 64))))""",
    "pcg-next": """(let* ([p0 (make-pcg 12345 1)]
       [r1 (pcg-next p0)]
       [out1 (car r1)]
       [p1 (cdr r1)]
       [r1b (pcg-next p0)])
  (and (= out1 (car r1b))
       (equal? p1 (cdr r1b))
       (<= 0 out1)
       (< out1 (expt 2 32))
       (= (pcg-inc p0) (pcg-inc p1))
       (= (pcg-state p1)
          (u64 (+ (* (pcg-state p0) #x5851f42d4c957f2d) (pcg-inc p0))))))""",
    "make-xorshift128": """(let* ([xs0 (make-xorshift128 0)]
       [xs1 (make-xorshift128 0)]
       [xs2 (make-xorshift128 1)])
  (and (equal? xs0 xs1)
       (not (equal? xs0 xs2))
       (eq? (car xs0) 'xorshift128)
       (not (= (xorshift128-s0 xs0) 0))
       (not (= (xorshift128-s1 xs0) 0))))""",
    "xorshift128-next": """(let* ([xs (make-xorshift128 42)]
       [r1 (xorshift128-next xs)]
       [v1 (car r1)]
       [xs1 (cdr r1)]
       [r1b (xorshift128-next xs)])
  (and (= v1 (car r1b))
       (equal? xs1 (cdr r1b))
       (= v1 (u64 (+ (xorshift128-s0 xs) (xorshift128-s1 xs))))
       (<= 0 v1)
       (< v1 (expt 2 64))
       (eq? (car xs1) 'xorshift128)))""",
}

DEPENDS: Dict[str, List[str]] = {
    "u32": [],
    "u64": [],
    "rotl32": ["u32"],
    "splitmix-next": ["u64"],
    "make-pcg": ["u64"],
    "pcg-next": ["u32", "u64", "rotl32", "make-pcg"],
    "make-xorshift128": ["u64", "splitmix-next"],
    "xorshift128-next": ["u64", "make-xorshift128", "splitmix-next"],
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")
TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "u32": """def u32(n):
    return n & ((1 << 32) - 1)""",
    "u64": """def u64(n):
    return n & ((1 << 64) - 1)""",
    "rotl32": """def rotl32(x, k):
    x = u32(x)
    k %= 32
    return u32((x << k) | (x >> (32 - k)))""",
    "splitmix-next": """def splitmix_next(sm):
    s = u64(splitmix_state(sm) + 0x9E3779B97F4A7C15)
    z = u64((s ^ (s >> 30)) * 0xBF58476D1CE4E5B9)
    z = u64((z ^ (z >> 27)) * 0x94D049BB133111EB)
    z = z ^ (z >> 31)
    return z, make_splitmix(s)""",
    "make-pcg": """def make_pcg(seed, stream):
    inc = u64((stream << 1) | 1)
    state0 = 0
    state1 = u64(state0 * 0x5851F42D4C957F2D + inc)
    state2 = u64(state1 + u64(seed))
    state3 = u64(state2 * 0x5851F42D4C957F2D + inc)
    return ["pcg", state3, inc]""",
    "pcg-next": """def pcg_next(p):
    state = pcg_state(p)
    inc = pcg_inc(p)
    xorshifted = u32(((state >> 18) ^ state) >> 27)
    rot = state >> 59
    output = rotr32(xorshifted, rot)
    new_state = u64(state * 0x5851F42D4C957F2D + inc)
    return output, ["pcg", new_state, inc]""",
    "make-xorshift128": """def make_xorshift128(seed):
    sm0 = make_splitmix(seed)
    r1 = splitmix_next(sm0)
    s0 = r1[0]
    sm1 = r1[1]
    r2 = splitmix_next(sm1)
    s1 = r2[0]
    return ["xorshift128", 1 if s0 == 0 else s0, 1 if s1 == 0 else s1]""",
    "xorshift128-next": """def xorshift128_next(xs):
    s0 = xorshift128_s0(xs)
    s1 = xorshift128_s1(xs)
    result = u64(s0 + s1)
    s1_new = s0 ^ s1
    new_s0 = u64((rotl64(s0, 24) ^ s1_new) ^ (s1_new << 16))
    new_s1 = rotl64(s1_new, 37)
    return result, ["xorshift128", new_s0, new_s1]""",
}

CHEZ_SNIPPETS = {
    "u32": """(define (to-u32 n)
  (bitwise-and n #xFFFFFFFF))""",
    "u64": """(define (to-u64 n)
  (bitwise-and n #xFFFFFFFFFFFFFFFF))""",
    "rotl32": """(define (rotate-left32 x k)
  (let ([x (u32 x)]
        [k (modulo k 32)])
    (u32 (bitwise-ior (ash x k)
                      (ash x (- k 32))))))""",
    "splitmix-next": """(define (splitmix-step sm)
  (let* ([s (u64 (+ (splitmix-state sm) #x9e3779b97f4a7c15))]
         [z (u64 (* (bitwise-xor s (ash s -30)) #xbf58476d1ce4e5b9))]
         [z (u64 (* (bitwise-xor z (ash z -27)) #x94d049bb133111eb))]
         [z (bitwise-xor z (ash z -31))])
    (cons z (make-splitmix s))))""",
    "make-pcg": """(define (pcg-init seed stream)
  (let* ([inc (u64 (bitwise-ior (ash stream 1) 1))]
         [state0 0]
         [state1 (u64 (+ (* state0 #x5851f42d4c957f2d) inc))]
         [state2 (u64 (+ state1 (u64 seed)))]
         [state3 (u64 (+ (* state2 #x5851f42d4c957f2d) inc))])
    (list 'pcg state3 inc)))""",
    "pcg-next": """(define (pcg-step p)
  (let* ([state (pcg-state p)]
         [inc (pcg-inc p)]
         [xorshifted (u32 (ash (bitwise-xor (ash state -18) state) -27))]
         [rot (ash state -59)]
         [output (rotr32 xorshifted rot)]
         [new-state (u64 (+ (* state #x5851f42d4c957f2d) inc))])
    (cons output (list 'pcg new-state inc))))""",
    "make-xorshift128": """(define (xorshift-init seed)
  (let* ([sm0 (make-splitmix seed)]
         [r1 (splitmix-next sm0)]
         [s0 (car r1)]
         [sm1 (cdr r1)]
         [r2 (splitmix-next sm1)]
         [s1 (car r2)])
    (list 'xorshift128
          (if (= s0 0) 1 s0)
          (if (= s1 0) 1 s1))))""",
    "xorshift128-next": """(define (xorshift-step xs)
  (let* ([s0 (xorshift128-s0 xs)]
         [s1 (xorshift128-s1 xs)]
         [result (u64 (+ s0 s1))]
         [s1-new (bitwise-xor s0 s1)]
         [new-s0 (u64 (bitwise-xor
                       (bitwise-xor (rotl64 s0 24) s1-new)
                       (ash s1-new 16)))]
         [new-s1 (rotl64 s1-new 37)])
    (cons result (list 'xorshift128 new-s0 new-s1))))""",
}

BUGGY_CASES = [
    {
        "fn": "u32",
        "buggy": """(define (u32 n)
  (bitwise-and n #x7FFFFFFF))""",
        "note": "The mask drops the high bit; unsigned 32-bit truncation must keep all 32 bits.",
    },
    {
        "fn": "u32",
        "buggy": """(define (u32 n)
  (u64 n))""",
        "note": "This keeps 64 bits instead of truncating to 32 bits.",
    },
    {
        "fn": "u64",
        "buggy": """(define (u64 n)
  (bitwise-and n #xFFFFFFFF))""",
        "note": "This truncates to 32 bits, not 64 bits.",
    },
    {
        "fn": "u64",
        "buggy": """(define (u64 n)
  (abs n))""",
        "note": "Unsigned wrapping is required; abs does not implement 64-bit masking semantics.",
    },
    {
        "fn": "rotl32",
        "buggy": """(define (rotl32 x k)
  (let ([x (u32 x)])
    (u32 (bitwise-ior (ash x k)
                      (ash x (- k 32))))))""",
        "note": "Shift amount must be reduced modulo 32 before rotation.",
    },
    {
        "fn": "rotl32",
        "buggy": """(define (rotl32 x k)
  (let ([x (u32 x)]
        [k (modulo k 32)])
    (u32 (bitwise-ior (ash x k)
                      (ash x (- k))))))""",
        "note": "The right-shift component uses -k instead of (k-32), breaking rotate behavior.",
    },
    {
        "fn": "splitmix-next",
        "buggy": """(define (splitmix-next sm)
  (let* ([s (u64 (+ (splitmix-state sm) #x9e3779b97f4a7c15))])
    (cons s (make-splitmix s))))""",
        "note": "Returning raw incremented state skips the required splitmix mixing rounds.",
    },
    {
        "fn": "splitmix-next",
        "buggy": """(define (splitmix-next sm)
  (let* ([s (u64 (+ (splitmix-state sm) #x9e3779b97f4a7c15))]
         [z (u64 (* (bitwise-xor s (ash s -30)) #xbf58476d1ce4e5b9))]
         [z (u64 (* (bitwise-xor z (ash z -27)) #x94d049bb133111eb))])
    (cons z (make-splitmix s))))""",
        "note": "The final `xor`/`ash -31` diffusion step is missing, so output quality is degraded.",
    },
    {
        "fn": "make-pcg",
        "buggy": """(define (make-pcg seed stream)
  (let* ([inc (u64 (ash stream 1))]
         [state0 0]
         [state1 (u64 (+ (* state0 #x5851f42d4c957f2d) inc))]
         [state2 (u64 (+ state1 (u64 seed)))]
         [state3 (u64 (+ (* state2 #x5851f42d4c957f2d) inc))])
    (list 'pcg state3 inc)))""",
        "note": "PCG increment must be odd; dropping the low-bit OR can create invalid streams.",
    },
    {
        "fn": "make-pcg",
        "buggy": """(define (make-pcg seed stream)
  (let* ([inc (u64 (bitwise-ior (ash stream 1) 1))])
    (list 'pcg (u64 seed) inc)))""",
        "note": "PCG seeding requires warm-up transitions, not direct state assignment.",
    },
    {
        "fn": "pcg-next",
        "buggy": """(define (pcg-next p)
  (let* ([state (pcg-state p)]
         [inc (pcg-inc p)]
         [xorshifted (u32 (ash
                           (bitwise-xor (ash state -18) state)
                           -27))]
         [rot (ash state -58)]
         [output (rotr32 xorshifted rot)]
         [new-state (u64 (+ (* state #x5851f42d4c957f2d) inc))])
    (cons output (list 'pcg new-state inc))))""",
        "note": "Rotation count must use the top 5 bits (state >> 59), not >> 58.",
    },
    {
        "fn": "pcg-next",
        "buggy": """(define (pcg-next p)
  (let* ([state (pcg-state p)]
         [inc (pcg-inc p)]
         [xorshifted (ash
                      (bitwise-xor (ash state -18) state)
                      -27)]
         [rot (ash state -59)]
         [output (rotr32 xorshifted rot)]
         [new-state (u64 (+ (* state #x5851f42d4c957f2d) inc))])
    (cons output (list 'pcg new-state inc))))""",
        "note": "xorshifted must be truncated to 32 bits before rotation.",
    },
    {
        "fn": "make-xorshift128",
        "buggy": """(define (make-xorshift128 seed)
  (let* ([sm0 (make-splitmix seed)]
         [r1 (splitmix-next sm0)]
         [s0 (car r1)]
         [sm1 (cdr r1)]
         [r2 (splitmix-next sm1)]
         [s1 (car r2)])
    (list 'xorshift128 s0 s1)))""",
        "note": "State words must be forced non-zero for robustness of xorshift128+.",
    },
    {
        "fn": "make-xorshift128",
        "buggy": """(define (make-xorshift128 seed)
  (let* ([sm0 (make-splitmix seed)]
         [r1 (splitmix-next sm0)]
         [s0 (car r1)]
         [s1 (car r1)])
    (list 'xorshift128 (if (= s0 0) 1 s0) (if (= s1 0) 1 s1))))""",
        "note": "The second state word must come from a second splitmix step, not reused from the first.",
    },
    {
        "fn": "xorshift128-next",
        "buggy": """(define (xorshift128-next xs)
  (let* ([s0 (xorshift128-s0 xs)]
         [s1 (xorshift128-s1 xs)]
         [result (u64 (bitwise-xor s0 s1))]
         [s1-new (bitwise-xor s0 s1)]
         [new-s0 (u64 (bitwise-xor
                       (bitwise-xor (rotl64 s0 24) s1-new)
                       (ash s1-new 16)))]
         [new-s1 (rotl64 s1-new 37)])
    (cons result (list 'xorshift128 new-s0 new-s1))))""",
        "note": "xorshift128+ output is sum mod 2^64, not xor.",
    },
    {
        "fn": "xorshift128-next",
        "buggy": """(define (xorshift128-next xs)
  (let* ([s0 (xorshift128-s0 xs)]
         [s1 (xorshift128-s1 xs)]
         [result (u64 (+ s0 s1))]
         [s1-new (bitwise-xor s0 s1)]
         [new-s0 (u64 (bitwise-xor
                       (bitwise-xor (rotl64 s0 24) s1-new)
                       (ash s1-new 16)))])
    (cons result (list 'xorshift128 new-s0 s1-new))))""",
        "note": "The second state word must be rotated by 37 bits each step.",
    },
]

DIFFICULTY = {
    "u32": "easy",
    "u64": "easy",
    "rotl32": "medium",
    "splitmix-next": "medium",
    "make-pcg": "hard",
    "pcg-next": "hard",
    "make-xorshift128": "hard",
    "xorshift128-next": "hard",
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
    sid = f"random_prng_{family}_{family_counter[family]:03d}"
    sample = {
        "id": sid,
        "family": family,
        "category": category,
        "difficulty": difficulty,
        "source_module": SOURCE_MODULE,
        "source_test": SOURCE_TEST,
        "source_function": source_function,
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
    parts = GLOBAL_DEFS + [DEFS[d] for d in dependency_closure(fn)] + [VERIFY_BY_FUNCTION[fn]]
    return "(let ()\n  " + "\n  ".join(parts) + ")"


def wrap_verify_expr(expr: str) -> str:
    parts = GLOBAL_DEFS + [DEFS[name] for name in FUNCTION_ORDER] + [expr]
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
        tags=["tier1", "random", "prng", "spec-to-code", fn],
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
        tags=["tier1", "random", "prng", "skeleton", fn],
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
        tags=["tier1", "random", "prng", "translation", "python", fn],
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
        tags=["tier1", "random", "prng", "translation", "chez", fn],
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
        tags=["tier1", "random", "prng", "bugfix", fn],
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
        tags=["tier1", "random", "prng", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # u32
    (
        "u32",
        "Truncate `#x100000001` to unsigned 32-bit.",
        "(u32 #x100000001)",
        "(equal? (u32 #x100000001) 1)",
        "easy",
        ["direct"],
    ),
    (
        "u32",
        "Map `u32` over the integers `'(0 -1 #x100000000)` and return the result list.",
        "(map u32 '(0 -1 #x100000000))",
        "(equal? (map u32 '(0 -1 #x100000000)) '(0 #xFFFFFFFF 0))",
        "easy",
        ["map"],
    ),
    (
        "u32",
        "Return #t iff applying `u32` twice is idempotent for `-17`.",
        "(= (u32 (u32 -17)) (u32 -17))",
        "(equal? (= (u32 (u32 -17)) (u32 -17)) #t)",
        "medium",
        ["property"],
    ),
    (
        "u32",
        "Compute the low 32 bits of `(+ #xFFFFFFFF 2)`.",
        "(u32 (+ #xFFFFFFFF 2))",
        "(equal? (u32 (+ #xFFFFFFFF 2)) 1)",
        "easy",
        ["arithmetic"],
    ),

    # u64
    (
        "u64",
        "Truncate `#x10000000000000005` to unsigned 64-bit.",
        "(u64 #x10000000000000005)",
        "(equal? (u64 #x10000000000000005) 5)",
        "easy",
        ["direct"],
    ),
    (
        "u64",
        "Map `u64` over `'(0 -1 #x10000000000000000)`.",
        "(map u64 '(0 -1 #x10000000000000000))",
        "(equal? (map u64 '(0 -1 #x10000000000000000)) '(0 #xFFFFFFFFFFFFFFFF 0))",
        "easy",
        ["map"],
    ),
    (
        "u64",
        "Return #t iff `u64` is idempotent on `-987654321`.",
        "(= (u64 (u64 -987654321)) (u64 -987654321))",
        "(equal? (= (u64 (u64 -987654321)) (u64 -987654321)) #t)",
        "medium",
        ["property"],
    ),
    (
        "u64",
        "Compute wrapped addition: low 64 bits of `(+ #xFFFFFFFFFFFFFFFF 2)`.",
        "(u64 (+ #xFFFFFFFFFFFFFFFF 2))",
        "(equal? (u64 (+ #xFFFFFFFFFFFFFFFF 2)) 1)",
        "easy",
        ["arithmetic"],
    ),

    # rotl32
    (
        "rotl32",
        "Rotate `#x12345678` left by 8 bits.",
        "(rotl32 #x12345678 8)",
        "(equal? (rotl32 #x12345678 8) #x34567812)",
        "medium",
        ["direct"],
    ),
    (
        "rotl32",
        "Return #t iff rotating by 40 is equivalent to rotating by 8.",
        "(= (rotl32 #xDEADBEEF 40) (rotl32 #xDEADBEEF 8))",
        "(equal? (= (rotl32 #xDEADBEEF 40) (rotl32 #xDEADBEEF 8)) #t)",
        "medium",
        ["modulo"],
    ),
    (
        "rotl32",
        "Rotate then undo with `rotr32` for `#x89ABCDEF` and shift 13; return whether original value is recovered.",
        "(= (rotr32 (rotl32 #x89ABCDEF 13) 13) (u32 #x89ABCDEF))",
        "(equal? (= (rotr32 (rotl32 #x89ABCDEF 13) 13) (u32 #x89ABCDEF)) #t)",
        "medium",
        ["inverse"],
    ),
    (
        "rotl32",
        "Rotate `1` left by 31 bits.",
        "(rotl32 1 31)",
        "(equal? (rotl32 1 31) #x80000000)",
        "easy",
        ["edge-case"],
    ),

    # splitmix-next
    (
        "splitmix-next",
        "Advance `(make-splitmix 5)` once and return the next generator state value.",
        "(splitmix-state (cdr (splitmix-next (make-splitmix 5))))",
        "(equal? (splitmix-state (cdr (splitmix-next (make-splitmix 5)))) (u64 (+ 5 #x9e3779b97f4a7c15)))",
        "medium",
        ["state"],
    ),
    (
        "splitmix-next",
        "Return #t iff the first splitmix output is deterministic for seed 7.",
        "(let ([a (car (splitmix-next (make-splitmix 7)))] [b (car (splitmix-next (make-splitmix 7)))]) (= a b))",
        "(equal? (let ([a (car (splitmix-next (make-splitmix 7)))] [b (car (splitmix-next (make-splitmix 7)))]) (= a b)) #t)",
        "medium",
        ["determinism"],
    ),
    (
        "splitmix-next",
        "Return #t iff two consecutive outputs from the same seed are different.",
        "(let* ([g (make-splitmix 7)] [r1 (splitmix-next g)] [r2 (splitmix-next (cdr r1))]) (not (= (car r1) (car r2))))",
        "(equal? (let* ([g (make-splitmix 7)] [r1 (splitmix-next g)] [r2 (splitmix-next (cdr r1))]) (not (= (car r1) (car r2)))) #t)",
        "hard",
        ["sequence"],
    ),
    (
        "splitmix-next",
        "Check that splitmix output always lies in unsigned 64-bit range for seed 123.",
        "(let ([v (car (splitmix-next (make-splitmix 123)))]) (and (<= 0 v) (< v (expt 2 64))))",
        "(equal? (let ([v (car (splitmix-next (make-splitmix 123)))]) (and (<= 0 v) (< v (expt 2 64)))) #t)",
        "medium",
        ["range"],
    ),

    # make-pcg
    (
        "make-pcg",
        "Create `(make-pcg 42 7)` and return its increment field.",
        "(pcg-inc (make-pcg 42 7))",
        "(equal? (modulo (pcg-inc (make-pcg 42 7)) 2) 1)",
        "medium",
        ["direct"],
    ),
    (
        "make-pcg",
        "Return #t iff `make-pcg` is deterministic for identical seed/stream.",
        "(equal? (make-pcg 42 7) (make-pcg 42 7))",
        "(equal? (equal? (make-pcg 42 7) (make-pcg 42 7)) #t)",
        "easy",
        ["determinism"],
    ),
    (
        "make-pcg",
        "Return #t iff different stream ids produce different PCG states for seed 42.",
        "(not (equal? (make-pcg 42 7) (make-pcg 42 8)))",
        "(equal? (not (equal? (make-pcg 42 7) (make-pcg 42 8))) #t)",
        "medium",
        ["streams"],
    ),
    (
        "make-pcg",
        "Generate one value from `(make-pcg 99 1)` and check that output is 32-bit.",
        "(let ([v (car (pcg-next (make-pcg 99 1)))]) (and (<= 0 v) (< v (expt 2 32))))",
        "(equal? (let ([v (car (pcg-next (make-pcg 99 1)))]) (and (<= 0 v) (< v (expt 2 32)))) #t)",
        "hard",
        ["integration"],
    ),

    # pcg-next
    (
        "pcg-next",
        "Run `pcg-next` on `(make-pcg 12345 1)` and return the output value.",
        "(car (pcg-next (make-pcg 12345 1)))",
        "(let ([v (car (pcg-next (make-pcg 12345 1)))]) (and (<= 0 v) (< v (expt 2 32))))",
        "hard",
        ["direct"],
    ),
    (
        "pcg-next",
        "Return #t iff one-step state transition matches the PCG LCG recurrence.",
        "(let* ([p0 (make-pcg 12345 1)] [p1 (cdr (pcg-next p0))]) (= (pcg-state p1) (u64 (+ (* (pcg-state p0) #x5851f42d4c957f2d) (pcg-inc p0)))))",
        "(equal? (let* ([p0 (make-pcg 12345 1)] [p1 (cdr (pcg-next p0))]) (= (pcg-state p1) (u64 (+ (* (pcg-state p0) #x5851f42d4c957f2d) (pcg-inc p0))))) #t)",
        "hard",
        ["state-update"],
    ),
    (
        "pcg-next",
        "Return #t iff `pcg-next` is deterministic for the same input state.",
        "(let* ([p (make-pcg 77 3)] [r1 (pcg-next p)] [r2 (pcg-next p)]) (and (= (car r1) (car r2)) (equal? (cdr r1) (cdr r2))))",
        "(equal? (let* ([p (make-pcg 77 3)] [r1 (pcg-next p)] [r2 (pcg-next p)]) (and (= (car r1) (car r2)) (equal? (cdr r1) (cdr r2)))) #t)",
        "hard",
        ["determinism"],
    ),
    (
        "pcg-next",
        "Run two PCG steps and return #t iff increment stays unchanged.",
        "(let* ([p0 (make-pcg 77 3)] [p1 (cdr (pcg-next p0))] [p2 (cdr (pcg-next p1))]) (= (pcg-inc p0) (pcg-inc p2)))",
        "(equal? (let* ([p0 (make-pcg 77 3)] [p1 (cdr (pcg-next p0))] [p2 (cdr (pcg-next p1))]) (= (pcg-inc p0) (pcg-inc p2))) #t)",
        "hard",
        ["invariant"],
    ),

    # make-xorshift128
    (
        "make-xorshift128",
        "Create `(make-xorshift128 0)` and return whether both state words are non-zero.",
        "(let ([xs (make-xorshift128 0)]) (and (not (= (xorshift128-s0 xs) 0)) (not (= (xorshift128-s1 xs) 0))))",
        "(equal? (let ([xs (make-xorshift128 0)]) (and (not (= (xorshift128-s0 xs) 0)) (not (= (xorshift128-s1 xs) 0)))) #t)",
        "hard",
        ["edge-case"],
    ),
    (
        "make-xorshift128",
        "Return #t iff xorshift constructor is deterministic for identical seed 42.",
        "(equal? (make-xorshift128 42) (make-xorshift128 42))",
        "(equal? (equal? (make-xorshift128 42) (make-xorshift128 42)) #t)",
        "easy",
        ["determinism"],
    ),
    (
        "make-xorshift128",
        "Return #t iff seeds 1 and 2 produce distinct initial xorshift states.",
        "(not (equal? (make-xorshift128 1) (make-xorshift128 2)))",
        "(equal? (not (equal? (make-xorshift128 1) (make-xorshift128 2))) #t)",
        "medium",
        ["seeds"],
    ),
    (
        "make-xorshift128",
        "Create `(make-xorshift128 9)` and check that tag symbol is `xorshift128`.",
        "(car (make-xorshift128 9))",
        "(equal? (car (make-xorshift128 9)) 'xorshift128)",
        "easy",
        ["shape"],
    ),

    # xorshift128-next
    (
        "xorshift128-next",
        "Run one xorshift step from `(make-xorshift128 42)` and return output value.",
        "(car (xorshift128-next (make-xorshift128 42)))",
        "(let ([v (car (xorshift128-next (make-xorshift128 42)))]) (and (<= 0 v) (< v (expt 2 64))))",
        "hard",
        ["direct"],
    ),
    (
        "xorshift128-next",
        "Return #t iff xorshift output equals `(u64 (+ s0 s1))` for current state.",
        "(let* ([xs (make-xorshift128 42)] [v (car (xorshift128-next xs))]) (= v (u64 (+ (xorshift128-s0 xs) (xorshift128-s1 xs)))))",
        "(equal? (let* ([xs (make-xorshift128 42)] [v (car (xorshift128-next xs))]) (= v (u64 (+ (xorshift128-s0 xs) (xorshift128-s1 xs))))) #t)",
        "hard",
        ["formula"],
    ),
    (
        "xorshift128-next",
        "Return #t iff `xorshift128-next` is deterministic for the same input state.",
        "(let* ([xs (make-xorshift128 5)] [r1 (xorshift128-next xs)] [r2 (xorshift128-next xs)]) (and (= (car r1) (car r2)) (equal? (cdr r1) (cdr r2))))",
        "(equal? (let* ([xs (make-xorshift128 5)] [r1 (xorshift128-next xs)] [r2 (xorshift128-next xs)]) (and (= (car r1) (car r2)) (equal? (cdr r1) (cdr r2)))) #t)",
        "hard",
        ["determinism"],
    ),
    (
        "xorshift128-next",
        "Run one step and check that the returned state tag remains `xorshift128`.",
        "(car (cdr (xorshift128-next (make-xorshift128 5))))",
        "(equal? (car (cdr (xorshift128-next (make-xorshift128 5)))) 'xorshift128)",
        "medium",
        ["state-shape"],
    ),
]

for fn, prompt, gt, verify_expr, difficulty, tags in composition_cases:
    add_composition(fn, prompt, gt, verify_expr, difficulty, tags)

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
