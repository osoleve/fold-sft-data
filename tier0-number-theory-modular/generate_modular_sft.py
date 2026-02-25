#!/usr/bin/env python3
"""Generate Tier-0 SFT samples for lattice/number-theory/modular.ss.

Outputs:
  - data/tier0-number-theory-modular/all.jsonl
  - data/tier0-number-theory-modular/train.jsonl
  - data/tier0-number-theory-modular/eval.jsonl
"""

from __future__ import annotations

import json
import hashlib
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

SOURCE_MODULE = "lattice/number-theory/modular.ss"
SOURCE_TEST = "lattice/number-theory/test-modular.ss"

DEFS: Dict[str, str] = {
    "mod+": """(define (mod+ a b m)
  (modulo (+ a b) m))""",
    "mod-": """(define (mod- a b m)
  (modulo (- a b) m))""",
    "mod*": """(define (mod* a b m)
  (modulo (* a b) m))""",
    "mod-expt": """(define (mod-expt base exp m)
  (let loop ([b (modulo base m)]
             [e exp]
             [result 1])
    (cond
      [(= e 0) (modulo result m)]
      [(odd? e)
       (loop (modulo (* b b) m)
             (quotient e 2)
             (modulo (* result b) m))]
      [else
       (loop (modulo (* b b) m)
             (quotient e 2)
             result)])))""",
    "gcd": """(define (gcd a b)
  (if (= b 0)
      (abs a)
      (gcd b (modulo a b))))""",
    "extended-gcd": """(define (extended-gcd a b)
  (let loop ([old-r a] [r b]
             [old-s 1] [s 0]
             [old-t 0] [t 1])
    (if (= r 0)
        (list old-r old-s old-t)
        (let* ([q (quotient old-r r)]
               [new-r (- old-r (* q r))]
               [new-s (- old-s (* q s))]
               [new-t (- old-t (* q t))])
          (loop r new-r s new-s t new-t)))))""",
    "mod-inverse": """(define (mod-inverse a m)
  (let* ([result (extended-gcd a m)]
         [g (car result)]
         [x (cadr result)])
    (if (= g 1)
        (modulo x m)
        #f)))""",
    "crt": """(define (crt remainders moduli)
  (if (or (null? remainders) (null? moduli))
      0
      (let ([M (fold-left * 1 moduli)])
        (let loop ([rs remainders]
                   [ms moduli]
                   [result 0])
          (if (null? rs)
              (modulo result M)
              (let* ([ai (car rs)]
                     [mi (car ms)]
                     [Mi (quotient M mi)]
                     [yi (mod-inverse Mi mi)])
                (if yi
                    (loop (cdr rs)
                          (cdr ms)
                          (+ result (* ai Mi yi)))
                    #f)))))))""",
    "montgomery-reduce": """(define (montgomery-reduce T m R m-prime)
  (let* ([t (modulo (* T m-prime) R)]
         [u (quotient (+ T (* t m)) R)])
    (if (>= u m)
        (- u m)
        u)))""",
    "montgomery-setup": """(define (montgomery-setup m)
  (let* ([R (let loop ([r 1])
              (if (> r m)
                  r
                  (loop (* r 2))))]
         [result (extended-gcd m R)]
         [m-inv (cadr result)]
         [m-prime (modulo (- m-inv) R)])
    (list R m-prime)))""",
    "montgomery-mult": """(define (montgomery-mult a b m R m-prime)
  (montgomery-reduce (* a b) m R m-prime))""",
    "to-montgomery": """(define (to-montgomery a m R)
  (modulo (* a R) m))""",
    "from-montgomery": """(define (from-montgomery a m R m-prime)
  (montgomery-reduce a m R m-prime))""",
    "quadratic-residue?": """(define (quadratic-residue? a p)
  (let ([a-mod (modulo a p)])
    (or (= a-mod 0)
        (= 1 (mod-expt a-mod (quotient (- p 1) 2) p)))))""",
    "legendre-symbol": """(define (legendre-symbol a p)
  (let ([a-mod (modulo a p)])
    (cond
      [(= a-mod 0) 0]
      [(= 1 (mod-expt a-mod (quotient (- p 1) 2) p)) 1]
      [else -1])))""",
    "mod-sqrt-both": """(define (mod-sqrt-both a p)
  (let ([r (mod-sqrt a p)])
    (if r
        (let ([r2 (modulo (- p r) p)])
          (if (= r r2)
              (list r)
              (if (<= r r2)
                  (list r r2)
                  (list r2 r))))
        #f)))""",
}

FUNCTION_ORDER = [
    "mod+",
    "mod-",
    "mod*",
    "mod-expt",
    "gcd",
    "extended-gcd",
    "mod-inverse",
    "crt",
    "montgomery-reduce",
    "montgomery-setup",
    "montgomery-mult",
    "to-montgomery",
    "from-montgomery",
    "quadratic-residue?",
    "legendre-symbol",
    "mod-sqrt-both",
]

FUNCTION_SPECS = {
    "mod+": "Return (a + b) mod m for m > 0.",
    "mod-": "Return (a - b) mod m for m > 0.",
    "mod*": "Return (a * b) mod m for m > 0.",
    "mod-expt": "Implement square-and-multiply modular exponentiation for exp >= 0.",
    "gcd": "Implement Euclidean gcd and normalize the final result to non-negative.",
    "extended-gcd": "Return (g x y) where g = gcd(a,b) and g = ax + by using iterative state updates.",
    "mod-inverse": "Return modular inverse of a mod m when gcd(a,m)=1, else #f.",
    "crt": "Solve a list of congruences; return #f when any inverse is missing.",
    "montgomery-reduce": "Implement Montgomery reduction with the final conditional subtraction.",
    "montgomery-setup": "Compute (R, m-prime) where R is the smallest power-of-two strictly greater than m.",
    "montgomery-mult": "Multiply two Montgomery-form values via montgomery-reduce.",
    "to-montgomery": "Map a to Montgomery representation aR mod m.",
    "from-montgomery": "Map a Montgomery value back via montgomery-reduce.",
    "quadratic-residue?": "Use Euler's criterion and treat 0 as a quadratic residue.",
    "legendre-symbol": "Return 1, -1, or 0 for odd prime p.",
    "mod-sqrt-both": "Return both square roots from mod-sqrt as an ordered list (smaller first), deduplicating when roots coincide.",
}

SKELETONS = {
    "mod+": """(define (mod+ a b m)
  ;; TODO: return (a + b) mod m
  <TODO>)""",
    "mod-": """(define (mod- a b m)
  ;; TODO: return (a - b) mod m
  <TODO>)""",
    "mod*": """(define (mod* a b m)
  ;; TODO: return (a * b) mod m
  <TODO>)""",
    "mod-expt": """(define (mod-expt base exp m)
  ;; TODO: implement square-and-multiply with a named let
  <TODO>)""",
    "gcd": """(define (gcd a b)
  ;; TODO: Euclidean algorithm with non-negative output
  <TODO>)""",
    "extended-gcd": """(define (extended-gcd a b)
  ;; TODO: iterative EEA returning (g x y)
  <TODO>)""",
    "mod-inverse": """(define (mod-inverse a m)
  ;; TODO: use extended-gcd and return #f when no inverse exists
  <TODO>)""",
    "crt": """(define (crt remainders moduli)
  ;; TODO: combine congruences with modular inverses
  <TODO>)""",
    "montgomery-reduce": """(define (montgomery-reduce T m R m-prime)
  ;; TODO: compute t then u, then conditionally subtract m
  <TODO>)""",
    "montgomery-setup": """(define (montgomery-setup m)
  ;; TODO: pick R as smallest power-of-two > m, compute m-prime
  <TODO>)""",
    "montgomery-mult": """(define (montgomery-mult a b m R m-prime)
  ;; TODO: multiply and reduce in Montgomery domain
  <TODO>)""",
    "to-montgomery": """(define (to-montgomery a m R)
  ;; TODO: convert to Montgomery form
  <TODO>)""",
    "from-montgomery": """(define (from-montgomery a m R m-prime)
  ;; TODO: convert out of Montgomery form
  <TODO>)""",
    "quadratic-residue?": """(define (quadratic-residue? a p)
  ;; TODO: Euler criterion with explicit 0 case
  <TODO>)""",
    "legendre-symbol": """(define (legendre-symbol a p)
  ;; TODO: return 1, -1, or 0
  <TODO>)""",
    "mod-sqrt-both": """(define (mod-sqrt-both a p)
  ;; TODO: call mod-sqrt and return one or two roots
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "mod+": "(and (= (mod+ 7 5 12) 0) (= (mod+ -3 5 7) 2) (= (mod+ -8 3 7) 2) (= (mod+ 10 15 13) 12))",
    "mod-": "(and (= (mod- 5 3 7) 2) (= (mod- 3 5 7) 5) (= (mod- 10 15 13) 8))",
    "mod*": "(and (= (mod* 3 4 7) 5) (= (mod* 5 6 13) 4) (= (mod* 123 456 1000) 88))",
    "mod-expt": "(and (= (mod-expt 2 10 1000) 24) (= (mod-expt 3 20 100) 1) (= (mod-expt 7 0 1) 0))",
    "gcd": "(and (= (gcd 12 8) 4) (= (gcd 7 0) 7) (= (gcd -12 8) 4) (= (gcd 12 -8) 4))",
    "extended-gcd": "(let* ([r (extended-gcd 240 46)] [g (car r)] [x (cadr r)] [y (caddr r)]) (and (= g 2) (= g (+ (* 240 x) (* 46 y)))))",
    "mod-inverse": "(and (= (mod-inverse 3 7) 5) (= (mod-inverse 5 13) 8) (not (mod-inverse 10 15)))",
    "crt": "(let ([x (crt '(2 3 2) '(3 5 7))] [bad (crt '(1 1) '(2 4))]) (and x (= (modulo x 3) 2) (= (modulo x 5) 3) (= (modulo x 7) 2) (not bad)))",
    "montgomery-reduce": "(let* ([setup (montgomery-setup 17)] [R (car setup)] [m-prime (cadr setup)]) (= (montgomery-reduce 17 17 R m-prime) 0))",
    "montgomery-setup": "(let* ([setup (montgomery-setup 1)] [R (car setup)] [m-prime (cadr setup)]) (and (> R 1) (= (bitwise-and R (- R 1)) 0) (= (modulo (+ (* 1 m-prime) 1) R) 0)))",
    "montgomery-mult": "(let* ([m 17] [setup (montgomery-setup m)] [R (car setup)] [m-prime (cadr setup)] [aM (to-montgomery 3 m R)] [bM (to-montgomery 5 m R)] [prodM (montgomery-mult aM bM m R m-prime)] [prod (from-montgomery prodM m R m-prime)]) (= prod 15))",
    "to-montgomery": "(let* ([m 17] [setup (montgomery-setup m)] [R (car setup)] [m-prime (cadr setup)] [x (to-montgomery 9 m R)]) (= (from-montgomery x m R m-prime) 9))",
    "from-montgomery": "(let* ([m 17] [setup (montgomery-setup m)] [R (car setup)] [m-prime (cadr setup)] [x (to-montgomery 11 m R)]) (= (from-montgomery x m R m-prime) 11))",
    "quadratic-residue?": "(and (quadratic-residue? 4 7) (not (quadratic-residue? 3 7)) (quadratic-residue? 0 7))",
    "legendre-symbol": "(and (= (legendre-symbol 1 7) 1) (= (legendre-symbol 3 7) -1) (= (legendre-symbol 0 7) 0))",
    "mod-sqrt-both": "(let ([roots (mod-sqrt-both 4 13)] [roots0 (mod-sqrt-both 0 13)]) (and roots (= (length roots) 2) (<= (car roots) (cadr roots)) (= (+ (car roots) (cadr roots)) 13) (= (modulo (* (car roots) (car roots)) 13) 4) (= (modulo (* (cadr roots) (cadr roots)) 13) 4) (equal? roots0 '(0))))",
}

DEPENDS: Dict[str, List[str]] = {
    "mod+": [],
    "mod-": [],
    "mod*": [],
    "mod-expt": [],
    "gcd": [],
    "extended-gcd": [],
    "mod-inverse": ["extended-gcd"],
    "crt": ["mod-inverse", "extended-gcd"],
    "montgomery-reduce": [],
    "montgomery-setup": ["extended-gcd"],
    "montgomery-mult": ["montgomery-reduce"],
    "to-montgomery": [],
    "from-montgomery": ["montgomery-reduce"],
    "quadratic-residue?": ["mod-expt"],
    "legendre-symbol": ["mod-expt"],
    "mod-sqrt-both": [],
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")

TRANSLATION_FUNCTIONS = [
    "mod+",
    "mod-",
    "mod*",
    "mod-expt",
    "gcd",
    "extended-gcd",
    "mod-inverse",
    "crt",
    "montgomery-setup",
    "montgomery-mult",
    "quadratic-residue?",
    "legendre-symbol",
]

PYTHON_SNIPPETS = {
    "mod+": "def mod_add(a, b, m):\n    return (a + b) % m",
    "mod-": "def mod_sub(a, b, m):\n    return (a - b) % m",
    "mod*": "def mod_mul(a, b, m):\n    return (a * b) % m",
    "mod-expt": "def mod_pow(base, exp, m):\n    b = base % m\n    result = 1\n    while exp > 0:\n        if exp & 1:\n            result = (result * b) % m\n        b = (b * b) % m\n        exp //= 2\n    return result % m",
    "gcd": "def gcd(a, b):\n    while b != 0:\n        a, b = b, a % b\n    return abs(a)",
    "extended-gcd": "def egcd(a, b):\n    old_r, r = a, b\n    old_s, s = 1, 0\n    old_t, t = 0, 1\n    while r != 0:\n        q = old_r // r\n        old_r, r = r, old_r - q * r\n        old_s, s = s, old_s - q * s\n        old_t, t = t, old_t - q * t\n    return old_r, old_s, old_t",
    "mod-inverse": "def mod_inverse(a, m):\n    g, x, _ = egcd(a, m)\n    return x % m if g == 1 else None",
    "crt": "def crt(remainders, moduli):\n    if not remainders or not moduli:\n        return 0\n    M = 1\n    for m in moduli:\n        M *= m\n    total = 0\n    for ai, mi in zip(remainders, moduli):\n        Mi = M // mi\n        yi = mod_inverse(Mi, mi)\n        if yi is None:\n            return None\n        total += ai * Mi * yi\n    return total % M",
    "montgomery-setup": "def mont_setup(m):\n    R = 1\n    while R <= m:\n        R <<= 1\n    _, m_inv, _ = egcd(m, R)\n    m_prime = (-m_inv) % R\n    return R, m_prime",
    "montgomery-mult": "def mont_mult(a, b, m, R, m_prime):\n    T = a * b\n    t = (T * m_prime) % R\n    u = (T + t * m) // R\n    return u - m if u >= m else u",
    "quadratic-residue?": "def is_qr(a, p):\n    a_mod = a % p\n    return a_mod == 0 or pow(a_mod, (p - 1) // 2, p) == 1",
    "legendre-symbol": "def legendre(a, p):\n    a_mod = a % p\n    if a_mod == 0:\n        return 0\n    return 1 if pow(a_mod, (p - 1) // 2, p) == 1 else -1",
}

CHEZ_SNIPPETS = {
    "mod+": "(define (mod-add a b m)\n  (remainder (+ a b) m))",
    "mod-": "(define (mod-sub a b m)\n  (remainder (- a b) m))",
    "mod*": "(define (mod-mul a b m)\n  (remainder (* a b) m))",
    "mod-expt": "(define (mod-pow base exp m)\n  (let loop ((b (modulo base m))\n             (e exp)\n             (acc 1))\n    (if (= e 0)\n        acc\n        (if (odd? e)\n            (loop (modulo (* b b) m) (quotient e 2) (modulo (* acc b) m))\n            (loop (modulo (* b b) m) (quotient e 2) acc)))))",
    "gcd": "(define (gcd0 a b)\n  (if (= b 0) a (gcd0 b (modulo a b))))",
    "extended-gcd": "(define (egcd a b)\n  (let loop ((old-r a) (r b)\n             (old-s 1) (s 0)\n             (old-t 0) (t 1))\n    (if (= r 0)\n        (list old-r old-s old-t)\n        (let* ((q (quotient old-r r))\n               (nr (- old-r (* q r)))\n               (ns (- old-s (* q s)))\n               (nt (- old-t (* q t))))\n          (loop r nr s ns t nt)))))",
    "mod-inverse": "(define (inv a m)\n  (let* ((triplet (extended-gcd a m))\n         (g (car triplet))\n         (x (cadr triplet)))\n    (if (= g 1) x #f)))",
    "crt": "(define (crt0 rs ms)\n  (let ((M (apply * ms)))\n    (let loop ((rs rs) (ms ms) (acc 0))\n      (if (null? rs)\n          (modulo acc M)\n          (let* ((ai (car rs))\n                 (mi (car ms))\n                 (Mi (quotient M mi))\n                 (yi (mod-inverse Mi mi)))\n            (loop (cdr rs) (cdr ms) (+ acc (* ai Mi yi))))))))",
    "montgomery-setup": "(define (mont-setup m)\n  (let loop ((r 1))\n    (if (> r m)\n        (let* ((eg (extended-gcd m r))\n               (m-inv (cadr eg)))\n          (list r (modulo (- m-inv) r)))\n        (loop (* r 2)))))",
    "montgomery-mult": "(define (mont-mult a b m R m-prime)\n  (montgomery-reduce (* a b) m R m-prime))",
    "quadratic-residue?": "(define (qr? a p)\n  (= 1 (mod-expt (modulo a p) (quotient (- p 1) 2) p)))",
    "legendre-symbol": "(define (legendre0 a p)\n  (cond ((= (modulo a p) 0) 0)\n        ((= 1 (mod-expt (modulo a p) (quotient (- p 1) 2) p)) 1)\n        (else -1)))",
}

BUGGY_CASES = [
    {
        "fn": "mod+",
        "buggy": "(define (mod+ a b m)\n  (remainder (+ a b) m))",
        "note": "Negative inputs should still normalize to [0, m).",
    },
    {
        "fn": "mod+",
        "buggy": "(define (mod+ a b m)\n  (+ a b))",
        "note": "The modulus is ignored.",
    },
    {
        "fn": "mod-",
        "buggy": "(define (mod- a b m)\n  (modulo (- b a) m))",
        "note": "Subtracting in the wrong order flips results.",
    },
    {
        "fn": "mod-",
        "buggy": "(define (mod- a b m)\n  (- a b))",
        "note": "Result is not reduced modulo m.",
    },
    {
        "fn": "mod*",
        "buggy": "(define (mod* a b m)\n  (* a b))",
        "note": "Must reduce the product modulo m.",
    },
    {
        "fn": "mod*",
        "buggy": "(define (mod* a b m)\n  (modulo (+ a b) m))",
        "note": "Addition is used instead of multiplication.",
    },
    {
        "fn": "mod-expt",
        "buggy": "(define (mod-expt base exp m)\n  (let loop ([b (modulo base m)] [e exp] [result 1])\n    (cond\n      [(= e 0) result]\n      [(odd? e) (loop (modulo (* b b) m) (quotient e 2) (modulo (* result b) m))]\n      [else (loop (modulo (* b b) m) (quotient e 2) result)])))",
        "note": "Base case must normalize for m=1.",
    },
    {
        "fn": "mod-expt",
        "buggy": "(define (mod-expt base exp m)\n  (let loop ([b (modulo base m)] [e exp] [result 1])\n    (cond\n      [(= e 0) (modulo result m)]\n      [(odd? e) (loop (modulo (* b b) m) (quotient e 2) (modulo (* result result) m))]\n      [else (loop (modulo (* b b) m) (quotient e 2) result)])))",
        "note": "Odd-step accumulator multiplies by itself instead of b.",
    },
    {
        "fn": "gcd",
        "buggy": "(define (gcd a b)\n  (if (= b 0)\n      a\n      (gcd b (modulo a b))))",
        "note": "Result sign should be normalized.",
    },
    {
        "fn": "gcd",
        "buggy": "(define (gcd a b)\n  (if (= b 0)\n      (abs a)\n      (gcd b (quotient a b))))",
        "note": "Euclidean step should use modulo, not quotient.",
    },
    {
        "fn": "extended-gcd",
        "buggy": "(define (extended-gcd a b)\n  (let loop ([old-r a] [r b] [old-s 1] [s 0] [old-t 0] [t 1])\n    (if (= r 0)\n        (list old-r old-s old-t)\n        (let* ([q (quotient old-r r)]\n               [new-r (- old-r (* q r))]\n               [new-s (- old-s (* q old-s))]\n               [new-t (- old-t (* q t))])\n          (loop r new-r s new-s t new-t)))))",
        "note": "new-s uses old-s twice; it must use current s.",
    },
    {
        "fn": "mod-inverse",
        "buggy": "(define (mod-inverse a m)\n  (let* ([result (extended-gcd a m)] [g (car result)] [x (cadr result)])\n    (if (= g 1) x #f)))",
        "note": "Inverse should be reduced modulo m.",
    },
    {
        "fn": "mod-inverse",
        "buggy": "(define (mod-inverse a m)\n  (let* ([result (extended-gcd a m)] [g (car result)] [x (cadr result)])\n    (if (= g -1) (modulo x m) #f)))",
        "note": "The invertibility check is against the wrong gcd value.",
    },
    {
        "fn": "crt",
        "buggy": "(define (crt remainders moduli)\n  (if (or (null? remainders) (null? moduli))\n      0\n      (let ([M (fold-left * 1 moduli)])\n        (let loop ([rs remainders] [ms moduli] [result 0])\n          (if (null? rs)\n              (modulo result M)\n              (let* ([ai (car rs)] [mi (car ms)] [Mi (quotient M mi)] [yi (mod-inverse Mi mi)])\n                (loop (cdr rs) (cdr ms) (+ result (* ai Mi yi)))))))))",
        "note": "When yi is #f, solver should fail instead of multiplying by #f.",
    },
    {
        "fn": "crt",
        "buggy": "(define (crt remainders moduli)\n  (if (or (null? remainders) (null? moduli))\n      0\n      (let ([M (fold-left + 0 moduli)])\n        (let loop ([rs remainders] [ms moduli] [result 0])\n          (if (null? rs)\n              (modulo result M)\n              (let* ([ai (car rs)] [mi (car ms)] [Mi (quotient M mi)] [yi (mod-inverse Mi mi)])\n                (if yi\n                    (loop (cdr rs) (cdr ms) (+ result (* ai Mi yi)))\n                    #f)))))))",
        "note": "M must be the product of moduli, not the sum.",
    },
    {
        "fn": "montgomery-reduce",
        "buggy": "(define (montgomery-reduce T m R m-prime)\n  (let* ([t (modulo (* T m-prime) R)]\n         [u (quotient (+ T (* t m)) R)])\n    u))",
        "note": "Need final conditional subtraction when u >= m.",
    },
    {
        "fn": "montgomery-setup",
        "buggy": "(define (montgomery-setup m)\n  (let* ([R (let loop ([r 1])\n              (if (>= r m) r (loop (* r 2))))]\n         [result (extended-gcd m R)]\n         [m-inv (cadr result)]\n         [m-prime (modulo (- m-inv) R)])\n    (list R m-prime)))",
        "note": "R must be strictly greater than m.",
    },
    {
        "fn": "montgomery-mult",
        "buggy": "(define (montgomery-mult a b m R m-prime)\n  (modulo (* a b) m))",
        "note": "Montgomery multiplication should call montgomery-reduce.",
    },
    {
        "fn": "to-montgomery",
        "buggy": "(define (to-montgomery a m R)\n  (modulo (* a m) m))",
        "note": "Conversion multiplies by R, not by m.",
    },
    {
        "fn": "from-montgomery",
        "buggy": "(define (from-montgomery a m R m-prime)\n  (modulo a m))",
        "note": "Conversion from Montgomery domain requires montgomery-reduce.",
    },
    {
        "fn": "quadratic-residue?",
        "buggy": "(define (quadratic-residue? a p)\n  (= 1 (mod-expt (modulo a p) (quotient (- p 1) 2) p)))",
        "note": "a == 0 mod p is also a quadratic residue.",
    },
    {
        "fn": "legendre-symbol",
        "buggy": "(define (legendre-symbol a p)\n  (let ([a-mod (modulo a p)])\n    (if (= a-mod 0)\n        0\n        (quadratic-residue? a-mod p))))",
        "note": "Return value must be 1 or -1, not booleans.",
    },
    {
        "fn": "mod-sqrt-both",
        "buggy": "(define (mod-sqrt-both a p)\n  (let ([r (mod-sqrt a p)])\n    (if r\n        (list r (- p r))\n        #f)))",
        "note": "Roots must be reduced modulo p, ordered smaller-first, and deduplicated.",
    },
    {
        "fn": "mod-sqrt-both",
        "buggy": "(define (mod-sqrt-both a p)\n  (let ([r (mod-sqrt a p)])\n    (if r\n        (list r r)\n        #f)))",
        "note": "Second root should be p-r (mod p) unless both roots coincide.",
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
    idx = family_counter[family]
    sid = f"nt_modular_{family}_{idx:03d}"
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
    return [name for name in FUNCTION_ORDER if name != fn and name in tokens]


def dependency_closure(fn: str) -> List[str]:
    ordered: List[str] = []
    seen: Set[str] = set()

    def visit(name: str) -> None:
        for dep in DEPENDS.get(name, []):
            if dep not in seen:
                seen.add(dep)
                visit(dep)
                ordered.append(dep)

    for dep in DEPENDS.get(fn, []):
        if dep not in seen:
            seen.add(dep)
            visit(dep)
            ordered.append(dep)

    for dep in verify_refs(fn):
        if dep not in seen:
            seen.add(dep)
            visit(dep)
            ordered.append(dep)

    return ordered


def def_verify(function_name: str) -> str:
    parts = [DEFS[dep] for dep in dependency_closure(function_name)] + [DEFS[function_name], VERIFY_BY_FUNCTION[function_name]]
    body = "\n  ".join(parts)
    return f"(let ()\n  {body})"


# -----------------------------------------------------------------------------
# Family 1: spec_to_code (32 samples)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty="medium" if fn in {"mod+", "mod-", "mod*", "gcd", "to-montgomery", "from-montgomery"} else "hard",
        source_function=fn,
        prompt=f"""You are implementing Tier-0 number-theory code in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "number-theory", "modular", "spec-to-code", fn],
    )

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty="medium" if fn in {"mod+", "mod-", "mod*", "gcd", "to-montgomery", "from-montgomery"} else "hard",
        source_function=fn,
        prompt=f"""Complete this Fold Scheme skeleton so it satisfies the modular arithmetic test suite.

```scheme
{SKELETONS[fn]}
```

Replace `<TODO>` and return only the completed definition for `{fn}`.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "number-theory", "modular", "skeleton-completion", fn],
    )

# -----------------------------------------------------------------------------
# Family 2: translation (24 samples)
# -----------------------------------------------------------------------------
for fn in TRANSLATION_FUNCTIONS:
    add_sample(
        family="translation",
        category="translation",
        difficulty="medium" if fn in {"mod+", "mod-", "mod*", "gcd"} else "hard",
        source_function=fn,
        prompt=f"""Translate the following Python function into Fold-native Scheme.
Preserve behavior exactly and keep the target function name `{fn}`.
Return only the Scheme definition.

```python
{PYTHON_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "number-theory", "modular", "python-to-scheme", fn],
    )

    add_sample(
        family="translation",
        category="translation",
        difficulty="medium" if fn in {"mod+", "mod-", "mod*", "gcd"} else "hard",
        source_function=fn,
        prompt=f"""Convert this Chez-style snippet into canonical Fold style for `{SOURCE_MODULE}`.
Target function name must be `{fn}`.
Return only the corrected Fold definition.

```scheme
{CHEZ_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "number-theory", "modular", "chez-to-fold", fn],
    )

# -----------------------------------------------------------------------------
# Family 3: bugfix (24 samples)
# -----------------------------------------------------------------------------
for case in BUGGY_CASES:
    fn = case["fn"]
    add_sample(
        family="bugfix",
        category="debugging",
        difficulty="medium" if fn in {"mod+", "mod-", "mod*", "gcd", "to-montgomery", "from-montgomery"} else "hard",
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
        tags=["tier0", "number-theory", "modular", "bugfix", fn],
    )

# -----------------------------------------------------------------------------
# Family 4: composition/use (60 samples)
# -----------------------------------------------------------------------------


def add_composition(
    source_function: str,
    prompt: str,
    ground_truth: str,
    verify_expr: str,
    difficulty: str = "easy",
    tags: List[str] | None = None,
) -> None:
    tag_list = ["tier0", "number-theory", "modular", "composition", source_function]
    if tags:
        tag_list.extend(tags)
    add_sample(
        family="composition",
        category="usage",
        difficulty=difficulty,
        source_function=source_function,
        prompt=prompt,
        ground_truth=ground_truth,
        verify_expr=verify_expr,
        tags=tag_list,
    )


composition_cases = [
    # --- Direct computation (16) ---
    ("mod+", "Compute (7 + 5) mod 12 using `mod+`. Return a single Scheme expression.", "(mod+ 7 5 12)", "(equal? (let () (mod+ 7 5 12)) (modulo (+ 7 5) 12))", "easy", ["direct"]),
    ("mod+", "Compute (-22 + 8) mod 17 using `mod+`.", "(mod+ -22 8 17)", "(equal? (let () (mod+ -22 8 17)) (modulo (+ -22 8) 17))", "easy", ["direct"]),
    ("mod+", "Compute (123 + 456) mod 97 using `mod+`.", "(mod+ 123 456 97)", "(equal? (let () (mod+ 123 456 97)) (modulo (+ 123 456) 97))", "easy", ["direct"]),
    ("mod+", "Compute (0 + 9) mod 11 using `mod+`.", "(mod+ 0 9 11)", "(equal? (let () (mod+ 0 9 11)) (modulo (+ 0 9) 11))", "easy", ["direct"]),
    ("mod-", "Compute (3 - 5) mod 7 using `mod-`.", "(mod- 3 5 7)", "(equal? (let () (mod- 3 5 7)) (modulo (- 3 5) 7))", "easy", ["direct"]),
    ("mod-", "Compute (42 - 58) mod 100 using `mod-`.", "(mod- 42 58 100)", "(equal? (let () (mod- 42 58 100)) (modulo (- 42 58) 100))", "easy", ["direct"]),
    ("mod-", "Compute (999 - 1) mod 10 using `mod-`.", "(mod- 999 1 10)", "(equal? (let () (mod- 999 1 10)) (modulo (- 999 1) 10))", "easy", ["direct"]),
    ("mod-", "Compute (0 - 9) mod 11 using `mod-`.", "(mod- 0 9 11)", "(equal? (let () (mod- 0 9 11)) (modulo (- 0 9) 11))", "easy", ["direct"]),
    ("mod*", "Compute (3 * 4) mod 7 with `mod*`.", "(mod* 3 4 7)", "(equal? (let () (mod* 3 4 7)) (modulo (* 3 4) 7))", "easy", ["direct"]),
    ("mod*", "Compute (-22 * 8) mod 17 with `mod*`.", "(mod* -22 8 17)", "(equal? (let () (mod* -22 8 17)) (modulo (* -22 8) 17))", "easy", ["direct"]),
    ("mod*", "Compute (123 * 456) mod 1000 with `mod*`.", "(mod* 123 456 1000)", "(equal? (let () (mod* 123 456 1000)) (modulo (* 123 456) 1000))", "easy", ["direct"]),
    ("mod*", "Compute (77 * 91) mod 97 with `mod*`.", "(mod* 77 91 97)", "(equal? (let () (mod* 77 91 97)) (modulo (* 77 91) 97))", "easy", ["direct"]),
    ("mod-expt", "Compute (2^100) mod 1000000007 using `mod-expt`.", "(mod-expt 2 100 1000000007)", "(equal? (let () (mod-expt 2 100 1000000007)) (modulo (expt 2 100) 1000000007))", "medium", ["direct"]),
    ("mod-expt", "Compute (7^0) mod 1 using `mod-expt`.", "(mod-expt 7 0 1)", "(equal? (let () (mod-expt 7 0 1)) (modulo (expt 7 0) 1))", "medium", ["direct", "edge-case"]),
    ("mod-expt", "Compute (5^3) mod 13 using `mod-expt`.", "(mod-expt 5 3 13)", "(equal? (let () (mod-expt 5 3 13)) (modulo (expt 5 3) 13))", "medium", ["direct"]),
    ("mod-expt", "Compute (123^45) mod 89 using `mod-expt`.", "(mod-expt 123 45 89)", "(equal? (let () (mod-expt 123 45 89)) (modulo (expt 123 45) 89))", "medium", ["direct"]),

    # --- Algebraic properties (10) ---
    ("mod+", "Return a boolean that checks commutativity: mod+(81,23,29) == mod+(23,81,29).", "(= (mod+ 81 23 29) (mod+ 23 81 29))", "(equal? (let () (= (mod+ 81 23 29) (mod+ 23 81 29))) #t)", "medium", ["property"]),
    ("mod+", "Check associativity for modular addition with a=5, b=9, c=14, m=17.", "(= (mod+ (mod+ 5 9 17) 14 17) (mod+ 5 (mod+ 9 14 17) 17))", "(equal? (let () (= (mod+ (mod+ 5 9 17) 14 17) (mod+ 5 (mod+ 9 14 17) 17))) #t)", "medium", ["property"]),
    ("mod-", "Check that modular subtraction equals adding a negated value for a=15,b=42,m=23.", "(= (mod- 15 42 23) (mod+ 15 (- 42) 23))", "(equal? (let () (= (mod- 15 42 23) (mod+ 15 (- 42) 23))) #t)", "medium", ["property"]),
    ("mod-", "Check the identity mod-(a,a,m)=0 for a=37,m=19.", "(= (mod- 37 37 19) 0)", "(equal? (let () (= (mod- 37 37 19) 0)) #t)", "medium", ["property"]),
    ("mod*", "Return a boolean that checks commutativity for mod* with a=12,b=31,m=37.", "(= (mod* 12 31 37) (mod* 31 12 37))", "(equal? (let () (= (mod* 12 31 37) (mod* 31 12 37))) #t)", "medium", ["property"]),
    ("mod*", "Check distributivity: a*(b+c) mod m equals a*b + a*c mod m for a=7,b=9,c=13,m=23.", "(= (mod* 7 (mod+ 9 13 23) 23) (mod+ (mod* 7 9 23) (mod* 7 13 23) 23))", "(equal? (let () (= (mod* 7 (mod+ 9 13 23) 23) (mod+ (mod* 7 9 23) (mod* 7 13 23) 23))) #t)", "medium", ["property"]),
    ("mod*", "Check multiplicative identity for a=-44,m=29.", "(= (mod* -44 1 29) (modulo -44 29))", "(equal? (let () (= (mod* -44 1 29) (modulo -44 29))) #t)", "medium", ["property"]),
    ("mod-expt", "Check exponent base case: a^0 mod m = 1 for a=9,m=13.", "(= (mod-expt 9 0 13) 1)", "(equal? (let () (= (mod-expt 9 0 13) 1)) #t)", "medium", ["property"]),
    ("mod-expt", "Check exponent one case: a^1 mod m = a mod m for a=27,m=19.", "(= (mod-expt 27 1 19) (modulo 27 19))", "(equal? (let () (= (mod-expt 27 1 19) (modulo 27 19))) #t)", "medium", ["property"]),
    ("mod-expt", "Check additive exponent law for a=7,e1=5,e2=8,m=19.", "(= (mod-expt 7 (+ 5 8) 19) (mod* (mod-expt 7 5 19) (mod-expt 7 8 19) 19))", "(equal? (let () (= (mod-expt 7 (+ 5 8) 19) (mod* (mod-expt 7 5 19) (mod-expt 7 8 19) 19))) #t)", "hard", ["property"]),

    # --- List/map/fold style usage (10) ---
    ("mod+", "Use map with `mod+` on pair list ((3 5) (10 9) (14 -2)) modulo 13.", "(map (lambda (p) (mod+ (car p) (cadr p) 13)) '((3 5) (10 9) (14 -2)))", "(equal? (let () (map (lambda (p) (mod+ (car p) (cadr p) 13)) '((3 5) (10 9) (14 -2)))) '(8 6 12))", "medium", ["list"]),
    ("mod-", "Use map with `mod-` on pair list ((2 9) (15 4) (-3 8)) modulo 13.", "(map (lambda (p) (mod- (car p) (cadr p) 13)) '((2 9) (15 4) (-3 8)))", "(equal? (let () (map (lambda (p) (mod- (car p) (cadr p) 13)) '((2 9) (15 4) (-3 8)))) '(6 11 2))", "medium", ["list"]),
    ("mod*", "Map `mod*` with scalar 7 modulo 17 over '(1 2 3 4 5).", "(map (lambda (x) (mod* x 7 17)) '(1 2 3 4 5))", "(equal? (let () (map (lambda (x) (mod* x 7 17)) '(1 2 3 4 5))) '(7 14 4 11 1))", "medium", ["list"]),
    ("mod-expt", "Build a powers table [3^e mod 17] for exponents 0..4.", "(map (lambda (e) (mod-expt 3 e 17)) '(0 1 2 3 4))", "(equal? (let () (map (lambda (e) (mod-expt 3 e 17)) '(0 1 2 3 4))) '(1 3 9 10 13))", "medium", ["list"]),
    ("mod+", "Compute modular sum of '(12 19 7 5) modulo 13 using a named-let accumulator.", "(let loop ([xs '(12 19 7 5)] [acc 0]) (if (null? xs) acc (loop (cdr xs) (mod+ acc (car xs) 13))))", "(equal? (let () (let loop ([xs '(12 19 7 5)] [acc 0]) (if (null? xs) acc (loop (cdr xs) (mod+ acc (car xs) 13))))) 4)", "medium", ["list", "fold"]),
    ("mod*", "Compute modular product of '(3 4 5 6) modulo 17 using a named-let accumulator.", "(let loop ([xs '(3 4 5 6)] [acc 1]) (if (null? xs) acc (loop (cdr xs) (mod* acc (car xs) 17))))", "(equal? (let () (let loop ([xs '(3 4 5 6)] [acc 1]) (if (null? xs) acc (loop (cdr xs) (mod* acc (car xs) 17))))) 3)", "medium", ["list", "fold"]),
    ("mod-expt", "Evaluate polynomial x^3 + 2x + 5 at x=4 over Z_17 using modular operators only.", "(let ([x 4]) (mod+ (mod+ (mod-expt x 3 17) (mod* 2 x 17) 17) 5 17))", "(equal? (let () (let ([x 4]) (mod+ (mod+ (mod-expt x 3 17) (mod* 2 x 17) 17) 5 17))) 9)", "hard", ["list", "composition"]),
    ("mod-inverse", "Map `mod-inverse` over all non-zero residues modulo 11.", "(map (lambda (a) (mod-inverse a 11)) '(1 2 3 4 5 6 7 8 9 10))", "(equal? (let () (map (lambda (a) (mod-inverse a 11)) '(1 2 3 4 5 6 7 8 9 10))) '(1 6 4 3 9 2 8 7 5 10))", "medium", ["list"]),
    ("mod-inverse", "Return #t iff every non-zero residue modulo 11 satisfies a*inv(a)=1.", "(let loop ([as '(1 2 3 4 5 6 7 8 9 10)]) (if (null? as) #t (let* ([a (car as)] [inv (mod-inverse a 11)]) (and inv (= (mod* a inv 11) 1) (loop (cdr as))))))", "(equal? (let () (let loop ([as '(1 2 3 4 5 6 7 8 9 10)]) (if (null? as) #t (let* ([a (car as)] [inv (mod-inverse a 11)]) (and inv (= (mod* a inv 11) 1) (loop (cdr as))))))) #t)", "hard", ["list", "property"]),
    ("crt", "Solve CRT for (2,3,2)/(3,5,7), then return a list confirming each congruence.", "(let ([x (crt '(2 3 2) '(3 5 7))]) (list (= (modulo x 3) 2) (= (modulo x 5) 3) (= (modulo x 7) 2)))", "(equal? (let () (let ([x (crt '(2 3 2) '(3 5 7))]) (list (= (modulo x 3) 2) (= (modulo x 5) 3) (= (modulo x 7) 2)))) '(#t #t #t))", "medium", ["list", "composition"]),

    # --- gcd / extended-gcd / mod-inverse / crt integration (12) ---
    ("gcd", "Evaluate gcd(240, 46) with `gcd`.", "(gcd 240 46)", "(equal? (let () (gcd 240 46)) 2)", "easy", ["integration"]),
    ("gcd", "Evaluate gcd(-99, 78) with `gcd` and rely on normalized sign.", "(gcd -99 78)", "(equal? (let () (gcd -99 78)) 3)", "easy", ["integration"]),
    ("gcd", "Compute lcm(12,18) from gcd using lcm(a,b)=|ab|/gcd(a,b).", "(let ([a 12] [b 18]) (quotient (abs (* a b)) (gcd a b)))", "(equal? (let () (let ([a 12] [b 18]) (quotient (abs (* a b)) (gcd a b)))) 36)", "medium", ["integration"]),
    ("extended-gcd", "Return #t iff extended-gcd satisfies Bezout identity for (240,46).", "(let* ([r (extended-gcd 240 46)] [g (car r)] [x (cadr r)] [y (caddr r)]) (= g (+ (* 240 x) (* 46 y))))", "(equal? (let () (let* ([r (extended-gcd 240 46)] [g (car r)] [x (cadr r)] [y (caddr r)]) (= g (+ (* 240 x) (* 46 y))))) #t)", "medium", ["integration"]),
    ("extended-gcd", "Extract gcd component from extended-gcd(17,13).", "(car (extended-gcd 17 13))", "(equal? (let () (car (extended-gcd 17 13))) 1)", "easy", ["integration"]),
    ("extended-gcd", "Extract gcd component from extended-gcd(391,299).", "(car (extended-gcd 391 299))", "(equal? (let () (car (extended-gcd 391 299))) 23)", "medium", ["integration"]),
    ("mod-inverse", "Compute inverse of 3 mod 7 with `mod-inverse`.", "(mod-inverse 3 7)", "(equal? (let () (mod-inverse 3 7)) 5)", "easy", ["integration"]),
    ("mod-inverse", "Compute inverse of 10 mod 15 with `mod-inverse` (expect #f).", "(mod-inverse 10 15)", "(equal? (let () (mod-inverse 10 15)) #f)", "easy", ["integration"]),
    ("mod-inverse", "Check multiplicative inverse property for a=5, m=13 in one expression.", "(let ([inv (mod-inverse 5 13)]) (and inv (= (mod* 5 inv 13) 1)))", "(equal? (let () (let ([inv (mod-inverse 5 13)]) (and inv (= (mod* 5 inv 13) 1)))) #t)", "medium", ["integration"]),
    ("crt", "Solve x=2 (mod 3), x=3 (mod 5), x=2 (mod 7) via `crt`.", "(crt '(2 3 2) '(3 5 7))", "(equal? (let () (crt '(2 3 2) '(3 5 7))) 23)", "easy", ["integration"]),
    ("crt", "Solve x=1 (mod 2), x=2 (mod 3), x=3 (mod 5) via `crt`.", "(crt '(1 2 3) '(2 3 5))", "(equal? (let () (crt '(1 2 3) '(2 3 5))) 23)", "easy", ["integration"]),
    ("crt", "Evaluate `crt` on empty congruence lists.", "(crt '() '())", "(equal? (let () (crt '() '())) 0)", "easy", ["integration", "edge-case"]),

    # --- Montgomery workflows (6) ---
    ("montgomery-setup", "Call `montgomery-setup` with modulus 17 and return R.", "(car (montgomery-setup 17))", "(let ([r (let () (car (montgomery-setup 17)))]) (and (> r 17) (= (bitwise-and r (- r 1)) 0)))", "medium", ["montgomery"]),
    ("to-montgomery", "Convert 5 into Montgomery form for modulus 17, then round-trip back in one expression.", "(let* ([setup (montgomery-setup 17)] [R (car setup)] [m-prime (cadr setup)] [x (to-montgomery 5 17 R)]) (from-montgomery x 17 R m-prime))", "(equal? (let () (let* ([setup (montgomery-setup 17)] [R (car setup)] [m-prime (cadr setup)] [x (to-montgomery 5 17 R)]) (from-montgomery x 17 R m-prime))) 5)", "medium", ["montgomery"]),
    ("montgomery-mult", "Multiply 3 and 5 modulo 17 using Montgomery-domain conversion and `montgomery-mult`.", "(let* ([m 17] [setup (montgomery-setup m)] [R (car setup)] [m-prime (cadr setup)] [aM (to-montgomery 3 m R)] [bM (to-montgomery 5 m R)] [prodM (montgomery-mult aM bM m R m-prime)]) (from-montgomery prodM m R m-prime))", "(equal? (let () (let* ([m 17] [setup (montgomery-setup m)] [R (car setup)] [m-prime (cadr setup)] [aM (to-montgomery 3 m R)] [bM (to-montgomery 5 m R)] [prodM (montgomery-mult aM bM m R m-prime)]) (from-montgomery prodM m R m-prime))) 15)", "hard", ["montgomery"]),
    ("montgomery-reduce", "Use `montgomery-reduce` on the edge case T=17, m=17 (with setup-derived R,m-prime).", "(let* ([setup (montgomery-setup 17)] [R (car setup)] [m-prime (cadr setup)]) (montgomery-reduce 17 17 R m-prime))", "(equal? (let () (let* ([setup (montgomery-setup 17)] [R (car setup)] [m-prime (cadr setup)]) (montgomery-reduce 17 17 R m-prime))) 0)", "hard", ["montgomery", "edge-case"]),
    ("mod-expt", "Return #t iff Montgomery exponentiation matches regular modular exponentiation for (7,13,19).", "(= (montgomery-expt 7 13 19) (mod-expt 7 13 19))", "(equal? (let () (= (montgomery-expt 7 13 19) (mod-expt 7 13 19))) #t)", "hard", ["montgomery"]),
    ("mod-expt", "Return #t iff Montgomery exponentiation matches regular modular exponentiation for (2,100,1000000007).", "(= (montgomery-expt 2 100 1000000007) (mod-expt 2 100 1000000007))", "(equal? (let () (= (montgomery-expt 2 100 1000000007) (mod-expt 2 100 1000000007))) #t)", "hard", ["montgomery"]),

    # --- Quadratic residues / Legendre / modular roots (6) ---
    ("quadratic-residue?", "Check whether 4 is a quadratic residue mod 7.", "(quadratic-residue? 4 7)", "(equal? (let () (quadratic-residue? 4 7)) #t)", "easy", ["quadratic"]),
    ("quadratic-residue?", "Check whether 3 is a quadratic residue mod 7.", "(quadratic-residue? 3 7)", "(equal? (let () (quadratic-residue? 3 7)) #f)", "easy", ["quadratic"]),
    ("legendre-symbol", "Compute Legendre symbol (3/7) with `legendre-symbol`.", "(legendre-symbol 3 7)", "(equal? (let () (legendre-symbol 3 7)) -1)", "easy", ["quadratic"]),
    ("legendre-symbol", "Compute Legendre symbol (4/7) with `legendre-symbol`.", "(legendre-symbol 4 7)", "(equal? (let () (legendre-symbol 4 7)) 1)", "easy", ["quadratic"]),
    ("mod-sqrt-both", "Return both square roots of 4 modulo 13 using `mod-sqrt-both`.", "(mod-sqrt-both 4 13)", "(let ([roots (let () (mod-sqrt-both 4 13))]) (and roots (= (length roots) 2) (= (modulo (* (car roots) (car roots)) 13) 4) (= (modulo (* (cadr roots) (cadr roots)) 13) 4)))", "medium", ["quadratic"]),
    ("mod-sqrt-both", "Return #t iff roots of 9 modulo 13 are valid and sum to 13.", "(let ([roots (mod-sqrt-both 9 13)]) (and roots (= (length roots) 2) (= (+ (car roots) (cadr roots)) 13) (= (modulo (* (car roots) (car roots)) 13) 9) (= (modulo (* (cadr roots) (cadr roots)) 13) 9)))", "(equal? (let () (let ([roots (mod-sqrt-both 9 13)]) (and roots (= (length roots) 2) (= (+ (car roots) (cadr roots)) 13) (= (modulo (* (car roots) (car roots)) 13) 9) (= (modulo (* (cadr roots) (cadr roots)) 13) 9)))) #t)", "medium", ["quadratic"]),
]

for fn, prompt, gt, verify, difficulty, extra_tags in composition_cases:
    add_composition(fn, prompt, gt, verify, difficulty=difficulty, tags=extra_tags)

if sum(1 for s in samples if s["family"] == "composition") != 60:
    raise ValueError("composition family must contain exactly 60 samples")


# -----------------------------------------------------------------------------
# Split train/eval by source_function (leakage-proof)
# -----------------------------------------------------------------------------
if len(samples) != 140:
    raise ValueError(f"expected 140 samples, got {len(samples)}")


def stable_hash_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


all_source_functions = sorted({str(s["source_function"]) for s in samples})
ranked_functions = sorted(all_source_functions, key=stable_hash_int)
target_eval_functions = max(3, round(len(all_source_functions) * 0.18))
eval_function_set = set(ranked_functions[:target_eval_functions])
eval_ids: Set[str] = {str(s["id"]) for s in samples if str(s["source_function"]) in eval_function_set}

by_family: Dict[str, List[Dict[str, object]]] = defaultdict(list)
for s in samples:
    by_family[str(s["family"])].append(s)

train, eval_ = [], []
for s in samples:
    if s["id"] in eval_ids:
        s2 = dict(s)
        s2["split"] = "eval"
        eval_.append(s2)
    else:
        s2 = dict(s)
        s2["split"] = "train"
        train.append(s2)

if len(train) + len(eval_) != len(samples):
    raise ValueError(f"split mismatch: train={len(train)}, eval={len(eval_)}, total={len(samples)}")


# -----------------------------------------------------------------------------
# Write files
# -----------------------------------------------------------------------------

def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


write_jsonl(ALL_PATH, [dict(s, split=("eval" if s["id"] in eval_ids else "train")) for s in samples])
write_jsonl(TRAIN_PATH, train)
write_jsonl(EVAL_PATH, eval_)

summary = {
    "total": len(samples),
    "train": len(train),
    "eval": len(eval_),
    "families": {
        fam: {
            "total": len(fam_samples),
            "eval": sum(1 for s in fam_samples if s["id"] in eval_ids),
            "train": sum(1 for s in fam_samples if s["id"] not in eval_ids),
        }
        for fam, fam_samples in sorted(by_family.items())
    },
}

with SUMMARY_PATH.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print(f"Wrote: {ALL_PATH}")
print(f"Wrote: {TRAIN_PATH}")
print(f"Wrote: {EVAL_PATH}")
