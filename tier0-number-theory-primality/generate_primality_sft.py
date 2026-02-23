#!/usr/bin/env python3
"""Generate SFT samples for lattice/number-theory/primality.ss."""

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

SOURCE_MODULE = "lattice/number-theory/primality.ss"
SOURCE_TEST = "lattice/number-theory/test-primality.ss"

DEFS: Dict[str, str] = {
    # Helpers
    "isqrt": """(define (isqrt n)
  (cond
   [(< n 0) 0]
   [(= n 0) 0]
   [(= n 1) 1]
   [else
    (let loop ([x n])
      (let ([x1 (quotient (+ x (quotient n x)) 2)])
        (if (>= x1 x)
            x
            (loop x1))))]))""",
    "factor-out-2s": """(define (factor-out-2s n)
  (let loop ([d n] [r 0])
    (if (even? d)
        (loop (quotient d 2) (+ r 1))
        (cons d r))))""",
    "miller-rabin-witness?": """(define (miller-rabin-witness? a n)
  (let* ([n-1 (- n 1)]
         [factor-result (factor-out-2s n-1)]
         [d (car factor-result)]
         [r (cdr factor-result)]
         [x (mod-expt a d n)])
    (cond
     [(or (= x 1) (= x n-1)) #f]
     [else
      (let loop ([x x] [i 0])
        (cond
         [(>= i (- r 1)) #t]
         [else
          (let ([x-new (mod-expt x 2 n)])
            (cond
             [(= x-new 1) #t]
             [(= x-new n-1) #f]
             [else (loop x-new (+ i 1))]))]))])))""",
    "deterministic-witnesses": """(define (deterministic-witnesses n k)
  (let ([small-primes '(2 3 5 7 11 13 17 19 23 29 31 37 41 43 47 53 59 61 67 71 73 79 83 89 97)])
    (take (min k (length small-primes)) small-primes)))""",
    "extract-small-factors": """(define (extract-small-factors n limit)
  (let loop ([n n] [d 2] [factors '()])
    (cond
     [(or (= n 1) (> d limit)) (cons n factors)]
     [(zero? (modulo n d))
      (loop (quotient n d) d (cons d factors))]
     [else
      (loop n (if (= d 2) 3 (+ d 2)) factors)])))""",
    "pollard-rho-backtrack": """(define (pollard-rho-backtrack n c xs-history ys-history)
  (let ([f (lambda (x) (modulo (+ (* x x) c) n))])
    (let loop ([xs xs-history] [ys ys-history] [x (if (null? xs-history) 2 (car xs-history))]
               [y (if (null? ys-history) 2 (car ys-history))])
      (if (null? xs)
          n
          (let* ([x-new (f x)]
                 [y-new (f (f y))]
                 [d (gcd (abs (- x-new y-new)) n)])
            (cond
             [(= d 1)
              (loop (cdr xs) (cdr ys) x-new y-new)]
             [(and (> d 1) (< d n)) d]
             [else n]))))))""",
    "pollard-rho-single": """(define (pollard-rho-single n c)
  (let ([f (lambda (x) (modulo (+ (* x x) c) n))]
        [batch-size 128])
    (let outer-loop ([x 2] [y 2] [iter 0])
      (if (> iter 1000000)
          n
          (let inner-loop ([x x] [y y] [prod 1] [batch-count 0]
                           [xs-history '()] [ys-history '()])
            (if (= batch-count batch-size)
                (let ([d (gcd prod n)])
                  (cond
                   [(= d 1)
                    (outer-loop x y (+ iter batch-size))]
                   [(and (> d 1) (< d n)) d]
                   [else
                    (pollard-rho-backtrack n c (reverse xs-history) (reverse ys-history))]))
                (let* ([x-new (f x)]
                       [y-new (f (f y))]
                       [diff (abs (- x-new y-new))]
                       [prod-new (modulo (* prod diff) n)])
                  (inner-loop x-new y-new prod-new (+ batch-count 1)
                              (cons x xs-history) (cons y ys-history)))))))))""",
    "pollard-rho": """(define (pollard-rho n)
  (cond
   [(even? n) 2]
   [else
    (let try-c ([c 1])
      (if (> c 20)
          n
          (let ([result (pollard-rho-single n c)])
            (if (and (> result 1) (< result n))
                result
                (try-c (+ c 1))))))]))""",
    "pollard-rho-factorize": """(define (pollard-rho-factorize n)
  (cond
   [(= n 1) '()]
   [(prime? n) (list n)]
   [else
    (let ([d (pollard-rho n)])
      (if (= d n)
          (trial-division n)
          (append (pollard-rho-factorize d)
                  (pollard-rho-factorize (quotient n d)))))]))""",
    "prime-powers": """(define (prime-powers p e)
  (let loop ([i 0] [pw 1] [result '()])
    (if (> i e)
        (reverse result)
        (loop (+ i 1) (* pw p) (cons pw result)))))""",
    "generate-divisors": """(define (generate-divisors pf)
  (if (null? pf)
      '(1)
      (let* ([p (caar pf)]
             [e (cdar pf)]
             [rest (generate-divisors (cdr pf))]
             [powers (prime-powers p e)])
        (append-map (lambda (pw)
                      (map (lambda (d) (* pw d)) rest))
                    powers))))""",

    # Targets
    "prime?": """(define (prime? n)
  (cond
   [(< n 2) #f]
   [(= n 2) #t]
   [(even? n) #f]
   [(= n 3) #t]
   [else
    (let ([limit (isqrt n)])
      (let loop ([d 3])
        (cond
         [(> d limit) #t]
         [(zero? (modulo n d)) #f]
         [else (loop (+ d 2))])))]))""",
    "miller-rabin?": """(define (miller-rabin? n . args)
  (let ([rounds (if (null? args) 20 (car args))])
    (cond
     [(< n 2) #f]
     [(= n 2) #t]
     [(even? n) #f]
     [(= n 3) #t]
     [(< n 2047)
      (not (miller-rabin-witness? 2 n))]
     [(< n 1373653)
      (not (or (miller-rabin-witness? 2 n)
               (miller-rabin-witness? 3 n)))]
     [(< n 9080191)
      (not (or (miller-rabin-witness? 31 n)
               (miller-rabin-witness? 73 n)))]
     [(< n 25326001)
      (not (or (miller-rabin-witness? 2 n)
               (miller-rabin-witness? 3 n)
               (miller-rabin-witness? 5 n)))]
     [(< n 3215031751)
      (not (or (miller-rabin-witness? 2 n)
               (miller-rabin-witness? 3 n)
               (miller-rabin-witness? 5 n)
               (miller-rabin-witness? 7 n)))]
     [(< n 4759123141)
      (not (or (miller-rabin-witness? 2 n)
               (miller-rabin-witness? 7 n)
               (miller-rabin-witness? 61 n)))]
     [(< n 3317044064679887385961981)
      (let ([witnesses '(2 3 5 7 11 13 17 19 23 29 31 37)])
        (not (exists (lambda (a) (miller-rabin-witness? a n)) witnesses)))]
     [else
      (let ([witnesses (deterministic-witnesses n rounds)])
        (not (exists (lambda (a) (miller-rabin-witness? a n)) witnesses)))])))""",
    "trial-division": """(define (trial-division n)
  (cond
   [(< n 2) '()]
   [else
    (let loop ([n n] [d 2] [factors '()])
      (cond
       [(= n 1) (reverse factors)]
       [(> (* d d) n) (reverse (cons n factors))]
       [(zero? (modulo n d))
        (loop (quotient n d) d (cons d factors))]
       [else
        (loop n (if (= d 2) 3 (+ d 2)) factors)]))]))""",
    "factorize": """(define (factorize n)
  (cond
   [(< n 2) '()]
   [(prime? n) (list n)]
   [else
    (let* ([small-factor-result (extract-small-factors n 1000)]
           [remaining (car small-factor-result)]
           [small-factors (cdr small-factor-result)])
      (if (= remaining 1)
          (sort-by < small-factors)
          (sort-by < (append small-factors (pollard-rho-factorize remaining)))))]))""",
    "prime-factorization": """(define (prime-factorization n)
  (let ([factors (factorize n)])
    (if (null? factors)
        '()
        (let loop ([fs (cdr factors)]
                   [current (car factors)]
                   [count 1]
                   [result '()])
          (cond
           [(null? fs)
            (reverse (cons (cons current count) result))]
           [(= (car fs) current)
            (loop (cdr fs) current (+ count 1) result)]
           [else
            (loop (cdr fs)
                  (car fs)
                  1
                  (cons (cons current count) result))])))))""",
    "divisors": """(define (divisors n)
  (if (< n 1)
      '()
      (let* ([pf (prime-factorization n)])
        (sort-by < (generate-divisors pf)))))""",
    "euler-totient": """(define (euler-totient n)
  (if (< n 1)
      0
      (let ([pf (prime-factorization n)])
        (fold-left
         (lambda (acc pe)
           (let ([p (car pe)]
                 [e (cdr pe)])
             (* acc (- p 1) (expt p (- e 1)))))
         1
         pf))))""",
    "jacobi-symbol": """(define (jacobi-symbol a n)
  (cond
   [(not (and (odd? n) (> n 0))) 0]
   [(= n 1) 1]
   [(= a 0) 0]
   [(= a 1) 1]
   [else
    (let ([a (modulo a n)])
      (cond
       [(= a 0) 0]
       [(= a 1) 1]
       [else
        (let* ([twos-result (factor-out-2s a)]
               [a-odd (car twos-result)]
               [e (cdr twos-result)]
               [two-contrib
                (if (even? e)
                    1
                    (let ([n-mod-8 (modulo n 8)])
                      (if (or (= n-mod-8 1) (= n-mod-8 7))
                          1
                          -1)))])
          (if (= a-odd 1)
              two-contrib
              (let* ([flip-sign
                      (if (and (= (modulo a-odd 4) 3)
                               (= (modulo n 4) 3))
                          -1
                          1)]
                     [recur (jacobi-symbol (modulo n a-odd) a-odd)])
                (* two-contrib flip-sign recur))))]))]))""",
}

DEPENDS: Dict[str, List[str]] = {
    # Helpers
    "isqrt": [],
    "factor-out-2s": [],
    "miller-rabin-witness?": ["factor-out-2s"],
    "deterministic-witnesses": [],
    "extract-small-factors": [],
    "pollard-rho-backtrack": [],
    "pollard-rho-single": ["pollard-rho-backtrack"],
    "pollard-rho": ["pollard-rho-single"],
    "pollard-rho-factorize": ["prime?", "pollard-rho", "trial-division"],
    "prime-powers": [],
    "generate-divisors": ["prime-powers"],

    # Targets
    "prime?": ["isqrt"],
    "miller-rabin?": ["miller-rabin-witness?", "deterministic-witnesses"],
    "trial-division": [],
    "factorize": ["prime?", "extract-small-factors", "pollard-rho-factorize"],
    "prime-factorization": ["factorize"],
    "divisors": ["prime-factorization", "generate-divisors"],
    "euler-totient": ["prime-factorization"],
    "jacobi-symbol": ["factor-out-2s"],
}

FUNCTION_ORDER = [
    "prime?",
    "miller-rabin?",
    "trial-division",
    "factorize",
    "prime-factorization",
    "divisors",
    "euler-totient",
    "jacobi-symbol",
]

FUNCTION_SPECS = {
    "prime?": "Deterministic primality test via trial division up to integer square root.",
    "miller-rabin?": "Deterministic/probabilistic Miller-Rabin primality test with optional rounds argument.",
    "trial-division": "Factor integer n into sorted prime factors with repetition using trial division.",
    "factorize": "General integer factorization: small-factor extraction plus Pollard-rho fallback.",
    "prime-factorization": "Return factorization as sorted (prime . exponent) pairs.",
    "divisors": "Generate all positive divisors of n in ascending order.",
    "euler-totient": "Compute Euler's totient φ(n) from prime factorization.",
    "jacobi-symbol": "Compute Jacobi symbol (a/n) for odd positive n using reciprocity and power-of-two extraction.",
}

SKELETONS = {
    "prime?": """(define (prime? n)
  ;; TODO: trial-division primality test
  <TODO>)""",
    "miller-rabin?": """(define (miller-rabin? n . args)
  ;; TODO: Miller-Rabin with deterministic witness sets for bounded ranges
  <TODO>)""",
    "trial-division": """(define (trial-division n)
  ;; TODO: factor n by dividing out small factors repeatedly
  <TODO>)""",
    "factorize": """(define (factorize n)
  ;; TODO: combine small-factor extraction and Pollard-rho fallback
  <TODO>)""",
    "prime-factorization": """(define (prime-factorization n)
  ;; TODO: compress factor list into (prime . exponent) pairs
  <TODO>)""",
    "divisors": """(define (divisors n)
  ;; TODO: derive all divisors from prime-factorization
  <TODO>)""",
    "euler-totient": """(define (euler-totient n)
  ;; TODO: compute phi(n) from prime powers
  <TODO>)""",
    "jacobi-symbol": """(define (jacobi-symbol a n)
  ;; TODO: implement Jacobi symbol recursion with reciprocity
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "prime?": "(and (not (prime? 0)) (not (prime? 1)) (prime? 2) (prime? 3) (not (prime? 4)) (prime? 97) (not (prime? 1001)))",
    "miller-rabin?": "(and (miller-rabin? 104729) (miller-rabin? 1299709 5) (not (miller-rabin? 561)) (not (miller-rabin? 1105)) (equal? (miller-rabin? 997) (prime? 997)) (equal? (miller-rabin? 1000) (prime? 1000)))",
    "trial-division": "(and (equal? (trial-division 1) '()) (equal? (trial-division 12) '(2 2 3)) (equal? (trial-division 1001) '(7 11 13)) (= (apply * (trial-division 360)) 360))",
    "factorize": "(and (equal? (factorize 1) '()) (equal? (factorize 100) '(2 2 5 5)) (equal? (factorize 997) '(997)) (= (apply * (factorize 12345)) 12345))",
    "prime-factorization": "(and (equal? (prime-factorization 1) '()) (equal? (prime-factorization 12) '((2 . 2) (3 . 1))) (equal? (prime-factorization 360) '((2 . 3) (3 . 2) (5 . 1))))",
    "divisors": "(and (equal? (divisors 1) '(1)) (equal? (divisors 12) '(1 2 3 4 6 12)) (= (length (divisors 360)) 24) (= (car (reverse (divisors 28))) 28))",
    "euler-totient": "(and (= (euler-totient 1) 1) (= (euler-totient 2) 1) (= (euler-totient 10) 4) (= (euler-totient 36) 12) (= (euler-totient 13) 12))",
    "jacobi-symbol": "(and (= (jacobi-symbol 2 15) 1) (= (jacobi-symbol 11 15) -1) (= (jacobi-symbol 14 15) -1) (= (jacobi-symbol 3 10) 0) (= (jacobi-symbol (+ 14 15) 15) (jacobi-symbol 14 15)))",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")
TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "prime?": "def is_prime(n):\n    if n < 2:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    if n == 3:\n        return True\n    d = 3\n    limit = isqrt(n)\n    while d <= limit:\n        if n % d == 0:\n            return False\n        d += 2\n    return True",
    "miller-rabin?": "def miller_rabin(n, rounds=20):\n    if n < 2:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    if n == 3:\n        return True\n    if n < 2047:\n        return not mr_witness(2, n)\n    if n < 1373653:\n        return not (mr_witness(2, n) or mr_witness(3, n))\n    if n < 9080191:\n        return not (mr_witness(31, n) or mr_witness(73, n))\n    if n < 25326001:\n        return not (mr_witness(2, n) or mr_witness(3, n) or mr_witness(5, n))\n    if n < 3215031751:\n        return not (mr_witness(2, n) or mr_witness(3, n) or mr_witness(5, n) or mr_witness(7, n))\n    if n < 4759123141:\n        return not (mr_witness(2, n) or mr_witness(7, n) or mr_witness(61, n))\n    if n < 3317044064679887385961981:\n        witnesses = [2,3,5,7,11,13,17,19,23,29,31,37]\n    else:\n        witnesses = deterministic_witnesses(n, rounds)\n    return not any(mr_witness(a, n) for a in witnesses)",
    "trial-division": "def trial_division(n):\n    if n < 2:\n        return []\n    factors = []\n    d = 2\n    while n > 1:\n        if d * d > n:\n            factors.append(n)\n            break\n        if n % d == 0:\n            factors.append(d)\n            n //= d\n        else:\n            d = 3 if d == 2 else d + 2\n    return factors",
    "factorize": "def factorize(n):\n    if n < 2:\n        return []\n    if is_prime(n):\n        return [n]\n    remaining, small = extract_small_factors(n, 1000)\n    if remaining == 1:\n        return sorted(small)\n    return sorted(small + pollard_rho_factorize(remaining))",
    "prime-factorization": "def prime_factorization(n):\n    factors = factorize(n)\n    if not factors:\n        return []\n    out = []\n    current = factors[0]\n    count = 1\n    for x in factors[1:]:\n        if x == current:\n            count += 1\n        else:\n            out.append((current, count))\n            current = x\n            count = 1\n    out.append((current, count))\n    return out",
    "divisors": "def divisors(n):\n    if n < 1:\n        return []\n    pf = prime_factorization(n)\n    return sorted(generate_divisors(pf))",
    "euler-totient": "def euler_totient(n):\n    if n < 1:\n        return 0\n    out = 1\n    for p, e in prime_factorization(n):\n        out *= (p - 1) * (p ** (e - 1))\n    return out",
    "jacobi-symbol": "def jacobi_symbol(a, n):\n    if n <= 0 or n % 2 == 0:\n        return 0\n    if n == 1:\n        return 1\n    if a == 0:\n        return 0\n    if a == 1:\n        return 1\n    a %= n\n    if a == 0:\n        return 0\n    if a == 1:\n        return 1\n    a_odd, e = factor_out_2s(a)\n    if e % 2 == 0:\n        two = 1\n    else:\n        two = 1 if n % 8 in (1, 7) else -1\n    if a_odd == 1:\n        return two\n    flip = -1 if (a_odd % 4 == 3 and n % 4 == 3) else 1\n    return two * flip * jacobi_symbol(n % a_odd, a_odd)",
}

CHEZ_SNIPPETS = {
    "prime?": "(define (is-prime n)\n  (cond ((< n 2) #f)\n        ((= n 2) #t)\n        ((even? n) #f)\n        ((= n 3) #t)\n        (else\n         (let ((limit (isqrt n)))\n           (let loop ((d 3))\n             (cond ((> d limit) #t)\n                   ((zero? (modulo n d)) #f)\n                   (else (loop (+ d 2)))))))))",
    "miller-rabin?": "(define (mr? n . args)\n  (let ((rounds (if (null? args) 20 (car args))))\n    (cond ((< n 2) #f)\n          ((= n 2) #t)\n          ((even? n) #f)\n          ((= n 3) #t)\n          ((< n 2047) (not (miller-rabin-witness? 2 n)))\n          ((< n 1373653) (not (or (miller-rabin-witness? 2 n) (miller-rabin-witness? 3 n))))\n          ((< n 9080191) (not (or (miller-rabin-witness? 31 n) (miller-rabin-witness? 73 n))))\n          ((< n 25326001) (not (or (miller-rabin-witness? 2 n) (miller-rabin-witness? 3 n) (miller-rabin-witness? 5 n))))\n          ((< n 3215031751) (not (or (miller-rabin-witness? 2 n) (miller-rabin-witness? 3 n) (miller-rabin-witness? 5 n) (miller-rabin-witness? 7 n))))\n          ((< n 4759123141) (not (or (miller-rabin-witness? 2 n) (miller-rabin-witness? 7 n) (miller-rabin-witness? 61 n))))\n          ((< n 3317044064679887385961981)\n           (not (exists (lambda (a) (miller-rabin-witness? a n)) '(2 3 5 7 11 13 17 19 23 29 31 37))))\n          (else\n           (let ((ws (deterministic-witnesses n rounds)))\n             (not (exists (lambda (a) (miller-rabin-witness? a n)) ws)))))))",
    "trial-division": "(define (trial-div n)\n  (cond ((< n 2) '())\n        (else\n         (let loop ((n n) (d 2) (factors '()))\n           (cond ((= n 1) (reverse factors))\n                 ((> (* d d) n) (reverse (cons n factors)))\n                 ((zero? (modulo n d)) (loop (quotient n d) d (cons d factors)))\n                 (else (loop n (if (= d 2) 3 (+ d 2)) factors)))))))",
    "factorize": "(define (factorize0 n)\n  (cond ((< n 2) '())\n        ((prime? n) (list n))\n        (else\n         (let* ((r (extract-small-factors n 1000))\n                (remaining (car r))\n                (small (cdr r)))\n           (if (= remaining 1)\n               (sort-by < small)\n               (sort-by < (append small (pollard-rho-factorize remaining))))))))",
    "prime-factorization": "(define (pf0 n)\n  (let ((factors (factorize n)))\n    (if (null? factors)\n        '()\n        (let loop ((fs (cdr factors)) (current (car factors)) (count 1) (result '()))\n          (cond ((null? fs) (reverse (cons (cons current count) result)))\n                ((= (car fs) current) (loop (cdr fs) current (+ count 1) result))\n                (else (loop (cdr fs) (car fs) 1 (cons (cons current count) result))))))))",
    "divisors": "(define (divisors0 n)\n  (if (< n 1)\n      '()\n      (let ((pf (prime-factorization n)))\n        (sort-by < (generate-divisors pf)))))",
    "euler-totient": "(define (phi n)\n  (if (< n 1)\n      0\n      (fold-left (lambda (acc pe)\n                   (let ((p (car pe)) (e (cdr pe)))\n                     (* acc (- p 1) (expt p (- e 1)))))\n                 1\n                 (prime-factorization n))))",
    "jacobi-symbol": "(define (jacobi0 a n)\n  (cond ((not (and (odd? n) (> n 0))) 0)\n        ((= n 1) 1)\n        ((= a 0) 0)\n        ((= a 1) 1)\n        (else\n         (let ((a (modulo a n)))\n           (cond ((= a 0) 0)\n                 ((= a 1) 1)\n                 (else\n                  (let* ((tw (factor-out-2s a))\n                         (a-odd (car tw))\n                         (e (cdr tw))\n                         (two (if (even? e) 1 (if (or (= (modulo n 8) 1) (= (modulo n 8) 7)) 1 -1))))\n                    (if (= a-odd 1)\n                        two\n                        (let* ((flip (if (and (= (modulo a-odd 4) 3) (= (modulo n 4) 3)) -1 1))\n                               (recur (jacobi0 (modulo n a-odd) a-odd)))\n                          (* two flip recur))))))))))",
}

BUGGY_CASES = [
    {
        "fn": "prime?",
        "buggy": "(define (prime? n)\n  (cond\n   [(< n 2) #f]\n   [(even? n) #f]\n   [(= n 2) #t]\n   [(= n 3) #t]\n   [else\n    (let ([limit (isqrt n)])\n      (let loop ([d 3])\n        (cond\n         [(> d limit) #t]\n         [(zero? (modulo n d)) #f]\n         [else (loop (+ d 2))])))]))",
        "note": "The n=2 special case must be checked before the even-number rejection.",
    },
    {
        "fn": "prime?",
        "buggy": "(define (prime? n)\n  (cond\n   [(< n 2) #f]\n   [(= n 2) #t]\n   [(even? n) #f]\n   [(= n 3) #t]\n   [else\n    (let ([limit (isqrt n)])\n      (let loop ([d 3])\n        (cond\n         [(> d limit) #f]\n         [(zero? (modulo n d)) #f]\n         [else (loop (+ d 2))])))]))",
        "note": "If no divisor is found up to sqrt(n), the number is prime (#t), not composite.",
    },
    {
        "fn": "miller-rabin?",
        "buggy": "(define (miller-rabin? n . args)\n  (let ([rounds (if (null? args) 20 (car args))])\n    (cond\n     [(< n 2) #f]\n     [(= n 2) #t]\n     [(even? n) #t]\n     [(= n 3) #t]\n     [(< n 2047)\n      (not (miller-rabin-witness? 2 n))]\n     [(< n 1373653)\n      (not (or (miller-rabin-witness? 2 n)\n               (miller-rabin-witness? 3 n)))]\n     [(< n 9080191)\n      (not (or (miller-rabin-witness? 31 n)\n               (miller-rabin-witness? 73 n)))]\n     [(< n 25326001)\n      (not (or (miller-rabin-witness? 2 n)\n               (miller-rabin-witness? 3 n)\n               (miller-rabin-witness? 5 n)))]\n     [(< n 3215031751)\n      (not (or (miller-rabin-witness? 2 n)\n               (miller-rabin-witness? 3 n)\n               (miller-rabin-witness? 5 n)\n               (miller-rabin-witness? 7 n)))]\n     [(< n 4759123141)\n      (not (or (miller-rabin-witness? 2 n)\n               (miller-rabin-witness? 7 n)\n               (miller-rabin-witness? 61 n)))]\n     [(< n 3317044064679887385961981)\n      (let ([witnesses '(2 3 5 7 11 13 17 19 23 29 31 37)])\n        (not (exists (lambda (a) (miller-rabin-witness? a n)) witnesses)))]\n     [else\n      (let ([witnesses (deterministic-witnesses n rounds)])\n        (not (exists (lambda (a) (miller-rabin-witness? a n)) witnesses)))])))",
        "note": "Even numbers greater than 2 are composite and must return #f.",
    },
    {
        "fn": "miller-rabin?",
        "buggy": "(define (miller-rabin? n . args)\n  (let ([rounds (if (null? args) 20 (car args))])\n    (cond\n     [(< n 2) #f]\n     [(= n 2) #t]\n     [(even? n) #f]\n     [(= n 3) #t]\n     [(< n 2047)\n      (miller-rabin-witness? 2 n)]\n     [(< n 1373653)\n      (not (or (miller-rabin-witness? 2 n)\n               (miller-rabin-witness? 3 n)))]\n     [(< n 9080191)\n      (not (or (miller-rabin-witness? 31 n)\n               (miller-rabin-witness? 73 n)))]\n     [(< n 25326001)\n      (not (or (miller-rabin-witness? 2 n)\n               (miller-rabin-witness? 3 n)\n               (miller-rabin-witness? 5 n)))]\n     [(< n 3215031751)\n      (not (or (miller-rabin-witness? 2 n)\n               (miller-rabin-witness? 3 n)\n               (miller-rabin-witness? 5 n)\n               (miller-rabin-witness? 7 n)))]\n     [(< n 4759123141)\n      (not (or (miller-rabin-witness? 2 n)\n               (miller-rabin-witness? 7 n)\n               (miller-rabin-witness? 61 n)))]\n     [(< n 3317044064679887385961981)\n      (let ([witnesses '(2 3 5 7 11 13 17 19 23 29 31 37)])\n        (not (exists (lambda (a) (miller-rabin-witness? a n)) witnesses)))]\n     [else\n      (let ([witnesses (deterministic-witnesses n rounds)])\n        (not (exists (lambda (a) (miller-rabin-witness? a n)) witnesses)))])))",
        "note": "Witness check returns compositeness evidence; primality result must negate it.",
    },
    {
        "fn": "trial-division",
        "buggy": "(define (trial-division n)\n  (cond\n   [(< n 2) '()]\n   [else\n    (let loop ([n n] [d 2] [factors '()])\n      (cond\n       [(= n 1) (reverse factors)]\n       [(> (* d d) n) (reverse (cons n factors))]\n       [(zero? (modulo n d))\n        (loop (quotient n d) (if (= d 2) 3 (+ d 2)) (cons d factors))]\n       [else\n        (loop n (if (= d 2) 3 (+ d 2)) factors)]))]))",
        "note": "When a factor divides n, keep dividing by the same d to preserve multiplicity.",
    },
    {
        "fn": "trial-division",
        "buggy": "(define (trial-division n)\n  (cond\n   [(< n 2) '()]\n   [else\n    (let loop ([n n] [d 2] [factors '()])\n      (cond\n       [(= n 1) (reverse factors)]\n       [(> (* d d) n) (reverse factors)]\n       [(zero? (modulo n d))\n        (loop (quotient n d) d (cons d factors))]\n       [else\n        (loop n (if (= d 2) 3 (+ d 2)) factors)]))]))",
        "note": "When d^2 > n and n>1, the remaining n is a prime factor and must be included.",
    },
    {
        "fn": "factorize",
        "buggy": "(define (factorize n)\n  (cond\n   [(< n 2) '()]\n   [(prime? n) '()]\n   [else\n    (let* ([small-factor-result (extract-small-factors n 1000)]\n           [remaining (car small-factor-result)]\n           [small-factors (cdr small-factor-result)])\n      (if (= remaining 1)\n          (sort-by < small-factors)\n          (sort-by < (append small-factors (pollard-rho-factorize remaining)))))]))",
        "note": "Prime input should return a singleton factor list, not empty.",
    },
    {
        "fn": "factorize",
        "buggy": "(define (factorize n)\n  (cond\n   [(< n 2) '()]\n   [(prime? n) (list n)]\n   [else\n    (let* ([small-factor-result (extract-small-factors n 1000)]\n           [remaining (car small-factor-result)]\n           [small-factors (cdr small-factor-result)])\n      (if (= remaining 1)\n          (sort-by < small-factors)\n          (sort-by < (append small-factors (list remaining)))))]))",
        "note": "If a composite remainder survives small extraction, it must be fully factored, not appended as one factor.",
    },
    {
        "fn": "prime-factorization",
        "buggy": "(define (prime-factorization n)\n  (let ([factors (factorize n)])\n    (map (lambda (p) (cons p 1)) factors)))",
        "note": "Repeated prime factors must be grouped into exponent counts.",
    },
    {
        "fn": "prime-factorization",
        "buggy": "(define (prime-factorization n)\n  (let ([factors (factorize n)])\n    (if (null? factors)\n        '()\n        (let loop ([fs (cdr factors)] [current (car factors)] [count 1] [result '()])\n          (cond\n           [(null? fs) (reverse result)]\n           [(= (car fs) current) (loop (cdr fs) current (+ count 1) result)]\n           [else (loop (cdr fs) (car fs) 1 (cons (cons current count) result))])))))",
        "note": "The final accumulated (prime . exponent) pair must be emitted when traversal ends.",
    },
    {
        "fn": "divisors",
        "buggy": "(define (divisors n)\n  (if (< n 1)\n      '()\n      (let* ([pf (prime-factorization n)])\n        (generate-divisors pf))))",
        "note": "Divisors output is expected in sorted ascending order.",
    },
    {
        "fn": "divisors",
        "buggy": "(define (divisors n)\n  (if (< n 1)\n      '()\n      (prime-factorization n)))",
        "note": "Divisors are integers, not exponent pairs.",
    },
    {
        "fn": "euler-totient",
        "buggy": "(define (euler-totient n)\n  (if (< n 1)\n      0\n      (let ([pf (prime-factorization n)])\n        (fold-left\n         (lambda (acc pe)\n           (let ([p (car pe)]\n                 [e (cdr pe)])\n             (* acc p (expt p (- e 1)))))\n         1\n         pf))))",
        "note": "Each prime-power contribution is (p-1)*p^(e-1), not p*p^(e-1).",
    },
    {
        "fn": "euler-totient",
        "buggy": "(define (euler-totient n)\n  (if (< n 1)\n      1\n      (let ([pf (prime-factorization n)])\n        (fold-left\n         (lambda (acc pe)\n           (let ([p (car pe)]\n                 [e (cdr pe)])\n             (* acc (- p 1) (expt p (- e 1)))))\n         1\n         pf))))",
        "note": "By contract, phi(n) is 0 for n < 1.",
    },
    {
        "fn": "jacobi-symbol",
        "buggy": "(define (jacobi-symbol a n)\n  (cond\n   [(not (and (odd? n) (> n 0))) 1]\n   [(= n 1) 1]\n   [(= a 0) 0]\n   [(= a 1) 1]\n   [else\n    (let ([a (modulo a n)])\n      (cond\n       [(= a 0) 0]\n       [(= a 1) 1]\n       [else\n        (let* ([twos-result (factor-out-2s a)]\n               [a-odd (car twos-result)]\n               [e (cdr twos-result)]\n               [two-contrib\n                (if (even? e)\n                    1\n                    (let ([n-mod-8 (modulo n 8)])\n                      (if (or (= n-mod-8 1) (= n-mod-8 7))\n                          1\n                          -1)))])\n          (if (= a-odd 1)\n              two-contrib\n              (let* ([flip-sign\n                      (if (and (= (modulo a-odd 4) 3)\n                               (= (modulo n 4) 3))\n                          -1\n                          1)]\n                     [recur (jacobi-symbol (modulo n a-odd) a-odd)])\n                (* two-contrib flip-sign recur))))]))]))",
        "note": "Jacobi symbol is only defined for odd positive n here; invalid inputs should return 0.",
    },
    {
        "fn": "jacobi-symbol",
        "buggy": "(define (jacobi-symbol a n)\n  (cond\n   [(not (and (odd? n) (> n 0))) 0]\n   [(= n 1) 1]\n   [(= a 0) 0]\n   [(= a 1) 1]\n   [else\n    (let ([a (modulo a n)])\n      (cond\n       [(= a 0) 0]\n       [(= a 1) 1]\n       [else\n        (let* ([twos-result (factor-out-2s a)]\n               [a-odd (car twos-result)]\n               [e (cdr twos-result)]\n               [two-contrib\n                (if (even? e)\n                    1\n                    (let ([n-mod-8 (modulo n 8)])\n                      (if (or (= n-mod-8 1) (= n-mod-8 7))\n                          -1\n                          1)))])\n          (if (= a-odd 1)\n              two-contrib\n              (let* ([flip-sign\n                      (if (and (= (modulo a-odd 4) 3)\n                               (= (modulo n 4) 3))\n                          -1\n                          1)]\n                     [recur (jacobi-symbol (modulo n a-odd) a-odd)])\n                (* two-contrib flip-sign recur))))]))]))",
        "note": "The (2/n) contribution for odd exponent e is +1 when n mod 8 is 1 or 7, otherwise -1.",
    },
]

DIFFICULTY = {
    "prime?": "easy",
    "miller-rabin?": "hard",
    "trial-division": "medium",
    "factorize": "hard",
    "prime-factorization": "medium",
    "divisors": "medium",
    "euler-totient": "medium",
    "jacobi-symbol": "hard",
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
    sid = f"primality_{family}_{family_counter[family]:03d}"
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
    parts = [DEFS[d] for d in dependency_closure(fn)] + [DEFS[fn], VERIFY_BY_FUNCTION[fn]]
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
        tags=["tier0", "number-theory", "primality", "spec-to-code", fn],
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
        tags=["tier0", "number-theory", "primality", "spec-to-code", "skeleton", fn],
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
Preserve behavior exactly.

Target function name: `{fn}`

```python
{PYTHON_SNIPPETS[fn]}
```

Return only the Scheme definition.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "number-theory", "primality", "translation", "python", fn],
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
        tags=["tier0", "number-theory", "primality", "translation", "chez", fn],
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
        tags=["tier0", "number-theory", "primality", "bugfix", fn],
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
        tags=["tier0", "number-theory", "primality", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # prime?
    ("prime?", "Check whether integer 97 is prime.", "(prime? 97)", "(equal? (prime? 97) #t)", "easy", ["direct"]),
    ("prime?", "Check primality at edge value n=1.", "(prime? 1)", "(equal? (prime? 1) #f)", "easy", ["edge-case"]),
    ("prime?", "Check primality of next-prime after 100.", "(prime? (next-prime 100))", "(equal? (prime? (next-prime 100)) #t)", "medium", ["integration"]),
    ("prime?", "Return #t iff product 17*19 is not prime.", "(not (prime? (* 17 19)))", "(equal? (not (prime? (* 17 19))) #t)", "medium", ["property"]),

    # miller-rabin?
    ("miller-rabin?", "Run Miller-Rabin on known prime 104729.", "(miller-rabin? 104729)", "(equal? (miller-rabin? 104729) #t)", "medium", ["direct"]),
    ("miller-rabin?", "Run Miller-Rabin on Carmichael number 561.", "(miller-rabin? 561)", "(equal? (miller-rabin? 561) #f)", "hard", ["edge-case"]),
    ("miller-rabin?", "Check that Miller-Rabin agrees with prime? on 997.", "(list (miller-rabin? 997) (prime? 997))", "(equal? (list (miller-rabin? 997) (prime? 997)) '(#t #t))", "hard", ["integration"]),
    ("miller-rabin?", "Run Miller-Rabin with explicit round count on 1299709.", "(miller-rabin? 1299709 5)", "(equal? (miller-rabin? 1299709 5) #t)", "hard", ["direct"]),

    # trial-division
    ("trial-division", "Factor 360 with trial division.", "(trial-division 360)", "(equal? (trial-division 360) '(2 2 2 3 3 5))", "medium", ["direct"]),
    ("trial-division", "Factor edge case n=1 with trial division.", "(trial-division 1)", "(equal? (trial-division 1) '())", "easy", ["edge-case"]),
    ("trial-division", "Verify product of trial-division factors equals original n for 12345.", "(= (apply * (trial-division 12345)) 12345)", "(equal? (= (apply * (trial-division 12345)) 12345) #t)", "medium", ["property"]),
    ("trial-division", "Compare trial-division and factorize on 360.", "(list (trial-division 360) (factorize 360))", "(equal? (list (trial-division 360) (factorize 360)) '((2 2 2 3 3 5) (2 2 2 3 3 5)))", "medium", ["integration"]),

    # factorize
    ("factorize", "Factor 1001 using factorize.", "(factorize 1001)", "(equal? (factorize 1001) '(7 11 13))", "medium", ["direct"]),
    ("factorize", "Factor a prime input 997.", "(factorize 997)", "(equal? (factorize 997) '(997))", "easy", ["edge-case"]),
    ("factorize", "Verify factorize factors multiply back to 1000.", "(= (apply * (factorize 1000)) 1000)", "(equal? (= (apply * (factorize 1000)) 1000) #t)", "medium", ["property"]),
    ("factorize", "Check factorize agrees with trial-division on 360.", "(list (factorize 360) (trial-division 360))", "(equal? (list (factorize 360) (trial-division 360)) '((2 2 2 3 3 5) (2 2 2 3 3 5)))", "hard", ["integration"]),

    # prime-factorization
    ("prime-factorization", "Compute exponent-form factorization of 360.", "(prime-factorization 360)", "(equal? (prime-factorization 360) '((2 . 3) (3 . 2) (5 . 1)))", "medium", ["direct"]),
    ("prime-factorization", "Compute exponent-form factorization of 1.", "(prime-factorization 1)", "(equal? (prime-factorization 1) '())", "easy", ["edge-case"]),
    ("prime-factorization", "Reconstruct 1000 from prime-factorization output.", "(fold-left * 1 (append-map (lambda (pe) (make-list (cdr pe) (car pe))) (prime-factorization 1000)))", "(equal? (fold-left * 1 (append-map (lambda (pe) (make-list (cdr pe) (car pe))) (prime-factorization 1000))) 1000)", "hard", ["property"]),
    ("prime-factorization", "Use prime-factorization terms to rederive phi(36).", "(let ([pf (prime-factorization 36)]) (fold-left * 1 (map (lambda (pe) (* (- (car pe) 1) (expt (car pe) (- (cdr pe) 1)))) pf)))", "(equal? (let ([pf (prime-factorization 36)]) (fold-left * 1 (map (lambda (pe) (* (- (car pe) 1) (expt (car pe) (- (cdr pe) 1)))) pf))) (euler-totient 36))", "hard", ["integration"]),

    # divisors
    ("divisors", "List all positive divisors of 12.", "(divisors 12)", "(equal? (divisors 12) '(1 2 3 4 6 12))", "easy", ["direct"]),
    ("divisors", "List all positive divisors of 1.", "(divisors 1)", "(equal? (divisors 1) '(1))", "easy", ["edge-case"]),
    ("divisors", "Return number of divisors of 360.", "(length (divisors 360))", "(equal? (length (divisors 360)) 24)", "medium", ["property"]),
    ("divisors", "Verify each divisor of 28 divides 28.", "(null? (filter (lambda (d) (not (zero? (modulo 28 d)))) (divisors 28)))", "(equal? (null? (filter (lambda (d) (not (zero? (modulo 28 d)))) (divisors 28))) #t)", "medium", ["integration"]),

    # euler-totient
    ("euler-totient", "Compute Euler totient phi(36).", "(euler-totient 36)", "(equal? (euler-totient 36) 12)", "medium", ["direct"]),
    ("euler-totient", "Compute Euler totient phi(1).", "(euler-totient 1)", "(equal? (euler-totient 1) 1)", "easy", ["edge-case"]),
    ("euler-totient", "Use prime rule phi(p)=p-1 for p=13.", "(= (euler-totient 13) 12)", "(equal? (= (euler-totient 13) 12) #t)", "medium", ["property"]),
    ("euler-totient", "Cross-check phi(10) with brute-force coprime count.", "(= (euler-totient 10) (length (filter (lambda (k) (= (gcd k 10) 1)) (range 1 11))))", "(equal? (= (euler-totient 10) (length (filter (lambda (k) (= (gcd k 10) 1)) (range 1 11)))) #t)", "hard", ["integration"]),

    # jacobi-symbol
    ("jacobi-symbol", "Compute Jacobi symbol (2/15).", "(jacobi-symbol 2 15)", "(equal? (jacobi-symbol 2 15) 1)", "hard", ["direct"]),
    ("jacobi-symbol", "Compute Jacobi symbol (11/15).", "(jacobi-symbol 11 15)", "(equal? (jacobi-symbol 11 15) -1)", "hard", ["direct"]),
    ("jacobi-symbol", "Return 0 for invalid even modulus in Jacobi symbol.", "(jacobi-symbol 3 10)", "(equal? (jacobi-symbol 3 10) 0)", "medium", ["edge-case"]),
    ("jacobi-symbol", "Check periodicity in first argument modulo n.", "(= (jacobi-symbol (+ 14 (* 3 15)) 15) (jacobi-symbol 14 15))", "(equal? (= (jacobi-symbol (+ 14 (* 3 15)) 15) (jacobi-symbol 14 15)) #t)", "hard", ["property"]),
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
