#!/usr/bin/env python3
"""Generate SFT samples for core/base/prelude.ss (focused subset)."""

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

SOURCE_MODULE = "core/base/prelude.ss"
SOURCE_TEST = "core/base/test-prelude.ss"

DEFS: Dict[str, str] = {
    "andmap": """(define (andmap pred lst)
  (or (null? lst)
      (and (pred (car lst))
           (andmap pred (cdr lst)))))""",
    "ormap": """(define (ormap pred lst)
  (and (pair? lst)
       (or (pred (car lst))
           (ormap pred (cdr lst)))))""",
    "filter": """(define (filter pred lst)
  (cond
    [(null? lst) '()]
    [(pred (car lst)) (cons (car lst) (filter pred (cdr lst)))]
    [else (filter pred (cdr lst))]))""",
    "filter-map": """(define (filter-map f lst)
  (let loop ([lst lst] [acc '()])
    (if (null? lst)
        (reverse acc)
        (let ([result (f (car lst))])
          (if result
              (loop (cdr lst) (cons result acc))
              (loop (cdr lst) acc))))))""",
    "fold-left": """(define (fold-left f acc lst)
  (if (null? lst)
      acc
      (fold-left f (f acc (car lst)) (cdr lst))))""",
    "fold-right": """(define (fold-right f acc lst)
  (if (null? lst)
      acc
      (f (car lst) (fold-right f acc (cdr lst)))))""",
    "zip": """(define (zip xs ys)
  (if (or (null? xs) (null? ys))
      '()
      (cons (cons (car xs) (car ys))
            (zip (cdr xs) (cdr ys)))))""",
    "iota": """(define (iota n)
  (let loop ([i 0] [acc '()])
    (if (= i n)
        (reverse acc)
        (loop (+ i 1) (cons i acc)))))""",
    "range": """(define (range start end)
  (let loop ([i start] [acc '()])
    (if (>= i end)
        (reverse acc)
        (loop (+ i 1) (cons i acc)))))""",
    "take": """(define (take n lst)
  (if (or (= n 0) (null? lst))
      '()
      (cons (car lst) (take (- n 1) (cdr lst)))))""",
    "drop": """(define (drop n lst)
  (if (or (= n 0) (null? lst))
      lst
      (drop (- n 1) (cdr lst))))""",
    "find": """(define (find pred lst)
  (cond
    [(null? lst) #f]
    [(pred (car lst)) (car lst)]
    [else (find pred (cdr lst))]))""",
    "last": """(define (last lst)
  (if (null? (cdr lst))
      (car lst)
      (last (cdr lst))))""",
    "init": """(define (init lst)
  (if (null? (cdr lst))
      '()
      (cons (car lst) (init (cdr lst)))))""",
    "replicate": """(define (replicate n x)
  (if (<= n 0)
      '()
      (cons x (replicate (- n 1) x))))""",
    "span": """(define (span pred lst)
  (cond
    [(null? lst) (values '() '())]
    [(pred (car lst))
     (let-values ([(pre suf) (span pred (cdr lst))])
       (values (cons (car lst) pre) suf))]
    [else (values '() lst)]))""",
    "break": """(define (break pred lst)
  (span (lambda (x) (not (pred x))) lst))""",
    "sum": """(define (sum lst)
  (fold-left + 0 lst))""",
    "product": """(define (product lst)
  (fold-left * 1 lst))""",
    "mean": """(define (mean lst)
  (if (null? lst)
      (error 'mean "empty list")
      (/ (sum lst) (length lst))))""",
    "identity": """(define (identity x)
  x)""",
    "flatten": """(define (flatten lst-of-lists)
  (fold-right append '() lst-of-lists))""",
    "append-map": """(define (append-map f lst)
  (fold-right (lambda (x acc) (append (f x) acc)) '() lst))""",
    "partition": """(define (partition pred lst)
  (let loop ([lst lst] [yes '()] [no '()])
    (cond
      [(null? lst) (list (reverse yes) (reverse no))]
      [(pred (car lst)) (loop (cdr lst) (cons (car lst) yes) no)]
      [else (loop (cdr lst) yes (cons (car lst) no))])))""",
    "group-by": """(define (group-by key-fn lst)
  (if (null? lst)
      '()
      (let ([first-key (key-fn (car lst))])
        (let loop ([remaining lst] [current-key first-key] [current-group '()] [result '()])
          (cond
            [(null? remaining)
             (reverse (cons (cons current-key (reverse current-group)) result))]
            [(equal? (key-fn (car remaining)) current-key)
             (loop (cdr remaining) current-key (cons (car remaining) current-group) result)]
            [else
             (let ([new-key (key-fn (car remaining))])
               (loop (cdr remaining) new-key (list (car remaining))
                     (cons (cons current-key (reverse current-group)) result)))])))))""",
    "distinct-by": """(define (distinct-by key-fn lst)
  (let loop ([lst lst] [seen '()] [result '()])
    (cond
      [(null? lst) (reverse result)]
      [else
       (let* ([elem (car lst)]
              [key (key-fn elem)])
         (if (member key seen)
             (loop (cdr lst) seen result)
             (loop (cdr lst) (cons key seen) (cons elem result))))])))""",
}

FUNCTION_ORDER = [
    "andmap",
    "ormap",
    "filter",
    "filter-map",
    "fold-left",
    "fold-right",
    "zip",
    "iota",
    "range",
    "take",
    "drop",
    "find",
    "last",
    "init",
    "replicate",
    "span",
    "break",
    "sum",
    "product",
    "mean",
    "identity",
    "flatten",
    "append-map",
    "partition",
    "group-by",
    "distinct-by",
]

DEPENDS: Dict[str, List[str]] = {
    "andmap": [],
    "ormap": [],
    "filter": [],
    "filter-map": [],
    "fold-left": [],
    "fold-right": [],
    "zip": [],
    "iota": [],
    "range": [],
    "take": [],
    "drop": [],
    "find": [],
    "last": [],
    "init": [],
    "replicate": [],
    "span": [],
    "break": ["span"],
    "sum": ["fold-left"],
    "product": ["fold-left"],
    "mean": ["sum", "fold-left"],
    "identity": [],
    "flatten": ["fold-right"],
    "append-map": ["fold-right"],
    "partition": [],
    "group-by": [],
    "distinct-by": [],
}

FUNCTION_SPECS = {
    "andmap": "Apply predicate to all elements; return #t for empty list.",
    "ormap": "Apply predicate to list; return first truthy result, or #f if none match.",
    "filter": "Return only elements satisfying predicate.",
    "filter-map": "Map function and keep only non-#f results.",
    "fold-left": "Left-associative fold over list.",
    "fold-right": "Right-associative fold over list.",
    "zip": "Pair elements from two lists, stopping at shorter list.",
    "iota": "Return list `(0 ... n-1)`.",
    "range": "Return list from start (inclusive) to end (exclusive).",
    "take": "Return first n elements from list.",
    "drop": "Drop first n elements from list.",
    "find": "Return first matching element, else #f.",
    "last": "Return final element of non-empty list.",
    "init": "Return all elements except last.",
    "replicate": "Return list containing n copies of value.",
    "span": "Split list at first element that fails predicate.",
    "break": "Split list at first element that satisfies predicate.",
    "sum": "Sum numeric list; return 0 for empty list.",
    "product": "Multiply numeric list; return 1 for empty list.",
    "mean": "Arithmetic mean; raise error on empty list.",
    "identity": "Return argument unchanged.",
    "flatten": "Concatenate list of lists.",
    "append-map": "Map to lists and concatenate results.",
    "partition": "Return `(list yes no)` by predicate.",
    "group-by": "Group consecutive elements by key function result.",
    "distinct-by": "Remove duplicates by key function, preserving first occurrence.",
}

SKELETONS = {fn: f"""(define ({fn} {' '.join(['arg'+str(i) for i in range(1, 4)])})\n  ;; TODO: implement {fn}\n  <TODO>)""" for fn in FUNCTION_ORDER}
# Replace with per-function readable signatures.
SKELETONS.update({
    "andmap": """(define (andmap pred lst)
  ;; TODO: #t iff every element satisfies pred
  <TODO>)""",
    "ormap": """(define (ormap pred lst)
  ;; TODO: first truthy predicate result, or #f
  <TODO>)""",
    "filter": """(define (filter pred lst)
  ;; TODO: keep only matching elements
  <TODO>)""",
    "filter-map": """(define (filter-map f lst)
  ;; TODO: map f and keep non-#f results
  <TODO>)""",
    "fold-left": """(define (fold-left f acc lst)
  ;; TODO: left-associative fold
  <TODO>)""",
    "fold-right": """(define (fold-right f acc lst)
  ;; TODO: right-associative fold
  <TODO>)""",
    "zip": """(define (zip xs ys)
  ;; TODO: zip into pairs
  <TODO>)""",
    "iota": """(define (iota n)
  ;; TODO: generate 0..n-1
  <TODO>)""",
    "range": """(define (range start end)
  ;; TODO: generate [start, end)
  <TODO>)""",
    "take": """(define (take n lst)
  ;; TODO: take first n elements
  <TODO>)""",
    "drop": """(define (drop n lst)
  ;; TODO: drop first n elements
  <TODO>)""",
    "find": """(define (find pred lst)
  ;; TODO: return first match or #f
  <TODO>)""",
    "last": """(define (last lst)
  ;; TODO: return last element
  <TODO>)""",
    "init": """(define (init lst)
  ;; TODO: all except final element
  <TODO>)""",
    "replicate": """(define (replicate n x)
  ;; TODO: list of n copies of x
  <TODO>)""",
    "span": """(define (span pred lst)
  ;; TODO: return (values prefix suffix)
  <TODO>)""",
    "break": """(define (break pred lst)
  ;; TODO: split at first satisfying element
  <TODO>)""",
    "sum": """(define (sum lst)
  ;; TODO: sum numeric list
  <TODO>)""",
    "product": """(define (product lst)
  ;; TODO: multiply numeric list
  <TODO>)""",
    "mean": """(define (mean lst)
  ;; TODO: arithmetic mean (error on empty)
  <TODO>)""",
    "identity": """(define (identity x)
  ;; TODO: return x
  <TODO>)""",
    "flatten": """(define (flatten lst-of-lists)
  ;; TODO: flatten list-of-lists
  <TODO>)""",
    "append-map": """(define (append-map f lst)
  ;; TODO: map to lists then append
  <TODO>)""",
    "partition": """(define (partition pred lst)
  ;; TODO: return (list yes no)
  <TODO>)""",
    "group-by": """(define (group-by key-fn lst)
  ;; TODO: group consecutive elements by key
  <TODO>)""",
    "distinct-by": """(define (distinct-by key-fn lst)
  ;; TODO: remove duplicate keys, keep first
  <TODO>)""",
})

VERIFY_BY_FUNCTION = {
    "andmap": "(and (andmap number? '()) (andmap number? '(1 2 3)) (not (andmap number? '(1 \"x\" 3))))",
    "ormap": "(and (not (ormap number? '())) (equal? (ormap (lambda (x) (and (number? x) x)) '(\"a\" 2 \"c\")) 2) (not (ormap number? '(\"a\" \"b\"))))",
    "filter": "(equal? (filter even? '(1 2 3 4 5)) '(2 4))",
    "filter-map": "(equal? (filter-map (lambda (x) (if (> x 0) (* 2 x) #f)) '(-2 -1 0 1 2)) '(2 4))",
    "fold-left": "(and (= (fold-left + 0 '(1 2 3 4)) 10) (equal? (fold-left (lambda (acc x) (list acc x)) 0 '(1 2)) '((0 1) 2)))",
    "fold-right": "(and (= (fold-right + 0 '(1 2 3 4)) 10) (equal? (fold-right (lambda (x acc) (list x acc)) 0 '(1 2)) '(1 (2 0))))",
    "zip": "(equal? (zip '(1 2 3) '(a b)) '((1 . a) (2 . b)))",
    "iota": "(and (equal? (iota 0) '()) (equal? (iota 5) '(0 1 2 3 4)))",
    "range": "(and (equal? (range 2 2) '()) (equal? (range 2 6) '(2 3 4 5)))",
    "take": "(equal? (take 3 '(a b c d e)) '(a b c))",
    "drop": "(equal? (drop 3 '(a b c d e)) '(d e))",
    "find": "(and (equal? (find even? '(1 3 4 6)) 4) (not (find even? '(1 3 5))))",
    "last": "(equal? (last '(x y z)) 'z)",
    "init": "(equal? (init '(x y z)) '(x y))",
    "replicate": "(and (equal? (replicate 3 'q) '(q q q)) (equal? (replicate 0 'q) '()))",
    "span": "(let-values ([(pre suf) (span (lambda (x) (< x 3)) '(1 2 3 4))]) (and (equal? pre '(1 2)) (equal? suf '(3 4))))",
    "break": "(let-values ([(pre suf) (break even? '(1 3 5 2 4))]) (and (equal? pre '(1 3 5)) (equal? suf '(2 4))))",
    "sum": "(and (= (sum '()) 0) (= (sum '(1 2 3 4)) 10))",
    "product": "(and (= (product '()) 1) (= (product '(2 3 4)) 24))",
    "mean": "(and (equal? (mean '(2 4 6 8)) 5) (guard (ex [else #t]) (begin (mean '()) #f)))",
    "identity": "(and (= (identity 42) 42) (equal? (identity '(a b)) '(a b)))",
    "flatten": "(equal? (flatten '((1 2) () (3 4))) '(1 2 3 4))",
    "append-map": "(equal? (append-map (lambda (x) (list x x)) '(1 2 3)) '(1 1 2 2 3 3))",
    "partition": "(equal? (partition even? '(1 2 3 4 5)) '((2 4) (1 3 5)))",
    "group-by": "(equal? (group-by string-length '(\"a\" \"b\" \"cc\" \"dd\")) '((1 \"a\" \"b\") (2 \"cc\" \"dd\")))",
    "distinct-by": "(equal? (distinct-by (lambda (s) (string-ref s 0)) '(\"apple\" \"apricot\" \"banana\" \"blueberry\")) '(\"apple\" \"banana\"))",
}

PYTHON_SNIPPETS = {
    "andmap": "def andmap(pred, xs):\n    return all(pred(x) for x in xs)",
    "ormap": "def ormap(pred, xs):\n    for x in xs:\n        v = pred(x)\n        if v:\n            return v\n    return False",
    "filter": "def filter_(pred, xs):\n    return [x for x in xs if pred(x)]",
    "filter-map": "def filter_map(f, xs):\n    out=[]\n    for x in xs:\n        v=f(x)\n        if v is not None:\n            out.append(v)\n    return out",
    "fold-left": "def fold_left(f, acc, xs):\n    for x in xs:\n        acc=f(acc,x)\n    return acc",
    "fold-right": "def fold_right(f, acc, xs):\n    for x in reversed(xs):\n        acc=f(x,acc)\n    return acc",
    "zip": "def zip_(xs, ys):\n    return list(zip(xs, ys))",
    "iota": "def iota(n):\n    return list(range(0, n))",
    "range": "def range_(start, end):\n    return list(range(start, end))",
    "take": "def take(n, xs):\n    return xs[:n]",
    "drop": "def drop(n, xs):\n    return xs[n:]",
    "find": "def find(pred, xs):\n    for x in xs:\n        if pred(x):\n            return x\n    return None",
    "last": "def last(xs):\n    return xs[-1]",
    "init": "def init(xs):\n    return xs[:-1]",
    "replicate": "def replicate(n, x):\n    return [x]*max(n,0)",
    "span": "def span(pred, xs):\n    i=0\n    while i < len(xs) and pred(xs[i]):\n        i+=1\n    return xs[:i], xs[i:]",
    "break": "def break_(pred, xs):\n    i=0\n    while i < len(xs) and not pred(xs[i]):\n        i+=1\n    return xs[:i], xs[i:]",
    "sum": "def sum_(xs):\n    return sum(xs)",
    "product": "def product(xs):\n    out=1\n    for x in xs:\n        out*=x\n    return out",
    "mean": "def mean(xs):\n    if len(xs)==0:\n        raise ValueError('empty list')\n    return sum(xs)/len(xs)",
    "identity": "def identity(x):\n    return x",
    "flatten": "def flatten(xss):\n    return [x for xs in xss for x in xs]",
    "append-map": "def append_map(f, xs):\n    out=[]\n    for x in xs:\n        out.extend(f(x))\n    return out",
    "partition": "def partition(pred, xs):\n    yes=[]; no=[]\n    for x in xs:\n        (yes if pred(x) else no).append(x)\n    return [yes,no]",
    "group-by": "def group_by(key_fn, xs):\n    if not xs:\n        return []\n    out=[]\n    cur_k=key_fn(xs[0]); cur=[]\n    for x in xs:\n        k=key_fn(x)\n        if k==cur_k:\n            cur.append(x)\n        else:\n            out.append((cur_k,cur)); cur_k=k; cur=[x]\n    out.append((cur_k,cur))\n    return out",
    "distinct-by": "def distinct_by(key_fn, xs):\n    seen=[]; out=[]\n    for x in xs:\n        k=key_fn(x)\n        if k not in seen:\n            seen.append(k); out.append(x)\n    return out",
}

BUGGY_CASES = [
    {"fn": "andmap", "buggy": "(define (andmap pred lst)\n  (or (null? lst)\n      (or (pred (car lst)) (andmap pred (cdr lst)))))", "note": "All elements must satisfy predicate, not just one."},
    {"fn": "ormap", "buggy": "(define (ormap pred lst)\n  (or (null? lst)\n      (and (pred (car lst)) (ormap pred (cdr lst)))))", "note": "Should return first truthy predicate result; empty list should return #f."},
    {"fn": "filter", "buggy": "(define (filter pred lst)\n  (cond [(null? lst) '()]\n        [(pred (car lst)) (filter pred (cdr lst))]\n        [else (cons (car lst) (filter pred (cdr lst)))]))", "note": "Predicate logic is inverted."},
    {"fn": "filter-map", "buggy": "(define (filter-map f lst)\n  (map f lst))", "note": "Must remove #f results."},
    {"fn": "fold-left", "buggy": "(define (fold-left f acc lst)\n  (if (null? lst) acc (f (car lst) (fold-left f acc (cdr lst)))))", "note": "This is right-associative, not left-associative."},
    {"fn": "fold-right", "buggy": "(define (fold-right f acc lst)\n  (if (null? lst) acc (fold-right f (f (car lst) acc) (cdr lst))))", "note": "This behaves like fold-left."},
    {"fn": "zip", "buggy": "(define (zip xs ys)\n  (map list xs ys))", "note": "Expected pairs `(a . b)`, not two-element lists."},
    {"fn": "iota", "buggy": "(define (iota n)\n  (let loop ([i 1] [acc '()]) (if (> i n) (reverse acc) (loop (+ i 1) (cons i acc)))))", "note": "Sequence must start at 0 and end at n-1."},
    {"fn": "range", "buggy": "(define (range start end)\n  (if (> start end) '() (cons start (range (+ start 1) end))))", "note": "End should be exclusive."},
    {"fn": "take", "buggy": "(define (take n lst)\n  (if (or (< n 0) (null? lst)) '() (cons (car lst) (take (- n 1) (cdr lst)))))", "note": "When n is 0, result must be empty."},
    {"fn": "drop", "buggy": "(define (drop n lst)\n  (take n lst))", "note": "Drop and take are not the same."},
    {"fn": "find", "buggy": "(define (find pred lst)\n  (ormap pred lst))", "note": "Find returns matching element, not boolean."},
    {"fn": "last", "buggy": "(define (last lst)\n  (car lst))", "note": "Must return final element."},
    {"fn": "init", "buggy": "(define (init lst)\n  lst)", "note": "Must remove final element."},
    {"fn": "replicate", "buggy": "(define (replicate n x)\n  (if (< n 0) '() (cons x (replicate (- n 1) x))))", "note": "For n=0 result should be empty, not one element."},
    {"fn": "span", "buggy": "(define (span pred lst)\n  (values lst '()))", "note": "Must split at first failure of predicate."},
    {"fn": "break", "buggy": "(define (break pred lst)\n  (span pred lst))", "note": "Break should negate predicate before delegating to span."},
    {"fn": "sum", "buggy": "(define (sum lst)\n  (fold-left * 1 lst))", "note": "Sum should add, not multiply."},
    {"fn": "product", "buggy": "(define (product lst)\n  (fold-left + 0 lst))", "note": "Product should multiply, not add."},
    {"fn": "mean", "buggy": "(define (mean lst)\n  (/ (sum lst) (length lst)))", "note": "Must error on empty list."},
    {"fn": "identity", "buggy": "(define (identity x)\n  #f)", "note": "Identity must return input unchanged."},
    {"fn": "flatten", "buggy": "(define (flatten lst-of-lists)\n  (if (null? lst-of-lists) '() (car lst-of-lists)))", "note": "Must concatenate all inner lists."},
    {"fn": "append-map", "buggy": "(define (append-map f lst)\n  (map f lst))", "note": "Must append mapped lists into one list."},
    {"fn": "partition", "buggy": "(define (partition pred lst)\n  (list (filter (lambda (x) (not (pred x))) lst) (filter pred lst)))", "note": "Output order is `(yes no)`.",},
    {"fn": "group-by", "buggy": "(define (group-by key-fn lst)\n  '())", "note": "Must produce grouped key/list pairs."},
    {"fn": "distinct-by", "buggy": "(define (distinct-by key-fn lst)\n  (reverse lst))", "note": "Must deduplicate by computed key while preserving first occurrences."},
]

DIFFICULTY = {
    "andmap": "easy",
    "ormap": "easy",
    "filter": "easy",
    "filter-map": "medium",
    "fold-left": "medium",
    "fold-right": "medium",
    "zip": "easy",
    "iota": "easy",
    "range": "easy",
    "take": "easy",
    "drop": "easy",
    "find": "easy",
    "last": "easy",
    "init": "easy",
    "replicate": "easy",
    "span": "medium",
    "break": "medium",
    "sum": "easy",
    "product": "easy",
    "mean": "medium",
    "identity": "easy",
    "flatten": "medium",
    "append-map": "medium",
    "partition": "medium",
    "group-by": "hard",
    "distinct-by": "hard",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")
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
    sid = f"core_prelude_{family}_{family_counter[family]:03d}"
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


def def_verify(fn: str) -> str:
    parts = [DEFS[d] for d in dependency_closure(fn)] + [DEFS[fn], VERIFY_BY_FUNCTION[fn]]
    body = "\n  ".join(parts)
    return f"(let ()\n  {body})"


# -----------------------------------------------------------------------------
# Family 1: spec_to_code (52)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""You are implementing core prelude utilities in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["core", "base", "prelude", "spec-to-code", fn],
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
        tags=["core", "base", "prelude", "skeleton-completion", fn],
    )


# -----------------------------------------------------------------------------
# Family 2: translation (26)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
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
        tags=["core", "base", "prelude", "python-to-scheme", fn],
    )


# -----------------------------------------------------------------------------
# Family 3: bugfix (26)
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
        tags=["core", "base", "prelude", "bugfix", fn],
    )


# -----------------------------------------------------------------------------
# Family 4: composition (30)
# -----------------------------------------------------------------------------

def add_composition(source_function: str, prompt: str, ground_truth: str, verify_expr: str, difficulty: str, extra_tags: List[str]) -> None:
    add_sample(
        family="composition",
        category="usage",
        difficulty=difficulty,
        source_function=source_function,
        prompt=prompt,
        ground_truth=ground_truth,
        verify_expr=verify_expr,
        tags=["core", "base", "prelude", "composition", source_function] + extra_tags,
    )


composition_cases = [
    ("andmap", "Return whether all elements in `(1 2 3)` are numbers.", "(andmap number? '(1 2 3))", "(equal? (andmap number? '(1 2 3)) #t)", "easy", ["direct"]),
    ("ormap", "Return whether any element in `(\"a\" 2 \"c\")` is a number.", "(ormap number? '(\"a\" 2 \"c\"))", "(equal? (ormap number? '(\"a\" 2 \"c\")) #t)", "easy", ["direct"]),
    ("filter", "Keep evens from `(1 2 3 4 5 6)`.", "(filter even? '(1 2 3 4 5 6))", "(equal? (filter even? '(1 2 3 4 5 6)) '(2 4 6))", "easy", ["direct"]),
    ("filter-map", "Double positive numbers and drop others from `(-1 0 2 3)`.", "(filter-map (lambda (x) (if (> x 0) (* 2 x) #f)) '(-1 0 2 3))", "(equal? (filter-map (lambda (x) (if (> x 0) (* 2 x) #f)) '(-1 0 2 3)) '(4 6))", "medium", ["direct"]),
    ("fold-left", "Compute sum of `(1 2 3 4)` using fold-left.", "(fold-left + 0 '(1 2 3 4))", "(equal? (fold-left + 0 '(1 2 3 4)) 10)", "easy", ["direct"]),
    ("fold-right", "Build nested pair form with fold-right over `(1 2)`.", "(fold-right (lambda (x acc) (list x acc)) 0 '(1 2))", "(equal? (fold-right (lambda (x acc) (list x acc)) 0 '(1 2)) '(1 (2 0)))", "medium", ["direct"]),
    ("zip", "Zip `(1 2 3)` and `(a b)`.", "(zip '(1 2 3) '(a b))", "(equal? (zip '(1 2 3) '(a b)) '((1 . a) (2 . b)))", "easy", ["direct"]),
    ("iota", "Generate `iota` for 5.", "(iota 5)", "(equal? (iota 5) '(0 1 2 3 4))", "easy", ["direct"]),
    ("range", "Generate range from 3 to 7.", "(range 3 7)", "(equal? (range 3 7) '(3 4 5 6))", "easy", ["direct"]),
    ("take", "Take 3 elements from `(a b c d e)`.", "(take 3 '(a b c d e))", "(equal? (take 3 '(a b c d e)) '(a b c))", "easy", ["direct"]),
    ("drop", "Drop 3 elements from `(a b c d e)`.", "(drop 3 '(a b c d e))", "(equal? (drop 3 '(a b c d e)) '(d e))", "easy", ["direct"]),
    ("find", "Find first even value in `(1 3 4 6)`.", "(find even? '(1 3 4 6))", "(equal? (find even? '(1 3 4 6)) 4)", "easy", ["direct"]),
    ("last", "Return last element of `(x y z)`.", "(last '(x y z))", "(equal? (last '(x y z)) 'z)", "easy", ["direct"]),
    ("init", "Return init of `(x y z)`.", "(init '(x y z))", "(equal? (init '(x y z)) '(x y))", "easy", ["direct"]),
    ("replicate", "Replicate symbol `q` three times.", "(replicate 3 'q)", "(equal? (replicate 3 'q) '(q q q))", "easy", ["direct"]),
    ("span", "Split `(1 2 3 4)` with predicate `<3` using span.", "(let-values ([(pre suf) (span (lambda (x) (< x 3)) '(1 2 3 4))]) (list pre suf))", "(equal? (let-values ([(pre suf) (span (lambda (x) (< x 3)) '(1 2 3 4))]) (list pre suf)) '((1 2) (3 4)))", "medium", ["direct"]),
    ("break", "Split `(1 3 5 2 4)` at first even using break.", "(let-values ([(pre suf) (break even? '(1 3 5 2 4))]) (list pre suf))", "(equal? (let-values ([(pre suf) (break even? '(1 3 5 2 4))]) (list pre suf)) '((1 3 5) (2 4)))", "medium", ["direct"]),
    ("sum", "Sum values `(10 20 30)`.", "(sum '(10 20 30))", "(equal? (sum '(10 20 30)) 60)", "easy", ["direct"]),
    ("product", "Product of `(2 3 4)`.", "(product '(2 3 4))", "(equal? (product '(2 3 4)) 24)", "easy", ["direct"]),
    ("mean", "Mean of `(2 4 6 8)`.", "(mean '(2 4 6 8))", "(equal? (mean '(2 4 6 8)) 5)", "medium", ["direct"]),
    ("identity", "Apply identity to list `(a b)`.", "(identity '(a b))", "(equal? (identity '(a b)) '(a b))", "easy", ["direct"]),
    ("flatten", "Flatten list-of-lists `((1 2) () (3 4))`.", "(flatten '((1 2) () (3 4)))", "(equal? (flatten '((1 2) () (3 4))) '(1 2 3 4))", "medium", ["direct"]),
    ("append-map", "Duplicate each element with append-map over `(1 2 3)`.", "(append-map (lambda (x) (list x x)) '(1 2 3))", "(equal? (append-map (lambda (x) (list x x)) '(1 2 3)) '(1 1 2 2 3 3))", "medium", ["direct"]),
    ("partition", "Partition `(1 2 3 4 5)` by even?.", "(partition even? '(1 2 3 4 5))", "(equal? (partition even? '(1 2 3 4 5)) '((2 4) (1 3 5)))", "medium", ["direct"]),
    ("group-by", "Group strings by length in `(\"a\" \"b\" \"cc\" \"dd\")`.", "(group-by string-length '(\"a\" \"b\" \"cc\" \"dd\"))", "(equal? (group-by string-length '(\"a\" \"b\" \"cc\" \"dd\")) '((1 \"a\" \"b\") (2 \"cc\" \"dd\")))", "hard", ["direct"]),
    ("distinct-by", "Keep first string per initial letter.", "(distinct-by (lambda (s) (string-ref s 0)) '(\"apple\" \"apricot\" \"banana\" \"blueberry\"))", "(equal? (distinct-by (lambda (s) (string-ref s 0)) '(\"apple\" \"apricot\" \"banana\" \"blueberry\")) '(\"apple\" \"banana\"))", "hard", ["direct"]),
    ("sum", "Return #t iff `sum (iota 6)` equals 15.", "(= (sum (iota 6)) 15)", "(equal? (= (sum (iota 6)) 15) #t)", "medium", ["integration"]),
    ("fold-left", "Build reversed list via fold-left and cons.", "(fold-left (lambda (acc x) (cons x acc)) '() '(1 2 3 4))", "(equal? (fold-left (lambda (acc x) (cons x acc)) '() '(1 2 3 4)) '(4 3 2 1))", "medium", ["integration"]),
    ("break", "Return #t iff `break` is the opposite split of `span` for predicate even? on `(1 3 5 2 4)`.", "(let-values ([(a b) (break even? '(1 3 5 2 4))]) (and (equal? a '(1 3 5)) (equal? b '(2 4))))", "(equal? (let-values ([(a b) (break even? '(1 3 5 2 4))]) (and (equal? a '(1 3 5)) (equal? b '(2 4)))) #t)", "medium", ["property"]),
    ("partition", "Count partition sizes for `(1 2 3 4 5 6)` by even?.", "(let* ([p (partition even? '(1 2 3 4 5 6))] [yes (car p)] [no (cadr p)]) (list (length yes) (length no)))", "(equal? (let* ([p (partition even? '(1 2 3 4 5 6))] [yes (car p)] [no (cadr p)]) (list (length yes) (length no))) '(3 3))", "medium", ["integration"]),
]

for fn, prompt, gt, verify, diff, tags in composition_cases:
    add_composition(fn, prompt, gt, verify, diff, tags)

if len([s for s in samples if s["family"] == "spec_to_code"]) != 52:
    raise ValueError("spec_to_code family must contain exactly 52 samples")
if len([s for s in samples if s["family"] == "translation"]) != 26:
    raise ValueError("translation family must contain exactly 26 samples")
if len([s for s in samples if s["family"] == "bugfix"]) != 26:
    raise ValueError("bugfix family must contain exactly 26 samples")
if len([s for s in samples if s["family"] == "composition"]) != 30:
    raise ValueError("composition family must contain exactly 30 samples")
if len(samples) != 134:
    raise ValueError(f"expected 134 samples, got {len(samples)}")


# -----------------------------------------------------------------------------
# Split
# -----------------------------------------------------------------------------
by_family: Dict[str, List[Dict[str, object]]] = defaultdict(list)
for s in samples:
    by_family[str(s["family"])].append(s)

EVAL_QUOTA = {
    "spec_to_code": 10,
    "translation": 6,
    "bugfix": 6,
    "composition": 8,
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

if len(train_rows) != 104 or len(eval_rows) != 30:
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
}

with SUMMARY_PATH.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print(f"Wrote: {ALL_PATH}")
print(f"Wrote: {TRAIN_PATH}")
print(f"Wrote: {EVAL_PATH}")
