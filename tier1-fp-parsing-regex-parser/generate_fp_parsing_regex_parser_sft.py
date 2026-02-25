#!/usr/bin/env python3
"""Generate Tier-1 FP parsing regex-parser SFT samples for lattice/fp/parsing/regex.ss."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

OUT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(__file__).resolve().parents[1]
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from sft_prompt_diversity import diversify_prompt
from sft_split_utils import compute_leakage_aware_eval_ids

ALL_PATH = OUT_DIR / "all.jsonl"
TRAIN_PATH = OUT_DIR / "train.jsonl"
EVAL_PATH = OUT_DIR / "eval.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.json"

SOURCE_MODULE = "lattice/fp/parsing/regex.ss"
SOURCE_TEST = "lattice/fp/parsing/test-regex.ss"

DEFS: Dict[str, str] = {
    "parse-class-range": """(define (parse-class-range)
  (doc 'type '(Parser (List Char)))
  (parser-bind
   (parse-class-char)
   (lambda (start)
     (parser-bind
      (parser-char #\\-)
      (lambda (_)
        (parser-bind
         (parse-class-char)
         (lambda (end)
           (if (char<=? start end)
               (let* ([start-int (char->integer start)]
                      [count (+ 1 (- (char->integer end) start-int))])
                 (parser-pure (map (lambda (i) (integer->char (+ i start-int)))
                                   (iota count))))
               (parser-fail "invalid range: start > end")))))))))""",
    "parse-interval": """(define (parse-interval)
  (doc 'type '(Parser (Pair Nat (Option Nat))))
  (doc 'description "Parse interval quantifier, returns (min . max) where max=#f means unbounded")
  (parser-bind
   (parser-char #\\{)
   (lambda (_)
     (parser-bind
      (parser-option-maybe parser-natural)
      (lambda (min-maybe)
        (parser-or
         (parser-bind
          (parser-char #\\})
          (lambda (_)
            (if (nothing? min-maybe)
                (parser-fail "empty interval {} is invalid")
                (let ([n (from-just min-maybe)])
                  (parser-pure (cons n n))))))
         (parser-bind
          (parser-char #\\,)
          (lambda (_)
            (parser-bind
             (parser-option-maybe parser-natural)
             (lambda (max-maybe)
               (parser-bind
                (parser-char #\\})
                (lambda (_)
                  (let ([min (if (nothing? min-maybe) 0 (from-just min-maybe))]
                        [max (if (nothing? max-maybe) #f (from-just max-maybe))])
                    (if (and max (> min max))
                        (parser-fail "interval min > max")
                        (parser-pure (cons min max))))))))))))))))""",
    "apply-postfix-op": """(define (apply-postfix-op expr op)
  (cond
   [(char? op)
    (cond
     [(char=? op #\\*) (regex-star expr)]
     [(char=? op #\\+) (regex-plus expr)]
     [(char=? op #\\?) (regex-opt expr)])]
   [(pair? op)
    (regex-repeat expr (car op) (cdr op))]))""",
    "parse-seq": """(define (parse-seq)
  (doc 'type '(Parser RegexAST))
  (parser-bind
   (parser-many (parse-postfix))
   (lambda (exprs)
     (parser-pure
      (cond
       [(null? exprs) (regex-empty)]
       [(null? (cdr exprs)) (car exprs)]
       [else (regex-seq exprs)])))))""",
    "parse-alt": """(define (parse-alt)
  (doc 'type '(Parser RegexAST))
  (parser-bind
   (parse-seq)
   (lambda (first)
     (parser-bind
      (parser-many (parser-then (parser-char #\\|) (parse-seq)))
      (lambda (rest)
        (parser-pure
         (if (null? rest)
             first
             (regex-alt (cons first rest)))))))))""",
    "compile-repeat": """(define (compile-repeat expr min max universe)
  (doc 'type '(-> RegexAST Nat (Option Nat) (List Char) FSM))
  (let ([base (regex-compile expr universe)])
    (cond
     [(and (= min 0) (eqv? max 0))
      (fsm-epsilon-lang)]
     [(and (= min 0) (not max))
      (fsm-star base)]
     [(= min 0)
      (fold-left (lambda (acc _)
                   (fsm-optional (fsm-concat base acc)))
                 (fsm-epsilon-lang)
                 (iota max))]
     [(eqv? min max)
      (fold-left fsm-concat base
                 (map (lambda (_) (regex-compile expr universe))
                      (iota (- min 1))))]
     [(not max)
      (let ([required (fold-left fsm-concat base
                                 (map (lambda (_) (regex-compile expr universe))
                                      (iota (- min 1))))])
        (fsm-concat required (fsm-star base)))]
     [else
      (let* ([required (fold-left fsm-concat base
                                  (map (lambda (_) (regex-compile expr universe))
                                       (iota (- min 1))))]
             [optional (fold-left (lambda (acc _)
                                    (fsm-concat (fsm-optional base) acc))
                                  (fsm-epsilon-lang)
                                  (iota (- max min)))])
        (fsm-concat required optional))])))""",
    "compile-anchor": """(define (compile-anchor type)
  (doc 'type '(-> Symbol FSM))
  (let ([s0 (fsm-fresh-state "anc")]
        [s1 (fsm-fresh-state "anc")])
    (make-fsm-with-assertions
     (list s0 s1) '() '() s0 (list s1) '()
     (list (list s0 'anchor type s1)))))""",
    "compile-lookahead": """(define (compile-lookahead expr positive? universe)
  (doc 'type '(-> RegexAST Boolean (List Char) FSM))
  (let ([inner (regex-compile expr universe)]
        [s0 (fsm-fresh-state "la")]
        [s1 (fsm-fresh-state "la")])
    (make-fsm-with-assertions
     (list s0 s1) '() '() s0 (list s1) '()
     (list (list s0 'lookahead inner positive? s1)))))""",
}

FUNCTION_ORDER = [
    "parse-class-range",
    "parse-interval",
    "apply-postfix-op",
    "parse-seq",
    "parse-alt",
    "compile-repeat",
    "compile-anchor",
    "compile-lookahead",
]

FUNCTION_SPECS = {
    "parse-class-range": "Parse and expand a character range like a-z into an inclusive character list; reject reversed ranges.",
    "parse-interval": "Parse interval quantifiers {n}, {n,m}, {n,}, {,m}; reject empty {} and invalid min>max.",
    "apply-postfix-op": "Apply regex postfix operator (*,+,?,interval-pair) to an AST node.",
    "parse-seq": "Parse concatenation of postfix terms; return regex-empty for none, single node for one, regex-seq for many.",
    "parse-alt": "Parse alternations separated by | with left term plus zero-or-more right terms.",
    "compile-repeat": "Compile bounded/unbounded repetition to an FSM matching regex quantifier semantics.",
    "compile-anchor": "Compile ^/$ anchor into assertion FSM with zero-width transition metadata.",
    "compile-lookahead": "Compile positive/negative lookahead into assertion FSM preserving inner compiled pattern and polarity.",
}

SKELETONS = {
    "parse-class-range": """(define (parse-class-range)
  ;; TODO: parse start-end class range and expand inclusive chars
  <TODO>)""",
    "parse-interval": """(define (parse-interval)
  ;; TODO: parse {n}, {n,m}, {n,}, {,m} with validation
  <TODO>)""",
    "apply-postfix-op": """(define (apply-postfix-op expr op)
  ;; TODO: map char operators and interval pairs to regex AST wrappers
  <TODO>)""",
    "parse-seq": """(define (parse-seq)
  ;; TODO: parse zero-or-more postfix items and normalize empty/singleton/many
  <TODO>)""",
    "parse-alt": """(define (parse-alt)
  ;; TODO: parse seq and trailing |seq clauses, building regex-alt when needed
  <TODO>)""",
    "compile-repeat": """(define (compile-repeat expr min max universe)
  ;; TODO: compile repeat bounds into equivalent FSM composition
  <TODO>)""",
    "compile-anchor": """(define (compile-anchor type)
  ;; TODO: emit assertion FSM for anchor type
  <TODO>)""",
    "compile-lookahead": """(define (compile-lookahead expr positive? universe)
  ;; TODO: emit assertion FSM for lookahead polarity and inner pattern
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "parse-class-range": "(let ([ok (parser-parse (parser-left (parse-class-range) parser-eof) \"a-z\")] [bad (parser-parse (parser-left (parse-class-range) parser-eof) \"z-a\")]) (and (right? ok) (= (length (from-right ok)) 26) (equal? (car (from-right ok)) #\\a) (equal? (car (reverse (from-right ok))) #\\z) (left? bad)))",
    "parse-interval": "(let ([e1 (parser-parse (parser-left (parse-interval) parser-eof) \"{3}\")] [e2 (parser-parse (parser-left (parse-interval) parser-eof) \"{2,}\")] [e3 (parser-parse (parser-left (parse-interval) parser-eof) \"{,4}\")] [bad1 (parser-parse (parser-left (parse-interval) parser-eof) \"{}\")] [bad2 (parser-parse (parser-left (parse-interval) parser-eof) \"{4,2}\")]) (and (right? e1) (equal? (from-right e1) (cons 3 3)) (right? e2) (equal? (from-right e2) (cons 2 #f)) (right? e3) (equal? (from-right e3) (cons 0 4)) (left? bad1) (left? bad2)))",
    "apply-postfix-op": "(let* ([lit (regex-lit #\\a)] [s (apply-postfix-op lit #\\*)] [p (apply-postfix-op lit #\\+)] [o (apply-postfix-op lit #\\?)] [r (apply-postfix-op lit (cons 2 4))]) (and (regex-star? s) (regex-plus? p) (regex-opt? o) (regex-repeat? r) (= (regex-repeat-min r) 2) (= (regex-repeat-max r) 4)))",
    "parse-seq": "(let ([e0 (parser-parse (parser-left (parse-seq) parser-eof) \"\")] [e1 (parser-parse (parser-left (parse-seq) parser-eof) \"a\")] [e2 (parser-parse (parser-left (parse-seq) parser-eof) \"ab\")]) (and (right? e0) (regex-empty? (from-right e0)) (right? e1) (regex-lit? (from-right e1)) (right? e2) (regex-seq? (from-right e2)) (= (length (regex-seq-exprs (from-right e2))) 2)))",
    "parse-alt": "(let* ([a (parser-parse (parser-left (parse-alt) parser-eof) \"a|b|c\")] [b (parser-parse (parser-left (parse-alt) parser-eof) \"ab\")] [c (parser-parse (parser-left (parse-alt) parser-eof) \"a|bc\")]) (and (right? a) (regex-alt? (from-right a)) (= (length (regex-alt-exprs (from-right a))) 3) (right? b) (regex-seq? (from-right b)) (right? c) (regex-alt? (from-right c)) (let ([alts (regex-alt-exprs (from-right c))]) (and (= (length alts) 2) (regex-seq? (cadr alts))))))",
    "compile-repeat": "(let* ([u '(#\\a #\\b)] [z (compile-repeat (regex-lit #\\a) 0 0 u)] [r24 (compile-repeat (regex-lit #\\a) 2 4 u)] [r2p (compile-repeat (regex-lit #\\a) 2 #f u)] [r02 (compile-repeat (regex-lit #\\a) 0 2 u)]) (and (fsm-accepts? z \"\") (not (fsm-accepts? z \"a\")) (fsm-accepts? r24 \"aa\") (fsm-accepts? r24 \"aaaa\") (not (fsm-accepts? r24 \"a\")) (not (fsm-accepts? r24 \"aaaaa\")) (fsm-accepts? r2p \"aa\") (fsm-accepts? r2p \"aaaaa\") (not (fsm-accepts? r2p \"a\")) (fsm-accepts? r02 \"\") (fsm-accepts? r02 \"aa\") (not (fsm-accepts? r02 \"aaa\"))))",
    "compile-anchor": "(let* ([s (compile-anchor 'start)] [e (compile-anchor 'end)] [sas (fsm-assertions s)] [eas (fsm-assertions e)]) (and (pair? sas) (pair? eas) (fsm-accepts? s \"\") (fsm-accepts? e \"\") (not (fsm-accepts? s \"x\")) (not (fsm-accepts? e \"x\")) (equal? (cadr (car sas)) 'anchor) (equal? (caddr (car sas)) 'start) (equal? (caddr (car eas)) 'end)))",
    "compile-lookahead": "(let* ([u '(#\\a #\\b #\\c)] [pos (fsm-concat (compile-lookahead (regex-lit #\\a) #t u) (fsm-char #\\a))] [neg (fsm-concat (compile-lookahead (regex-lit #\\b) #f u) (fsm-char #\\a))] [pa (car (fsm-assertions (compile-lookahead (regex-lit #\\a) #t u)))] [na (car (fsm-assertions (compile-lookahead (regex-lit #\\b) #f u)))]) (and (fsm-accepts? pos \"a\") (not (fsm-accepts? pos \"b\")) (fsm-accepts? neg \"a\") (not (fsm-accepts? neg \"b\")) (equal? (cadr pa) 'lookahead) (equal? (cadddr pa) #t) (equal? (cadddr na) #f)))",
}

TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "parse-class-range": """def parse_class_range():
    start = parse_class_char()
    expect('-')
    end = parse_class_char()
    if start <= end:
        return [chr(ord(start) + i) for i in range(ord(end) - ord(start) + 1)]
    raise ParseError('invalid range: start > end')""",
    "parse-interval": """def parse_interval():
    expect('{')
    min_v = maybe_parse_nat()
    if peek() == '}':
        expect('}')
        if min_v is None:
            raise ParseError('empty interval {} is invalid')
        return (min_v, min_v)
    expect(',')
    max_v = maybe_parse_nat()
    expect('}')
    min_out = 0 if min_v is None else min_v
    max_out = max_v
    if max_out is not None and min_out > max_out:
        raise ParseError('interval min > max')
    return (min_out, max_out)""",
    "apply-postfix-op": """def apply_postfix_op(expr, op):
    if isinstance(op, str):
        if op == '*':
            return regex_star(expr)
        if op == '+':
            return regex_plus(expr)
        if op == '?':
            return regex_opt(expr)
    if isinstance(op, tuple):
        mn, mx = op
        return regex_repeat(expr, mn, mx)""",
    "parse-seq": """def parse_seq():
    exprs = many(parse_postfix)
    if not exprs:
        return regex_empty()
    if len(exprs) == 1:
        return exprs[0]
    return regex_seq(exprs)""",
    "parse-alt": """def parse_alt():
    first = parse_seq()
    rest = many(lambda: then(char('|'), parse_seq()))
    if not rest:
        return first
    return regex_alt([first] + rest)""",
    "compile-repeat": """def compile_repeat(expr, mn, mx, universe):
    base = regex_compile(expr, universe)
    if mn == 0 and mx == 0:
        return fsm_epsilon_lang()
    if mn == 0 and mx is None:
        return fsm_star(base)
    if mn == 0:
        acc = fsm_epsilon_lang()
        for _ in range(mx):
            acc = fsm_optional(fsm_concat(base, acc))
        return acc
    if mx == mn:
        return concat_n(base, mn)
    if mx is None:
        return fsm_concat(concat_n(base, mn), fsm_star(base))
    return fsm_concat(concat_n(base, mn), concat_optional(base, mx - mn))""",
    "compile-anchor": """def compile_anchor(anchor_type):
    s0 = fsm_fresh_state('anc')
    s1 = fsm_fresh_state('anc')
    assertions = [(s0, 'anchor', anchor_type, s1)]
    return make_fsm_with_assertions([s0, s1], [], [], s0, [s1], [], assertions)""",
    "compile-lookahead": """def compile_lookahead(expr, positive, universe):
    inner = regex_compile(expr, universe)
    s0 = fsm_fresh_state('la')
    s1 = fsm_fresh_state('la')
    assertions = [(s0, 'lookahead', inner, positive, s1)]
    return make_fsm_with_assertions([s0, s1], [], [], s0, [s1], [], assertions)""",
}

CHEZ_SNIPPETS = {
    "parse-class-range": """(define (class-range)
  (parser-bind
   (parse-class-char)
   (lambda (start)
     (parser-bind
      (parser-char #\\-)
      (lambda (_)
        (parser-bind
         (parse-class-char)
         (lambda (end)
           (if (char<=? start end)
               (let* ([si (char->integer start)]
                      [n (+ 1 (- (char->integer end) si))])
                 (parser-pure (map (lambda (i) (integer->char (+ i si)))
                                   (iota n))))
               (parser-fail "invalid range: start > end")))))))))""",
    "parse-interval": """(define (interval-parser)
  (parser-bind
   (parser-char #\\{)
   (lambda (_)
     (parser-bind
      (parser-option-maybe parser-natural)
      (lambda (mn)
        (parser-or
         (parser-bind (parser-char #\\})
                      (lambda (_)
                        (if (nothing? mn)
                            (parser-fail "empty interval {} is invalid")
                            (let ([n (from-just mn)])
                              (parser-pure (cons n n))))))
         (parser-bind (parser-char #\\,)
                      (lambda (_)
                        (parser-bind
                         (parser-option-maybe parser-natural)
                         (lambda (mx)
                           (parser-bind
                            (parser-char #\\})
                            (lambda (_)
                              (let ([min (if (nothing? mn) 0 (from-just mn))]
                                    [max (if (nothing? mx) #f (from-just mx))])
                                (if (and max (> min max))
                                    (parser-fail "interval min > max")
                                    (parser-pure (cons min max))))))))))))))))""",
    "apply-postfix-op": """(define (postfix->ast expr op)
  (cond
   [(char? op)
    (cond
     [(char=? op #\\*) (regex-star expr)]
     [(char=? op #\\+) (regex-plus expr)]
     [(char=? op #\\?) (regex-opt expr)])]
   [(pair? op)
    (regex-repeat expr (car op) (cdr op))]))""",
    "parse-seq": """(define (seq-parser)
  (parser-bind
   (parser-many (parse-postfix))
   (lambda (exprs)
     (parser-pure
      (cond
       [(null? exprs) (regex-empty)]
       [(null? (cdr exprs)) (car exprs)]
       [else (regex-seq exprs)])))))""",
    "parse-alt": """(define (alt-parser)
  (parser-bind
   (parse-seq)
   (lambda (first)
     (parser-bind
      (parser-many (parser-then (parser-char #\\|) (parse-seq)))
      (lambda (rest)
        (parser-pure
         (if (null? rest)
             first
             (regex-alt (cons first rest)))))))))""",
    "compile-repeat": """(define (compile-repeat* expr mn mx universe)
  (let ([base (regex-compile expr universe)])
    (cond
     [(and (= mn 0) (eqv? mx 0)) (fsm-epsilon-lang)]
     [(and (= mn 0) (not mx)) (fsm-star base)]
     [(= mn 0)
      (fold-left (lambda (acc _) (fsm-optional (fsm-concat base acc)))
                 (fsm-epsilon-lang)
                 (iota mx))]
     [(eqv? mn mx)
      (fold-left fsm-concat base
                 (map (lambda (_) (regex-compile expr universe))
                      (iota (- mn 1))))]
     [(not mx)
      (let ([req (fold-left fsm-concat base
                            (map (lambda (_) (regex-compile expr universe))
                                 (iota (- mn 1))))])
        (fsm-concat req (fsm-star base)))]
     [else
      (let* ([req (fold-left fsm-concat base
                             (map (lambda (_) (regex-compile expr universe))
                                  (iota (- mn 1))))]
             [opt (fold-left (lambda (acc _)
                               (fsm-concat (fsm-optional base) acc))
                             (fsm-epsilon-lang)
                             (iota (- mx mn)))])
        (fsm-concat req opt))])))""",
    "compile-anchor": """(define (anchor-fsm type)
  (let ([s0 (fsm-fresh-state "anc")]
        [s1 (fsm-fresh-state "anc")])
    (make-fsm-with-assertions
     (list s0 s1) '() '() s0 (list s1) '()
     (list (list s0 'anchor type s1)))))""",
    "compile-lookahead": """(define (lookahead-fsm expr positive? universe)
  (let ([inner (regex-compile expr universe)]
        [s0 (fsm-fresh-state "la")]
        [s1 (fsm-fresh-state "la")])
    (make-fsm-with-assertions
     (list s0 s1) '() '() s0 (list s1) '()
     (list (list s0 'lookahead inner positive? s1)))))""",
}

BUGGY_CASES = [
    {
        "fn": "parse-class-range",
        "buggy": """(define (parse-class-range)
  (parser-bind
   (parse-class-char)
   (lambda (start)
     (parser-bind
      (parser-char #\\-)
      (lambda (_)
        (parser-bind
         (parse-class-char)
         (lambda (end)
           (if (char<=? start end)
               (let* ([start-int (char->integer start)]
                      [count (- (char->integer end) start-int)])
                 (parser-pure (map (lambda (i) (integer->char (+ i start-int)))
                                   (iota count))))
               (parser-fail "invalid range: start > end")))))))))""",
        "note": "Range expansion must be inclusive; dropping +1 omits the upper bound character.",
    },
    {
        "fn": "parse-class-range",
        "buggy": """(define (parse-class-range)
  (parser-bind
   (parse-class-char)
   (lambda (start)
     (parser-bind
      (parser-char #\\-)
      (lambda (_)
        (parser-bind
         (parse-class-char)
         (lambda (end)
           (parser-pure (list start #\\- end)))))))))""",
        "note": "Character ranges must expand inclusively; returning literal start-dash-end loses range semantics.",
    },
    {
        "fn": "parse-interval",
        "buggy": """(define (parse-interval)
  (parser-bind
   (parser-char #\\{)
   (lambda (_)
     (parser-bind
      (parser-option-maybe parser-natural)
      (lambda (min-maybe)
        (parser-or
         (parser-bind
          (parser-char #\\})
          (lambda (_)
            (if (nothing? min-maybe)
                (parser-fail "empty interval {} is invalid")
                (let ([n (from-just min-maybe)])
                  (parser-pure (cons n #f))))))
         (parser-bind
          (parser-char #\\,)
          (lambda (_)
            (parser-bind
             (parser-option-maybe parser-natural)
             (lambda (max-maybe)
               (parser-bind
                (parser-char #\\})
                (lambda (_)
                  (let ([min (if (nothing? min-maybe) 0 (from-just min-maybe))]
                        [max (if (nothing? max-maybe) #f (from-just max-maybe))])
                    (if (and max (> min max))
                        (parser-fail "interval min > max")
                        (parser-pure (cons min max))))))))))))))))""",
        "note": "Exact intervals {n} must yield (n . n), not an unbounded upper bound.",
    },
    {
        "fn": "parse-interval",
        "buggy": """(define (parse-interval)
  (parser-bind
   (parser-char #\\{)
   (lambda (_)
     (parser-bind
      (parser-option-maybe parser-natural)
      (lambda (min-maybe)
        (parser-or
         (parser-bind
          (parser-char #\\})
          (lambda (_)
            (if (nothing? min-maybe)
                (parser-fail "empty interval {} is invalid")
                (let ([n (from-just min-maybe)])
                  (parser-pure (cons n n))))))
         (parser-bind
          (parser-char #\\,)
          (lambda (_)
            (parser-bind
             (parser-option-maybe parser-natural)
             (lambda (max-maybe)
               (parser-bind
                (parser-char #\\})
                (lambda (_)
                  (let ([min (if (nothing? min-maybe) 1 (from-just min-maybe))]
                        [max (if (nothing? max-maybe) #f (from-just max-maybe))])
                    (parser-pure (cons min max)))))))))))))))""",
        "note": "Open lower bound {,m} must default min to 0 and still enforce min<=max validation.",
    },
    {
        "fn": "apply-postfix-op",
        "buggy": """(define (apply-postfix-op expr op)
  (cond
   [(char? op)
    (cond
     [(char=? op #\\*) (regex-plus expr)]
     [(char=? op #\\+) (regex-star expr)]
     [(char=? op #\\?) (regex-opt expr)])]
   [(pair? op)
    (regex-repeat expr (car op) (cdr op))]))""",
        "note": "Star and plus semantics are swapped; postfix mapping must preserve operator meaning.",
    },
    {
        "fn": "apply-postfix-op",
        "buggy": """(define (apply-postfix-op expr op)
  (cond
   [(char? op)
    (cond
     [(char=? op #\\*) (regex-star expr)]
     [(char=? op #\\+) (regex-plus expr)]
     [(char=? op #\\?) (regex-opt expr)])]
   [(pair? op)
    (regex-repeat expr (cdr op) (car op))]))""",
        "note": "Interval pairs are (min . max); swapping bounds corrupts repeat semantics.",
    },
    {
        "fn": "parse-seq",
        "buggy": """(define (parse-seq)
  (parser-bind
   (parser-many (parse-postfix))
   (lambda (exprs)
     (parser-pure (regex-seq exprs)))))""",
        "note": "Empty and singleton sequences must normalize to regex-empty/single-node, not always regex-seq.",
    },
    {
        "fn": "parse-seq",
        "buggy": """(define (parse-seq)
  (parser-bind
   (parser-some (parse-postfix))
   (lambda (exprs)
     (parser-pure
      (if (null? (cdr exprs))
          (car exprs)
          (regex-seq exprs))))))""",
        "note": "Sequence parser must allow zero terms so empty pattern parses as regex-empty.",
    },
    {
        "fn": "parse-alt",
        "buggy": """(define (parse-alt)
  (parser-bind
   (parse-seq)
   (lambda (first)
     (parser-bind
      (parser-option-maybe (parser-then (parser-char #\\|) (parse-seq)))
      (lambda (rest-maybe)
        (if (nothing? rest-maybe)
            (parser-pure first)
            (parser-pure (regex-alt (list first (from-just rest-maybe))))))))))""",
        "note": "Alternation must support repeated | branches, not just a single optional branch.",
    },
    {
        "fn": "parse-alt",
        "buggy": """(define (parse-alt)
  (parser-bind
   (parse-seq)
   (lambda (first)
     (parser-bind
      (parser-many (parser-then (parser-char #\\|) (parse-seq)))
      (lambda (rest)
        (parser-pure (regex-alt (cons first rest))))))))""",
        "note": "When no | appears, parser must return the sequence directly instead of wrapping singleton alt.",
    },
    {
        "fn": "compile-repeat",
        "buggy": """(define (compile-repeat expr min max universe)
  (let ([base (regex-compile expr universe)])
    (cond
     [(and (= min 0) (eqv? max 0))
      (make-fsm '(dead) '() '() 'dead '())]
     [(and (= min 0) (not max))
      (fsm-star base)]
     [(= min 0)
      (fold-left (lambda (acc _)
                   (fsm-optional (fsm-concat base acc)))
                 (fsm-epsilon-lang)
                 (iota max))]
     [(eqv? min max)
      (fold-left fsm-concat base
                 (map (lambda (_) (regex-compile expr universe))
                      (iota (- min 1))))]
     [(not max)
      (let ([required (fold-left fsm-concat base
                                 (map (lambda (_) (regex-compile expr universe))
                                      (iota (- min 1))))])
        (fsm-concat required (fsm-star base)))]
     [else
      (let* ([required (fold-left fsm-concat base
                                  (map (lambda (_) (regex-compile expr universe))
                                       (iota (- min 1))))]
             [optional (fold-left (lambda (acc _)
                                    (fsm-concat (fsm-optional base) acc))
                                  (fsm-epsilon-lang)
                                  (iota (- max min)))])
        (fsm-concat required optional))])))""",
        "note": "{0,0} must match empty string, not empty language.",
    },
    {
        "fn": "compile-repeat",
        "buggy": """(define (compile-repeat expr min max universe)
  (let ([base (regex-compile expr universe)])
    (cond
     [(and (= min 0) (eqv? max 0))
      (fsm-epsilon-lang)]
     [(and (= min 0) (not max))
      (fsm-star base)]
     [(= min 0)
      (fold-left (lambda (acc _)
                   (fsm-optional (fsm-concat base acc)))
                 (fsm-epsilon-lang)
                 (iota max))]
     [(eqv? min max)
      (fold-left fsm-concat base
                 (map (lambda (_) (regex-compile expr universe))
                      (iota (- min 1))))]
     [(not max)
      (fold-left fsm-concat base
                 (map (lambda (_) (regex-compile expr universe))
                      (iota (- min 1))))]
     [else
      (let* ([required (fold-left fsm-concat base
                                  (map (lambda (_) (regex-compile expr universe))
                                       (iota (- min 1))))]
             [optional (fold-left (lambda (acc _)
                                    (fsm-concat (fsm-optional base) acc))
                                  (fsm-epsilon-lang)
                                  (iota (- max min)))])
        (fsm-concat required optional))])))""",
        "note": "Unbounded upper ranges {n,} require concatenating a trailing star for extra repetitions.",
    },
    {
        "fn": "compile-anchor",
        "buggy": """(define (compile-anchor type)
  (let ([s0 (fsm-fresh-state "anc")]
        [s1 (fsm-fresh-state "anc")])
    (make-fsm (list s0 s1) '() '() s0 (list s1) (list (list s0 s1)))))""",
        "note": "Anchors are assertion transitions, not epsilon edges on a plain FSM.",
    },
    {
        "fn": "compile-anchor",
        "buggy": """(define (compile-anchor type)
  (let ([s0 (fsm-fresh-state "anc")]
        [s1 (fsm-fresh-state "anc")])
    (make-fsm-with-assertions
     (list s0 s1) '() '() s0 (list s1) '()
     (list (list s0 'anchor 'start s1)))))""",
        "note": "compile-anchor must preserve caller-supplied anchor type instead of hardcoding start.",
    },
    {
        "fn": "compile-lookahead",
        "buggy": """(define (compile-lookahead expr positive? universe)
  (let ([inner (regex-compile expr universe)]
        [s0 (fsm-fresh-state "la")]
        [s1 (fsm-fresh-state "la")])
    (make-fsm-with-assertions
     (list s0 s1) '() '() s0 (list s1) '()
     (list (list s0 'lookahead inner #t s1)))))""",
        "note": "Lookahead polarity must follow the positive? argument; forcing #t breaks negative lookahead.",
    },
    {
        "fn": "compile-lookahead",
        "buggy": """(define (compile-lookahead expr positive? universe)
  (let ([inner (regex-compile expr universe)]
        [s0 (fsm-fresh-state "la")]
        [s1 (fsm-fresh-state "la")])
    (make-fsm-with-assertions
     (list s0 s1) '() '() s0 (list s1) '()
     (list (list s0 'anchor 'start s1)))))""",
        "note": "Lookahead compilation must emit lookahead assertions carrying the inner regex machine.",
    },
]

DIFFICULTY = {
    "parse-class-range": "medium",
    "parse-interval": "hard",
    "apply-postfix-op": "easy",
    "parse-seq": "medium",
    "parse-alt": "medium",
    "compile-repeat": "hard",
    "compile-anchor": "easy",
    "compile-lookahead": "hard",
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
    sid = f"fp_parsing_regex_parser_{family}_{family_counter[family]:03d}"
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
            ground_truth=ground_truth,
        ),
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


# -----------------------------------------------------------------------------
# Family 1: spec_to_code (16)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Implement this regex parsing/compilation utility in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "fp", "parsing", "regex", "spec-to-code", fn],
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
        tags=["tier1", "fp", "parsing", "regex", "skeleton-completion", fn],
    )

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Implement `{fn}` from this regex parser contract.

Module: `{SOURCE_MODULE}`
Contract focus: {FUNCTION_SPECS[fn]}

Requirements:
1. Preserve regex/parse semantics and edge cases.
2. Keep the exact function name/signature for `{fn}`.
3. Return one production-ready definition only.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "fp", "parsing", "regex", "contract-implementation", fn],
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
        tags=["tier1", "fp", "parsing", "regex", "python-to-scheme", fn],
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
        tags=["tier1", "fp", "parsing", "regex", "chez-to-fold", fn],
    )

    add_sample(
        family="translation",
        category="translation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Translate this reference implementation into canonical Fold Scheme for `{fn}`.

Preserve observable behavior exactly (including failure cases).
Keep the target function name/signature as `{fn}`.
Return only the final Scheme definition.

```python
{PYTHON_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "fp", "parsing", "regex", "reference-translation", fn],
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
        tags=["tier1", "fp", "parsing", "regex", "bugfix", fn],
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
    composition_prompt = (
        f"{prompt.rstrip()}\n\n"
        f"Ensure `{source_function}` is part of the composed solution."
    )
    add_sample(
        family="composition",
        category="usage",
        difficulty=difficulty,
        source_function=source_function,
        prompt=composition_prompt,
        ground_truth=ground_truth,
        verify_expr=verify_check.strip(),
        tags=["tier1", "fp", "parsing", "regex", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # parse-class-range
    {
        "fn": "parse-class-range",
        "prompt": "Parse class range `a-z` and return expanded character count.",
        "gt": "(let ([r (parser-parse (parser-left (parse-class-range) parser-eof) \"a-z\")]) (if (right? r) (length (from-right r)) -1))",
        "verify": "(= (let ([r (parser-parse (parser-left (parse-class-range) parser-eof) \"a-z\")]) (if (right? r) (length (from-right r)) -1)) 26)",
        "difficulty": "medium",
        "tags": ["range-size"],
    },
    {
        "fn": "parse-class-range",
        "prompt": "Parse class range `0-2` and return the expanded character list.",
        "gt": "(let ([r (parser-parse (parser-left (parse-class-range) parser-eof) \"0-2\")]) (if (right? r) (from-right r) '()))",
        "verify": "(equal? (let ([r (parser-parse (parser-left (parse-class-range) parser-eof) \"0-2\")]) (if (right? r) (from-right r) '())) '(#\\0 #\\1 #\\2))",
        "difficulty": "easy",
        "tags": ["digits"],
    },
    {
        "fn": "parse-class-range",
        "prompt": "Return whether parsing reversed class range `z-a` fails.",
        "gt": "(left? (parser-parse (parser-left (parse-class-range) parser-eof) \"z-a\"))",
        "verify": "(equal? (left? (parser-parse (parser-left (parse-class-range) parser-eof) \"z-a\")) #t)",
        "difficulty": "medium",
        "tags": ["invalid-range"],
    },
    {
        "fn": "parse-class-range",
        "prompt": "Parse class range `m-m` and return whether it expands to a singleton list.",
        "gt": "(let ([r (parser-parse (parser-left (parse-class-range) parser-eof) \"m-m\")]) (and (right? r) (equal? (from-right r) '(#\\m))))",
        "verify": "(equal? (let ([r (parser-parse (parser-left (parse-class-range) parser-eof) \"m-m\")]) (and (right? r) (equal? (from-right r) '(#\\m)))) #t)",
        "difficulty": "easy",
        "tags": ["singleton"],
    },

    # parse-interval
    {
        "fn": "parse-interval",
        "prompt": "Parse interval token `{3}` and return the parsed min/max pair.",
        "gt": "(let ([r (parser-parse (parser-left (parse-interval) parser-eof) \"{3}\")]) (if (right? r) (from-right r) '(bad)))",
        "verify": "(equal? (let ([r (parser-parse (parser-left (parse-interval) parser-eof) \"{3}\")]) (if (right? r) (from-right r) '(bad))) (cons 3 3))",
        "difficulty": "easy",
        "tags": ["exact"],
    },
    {
        "fn": "parse-interval",
        "prompt": "Parse interval token `{2,}` and return parsed pair.",
        "gt": "(let ([r (parser-parse (parser-left (parse-interval) parser-eof) \"{2,}\")]) (if (right? r) (from-right r) '(bad)))",
        "verify": "(equal? (let ([r (parser-parse (parser-left (parse-interval) parser-eof) \"{2,}\")]) (if (right? r) (from-right r) '(bad))) (cons 2 #f))",
        "difficulty": "medium",
        "tags": ["unbounded"],
    },
    {
        "fn": "parse-interval",
        "prompt": "Parse interval token `{,4}` and return parsed pair.",
        "gt": "(let ([r (parser-parse (parser-left (parse-interval) parser-eof) \"{,4}\")]) (if (right? r) (from-right r) '(bad)))",
        "verify": "(equal? (let ([r (parser-parse (parser-left (parse-interval) parser-eof) \"{,4}\")]) (if (right? r) (from-right r) '(bad))) (cons 0 4))",
        "difficulty": "medium",
        "tags": ["lower-default"],
    },
    {
        "fn": "parse-interval",
        "prompt": "Return whether invalid interval `{4,2}` fails to parse.",
        "gt": "(left? (parser-parse (parser-left (parse-interval) parser-eof) \"{4,2}\"))",
        "verify": "(equal? (left? (parser-parse (parser-left (parse-interval) parser-eof) \"{4,2}\")) #t)",
        "difficulty": "hard",
        "tags": ["validation"],
    },

    # apply-postfix-op
    {
        "fn": "apply-postfix-op",
        "prompt": "Apply `*` postfix to literal `a` and return whether result is regex-star.",
        "gt": "(regex-star? (apply-postfix-op (regex-lit #\\a) #\\*))",
        "verify": "(equal? (regex-star? (apply-postfix-op (regex-lit #\\a) #\\*)) #t)",
        "difficulty": "easy",
        "tags": ["star"],
    },
    {
        "fn": "apply-postfix-op",
        "prompt": "Apply `+` postfix to literal `a` and return whether result is regex-plus.",
        "gt": "(regex-plus? (apply-postfix-op (regex-lit #\\a) #\\+))",
        "verify": "(equal? (regex-plus? (apply-postfix-op (regex-lit #\\a) #\\+)) #t)",
        "difficulty": "easy",
        "tags": ["plus"],
    },
    {
        "fn": "apply-postfix-op",
        "prompt": "Apply interval postfix (2 . 4) to literal `a` and return parsed bounds from regex-repeat.",
        "gt": "(let ([r (apply-postfix-op (regex-lit #\\a) (cons 2 4))]) (list (regex-repeat-min r) (regex-repeat-max r)))",
        "verify": "(equal? (let ([r (apply-postfix-op (regex-lit #\\a) (cons 2 4))]) (list (regex-repeat-min r) (regex-repeat-max r))) '(2 4))",
        "difficulty": "medium",
        "tags": ["interval"],
    },
    {
        "fn": "apply-postfix-op",
        "prompt": "Compile optional postfix on literal `a` and return whether resulting FSM accepts empty and `a` but rejects `aa`.",
        "gt": "(let ([m (regex-compile (apply-postfix-op (regex-lit #\\a) #\\?) '(#\\a #\\b))]) (and (fsm-accepts? m \"\") (fsm-accepts? m \"a\") (not (fsm-accepts? m \"aa\"))))",
        "verify": "(equal? (let ([m (regex-compile (apply-postfix-op (regex-lit #\\a) #\\?) '(#\\a #\\b))]) (and (fsm-accepts? m \"\") (fsm-accepts? m \"a\") (not (fsm-accepts? m \"aa\")))) #t)",
        "difficulty": "medium",
        "tags": ["optional"],
    },

    # parse-seq
    {
        "fn": "parse-seq",
        "prompt": "Parse empty input with parse-seq and return whether AST is regex-empty.",
        "gt": "(let ([r (parser-parse (parser-left (parse-seq) parser-eof) \"\")]) (and (right? r) (regex-empty? (from-right r))))",
        "verify": "(equal? (let ([r (parser-parse (parser-left (parse-seq) parser-eof) \"\")]) (and (right? r) (regex-empty? (from-right r)))) #t)",
        "difficulty": "easy",
        "tags": ["empty"],
    },
    {
        "fn": "parse-seq",
        "prompt": "Parse `ab` with parse-seq and return concatenation arity.",
        "gt": "(let ([r (parser-parse (parser-left (parse-seq) parser-eof) \"ab\")]) (if (and (right? r) (regex-seq? (from-right r))) (length (regex-seq-exprs (from-right r))) -1))",
        "verify": "(= (let ([r (parser-parse (parser-left (parse-seq) parser-eof) \"ab\")]) (if (and (right? r) (regex-seq? (from-right r))) (length (regex-seq-exprs (from-right r))) -1)) 2)",
        "difficulty": "medium",
        "tags": ["concat"],
    },
    {
        "fn": "parse-seq",
        "prompt": "Parse `a*` with parse-seq and return whether singleton result is regex-star.",
        "gt": "(let ([r (parser-parse (parser-left (parse-seq) parser-eof) \"a*\")]) (and (right? r) (regex-star? (from-right r))))",
        "verify": "(equal? (let ([r (parser-parse (parser-left (parse-seq) parser-eof) \"a*\")]) (and (right? r) (regex-star? (from-right r)))) #t)",
        "difficulty": "medium",
        "tags": ["singleton"],
    },
    {
        "fn": "parse-seq",
        "prompt": "Parse `a(?=b)` with parse-seq and return whether it becomes a 2-term sequence.",
        "gt": "(let ([r (parser-parse (parser-left (parse-seq) parser-eof) \"a(?=b)\")]) (and (right? r) (regex-seq? (from-right r)) (= (length (regex-seq-exprs (from-right r))) 2)))",
        "verify": "(equal? (let ([r (parser-parse (parser-left (parse-seq) parser-eof) \"a(?=b)\")]) (and (right? r) (regex-seq? (from-right r)) (= (length (regex-seq-exprs (from-right r))) 2))) #t)",
        "difficulty": "hard",
        "tags": ["lookahead-seq"],
    },

    # parse-alt
    {
        "fn": "parse-alt",
        "prompt": "Parse `a|b|c` and return alternation branch count.",
        "gt": "(let ([r (parser-parse (parser-left (parse-alt) parser-eof) \"a|b|c\")]) (if (and (right? r) (regex-alt? (from-right r))) (length (regex-alt-exprs (from-right r))) -1))",
        "verify": "(= (let ([r (parser-parse (parser-left (parse-alt) parser-eof) \"a|b|c\")]) (if (and (right? r) (regex-alt? (from-right r))) (length (regex-alt-exprs (from-right r))) -1)) 3)",
        "difficulty": "medium",
        "tags": ["nary-alt"],
    },
    {
        "fn": "parse-alt",
        "prompt": "Parse `ab` with parse-alt and return whether AST remains sequence (not alternation).",
        "gt": "(let ([r (parser-parse (parser-left (parse-alt) parser-eof) \"ab\")]) (and (right? r) (regex-seq? (from-right r))))",
        "verify": "(equal? (let ([r (parser-parse (parser-left (parse-alt) parser-eof) \"ab\")]) (and (right? r) (regex-seq? (from-right r)))) #t)",
        "difficulty": "easy",
        "tags": ["no-alt"],
    },
    {
        "fn": "parse-alt",
        "prompt": "Parse `a|bc` and return whether second alternation branch is a sequence.",
        "gt": "(let ([r (parser-parse (parser-left (parse-alt) parser-eof) \"a|bc\")]) (and (right? r) (regex-alt? (from-right r)) (let ([alts (regex-alt-exprs (from-right r))]) (and (= (length alts) 2) (regex-seq? (cadr alts))))))",
        "verify": "(equal? (let ([r (parser-parse (parser-left (parse-alt) parser-eof) \"a|bc\")]) (and (right? r) (regex-alt? (from-right r)) (let ([alts (regex-alt-exprs (from-right r))]) (and (= (length alts) 2) (regex-seq? (cadr alts)))))) #t)",
        "difficulty": "medium",
        "tags": ["mixed-branches"],
    },
    {
        "fn": "parse-alt",
        "prompt": "Parse `a|b` and return whether first branch is literal `a`.",
        "gt": "(let ([r (parser-parse (parser-left (parse-alt) parser-eof) \"a|b\")]) (and (right? r) (regex-alt? (from-right r)) (let ([a1 (car (regex-alt-exprs (from-right r)))]) (and (regex-lit? a1) (equal? (regex-lit-char a1) #\\a)))))",
        "verify": "(equal? (let ([r (parser-parse (parser-left (parse-alt) parser-eof) \"a|b\")]) (and (right? r) (regex-alt? (from-right r)) (let ([a1 (car (regex-alt-exprs (from-right r)))]) (and (regex-lit? a1) (equal? (regex-lit-char a1) #\\a))))) #t)",
        "difficulty": "medium",
        "tags": ["branch-shape"],
    },

    # compile-repeat
    {
        "fn": "compile-repeat",
        "prompt": "Compile repeat bounds {0,0} for literal `a` and return whether FSM accepts only empty string.",
        "gt": "(let ([m (compile-repeat (regex-lit #\\a) 0 0 '(#\\a #\\b))]) (and (fsm-accepts? m \"\") (not (fsm-accepts? m \"a\"))))",
        "verify": "(equal? (let ([m (compile-repeat (regex-lit #\\a) 0 0 '(#\\a #\\b))]) (and (fsm-accepts? m \"\") (not (fsm-accepts? m \"a\")))) #t)",
        "difficulty": "medium",
        "tags": ["zero-zero"],
    },
    {
        "fn": "compile-repeat",
        "prompt": "Compile repeat bounds {2,4} for literal `a` and return acceptance for aa/aaaa with rejection for a/aaaaa.",
        "gt": "(let ([m (compile-repeat (regex-lit #\\a) 2 4 '(#\\a #\\b))]) (and (fsm-accepts? m \"aa\") (fsm-accepts? m \"aaaa\") (not (fsm-accepts? m \"a\")) (not (fsm-accepts? m \"aaaaa\"))))",
        "verify": "(equal? (let ([m (compile-repeat (regex-lit #\\a) 2 4 '(#\\a #\\b))]) (and (fsm-accepts? m \"aa\") (fsm-accepts? m \"aaaa\") (not (fsm-accepts? m \"a\")) (not (fsm-accepts? m \"aaaaa\")))) #t)",
        "difficulty": "hard",
        "tags": ["bounded"],
    },
    {
        "fn": "compile-repeat",
        "prompt": "Compile repeat bounds {2,} for literal `a` and return acceptance for aa/aaaaa with rejection for a.",
        "gt": "(let ([m (compile-repeat (regex-lit #\\a) 2 #f '(#\\a #\\b))]) (and (fsm-accepts? m \"aa\") (fsm-accepts? m \"aaaaa\") (not (fsm-accepts? m \"a\"))))",
        "verify": "(equal? (let ([m (compile-repeat (regex-lit #\\a) 2 #f '(#\\a #\\b))]) (and (fsm-accepts? m \"aa\") (fsm-accepts? m \"aaaaa\") (not (fsm-accepts? m \"a\")))) #t)",
        "difficulty": "hard",
        "tags": ["unbounded"],
    },
    {
        "fn": "compile-repeat",
        "prompt": "Compile repeat bounds {0,2} for literal `a` and return whether empty/a/aa pass but aaa fails.",
        "gt": "(let ([m (compile-repeat (regex-lit #\\a) 0 2 '(#\\a #\\b))]) (and (fsm-accepts? m \"\") (fsm-accepts? m \"a\") (fsm-accepts? m \"aa\") (not (fsm-accepts? m \"aaa\"))))",
        "verify": "(equal? (let ([m (compile-repeat (regex-lit #\\a) 0 2 '(#\\a #\\b))]) (and (fsm-accepts? m \"\") (fsm-accepts? m \"a\") (fsm-accepts? m \"aa\") (not (fsm-accepts? m \"aaa\")))) #t)",
        "difficulty": "hard",
        "tags": ["zero-upper"],
    },

    # compile-anchor
    {
        "fn": "compile-anchor",
        "prompt": "Compile start anchor and return whether assertion metadata encodes anchor/start.",
        "gt": "(let ([a (car (fsm-assertions (compile-anchor 'start)))]) (and (equal? (cadr a) 'anchor) (equal? (caddr a) 'start)))",
        "verify": "(equal? (let ([a (car (fsm-assertions (compile-anchor 'start)))]) (and (equal? (cadr a) 'anchor) (equal? (caddr a) 'start))) #t)",
        "difficulty": "easy",
        "tags": ["metadata"],
    },
    {
        "fn": "compile-anchor",
        "prompt": "Compile end anchor and return whether assertion metadata encodes anchor/end.",
        "gt": "(let ([a (car (fsm-assertions (compile-anchor 'end)))]) (and (equal? (cadr a) 'anchor) (equal? (caddr a) 'end)))",
        "verify": "(equal? (let ([a (car (fsm-assertions (compile-anchor 'end)))]) (and (equal? (cadr a) 'anchor) (equal? (caddr a) 'end))) #t)",
        "difficulty": "easy",
        "tags": ["metadata"],
    },
    {
        "fn": "compile-anchor",
        "prompt": "Compose start anchor with literal `a` and return whether it accepts `a` but rejects `ba`.",
        "gt": "(let ([m (fsm-concat (compile-anchor 'start) (fsm-literal \"a\"))]) (and (fsm-accepts? m \"a\") (not (fsm-accepts? m \"ba\"))))",
        "verify": "(equal? (let ([m (fsm-concat (compile-anchor 'start) (fsm-literal \"a\"))]) (and (fsm-accepts? m \"a\") (not (fsm-accepts? m \"ba\")))) #t)",
        "difficulty": "medium",
        "tags": ["anchored-prefix"],
    },
    {
        "fn": "compile-anchor",
        "prompt": "Concatenate start and end anchors and return whether result matches only empty string.",
        "gt": "(let ([m (fsm-concat (compile-anchor 'start) (compile-anchor 'end))]) (and (fsm-accepts? m \"\") (not (fsm-accepts? m \"a\"))))",
        "verify": "(equal? (let ([m (fsm-concat (compile-anchor 'start) (compile-anchor 'end))]) (and (fsm-accepts? m \"\") (not (fsm-accepts? m \"a\")))) #t)",
        "difficulty": "medium",
        "tags": ["empty-only"],
    },

    # compile-lookahead
    {
        "fn": "compile-lookahead",
        "prompt": "Compose positive lookahead for `a` with literal `a` and return acceptance for `a` with rejection for `b`.",
        "gt": "(let* ([u '(#\\a #\\b #\\c)] [m (fsm-concat (compile-lookahead (regex-lit #\\a) #t u) (fsm-char #\\a))]) (and (fsm-accepts? m \"a\") (not (fsm-accepts? m \"b\"))))",
        "verify": "(equal? (let* ([u '(#\\a #\\b #\\c)] [m (fsm-concat (compile-lookahead (regex-lit #\\a) #t u) (fsm-char #\\a))]) (and (fsm-accepts? m \"a\") (not (fsm-accepts? m \"b\")))) #t)",
        "difficulty": "hard",
        "tags": ["positive"],
    },
    {
        "fn": "compile-lookahead",
        "prompt": "Compose negative lookahead for `b` with literal `a` and return acceptance for `a` with rejection for `b`.",
        "gt": "(let* ([u '(#\\a #\\b #\\c)] [m (fsm-concat (compile-lookahead (regex-lit #\\b) #f u) (fsm-char #\\a))]) (and (fsm-accepts? m \"a\") (not (fsm-accepts? m \"b\"))))",
        "verify": "(equal? (let* ([u '(#\\a #\\b #\\c)] [m (fsm-concat (compile-lookahead (regex-lit #\\b) #f u) (fsm-char #\\a))]) (and (fsm-accepts? m \"a\") (not (fsm-accepts? m \"b\")))) #t)",
        "difficulty": "hard",
        "tags": ["negative"],
    },
    {
        "fn": "compile-lookahead",
        "prompt": "Compile positive lookahead and return whether assertion metadata stores lookahead with #t polarity.",
        "gt": "(let ([a (car (fsm-assertions (compile-lookahead (regex-lit #\\a) #t '(#\\a #\\b #\\c))))]) (and (equal? (cadr a) 'lookahead) (equal? (cadddr a) #t)))",
        "verify": "(equal? (let ([a (car (fsm-assertions (compile-lookahead (regex-lit #\\a) #t '(#\\a #\\b #\\c))))]) (and (equal? (cadr a) 'lookahead) (equal? (cadddr a) #t))) #t)",
        "difficulty": "medium",
        "tags": ["metadata"],
    },
    {
        "fn": "compile-lookahead",
        "prompt": "Build `a(?=b)b` from primitives and return whether it matches `ab` but rejects `ac`.",
        "gt": "(let* ([u '(#\\a #\\b #\\c)] [m (fsm-concat (fsm-char #\\a) (fsm-concat (compile-lookahead (regex-lit #\\b) #t u) (fsm-char #\\b)))]) (and (fsm-accepts? m \"ab\") (not (fsm-accepts? m \"ac\"))))",
        "verify": "(equal? (let* ([u '(#\\a #\\b #\\c)] [m (fsm-concat (fsm-char #\\a) (fsm-concat (compile-lookahead (regex-lit #\\b) #t u) (fsm-char #\\b)))]) (and (fsm-accepts? m \"ab\") (not (fsm-accepts? m \"ac\")))) #t)",
        "difficulty": "hard",
        "tags": ["sequence-integration"],
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

# Strengthen non-composition rows with an independent behavior check selected
# from composition verifies of the same source function.
composition_verify_by_fn: Dict[str, List[str]] = defaultdict(list)
for case in composition_cases:
    fn = str(case["fn"])
    check = str(case["verify"]).strip()
    if check not in composition_verify_by_fn[fn]:
        composition_verify_by_fn[fn].append(check)

for sample in samples:
    family = str(sample["family"])
    if family == "composition":
        continue

    fn = str(sample["source_function"])
    checks = composition_verify_by_fn.get(fn, [])
    if not checks:
        continue

    sid = str(sample["id"])
    base = sum(ord(ch) for ch in sid)
    pick1 = base % len(checks)
    selected_checks = [checks[pick1]]
    if len(checks) > 1:
        pick2 = (base * 7 + len(sid)) % len(checks)
        if pick2 == pick1:
            pick2 = (pick1 + 1) % len(checks)
        selected_checks.append(checks[pick2])

    sample["verify_expr"] = (
        f"(and {str(sample['verify_expr']).strip()} {' '.join(selected_checks)})"
    )
    checks_block = "\n\n".join(
        f"Check {idx + 1}:\n```scheme\n{check}\n```"
        for idx, check in enumerate(selected_checks)
    )
    sample["prompt"] = (
        f"{str(sample['prompt']).rstrip()}\n\n"
        "Independent behavior checks to satisfy:\n"
        f"{checks_block}"
    )

if sum(1 for s in samples if s["family"] == "composition") != 32:
    raise ValueError("composition family must contain exactly 32 samples")

by_family: Dict[str, List[Dict[str, object]]] = defaultdict(list)
for sample in samples:
    by_family[str(sample["family"])].append(sample)

EVAL_RATIO = 0.18
EVAL_MIN_BY_FAMILY = {
    "spec_to_code": 3,
    "translation": 3,
    "bugfix": 2,
    "composition": 5,
}

eval_ids = compute_leakage_aware_eval_ids(
    samples,
    eval_ratio=EVAL_RATIO,
    eval_min_by_family=EVAL_MIN_BY_FAMILY,
    enforce_source_function_coverage=True,
)

id_to_sample: Dict[str, Dict[str, object]] = {str(sample["id"]): sample for sample in samples}
all_source_functions = sorted({str(sample["source_function"]) for sample in samples})
missing_after = [
    fn
    for fn in all_source_functions
    if not any(str(id_to_sample[sid]["source_function"]) == fn for sid in eval_ids)
]
if missing_after:
    raise ValueError(f"eval split is missing source functions: {missing_after}")

train_rows: List[Dict[str, object]] = []
eval_rows: List[Dict[str, object]] = []
for sample in samples:
    row = dict(sample)
    if sample["id"] in eval_ids:
        row["split"] = "eval"
        eval_rows.append(row)
    else:
        row["split"] = "train"
        train_rows.append(row)

if not train_rows or not eval_rows:
    raise ValueError(f"split mismatch: train={len(train_rows)}, eval={len(eval_rows)}")


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


write_jsonl(ALL_PATH, [dict(sample, split=("eval" if sample["id"] in eval_ids else "train")) for sample in samples])
write_jsonl(TRAIN_PATH, train_rows)
write_jsonl(EVAL_PATH, eval_rows)

summary = {
    "total": len(samples),
    "train": len(train_rows),
    "eval": len(eval_rows),
    "families": {
        family: {
            "total": len(family_samples),
            "eval": sum(1 for sample in family_samples if sample["id"] in eval_ids),
            "train": sum(1 for sample in family_samples if sample["id"] not in eval_ids),
        }
        for family, family_samples in sorted(by_family.items())
    },
    "difficulty": dict(sorted(Counter(str(sample["difficulty"]) for sample in samples).items())),
    "source_functions": dict(sorted(Counter(str(sample["source_function"]) for sample in samples).items())),
}

with SUMMARY_PATH.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print(f"Wrote: {ALL_PATH}")
print(f"Wrote: {TRAIN_PATH}")
print(f"Wrote: {EVAL_PATH}")
