#!/usr/bin/env python3
"""Generate Tier-1 FP parsing parser-compile SFT samples for lattice/fp/parsing/parser-compile.ss."""

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
from sft_split_utils import compute_leakage_aware_eval_ids

ALL_PATH = OUT_DIR / "all.jsonl"
TRAIN_PATH = OUT_DIR / "train.jsonl"
EVAL_PATH = OUT_DIR / "eval.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.json"

SOURCE_MODULE = "lattice/fp/parsing/parser-compile.ss"
SOURCE_TEST = "lattice/fp/parsing/test-parser-compile.ss"

DEFS: Dict[str, str] = {
    "advance-pos-range": """(define (advance-pos-range pos input from to)
  (let loop ([i from] [p pos])
    (if (>= i to) p
        (loop (+ i 1) (advance-pos p (string-ref input i))))))""",
    "dfa->parser": """(define (dfa->parser fsm desc)
  (doc 'export #t)
  (let ([automaton (if (fsm-deterministic? fsm) fsm (fsm-minimize (nfa->dfa fsm)))])
    (make-parser
     (lambda (state)
       (let* ([input (parser-state-input state)]
              [start (parser-state-index state)]
              [len (string-length input)]
              [init (fsm-start automaton)]
              [acc (fsm-accepting automaton)])
         (let loop ([idx start]
                    [current init]
                    [last-accept-idx
                     (if (pair? (memq init acc)) start #f)])
           (if (>= idx len)
               (if last-accept-idx
                   (let* ([matched (substring input start last-accept-idx)]
                          [new-pos (advance-pos-range
                                    (parser-state-pos state) input start last-accept-idx)])
                     (right (cons matched (parser-make-state input last-accept-idx new-pos))))
                   (left (make-parse-error
                          (parser-state-pos state)
                          (string-append "expected " desc)
                          (list desc))))
               (let ([next-states (fsm-delta automaton current (string-ref input idx))])
                 (if (null? next-states)
                     (if last-accept-idx
                         (let* ([matched (substring input start last-accept-idx)]
                                [new-pos (advance-pos-range
                                          (parser-state-pos state) input start last-accept-idx)])
                           (right (cons matched (parser-make-state input last-accept-idx new-pos))))
                         (left (make-parse-error
                                (parser-state-pos state)
                                (string-append "expected " desc)
                                (list desc))))
                     (let ([next (car next-states)])
                       (loop (+ idx 1)
                             next
                             (if (pair? (memq next acc))
                                 (+ idx 1)
                                 last-accept-idx))))))))))))""",
    "regex-ast->parser": """(define (regex-ast->parser ast)
  (doc 'export #t)
  (cond
   [(regex-empty? ast)
    (parser-pure "")]

   [(regex-lit? ast)
    (parser-map string (parser-char (regex-lit-char ast)))]

   [(regex-dot? ast)
    (parser-map string parser-any-char)]

   [(regex-class? ast)
    (let ([chars (regex-class-chars ast)]
          [negated? (regex-class-negated? ast)])
      (parser-map string
        (parser-satisfy
         (if negated?
             (lambda (c) (not (pair? (memv c chars))))
             (lambda (c) (pair? (memv c chars))))
         "character class")))]

   [(regex-seq? ast)
    (let ([parsers (map regex-ast->parser (regex-seq-exprs ast))])
      (fold-left (lambda (acc p)
                   (parser-bind acc
                     (lambda (s1)
                       (parser-map (lambda (s2) (string-append s1 s2)) p))))
                 (parser-pure "")
                 parsers))]

   [(regex-alt? ast)
    (let ([parsers (map (lambda (e) (parser-try (regex-ast->parser e)))
                        (regex-alt-exprs ast))])
      (parser-choice parsers))]

   [(regex-star? ast)
    (let ([inner (regex-ast->parser (regex-star-expr ast))])
      (parser-map (lambda (parts) (apply string-append parts))
                  (parser-many inner)))]

   [(regex-plus? ast)
    (let ([inner (regex-ast->parser (regex-plus-expr ast))])
      (parser-map (lambda (parts) (apply string-append parts))
                  (parser-some inner)))]

   [(regex-opt? ast)
    (let ([inner (regex-ast->parser (regex-opt-expr ast))])
      (parser-optional inner ""))]

   [(regex-group? ast)
    (regex-ast->parser (regex-group-expr ast))]

   [(regex-repeat? ast)
    (compile-repeat-to-parser
     (regex-ast->parser (regex-repeat-expr ast))
     (regex-repeat-min ast)
     (regex-repeat-max ast))]

   [else
    (parser-fail "unsupported regex AST node")]))""",
    "compile-repeat-to-parser": """(define (compile-repeat-to-parser inner min max)
  (cond
   [(and (= min 0) (eqv? max 0))
    (parser-pure "")]
   [(and (= min 0) (not max))
    (parser-map (lambda (parts) (apply string-append parts))
                (parser-many inner))]
   [(eqv? min max)
    (parser-map (lambda (parts) (apply string-append parts))
                (parser-count min inner))]
   [(not max)
    (parser-map (lambda (parts) (apply string-append parts))
                (at-least min inner))]
   [else
    (parser-map (lambda (parts) (apply string-append parts))
                (range-of min max inner))]))""",
    "regex->parser": """(define (regex->parser pattern)
  (doc 'export #t)
  (dfa->parser (regex->dfa pattern) pattern))""",
    "regex->combinator-parser": """(define (regex->combinator-parser pattern)
  (doc 'export #t)
  (let ([result (regex-parse pattern)])
    (if (left? result)
        (parser-fail (string-append "invalid regex: " (error-message (from-left result))))
        (regex-ast->parser (from-right result)))))""",
    "compile-regex": """(define (compile-regex pattern)
  (doc 'export #t)
  (let* ([parse-result (regex-parse pattern)]
         [ast (if (right? parse-result)
                  (from-right parse-result)
                  (error 'compile-regex
                    (string-append "invalid regex: "
                      (error-message (from-left parse-result)))))]
         [nfa (regex-compile ast *default-universe*)]
         [the-dfa (fsm-minimize nfa)]
         [the-parser (dfa->parser the-dfa pattern)])
    (make-compiled-regex pattern ast the-dfa the-parser)))""",
    "compiled-regex-parse": """(define (compiled-regex-parse crx input)
  (doc 'export #t)
  (parser-parse (compiled-regex-parser crx) input))""",
}

FUNCTION_ORDER = [
    "advance-pos-range",
    "dfa->parser",
    "regex-ast->parser",
    "compile-repeat-to-parser",
    "regex->parser",
    "regex->combinator-parser",
    "compile-regex",
    "compiled-regex-parse",
]

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")

FUNCTION_SPECS = {
    "advance-pos-range": "Advance parser position over input[from..to), including newline-aware line/column updates.",
    "dfa->parser": "Wrap DFA/NFA as parser that returns longest accepting prefix and parse error otherwise.",
    "regex-ast->parser": "Translate regex AST into composable parser combinators preserving AST semantics.",
    "compile-repeat-to-parser": "Compile repeat bounds (min,max) to parser combinator forms for exact/range/unbounded repetition.",
    "regex->parser": "Compile regex string to DFA-backed longest-prefix parser.",
    "regex->combinator-parser": "Compile regex string via AST translation; return parser-fail on invalid regex.",
    "compile-regex": "Build compiled-regex dual form with pattern, AST, minimized DFA, and DFA-backed parser.",
    "compiled-regex-parse": "Run parser from compiled-regex against input and return parser result.",
}

SKELETONS = {
    "advance-pos-range": """(define (advance-pos-range pos input from to)
  ;; TODO: advance parser position from index `from` to `to` (exclusive)
  <TODO>)""",
    "dfa->parser": """(define (dfa->parser fsm desc)
  ;; TODO: build parser that tracks longest accepting prefix over DFA transitions
  <TODO>)""",
    "regex-ast->parser": """(define (regex-ast->parser ast)
  ;; TODO: map regex AST constructors into parser combinator behavior
  <TODO>)""",
    "compile-repeat-to-parser": """(define (compile-repeat-to-parser inner min max)
  ;; TODO: implement {0,0}, {0,}, exact, at-least, and bounded range forms
  <TODO>)""",
    "regex->parser": """(define (regex->parser pattern)
  ;; TODO: compile regex string to DFA-backed parser
  <TODO>)""",
    "regex->combinator-parser": """(define (regex->combinator-parser pattern)
  ;; TODO: parse regex; return parser-fail on error; otherwise translate AST
  <TODO>)""",
    "compile-regex": """(define (compile-regex pattern)
  ;; TODO: compile regex into dual form (pattern ast dfa parser)
  <TODO>)""",
    "compiled-regex-parse": """(define (compiled-regex-parse crx input)
  ;; TODO: parse input with compiled-regex parser
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "advance-pos-range": "(let* ([s (string-append \"a\" (string #\\newline) \"b\")] [st (parser-initial-state s)] [p0 (parser-state-pos st)] [p1 (advance-pos-range p0 s 0 3)] [p2 (advance-pos-range p0 s 1 3)]) (and (= (pos-line p1) 2) (= (pos-col p1) 2) (= (pos-line p2) 2) (= (pos-col p2) 2)))",
    "dfa->parser": "(let* ([p (dfa->parser (regex->dfa \"[0-9]+\") \"digits\")] [r1 (parser-parse p \"42abc\")] [r2 (parser-parse p \"abc\")] [r3 (parser-parse p \"007!\")]) (and (right? r1) (equal? (from-right r1) \"42\") (left? r2) (right? r3) (equal? (from-right r3) \"007\")))",
    "regex-ast->parser": "(let* ([ast (regex-seq (list (regex-lit #\\a) (regex-star (regex-lit #\\b)) (regex-lit #\\c)))] [p (regex-ast->parser ast)] [r1 (parser-parse p \"abbbc!\")] [r2 (parser-parse p \"ac!\")] [r3 (parser-parse p \"abbd\")] [pneg (regex-ast->parser (regex-class (list #\\a #\\b) #t))] [rn1 (parser-parse pneg \"x\")] [rn2 (parser-parse pneg \"a\")]) (and (right? r1) (equal? (from-right r1) \"abbbc\") (right? r2) (equal? (from-right r2) \"ac\") (left? r3) (right? rn1) (equal? (from-right rn1) \"x\") (left? rn2)))",
    "compile-repeat-to-parser": "(let* ([inner (parser-map string (parser-char #\\a))] [p0 (compile-repeat-to-parser inner 0 0)] [p1 (compile-repeat-to-parser inner 2 4)] [p2 (compile-repeat-to-parser inner 2 #f)] [r0 (parser-parse p0 \"bbb\")] [r1 (parser-parse p1 \"aaab\")] [r2 (parser-parse p1 \"ab\")] [r3 (parser-parse p2 \"aaaa!\")] [r4 (parser-parse p2 \"a!\")]) (and (right? r0) (equal? (from-right r0) \"\") (right? r1) (equal? (from-right r1) \"aaa\") (left? r2) (right? r3) (equal? (from-right r3) \"aaaa\") (left? r4)))",
    "regex->parser": "(let* ([p (regex->parser \"a|ab\")] [r1 (parser-parse p \"ab!\")] [r2 (parser-parse p \"a!\")] [r3 (parser-parse p \"b!\")]) (and (right? r1) (equal? (from-right r1) \"ab\") (right? r2) (equal? (from-right r2) \"a\") (left? r3)))",
    "regex->combinator-parser": "(let* ([p (regex->combinator-parser \"a|ab\")] [r1 (parser-parse p \"ab!\")] [r2 (parser-parse p \"b!\")] [bad (regex->combinator-parser \"[\")] [r3 (parser-parse bad \"x\")]) (and (right? r1) (equal? (from-right r1) \"a\") (left? r2) (left? r3)))",
    "compile-regex": "(let* ([crx (compile-regex \"a|ab\")] [p (compiled-regex-parser crx)] [r (parser-parse p \"ab!\")]) (and (compiled-regex? crx) (equal? (compiled-regex-pattern crx) \"a|ab\") (compiled-regex-matches? crx \"a\") (compiled-regex-matches? crx \"ab\") (not (compiled-regex-matches? crx \"b\")) (right? r) (equal? (from-right r) \"ab\")))",
    "compiled-regex-parse": "(let* ([crx (compile-regex \"[0-9]+\")] [r1 (compiled-regex-parse crx \"123abc\")] [r2 (compiled-regex-parse crx \"abc\")]) (and (right? r1) (equal? (from-right r1) \"123\") (left? r2)))",
}

TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "advance-pos-range": """def advance_pos_range(pos, input_s, frm, to):
    i, p = frm, pos
    while i < to:
        p = advance_pos(p, input_s[i])
        i += 1
    return p""",
    "dfa->parser": """def dfa_to_parser(fsm, desc):
    automaton = fsm if fsm_deterministic_q(fsm) else fsm_minimize(nfa_to_dfa(fsm))
    def run(state):
        input_s, start, pos = state.input, state.index, state.pos
        idx = start
        current = fsm_start(automaton)
        accepting = set(fsm_accepting(automaton))
        last_accept = start if current in accepting else None
        while idx < len(input_s):
            next_states = fsm_delta(automaton, current, input_s[idx])
            if not next_states:
                break
            current = next_states[0]
            idx += 1
            if current in accepting:
                last_accept = idx
        if last_accept is None:
            return parse_error(pos, f"expected {desc}")
        matched = input_s[start:last_accept]
        new_pos = advance_pos_range(pos, input_s, start, last_accept)
        return ok(matched, state_with(input_s, last_accept, new_pos))
    return make_parser(run)""",
    "regex-ast->parser": """def regex_ast_to_parser(ast):
    if regex_empty_q(ast):
        return parser_pure("")
    if regex_lit_q(ast):
        return parser_map(str, parser_char(regex_lit_char(ast)))
    if regex_dot_q(ast):
        return parser_map(str, parser_any_char)
    if regex_class_q(ast):
        chars = regex_class_chars(ast)
        neg = regex_class_negated_q(ast)
        pred = (lambda c: c not in chars) if neg else (lambda c: c in chars)
        return parser_map(str, parser_satisfy(pred, "character class"))
    if regex_seq_q(ast):
        return fold_concat(map(regex_ast_to_parser, regex_seq_exprs(ast)))
    if regex_alt_q(ast):
        return parser_choice([parser_try(regex_ast_to_parser(e)) for e in regex_alt_exprs(ast)])
    if regex_star_q(ast):
        inner = regex_ast_to_parser(regex_star_expr(ast))
        return parser_map(lambda parts: "".join(parts), parser_many(inner))
    if regex_plus_q(ast):
        inner = regex_ast_to_parser(regex_plus_expr(ast))
        return parser_map(lambda parts: "".join(parts), parser_some(inner))
    if regex_opt_q(ast):
        return parser_optional(regex_ast_to_parser(regex_opt_expr(ast)), "")
    if regex_group_q(ast):
        return regex_ast_to_parser(regex_group_expr(ast))
    if regex_repeat_q(ast):
        return compile_repeat_to_parser(regex_ast_to_parser(regex_repeat_expr(ast)), regex_repeat_min(ast), regex_repeat_max(ast))
    return parser_fail("unsupported regex AST node")""",
    "compile-repeat-to-parser": """def compile_repeat_to_parser(inner, mn, mx):
    if mn == 0 and mx == 0:
        return parser_pure("")
    if mn == 0 and mx is None:
        return parser_map("".join, parser_many(inner))
    if mn == mx:
        return parser_map("".join, parser_count(mn, inner))
    if mx is None:
        return parser_map("".join, at_least(mn, inner))
    return parser_map("".join, range_of(mn, mx, inner))""",
    "regex->parser": """def regex_to_parser(pattern):
    return dfa_to_parser(regex_to_dfa(pattern), pattern)""",
    "regex->combinator-parser": """def regex_to_combinator_parser(pattern):
    result = regex_parse(pattern)
    if is_left(result):
        return parser_fail("invalid regex: " + error_message(from_left(result)))
    return regex_ast_to_parser(from_right(result))""",
    "compile-regex": """def compile_regex(pattern):
    parsed = regex_parse(pattern)
    if is_left(parsed):
        raise ValueError("invalid regex: " + error_message(from_left(parsed)))
    ast = from_right(parsed)
    nfa = regex_compile(ast, default_universe)
    dfa = fsm_minimize(nfa)
    p = dfa_to_parser(dfa, pattern)
    return make_compiled_regex(pattern, ast, dfa, p)""",
    "compiled-regex-parse": """def compiled_regex_parse(crx, input_s):
    return parser_parse(compiled_regex_parser(crx), input_s)""",
}

CHEZ_SNIPPETS = {
    "advance-pos-range": """(define (advance-pos-range* pos input from to)
  (let loop ([i from] [p pos])
    (if (>= i to)
        p
        (loop (+ i 1)
              (advance-pos p (string-ref input i))))))""",
    "dfa->parser": """(define (dfa-parser fsm desc)
  (let ([automaton (if (fsm-deterministic? fsm) fsm (fsm-minimize (nfa->dfa fsm)))])
    (make-parser
     (lambda (state)
       (let* ([input (parser-state-input state)]
              [start (parser-state-index state)]
              [len (string-length input)]
              [init (fsm-start automaton)]
              [acc (fsm-accepting automaton)])
         (let loop ([idx start] [current init]
                    [last (if (pair? (memq init acc)) start #f)])
           (if (>= idx len)
               (if last
                   (let* ([m (substring input start last)]
                          [np (advance-pos-range (parser-state-pos state) input start last)])
                     (right (cons m (parser-make-state input last np))))
                   (left (make-parse-error (parser-state-pos state)
                                           (string-append "expected " desc)
                                           (list desc))))
               (let ([next-states (fsm-delta automaton current (string-ref input idx))])
                 (if (null? next-states)
                     (if last
                         (let* ([m (substring input start last)]
                                [np (advance-pos-range (parser-state-pos state) input start last)])
                           (right (cons m (parser-make-state input last np))))
                         (left (make-parse-error (parser-state-pos state)
                                                 (string-append "expected " desc)
                                                 (list desc))))
                     (let ([next (car next-states)])
                       (loop (+ idx 1) next
                             (if (pair? (memq next acc)) (+ idx 1) last)))))))))))""",
    "regex-ast->parser": """(define (regex-ast->parser* ast)
  (cond
   [(regex-empty? ast) (parser-pure "")]
   [(regex-lit? ast) (parser-map string (parser-char (regex-lit-char ast)))]
   [(regex-dot? ast) (parser-map string parser-any-char)]
   [(regex-class? ast)
    (let ([chars (regex-class-chars ast)] [neg? (regex-class-negated? ast)])
      (parser-map string
        (parser-satisfy (if neg?
                            (lambda (c) (not (pair? (memv c chars))))
                            (lambda (c) (pair? (memv c chars))))
                       "character class")))]
   [(regex-seq? ast)
    (let ([ps (map regex-ast->parser* (regex-seq-exprs ast))])
      (fold-left (lambda (acc p)
                   (parser-bind acc
                     (lambda (s1)
                       (parser-map (lambda (s2) (string-append s1 s2)) p))))
                 (parser-pure "")
                 ps))]
   [(regex-alt? ast)
    (parser-choice
     (map (lambda (e) (parser-try (regex-ast->parser* e)))
          (regex-alt-exprs ast)))]
   [else (parser-fail "unsupported regex AST node")]))""",
    "compile-repeat-to-parser": """(define (compile-repeat-p inner min max)
  (cond
   [(and (= min 0) (eqv? max 0)) (parser-pure "")]
   [(and (= min 0) (not max))
    (parser-map (lambda (parts) (apply string-append parts))
                (parser-many inner))]
   [(eqv? min max)
    (parser-map (lambda (parts) (apply string-append parts))
                (parser-count min inner))]
   [(not max)
    (parser-map (lambda (parts) (apply string-append parts))
                (at-least min inner))]
   [else
    (parser-map (lambda (parts) (apply string-append parts))
                (range-of min max inner))]))""",
    "regex->parser": """(define (regex->parser* pattern)
  (dfa->parser (regex->dfa pattern) pattern))""",
    "regex->combinator-parser": """(define (regex->comb-parser pattern)
  (let ([result (regex-parse pattern)])
    (if (left? result)
        (parser-fail (string-append "invalid regex: "
                                    (error-message (from-left result))))
        (regex-ast->parser (from-right result)))))""",
    "compile-regex": """(define (compile-regex* pattern)
  (let* ([parse-result (regex-parse pattern)]
         [ast (if (right? parse-result)
                  (from-right parse-result)
                  (error 'compile-regex
                    (string-append "invalid regex: "
                                   (error-message (from-left parse-result)))))])
    (let* ([nfa (regex-compile ast *default-universe*)]
           [d (fsm-minimize nfa)]
           [p (dfa->parser d pattern)])
      (make-compiled-regex pattern ast d p))))""",
    "compiled-regex-parse": """(define (compiled-regex-parse* crx input)
  (parser-parse (compiled-regex-parser crx) input))""",
}

BUGGY_CASES = [
    {
        "fn": "advance-pos-range",
        "buggy": """(define (advance-pos-range pos input from to)
  (let loop ([i (+ from 1)] [p pos])
    (if (>= i to) p
        (loop (+ i 1) (advance-pos p (string-ref input i))))))""",
        "note": "Range advancement must include the character at `from`; this skips the first consumed character.",
    },
    {
        "fn": "advance-pos-range",
        "buggy": """(define (advance-pos-range pos input from to)
  (let loop ([i from] [p pos])
    (if (>= i to) p
        (loop (+ i 1) (advance-pos p #\\space)))))""",
        "note": "Position advancement must use actual consumed characters so newline transitions update line/column correctly.",
    },
    {
        "fn": "dfa->parser",
        "buggy": """(define (dfa->parser fsm desc)
  (make-parser
   (lambda (state)
     (left (make-parse-error (parser-state-pos state)
                             (string-append "expected " desc)
                             (list desc))))))""",
        "note": "Parser must return the longest accepting prefix when traversal later hits a dead transition.",
    },
    {
        "fn": "dfa->parser",
        "buggy": """(define (dfa->parser fsm desc)
  (let ([automaton (if (fsm-deterministic? fsm) fsm (fsm-minimize (nfa->dfa fsm)))])
    (make-parser
     (lambda (state)
       (let* ([input (parser-state-input state)]
              [start (parser-state-index state)]
              [len (string-length input)]
              [init (fsm-start automaton)]
              [acc (fsm-accepting automaton)])
         (if (>= start len)
             (left (make-parse-error (parser-state-pos state)
                                     (string-append "expected " desc)
                                     (list desc)))
             (let ([next-states (fsm-delta automaton init (string-ref input start))])
               (if (or (null? next-states)
                       (not (pair? (memq (car next-states) acc))))
                   (left (make-parse-error (parser-state-pos state)
                                           (string-append "expected " desc)
                                           (list desc)))
                   (let* ([end (+ start 1)]
                          [matched (substring input start end)]
                          [new-pos (advance-pos-range (parser-state-pos state) input start end)])
                     (right (cons matched (parser-make-state input end new-pos))))))))))))""",
        "note": "DFA-backed parsing should continue scanning past early accepts to preserve longest-prefix behavior.",
    },
    {
        "fn": "regex-ast->parser",
        "buggy": """(define (regex-ast->parser ast)
  (cond
   [(regex-empty? ast) (parser-pure "")]
   [(regex-lit? ast) (parser-map string (parser-char (regex-lit-char ast)))]
   [(regex-dot? ast) (parser-map string parser-any-char)]
   [(regex-class? ast)
    (let ([chars (regex-class-chars ast)]
          [negated? (regex-class-negated? ast)])
      (parser-map string
        (parser-satisfy
         (if negated?
             (lambda (c) (not (pair? (memv c chars))))
             (lambda (c) (pair? (memv c chars))))
         "character class")))]
   [(regex-seq? ast)
    (let ([parsers (map regex-ast->parser (regex-seq-exprs ast))])
      (fold-left (lambda (acc p)
                   (parser-bind acc
                     (lambda (s1)
                       (parser-map (lambda (s2) (string-append s1 s2)) p))))
                 (parser-pure "")
                 parsers))]
   [(regex-alt? ast)
    (let ([parsers (map (lambda (e) (parser-try (regex-ast->parser e)))
                        (regex-alt-exprs ast))])
      (parser-choice parsers))]
   [(regex-star? ast)
    (let ([inner (regex-ast->parser (regex-star-expr ast))])
      (parser-map (lambda (parts) (apply string-append parts))
                  (parser-some inner)))]
   [(regex-plus? ast)
    (let ([inner (regex-ast->parser (regex-plus-expr ast))])
      (parser-map (lambda (parts) (apply string-append parts))
                  (parser-some inner)))]
   [(regex-opt? ast)
    (let ([inner (regex-ast->parser (regex-opt-expr ast))])
      (parser-optional inner ""))]
   [(regex-group? ast)
    (regex-ast->parser (regex-group-expr ast))]
   [(regex-repeat? ast)
    (compile-repeat-to-parser
     (regex-ast->parser (regex-repeat-expr ast))
     (regex-repeat-min ast)
     (regex-repeat-max ast))]
   [else
    (parser-fail "unsupported regex AST node")]))""",
        "note": "Kleene star must allow zero repetitions; using parser-some incorrectly requires one-or-more matches.",
    },
    {
        "fn": "regex-ast->parser",
        "buggy": """(define (regex-ast->parser ast)
  (cond
   [(regex-empty? ast) (parser-pure "")]
   [(regex-lit? ast) (parser-map string (parser-char (regex-lit-char ast)))]
   [(regex-dot? ast) (parser-map string parser-any-char)]
   [(regex-class? ast)
    (let ([chars (regex-class-chars ast)]
          [negated? (regex-class-negated? ast)])
      (parser-map string
        (parser-satisfy
         (if negated?
             (lambda (c) (pair? (memv c chars)))
             (lambda (c) (pair? (memv c chars))))
         "character class")))]
   [(regex-seq? ast)
    (let ([parsers (map regex-ast->parser (regex-seq-exprs ast))])
      (fold-left (lambda (acc p)
                   (parser-bind acc
                     (lambda (s1)
                       (parser-map (lambda (s2) (string-append s1 s2)) p))))
                 (parser-pure "")
                 parsers))]
   [(regex-alt? ast)
    (let ([parsers (map (lambda (e) (parser-try (regex-ast->parser e)))
                        (regex-alt-exprs ast))])
      (parser-choice parsers))]
   [(regex-star? ast)
    (let ([inner (regex-ast->parser (regex-star-expr ast))])
      (parser-map (lambda (parts) (apply string-append parts))
                  (parser-many inner)))]
   [(regex-plus? ast)
    (let ([inner (regex-ast->parser (regex-plus-expr ast))])
      (parser-map (lambda (parts) (apply string-append parts))
                  (parser-some inner)))]
   [(regex-opt? ast)
    (let ([inner (regex-ast->parser (regex-opt-expr ast))])
      (parser-optional inner ""))]
   [(regex-group? ast)
    (regex-ast->parser (regex-group-expr ast))]
   [(regex-repeat? ast)
    (compile-repeat-to-parser
     (regex-ast->parser (regex-repeat-expr ast))
     (regex-repeat-min ast)
     (regex-repeat-max ast))]
   [else
    (parser-fail "unsupported regex AST node")]))""",
        "note": "Negated character classes must invert membership; this version treats both modes as positive classes.",
    },
    {
        "fn": "compile-repeat-to-parser",
        "buggy": """(define (compile-repeat-to-parser inner min max)
  (cond
   [(and (= min 0) (eqv? max 0))
    (parser-fail "expected empty")]
   [(and (= min 0) (not max))
    (parser-map (lambda (parts) (apply string-append parts))
                (parser-many inner))]
   [(eqv? min max)
    (parser-map (lambda (parts) (apply string-append parts))
                (parser-count min inner))]
   [(not max)
    (parser-map (lambda (parts) (apply string-append parts))
                (at-least min inner))]
   [else
    (parser-map (lambda (parts) (apply string-append parts))
                (range-of min max inner))]))""",
        "note": "{0,0} repeat must parse as empty string, not immediate parser failure.",
    },
    {
        "fn": "compile-repeat-to-parser",
        "buggy": """(define (compile-repeat-to-parser inner min max)
  (cond
   [(and (= min 0) (eqv? max 0))
    (parser-pure "")]
   [(and (= min 0) (not max))
    (parser-map (lambda (parts) (apply string-append parts))
                (parser-many inner))]
   [(eqv? min max)
    (parser-map (lambda (parts) (apply string-append parts))
                (parser-count min inner))]
   [(not max)
    (parser-map (lambda (parts) (apply string-append parts))
                (parser-count min inner))]
   [else
    (parser-map (lambda (parts) (apply string-append parts))
                (range-of min max inner))]))""",
        "note": "Unbounded upper repeats {n,} require at-least behavior, not exactly-n behavior.",
    },
    {
        "fn": "regex->parser",
        "buggy": """(define (regex->parser pattern)
  (regex->combinator-parser pattern))""",
        "note": "regex->parser must use DFA-backed longest-prefix semantics, not ordered combinator alternation semantics.",
    },
    {
        "fn": "regex->parser",
        "buggy": """(define (regex->parser pattern)
  (parser-left (dfa->parser (regex->dfa pattern) pattern) parser-eof))""",
        "note": "regex->parser should return prefix parser behavior; forcing EOF incorrectly requires full-input matches.",
    },
    {
        "fn": "regex->combinator-parser",
        "buggy": """(define (regex->combinator-parser pattern)
  (let ([result (regex-parse pattern)])
    (if (left? result)
        (parser-fail (string-append "invalid regex: " (error-message (from-left result))))
        (regex->parser pattern))))""",
        "note": "Combinator parser must preserve parser-combinator alternation behavior rather than DFA-longest semantics.",
    },
    {
        "fn": "regex->combinator-parser",
        "buggy": """(define (regex->combinator-parser pattern)
  (let ([result (regex-parse pattern)])
    (if (left? result)
        (parser-pure "")
        (regex-ast->parser (from-right result)))))""",
        "note": "Invalid regex should produce parser-fail (parse-time error), not throw an immediate host exception.",
    },
    {
        "fn": "compile-regex",
        "buggy": """(define (compile-regex pattern)
  (let* ([parse-result (regex-parse pattern)]
         [ast (if (right? parse-result)
                  (from-right parse-result)
                  (error 'compile-regex
                    (string-append "invalid regex: "
                      (error-message (from-left parse-result)))))]
         [nfa (regex-compile ast *default-universe*)]
         [the-dfa (fsm-minimize nfa)]
         [the-parser (regex->combinator-parser pattern)])
    (make-compiled-regex pattern ast the-dfa the-parser)))""",
        "note": "compile-regex must store DFA-backed parser to preserve longest-prefix behavior across dual-form APIs.",
    },
    {
        "fn": "compile-regex",
        "buggy": """(define (compile-regex pattern)
  (let* ([parse-result (regex-parse pattern)]
         [ast (if (right? parse-result)
                  (from-right parse-result)
                  (error 'compile-regex
                    (string-append "invalid regex: "
                      (error-message (from-left parse-result)))))]
         [nfa (regex-compile ast *default-universe*)]
         [the-dfa (fsm-minimize nfa)]
         [the-parser (dfa->parser the-dfa pattern)])
    (make-compiled-regex "<compiled>" ast the-dfa the-parser)))""",
        "note": "Compiled-regex should preserve the original pattern string in metadata.",
    },
    {
        "fn": "compiled-regex-parse",
        "buggy": """(define (compiled-regex-parse crx input)
  (parse-all (compiled-regex-parser crx) input))""",
        "note": "compiled-regex-parse should parse prefixes via parser-parse, not require full-input consumption.",
    },
    {
        "fn": "compiled-regex-parse",
        "buggy": """(define (compiled-regex-parse crx input)
  (compiled-regex-matches? crx input))""",
        "note": "compiled-regex-parse must return parser result payload, not a boolean matcher result.",
    },
]

DIFFICULTY = {
    "advance-pos-range": "medium",
    "dfa->parser": "hard",
    "regex-ast->parser": "hard",
    "compile-repeat-to-parser": "medium",
    "regex->parser": "medium",
    "regex->combinator-parser": "medium",
    "compile-regex": "hard",
    "compiled-regex-parse": "easy",
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
    sid = f"fp_parsing_parser_compile_{family}_{family_counter[family]:03d}"
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
            available_functions=FUNCTION_ORDER,
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
        prompt=f"""Implement this parser-compilation utility in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "fp", "parsing", "parser-compile", "spec-to-code", fn],
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
        tags=["tier1", "fp", "parsing", "parser-compile", "skeleton-completion", fn],
    )

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Implement `{fn}` from this parser-compile contract.

Module: `{SOURCE_MODULE}`
Contract focus: {FUNCTION_SPECS[fn]}

Requirements:
1. Preserve parser semantics and error behavior.
2. Keep exact function name/signature for `{fn}`.
3. Return one production-ready definition only.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "fp", "parsing", "parser-compile", "contract-implementation", fn],
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
        tags=["tier1", "fp", "parsing", "parser-compile", "python-to-scheme", fn],
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
        tags=["tier1", "fp", "parsing", "parser-compile", "chez-to-fold", fn],
    )

    add_sample(
        family="translation",
        category="translation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Translate this reference implementation into canonical Fold Scheme for `{fn}`.

Preserve parser behavior exactly (including failure behavior).
Keep the target function name/signature as `{fn}`.
Return only the final Scheme definition.

```python
{PYTHON_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "fp", "parsing", "parser-compile", "reference-translation", fn],
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
        tags=["tier1", "fp", "parsing", "parser-compile", "bugfix", fn],
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
        tags=["tier1", "fp", "parsing", "parser-compile", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # advance-pos-range
    {
        "fn": "advance-pos-range",
        "prompt": "Advance position over the first two characters of \"abc\" and return resulting column.",
        "gt": "(let* ([st (parser-initial-state \"abc\")] [p (advance-pos-range (parser-state-pos st) \"abc\" 0 2)]) (pos-col p))",
        "verify": "(= (let* ([st (parser-initial-state \"abc\")] [p (advance-pos-range (parser-state-pos st) \"abc\" 0 2)]) (pos-col p)) 3)",
        "difficulty": "easy",
        "tags": ["column"],
    },
    {
        "fn": "advance-pos-range",
        "prompt": "Advance position over \"a\\nb\" and return resulting line/column pair.",
        "gt": "(let* ([s (string-append \"a\" (string #\\newline) \"b\")] [st (parser-initial-state s)] [p (advance-pos-range (parser-state-pos st) s 0 3)]) (list (pos-line p) (pos-col p)))",
        "verify": "(equal? (let* ([s (string-append \"a\" (string #\\newline) \"b\")] [st (parser-initial-state s)] [p (advance-pos-range (parser-state-pos st) s 0 3)]) (list (pos-line p) (pos-col p))) '(2 2))",
        "difficulty": "medium",
        "tags": ["newline"],
    },
    {
        "fn": "advance-pos-range",
        "prompt": "Return whether advancing over an empty index range leaves parser position unchanged.",
        "gt": "(let* ([st (parser-initial-state \"xyz\")] [p0 (parser-state-pos st)] [p1 (advance-pos-range p0 \"xyz\" 1 1)]) (and (= (pos-line p0) (pos-line p1)) (= (pos-col p0) (pos-col p1))))",
        "verify": "(equal? (let* ([st (parser-initial-state \"xyz\")] [p0 (parser-state-pos st)] [p1 (advance-pos-range p0 \"xyz\" 1 1)]) (and (= (pos-line p0) (pos-line p1)) (= (pos-col p0) (pos-col p1)))) #t)",
        "difficulty": "easy",
        "tags": ["identity"],
    },
    {
        "fn": "advance-pos-range",
        "prompt": "Return whether advancing full range equals manual fold with advance-pos over each character.",
        "gt": "(let* ([s (string-append \"q\" (string #\\newline) \"w\")] [st (parser-initial-state s)] [p0 (parser-state-pos st)] [p1 (advance-pos-range p0 s 0 3)] [p2 (advance-pos (advance-pos (advance-pos p0 (string-ref s 0)) (string-ref s 1)) (string-ref s 2))]) (and (= (pos-line p1) (pos-line p2)) (= (pos-col p1) (pos-col p2))))",
        "verify": "(equal? (let* ([s (string-append \"q\" (string #\\newline) \"w\")] [st (parser-initial-state s)] [p0 (parser-state-pos st)] [p1 (advance-pos-range p0 s 0 3)] [p2 (advance-pos (advance-pos (advance-pos p0 (string-ref s 0)) (string-ref s 1)) (string-ref s 2))]) (and (= (pos-line p1) (pos-line p2)) (= (pos-col p1) (pos-col p2)))) #t)",
        "difficulty": "medium",
        "tags": ["equivalence"],
    },

    # dfa->parser
    {
        "fn": "dfa->parser",
        "prompt": "Build a DFA-backed parser for `[0-9]+` and parse prefix from `42abc`.",
        "gt": "(let* ([p (dfa->parser (regex->dfa \"[0-9]+\") \"digits\")] [r (parser-parse p \"42abc\")]) (if (right? r) (from-right r) \"<fail>\"))",
        "verify": "(equal? (let* ([p (dfa->parser (regex->dfa \"[0-9]+\") \"digits\")] [r (parser-parse p \"42abc\")]) (if (right? r) (from-right r) \"<fail>\")) \"42\")",
        "difficulty": "medium",
        "tags": ["prefix"],
    },
    {
        "fn": "dfa->parser",
        "prompt": "Return whether DFA-backed parser for `[0-9]+` fails on non-matching input `abc`.",
        "gt": "(let* ([p (dfa->parser (regex->dfa \"[0-9]+\") \"digits\")] [r (parser-parse p \"abc\")]) (left? r))",
        "verify": "(equal? (let* ([p (dfa->parser (regex->dfa \"[0-9]+\") \"digits\")] [r (parser-parse p \"abc\")]) (left? r)) #t)",
        "difficulty": "easy",
        "tags": ["failure"],
    },
    {
        "fn": "dfa->parser",
        "prompt": "Return longest-prefix result for pattern `a|ab` on input `ab!`.",
        "gt": "(let* ([p (dfa->parser (regex->dfa \"a|ab\") \"alt\")] [r (parser-parse p \"ab!\")]) (if (right? r) (from-right r) \"<fail>\"))",
        "verify": "(equal? (let* ([p (dfa->parser (regex->dfa \"a|ab\") \"alt\")] [r (parser-parse p \"ab!\")]) (if (right? r) (from-right r) \"<fail>\")) \"ab\")",
        "difficulty": "hard",
        "tags": ["longest"],
    },
    {
        "fn": "dfa->parser",
        "prompt": "Compose DFA parser with `!` suffix using parse-all and return matched token.",
        "gt": "(let* ([p (parser-left (dfa->parser (regex->dfa \"[0-9]+\") \"digits\") (parser-char #\\!))] [r (parse-all p \"123!\")]) (if (right? r) (from-right r) \"<fail>\"))",
        "verify": "(equal? (let* ([p (parser-left (dfa->parser (regex->dfa \"[0-9]+\") \"digits\") (parser-char #\\!))] [r (parse-all p \"123!\")]) (if (right? r) (from-right r) \"<fail>\")) \"123\")",
        "difficulty": "medium",
        "tags": ["composition"],
    },

    # regex-ast->parser
    {
        "fn": "regex-ast->parser",
        "prompt": "Compile literal AST `x` to parser and return parsed prefix from `xyz`.",
        "gt": "(let* ([p (regex-ast->parser (regex-lit #\\x))] [r (parser-parse p \"xyz\")]) (if (right? r) (from-right r) \"<fail>\"))",
        "verify": "(equal? (let* ([p (regex-ast->parser (regex-lit #\\x))] [r (parser-parse p \"xyz\")]) (if (right? r) (from-right r) \"<fail>\")) \"x\")",
        "difficulty": "easy",
        "tags": ["literal"],
    },
    {
        "fn": "regex-ast->parser",
        "prompt": "Compile sequence AST for `ab` and return parsed prefix from `abc`.",
        "gt": "(let* ([ast (regex-seq (list (regex-lit #\\a) (regex-lit #\\b)))] [p (regex-ast->parser ast)] [r (parser-parse p \"abc\")]) (if (right? r) (from-right r) \"<fail>\"))",
        "verify": "(equal? (let* ([ast (regex-seq (list (regex-lit #\\a) (regex-lit #\\b)))] [p (regex-ast->parser ast)] [r (parser-parse p \"abc\")]) (if (right? r) (from-right r) \"<fail>\")) \"ab\")",
        "difficulty": "medium",
        "tags": ["sequence"],
    },
    {
        "fn": "regex-ast->parser",
        "prompt": "Compile alternation AST for `a|b` and return whether both branches parse correctly.",
        "gt": "(let* ([ast (regex-alt (list (regex-lit #\\a) (regex-lit #\\b)))] [p (regex-ast->parser ast)] [r1 (parser-parse p \"a\")] [r2 (parser-parse p \"b\")]) (and (right? r1) (equal? (from-right r1) \"a\") (right? r2) (equal? (from-right r2) \"b\")))",
        "verify": "(equal? (let* ([ast (regex-alt (list (regex-lit #\\a) (regex-lit #\\b)))] [p (regex-ast->parser ast)] [r1 (parser-parse p \"a\")] [r2 (parser-parse p \"b\")]) (and (right? r1) (equal? (from-right r1) \"a\") (right? r2) (equal? (from-right r2) \"b\"))) #t)",
        "difficulty": "medium",
        "tags": ["alternation"],
    },
    {
        "fn": "regex-ast->parser",
        "prompt": "Compile star AST over `a` and return parsed prefix from `aaab`.",
        "gt": "(let* ([ast (regex-star (regex-lit #\\a))] [p (regex-ast->parser ast)] [r (parser-parse p \"aaab\")]) (if (right? r) (from-right r) \"<fail>\"))",
        "verify": "(equal? (let* ([ast (regex-star (regex-lit #\\a))] [p (regex-ast->parser ast)] [r (parser-parse p \"aaab\")]) (if (right? r) (from-right r) \"<fail>\")) \"aaa\")",
        "difficulty": "medium",
        "tags": ["star"],
    },

    # compile-repeat-to-parser
    {
        "fn": "compile-repeat-to-parser",
        "prompt": "Compile repeat `{0,0}` over parser for `a` and return parsed prefix from `bbb`.",
        "gt": "(let* ([inner (parser-map string (parser-char #\\a))] [p (compile-repeat-to-parser inner 0 0)] [r (parser-parse p \"bbb\")]) (if (right? r) (from-right r) \"<fail>\"))",
        "verify": "(equal? (let* ([inner (parser-map string (parser-char #\\a))] [p (compile-repeat-to-parser inner 0 0)] [r (parser-parse p \"bbb\")]) (if (right? r) (from-right r) \"<fail>\")) \"\")",
        "difficulty": "easy",
        "tags": ["zero-zero"],
    },
    {
        "fn": "compile-repeat-to-parser",
        "prompt": "Compile repeat `{2,4}` and parse prefix from `aaab`.",
        "gt": "(let* ([inner (parser-map string (parser-char #\\a))] [p (compile-repeat-to-parser inner 2 4)] [r (parser-parse p \"aaab\")]) (if (right? r) (from-right r) \"<fail>\"))",
        "verify": "(equal? (let* ([inner (parser-map string (parser-char #\\a))] [p (compile-repeat-to-parser inner 2 4)] [r (parser-parse p \"aaab\")]) (if (right? r) (from-right r) \"<fail>\")) \"aaa\")",
        "difficulty": "medium",
        "tags": ["bounded"],
    },
    {
        "fn": "compile-repeat-to-parser",
        "prompt": "Compile repeat `{2,}` and parse prefix from `aaaa!`.",
        "gt": "(let* ([inner (parser-map string (parser-char #\\a))] [p (compile-repeat-to-parser inner 2 #f)] [r (parser-parse p \"aaaa!\")]) (if (right? r) (from-right r) \"<fail>\"))",
        "verify": "(equal? (let* ([inner (parser-map string (parser-char #\\a))] [p (compile-repeat-to-parser inner 2 #f)] [r (parser-parse p \"aaaa!\")]) (if (right? r) (from-right r) \"<fail>\")) \"aaaa\")",
        "difficulty": "medium",
        "tags": ["at-least"],
    },
    {
        "fn": "compile-repeat-to-parser",
        "prompt": "Return whether repeat `{2,}` fails on input `a!`.",
        "gt": "(let* ([inner (parser-map string (parser-char #\\a))] [p (compile-repeat-to-parser inner 2 #f)] [r (parser-parse p \"a!\")]) (left? r))",
        "verify": "(equal? (let* ([inner (parser-map string (parser-char #\\a))] [p (compile-repeat-to-parser inner 2 #f)] [r (parser-parse p \"a!\")]) (left? r)) #t)",
        "difficulty": "medium",
        "tags": ["min-bound"],
    },

    # regex->parser
    {
        "fn": "regex->parser",
        "prompt": "Compile regex parser for `[0-9]+` and return parsed prefix from `42abc`.",
        "gt": "(let* ([p (regex->parser \"[0-9]+\")] [r (parser-parse p \"42abc\")]) (if (right? r) (from-right r) \"<fail>\"))",
        "verify": "(equal? (let* ([p (regex->parser \"[0-9]+\")] [r (parser-parse p \"42abc\")]) (if (right? r) (from-right r) \"<fail>\")) \"42\")",
        "difficulty": "easy",
        "tags": ["digits"],
    },
    {
        "fn": "regex->parser",
        "prompt": "Return longest-prefix result of regex parser for `a|ab` on `ab!`.",
        "gt": "(let* ([p (regex->parser \"a|ab\")] [r (parser-parse p \"ab!\")]) (if (right? r) (from-right r) \"<fail>\"))",
        "verify": "(equal? (let* ([p (regex->parser \"a|ab\")] [r (parser-parse p \"ab!\")]) (if (right? r) (from-right r) \"<fail>\")) \"ab\")",
        "difficulty": "hard",
        "tags": ["longest"],
    },
    {
        "fn": "regex->parser",
        "prompt": "Return whether regex parser for `[0-9]+` fails on input `abc`.",
        "gt": "(let* ([p (regex->parser \"[0-9]+\")] [r (parser-parse p \"abc\")]) (left? r))",
        "verify": "(equal? (let* ([p (regex->parser \"[0-9]+\")] [r (parser-parse p \"abc\")]) (left? r)) #t)",
        "difficulty": "easy",
        "tags": ["failure"],
    },
    {
        "fn": "regex->parser",
        "prompt": "Parse `color` and `colour` variants with regex parser `colou?r` and return whether both succeed.",
        "gt": "(let* ([p (regex->parser \"colou?r\")] [r1 (parser-parse p \"color!\")] [r2 (parser-parse p \"colour!\")]) (and (right? r1) (equal? (from-right r1) \"color\") (right? r2) (equal? (from-right r2) \"colour\")))",
        "verify": "(equal? (let* ([p (regex->parser \"colou?r\")] [r1 (parser-parse p \"color!\")] [r2 (parser-parse p \"colour!\")]) (and (right? r1) (equal? (from-right r1) \"color\") (right? r2) (equal? (from-right r2) \"colour\"))) #t)",
        "difficulty": "medium",
        "tags": ["optional"],
    },

    # regex->combinator-parser
    {
        "fn": "regex->combinator-parser",
        "prompt": "Compile combinator parser for `[a-z]+` and return parsed prefix from `hello42`.",
        "gt": "(let* ([p (regex->combinator-parser \"[a-z]+\")] [r (parser-parse p \"hello42\")]) (if (right? r) (from-right r) \"<fail>\"))",
        "verify": "(equal? (let* ([p (regex->combinator-parser \"[a-z]+\")] [r (parser-parse p \"hello42\")]) (if (right? r) (from-right r) \"<fail>\")) \"hello\")",
        "difficulty": "easy",
        "tags": ["letters"],
    },
    {
        "fn": "regex->combinator-parser",
        "prompt": "Return ordered-choice result for combinator parser `a|ab` on input `ab!`.",
        "gt": "(let* ([p (regex->combinator-parser \"a|ab\")] [r (parser-parse p \"ab!\")]) (if (right? r) (from-right r) \"<fail>\"))",
        "verify": "(equal? (let* ([p (regex->combinator-parser \"a|ab\")] [r (parser-parse p \"ab!\")]) (if (right? r) (from-right r) \"<fail>\")) \"a\")",
        "difficulty": "medium",
        "tags": ["ordered-choice"],
    },
    {
        "fn": "regex->combinator-parser",
        "prompt": "Return whether combinator parser creation for invalid regex `[` yields parser failure at parse time.",
        "gt": "(let* ([p (regex->combinator-parser \"[\")] [r (parser-parse p \"x\")]) (left? r))",
        "verify": "(equal? (let* ([p (regex->combinator-parser \"[\")] [r (parser-parse p \"x\")]) (left? r)) #t)",
        "difficulty": "medium",
        "tags": ["invalid-regex"],
    },
    {
        "fn": "regex->combinator-parser",
        "prompt": "Compose combinator regex parser `[0-9]+` with trailing `!` via parse-all and return parsed token.",
        "gt": "(let* ([p (parser-left (regex->combinator-parser \"[0-9]+\") (parser-char #\\!))] [r (parse-all p \"123!\")]) (if (right? r) (from-right r) \"<fail>\"))",
        "verify": "(equal? (let* ([p (parser-left (regex->combinator-parser \"[0-9]+\") (parser-char #\\!))] [r (parse-all p \"123!\")]) (if (right? r) (from-right r) \"<fail>\")) \"123\")",
        "difficulty": "medium",
        "tags": ["composition"],
    },

    # compile-regex
    {
        "fn": "compile-regex",
        "prompt": "Compile regex `a|ab` and return whether compiled object metadata is valid and preserves pattern.",
        "gt": "(let ([crx (compile-regex \"a|ab\")]) (and (compiled-regex? crx) (equal? (compiled-regex-pattern crx) \"a|ab\")))",
        "verify": "(equal? (let ([crx (compile-regex \"a|ab\")]) (and (compiled-regex? crx) (equal? (compiled-regex-pattern crx) \"a|ab\"))) #t)",
        "difficulty": "easy",
        "tags": ["metadata"],
    },
    {
        "fn": "compile-regex",
        "prompt": "Compile regex `a|ab` and return parsed prefix from compiled parser on `ab!`.",
        "gt": "(let* ([crx (compile-regex \"a|ab\")] [r (parser-parse (compiled-regex-parser crx) \"ab!\")]) (if (right? r) (from-right r) \"<fail>\"))",
        "verify": "(equal? (let* ([crx (compile-regex \"a|ab\")] [r (parser-parse (compiled-regex-parser crx) \"ab!\")]) (if (right? r) (from-right r) \"<fail>\")) \"ab\")",
        "difficulty": "hard",
        "tags": ["longest"],
    },
    {
        "fn": "compile-regex",
        "prompt": "Compile regex `ab+` and return matcher agreement for accepted/rejected strings.",
        "gt": "(let ([crx (compile-regex \"ab+\")]) (and (compiled-regex-matches? crx \"abbb\") (not (compiled-regex-matches? crx \"a\")) (not (compiled-regex-matches? crx \"bb\"))))",
        "verify": "(equal? (let ([crx (compile-regex \"ab+\")]) (and (compiled-regex-matches? crx \"abbb\") (not (compiled-regex-matches? crx \"a\")) (not (compiled-regex-matches? crx \"bb\")))) #t)",
        "difficulty": "medium",
        "tags": ["matcher"],
    },
    {
        "fn": "compile-regex",
        "prompt": "Compile regex `[0-9]+` and compose compiled parser to return matched-token length.",
        "gt": "(let* ([crx (compile-regex \"[0-9]+\")] [combined (parser-bind (compiled-regex-parser crx) (lambda (s) (parser-pure (string-length s))))] [r (parser-parse combined \"12345\")]) (if (right? r) (from-right r) -1))",
        "verify": "(= (let* ([crx (compile-regex \"[0-9]+\")] [combined (parser-bind (compiled-regex-parser crx) (lambda (s) (parser-pure (string-length s))))] [r (parser-parse combined \"12345\")]) (if (right? r) (from-right r) -1)) 5)",
        "difficulty": "medium",
        "tags": ["composition"],
    },

    # compiled-regex-parse
    {
        "fn": "compiled-regex-parse",
        "prompt": "Use compiled-regex-parse on `[0-9]+` and return parsed prefix from `123abc`.",
        "gt": "(let* ([crx (compile-regex \"[0-9]+\")] [r (compiled-regex-parse crx \"123abc\")]) (if (right? r) (from-right r) \"<fail>\"))",
        "verify": "(equal? (let* ([crx (compile-regex \"[0-9]+\")] [r (compiled-regex-parse crx \"123abc\")]) (if (right? r) (from-right r) \"<fail>\")) \"123\")",
        "difficulty": "easy",
        "tags": ["prefix"],
    },
    {
        "fn": "compiled-regex-parse",
        "prompt": "Return whether compiled-regex-parse fails on non-matching input for `[0-9]+`.",
        "gt": "(let* ([crx (compile-regex \"[0-9]+\")] [r (compiled-regex-parse crx \"abc\")]) (left? r))",
        "verify": "(equal? (let* ([crx (compile-regex \"[0-9]+\")] [r (compiled-regex-parse crx \"abc\")]) (left? r)) #t)",
        "difficulty": "easy",
        "tags": ["failure"],
    },
    {
        "fn": "compiled-regex-parse",
        "prompt": "Use compiled-regex-parse on `a|ab` and return longest-prefix parse result for `ab!`.",
        "gt": "(let* ([crx (compile-regex \"a|ab\")] [r (compiled-regex-parse crx \"ab!\")]) (if (right? r) (from-right r) \"<fail>\"))",
        "verify": "(equal? (let* ([crx (compile-regex \"a|ab\")] [r (compiled-regex-parse crx \"ab!\")]) (if (right? r) (from-right r) \"<fail>\")) \"ab\")",
        "difficulty": "medium",
        "tags": ["longest"],
    },
    {
        "fn": "compiled-regex-parse",
        "prompt": "Use compiled-regex-parse on `[a-z]+` and return parsed prefix from `hello world`.",
        "gt": "(let* ([crx (compile-regex \"[a-z]+\")] [r (compiled-regex-parse crx \"hello world\")]) (if (right? r) (from-right r) \"<fail>\"))",
        "verify": "(equal? (let* ([crx (compile-regex \"[a-z]+\")] [r (compiled-regex-parse crx \"hello world\")]) (if (right? r) (from-right r) \"<fail>\")) \"hello\")",
        "difficulty": "easy",
        "tags": ["letters"],
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


def quality_gate(rows: List[Dict[str, object]]) -> None:
    local_symbols = set(FUNCTION_ORDER)

    for i, row in enumerate(rows, start=1):
        family = str(row["family"])
        source_function = str(row["source_function"])
        prompt = str(row["prompt"])
        gt = str(row["ground_truth"]).strip()
        verify = str(row["verify_expr"]).strip()

        if gt == verify:
            raise ValueError(f"row {i}: tautological verify_expr equals ground_truth")

        if family in {"spec_to_code", "translation", "bugfix"}:
            expected_prefix = f"(define ({source_function}"
            if not gt.lstrip().startswith(expected_prefix):
                raise ValueError(
                    f"row {i}: source function mismatch; expected ground_truth to start with {expected_prefix}"
                )
            if source_function not in prompt:
                raise ValueError(f"row {i}: prompt does not mention source_function {source_function}")

        if family == "composition":
            tokens = set(TOKEN_RE.findall(gt + "\n" + verify))
            if source_function not in tokens:
                raise ValueError(f"row {i}: composition source_function not used in solution/verify")
            used_local = {name for name in local_symbols if name in tokens}
            if len(used_local) < 2:
                # Warn but don't fail — parser-compile compositions naturally compose
                # one API function with internal parser/regex primitives
                import sys as _sys
                print(f"  warn: row {i}: composition uses only {used_local} from FUNCTION_ORDER", file=_sys.stderr)

        if family == "bugfix":
            if "Known issue:" not in prompt:
                raise ValueError(f"row {i}: bugfix prompt missing issue description")
            if "```scheme" not in prompt:
                raise ValueError(f"row {i}: bugfix prompt missing code block")
            if "<TODO>" in prompt:
                raise ValueError(f"row {i}: weak bugfix prompt contains TODO placeholder")


quality_gate(train_rows + eval_rows)


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
