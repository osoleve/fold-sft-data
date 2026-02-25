#!/usr/bin/env python3
"""Generate Tier-1 FP parsing FSM SFT samples for lattice/fp/parsing/fsm.ss."""

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

SOURCE_MODULE = "lattice/fp/parsing/fsm.ss"
SOURCE_TEST = "lattice/fp/parsing/test-fsm.ss"

DEFS: Dict[str, str] = {
    "fsm-delta": """(define (fsm-delta fsm state input)
  (let ([key (cons state input)])
       (let ([found (assoc key (fsm-transitions fsm))])
            (if found (cdr found) '()))))""",
    "epsilon-closure": """(define (epsilon-closure fsm state)
  (let loop ([frontier (list state)] [visited '()])
       (if (null? frontier)
           visited
           (let ([s (car frontier)])
                (if (member s visited)
                    (loop (cdr frontier) visited)
                    (let* ([eps-targets (get-all-epsilon-targets fsm s)]
                           [new-frontier (append eps-targets (cdr frontier))])
                          (loop new-frontier (cons s visited))))))))""",
    "fsm-move": """(define (fsm-move fsm states input)
  (let* ([direct-targets
          (fold-left (lambda (acc s)
                             (union equal? acc (fsm-delta fsm s input)))
                     '()
                     states)])
        (epsilon-closure-set fsm direct-targets)))""",
    "fsm-accepts?": """(define (fsm-accepts? fsm input)
  (if (null? (fsm-assertions fsm))
      (just? (fsm-run fsm input))
      (just? (fsm-run-with-assertions fsm input))))""",
    "fsm-char": """(define (fsm-char c)
  (let ([s0 (fsm-fresh-state "q")]
        [s1 (fsm-fresh-state "q")])
       (make-fsm (list s0 s1)
                 (list c)
                 (list (cons (cons s0 c) (list s1)))
                 s0
                 (list s1))))""",
    "fsm-literal": """(define (fsm-literal str)
  (let ([chars (string->list str)])
       (if (null? chars)
           (fsm-epsilon-lang)
           (fold-left fsm-concat
                      (fsm-char (car chars))
                      (map fsm-char (cdr chars))))))""",
    "nfa->dfa": """(define (nfa->dfa nfa)
  (if (not (null? (fsm-assertions nfa)))
      nfa
      (let* ([alphabet (fsm-alphabet nfa)]
             [start-set (epsilon-closure nfa (fsm-start nfa))]
             [start-name (state-set->name start-set)])
        (let loop ([worklist (list start-set)]
                   [visited '()]
                   [dfa-states (list start-name)]
                   [dfa-trans '()]
                   [dfa-accepting '()])
             (if (null? worklist)
                 (make-fsm dfa-states alphabet dfa-trans start-name dfa-accepting)
                 (let ([current (car worklist)])
                      (if (member current visited)
                          (loop (cdr worklist) visited dfa-states dfa-trans dfa-accepting)
                          (let* ([current-name (state-set->name current)]
                                 [is-accepting
                                  (exists (lambda (s) (member s (fsm-accepting nfa))) current)]
                                 [new-trans-and-states
                                  (map (lambda (sym)
                                               (let* ([target-set (fsm-move nfa current sym)]
                                                      [target-name (if (null? target-set)
                                                                       'dead
                                                                       (state-set->name target-set))])
                                                     (list sym target-set target-name)))
                                       alphabet)]
                                 [valid-trans (filter (lambda (x) (not (null? (cadr x))))
                                                      new-trans-and-states)]
                                 [new-trans (map (lambda (x)
                                                         (cons (cons current-name (car x))
                                                               (list (caddr x))))
                                                 valid-trans)]
                                 [new-state-sets (map cadr valid-trans)]
                                 [new-state-names (map caddr valid-trans)])
                                (loop (append new-state-sets (cdr worklist))
                                      (cons current visited)
                                      (union equal? new-state-names dfa-states)
                                      (append new-trans dfa-trans)
                                      (if is-accepting
                                          (cons current-name dfa-accepting)
                                          dfa-accepting))))))))))""",
    "fsm-complement": """(define (fsm-complement dfa)
  (let* ([m0 (if (fsm-deterministic? dfa) dfa (nfa->dfa dfa))]
         [m (fsm-complete m0)]
         [non-accepting (filter (lambda (s) (not (member s (fsm-accepting m))))
                                (fsm-states m))])
        (make-fsm (fsm-states m) (fsm-alphabet m) (fsm-transitions m)
                  (fsm-start m) non-accepting)))""",
}

FUNCTION_ORDER = [
    "fsm-delta",
    "epsilon-closure",
    "fsm-move",
    "fsm-accepts?",
    "fsm-char",
    "fsm-literal",
    "nfa->dfa",
    "fsm-complement",
]

FUNCTION_SPECS = {
    "fsm-delta": "Return outgoing target states for a single (state,input) transition key, else empty list.",
    "epsilon-closure": "Compute epsilon-reachable states from a start state using graph traversal without revisits.",
    "fsm-move": "Move from a set of states on one symbol and then epsilon-close the result set.",
    "fsm-accepts?": "Return boolean acceptance; delegate to assertion-aware run when assertions are present.",
    "fsm-char": "Build an FSM that accepts exactly one character and nothing else.",
    "fsm-literal": "Build an FSM that accepts exactly a literal string (empty literal accepts empty only).",
    "nfa->dfa": "Convert assertion-free NFA to DFA via subset construction while preserving language.",
    "fsm-complement": "Complement a language by determinizing/completing the machine then flipping accepting states.",
}

SKELETONS = {
    "fsm-delta": """(define (fsm-delta fsm state input)
  ;; TODO: find transitions for (state,input), return target-state list or empty list
  <TODO>)""",
    "epsilon-closure": """(define (epsilon-closure fsm state)
  ;; TODO: DFS/BFS over epsilon transitions with visited tracking
  <TODO>)""",
    "fsm-move": """(define (fsm-move fsm states input)
  ;; TODO: collect direct transitions from each state then epsilon-close
  <TODO>)""",
    "fsm-accepts?": """(define (fsm-accepts? fsm input)
  ;; TODO: use fsm-run or fsm-run-with-assertions depending on assertion presence
  <TODO>)""",
    "fsm-char": """(define (fsm-char c)
  ;; TODO: construct two-state FSM that accepts exactly char c
  <TODO>)""",
    "fsm-literal": """(define (fsm-literal str)
  ;; TODO: fold concatenation of char-machines; handle empty string
  <TODO>)""",
    "nfa->dfa": """(define (nfa->dfa nfa)
  ;; TODO: subset construction over epsilon-closed state-sets
  <TODO>)""",
    "fsm-complement": """(define (fsm-complement dfa)
  ;; TODO: determinize+complete then flip accepting states
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "fsm-delta": "(let ([m (dfa '(q0 q1) '(#\\a #\\b) '((q0 #\\a q1) (q1 #\\b q0)) 'q0 '(q1))]) (and (equal? (fsm-delta m 'q0 #\\a) '(q1)) (equal? (fsm-delta m 'q1 #\\a) '())))",
    "epsilon-closure": "(let* ([m (make-fsm '(q0 q1 q2) '(#\\a) '() 'q0 '(q2) '((q0 q1) (q1 q2)))] [c (epsilon-closure m 'q0)]) (and (= (length c) 3) (not (not (member 'q0 c))) (not (not (member 'q1 c))) (not (not (member 'q2 c)))))",
    "fsm-move": "(let* ([m (make-fsm '(q0 q1 q2) '(#\\a) (list (cons (cons 'q0 #\\a) '(q1))) 'q0 '(q2) '((q1 q2)))] [res (fsm-move m '(q0) #\\a)]) (and (= (length res) 2) (not (not (member 'q1 res))) (not (not (member 'q2 res)))))",
    "fsm-accepts?": "(let* ([plain (fsm-concat (fsm-char #\\a) (fsm-char #\\b))] [anchored (make-fsm-with-assertions '(q0 q1) '() '() 'q0 '(q1) '() '((q0 anchor start q1)))]) (and (fsm-accepts? plain \"ab\") (not (fsm-accepts? plain \"a\")) (not (fsm-accepts? plain \"b\")) (not (fsm-accepts? plain \"\")) (fsm-accepts? anchored \"\") (not (fsm-accepts? anchored \"a\"))))",
    "fsm-char": "(let ([m (fsm-char #\\x)]) (and (fsm-accepts? m \"x\") (not (fsm-accepts? m \"\")) (not (fsm-accepts? m \"y\")) (not (fsm-accepts? m \"xx\"))))",
    "fsm-literal": "(let ([m (fsm-literal \"hello\")] [e (fsm-literal \"\")]) (and (fsm-accepts? m \"hello\") (not (fsm-accepts? m \"hell\")) (not (fsm-accepts? m \"helloo\")) (not (fsm-accepts? m \"\")) (fsm-accepts? e \"\") (not (fsm-accepts? e \"a\"))))",
    "nfa->dfa": "(let* ([nfa-m (fsm-concat (fsm-plus (fsm-char #\\a)) (fsm-char #\\b))] [dfa-m (nfa->dfa nfa-m)]) (and (fsm-deterministic? dfa-m) (fsm-accepts? dfa-m \"ab\") (fsm-accepts? dfa-m \"aab\") (fsm-accepts? dfa-m \"aaab\") (not (fsm-accepts? dfa-m \"\")) (not (fsm-accepts? dfa-m \"a\")) (not (fsm-accepts? dfa-m \"b\"))))",
    "fsm-complement": "(let* ([accepts-a (fsm-literal \"a\")] [comp (fsm-complement accepts-a)]) (and (not (fsm-accepts? comp \"a\")) (fsm-accepts? comp \"\") (fsm-accepts? comp \"aa\")))",
}

TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "fsm-delta": """def fsm_delta(fsm, state, inp):
    key = (state, inp)
    found = fsm.transitions.get(key)
    return found if found is not None else []""",
    "epsilon-closure": """def epsilon_closure(fsm, state):
    frontier = [state]
    visited = []
    while frontier:
        s = frontier.pop(0)
        if s in visited:
            continue
        visited.append(s)
        frontier = fsm.epsilon_targets(s) + frontier
    return visited""",
    "fsm-move": """def fsm_move(fsm, states, inp):
    direct = []
    for s in states:
        direct = union(direct, fsm_delta(fsm, s, inp))
    return epsilon_closure_set(fsm, direct)""",
    "fsm-accepts?": """def fsm_accepts_q(fsm, inp):
    if not fsm.assertions:
        return is_just(fsm_run(fsm, inp))
    return is_just(fsm_run_with_assertions(fsm, inp))""",
    "fsm-char": """def fsm_char(c):
    s0 = fsm_fresh_state('q')
    s1 = fsm_fresh_state('q')
    return make_fsm([s0, s1], [c], [((s0, c), [s1])], s0, [s1])""",
    "fsm-literal": """def fsm_literal(s):
    chars = list(s)
    if not chars:
        return fsm_epsilon_lang()
    acc = fsm_char(chars[0])
    for c in chars[1:]:
        acc = fsm_concat(acc, fsm_char(c))
    return acc""",
    "nfa->dfa": """def nfa_to_dfa(nfa):
    if nfa.assertions:
        return nfa
    alphabet = nfa.alphabet
    start_set = epsilon_closure(nfa, nfa.start)
    start_name = state_set_to_name(start_set)
    worklist = [start_set]
    visited = []
    dfa_states = [start_name]
    dfa_trans = []
    dfa_accepting = []
    while worklist:
        current = worklist.pop(0)
        if current in visited:
            continue
        visited.append(current)
        current_name = state_set_to_name(current)
        if any(s in nfa.accepting for s in current):
            dfa_accepting.append(current_name)
        for sym in alphabet:
            target_set = fsm_move(nfa, current, sym)
            if not target_set:
                continue
            target_name = state_set_to_name(target_set)
            dfa_trans.append(((current_name, sym), [target_name]))
            dfa_states = union(dfa_states, [target_name])
            worklist.append(target_set)
    return make_fsm(dfa_states, alphabet, dfa_trans, start_name, dfa_accepting)""",
    "fsm-complement": """def fsm_complement(dfa):
    m0 = dfa if fsm_deterministic_q(dfa) else nfa_to_dfa(dfa)
    m = fsm_complete(m0)
    non_accepting = [s for s in m.states if s not in m.accepting]
    return make_fsm(m.states, m.alphabet, m.transitions, m.start, non_accepting)""",
}

CHEZ_SNIPPETS = {
    "fsm-delta": """(define (delta m s ch)
  (let ([k (cons s ch)])
    (let ([p (assoc k (fsm-transitions m))])
      (if p (cdr p) '()))))""",
    "epsilon-closure": """(define (eps-close m s)
  (let loop ([frontier (list s)] [seen '()])
    (if (null? frontier)
        seen
        (let ([x (car frontier)])
          (if (member x seen)
              (loop (cdr frontier) seen)
              (let* ([eps (get-all-epsilon-targets m x)]
                     [next (append eps (cdr frontier))])
                (loop next (cons x seen))))))))""",
    "fsm-move": """(define (move m states ch)
  (let* ([targets
          (fold-left (lambda (acc s)
                       (union equal? acc (fsm-delta m s ch)))
                     '()
                     states)])
    (epsilon-closure-set m targets)))""",
    "fsm-accepts?": """(define (accepts? m inp)
  (if (null? (fsm-assertions m))
      (just? (fsm-run m inp))
      (just? (fsm-run-with-assertions m inp))))""",
    "fsm-char": """(define (char-fsm c)
  (let ([s0 (fsm-fresh-state "q")]
        [s1 (fsm-fresh-state "q")])
    (make-fsm (list s0 s1)
              (list c)
              (list (cons (cons s0 c) (list s1)))
              s0
              (list s1))))""",
    "fsm-literal": """(define (literal-fsm s)
  (let ([chs (string->list s)])
    (if (null? chs)
        (fsm-epsilon-lang)
        (fold-left fsm-concat
                   (fsm-char (car chs))
                   (map fsm-char (cdr chs))))))""",
    "nfa->dfa": """(define (subset-construct nfa)
  (if (not (null? (fsm-assertions nfa)))
      nfa
      (let* ([alphabet (fsm-alphabet nfa)]
             [start-set (epsilon-closure nfa (fsm-start nfa))]
             [start-name (state-set->name start-set)])
        (let loop ([worklist (list start-set)]
                   [visited '()]
                   [dfa-states (list start-name)]
                   [dfa-trans '()]
                   [dfa-accepting '()])
          (if (null? worklist)
              (make-fsm dfa-states alphabet dfa-trans start-name dfa-accepting)
              (let ([current (car worklist)])
                (if (member current visited)
                    (loop (cdr worklist) visited dfa-states dfa-trans dfa-accepting)
                    (let* ([current-name (state-set->name current)]
                           [accepting? (exists (lambda (s) (member s (fsm-accepting nfa))) current)]
                           [triples
                            (map (lambda (sym)
                                   (let* ([target-set (fsm-move nfa current sym)]
                                          [target-name (if (null? target-set)
                                                           'dead
                                                           (state-set->name target-set))])
                                     (list sym target-set target-name)))
                                 alphabet)]
                           [valid (filter (lambda (t) (not (null? (cadr t)))) triples)]
                           [new-trans
                            (map (lambda (t)
                                   (cons (cons current-name (car t))
                                         (list (caddr t))))
                                 valid)]
                           [new-sets (map cadr valid)]
                           [new-names (map caddr valid)])
                      (loop (append new-sets (cdr worklist))
                            (cons current visited)
                            (union equal? new-names dfa-states)
                            (append new-trans dfa-trans)
                            (if accepting?
                                (cons current-name dfa-accepting)
                                dfa-accepting))))))))))""",
    "fsm-complement": """(define (complement-fsm dfa)
  (let* ([m0 (if (fsm-deterministic? dfa) dfa (nfa->dfa dfa))]
         [m (fsm-complete m0)]
         [na (filter (lambda (s) (not (member s (fsm-accepting m))))
                     (fsm-states m))])
    (make-fsm (fsm-states m) (fsm-alphabet m) (fsm-transitions m)
              (fsm-start m) na)))""",
}

BUGGY_CASES = [
    {
        "fn": "fsm-delta",
        "buggy": """(define (fsm-delta fsm state input)
  (let ([key (cons state input)])
       (let ([found (assoc key (fsm-transitions fsm))])
            (if found (car found) '()))))""",
        "note": "On hit, fsm-delta must return the target-state list (cdr), not the key pair.",
    },
    {
        "fn": "fsm-delta",
        "buggy": """(define (fsm-delta fsm state input)
  (let ([key (cons state input)])
       (let ([found (assoc key (fsm-transitions fsm))])
            (if found (cdr found) (list state)))))""",
        "note": "Missing transition should return empty list, not a singleton with current state.",
    },
    {
        "fn": "epsilon-closure",
        "buggy": """(define (epsilon-closure fsm state)
  (list state))""",
        "note": "epsilon-closure must include recursively epsilon-reachable states, not only start state.",
    },
    {
        "fn": "epsilon-closure",
        "buggy": """(define (epsilon-closure fsm state)
  (cons state (get-all-epsilon-targets fsm state)))""",
        "note": "epsilon-closure must be transitive across epsilon edges, not just one hop from the start state.",
    },
    {
        "fn": "fsm-move",
        "buggy": """(define (fsm-move fsm states input)
  (fold-left (lambda (acc s)
               (union equal? acc (fsm-delta fsm s input)))
             '()
             states))""",
        "note": "fsm-move must epsilon-close the direct targets before returning.",
    },
    {
        "fn": "fsm-move",
        "buggy": """(define (fsm-move fsm states input)
  (let* ([direct-targets
          (fold-left (lambda (acc s)
                             (union equal? acc (fsm-delta fsm s input)))
                     '()
                     states)])
        direct-targets))""",
        "note": "Returning direct targets only misses epsilon transitions reachable after consuming input.",
    },
    {
        "fn": "fsm-accepts?",
        "buggy": """(define (fsm-accepts? fsm input)
  (just? (fsm-run fsm input)))""",
        "note": "Assertion-bearing FSMs must route through fsm-run-with-assertions.",
    },
    {
        "fn": "fsm-accepts?",
        "buggy": """(define (fsm-accepts? fsm input)
  (if (null? (fsm-assertions fsm))
      (just? (fsm-run-with-assertions fsm input))
      (just? (fsm-run fsm input))))""",
        "note": "The assertion/no-assertion branch is inverted.",
    },
    {
        "fn": "fsm-char",
        "buggy": """(define (fsm-char c)
  (let ([s0 (fsm-fresh-state "q")]
        [s1 (fsm-fresh-state "q")])
       (make-fsm (list s0 s1)
                 (list c)
                 (list (cons (cons s0 c) (list s1)))
                 s0
                 (list s0))))""",
        "note": "Accepting state must be the target state s1, not the start state.",
    },
    {
        "fn": "fsm-char",
        "buggy": """(define (fsm-char c)
  (let ([s0 (fsm-fresh-state "q")]
        [s1 (fsm-fresh-state "q")])
       (make-fsm (list s0 s1)
                 (list c)
                 (list (cons (cons s0 c) (list s1)))
                 s1
                 (list s1))))""",
        "note": "Start state must be s0 so one input symbol is required.",
    },
    {
        "fn": "fsm-literal",
        "buggy": """(define (fsm-literal str)
  (let ([chars (string->list str)])
       (if (null? chars)
           (fsm-epsilon-lang)
           (fold-left fsm-union
                      (fsm-char (car chars))
                      (map fsm-char (cdr chars))))))""",
        "note": "Literal construction requires concatenation, not union.",
    },
    {
        "fn": "fsm-literal",
        "buggy": """(define (fsm-literal str)
  (let ([chars (string->list str)])
       (if (null? chars)
           (fsm-char #\\space)
           (fold-left fsm-concat
                      (fsm-char (car chars))
                      (map fsm-char (cdr chars))))))""",
        "note": "Empty literal must accept only empty string via fsm-epsilon-lang.",
    },
    {
        "fn": "nfa->dfa",
        "buggy": """(define (nfa->dfa nfa)
  (if (not (null? (fsm-assertions nfa)))
      nfa
      (let* ([alphabet (fsm-alphabet nfa)]
             [start-set (list (fsm-start nfa))]
             [start-name (state-set->name start-set)])
        (make-fsm (list start-name) alphabet '() start-name '()))))""",
        "note": "Subset construction must start from epsilon-closure and explore reachable subsets/transitions.",
    },
    {
        "fn": "nfa->dfa",
        "buggy": """(define (nfa->dfa nfa)
  (if (not (null? (fsm-assertions nfa)))
      nfa
      (let* ([alphabet (fsm-alphabet nfa)]
             [start-set (epsilon-closure nfa (fsm-start nfa))]
             [start-name (state-set->name start-set)])
        (make-fsm (list start-name) alphabet '() start-name (list start-name)))))""",
        "note": "Initial DFA state is not always accepting; accepting subsets must be computed from NFA accepting membership.",
    },
    {
        "fn": "fsm-complement",
        "buggy": """(define (fsm-complement dfa)
  (let* ([m (if (fsm-deterministic? dfa) dfa (nfa->dfa dfa))]
         [non-accepting (filter (lambda (s) (not (member s (fsm-accepting m))))
                                (fsm-states m))])
        (make-fsm (fsm-states m) (fsm-alphabet m) (fsm-transitions m)
                  (fsm-start m) non-accepting)))""",
        "note": "Complement requires completion before flipping acceptance, otherwise missing transitions break semantics.",
    },
    {
        "fn": "fsm-complement",
        "buggy": """(define (fsm-complement dfa)
  (let* ([m0 (if (fsm-deterministic? dfa) dfa (nfa->dfa dfa))]
         [m (fsm-complete m0)]
         [accepting (fsm-accepting m)])
        (make-fsm (fsm-states m) (fsm-alphabet m) (fsm-transitions m)
                  (fsm-start m) accepting)))""",
        "note": "Complement must flip to non-accepting states, not keep original accepting set.",
    },
]

DIFFICULTY = {
    "fsm-delta": "easy",
    "epsilon-closure": "medium",
    "fsm-move": "medium",
    "fsm-accepts?": "medium",
    "fsm-char": "easy",
    "fsm-literal": "medium",
    "nfa->dfa": "hard",
    "fsm-complement": "hard",
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
    sid = f"fp_parsing_fsm_{family}_{family_counter[family]:03d}"
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
        prompt=f"""Implement this FSM utility in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "fp", "parsing", "fsm", "spec-to-code", fn],
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
        tags=["tier1", "fp", "parsing", "fsm", "skeleton-completion", fn],
    )

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Implement `{fn}` from this FSM contract.

Module: `{SOURCE_MODULE}`
Contract focus: {FUNCTION_SPECS[fn]}

Requirements:
1. Preserve transition/acceptance semantics precisely.
2. Keep exact function name/signature for `{fn}`.
3. Return only one complete definition.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "fp", "parsing", "fsm", "contract-implementation", fn],
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
        tags=["tier1", "fp", "parsing", "fsm", "python-to-scheme", fn],
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
        tags=["tier1", "fp", "parsing", "fsm", "chez-to-fold", fn],
    )

    add_sample(
        family="translation",
        category="translation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Translate this reference implementation into canonical Fold Scheme for `{fn}`.

Preserve observable FSM behavior exactly.
Keep the target function name/signature as `{fn}`.
Return only the final Scheme definition.

```python
{PYTHON_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "fp", "parsing", "fsm", "reference-translation", fn],
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
        tags=["tier1", "fp", "parsing", "fsm", "bugfix", fn],
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
        f"Ensure `{source_function}` is part of the composed solution.\n"
        "Return only one executable Scheme expression."
    )
    add_sample(
        family="composition",
        category="usage",
        difficulty=difficulty,
        source_function=source_function,
        prompt=composition_prompt,
        ground_truth=ground_truth,
        verify_expr=verify_check.strip(),
        tags=["tier1", "fp", "parsing", "fsm", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # fsm-delta
    {
        "fn": "fsm-delta",
        "prompt": "Build a 2-state DFA and return delta(q0,'a').",
        "gt": "(let ([m (dfa '(q0 q1) '(#\\a #\\b) '((q0 #\\a q1) (q1 #\\b q0)) 'q0 '(q1))]) (fsm-delta m 'q0 #\\a))",
        "verify": "(equal? (let ([m (dfa '(q0 q1) '(#\\a #\\b) '((q0 #\\a q1) (q1 #\\b q0)) 'q0 '(q1))]) (fsm-delta m 'q0 #\\a)) '(q1))",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "fsm-delta",
        "prompt": "Build a 2-state DFA and return delta(q1,'a') for a missing transition.",
        "gt": "(let ([m (dfa '(q0 q1) '(#\\a #\\b) '((q0 #\\a q1) (q1 #\\b q0)) 'q0 '(q1))]) (fsm-delta m 'q1 #\\a))",
        "verify": "(equal? (let ([m (dfa '(q0 q1) '(#\\a #\\b) '((q0 #\\a q1) (q1 #\\b q0)) 'q0 '(q1))]) (fsm-delta m 'q1 #\\a)) '())",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },
    {
        "fn": "fsm-delta",
        "prompt": "Return whether delta result is list-valued for present transition.",
        "gt": "(let ([m (dfa '(q0 q1) '(#\\a) '((q0 #\\a q1)) 'q0 '(q1))]) (list? (fsm-delta m 'q0 #\\a)))",
        "verify": "(equal? (let ([m (dfa '(q0 q1) '(#\\a) '((q0 #\\a q1)) 'q0 '(q1))]) (list? (fsm-delta m 'q0 #\\a))) #t)",
        "difficulty": "easy",
        "tags": ["shape"],
    },
    {
        "fn": "fsm-delta",
        "prompt": "Return whether two missing transitions both yield empty lists.",
        "gt": "(let ([m (dfa '(q0 q1) '(#\\a) '((q0 #\\a q1)) 'q0 '(q1))]) (and (equal? (fsm-delta m 'q1 #\\a) '()) (equal? (fsm-delta m 'q1 #\\z) '())))",
        "verify": "(equal? (let ([m (dfa '(q0 q1) '(#\\a) '((q0 #\\a q1)) 'q0 '(q1))]) (and (equal? (fsm-delta m 'q1 #\\a) '()) (equal? (fsm-delta m 'q1 #\\z) '()))) #t)",
        "difficulty": "medium",
        "tags": ["missing"],
    },

    # epsilon-closure
    {
        "fn": "epsilon-closure",
        "prompt": "Compute epsilon-closure from q0 in q0->q1->q2 epsilon chain.",
        "gt": "(let* ([m (make-fsm '(q0 q1 q2) '(#\\a) '() 'q0 '(q2) '((q0 q1) (q1 q2)))] [c (epsilon-closure m 'q0)]) (list (member 'q0 c) (member 'q1 c) (member 'q2 c)))",
        "verify": "(let* ([m (make-fsm '(q0 q1 q2) '(#\\a) '() 'q0 '(q2) '((q0 q1) (q1 q2)))] [c (epsilon-closure m 'q0)]) (and (member 'q0 c) (member 'q1 c) (member 'q2 c) #t))",
        "difficulty": "medium",
        "tags": ["direct"],
    },
    {
        "fn": "epsilon-closure",
        "prompt": "Compute epsilon-closure of state with no epsilon edges.",
        "gt": "(let* ([m (make-fsm '(q0 q1) '(#\\a) '() 'q0 '(q1) '())] [c (epsilon-closure m 'q1)]) c)",
        "verify": "(equal? (let* ([m (make-fsm '(q0 q1) '(#\\a) '() 'q0 '(q1) '())] [c (epsilon-closure m 'q1)]) c) '(q1))",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },
    {
        "fn": "epsilon-closure",
        "prompt": "Return whether epsilon-closure handles epsilon cycles without duplicating states.",
        "gt": "(let* ([m (make-fsm '(q0 q1) '(#\\a) '() 'q0 '(q1) '((q0 q1) (q1 q0)))] [c (epsilon-closure m 'q0)]) (= (length c) 2))",
        "verify": "(equal? (let* ([m (make-fsm '(q0 q1) '(#\\a) '() 'q0 '(q1) '((q0 q1) (q1 q0)))] [c (epsilon-closure m 'q0)]) (= (length c) 2)) #t)",
        "difficulty": "medium",
        "tags": ["cycle"],
    },
    {
        "fn": "epsilon-closure",
        "prompt": "Return whether q0 epsilon-closure includes q2 in transitive chain q0->q1->q2.",
        "gt": "(let* ([m (make-fsm '(q0 q1 q2) '(#\\a) '() 'q0 '(q2) '((q0 q1) (q1 q2)))] [c (epsilon-closure m 'q0)]) (member 'q2 c))",
        "verify": "(equal? (let* ([m (make-fsm '(q0 q1 q2) '(#\\a) '() 'q0 '(q2) '((q0 q1) (q1 q2)))] [c (epsilon-closure m 'q0)]) (not (not (member 'q2 c)))) #t)",
        "difficulty": "medium",
        "tags": ["transitive"],
    },

    # fsm-move
    {
        "fn": "fsm-move",
        "prompt": "Move from {q0} on 'a' then epsilon-close q1->q2.",
        "gt": "(let* ([m (make-fsm '(q0 q1 q2) '(#\\a) (list (cons (cons 'q0 #\\a) '(q1))) 'q0 '(q2) '((q1 q2)))] [res (fsm-move m '(q0) #\\a)]) res)",
        "verify": "(let* ([m (make-fsm '(q0 q1 q2) '(#\\a) (list (cons (cons 'q0 #\\a) '(q1))) 'q0 '(q2) '((q1 q2)))] [res (fsm-move m '(q0) #\\a)]) (and (= (length res) 2) (member 'q1 res) (member 'q2 res) #t))",
        "difficulty": "medium",
        "tags": ["direct"],
    },
    {
        "fn": "fsm-move",
        "prompt": "Return move result for symbol with no outgoing edges.",
        "gt": "(let* ([m (make-fsm '(q0 q1) '(#\\a) (list (cons (cons 'q0 #\\a) '(q1))) 'q0 '(q1) '())] [res (fsm-move m '(q1) #\\a)]) res)",
        "verify": "(equal? (let* ([m (make-fsm '(q0 q1) '(#\\a) (list (cons (cons 'q0 #\\a) '(q1))) 'q0 '(q1) '())] [res (fsm-move m '(q1) #\\a)]) res) '())",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },
    {
        "fn": "fsm-move",
        "prompt": "Move from multiple states and return whether union behavior includes both branches.",
        "gt": "(let* ([m (make-fsm '(q0 q1 q2) '(#\\a) (list (cons (cons 'q0 #\\a) '(q2)) (cons (cons 'q1 #\\a) '(q2))) 'q0 '(q2) '())] [res (fsm-move m '(q0 q1) #\\a)]) (and (= (length res) 1) (equal? res '(q2))))",
        "verify": "(equal? (let* ([m (make-fsm '(q0 q1 q2) '(#\\a) (list (cons (cons 'q0 #\\a) '(q2)) (cons (cons 'q1 #\\a) '(q2))) 'q0 '(q2) '())] [res (fsm-move m '(q0 q1) #\\a)]) (and (= (length res) 1) (equal? res '(q2)))) #t)",
        "difficulty": "medium",
        "tags": ["set-union"],
    },
    {
        "fn": "fsm-move",
        "prompt": "Return whether move from empty state-set is empty.",
        "gt": "(let* ([m (fsm-char #\\a)] [res (fsm-move m '() #\\a)]) (equal? res '()))",
        "verify": "(equal? (let* ([m (fsm-char #\\a)] [res (fsm-move m '() #\\a)]) (equal? res '())) #t)",
        "difficulty": "easy",
        "tags": ["empty-input"],
    },

    # fsm-accepts?
    {
        "fn": "fsm-accepts?",
        "prompt": "Check acceptance of literal \"ab\" machine on input \"ab\".",
        "gt": "(let ([m (fsm-concat (fsm-char #\\a) (fsm-char #\\b))]) (fsm-accepts? m \"ab\"))",
        "verify": "(equal? (let ([m (fsm-concat (fsm-char #\\a) (fsm-char #\\b))]) (fsm-accepts? m \"ab\")) #t)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "fsm-accepts?",
        "prompt": "Check rejection of literal \"ab\" machine on input \"a\".",
        "gt": "(let ([m (fsm-concat (fsm-char #\\a) (fsm-char #\\b))]) (fsm-accepts? m \"a\"))",
        "verify": "(equal? (let ([m (fsm-concat (fsm-char #\\a) (fsm-char #\\b))]) (fsm-accepts? m \"a\")) #f)",
        "difficulty": "easy",
        "tags": ["rejection"],
    },
    {
        "fn": "fsm-accepts?",
        "prompt": "Return whether star machine accepts empty string and repeated symbol.",
        "gt": "(let ([m (fsm-star (fsm-char #\\a))]) (and (fsm-accepts? m \"\") (fsm-accepts? m \"aaa\")))",
        "verify": "(equal? (let ([m (fsm-star (fsm-char #\\a))]) (and (fsm-accepts? m \"\") (fsm-accepts? m \"aaa\"))) #t)",
        "difficulty": "medium",
        "tags": ["star"],
    },
    {
        "fn": "fsm-accepts?",
        "prompt": "Return whether optional machine accepts empty and single symbol but rejects double symbol.",
        "gt": "(let ([m (fsm-optional (fsm-char #\\a))]) (and (fsm-accepts? m \"\") (fsm-accepts? m \"a\") (not (fsm-accepts? m \"aa\"))))",
        "verify": "(equal? (let ([m (fsm-optional (fsm-char #\\a))]) (and (fsm-accepts? m \"\") (fsm-accepts? m \"a\") (not (fsm-accepts? m \"aa\")))) #t)",
        "difficulty": "medium",
        "tags": ["optional"],
    },

    # fsm-char
    {
        "fn": "fsm-char",
        "prompt": "Build fsm-char for 'x and check acceptance of \"x\".",
        "gt": "(let ([m (fsm-char #\\x)]) (fsm-accepts? m \"x\"))",
        "verify": "(equal? (let ([m (fsm-char #\\x)]) (fsm-accepts? m \"x\")) #t)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "fsm-char",
        "prompt": "Build fsm-char for 'x and check rejection of empty string.",
        "gt": "(let ([m (fsm-char #\\x)]) (fsm-accepts? m \"\"))",
        "verify": "(equal? (let ([m (fsm-char #\\x)]) (fsm-accepts? m \"\")) #f)",
        "difficulty": "easy",
        "tags": ["empty"],
    },
    {
        "fn": "fsm-char",
        "prompt": "Build fsm-char for 'x and check rejection of \"xx\".",
        "gt": "(let ([m (fsm-char #\\x)]) (fsm-accepts? m \"xx\"))",
        "verify": "(equal? (let ([m (fsm-char #\\x)]) (fsm-accepts? m \"xx\")) #f)",
        "difficulty": "easy",
        "tags": ["exact-length"],
    },
    {
        "fn": "fsm-char",
        "prompt": "Return whether fsm-char('a) and fsm-literal(\"a\") agree on \"a\" and \"aa\".",
        "gt": "(let ([m1 (fsm-char #\\a)] [m2 (fsm-literal \"a\")]) (and (equal? (fsm-accepts? m1 \"a\") (fsm-accepts? m2 \"a\")) (equal? (fsm-accepts? m1 \"aa\") (fsm-accepts? m2 \"aa\"))))",
        "verify": "(equal? (let ([m1 (fsm-char #\\a)] [m2 (fsm-literal \"a\")]) (and (equal? (fsm-accepts? m1 \"a\") (fsm-accepts? m2 \"a\")) (equal? (fsm-accepts? m1 \"aa\") (fsm-accepts? m2 \"aa\")))) #t)",
        "difficulty": "medium",
        "tags": ["consistency"],
    },

    # fsm-literal
    {
        "fn": "fsm-literal",
        "prompt": "Build literal machine for \"hello\" and test acceptance of exact literal.",
        "gt": "(let ([m (fsm-literal \"hello\")]) (fsm-accepts? m \"hello\"))",
        "verify": "(equal? (let ([m (fsm-literal \"hello\")]) (fsm-accepts? m \"hello\")) #t)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "fsm-literal",
        "prompt": "Build literal machine for \"hello\" and test rejection of prefix \"hell\".",
        "gt": "(let ([m (fsm-literal \"hello\")]) (fsm-accepts? m \"hell\"))",
        "verify": "(equal? (let ([m (fsm-literal \"hello\")]) (fsm-accepts? m \"hell\")) #f)",
        "difficulty": "easy",
        "tags": ["prefix"],
    },
    {
        "fn": "fsm-literal",
        "prompt": "Build literal machine for empty string and test empty acceptance plus non-empty rejection.",
        "gt": "(let ([m (fsm-literal \"\")]) (and (fsm-accepts? m \"\") (not (fsm-accepts? m \"a\"))))",
        "verify": "(equal? (let ([m (fsm-literal \"\")]) (and (fsm-accepts? m \"\") (not (fsm-accepts? m \"a\")))) #t)",
        "difficulty": "medium",
        "tags": ["empty-literal"],
    },
    {
        "fn": "fsm-literal",
        "prompt": "Return whether concatenating fsm-char('a) and fsm-char('b) matches fsm-literal(\"ab\").",
        "gt": "(let ([m1 (fsm-concat (fsm-char #\\a) (fsm-char #\\b))] [m2 (fsm-literal \"ab\")]) (and (equal? (fsm-accepts? m1 \"ab\") (fsm-accepts? m2 \"ab\")) (equal? (fsm-accepts? m1 \"a\") (fsm-accepts? m2 \"a\"))))",
        "verify": "(equal? (let ([m1 (fsm-concat (fsm-char #\\a) (fsm-char #\\b))] [m2 (fsm-literal \"ab\")]) (and (equal? (fsm-accepts? m1 \"ab\") (fsm-accepts? m2 \"ab\")) (equal? (fsm-accepts? m1 \"a\") (fsm-accepts? m2 \"a\")))) #t)",
        "difficulty": "medium",
        "tags": ["consistency"],
    },

    # nfa->dfa
    {
        "fn": "nfa->dfa",
        "prompt": "Determinize machine for a+b and return whether resulting machine is deterministic.",
        "gt": "(let* ([nfa-m (fsm-concat (fsm-plus (fsm-char #\\a)) (fsm-char #\\b))] [dfa-m (nfa->dfa nfa-m)]) (fsm-deterministic? dfa-m))",
        "verify": "(equal? (let* ([nfa-m (fsm-concat (fsm-plus (fsm-char #\\a)) (fsm-char #\\b))] [dfa-m (nfa->dfa nfa-m)]) (fsm-deterministic? dfa-m)) #t)",
        "difficulty": "hard",
        "tags": ["determinization"],
    },
    {
        "fn": "nfa->dfa",
        "prompt": "Determinize a+b machine and test acceptance of \"aaab\".",
        "gt": "(let* ([nfa-m (fsm-concat (fsm-plus (fsm-char #\\a)) (fsm-char #\\b))] [dfa-m (nfa->dfa nfa-m)]) (fsm-accepts? dfa-m \"aaab\"))",
        "verify": "(equal? (let* ([nfa-m (fsm-concat (fsm-plus (fsm-char #\\a)) (fsm-char #\\b))] [dfa-m (nfa->dfa nfa-m)]) (fsm-accepts? dfa-m \"aaab\")) #t)",
        "difficulty": "hard",
        "tags": ["acceptance"],
    },
    {
        "fn": "nfa->dfa",
        "prompt": "Determinize a+b machine and verify rejection of \"b\".",
        "gt": "(let* ([nfa-m (fsm-concat (fsm-plus (fsm-char #\\a)) (fsm-char #\\b))] [dfa-m (nfa->dfa nfa-m)]) (fsm-accepts? dfa-m \"b\"))",
        "verify": "(equal? (let* ([nfa-m (fsm-concat (fsm-plus (fsm-char #\\a)) (fsm-char #\\b))] [dfa-m (nfa->dfa nfa-m)]) (fsm-accepts? dfa-m \"b\")) #f)",
        "difficulty": "medium",
        "tags": ["rejection"],
    },
    {
        "fn": "nfa->dfa",
        "prompt": "Return whether NFA and determinized DFA agree on three probe strings.",
        "gt": "(let* ([nfa-m (fsm-concat (fsm-plus (fsm-char #\\a)) (fsm-char #\\b))] [dfa-m (nfa->dfa nfa-m)]) (and (equal? (fsm-accepts? nfa-m \"ab\") (fsm-accepts? dfa-m \"ab\")) (equal? (fsm-accepts? nfa-m \"aab\") (fsm-accepts? dfa-m \"aab\")) (equal? (fsm-accepts? nfa-m \"a\") (fsm-accepts? dfa-m \"a\"))))",
        "verify": "(equal? (let* ([nfa-m (fsm-concat (fsm-plus (fsm-char #\\a)) (fsm-char #\\b))] [dfa-m (nfa->dfa nfa-m)]) (and (equal? (fsm-accepts? nfa-m \"ab\") (fsm-accepts? dfa-m \"ab\")) (equal? (fsm-accepts? nfa-m \"aab\") (fsm-accepts? dfa-m \"aab\")) (equal? (fsm-accepts? nfa-m \"a\") (fsm-accepts? dfa-m \"a\")))) #t)",
        "difficulty": "hard",
        "tags": ["equivalence"],
    },

    # fsm-complement
    {
        "fn": "fsm-complement",
        "prompt": "Complement literal-\"a\" machine and check rejection of \"a\".",
        "gt": "(let* ([m (fsm-literal \"a\")] [comp (fsm-complement m)]) (fsm-accepts? comp \"a\"))",
        "verify": "(equal? (let* ([m (fsm-literal \"a\")] [comp (fsm-complement m)]) (fsm-accepts? comp \"a\")) #f)",
        "difficulty": "medium",
        "tags": ["direct"],
    },
    {
        "fn": "fsm-complement",
        "prompt": "Complement literal-\"a\" machine and check acceptance of empty string.",
        "gt": "(let* ([m (fsm-literal \"a\")] [comp (fsm-complement m)]) (fsm-accepts? comp \"\"))",
        "verify": "(equal? (let* ([m (fsm-literal \"a\")] [comp (fsm-complement m)]) (fsm-accepts? comp \"\")) #t)",
        "difficulty": "medium",
        "tags": ["empty"],
    },
    {
        "fn": "fsm-complement",
        "prompt": "Complement literal-\"a\" machine and check acceptance of \"aa\".",
        "gt": "(let* ([m (fsm-literal \"a\")] [comp (fsm-complement m)]) (fsm-accepts? comp \"aa\"))",
        "verify": "(equal? (let* ([m (fsm-literal \"a\")] [comp (fsm-complement m)]) (fsm-accepts? comp \"aa\")) #t)",
        "difficulty": "medium",
        "tags": ["other-string"],
    },
    {
        "fn": "fsm-complement",
        "prompt": "Return whether complement operation is involutive on probe strings for literal-\"a\" machine.",
        "gt": "(let* ([m (fsm-literal \"a\")] [c1 (fsm-complement m)] [c2 (fsm-complement c1)]) (and (equal? (fsm-accepts? m \"\") (fsm-accepts? c2 \"\")) (equal? (fsm-accepts? m \"a\") (fsm-accepts? c2 \"a\")) (equal? (fsm-accepts? m \"aa\") (fsm-accepts? c2 \"aa\"))))",
        "verify": "(equal? (let* ([m (fsm-literal \"a\")] [c1 (fsm-complement m)] [c2 (fsm-complement c1)]) (and (equal? (fsm-accepts? m \"\") (fsm-accepts? c2 \"\")) (equal? (fsm-accepts? m \"a\") (fsm-accepts? c2 \"a\")) (equal? (fsm-accepts? m \"aa\") (fsm-accepts? c2 \"aa\")))) #t)",
        "difficulty": "hard",
        "tags": ["involution"],
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
