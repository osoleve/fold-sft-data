#!/usr/bin/env python3
"""Generate Tier-1 egraph match SFT samples for lattice/egraph/match.ss."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

OUT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DATA_ROOT.parent
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from sft_split_utils import compute_leakage_aware_eval_ids

ALL_PATH = OUT_DIR / "all.jsonl"
TRAIN_PATH = OUT_DIR / "train.jsonl"
EVAL_PATH = OUT_DIR / "eval.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.json"
PRE_DIVERSIFY_PATH = OUT_DIR / ".pre_diversify.jsonl"
SFT_GENERATOR_PATH = REPO_ROOT / "user" / "sft" / "generate.ss"

SOURCE_MODULE = "lattice/egraph/match.ss"
SOURCE_TEST = "lattice/egraph/test-match.ss"

DEFS: Dict[str, str] = {
    "pattern-var?": """(define (pattern-var? x)
  (doc 'type (-> Any Boolean))
  (doc 'description "Check if x is a pattern variable (symbol starting with ?).")
  (doc 'export #t)
  (and (symbol? x)
       (let ([s (symbol->string x)])
         (and (> (string-length s) 1)
              (char=? (string-ref s 0) #\\?)))))""",
    "subst-try-extend": """(define (subst-try-extend subst var class-id)
  (doc 'type (-> Substitution PatternVar EClassId (Or Substitution #f)))
  (doc 'description "Try to extend substitution. Returns #f if incompatible.")
  (doc 'export #t)
  (let ([existing (subst-lookup subst var)])
    (cond
      [(not existing) (subst-extend subst var class-id)]
      [(= existing class-id) subst]
      [else #f])))""",
    "subst-merge": """(define (subst-merge subst1 subst2)
  (doc 'type (-> Substitution Substitution (Or Substitution #f)))
  (doc 'description "Merge two substitutions. Returns #f if incompatible.")
  (doc 'export #t)
  (let loop ([s1 subst1] [result subst2])
    (if (null? s1)
        result
        (let* ([binding (car s1)]
               [var (car binding)]
               [class-id (cdr binding)]
               [new-result (subst-try-extend result var class-id)])
          (if new-result
              (loop (cdr s1) new-result)
              #f)))))""",
    "ematch-pattern": """(define (ematch-pattern eg pattern class-id subst)
  (doc 'type (-> EGraph Pattern EClassId Substitution (List Substitution)))
  (doc 'description "Match pattern against e-class, return all valid substitutions.")
  (doc 'export #t)
  (let ([root (egraph-find eg class-id)])
    (cond
      [(pattern-var? pattern)
       (let ([new-subst (subst-try-extend subst pattern root)])
         (if new-subst (list new-subst) '()))]
      [(or (symbol? pattern) (number? pattern))
       (let ([nodes (egraph-class-nodes eg root)])
         (if (exists (lambda (node)
                       (and (eqv? (enode-op node) pattern)
                            (zero? (enode-arity node))))
                     nodes)
             (list subst)
             '()))]
      [(pair? pattern)
       (let ([op (car pattern)]
             [arg-patterns (cdr pattern)]
             [nodes (egraph-class-nodes eg root)])
         (let ([matching-nodes
                (filter (lambda (node)
                          (and (eqv? (enode-op node) op)
                               (= (enode-arity node) (length arg-patterns))))
                        nodes)])
           (append-map (lambda (node)
                         (ematch-children eg arg-patterns
                                         (vector->list (enode-children node))
                                         subst))
                       matching-nodes)))]
      [else '()])))""",
    "ematch": """(define (ematch eg pattern class-id)
  (doc 'type (-> EGraph Pattern EClassId (List Substitution)))
  (doc 'description "Match pattern against e-class, return all substitutions.")
  (doc 'export #t)
  (ematch-pattern eg pattern class-id (empty-subst)))""",
    "pattern-apply": """(define (pattern-apply subst pattern)
  (doc 'type (-> Substitution Pattern Term))
  (doc 'description "Apply substitution to pattern, yielding a term with e-class refs.")
  (doc 'export #t)
  (cond
    [(pattern-var? pattern)
     (let ([binding (subst-lookup subst pattern)])
       (if binding
           (make-eclass-ref binding)
           (error 'pattern-apply "Unbound pattern variable" pattern)))]
    [(pair? pattern)
     (cons (car pattern)
           (map (lambda (p) (pattern-apply subst p))
                (cdr pattern)))]
    [else pattern]))""",
    "apply-rule": """(define (apply-rule eg rule class-id)
  (doc 'type (-> EGraph Rule EClassId Boolean))
  (doc 'description "Apply rule to e-class. Returns #t if matched.")
  (doc 'export #t)
  (let* ([lhs (rule-lhs rule)]
         [rhs (rule-rhs rule)]
         [matches (ematch eg lhs class-id)])
    (if (null? matches)
        #f
        (begin
          (for-each
           (lambda (subst)
             (let ([rhs-term (pattern-apply subst rhs)])
               (let ([rhs-id (add-instantiated-term eg rhs-term)])
                 (egraph-merge! eg class-id rhs-id))))
           matches)
          #t))))""",
    "apply-rules": """(define (apply-rules eg rules)
  (doc 'type (-> EGraph (List Rule) Nat))
  (doc 'description "Apply all rules to all e-classes once. Returns match count.")
  (doc 'export #t)
  (let ([uf (egraph-uf eg)]
        [count 0])
    (for-each
     (lambda (root)
       (for-each
        (lambda (rule)
          (when (apply-rule eg rule root)
            (set! count (+ count 1))))
        rules))
     (uf-roots uf))
    count))""",
}


def strip_doc_forms(defn: str) -> str:
    lines = [line for line in defn.splitlines() if not line.strip().startswith("(doc ")]
    return "\n".join(lines)


DOC_FREE_DEFS: Dict[str, str] = {fn: strip_doc_forms(code) for fn, code in DEFS.items()}

FUNCTION_ORDER = [
    "pattern-var?",
    "subst-try-extend",
    "subst-merge",
    "ematch-pattern",
    "ematch",
    "pattern-apply",
    "apply-rule",
    "apply-rules",
]

FUNCTION_SPECS = {
    "pattern-var?": "Return #t iff input is a symbol whose textual form starts with ? and has length > 1.",
    "subst-try-extend": "Add a binding when absent, preserve same binding, and reject conflicting binding with #f.",
    "subst-merge": "Merge two substitution alists and fail with #f on any variable/class conflict.",
    "ematch-pattern": "Match a pattern against an e-class recursively, returning every compatible substitution.",
    "ematch": "Entry-point wrapper that runs ematch-pattern using an empty substitution.",
    "pattern-apply": "Instantiate a pattern using substitution bindings, wrapping variable replacements as eclass refs.",
    "apply-rule": "Match rule lhs in a class, instantiate rhs for each match, add term, and merge into matched class.",
    "apply-rules": "Sweep all roots and rules once, counting how many rule applications matched.",
}

SKELETONS = {
    "pattern-var?": """(define (pattern-var? x)
  ;; TODO: detect pattern variables like ?x but reject plain symbols and bare ?
  <TODO>)""",
    "subst-try-extend": """(define (subst-try-extend subst var class-id)
  ;; TODO: insert new bindings, preserve identical bindings, reject conflicts
  <TODO>)""",
    "subst-merge": """(define (subst-merge subst1 subst2)
  ;; TODO: merge all bindings from subst1 into subst2 with compatibility checks
  <TODO>)""",
    "ematch-pattern": """(define (ematch-pattern eg pattern class-id subst)
  ;; TODO: match variable, leaf, and pair patterns over the e-class nodes
  <TODO>)""",
    "ematch": """(define (ematch eg pattern class-id)
  ;; TODO: call ematch-pattern with an empty substitution
  <TODO>)""",
    "pattern-apply": """(define (pattern-apply subst pattern)
  ;; TODO: replace pattern vars with eclass refs and recurse through pairs
  <TODO>)""",
    "apply-rule": """(define (apply-rule eg rule class-id)
  ;; TODO: match lhs, instantiate rhs, add term, and merge results
  <TODO>)""",
    "apply-rules": """(define (apply-rules eg rules)
  ;; TODO: apply every rule to every root and count successful matches
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "pattern-var?": "(and (pattern-var? '?x) (pattern-var? '?foo) (not (pattern-var? 'x)) (not (pattern-var? 42)) (not (pattern-var? '?)))",
    "subst-try-extend": "(let* ([s0 (empty-subst)] [s1 (subst-try-extend s0 '?x 5)] [s2 (and s1 (subst-try-extend s1 '?x 5))] [s3 (and s1 (subst-try-extend s1 '?x 6))]) (and (pair? s1) (pair? s2) (not s3) (= (subst-lookup s1 '?x) 5)))",
    "subst-merge": "(let* ([a (subst-extend (empty-subst) '?x 1)] [b (subst-extend (empty-subst) '?y 2)] [ok (subst-merge a b)] [c (subst-extend (empty-subst) '?x 1)] [d (subst-extend (empty-subst) '?x 3)] [bad (subst-merge c d)]) (and (pair? ok) (= (subst-lookup ok '?x) 1) (= (subst-lookup ok '?y) 2) (not bad)))",
    "ematch-pattern": "(let* ([eg (make-egraph)] [id (egraph-add-term! eg '(+ a b))] [m1 (ematch-pattern eg '(+ ?x ?y) id (empty-subst))] [m2 (ematch-pattern eg '(+ ?z ?z) id (empty-subst))]) (and (= (length m1) 1) (null? m2) (let* ([s (car m1)] [x (subst-lookup s '?x)] [y (subst-lookup s '?y)]) (and (integer? x) (integer? y) (not (= x y))))))",
    "ematch": "(and (let* ([eg (make-egraph)] [id (egraph-add-term! eg '(f x))] [mv (ematch eg '?a id)] [mf (ematch eg '(f ?z) id)] [mg (ematch eg '(g ?z) id)] [sv (and (pair? mv) (car mv))]) (and (= (length mv) 1) (= (length mf) 1) (null? mg) sv (not (subst-lookup sv '?seed)))) (= (let* ([eg (make-egraph)] [fx (egraph-add-term! eg '(f x))] [fy (egraph-add-term! eg '(f y))]) (egraph-merge! eg fx fy) (egraph-saturate-rebuild! eg) (length (ematch eg '(f ?a) fx))) 2))",
    "pattern-apply": "(let* ([s (subst-extend (subst-extend (empty-subst) '?x 1) '?y 2)] [t (pattern-apply s '(+ ?x ?y 42))]) (and (eq? (car t) '+) (eclass-ref? (cadr t)) (= (eclass-ref-id (cadr t)) 1) (eclass-ref? (caddr t)) (= (eclass-ref-id (caddr t)) 2) (= (cadddr t) 42)))",
    "apply-rule": "(and (let* ([eg (make-egraph)] [r (make-rule '(+ ?x 0) '?x)] [term (egraph-add-term! eg '(+ a 0))] [a-id (egraph-add-term! eg 'a)] [ok (apply-rule eg r term)]) (egraph-saturate-rebuild! eg) (and ok (= (egraph-find eg term) (egraph-find eg a-id)))) (let* ([eg (make-egraph)] [r (make-rule '(* ?x 1) '?x)] [term (egraph-add-term! eg '(+ a 0))]) (not (apply-rule eg r term))))",
    "apply-rules": "(and (= (let* ([eg (make-egraph)] [rules (list (make-rule '(+ ?x 0) '?x))] [_1 (egraph-add-term! eg '(+ a 0))] [_2 (egraph-add-term! eg '(+ b 0))] [count (apply-rules eg rules)]) count) 2) (= (let* ([eg (make-egraph)] [rules (list (make-rule '(+ ?x 0) '?x) (make-rule '(* ?x 1) '?x))] [_1 (egraph-add-term! eg '(+ a 0))] [_2 (egraph-add-term! eg '(* b 1))] [count (apply-rules eg rules)]) count) 2))",
}

PYTHON_SNIPPETS = {
    "pattern-var?": """def pattern_var(x):
    if not isinstance(x, Symbol):
        return False
    s = str(x)
    return len(s) > 1 and s[0] == '?'""",
    "subst-try-extend": """def subst_try_extend(subst, var, class_id):
    existing = subst_lookup(subst, var)
    if existing is None:
        return subst_extend(subst, var, class_id)
    if existing == class_id:
        return subst
    return None""",
    "subst-merge": """def subst_merge(subst1, subst2):
    result = subst2
    for var, class_id in subst1:
        result = subst_try_extend(result, var, class_id)
        if result is None:
            return None
    return result""",
    "ematch-pattern": """def ematch_pattern(eg, pattern, class_id, subst):
    root = egraph_find(eg, class_id)
    if pattern_var(pattern):
        s = subst_try_extend(subst, pattern, root)
        return [s] if s is not None else []
    if is_leaf(pattern):
        return [subst] if class_has_leaf(eg, root, pattern) else []
    op, args = pattern[0], pattern[1:]
    out = []
    for node in class_nodes(eg, root):
        if node.op == op and len(node.children) == len(args):
            out.extend(ematch_children(eg, args, node.children, subst))
    return out""",
    "ematch": """def ematch(eg, pattern, class_id):
    return ematch_pattern(eg, pattern, class_id, empty_subst())""",
    "pattern-apply": """def pattern_apply(subst, pattern):
    if pattern_var(pattern):
        cid = subst_lookup(subst, pattern)
        if cid is None:
            raise ValueError('unbound pattern variable')
        return make_eclass_ref(cid)
    if isinstance(pattern, list):
        return [pattern[0]] + [pattern_apply(subst, p) for p in pattern[1:]]
    return pattern""",
    "apply-rule": """def apply_rule(eg, rule, class_id):
    lhs, rhs = rule_lhs(rule), rule_rhs(rule)
    matches = ematch(eg, lhs, class_id)
    if not matches:
        return False
    for subst in matches:
        rhs_term = pattern_apply(subst, rhs)
        rhs_id = add_instantiated_term(eg, rhs_term)
        egraph_merge(eg, class_id, rhs_id)
    return True""",
    "apply-rules": """def apply_rules(eg, rules):
    count = 0
    for root in uf_roots(egraph_uf(eg)):
        for rule in rules:
            if apply_rule(eg, rule, root):
                count += 1
    return count""",
}

CHEZ_SNIPPETS = {
    "pattern-var?": """(define (pattern-var? x)
  (if (not (symbol? x))
      #f
      (let* ([txt (symbol->string x)]
             [n (string-length txt)])
        (and (> n 1)
             (char=? #\\? (string-ref txt 0))))))""",
    "subst-try-extend": """(define (subst-try-extend subst var class-id)
  (let ([existing (subst-lookup subst var)])
    (if existing
        (if (= existing class-id)
            subst
            #f)
        (subst-extend subst var class-id))))""",
    "subst-merge": """(define (subst-merge subst1 subst2)
  (let loop ([s1 subst1] [result subst2])
    (if (null? s1)
        result
        (let* ([binding (car s1)]
               [var (car binding)]
               [class-id (cdr binding)]
               [next (subst-try-extend result var class-id)])
          (and next (loop (cdr s1) next))))))""",
    "ematch-pattern": """(define (ematch-pattern eg pattern class-id subst)
  (let ([root (egraph-find eg class-id)])
    (cond
     [(pattern-var? pattern)
      (let ([new-subst (subst-try-extend subst pattern root)])
        (if new-subst (list new-subst) '()))]
     [(or (symbol? pattern) (number? pattern))
      (if (exists (lambda (node)
                    (and (eqv? (enode-op node) pattern)
                         (zero? (enode-arity node))))
                  (egraph-class-nodes eg root))
          (list subst)
          '())]
     [(pair? pattern)
      (let* ([op (car pattern)]
             [arg-patterns (cdr pattern)]
             [nodes (egraph-class-nodes eg root)]
             [matching
              (filter (lambda (node)
                        (and (eqv? (enode-op node) op)
                             (= (enode-arity node) (length arg-patterns))))
                      nodes)])
        (apply append
               (map (lambda (node)
                      (ematch-children eg arg-patterns
                                      (vector->list (enode-children node))
                                      subst))
                    matching)))]
     [else '()])))""",
    "ematch": """(define (ematch eg pattern class-id)
  (let ([seed (empty-subst)])
    (ematch-pattern eg pattern class-id seed)))""",
    "pattern-apply": """(define (pattern-apply subst pattern)
  (cond
   [(pattern-var? pattern)
    (let ([bound (subst-lookup subst pattern)])
      (if bound
          (make-eclass-ref bound)
          (error 'pattern-apply "Unbound pattern variable" pattern)))]
   [(pair? pattern)
    (let ([head (car pattern)]
          [tail (cdr pattern)])
      (cons head
            (map (lambda (part)
                   (pattern-apply subst part))
                 tail)))]
   [else pattern]))""",
    "apply-rule": """(define (apply-rule eg rule class-id)
  (let* ([lhs (rule-lhs rule)]
         [rhs (rule-rhs rule)]
         [subs (ematch eg lhs class-id)])
    (cond
     [(null? subs) #f]
     [else
      (for-each
       (lambda (s)
         (let* ([rhs-term (pattern-apply s rhs)]
                [rhs-id (add-instantiated-term eg rhs-term)])
           (egraph-merge! eg class-id rhs-id)))
       subs)
      #t])))""",
    "apply-rules": """(define (apply-rules eg rules)
  (let ([count 0])
    (for-each
     (lambda (root)
       (for-each
        (lambda (rule)
          (if (apply-rule eg rule root)
              (set! count (+ count 1))
              #f))
        rules))
     (uf-roots (egraph-uf eg)))
    count))""",
}

BUGGY_CASES = [
    {
        "fn": "pattern-var?",
        "buggy": """(define (pattern-var? x)
  (and (symbol? x)
       (let ([s (symbol->string x)])
         (> (string-length s) 0))))""",
        "note": "A pattern variable must start with ? and have at least one character after it.",
    },
    {
        "fn": "pattern-var?",
        "buggy": """(define (pattern-var? x)
  (and (symbol? x)
       (let ([s (symbol->string x)])
         (and (> (string-length s) 1)
              (char=? (string-ref s (- (string-length s) 1)) #\\?)))))""",
        "note": "The '?' marker must be the first character, not just present somewhere in the symbol.",
    },
    {
        "fn": "subst-try-extend",
        "buggy": """(define (subst-try-extend subst var class-id)
  (let ([existing (subst-lookup subst var)])
    (if existing
        (subst-extend subst var class-id)
        (subst-extend subst var class-id))))""",
        "note": "Conflicting rebinding must return #f, not append another binding.",
    },
    {
        "fn": "subst-try-extend",
        "buggy": """(define (subst-try-extend subst var class-id)
  (let ([existing (subst-lookup subst var)])
    (cond
      [(not existing) subst]
      [(= existing class-id) subst]
      [else #f])))""",
        "note": "When the variable is absent, extend with the new binding instead of leaving substitution unchanged.",
    },
    {
        "fn": "subst-merge",
        "buggy": """(define (subst-merge subst1 subst2)
  (append subst1 subst2))""",
        "note": "Merge must detect conflicts and return #f on incompatible bindings.",
    },
    {
        "fn": "subst-merge",
        "buggy": """(define (subst-merge subst1 subst2)
  (let loop ([s1 subst1] [result (empty-subst)])
    (if (null? s1)
        result
        (let* ([b (car s1)]
               [v (car b)]
               [c (cdr b)]
               [next (subst-try-extend result v c)])
          (and next (loop (cdr s1) next))))))""",
        "note": "Merge must preserve existing bindings from subst2, not rebuild from empty substitution.",
    },
    {
        "fn": "ematch-pattern",
        "buggy": """(define (ematch-pattern eg pattern class-id subst)
  (let ([root (egraph-find eg class-id)])
    (cond
      [(pattern-var? pattern)
       (let ([new-subst (subst-try-extend subst pattern root)])
         (if new-subst (list new-subst) '()))]
      [(or (symbol? pattern) (number? pattern))
       (if (pair? (egraph-class-nodes eg root))
           (list subst)
           '())]
      [else '()])))""",
        "note": "Leaf matching must verify operator equality and zero arity, not only class non-emptiness.",
    },
    {
        "fn": "ematch-pattern",
        "buggy": """(define (ematch-pattern eg pattern class-id subst)
  (let ([root (egraph-find eg class-id)])
    (cond
      [(pattern-var? pattern)
       (let ([new-subst (subst-try-extend subst pattern root)])
         (if new-subst (list new-subst) '()))]
      [(pair? pattern)
       (let* ([op (car pattern)]
              [args (cdr pattern)]
              [nodes (egraph-class-nodes eg root)]
              [matching
               (filter (lambda (node)
                         (and (eqv? (enode-op node) op)
                              (= (enode-arity node) (length args))))
                       nodes)])
         (append-map
          (lambda (node)
            (if (null? args)
                (list subst)
                (ematch-pattern eg (car args) (vector-ref (enode-children node) 0) subst)))
          matching))]
      [else '()])))""",
        "note": "For application patterns, every child pattern must be checked; matching only the first child is incorrect.",
    },
    {
        "fn": "ematch",
        "buggy": """(define (ematch eg pattern class-id)
  (let ([matches (ematch-pattern eg pattern class-id (empty-subst))])
    (if (null? matches) '() (list (car matches)))))""",
        "note": "ematch must return all substitutions, not truncate to one.",
    },
    {
        "fn": "ematch",
        "buggy": """(define (ematch eg pattern class-id)
  (ematch-pattern eg pattern class-id (subst-extend (empty-subst) '?seed 0)))""",
        "note": "Matching should begin with an empty substitution, without introducing extraneous bindings.",
    },
    {
        "fn": "pattern-apply",
        "buggy": """(define (pattern-apply subst pattern)
  (cond
    [(pattern-var? pattern)
     (let ([binding (subst-lookup subst pattern)])
       (if binding
           binding
           (error 'pattern-apply "Unbound pattern variable" pattern)))]
    [(pair? pattern)
     (cons (car pattern)
           (map (lambda (p) (pattern-apply subst p))
                (cdr pattern)))]
    [else pattern]))""",
        "note": "Variable replacements must be wrapped with make-eclass-ref to disambiguate from numeric literals.",
    },
    {
        "fn": "pattern-apply",
        "buggy": """(define (pattern-apply subst pattern)
  (cond
    [(pattern-var? pattern)
     pattern]
    [(pair? pattern)
     (cons (car pattern)
           (map (lambda (p) (pattern-apply subst p))
                (cdr pattern)))]
    [else pattern]))""",
        "note": "Bound pattern variables must be substituted, not left unchanged.",
    },
    {
        "fn": "apply-rule",
        "buggy": """(define (apply-rule eg rule class-id)
  (let* ([lhs (rule-lhs rule)]
         [rhs (rule-rhs rule)]
         [matches (ematch eg lhs class-id)])
    (if (null? matches)
        #t
        (begin
          (for-each
           (lambda (subst)
             (let ([rhs-term (pattern-apply subst rhs)])
               (let ([rhs-id (add-instantiated-term eg rhs-term)])
                 (egraph-merge! eg class-id rhs-id))))
           matches)
          #t))))""",
        "note": "If there are no matches, apply-rule must return #f.",
    },
    {
        "fn": "apply-rule",
        "buggy": """(define (apply-rule eg rule class-id)
  (let* ([lhs (rule-lhs rule)]
         [rhs (rule-rhs rule)]
         [matches (ematch eg lhs class-id)])
    (if (null? matches)
        #f
        (begin
          (for-each
           (lambda (subst)
             (let ([rhs-term (pattern-apply subst rhs)])
               (let ([rhs-id (add-instantiated-term eg rhs-term)])
                 (egraph-merge! eg rhs-id rhs-id))))
           matches)
          #t))))""",
        "note": "New rhs term must be merged into the matched class-id, not merged with itself.",
    },
    {
        "fn": "apply-rules",
        "buggy": """(define (apply-rules eg rules)
  (let ([uf (egraph-uf eg)]
        [count 0])
    (for-each
     (lambda (root)
       (for-each
        (lambda (rule)
          (apply-rule eg rule root)
          (set! count (+ count 1)))
        rules))
     (uf-roots uf))
    count))""",
        "note": "Only successful apply-rule matches should increment count.",
    },
    {
        "fn": "apply-rules",
        "buggy": """(define (apply-rules eg rules)
  (let ([uf (egraph-uf eg)]
        [count 0])
    (for-each
     (lambda (root)
       (when (pair? rules)
         (when (apply-rule eg (car rules) root)
           (set! count (+ count 1)))))
     (uf-roots uf))
    count))""",
        "note": "Must apply every rule, not only the first rule in the list.",
    },
]

BASE_DIFFICULTY = {
    "pattern-var?": "easy",
    "subst-try-extend": "medium",
    "subst-merge": "medium",
    "ematch-pattern": "hard",
    "ematch": "easy",
    "pattern-apply": "medium",
    "apply-rule": "hard",
    "apply-rules": "hard",
}

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
DIFFICULTY_INDEX = {name: idx for idx, name in enumerate(DIFFICULTY_LEVELS)}

REQUIRED_KEYS = [
    "id",
    "family",
    "category",
    "difficulty",
    "source_module",
    "source_test",
    "source_function",
    "prompt_body",
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
    sid = f"egraph_match_{family}_{family_counter[family]:03d}"
    sample = {
        "id": sid,
        "family": family,
        "category": category,
        "difficulty": difficulty,
        "source_module": SOURCE_MODULE,
        "source_test": SOURCE_TEST,
        "source_function": source_function,
        "prompt_body": prompt.strip(),
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


def bump_difficulty(level: str, delta: int) -> str:
    idx = DIFFICULTY_INDEX[level] + delta
    idx = max(0, min(idx, len(DIFFICULTY_LEVELS) - 1))
    return DIFFICULTY_LEVELS[idx]


def task_difficulty(fn: str, family: str, task_kind: str, override: str | None = None) -> str:
    if override:
        return override

    base = BASE_DIFFICULTY[fn]
    if family == "spec_to_code":
        if task_kind == "skeleton":
            return bump_difficulty(base, -1)
        if task_kind == "contract":
            return bump_difficulty(base, +1)
        return base

    if family == "translation":
        if task_kind == "chez":
            return bump_difficulty(base, -1)
        if task_kind == "excerpt":
            return base
        return base

    if family == "bugfix":
        return base

    return base


FUNCTION_SECTION = {
    "pattern-var?": "patterns",
    "subst-try-extend": "substitutions",
    "subst-merge": "substitutions",
    "ematch-pattern": "matching",
    "ematch": "matching",
    "pattern-apply": "instantiation",
    "apply-rule": "rewrites",
    "apply-rules": "rewrites",
}


def make_source_excerpt(fn: str, snippet: str) -> str:
    section = FUNCTION_SECTION[fn]
    indented = "\n".join(f"  {line}" for line in snippet.splitlines())
    return (
        ";;; lattice/egraph/match.ss excerpt\n"
        "(require 'prelude)\n"
        "(require 'egraph/core)\n"
        "(require 'egraph/union-find)\n"
        "\n"
        "(doc 'module 'egraph/match)\n"
        f"(doc 'section '{section})\n"
        "\n"
        "(define (local-helper x) x)\n"
        "\n"
        f"{indented}\n"
    )


# -----------------------------------------------------------------------------
# Family 1: spec_to_code (24)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=task_difficulty(fn, "spec_to_code", "direct"),
        source_function=fn,
        prompt=f"""Implement this e-graph matching utility in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "egraph", "matching", "rewrite", "spec-to-code", fn],
    )

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=task_difficulty(fn, "spec_to_code", "skeleton"),
        source_function=fn,
        prompt=f"""Complete this Fold Scheme skeleton.

```scheme
{SKELETONS[fn]}
```

Replace `<TODO>` and return only the completed definition for `{fn}`.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "egraph", "matching", "rewrite", "skeleton-completion", fn],
    )

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=task_difficulty(fn, "spec_to_code", "contract"),
        source_function=fn,
        prompt=f"""Implement `{fn}` from this contract.

Module: `{SOURCE_MODULE}`
Contract focus: {FUNCTION_SPECS[fn]}

Requirements:
1. Keep the exact function name/signature.
2. Preserve behavior for edge cases in matching/rewriting.
3. Return only one production-ready definition.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "egraph", "matching", "rewrite", "contract-implementation", fn],
    )


# -----------------------------------------------------------------------------
# Family 2: translation (24)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    add_sample(
        family="translation",
        category="translation",
        difficulty=task_difficulty(fn, "translation", "python"),
        source_function=fn,
        prompt=f"""Translate this Python function into Fold-native Scheme.
Preserve behavior and keep the function name `{fn}`.
Return only the Scheme definition.

```python
{PYTHON_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "egraph", "matching", "rewrite", "python-to-scheme", fn],
    )

    add_sample(
        family="translation",
        category="translation",
        difficulty=task_difficulty(fn, "translation", "chez"),
        source_function=fn,
        prompt=f"""Convert this Chez-style snippet to canonical Fold style.
The target function must be named `{fn}`.
Return only the final Fold definition.

```scheme
{CHEZ_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "egraph", "matching", "rewrite", "chez-to-fold", fn],
    )

    add_sample(
        family="translation",
        category="translation",
        difficulty=task_difficulty(fn, "translation", "excerpt"),
        source_function=fn,
        prompt=f"""Extract and translate the target function from this source-style module excerpt.
Return only a single Fold definition for `{fn}`.
Drop metadata doc forms from the output and keep executable behavior unchanged.

```scheme
{make_source_excerpt(fn, CHEZ_SNIPPETS[fn])}
```""",
        ground_truth=DOC_FREE_DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "egraph", "matching", "rewrite", "source-excerpt-to-fold", "doc-free-target", fn],
    )


# -----------------------------------------------------------------------------
# Family 3: bugfix (16)
# -----------------------------------------------------------------------------
for case in BUGGY_CASES:
    fn = case["fn"]
    add_sample(
        family="bugfix",
        category="debugging",
        difficulty=task_difficulty(fn, "bugfix", "bugfix", str(case.get("difficulty", "")) or None),
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
        tags=["tier1", "egraph", "matching", "rewrite", "bugfix", fn],
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
        "Return only the final Fold expression."
    )
    add_sample(
        family="composition",
        category="usage",
        difficulty=difficulty,
        source_function=source_function,
        prompt=composition_prompt,
        ground_truth=ground_truth,
        verify_expr=verify_check.strip(),
        tags=["tier1", "egraph", "matching", "rewrite", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # pattern-var?
    {
        "fn": "pattern-var?",
        "prompt": "Count how many entries are pattern variables in the list `(?x y ?z 10 ?foo)`.",
        "gt": "(length (filter pattern-var? '(?x y ?z 10 ?foo)))",
        "verify": "(= (length (filter pattern-var? '(?x y ?z 10 ?foo))) 3)",
        "difficulty": "easy",
        "tags": ["counting"],
    },
    {
        "fn": "pattern-var?",
        "prompt": "Map pattern-var? over `(?x x ?foo ?)` and return the boolean list.",
        "gt": "(map pattern-var? '(?x x ?foo ?))",
        "verify": "(equal? (map pattern-var? '(?x x ?foo ?)) '(#t #f #t #f))",
        "difficulty": "easy",
        "tags": ["mapping"],
    },
    {
        "fn": "pattern-var?",
        "prompt": "Extract normalized names from the pattern-variable subset of `(?left plus ?right ?tmp)`.",
        "gt": "(map pattern-var-name (filter pattern-var? '(?left plus ?right ?tmp)))",
        "verify": "(equal? (map pattern-var-name (filter pattern-var? '(?left plus ?right ?tmp))) '(left right tmp))",
        "difficulty": "medium",
        "tags": ["names"],
    },
    {
        "fn": "pattern-var?",
        "prompt": "Return whether all entries in `(?a ?b ?c)` are valid pattern variables.",
        "gt": "(and (pattern-var? '?a) (pattern-var? '?b) (pattern-var? '?c))",
        "verify": "(equal? (and (pattern-var? '?a) (pattern-var? '?b) (pattern-var? '?c)) #t)",
        "difficulty": "easy",
        "tags": ["all-true"],
    },

    # subst-try-extend
    {
        "fn": "subst-try-extend",
        "prompt": "Extend an empty substitution with `?x -> 7` and return the looked-up value for `?x`.",
        "gt": "(let ([s (subst-try-extend (empty-subst) '?x 7)]) (if s (subst-lookup s '?x) -1))",
        "verify": "(= (let ([s (subst-try-extend (empty-subst) '?x 7)]) (if s (subst-lookup s '?x) -1)) 7)",
        "difficulty": "easy",
        "tags": ["insert"],
    },
    {
        "fn": "subst-try-extend",
        "prompt": "Attempt to rebind `?x` from 2 to 3 and return whether the extension fails.",
        "gt": "(let* ([s1 (subst-try-extend (empty-subst) '?x 2)] [s2 (and s1 (subst-try-extend s1 '?x 3))]) (not s2))",
        "verify": "(equal? (let* ([s1 (subst-try-extend (empty-subst) '?x 2)] [s2 (and s1 (subst-try-extend s1 '?x 3))]) (not s2)) #t)",
        "difficulty": "medium",
        "tags": ["conflict"],
    },
    {
        "fn": "subst-try-extend",
        "prompt": "Build a substitution with `subst-try-extend`, merge in `?z -> 9`, and verify all bindings survive.",
        "gt": "(let* ([s1 (subst-try-extend (empty-subst) '?x 5)] [s2 (and s1 (subst-try-extend s1 '?y 8))] [m (and s2 (subst-merge s2 (subst-extend (empty-subst) '?z 9)))]) (and m (= (subst-lookup m '?x) 5) (= (subst-lookup m '?y) 8) (= (subst-lookup m '?z) 9)))",
        "verify": "(equal? (let* ([s1 (subst-try-extend (empty-subst) '?x 5)] [s2 (and s1 (subst-try-extend s1 '?y 8))] [m (and s2 (subst-merge s2 (subst-extend (empty-subst) '?z 9)))]) (and m (= (subst-lookup m '?x) 5) (= (subst-lookup m '?y) 8) (= (subst-lookup m '?z) 9))) #t)",
        "difficulty": "hard",
        "tags": ["multi-fn", "merge"],
    },
    {
        "fn": "subst-try-extend",
        "prompt": "Extend with `?x -> 1`, then `?y -> 2`, then test whether changing `?y` to 3 fails.",
        "gt": "(let* ([s1 (subst-try-extend (empty-subst) '?x 1)] [s2 (and s1 (subst-try-extend s1 '?y 2))] [s3 (and s2 (subst-try-extend s2 '?y 3))]) (and s2 (not s3)))",
        "verify": "(equal? (let* ([s1 (subst-try-extend (empty-subst) '?x 1)] [s2 (and s1 (subst-try-extend s1 '?y 2))] [s3 (and s2 (subst-try-extend s2 '?y 3))]) (and s2 (not s3))) #t)",
        "difficulty": "medium",
        "tags": ["multi-var"],
    },

    # subst-merge
    {
        "fn": "subst-merge",
        "prompt": "Merge `(?x . 1)` with `(?y . 2)` and return whether both bindings are present.",
        "gt": "(let* ([a (subst-extend (empty-subst) '?x 1)] [b (subst-extend (empty-subst) '?y 2)] [m (subst-merge a b)]) (and m (= (subst-lookup m '?x) 1) (= (subst-lookup m '?y) 2)))",
        "verify": "(equal? (let* ([a (subst-extend (empty-subst) '?x 1)] [b (subst-extend (empty-subst) '?y 2)] [m (subst-merge a b)]) (and m (= (subst-lookup m '?x) 1) (= (subst-lookup m '?y) 2))) #t)",
        "difficulty": "medium",
        "tags": ["compatible"],
    },
    {
        "fn": "subst-merge",
        "prompt": "Merge conflicting substitutions for `?x` and return whether merge fails.",
        "gt": "(let* ([a (subst-extend (empty-subst) '?x 1)] [b (subst-extend (empty-subst) '?x 2)]) (not (subst-merge a b)))",
        "verify": "(equal? (let* ([a (subst-extend (empty-subst) '?x 1)] [b (subst-extend (empty-subst) '?x 2)]) (not (subst-merge a b))) #t)",
        "difficulty": "medium",
        "tags": ["conflict"],
    },
    {
        "fn": "subst-merge",
        "prompt": "Merge overlapping substitutions where `?x` is consistent and verify both vars remain available.",
        "gt": "(let* ([a (subst-extend (empty-subst) '?x 3)] [b (subst-extend (subst-extend (empty-subst) '?x 3) '?z 9)] [m (subst-merge a b)]) (and m (= (subst-lookup m '?x) 3) (= (subst-lookup m '?z) 9)))",
        "verify": "(equal? (let* ([a (subst-extend (empty-subst) '?x 3)] [b (subst-extend (subst-extend (empty-subst) '?x 3) '?z 9)] [m (subst-merge a b)]) (and m (= (subst-lookup m '?x) 3) (= (subst-lookup m '?z) 9))) #t)",
        "difficulty": "medium",
        "tags": ["overlap"],
    },
    {
        "fn": "subst-merge",
        "prompt": "Create one substitution via `subst-try-extend`, merge with another, and verify all three variables are preserved.",
        "gt": "(let* ([a0 (subst-try-extend (empty-subst) '?x 7)] [a (and a0 (subst-try-extend a0 '?y 8))] [b (subst-try-extend (empty-subst) '?z 9)] [m (and a b (subst-merge a b))]) (and m (= (subst-lookup m '?x) 7) (= (subst-lookup m '?y) 8) (= (subst-lookup m '?z) 9)))",
        "verify": "(equal? (let* ([a0 (subst-try-extend (empty-subst) '?x 7)] [a (and a0 (subst-try-extend a0 '?y 8))] [b (subst-try-extend (empty-subst) '?z 9)] [m (and a b (subst-merge a b))]) (and m (= (subst-lookup m '?x) 7) (= (subst-lookup m '?y) 8) (= (subst-lookup m '?z) 9))) #t)",
        "difficulty": "hard",
        "tags": ["multi-fn", "transitive"],
    },

    # ematch-pattern
    {
        "fn": "ematch-pattern",
        "prompt": "Match `(+ ?x ?y)` against `(+ a b)`, then instantiate `(+ ?y ?x)` from the first substitution.",
        "gt": "(let* ([eg (make-egraph)] [id (egraph-add-term! eg '(+ a b))] [ms (ematch-pattern eg '(+ ?x ?y) id (empty-subst))] [s (and (pair? ms) (car ms))] [rhs (and s (pattern-apply s '(+ ?y ?x)))]) (and s (pair? rhs) (eq? (car rhs) '+) (eclass-ref? (cadr rhs)) (eclass-ref? (caddr rhs))))",
        "verify": "(equal? (let* ([eg (make-egraph)] [id (egraph-add-term! eg '(+ a b))] [ms (ematch-pattern eg '(+ ?x ?y) id (empty-subst))] [s (and (pair? ms) (car ms))] [rhs (and s (pattern-apply s '(+ ?y ?x)))]) (and s (pair? rhs) (eq? (car rhs) '+) (eclass-ref? (cadr rhs)) (eclass-ref? (caddr rhs)))) #t)",
        "difficulty": "hard",
        "tags": ["compound", "multi-fn"],
    },
    {
        "fn": "ematch-pattern",
        "prompt": "Match repeated-var pattern `(+ ?a ?a)` against `(+ x y)` and return whether it fails.",
        "gt": "(let* ([eg (make-egraph)] [id (egraph-add-term! eg '(+ x y))]) (null? (ematch-pattern eg '(+ ?a ?a) id (empty-subst))))",
        "verify": "(equal? (let* ([eg (make-egraph)] [id (egraph-add-term! eg '(+ x y))]) (null? (ematch-pattern eg '(+ ?a ?a) id (empty-subst)))) #t)",
        "difficulty": "hard",
        "tags": ["repeated-var"],
    },
    {
        "fn": "ematch-pattern",
        "prompt": "Match leaf pattern `x` against class for `x` and return whether exactly one substitution is returned.",
        "gt": "(let* ([eg (make-egraph)] [id (egraph-add-term! eg 'x)] [ms (ematch-pattern eg 'x id (empty-subst))]) (= (length ms) 1))",
        "verify": "(equal? (let* ([eg (make-egraph)] [id (egraph-add-term! eg 'x)] [ms (ematch-pattern eg 'x id (empty-subst))]) (= (length ms) 1)) #t)",
        "difficulty": "medium",
        "tags": ["leaf"],
    },
    {
        "fn": "ematch-pattern",
        "prompt": "Match wrong operator `(* ?x ?y)` against `(+ a b)` and return whether no matches are produced.",
        "gt": "(let* ([eg (make-egraph)] [id (egraph-add-term! eg '(+ a b))]) (null? (ematch-pattern eg '(* ?x ?y) id (empty-subst))))",
        "verify": "(equal? (let* ([eg (make-egraph)] [id (egraph-add-term! eg '(+ a b))]) (null? (ematch-pattern eg '(* ?x ?y) id (empty-subst)))) #t)",
        "difficulty": "hard",
        "tags": ["op-mismatch"],
    },

    # ematch
    {
        "fn": "ematch",
        "prompt": "Run `ematch` with variable pattern `?v` on class of `x` and return whether there is one match.",
        "gt": "(let* ([eg (make-egraph)] [x (egraph-add-term! eg 'x)]) (= (length (ematch eg '?v x)) 1))",
        "verify": "(equal? (let* ([eg (make-egraph)] [x (egraph-add-term! eg 'x)]) (= (length (ematch eg '?v x)) 1)) #t)",
        "difficulty": "easy",
        "tags": ["variable"],
    },
    {
        "fn": "ematch",
        "prompt": "Compare `ematch` vs `ematch-pattern` on `(f ?x)` for class `(f a)` and return whether they agree on match count.",
        "gt": "(let* ([eg (make-egraph)] [id (egraph-add-term! eg '(f a))] [m1 (ematch eg '(f ?x) id)] [m2 (ematch-pattern eg '(f ?x) id (empty-subst))]) (= (length m1) (length m2)))",
        "verify": "(equal? (let* ([eg (make-egraph)] [id (egraph-add-term! eg '(f a))] [m1 (ematch eg '(f ?x) id)] [m2 (ematch-pattern eg '(f ?x) id (empty-subst))]) (= (length m1) (length m2))) #t)",
        "difficulty": "medium",
        "tags": ["wrapper-consistency", "multi-fn"],
    },
    {
        "fn": "ematch",
        "prompt": "Run `ematch` for `(g ?x)` on class of `(f a)` and return whether it fails.",
        "gt": "(let* ([eg (make-egraph)] [id (egraph-add-term! eg '(f a))]) (null? (ematch eg '(g ?x) id)))",
        "verify": "(equal? (let* ([eg (make-egraph)] [id (egraph-add-term! eg '(f a))]) (null? (ematch eg '(g ?x) id))) #t)",
        "difficulty": "medium",
        "tags": ["mismatch"],
    },
    {
        "fn": "ematch",
        "prompt": "Merge classes of `(f x)` and `(f y)`, then count matches of `(f ?a)` from merged class.",
        "gt": "(let* ([eg (make-egraph)] [fx (egraph-add-term! eg '(f x))] [fy (egraph-add-term! eg '(f y))]) (egraph-merge! eg fx fy) (egraph-saturate-rebuild! eg) (length (ematch eg '(f ?a) fx)))",
        "verify": "(= (let* ([eg (make-egraph)] [fx (egraph-add-term! eg '(f x))] [fy (egraph-add-term! eg '(f y))]) (egraph-merge! eg fx fy) (egraph-saturate-rebuild! eg) (length (ematch eg '(f ?a) fx))) 2)",
        "difficulty": "hard",
        "tags": ["merged-class"],
    },

    # pattern-apply
    {
        "fn": "pattern-apply",
        "prompt": "Apply substitution `{?x->5}` to `?x` and return whether output is an eclass-ref with id 5.",
        "gt": "(let* ([s (subst-extend (empty-subst) '?x 5)] [r (pattern-apply s '?x)]) (and (eclass-ref? r) (= (eclass-ref-id r) 5)))",
        "verify": "(equal? (let* ([s (subst-extend (empty-subst) '?x 5)] [r (pattern-apply s '?x)]) (and (eclass-ref? r) (= (eclass-ref-id r) 5))) #t)",
        "difficulty": "easy",
        "tags": ["var-instantiate"],
    },
    {
        "fn": "pattern-apply",
        "prompt": "Apply substitution on `(+ ?x ?y)` and return whether both children are eclass refs.",
        "gt": "(let* ([s (subst-extend (subst-extend (empty-subst) '?x 1) '?y 2)] [r (pattern-apply s '(+ ?x ?y))]) (and (eclass-ref? (cadr r)) (eclass-ref? (caddr r))))",
        "verify": "(equal? (let* ([s (subst-extend (subst-extend (empty-subst) '?x 1) '?y 2)] [r (pattern-apply s '(+ ?x ?y))]) (and (eclass-ref? (cadr r)) (eclass-ref? (caddr r)))) #t)",
        "difficulty": "medium",
        "tags": ["compound"],
    },
    {
        "fn": "pattern-apply",
        "prompt": "Apply substitution to `(f ?x 42)` and return whether literal 42 is preserved.",
        "gt": "(let* ([s (subst-extend (empty-subst) '?x 9)] [r (pattern-apply s '(f ?x 42))]) (= (caddr r) 42))",
        "verify": "(equal? (let* ([s (subst-extend (empty-subst) '?x 9)] [r (pattern-apply s '(f ?x 42))]) (= (caddr r) 42)) #t)",
        "difficulty": "medium",
        "tags": ["literal-preserve"],
    },
    {
        "fn": "pattern-apply",
        "prompt": "Apply substitution to nested pattern `(f (g ?x))` and return whether nested arg is an eclass-ref.",
        "gt": "(let* ([s (subst-extend (empty-subst) '?x 4)] [r (pattern-apply s '(f (g ?x)))]) (eclass-ref? (cadr (cadr r))))",
        "verify": "(equal? (let* ([s (subst-extend (empty-subst) '?x 4)] [r (pattern-apply s '(f (g ?x)))]) (eclass-ref? (cadr (cadr r)))) #t)",
        "difficulty": "medium",
        "tags": ["nested"],
    },

    # apply-rule
    {
        "fn": "apply-rule",
        "prompt": "Apply `(+ ?x 0) => ?x` to class of `(+ a 0)` and return whether it becomes equivalent to `a`.",
        "gt": "(let* ([eg (make-egraph)] [r (make-rule '(+ ?x 0) '?x)] [term (egraph-add-term! eg '(+ a 0))] [a-id (egraph-add-term! eg 'a)] [ok (apply-rule eg r term)]) (egraph-saturate-rebuild! eg) (and ok (= (egraph-find eg term) (egraph-find eg a-id))))",
        "verify": "(equal? (let* ([eg (make-egraph)] [r (make-rule '(+ ?x 0) '?x)] [term (egraph-add-term! eg '(+ a 0))] [a-id (egraph-add-term! eg 'a)] [ok (apply-rule eg r term)]) (egraph-saturate-rebuild! eg) (and ok (= (egraph-find eg term) (egraph-find eg a-id)))) #t)",
        "difficulty": "hard",
        "tags": ["identity"],
    },
    {
        "fn": "apply-rule",
        "prompt": "Apply `(* ?x 1) => ?x` to class of `(+ a 0)` and return whether no match is reported.",
        "gt": "(let* ([eg (make-egraph)] [r (make-rule '(* ?x 1) '?x)] [term (egraph-add-term! eg '(+ a 0))]) (not (apply-rule eg r term)))",
        "verify": "(equal? (let* ([eg (make-egraph)] [r (make-rule '(* ?x 1) '?x)] [term (egraph-add-term! eg '(+ a 0))]) (not (apply-rule eg r term))) #t)",
        "difficulty": "medium",
        "tags": ["no-match"],
    },
    {
        "fn": "apply-rule",
        "prompt": "Apply commutativity rule `(+ ?x ?y) => (+ ?y ?x)` to `(+ a b)` and return class node count after rebuild.",
        "gt": "(let* ([eg (make-egraph)] [r (make-rule '(+ ?x ?y) '(+ ?y ?x))] [term (egraph-add-term! eg '(+ a b))]) (apply-rule eg r term) (egraph-saturate-rebuild! eg) (length (egraph-class-nodes eg term)))",
        "verify": "(= (let* ([eg (make-egraph)] [r (make-rule '(+ ?x ?y) '(+ ?y ?x))] [term (egraph-add-term! eg '(+ a b))]) (apply-rule eg r term) (egraph-saturate-rebuild! eg) (length (egraph-class-nodes eg term))) 2)",
        "difficulty": "hard",
        "tags": ["commutativity"],
    },
    {
        "fn": "apply-rule",
        "prompt": "Apply expansion rule `?x => (+ ?x 0)` to `a` and return whether `a` becomes equivalent to `(+ a 0)`.",
        "gt": "(let* ([eg (make-egraph)] [r (make-rule '?x '(+ ?x 0))] [a-id (egraph-add-term! eg 'a)] [sum-id (egraph-add-term! eg '(+ a 0))] [ok (apply-rule eg r a-id)]) (egraph-saturate-rebuild! eg) (and ok (= (egraph-find eg a-id) (egraph-find eg sum-id))))",
        "verify": "(equal? (let* ([eg (make-egraph)] [r (make-rule '?x '(+ ?x 0))] [a-id (egraph-add-term! eg 'a)] [sum-id (egraph-add-term! eg '(+ a 0))] [ok (apply-rule eg r a-id)]) (egraph-saturate-rebuild! eg) (and ok (= (egraph-find eg a-id) (egraph-find eg sum-id)))) #t)",
        "difficulty": "hard",
        "tags": ["expansion"],
    },

    # apply-rules
    {
        "fn": "apply-rules",
        "prompt": "Run two simplification rules once, then confirm both rewritten terms are equivalent to their simplified symbols.",
        "gt": "(let* ([eg (make-egraph)] [rules (list (make-rule '(+ ?x 0) '?x) (make-rule '(* ?x 1) '?x))] [ta (egraph-add-term! eg '(+ a 0))] [tb (egraph-add-term! eg '(* b 1))] [a-id (egraph-add-term! eg 'a)] [b-id (egraph-add-term! eg 'b)] [count (apply-rules eg rules)]) (egraph-saturate-rebuild! eg) (and (= count 2) (= (egraph-find eg ta) (egraph-find eg a-id)) (= (egraph-find eg tb) (egraph-find eg b-id))))",
        "verify": "(equal? (let* ([eg (make-egraph)] [rules (list (make-rule '(+ ?x 0) '?x) (make-rule '(* ?x 1) '?x))] [ta (egraph-add-term! eg '(+ a 0))] [tb (egraph-add-term! eg '(* b 1))] [a-id (egraph-add-term! eg 'a)] [b-id (egraph-add-term! eg 'b)] [count (apply-rules eg rules)]) (egraph-saturate-rebuild! eg) (and (= count 2) (= (egraph-find eg ta) (egraph-find eg a-id)) (= (egraph-find eg tb) (egraph-find eg b-id)))) #t)",
        "difficulty": "hard",
        "tags": ["rewrite-effect", "multi-fn"],
    },
    {
        "fn": "apply-rules",
        "prompt": "Apply a commutativity+identity rule set and check that `ematch` can still find an addition shape in the rewritten class.",
        "gt": "(let* ([eg (make-egraph)] [rules (list (make-rule '(+ ?x ?y) '(+ ?y ?x)) (make-rule '(+ ?x 0) '?x))] [t (egraph-add-term! eg '(+ a 0))]) (apply-rules eg rules) (egraph-saturate-rebuild! eg) (>= (length (ematch eg '(+ ?u ?v) t)) 1))",
        "verify": "(equal? (let* ([eg (make-egraph)] [rules (list (make-rule '(+ ?x ?y) '(+ ?y ?x)) (make-rule '(+ ?x 0) '?x))] [t (egraph-add-term! eg '(+ a 0))]) (apply-rules eg rules) (egraph-saturate-rebuild! eg) (>= (length (ematch eg '(+ ?u ?v) t)) 1)) #t)",
        "difficulty": "medium",
        "tags": ["post-match", "multi-fn"],
    },
    {
        "fn": "apply-rules",
        "prompt": "Run a simplification rule set over non-matching terms and return whether the sweep records zero matches.",
        "gt": "(let* ([eg (make-egraph)] [rules (list (make-rule '(+ ?x 0) '?x) (make-rule '(* ?x 1) '?x))]) (egraph-add-term! eg '(+ a 1)) (egraph-add-term! eg '(* b 2)) (= (apply-rules eg rules) 0))",
        "verify": "(equal? (let* ([eg (make-egraph)] [rules (list (make-rule '(+ ?x 0) '?x) (make-rule '(* ?x 1) '?x))]) (egraph-add-term! eg '(+ a 1)) (egraph-add-term! eg '(* b 2)) (= (apply-rules eg rules) 0)) #t)",
        "difficulty": "medium",
        "tags": ["no-op-sweep"],
    },
    {
        "fn": "apply-rules",
        "prompt": "Use `apply-rules` first, then a direct `apply-rule` call, and verify both rewrites contribute to expected equivalences.",
        "gt": "(let* ([eg (make-egraph)] [rules (list (make-rule '(+ ?x 0) '?x))] [sum (egraph-add-term! eg '(+ a 0))] [prod (egraph-add-term! eg '(* b 1))] [a-id (egraph-add-term! eg 'a)] [b-id (egraph-add-term! eg 'b)] [extra (make-rule '(* ?x 1) '?x)] [count (apply-rules eg rules)] [ok2 (apply-rule eg extra prod)]) (egraph-saturate-rebuild! eg) (and (= count 1) ok2 (= (egraph-find eg sum) (egraph-find eg a-id)) (= (egraph-find eg prod) (egraph-find eg b-id))))",
        "verify": "(equal? (let* ([eg (make-egraph)] [rules (list (make-rule '(+ ?x 0) '?x))] [sum (egraph-add-term! eg '(+ a 0))] [prod (egraph-add-term! eg '(* b 1))] [a-id (egraph-add-term! eg 'a)] [b-id (egraph-add-term! eg 'b)] [extra (make-rule '(* ?x 1) '?x)] [count (apply-rules eg rules)] [ok2 (apply-rule eg extra prod)]) (egraph-saturate-rebuild! eg) (and (= count 1) ok2 (= (egraph-find eg sum) (egraph-find eg a-id)) (= (egraph-find eg prod) (egraph-find eg b-id)))) #t)",
        "difficulty": "hard",
        "tags": ["staged-rewrite", "multi-fn"],
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

if len(samples) != 96:
    raise ValueError(f"expected 96 samples, found {len(samples)}")

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


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


pre_diversify_rows = [dict(sample, split=("eval" if sample["id"] in eval_ids else "train")) for sample in samples]
write_jsonl(PRE_DIVERSIFY_PATH, pre_diversify_rows)

env = os.environ.copy()
env["SFT_EVAL_RATIO"] = str(EVAL_RATIO)
proc = subprocess.run(
    [
        "scheme",
        "--script",
        str(SFT_GENERATOR_PATH),
        str(PRE_DIVERSIFY_PATH),
        str(OUT_DIR),
    ],
    cwd=str(REPO_ROOT),
    env=env,
    text=True,
    capture_output=True,
)
if proc.returncode != 0:
    raise RuntimeError(
        "DSL prompt generation failed:\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )

all_rows = read_jsonl(ALL_PATH)
train_rows_out = read_jsonl(TRAIN_PATH)
eval_rows_out = read_jsonl(EVAL_PATH)
if len(all_rows) != len(samples):
    raise ValueError(f"dsl output size mismatch: expected {len(samples)}, got {len(all_rows)}")
if len(train_rows_out) != len(train_rows) or len(eval_rows_out) != len(eval_rows):
    raise ValueError(
        "dsl split mismatch: "
        f"train={len(train_rows_out)} (expected {len(train_rows)}), "
        f"eval={len(eval_rows_out)} (expected {len(eval_rows)})"
    )
if any("prompt" not in row or not str(row["prompt"]).strip() for row in all_rows):
    raise ValueError("dsl output missing prompt in one or more rows")
if any("prompt_body" not in row or not str(row["prompt_body"]).strip() for row in all_rows):
    raise ValueError("dsl output missing prompt_body in one or more rows")
PRE_DIVERSIFY_PATH.unlink(missing_ok=True)

summary = {
    "total": len(samples),
    "train": len(train_rows_out),
    "eval": len(eval_rows_out),
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
