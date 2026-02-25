#!/usr/bin/env python3
"""Generate Tier-1 egraph extract SFT samples for lattice/egraph/extract.ss."""

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

SOURCE_MODULE = "lattice/egraph/extract.ss"
SOURCE_TEST = "lattice/egraph/test-extract.ss"

DEFS: Dict[str, str] = {
    "make-extraction-state": """(define (make-extraction-state eg cost-model)
  (doc 'type (-> EGraph CostModel ExtractionState))
  (doc 'description "Create extraction state with precomputed best nodes.")
  (doc 'export #t)
  (let ([costs (compute-costs eg cost-model)]
        [uf (egraph-uf eg)]
        [store (egraph-classes eg)]
        [node-cost-fn (cost-model-fn cost-model)])
    ;; Find best node for each class
    (let ([best-nodes
           (fold-left
            (lambda (bn root)
              (let find-best ([nodes (eclass-get-nodes store root)]
                              [best-node #f]
                              [best-cost +inf.0])
                (if (null? nodes)
                    (if best-node (hamt-assoc root best-node bn) bn)
                    (let ([c (compute-node-cost (car nodes) costs node-cost-fn)])
                      (if (< c best-cost)
                          (find-best (cdr nodes) (car nodes) c)
                          (find-best (cdr nodes) best-node best-cost))))))
            hamt-empty
            (uf-roots uf))])
      (vector extraction-state-tag eg costs best-nodes))))""",
    "extraction-state?": """(define (extraction-state? x)
  (doc 'type (-> Any Boolean))
  (doc 'description "Check if x is an extraction state.")
  (doc 'export #t)
  (and (vector? x)
       (>= (vector-length x) 4)
       (eq? (vector-ref x 0) extraction-state-tag)))""",
    "extract": """(define (extract state class-id)
  (doc 'type (-> ExtractionState ClassId Term))
  (doc 'description "Extract minimum-cost term from e-class.")
  (doc 'export #t)
  (let* ([eg (state-egraph state)]
         [root (egraph-find eg class-id)]
         [best-nodes (state-best-nodes state)]
         [best-node (hamt-lookup root best-nodes)])
    (if (not best-node)
        ;; Shouldn't happen in a well-formed e-graph
        (error 'extract "no best node for class" root)
        (let ([op (enode-op best-node)]
              [children (enode-children best-node)])
          (if (zero? (vector-length children))
              ;; Leaf: return the operator as-is
              op
              ;; Compound: recursively extract children
              (cons op
                    (map (lambda (i)
                           (extract state (vector-ref children i)))
                         (iota (vector-length children)))))))))""",
    "extract-term": """(define (extract-term eg term cost-model)
  (doc 'type (-> EGraph Term CostModel Term))
  (doc 'description "Add term to e-graph and extract optimal equivalent.")
  (doc 'export #t)
  (let* ([class-id (egraph-add-term! eg term)]
         [state (make-extraction-state eg cost-model)])
    (extract state class-id)))""",
    "extract-all": """(define (extract-all state)
  (doc 'type (-> ExtractionState (List (Pair ClassId Term))))
  (doc 'description "Extract optimal terms for all root e-classes.")
  (doc 'export #t)
  (let* ([eg (state-egraph state)]
         [uf (egraph-uf eg)])
    (map (lambda (root)
           (cons root (extract state root)))
         (uf-roots uf))))""",
    "optimize": """(define (optimize term rules cost-model)
  (doc 'type (-> Term (List Rule) CostModel Term))
  (doc 'description "Apply rules via equality saturation, extract optimal form.")
  (doc 'export #t)
  (let ([eg (make-egraph)])
    (let ([root (egraph-add-term! eg term)])
      (saturate-simple eg rules)
      (let ([state (make-extraction-state eg cost-model)])
        (extract state root)))))""",
    "optimize-with-config": """(define (optimize-with-config term rules cost-model config)
  (doc 'type (-> Term (List Rule) CostModel SaturationConfig Term))
  (doc 'description "Optimize with custom saturation config.")
  (doc 'export #t)
  (let ([eg (make-egraph)])
    (let ([root (egraph-add-term! eg term)])
      (saturate eg rules config)
      (let ([state (make-extraction-state eg cost-model)])
        (extract state root)))))""",
    "compare-extractions": """(define (compare-extractions eg class-id cost-models)
  (doc 'type (-> EGraph ClassId (List CostModel) (List (Triple CostModel Term Nat))))
  (doc 'description "Compare extractions across different cost models.")
  (doc 'export #t)
  (map (lambda (cm)
         (let* ([state (make-extraction-state eg cm)]
                [term (extract state class-id)]
                [cost (class-cost (state-costs state) (egraph-find eg class-id))])
           (list cm term cost)))
       cost-models))""",
}


def strip_doc_forms(defn: str) -> str:
    lines = [line for line in defn.splitlines() if not line.strip().startswith("(doc ")]
    return "\n".join(lines)


DOC_FREE_DEFS: Dict[str, str] = {fn: strip_doc_forms(code) for fn, code in DEFS.items()}

FUNCTION_ORDER = [
    "make-extraction-state",
    "extraction-state?",
    "extract",
    "extract-term",
    "extract-all",
    "optimize",
    "optimize-with-config",
    "compare-extractions",
]

FUNCTION_SPECS = {
    "make-extraction-state": "Compute class costs and choose minimum-cost enode for each root e-class.",
    "extraction-state?": "Return #t only for extraction-state vectors tagged with extraction-state-tag and arity >= 4.",
    "extract": "Extract the minimum-cost term for a class by following selected enodes recursively.",
    "extract-term": "Insert a term into an egraph, build extraction state, then extract optimal equivalent.",
    "extract-all": "Return (root . extracted-term) for every root e-class in the state's egraph.",
    "optimize": "Run saturation with rules using default config, then extract optimal term from the original root.",
    "optimize-with-config": "Same as optimize, but use caller-provided saturation config.",
    "compare-extractions": "For each cost model, return (cost-model extracted-term class-cost) for the target class.",
}

SKELETONS = {
    "make-extraction-state": """(define (make-extraction-state eg cost-model)
  ;; TODO: compute costs and choose best enode per root class
  <TODO>)""",
    "extraction-state?": """(define (extraction-state? x)
  ;; TODO: validate extraction-state tag and structural shape
  <TODO>)""",
    "extract": """(define (extract state class-id)
  ;; TODO: lookup best node for class and recursively rebuild term
  <TODO>)""",
    "extract-term": """(define (extract-term eg term cost-model)
  ;; TODO: add term, make extraction state, extract best equivalent
  <TODO>)""",
    "extract-all": """(define (extract-all state)
  ;; TODO: extract one optimal term for each root class
  <TODO>)""",
    "optimize": """(define (optimize term rules cost-model)
  ;; TODO: saturate then extract from the original root
  <TODO>)""",
    "optimize-with-config": """(define (optimize-with-config term rules cost-model config)
  ;; TODO: saturate with config then extract from the original root
  <TODO>)""",
    "compare-extractions": """(define (compare-extractions eg class-id cost-models)
  ;; TODO: run extraction under each cost model and report (model term cost)
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "make-extraction-state": "(let* ([eg (make-egraph)] [exp-id (egraph-add-term! eg '(+ (+ x 0) 0))] [x-id (egraph-add-term! eg 'x)]) (egraph-merge! eg exp-id x-id) (egraph-rebuild! eg) (let* ([st (make-extraction-state eg ast-size-cost)] [root (egraph-find eg exp-id)] [best (hamt-lookup root (state-best-nodes st))]) (and (extraction-state? st) (eq? (state-egraph st) eg) (not (hamt-empty? (state-costs st))) best (equal? (extract st exp-id) 'x) (eq? (enode-op best) 'x))))",
    "extraction-state?": "(let ([eg (make-egraph)]) (egraph-add-term! eg 'x) (let ([st (make-extraction-state eg ast-size-cost)]) (and (extraction-state? st) (not (extraction-state? '#(extraction-state 1 2))) (not (extraction-state? '#(wrong-tag 1 2 3 4))) (not (extraction-state? 42)))))",
    "extract": "(let* ([eg (make-egraph)] [id1 (egraph-add-term! eg '(+ a b))] [id2 (egraph-add-term! eg '(* (+ a b) c))] [x-id (egraph-add-term! eg 'x)] [sum-id (egraph-add-term! eg '(+ x 0))]) (egraph-merge! eg x-id sum-id) (egraph-rebuild! eg) (let* ([st (make-extraction-state eg ast-size-cost)] [t1 (extract st id1)] [t2 (extract st id2)] [t3 (extract st sum-id)]) (and (equal? t1 '(+ a b)) (equal? t2 '(* (+ a b) c)) (equal? t3 'x))))",
    "extract-term": "(let* ([eg1 (make-egraph)] [t1 (extract-term eg1 '(+ a b) ast-size-cost)] [eg2 (make-egraph)] [x (egraph-add-term! eg2 'x)] [sum (egraph-add-term! eg2 '(+ x 0))]) (egraph-merge! eg2 x sum) (egraph-rebuild! eg2) (let ([t2 (extract-term eg2 '(+ x 0) ast-size-cost)]) (and (equal? t1 '(+ a b)) (equal? t2 'x))))",
    "extract-all": "(let ([eg (make-egraph)]) (let ([a (egraph-add-term! eg 'a)] [b (egraph-add-term! eg 'b)] [ab (egraph-add-term! eg '(+ a b))]) (let* ([st (make-extraction-state eg ast-size-cost)] [all (extract-all st)] [entry-a (find (lambda (p) (= (car p) a)) all)]) (and (= (length all) 3) (pair? (car all)) entry-a (equal? (cdr entry-a) 'a)))))",
    "optimize": "(and (equal? (optimize '(+ x 0) arith-identity-rules ast-size-cost) 'x) (equal? (optimize '(* x 0) arith-identity-rules ast-size-cost) '0) (equal? (optimize '(+ a b) '() ast-size-cost) '(+ a b)))",
    "optimize-with-config": "(let* ([cfg (make-saturation-config 1 10000 0)] [r1 (optimize-with-config '(+ x 0) arith-identity-rules ast-size-cost cfg)] [r2 (optimize-with-config '(* x 1) arith-identity-rules ast-size-cost cfg)]) (and (equal? r1 'x) (equal? r2 'x)))",
    "compare-extractions": "(let ([eg (make-egraph)]) (let ([id (egraph-add-term! eg '(+ x y))]) (let ([results (compare-extractions eg id (list ast-size-cost ast-depth-cost))]) (and (= (length results) 2) (let ([r1 (car results)] [r2 (cadr results)]) (and (cost-model? (car r1)) (equal? (cadr r1) '(+ x y)) (number? (caddr r1)) (cost-model? (car r2)) (equal? (cadr r2) '(+ x y)) (number? (caddr r2))))))))",
}

PYTHON_SNIPPETS = {
    "make-extraction-state": """def make_extraction_state(eg, cost_model):
    costs = compute_costs(eg, cost_model)
    best_nodes = {}
    node_cost = cost_model.fn
    for root in uf_roots(egraph_uf(eg)):
        best_node = None
        best_cost = float('inf')
        for node in eclass_nodes(egraph_classes(eg), root):
            c = compute_node_cost(node, costs, node_cost)
            if c < best_cost:
                best_node = node
                best_cost = c
        if best_node is not None:
            best_nodes[root] = best_node
    return ('extraction-state', eg, costs, best_nodes)""",
    "extraction-state?": """def extraction_state_p(x):
    return (isinstance(x, tuple)
            and len(x) >= 4
            and x[0] == 'extraction-state')""",
    "extract": """def extract(state, class_id):
    eg = state_egraph(state)
    root = egraph_find(eg, class_id)
    best = state_best_nodes(state).get(root)
    if best is None:
        raise ValueError('no best node for class')
    if len(best.children) == 0:
        return best.op
    return [best.op] + [extract(state, cid) for cid in best.children]""",
    "extract-term": """def extract_term(eg, term, cost_model):
    cid = egraph_add_term(eg, term)
    state = make_extraction_state(eg, cost_model)
    return extract(state, cid)""",
    "extract-all": """def extract_all(state):
    eg = state_egraph(state)
    return [(root, extract(state, root)) for root in uf_roots(egraph_uf(eg))]""",
    "optimize": """def optimize(term, rules, cost_model):
    eg = make_egraph()
    root = egraph_add_term(eg, term)
    saturate_simple(eg, rules)
    state = make_extraction_state(eg, cost_model)
    return extract(state, root)""",
    "optimize-with-config": """def optimize_with_config(term, rules, cost_model, config):
    eg = make_egraph()
    root = egraph_add_term(eg, term)
    saturate(eg, rules, config)
    state = make_extraction_state(eg, cost_model)
    return extract(state, root)""",
    "compare-extractions": """def compare_extractions(eg, class_id, models):
    out = []
    for cm in models:
        state = make_extraction_state(eg, cm)
        term = extract(state, class_id)
        cost = class_cost(state_costs(state), egraph_find(eg, class_id))
        out.append((cm, term, cost))
    return out""",
}

CHEZ_SNIPPETS = {
    "make-extraction-state": """(define (make-extraction-state eg cost-model)
  (let* ([costs (compute-costs eg cost-model)]
         [uf (egraph-uf eg)]
         [store (egraph-classes eg)]
         [node-cost-fn (cost-model-fn cost-model)])
    (define (best-node-for root)
      (let pick ([nodes (eclass-get-nodes store root)]
                 [best #f]
                 [best-cost +inf.0])
        (if (null? nodes)
            best
            (let* ([node (car nodes)]
                   [c (compute-node-cost node costs node-cost-fn)])
              (if (< c best-cost)
                  (pick (cdr nodes) node c)
                  (pick (cdr nodes) best best-cost))))))
    (let loop-roots ([roots (uf-roots uf)]
                     [best-nodes hamt-empty])
      (if (null? roots)
          (vector extraction-state-tag eg costs best-nodes)
          (let* ([root (car roots)]
                 [best (best-node-for root)]
                 [next (if best (hamt-assoc root best best-nodes) best-nodes)])
            (loop-roots (cdr roots) next)))))""",
    "extraction-state?": """(define (extraction-state? x)
  (if (vector? x)
      (let ([n (vector-length x)])
        (if (>= n 4)
            (eq? (vector-ref x 0) extraction-state-tag)
            #f))
      #f))""",
    "extract": """(define (extract state class-id)
  (let* ([eg (state-egraph state)]
         [root (egraph-find eg class-id)]
         [best (hamt-lookup root (state-best-nodes state))])
    (if (not best)
        (error 'extract "no best node for class" root)
        (let ([children (enode-children best)])
          (if (zero? (vector-length children))
              (enode-op best)
              (cons (enode-op best)
                    (map (lambda (i)
                           (extract state (vector-ref children i)))
                         (iota (vector-length children)))))))))""",
    "extract-term": """(define (extract-term eg term cost-model)
  (let ([class-id (egraph-add-term! eg term)])
    (extract (make-extraction-state eg cost-model) class-id)))""",
    "extract-all": """(define (extract-all state)
  (let* ([eg (state-egraph state)]
         [uf (egraph-uf eg)])
    (reverse
     (fold-left
      (lambda (acc root)
        (cons (cons root (extract state root)) acc))
      '()
      (uf-roots uf)))))""",
    "optimize": """(define (optimize term rules cost-model)
  (let* ([eg (make-egraph)]
         [root (egraph-add-term! eg term)]
         [_ (saturate-simple eg rules)])
    (extract (make-extraction-state eg cost-model) root))""",
    "optimize-with-config": """(define (optimize-with-config term rules cost-model config)
  (let* ([eg (make-egraph)]
         [root (egraph-add-term! eg term)]
         [_ (saturate eg rules config)])
    (extract (make-extraction-state eg cost-model) root))""",
    "compare-extractions": """(define (compare-extractions eg class-id cost-models)
  (fold-right
   (lambda (cm acc)
     (let* ([state (make-extraction-state eg cm)]
            [term (extract state class-id)]
            [cost (class-cost (state-costs state) (egraph-find eg class-id))])
       (cons (list cm term cost) acc)))
   '()
   cost-models))""",
}

BUGGY_CASES = [
    {
        "fn": "make-extraction-state",
        "buggy": """(define (make-extraction-state eg cost-model)
  (let ([costs (compute-costs eg cost-model)]
        [uf (egraph-uf eg)])
    (vector extraction-state-tag eg costs hamt-empty)))""",
        "note": "State must include best-node selections for each root class; leaving table empty breaks extraction.",
    },
    {
        "fn": "make-extraction-state",
        "buggy": """(define (make-extraction-state eg cost-model)
  (let ([costs (compute-costs eg cost-model)]
        [uf (egraph-uf eg)]
        [store (egraph-classes eg)])
    (let ([best-nodes
           (fold-left
            (lambda (bn root)
              (let ([nodes (eclass-get-nodes store root)])
                (if (null? nodes)
                    bn
                    (hamt-assoc root (car nodes) bn))))
            hamt-empty
            (uf-roots uf))])
      (vector extraction-state-tag eg hamt-empty best-nodes))))""",
        "note": "The state must preserve computed costs; replacing them with an empty table breaks extraction analytics.",
    },
    {
        "fn": "extraction-state?",
        "buggy": """(define (extraction-state? x)
  (and (vector? x)
       (>= (vector-length x) 4)))""",
        "note": "Predicate must validate extraction-state-tag, not just vector shape.",
    },
    {
        "fn": "extraction-state?",
        "buggy": """(define (extraction-state? x)
  (and (vector? x)
       (> (vector-length x) 4)
       (eq? (vector-ref x 0) extraction-state-tag)))""",
        "note": "Valid extraction states have length 4; requiring > 4 rejects correct values.",
    },
    {
        "fn": "extract",
        "buggy": """(define (extract state class-id)
  (let* ([best-nodes (state-best-nodes state)]
         [best-node (hamt-lookup class-id best-nodes)])
    (if (not best-node)
        (error 'extract "no best node for class" class-id)
        (let ([op (enode-op best-node)]
              [children (enode-children best-node)])
          (if (zero? (vector-length children))
              op
              (cons op
                    (map (lambda (i)
                           (extract state (vector-ref children i)))
                         (iota (vector-length children)))))))))""",
        "note": "Must canonicalize class-id with egraph-find before best-node lookup.",
    },
    {
        "fn": "extract",
        "buggy": """(define (extract state class-id)
  (let* ([eg (state-egraph state)]
         [root (egraph-find eg class-id)]
         [best-node (hamt-lookup root (state-best-nodes state))])
    (if (not best-node)
        (error 'extract "no best node for class" root)
        (enode-op best-node))))""",
        "note": "Compound nodes must recurse into children; returning only the operator loses structure.",
    },
    {
        "fn": "extract-term",
        "buggy": """(define (extract-term eg term cost-model)
  (egraph-add-term! eg term)
  term)""",
        "note": "extract-term must run extraction, not return the input term directly.",
    },
    {
        "fn": "extract-term",
        "buggy": """(define (extract-term eg term cost-model)
  (let* ([state (make-extraction-state eg cost-model)]
         [class-id (egraph-add-term! eg term)])
    (extract state class-id)))""",
        "note": "State must be built after inserting the term so best-node tables include the new class.",
    },
    {
        "fn": "extract-all",
        "buggy": """(define (extract-all state)
  (let* ([eg (state-egraph state)]
         [uf (egraph-uf eg)])
    (map (lambda (root)
           (extract state root))
         (uf-roots uf))))""",
        "note": "extract-all must return (root . term) pairs, not bare terms.",
    },
    {
        "fn": "extract-all",
        "buggy": """(define (extract-all state)
  (let* ([eg (state-egraph state)]
         [uf (egraph-uf eg)])
    (map (lambda (root)
           (cons root root))
         (uf-roots uf))))""",
        "note": "Pair values must be extracted terms, not repeated class ids.",
    },
    {
        "fn": "optimize",
        "buggy": """(define (optimize term rules cost-model)
  (let ([eg (make-egraph)])
    (let ([root (egraph-add-term! eg term)])
      (let ([state (make-extraction-state eg cost-model)])
        (extract state root)))))""",
        "note": "optimize must run saturation before extraction.",
    },
    {
        "fn": "optimize",
        "buggy": """(define (optimize term rules cost-model)
  (let ([eg (make-egraph)])
    (egraph-add-term! eg term)
    (saturate-simple eg rules)
    term))""",
        "note": "After saturation, optimize must extract the best equivalent term instead of returning input term.",
    },
    {
        "fn": "optimize-with-config",
        "buggy": """(define (optimize-with-config term rules cost-model config)
  (let ([eg (make-egraph)])
    (let ([root (egraph-add-term! eg term)])
      (let ([state (make-extraction-state eg cost-model)])
        (extract state root)))))""",
        "note": "optimize-with-config must actually run saturation with the provided config.",
    },
    {
        "fn": "optimize-with-config",
        "buggy": """(define (optimize-with-config term rules cost-model config)
  (let ([eg (make-egraph)])
    (let ([root (egraph-add-term! eg term)])
      (saturate eg rules config)
      term)))""",
        "note": "Function must return extracted optimum, not the original input term.",
    },
    {
        "fn": "compare-extractions",
        "buggy": """(define (compare-extractions eg class-id cost-models)
  (if (null? cost-models)
      '()
      (let* ([cm (car cost-models)]
             [state (make-extraction-state eg cm)]
             [term (extract state class-id)]
             [cost (class-cost (state-costs state) (egraph-find eg class-id))])
        (list (list cm term cost)))))""",
        "note": "All provided cost models must be compared, not only the first one.",
    },
    {
        "fn": "compare-extractions",
        "buggy": """(define (compare-extractions eg class-id cost-models)
  (map (lambda (cm)
         (let* ([state (make-extraction-state eg cm)]
                [cost (class-cost (state-costs state) (egraph-find eg class-id))])
           (list cm class-id cost)))
       cost-models))""",
        "note": "Each result tuple must include the extracted term, not the numeric class id.",
    },
]

BASE_DIFFICULTY = {
    "make-extraction-state": "hard",
    "extraction-state?": "easy",
    "extract": "hard",
    "extract-term": "medium",
    "extract-all": "medium",
    "optimize": "hard",
    "optimize-with-config": "hard",
    "compare-extractions": "medium",
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
    sid = f"egraph_extract_{family}_{family_counter[family]:03d}"
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
    "make-extraction-state": "state",
    "extraction-state?": "state",
    "extract": "extract",
    "extract-term": "extract",
    "extract-all": "extract",
    "optimize": "convenience",
    "optimize-with-config": "convenience",
    "compare-extractions": "analysis",
}


def make_source_excerpt(fn: str, snippet: str) -> str:
    section = FUNCTION_SECTION[fn]
    indented = "\n".join(f"  {line}" for line in snippet.splitlines())
    return (
        ";;; lattice/egraph/extract.ss excerpt\n"
        "(require 'prelude)\n"
        "(require 'hamt)\n"
        "(require 'egraph/cost)\n"
        "(require 'egraph/saturation)\n"
        "\n"
        "(doc 'module 'egraph/extract)\n"
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
        prompt=f"""Implement this e-graph extraction utility in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "egraph", "extraction", "optimization", "spec-to-code", fn],
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
        tags=["tier1", "egraph", "extraction", "optimization", "skeleton-completion", fn],
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
2. Preserve extraction semantics and edge behavior.
3. Return only one production-ready definition.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "egraph", "extraction", "optimization", "contract-implementation", fn],
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
        tags=["tier1", "egraph", "extraction", "optimization", "python-to-scheme", fn],
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
        tags=["tier1", "egraph", "extraction", "optimization", "chez-to-fold", fn],
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
        tags=["tier1", "egraph", "extraction", "optimization", "source-excerpt-to-fold", "doc-free-target", fn],
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
        tags=["tier1", "egraph", "extraction", "optimization", "bugfix", fn],
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
        tags=["tier1", "egraph", "extraction", "optimization", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # make-extraction-state
    {
        "fn": "make-extraction-state",
        "prompt": "Build a merged class for `x` and `(+ x 0)`, create state, and extract the representative.",
        "gt": "(let* ([eg (make-egraph)] [x-id (egraph-add-term! eg 'x)] [sum-id (egraph-add-term! eg '(+ x 0))] [_m (egraph-merge! eg x-id sum-id)] [_r (egraph-rebuild! eg)] [st (make-extraction-state eg ast-size-cost)]) (extract st sum-id))",
        "verify": "(equal? (let* ([eg (make-egraph)] [x-id (egraph-add-term! eg 'x)] [sum-id (egraph-add-term! eg '(+ x 0))] [_m (egraph-merge! eg x-id sum-id)] [_r (egraph-rebuild! eg)] [st (make-extraction-state eg ast-size-cost)]) (extract st sum-id)) 'x)",
        "difficulty": "hard",
        "tags": ["multi-fn", "equivalence"],
    },
    {
        "fn": "make-extraction-state",
        "prompt": "Create state for `(+ a b)` and check that the root class has a selected best node.",
        "gt": "(let* ([eg (make-egraph)] [id (egraph-add-term! eg '(+ a b))] [st (make-extraction-state eg ast-size-cost)] [root (egraph-find eg id)] [best (hamt-lookup root (state-best-nodes st))]) (and best (eq? (enode-op best) '+)))",
        "verify": "(equal? (let* ([eg (make-egraph)] [id (egraph-add-term! eg '(+ a b))] [st (make-extraction-state eg ast-size-cost)] [root (egraph-find eg id)] [best (hamt-lookup root (state-best-nodes st))]) (and best (eq? (enode-op best) '+))) #t)",
        "difficulty": "medium",
        "tags": ["best-node"],
    },
    {
        "fn": "make-extraction-state",
        "prompt": "Compute state over a leaf and return the class cost from the state.",
        "gt": "(let* ([eg (make-egraph)] [id (egraph-add-term! eg 'x)] [st (make-extraction-state eg ast-size-cost)] [root (egraph-find eg id)]) (class-cost (state-costs st) root))",
        "verify": "(= (let* ([eg (make-egraph)] [id (egraph-add-term! eg 'x)] [st (make-extraction-state eg ast-size-cost)] [root (egraph-find eg id)]) (class-cost (state-costs st) root)) 1)",
        "difficulty": "medium",
        "tags": ["costs"],
    },
    {
        "fn": "make-extraction-state",
        "prompt": "Build extraction state and confirm the state tag check plus non-empty best-node table.",
        "gt": "(let* ([eg (make-egraph)] [_a (egraph-add-term! eg 'a)] [_b (egraph-add-term! eg '(id a))] [st (make-extraction-state eg ast-size-cost)]) (and (extraction-state? st) (not (hamt-empty? (state-best-nodes st)))))",
        "verify": "(equal? (let* ([eg (make-egraph)] [_a (egraph-add-term! eg 'a)] [_b (egraph-add-term! eg '(id a))] [st (make-extraction-state eg ast-size-cost)]) (and (extraction-state? st) (not (hamt-empty? (state-best-nodes st))))) #t)",
        "difficulty": "medium",
        "tags": ["multi-fn", "state-shape"],
    },

    # extraction-state?
    {
        "fn": "extraction-state?",
        "prompt": "Construct a real extraction state and verify `extraction-state?` returns true.",
        "gt": "(let* ([eg (make-egraph)] [_x (egraph-add-term! eg 'x)] [st (make-extraction-state eg ast-size-cost)]) (extraction-state? st))",
        "verify": "(equal? (let* ([eg (make-egraph)] [_x (egraph-add-term! eg 'x)] [st (make-extraction-state eg ast-size-cost)]) (extraction-state? st)) #t)",
        "difficulty": "easy",
        "tags": ["predicate", "multi-fn"],
    },
    {
        "fn": "extraction-state?",
        "prompt": "Check that a wrong-tag vector is rejected by `extraction-state?`.",
        "gt": "(extraction-state? '#(wrong-tag 1 2 3 4))",
        "verify": "(equal? (extraction-state? '#(wrong-tag 1 2 3 4)) #f)",
        "difficulty": "easy",
        "tags": ["negative"],
    },
    {
        "fn": "extraction-state?",
        "prompt": "Count how many values in a mixed list satisfy `extraction-state?`.",
        "gt": "(let* ([eg (make-egraph)] [_x (egraph-add-term! eg 'x)] [st (make-extraction-state eg ast-size-cost)] [vals (list st '#(wrong-tag 1 2 3 4) 42)]) (length (filter extraction-state? vals)))",
        "verify": "(= (let* ([eg (make-egraph)] [_x (egraph-add-term! eg 'x)] [st (make-extraction-state eg ast-size-cost)] [vals (list st '#(wrong-tag 1 2 3 4) 42)]) (length (filter extraction-state? vals))) 1)",
        "difficulty": "medium",
        "tags": ["collection", "multi-fn"],
    },
    {
        "fn": "extraction-state?",
        "prompt": "Use `extraction-state?` to guard extraction and verify the guarded path returns true.",
        "gt": "(let* ([eg (make-egraph)] [id (egraph-add-term! eg 'x)] [st (make-extraction-state eg ast-size-cost)]) (and (extraction-state? st) (equal? (extract st id) 'x)))",
        "verify": "(equal? (let* ([eg (make-egraph)] [id (egraph-add-term! eg 'x)] [st (make-extraction-state eg ast-size-cost)]) (and (extraction-state? st) (equal? (extract st id) 'x))) #t)",
        "difficulty": "medium",
        "tags": ["multi-fn", "guarded-flow"],
    },

    # extract
    {
        "fn": "extract",
        "prompt": "Extract a leaf term from a one-node egraph.",
        "gt": "(let* ([eg (make-egraph)] [id (egraph-add-term! eg 'x)] [st (make-extraction-state eg ast-size-cost)]) (extract st id))",
        "verify": "(equal? (let* ([eg (make-egraph)] [id (egraph-add-term! eg 'x)] [st (make-extraction-state eg ast-size-cost)]) (extract st id)) 'x)",
        "difficulty": "easy",
        "tags": ["leaf", "multi-fn"],
    },
    {
        "fn": "extract",
        "prompt": "Extract from a nested expression and preserve nested structure.",
        "gt": "(let* ([eg (make-egraph)] [id (egraph-add-term! eg '(f (g x)))] [st (make-extraction-state eg ast-size-cost)]) (extract st id))",
        "verify": "(equal? (let* ([eg (make-egraph)] [id (egraph-add-term! eg '(f (g x)))] [st (make-extraction-state eg ast-size-cost)]) (extract st id)) '(f (g x)))",
        "difficulty": "medium",
        "tags": ["nested", "multi-fn"],
    },
    {
        "fn": "extract",
        "prompt": "Merge `a` with `(id a)` and verify extraction chooses `a`.",
        "gt": "(let* ([eg (make-egraph)] [a (egraph-add-term! eg 'a)] [ida (egraph-add-term! eg '(id a))]) (egraph-merge! eg a ida) (egraph-rebuild! eg) (let ([st (make-extraction-state eg ast-size-cost)]) (extract st ida)))",
        "verify": "(equal? (let* ([eg (make-egraph)] [a (egraph-add-term! eg 'a)] [ida (egraph-add-term! eg '(id a))]) (egraph-merge! eg a ida) (egraph-rebuild! eg) (let ([st (make-extraction-state eg ast-size-cost)]) (extract st ida))) 'a)",
        "difficulty": "hard",
        "tags": ["equivalence", "multi-fn"],
    },
    {
        "fn": "extract",
        "prompt": "Extract a shared-subexpression form `(+ x x)` and verify exact structure.",
        "gt": "(let* ([eg (make-egraph)] [id (egraph-add-term! eg '(+ x x))] [st (make-extraction-state eg ast-size-cost)]) (extract st id))",
        "verify": "(equal? (let* ([eg (make-egraph)] [id (egraph-add-term! eg '(+ x x))] [st (make-extraction-state eg ast-size-cost)]) (extract st id)) '(+ x x))",
        "difficulty": "medium",
        "tags": ["shared", "multi-fn"],
    },

    # extract-term
    {
        "fn": "extract-term",
        "prompt": "Run `extract-term` on `(+ a b)` in a fresh egraph.",
        "gt": "(let ([eg (make-egraph)]) (extract-term eg '(+ a b) ast-size-cost))",
        "verify": "(equal? (let ([eg (make-egraph)]) (extract-term eg '(+ a b) ast-size-cost)) '(+ a b))",
        "difficulty": "easy",
        "tags": ["fresh-egraph"],
    },
    {
        "fn": "extract-term",
        "prompt": "Run `extract-term` on nested expression and keep shape stable.",
        "gt": "(let ([eg (make-egraph)]) (extract-term eg '(* (+ x y) z) ast-size-cost))",
        "verify": "(equal? (let ([eg (make-egraph)]) (extract-term eg '(* (+ x y) z) ast-size-cost)) '(* (+ x y) z))",
        "difficulty": "medium",
        "tags": ["nested"],
    },
    {
        "fn": "extract-term",
        "prompt": "Pre-merge `x` with `(+ x 0)` then call `extract-term` on `(+ x 0)` and verify simplification.",
        "gt": "(let* ([eg (make-egraph)] [x (egraph-add-term! eg 'x)] [sum (egraph-add-term! eg '(+ x 0))]) (egraph-merge! eg x sum) (egraph-rebuild! eg) (extract-term eg '(+ x 0) ast-size-cost))",
        "verify": "(equal? (let* ([eg (make-egraph)] [x (egraph-add-term! eg 'x)] [sum (egraph-add-term! eg '(+ x 0))]) (egraph-merge! eg x sum) (egraph-rebuild! eg) (extract-term eg '(+ x 0) ast-size-cost)) 'x)",
        "difficulty": "hard",
        "tags": ["multi-fn", "equivalence"],
    },
    {
        "fn": "extract-term",
        "prompt": "Compare `extract-term` and explicit `extract` on the same term in one egraph.",
        "gt": "(let* ([eg (make-egraph)] [t1 (extract-term eg '(neg x) ast-size-cost)] [id (egraph-add-term! eg '(neg x))] [st (make-extraction-state eg ast-size-cost)] [t2 (extract st id)]) (equal? t1 t2))",
        "verify": "(equal? (let* ([eg (make-egraph)] [t1 (extract-term eg '(neg x) ast-size-cost)] [id (egraph-add-term! eg '(neg x))] [st (make-extraction-state eg ast-size-cost)] [t2 (extract st id)]) (equal? t1 t2)) #t)",
        "difficulty": "medium",
        "tags": ["multi-fn", "consistency"],
    },

    # extract-all
    {
        "fn": "extract-all",
        "prompt": "Extract all roots from egraph containing `a`, `b`, and `(+ a b)` and return count.",
        "gt": "(let ([eg (make-egraph)]) (egraph-add-term! eg 'a) (egraph-add-term! eg 'b) (egraph-add-term! eg '(+ a b)) (let* ([st (make-extraction-state eg ast-size-cost)] [all (extract-all st)]) (length all)))",
        "verify": "(= (let ([eg (make-egraph)]) (egraph-add-term! eg 'a) (egraph-add-term! eg 'b) (egraph-add-term! eg '(+ a b)) (let* ([st (make-extraction-state eg ast-size-cost)] [all (extract-all st)]) (length all))) 3)",
        "difficulty": "medium",
        "tags": ["cardinality", "multi-fn"],
    },
    {
        "fn": "extract-all",
        "prompt": "Find the pair for class of `x` inside `extract-all` output.",
        "gt": "(let ([eg (make-egraph)]) (let ([id (egraph-add-term! eg 'x)]) (let* ([st (make-extraction-state eg ast-size-cost)] [all (extract-all st)] [entry (find (lambda (p) (= (car p) id)) all)]) (and entry (equal? (cdr entry) 'x)))))",
        "verify": "(equal? (let ([eg (make-egraph)]) (let ([id (egraph-add-term! eg 'x)]) (let* ([st (make-extraction-state eg ast-size-cost)] [all (extract-all st)] [entry (find (lambda (p) (= (car p) id)) all)]) (and entry (equal? (cdr entry) 'x))))) #t)",
        "difficulty": "medium",
        "tags": ["lookup", "multi-fn"],
    },
    {
        "fn": "extract-all",
        "prompt": "Merge `a` and `(id a)`, then ensure `extract-all` has one root entry.",
        "gt": "(let ([eg (make-egraph)]) (let ([a (egraph-add-term! eg 'a)] [ida (egraph-add-term! eg '(id a))]) (egraph-merge! eg a ida) (egraph-rebuild! eg) (let* ([st (make-extraction-state eg ast-size-cost)] [all (extract-all st)]) (= (length all) 1))))",
        "verify": "(equal? (let ([eg (make-egraph)]) (let ([a (egraph-add-term! eg 'a)] [ida (egraph-add-term! eg '(id a))]) (egraph-merge! eg a ida) (egraph-rebuild! eg) (let* ([st (make-extraction-state eg ast-size-cost)] [all (extract-all st)]) (= (length all) 1)))) #t)",
        "difficulty": "hard",
        "tags": ["equivalence", "multi-fn"],
    },
    {
        "fn": "extract-all",
        "prompt": "Collect extracted terms from `extract-all` and verify expected symbol membership.",
        "gt": "(let ([eg (make-egraph)]) (egraph-add-term! eg 'x) (egraph-add-term! eg 'y) (let* ([st (make-extraction-state eg ast-size-cost)] [terms (map cdr (extract-all st))]) (and (not (not (member 'x terms))) (not (not (member 'y terms))))))",
        "verify": "(equal? (let ([eg (make-egraph)]) (egraph-add-term! eg 'x) (egraph-add-term! eg 'y) (let* ([st (make-extraction-state eg ast-size-cost)] [terms (map cdr (extract-all st))]) (and (not (not (member 'x terms))) (not (not (member 'y terms)))))) #t)",
        "difficulty": "medium",
        "tags": ["projection", "multi-fn"],
    },

    # optimize
    {
        "fn": "optimize",
        "prompt": "Optimize `(+ x 0)` with identity rules.",
        "gt": "(optimize '(+ x 0) arith-identity-rules ast-size-cost)",
        "verify": "(equal? (optimize '(+ x 0) arith-identity-rules ast-size-cost) 'x)",
        "difficulty": "medium",
        "tags": ["identity"],
    },
    {
        "fn": "optimize",
        "prompt": "Optimize nested identity `(+ (* x 1) 0)`.",
        "gt": "(optimize '(+ (* x 1) 0) arith-identity-rules ast-size-cost)",
        "verify": "(equal? (optimize '(+ (* x 1) 0) arith-identity-rules ast-size-cost) 'x)",
        "difficulty": "hard",
        "tags": ["nested"],
    },
    {
        "fn": "optimize",
        "prompt": "Optimize `(+ a b)` with commutativity and verify one of canonical equivalents.",
        "gt": "(optimize '(+ a b) arith-comm-rules ast-size-cost)",
        "verify": "(let ([r (optimize '(+ a b) arith-comm-rules ast-size-cost)]) (or (equal? r '(+ a b)) (equal? r '(+ b a))))",
        "difficulty": "hard",
        "tags": ["commutativity"],
    },
    {
        "fn": "optimize",
        "prompt": "Optimize `(+ (+ x 0) (* y 1))` with basic arithmetic rules.",
        "gt": "(optimize '(+ (+ x 0) (* y 1)) basic-arith-rules ast-size-cost)",
        "verify": "(let ([r (optimize '(+ (+ x 0) (* y 1)) basic-arith-rules ast-size-cost)]) (or (equal? r '(+ x y)) (equal? r '(+ y x))))",
        "difficulty": "hard",
        "tags": ["basic-rules"],
    },

    # optimize-with-config
    {
        "fn": "optimize-with-config",
        "prompt": "Optimize `(+ x 0)` using explicit saturation config.",
        "gt": "(let ([cfg (make-saturation-config 1 10000 0)]) (optimize-with-config '(+ x 0) arith-identity-rules ast-size-cost cfg))",
        "verify": "(equal? (let ([cfg (make-saturation-config 1 10000 0)]) (optimize-with-config '(+ x 0) arith-identity-rules ast-size-cost cfg)) 'x)",
        "difficulty": "medium",
        "tags": ["config"],
    },
    {
        "fn": "optimize-with-config",
        "prompt": "Use config-driven optimization on `(* x 1)` and verify simplification.",
        "gt": "(let ([cfg (make-saturation-config 2 10000 0)]) (optimize-with-config '(* x 1) arith-identity-rules ast-size-cost cfg))",
        "verify": "(equal? (let ([cfg (make-saturation-config 2 10000 0)]) (optimize-with-config '(* x 1) arith-identity-rules ast-size-cost cfg)) 'x)",
        "difficulty": "medium",
        "tags": ["identity"],
    },
    {
        "fn": "optimize-with-config",
        "prompt": "Optimize `(+ (+ x 0) (* y 1))` with config and verify canonical simplified form.",
        "gt": "(let ([cfg (make-saturation-config 5 10000 0)]) (optimize-with-config '(+ (+ x 0) (* y 1)) basic-arith-rules ast-size-cost cfg))",
        "verify": "(let ([r (let ([cfg (make-saturation-config 5 10000 0)]) (optimize-with-config '(+ (+ x 0) (* y 1)) basic-arith-rules ast-size-cost cfg))]) (or (equal? r '(+ x y)) (equal? r '(+ y x))))",
        "difficulty": "hard",
        "tags": ["multi-rule"],
    },
    {
        "fn": "optimize-with-config",
        "prompt": "Run commutativity optimization with config and verify result remains in the equivalence set.",
        "gt": "(let ([cfg (make-saturation-config 1 10000 0)]) (optimize-with-config '(+ a b) arith-comm-rules ast-size-cost cfg))",
        "verify": "(let ([r (let ([cfg (make-saturation-config 1 10000 0)]) (optimize-with-config '(+ a b) arith-comm-rules ast-size-cost cfg))]) (or (equal? r '(+ a b)) (equal? r '(+ b a))))",
        "difficulty": "hard",
        "tags": ["commutativity"],
    },

    # compare-extractions
    {
        "fn": "compare-extractions",
        "prompt": "Compare two cost models for a single class and return number of result tuples.",
        "gt": "(let ([eg (make-egraph)]) (let ([id (egraph-add-term! eg '(+ x y))]) (length (compare-extractions eg id (list ast-size-cost ast-depth-cost)))))",
        "verify": "(= (let ([eg (make-egraph)]) (let ([id (egraph-add-term! eg '(+ x y))]) (length (compare-extractions eg id (list ast-size-cost ast-depth-cost))))) 2)",
        "difficulty": "medium",
        "tags": ["cardinality", "multi-fn"],
    },
    {
        "fn": "compare-extractions",
        "prompt": "Compare model outputs and verify both extracted terms equal the only available term.",
        "gt": "(let ([eg (make-egraph)]) (let* ([id (egraph-add-term! eg '(+ x y))] [results (compare-extractions eg id (list ast-size-cost ast-depth-cost))]) (and (equal? (cadr (car results)) '(+ x y)) (equal? (cadr (cadr results)) '(+ x y)))))",
        "verify": "(equal? (let ([eg (make-egraph)]) (let* ([id (egraph-add-term! eg '(+ x y))] [results (compare-extractions eg id (list ast-size-cost ast-depth-cost))]) (and (equal? (cadr (car results)) '(+ x y)) (equal? (cadr (cadr results)) '(+ x y))))) #t)",
        "difficulty": "medium",
        "tags": ["term-equality"],
    },
    {
        "fn": "compare-extractions",
        "prompt": "Verify compare-extractions returns triples with model, term, and numeric cost fields.",
        "gt": "(let ([eg (make-egraph)]) (let* ([id (egraph-add-term! eg '(+ x y))] [results (compare-extractions eg id (list ast-size-cost ast-depth-cost))] [r1 (car results)] [r2 (cadr results)]) (and (cost-model? (car r1)) (number? (caddr r1)) (cost-model? (car r2)) (number? (caddr r2)))))",
        "verify": "(equal? (let ([eg (make-egraph)]) (let* ([id (egraph-add-term! eg '(+ x y))] [results (compare-extractions eg id (list ast-size-cost ast-depth-cost))] [r1 (car results)] [r2 (cadr results)]) (and (cost-model? (car r1)) (number? (caddr r1)) (cost-model? (car r2)) (number? (caddr r2))))) #t)",
        "difficulty": "medium",
        "tags": ["shape-check"],
    },
    {
        "fn": "compare-extractions",
        "prompt": "Return whether model names in compare-extractions are ast-size then ast-depth for the given list order.",
        "gt": "(let ([eg (make-egraph)]) (let* ([id (egraph-add-term! eg '(+ x y))] [results (compare-extractions eg id (list ast-size-cost ast-depth-cost))]) (and (eq? (cost-model-name (car (car results))) 'ast-size) (eq? (cost-model-name (car (cadr results))) 'ast-depth))))",
        "verify": "(equal? (let ([eg (make-egraph)]) (let* ([id (egraph-add-term! eg '(+ x y))] [results (compare-extractions eg id (list ast-size-cost ast-depth-cost))]) (and (eq? (cost-model-name (car (car results))) 'ast-size) (eq? (cost-model-name (car (cadr results))) 'ast-depth)))) #t)",
        "difficulty": "medium",
        "tags": ["ordering"],
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
