#!/usr/bin/env python3
"""Generate SFT samples for KG usage functions in lattice/meta/search.ss."""

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

SOURCE_MODULE = "lattice/meta/search.ss"
SOURCE_TEST = "lattice/meta/test-meta.ss"

GLOBAL_DEFS = [
    """(define (fold-left f init lst)
  (let loop ([acc init] [xs lst])
    (if (null? xs)
        acc
        (loop (f acc (car xs)) (cdr xs)))))""",
    """(define (filter pred lst)
  (let loop ([xs lst] [acc '()])
    (cond
      [(null? xs) (reverse acc)]
      [(pred (car xs)) (loop (cdr xs) (cons (car xs) acc))]
      [else (loop (cdr xs) acc)])))""",
    """(define (filter-map f lst)
  (let loop ([xs lst] [acc '()])
    (if (null? xs)
        (reverse acc)
        (let ([v (f (car xs))])
          (loop (cdr xs) (if v (cons v acc) acc))))))""",
    """(define (take-at-most n lst)
  (if (or (<= n 0) (null? lst))
      '()
      (cons (car lst) (take-at-most (- n 1) (cdr lst)))))""",
    """(define (insert-sorted cmp x lst)
  (cond
    [(null? lst) (list x)]
    [(cmp x (car lst)) (cons x lst)]
    [else (cons (car lst) (insert-sorted cmp x (cdr lst)))]))""",
    """(define (sort-by cmp lst)
  (fold-left (lambda (acc x) (insert-sorted cmp x acc)) '() lst))""",
    """(define hamt-empty '())""",
    """(define (hamt-lookup key m)
  (let ([p (assq key m)])
    (if p (cdr p) #f)))""",
    """(define (hamt-assoc key value m)
  (let loop ([xs m] [acc '()])
    (cond
      [(null? xs) (reverse (cons (cons key value) acc))]
      [(eq? (caar xs) key)
       (append (reverse acc) (cons (cons key value) (cdr xs)))]
      [else (loop (cdr xs) (cons (car xs) acc))])))""",
    """(define (hamt-size m)
  (length m))""",
    """(define (ensure-indexed!) 'ok)""",
    """(define CONCEPT-BOOST 0.25)""",
    """(define CONCEPT-BOOST-CAP 2.0)""",
    """(define *kg-skills-list* '(linalg meta query crypto graphics))""",
    """(define *kg-skill-data*
  '((linalg . ((name . linalg)
               (tier . 0)
               (purity . total)
               (exports . ((vec vec-add vec-sub vec-dot)
                           (matrix matrix-mul matrix-det)))))
    (meta . ((name . meta)
             (tier . 1)
             (purity . partial)
             (exports . ((search lattice-find lattice-find-prefix lattice-find-substring)
                         (kg kg-build! kg-skills)))))
    (query . ((name . query)
              (tier . 1)
              (purity . total)
              (exports . ((query-dsl query-run query-plan)
                          (sql sql-parse sql-run)))))
    (crypto . ((name . crypto)
               (tier . 1)
               (purity . total)
               (exports . (sha256 hmac hash-verify))))
    (graphics . ((name . graphics)
                 (tier . 2)
                 (purity . partial)
                 (exports . ((render draw-line draw-circle)))))))""",
    """(define *kg-exports-list*
  '((vec-add . #t)
    (vec-sub . #t)
    (vec-dot . #t)
    (lattice-find . #t)
    (lattice-find-prefix . #t)
    (lattice-find-substring . #t)
    (kg-build! . #t)
    (kg-skills . #t)
    (query-run . #t)
    (query-plan . #t)
    (sql-run . #t)
    (sha256 . #t)
    (hmac . #t)
    (draw-line . #t)))""",
    """(define *export-module-map*
  '((vec-add . vec)
    (vec-sub . vec)
    (vec-dot . vec)
    (lattice-find . search)
    (lattice-find-prefix . search)
    (lattice-find-substring . search)
    (kg-build! . kg)
    (kg-skills . kg)
    (query-run . query-dsl)
    (query-plan . query-dsl)
    (sql-run . sql)
    (sha256 . sha256)
    (hmac . sha256)
    (draw-line . render)))""",
    """(define *module-skill-map*
  '((vec . linalg)
    (matrix . linalg)
    (search . meta)
    (kg . meta)
    (query-dsl . query)
    (sql . query)
    (sha256 . crypto)
    (render . graphics)))""",
    """(define *kg-concept-skill-map*
  '((optimization . (linalg query))
    (search . (meta query))
    (crypto . (crypto))
    (rendering . (graphics))
    (graph . (meta query graphics))))""",
    """(define *mock-search-results*
  '((vector . ((linalg 1.20 skill ((tier . 0) (purity . total)))
                (vec-add 0.92 export ((module . vec)))
                (vec-sub 0.89 export ((module . vec)))
                (graphics 0.31 skill ((tier . 2) (purity . partial)))))
    (search . ((meta 1.05 skill ((tier . 1) (purity . partial)))
               (query 0.96 skill ((tier . 1) (purity . total)))
               (lattice-find 0.88 export ((module . search)))
               (query-run 0.77 export ((module . query-dsl)))))
    (crypto . ((crypto 1.07 skill ((tier . 1) (purity . total)))
               (sha256 0.95 export ((module . sha256)))
               (hmac 0.82 export ((module . sha256)))))))""",
    """(define (kg-skills)
  *kg-skills-list*)""",
    """(define (kg-skill-data skill-name)
  (let ([entry (assq skill-name *kg-skill-data*)])
    (if entry (cdr entry) #f)))""",
    """(define (kg-exports)
  *kg-exports-list*)""",
    """(define (kg-concept-skills concept-name)
  (let ([entry (assq concept-name *kg-concept-skill-map*)])
    (if entry (cdr entry) '())))""",
    """(define (lattice-find query . options)
  (let* ([k (if (and (pair? options) (number? (car options)))
                (car options)
                10)]
         [type (if (and (pair? options) (pair? (cdr options)))
                   (cadr options)
                   'all)]
         [key (string->symbol (string-downcase query))]
         [entry (assq key *mock-search-results*)]
         [base (if entry (cdr entry) '())]
         [typed (case type
                  [(skill skills) (filter (lambda (r) (eq? (caddr r) 'skill)) base)]
                  [(module modules) (filter (lambda (r) (eq? (caddr r) 'module)) base)]
                  [(export exports) (filter (lambda (r) (eq? (caddr r) 'export)) base)]
                  [else base])])
    (take-at-most k typed)))""",
    """(define (string-contains? haystack needle)
  (let ([h-len (string-length haystack)]
        [n-len (string-length needle)])
    (if (> n-len h-len)
        #f
        (let loop ([i 0])
          (cond
            [(> (+ i n-len) h-len) #f]
            [(string=? (substring haystack i (+ i n-len)) needle) #t]
            [else (loop (+ i 1))])))))""",
]

DEFS: Dict[str, str] = {
    "concept-boost-for-query": """(define (concept-boost-for-query query-terms)
  (fold-left
   (lambda (boost-set term)
     (let ([skills (kg-concept-skills term)])
       (fold-left
        (lambda (bs skill-name)
          (let ([current (or (hamt-lookup skill-name bs) 0)])
            (hamt-assoc skill-name (+ current 1) bs)))
        boost-set
        skills)))
   hamt-empty
   query-terms))""",
    "result->skill-name": """(define (result->skill-name result)
  (let ([id (car result)]
        [type (caddr result)]
        [data (cadddr result)])
    (case type
      [(skill) id]
      [(module)
       (and data
            (let ([skill (assq 'skill data)])
              (and skill (cdr skill))))]
      [(export)
       (and data
            (let ([mod (assq 'module data)])
              (and mod (hamt-lookup (cdr mod) *module-skill-map*))))]
      [else #f])))""",
    "apply-concept-boosts": """(define (apply-concept-boosts results boost-map)
  (if (zero? (hamt-size boost-map))
      results
      (map (lambda (result)
             (let* ([skill (result->skill-name result)]
                    [count (and skill (hamt-lookup skill boost-map))])
               (if count
                   (let* ([multiplier (min CONCEPT-BOOST-CAP
                                           (+ 1.0 (* CONCEPT-BOOST count)))]
                          [old-score (cadr result)]
                          [new-score (* old-score multiplier)])
                     (list (car result) new-score (caddr result) (cadddr result)))
                   result)))
           results)))""",
    "lattice-export-source": """(define (lattice-export-source sym)
  (let loop ([skills (kg-skills)])
    (if (null? skills) #f
        (let* ([skill-name (car skills)]
               [data (kg-skill-data skill-name)]
               [exports-raw (if data
                                (let ([e (assq 'exports data)])
                                  (if e (cdr e) '()))
                                '())])
          (cond
           [(not (list? exports-raw)) (loop (cdr skills))]
           [(null? exports-raw) (loop (cdr skills))]
           [(symbol? (car exports-raw))
            (if (memq sym exports-raw)
                skill-name
                (loop (cdr skills)))]
           [else
            (let group-loop ([groups exports-raw])
              (if (null? groups)
                  (loop (cdr skills))
                  (let ([group (car groups)])
                    (if (and (pair? group) (memq sym group))
                        skill-name
                        (group-loop (cdr groups))))))])))))""",
    "lattice-find-prefix": """(define (lattice-find-prefix prefix-sym . options)
  (ensure-indexed!)
  (let* ([k (if (pair? options) (car options) 20)]
         [prefix-str (string-downcase (symbol->string prefix-sym))]
         [prefix-len (string-length prefix-str)])
    (let* ([export-matches
            (filter-map
             (lambda (export-entry)
               (let* ([export-name (car export-entry)]
                      [name-str (string-downcase (symbol->string export-name))])
                 (if (and (>= (string-length name-str) prefix-len)
                          (string=? (substring name-str 0 prefix-len) prefix-str))
                     (let ([mod (hamt-lookup export-name *export-module-map*)])
                       (list export-name 0.9 'export
                             `((name . ,export-name)
                               ,@(if mod `((module . ,mod)) '()))))
                     #f)))
             (kg-exports))]
           [skill-matches
            (filter-map
             (lambda (skill-name)
               (let ([name-str (string-downcase (symbol->string skill-name))])
                 (if (and (>= (string-length name-str) prefix-len)
                          (string=? (substring name-str 0 prefix-len) prefix-str))
                     (list skill-name 0.95 'skill (kg-skill-data skill-name))
                     #f)))
             (kg-skills))]
           [all-matches (append skill-matches export-matches)]
           [sorted (sort-by (lambda (a b) (> (cadr a) (cadr b))) all-matches)])
      (take-at-most k sorted))))""",
    "lattice-find-substring": """(define (lattice-find-substring substr-sym . options)
  (ensure-indexed!)
  (let* ([k (if (pair? options) (car options) 20)]
         [substr-str (string-downcase (symbol->string substr-sym))])
    (let* ([export-matches
            (filter-map
             (lambda (export-entry)
               (let* ([export-name (car export-entry)]
                      [name-str (string-downcase (symbol->string export-name))])
                 (if (string-contains? name-str substr-str)
                     (let ([mod (hamt-lookup export-name *export-module-map*)])
                       (list export-name 0.8 'export
                             `((name . ,export-name)
                               ,@(if mod `((module . ,mod)) '()))))
                     #f)))
             (kg-exports))]
           [skill-matches
            (filter-map
             (lambda (skill-name)
               (let ([name-str (string-downcase (symbol->string skill-name))])
                 (if (string-contains? name-str substr-str)
                     (list skill-name 0.85 'skill (kg-skill-data skill-name))
                     #f)))
             (kg-skills))]
           [all-matches (append skill-matches export-matches)]
           [sorted (sort-by (lambda (a b) (> (cadr a) (cadr b))) all-matches)])
      (take-at-most k sorted))))""",
    "lattice-find-by-tier": """(define (lattice-find-by-tier query tier . options)
  (let* ([k (if (pair? options) (car options) 10)]
         [results (lattice-find query k 'skill)])
    (filter
     (lambda (result)
       (let ([data (cadddr result)])
         (and data
              (let ([t (assq 'tier data)])
                (and t (= (cdr t) tier))))))
     results)))""",
    "lattice-find-by-purity": """(define (lattice-find-by-purity query purity . options)
  (let* ([k (if (pair? options) (car options) 10)]
         [results (lattice-find query k 'skill)])
    (filter
     (lambda (result)
       (let ([data (cadddr result)])
         (and data
              (let ([p (assq 'purity data)])
                (and p (eq? (cdr p) purity))))))
     results)))""",
}

DEPENDS: Dict[str, List[str]] = {
    "concept-boost-for-query": [],
    "result->skill-name": [],
    "apply-concept-boosts": ["result->skill-name"],
    "lattice-export-source": [],
    "lattice-find-prefix": [],
    "lattice-find-substring": [],
    "lattice-find-by-tier": [],
    "lattice-find-by-purity": [],
}

FUNCTION_ORDER = [
    "concept-boost-for-query",
    "result->skill-name",
    "apply-concept-boosts",
    "lattice-export-source",
    "lattice-find-prefix",
    "lattice-find-substring",
    "lattice-find-by-tier",
    "lattice-find-by-purity",
]

FUNCTION_SPECS = {
    "concept-boost-for-query": "Build a skill->count map from query concept terms; increment count for each matched concept-skill relation.",
    "result->skill-name": "Resolve the parent skill for a search result tuple based on result type and attached metadata.",
    "apply-concept-boosts": "Apply multiplicative score boosts to search results based on concept-match counts with a cap.",
    "lattice-export-source": "Find which skill exports a symbol, handling both flat and grouped manifest export layouts.",
    "lattice-find-prefix": "Return skill/export matches whose names start with a prefix, ranked by score and truncated by k.",
    "lattice-find-substring": "Return skill/export matches whose names contain a substring, ranked by score and truncated by k.",
    "lattice-find-by-tier": "Filter skill search results to a specific numeric tier.",
    "lattice-find-by-purity": "Filter skill search results to a specific purity tag (e.g., total/partial).",
}

SKELETONS = {
    "concept-boost-for-query": """(define (concept-boost-for-query query-terms)
  ;; TODO: accumulate concept matches into skill score boosts
  <TODO>)""",
    "result->skill-name": """(define (result->skill-name result)
  ;; TODO: decode skill identity from skill/module/export result tuple
  <TODO>)""",
    "apply-concept-boosts": """(define (apply-concept-boosts results boost-map)
  ;; TODO: scale scores by concept counts with cap and preserve structure
  <TODO>)""",
    "lattice-export-source": """(define (lattice-export-source sym)
  ;; TODO: walk skill manifests and locate which skill exports sym
  <TODO>)""",
    "lattice-find-prefix": """(define (lattice-find-prefix prefix-sym . options)
  ;; TODO: prefix match across skills and exports, then sort by score
  <TODO>)""",
    "lattice-find-substring": """(define (lattice-find-substring substr-sym . options)
  ;; TODO: substring match across skills and exports, then sort by score
  <TODO>)""",
    "lattice-find-by-tier": """(define (lattice-find-by-tier query tier . options)
  ;; TODO: filter skill search results by tier
  <TODO>)""",
    "lattice-find-by-purity": """(define (lattice-find-by-purity query purity . options)
  ;; TODO: filter skill search results by purity symbol
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "concept-boost-for-query": """(and
  (let ([m (concept-boost-for-query '(search optimization search))])
    (and (= (hamt-lookup 'meta m) 2)
         (= (hamt-lookup 'query m) 3)
         (= (hamt-lookup 'linalg m) 1)
         (not (hamt-lookup 'crypto m))))
  (= (hamt-size (concept-boost-for-query '())) 0)
  (let ([m (concept-boost-for-query '(graph graph graph))])
    (= (hamt-lookup 'graphics m) 3)))""",
    "result->skill-name": """(and
  (eq? (result->skill-name '(meta 1.0 skill ((tier . 1)))) 'meta)
  (eq? (result->skill-name '(meta/search 1.0 module ((skill . meta)))) 'meta)
  (eq? (result->skill-name '(lattice-find 0.8 export ((module . search)))) 'meta)
  (eq? (result->skill-name '(sha256 0.9 export ((module . sha256)))) 'crypto)
  (eq? (result->skill-name '(unknown 0.1 export ((module . unknown)))) #f))""",
    "apply-concept-boosts": """(and
  (equal?
   (apply-concept-boosts
    (list '(meta 1.0 skill ((tier . 1)))
          '(query 0.9 skill ((tier . 1))))
    hamt-empty)
   (list '(meta 1.0 skill ((tier . 1)))
         '(query 0.9 skill ((tier . 1)))))
  (let* ([out (apply-concept-boosts
               (list '(meta 1.0 skill ((tier . 1)))
                     '(lattice-find 0.8 export ((module . search)))
                     '(sha256 0.7 export ((module . sha256))))
               '((meta . 2)))])
    (and (< (abs (- (cadr (list-ref out 0)) 1.5)) 0.00001)
         (< (abs (- (cadr (list-ref out 1)) 1.2)) 0.00001)
         (< (abs (- (cadr (list-ref out 2)) 0.7)) 0.00001)))
  (let* ([out (apply-concept-boosts
               (list '(meta 1.0 skill ((tier . 1))))
               '((meta . 99)))])
    (< (abs (- (cadr (car out)) 2.0)) 0.00001)))""",
    "lattice-export-source": """(and
  (eq? (lattice-export-source 'lattice-find) 'meta)
  (eq? (lattice-export-source 'sha256) 'crypto)
  (eq? (lattice-export-source 'sql-run) 'query)
  (eq? (lattice-export-source 'missing-export) #f))""",
    "lattice-find-prefix": """(and
  (equal? (map car (lattice-find-prefix 'vec 3)) '(vec-add vec-sub vec-dot))
  (equal? (map car (lattice-find-prefix 'q 2)) '(query query-run))
  (equal? (map car (lattice-find-prefix 'm 5)) '(meta))
  (= (length (lattice-find-prefix 'z 10)) 0))""",
    "lattice-find-substring": """(and
  (let ([ids (map car (lattice-find-substring 'find 10))])
    (and (memq 'lattice-find ids)
         (memq 'lattice-find-prefix ids)
         (memq 'lattice-find-substring ids)))
  (eq? (car (car (lattice-find-substring 'query 3))) 'query)
  (= (length (lattice-find-substring 'run 1)) 1)
  (= (length (lattice-find-substring 'xyz 5)) 0))""",
    "lattice-find-by-tier": """(and
  (equal? (map car (lattice-find-by-tier "search" 1 10)) '(meta query))
  (equal? (map car (lattice-find-by-tier "vector" 0 10)) '(linalg))
  (equal? (map car (lattice-find-by-tier "vector" 2 10)) '(graphics))
  (equal? (map car (lattice-find-by-tier "crypto" 1 1)) '(crypto)))""",
    "lattice-find-by-purity": """(and
  (equal? (map car (lattice-find-by-purity "search" 'partial 10)) '(meta))
  (equal? (map car (lattice-find-by-purity "search" 'total 10)) '(query))
  (equal? (map car (lattice-find-by-purity "vector" 'total 10)) '(linalg))
  (equal? (map car (lattice-find-by-purity "vector" 'partial 10)) '(graphics)))""",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")
TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "concept-boost-for-query": """def concept_boost_for_query(query_terms):
    boost_set = {}
    for term in query_terms:
        for skill_name in kg_concept_skills(term):
            current = boost_set.get(skill_name, 0)
            boost_set[skill_name] = current + 1
    return boost_set""",
    "result->skill-name": """def result_to_skill_name(result):
    _id, _score, kind, data = result
    if kind == "skill":
        return _id
    if kind == "module":
        return data.get("skill") if data else None
    if kind == "export":
        module = data.get("module") if data else None
        return module_skill_map.get(module) if module else None
    return None""",
    "apply-concept-boosts": """def apply_concept_boosts(results, boost_map):
    if len(boost_map) == 0:
        return results
    out = []
    for _id, score, kind, data in results:
        skill = result_to_skill_name((_id, score, kind, data))
        count = boost_map.get(skill) if skill else None
        if count:
            mult = min(CONCEPT_BOOST_CAP, 1.0 + CONCEPT_BOOST * count)
            out.append((_id, score * mult, kind, data))
        else:
            out.append((_id, score, kind, data))
    return out""",
    "lattice-export-source": """def lattice_export_source(sym):
    for skill_name in kg_skills():
        data = kg_skill_data(skill_name)
        exports_raw = data.get("exports", []) if data else []
        if not isinstance(exports_raw, list) or len(exports_raw) == 0:
            continue
        if isinstance(exports_raw[0], str):
            if sym in exports_raw:
                return skill_name
        else:
            for group in exports_raw:
                if isinstance(group, list) and sym in group:
                    return skill_name
    return None""",
    "lattice-find-prefix": """def lattice_find_prefix(prefix_sym, k=20):
    prefix = str(prefix_sym).lower()
    export_matches = []
    for export_name, _ in kg_exports():
        name = str(export_name).lower()
        if len(name) >= len(prefix) and name[: len(prefix)] == prefix:
            row = [export_name, 0.9, "export", {"name": export_name}]
            mod = export_module_map.get(export_name)
            if mod:
                row[3]["module"] = mod
            export_matches.append(row)
    skill_matches = []
    for skill_name in kg_skills():
        name = str(skill_name).lower()
        if len(name) >= len(prefix) and name[: len(prefix)] == prefix:
            skill_matches.append([skill_name, 0.95, "skill", kg_skill_data(skill_name)])
    merged = sorted(skill_matches + export_matches, key=lambda r: -r[1])
    return merged[:k]""",
    "lattice-find-substring": """def lattice_find_substring(substr_sym, k=20):
    needle = str(substr_sym).lower()
    export_matches = []
    for export_name, _ in kg_exports():
        name = str(export_name).lower()
        if needle in name:
            row = [export_name, 0.8, "export", {"name": export_name}]
            mod = export_module_map.get(export_name)
            if mod:
                row[3]["module"] = mod
            export_matches.append(row)
    skill_matches = []
    for skill_name in kg_skills():
        name = str(skill_name).lower()
        if needle in name:
            skill_matches.append([skill_name, 0.85, "skill", kg_skill_data(skill_name)])
    merged = sorted(skill_matches + export_matches, key=lambda r: -r[1])
    return merged[:k]""",
    "lattice-find-by-tier": """def lattice_find_by_tier(query, tier, k=10):
    rows = lattice_find(query, k, "skill")
    out = []
    for row in rows:
        data = row[3]
        if data is not None and data.get("tier") == tier:
            out.append(row)
    return out""",
    "lattice-find-by-purity": """def lattice_find_by_purity(query, purity, k=10):
    rows = lattice_find(query, k, "skill")
    out = []
    for row in rows:
        data = row[3]
        if data is not None and data.get("purity") == purity:
            out.append(row)
    return out""",
}

CHEZ_SNIPPETS = {
    "concept-boost-for-query": """(define (concept-boost query-terms)
  (fold-left
   (lambda (boost-set term)
     (let ((skills (kg-concept-skills term)))
       (fold-left
        (lambda (bs skill-name)
          (let ((current (or (hamt-lookup skill-name bs) 0)))
            (hamt-assoc skill-name (+ current 1) bs)))
        boost-set
        skills)))
   hamt-empty
   query-terms))""",
    "result->skill-name": """(define (result->skill result)
  (let ((id (car result))
        (kind (caddr result))
        (data (cadddr result)))
    (case kind
      ((skill) id)
      ((module)
       (and data
            (let ((s (assq 'skill data)))
              (and s (cdr s)))))
      ((export)
       (and data
            (let ((m (assq 'module data)))
              (and m (hamt-lookup (cdr m) *module-skill-map*)))))
      (else #f))))""",
    "apply-concept-boosts": """(define (boost-results results boost-map)
  (if (zero? (hamt-size boost-map))
      results
      (map (lambda (result)
             (let* ((skill (result->skill-name result))
                    (count (and skill (hamt-lookup skill boost-map))))
               (if count
                   (let* ((mult (min CONCEPT-BOOST-CAP (+ 1.0 (* CONCEPT-BOOST count))))
                          (old (cadr result)))
                     (list (car result) (* old mult) (caddr result) (cadddr result)))
                   result)))
           results)))""",
    "lattice-export-source": """(define (export-source sym)
  (let loop ((skills (kg-skills)))
    (if (null? skills)
        #f
        (let* ((skill-name (car skills))
               (data (kg-skill-data skill-name))
               (exports-raw (if data
                                (let ((e (assq 'exports data)))
                                  (if e (cdr e) '()))
                                '())))
          (cond
            ((not (list? exports-raw)) (loop (cdr skills)))
            ((null? exports-raw) (loop (cdr skills)))
            ((symbol? (car exports-raw))
             (if (memq sym exports-raw) skill-name (loop (cdr skills))))
            (else
             (let group-loop ((groups exports-raw))
               (if (null? groups)
                   (loop (cdr skills))
                   (let ((group (car groups)))
                     (if (and (pair? group) (memq sym group))
                         skill-name
                         (group-loop (cdr groups))))))))))))""",
    "lattice-find-prefix": """(define (find-prefix prefix-sym . options)
  (ensure-indexed!)
  (let* ((k (if (pair? options) (car options) 20))
         (prefix-str (string-downcase (symbol->string prefix-sym)))
         (prefix-len (string-length prefix-str)))
    (let* ((export-matches
            (filter-map
             (lambda (entry)
               (let* ((name (car entry))
                      (name-str (string-downcase (symbol->string name))))
                 (if (and (>= (string-length name-str) prefix-len)
                          (string=? (substring name-str 0 prefix-len) prefix-str))
                     (let ((mod (hamt-lookup name *export-module-map*)))
                       (list name 0.9 'export `((name . ,name) ,@(if mod `((module . ,mod)) '()))))
                     #f)))
             (kg-exports)))
           (skill-matches
            (filter-map
             (lambda (skill)
               (let ((name-str (string-downcase (symbol->string skill))))
                 (if (and (>= (string-length name-str) prefix-len)
                          (string=? (substring name-str 0 prefix-len) prefix-str))
                     (list skill 0.95 'skill (kg-skill-data skill))
                     #f)))
             (kg-skills)))
           (sorted (sort-by (lambda (a b) (> (cadr a) (cadr b)))
                            (append skill-matches export-matches))))
      (take-at-most k sorted))))""",
    "lattice-find-substring": """(define (find-substr substr-sym . options)
  (ensure-indexed!)
  (let* ((k (if (pair? options) (car options) 20))
         (needle (string-downcase (symbol->string substr-sym))))
    (let* ((export-matches
            (filter-map
             (lambda (entry)
               (let* ((name (car entry))
                      (name-str (string-downcase (symbol->string name))))
                 (if (string-contains? name-str needle)
                     (let ((mod (hamt-lookup name *export-module-map*)))
                       (list name 0.8 'export `((name . ,name) ,@(if mod `((module . ,mod)) '()))))
                     #f)))
             (kg-exports)))
           (skill-matches
            (filter-map
             (lambda (skill)
               (let ((name-str (string-downcase (symbol->string skill))))
                 (if (string-contains? name-str needle)
                     (list skill 0.85 'skill (kg-skill-data skill))
                     #f)))
             (kg-skills)))
           (sorted (sort-by (lambda (a b) (> (cadr a) (cadr b)))
                            (append skill-matches export-matches))))
      (take-at-most k sorted))))""",
    "lattice-find-by-tier": """(define (find-by-tier query tier . options)
  (let* ((k (if (pair? options) (car options) 10))
         (results (lattice-find query k 'skill)))
    (filter
     (lambda (result)
       (let ((data (cadddr result)))
         (and data
              (let ((t (assq 'tier data)))
                (and t (= (cdr t) tier))))))
     results)))""",
    "lattice-find-by-purity": """(define (find-by-purity query purity . options)
  (let* ((k (if (pair? options) (car options) 10))
         (results (lattice-find query k 'skill)))
    (filter
     (lambda (result)
       (let ((data (cadddr result)))
         (and data
              (let ((p (assq 'purity data)))
                (and p (eq? (cdr p) purity))))))
     results)))""",
}

BUGGY_CASES = [
    {
        "fn": "concept-boost-for-query",
        "buggy": """(define (concept-boost-for-query query-terms)
  (fold-left
   (lambda (boost-set term)
     (let ([skills (kg-concept-skills term)])
       (fold-left
        (lambda (bs skill-name)
          (hamt-assoc skill-name 1 bs))
        boost-set
        skills)))
   hamt-empty
   query-terms))""",
        "note": "Repeated concept hits must accumulate counts; overwriting to 1 loses frequency information.",
    },
    {
        "fn": "concept-boost-for-query",
        "buggy": """(define (concept-boost-for-query query-terms)
  (fold-left
   (lambda (boost-set term)
     (let ([skills (kg-concept-skills term)])
       (fold-left
        (lambda (bs skill-name)
          (let ([current (or (hamt-lookup skill-name boost-set) 0)])
            (hamt-assoc skill-name (+ current 1) bs)))
        boost-set
        skills)))
   hamt-empty
   query-terms))""",
        "note": "Use the inner accumulator when reading current counts; reading the outer snapshot drops increments in the same fold.",
    },
    {
        "fn": "result->skill-name",
        "buggy": """(define (result->skill-name result)
  (let ([id (car result)]
        [type (caddr result)]
        [data (cadddr result)])
    (case type
      [(skill) id]
      [(module)
       (and data
            (let ([name (assq 'name data)])
              (and name (cdr name))))]
      [(export)
       (and data
            (let ([mod (assq 'module data)])
              (and mod (hamt-lookup (cdr mod) *module-skill-map*))))]
      [else #f])))""",
        "note": "Module results carry parent skill under the 'skill field, not under 'name.",
    },
    {
        "fn": "result->skill-name",
        "buggy": """(define (result->skill-name result)
  (let ([id (car result)]
        [type (caddr result)]
        [data (cadddr result)])
    (case type
      [(skill) id]
      [(module)
       (and data
            (let ([skill (assq 'skill data)])
              (and skill (cdr skill))))]
      [(export)
       (and data
            (let ([mod (assq 'module data)])
              (and mod (cdr mod))))]
      [else #f])))""",
        "note": "Export results must map module -> skill via *module-skill-map*; returning module name is incorrect.",
    },
    {
        "fn": "apply-concept-boosts",
        "buggy": """(define (apply-concept-boosts results boost-map)
  (if (zero? (hamt-size boost-map))
      '()
      (map (lambda (result)
             (let* ([skill (result->skill-name result)]
                    [count (and skill (hamt-lookup skill boost-map))])
               (if count
                   (let* ([multiplier (min CONCEPT-BOOST-CAP
                                           (+ 1.0 (* CONCEPT-BOOST count)))]
                          [old-score (cadr result)]
                          [new-score (* old-score multiplier)])
                     (list (car result) new-score (caddr result) (cadddr result)))
                   result)))
           results)))""",
        "note": "With an empty boost map, results should be returned unchanged, not dropped.",
    },
    {
        "fn": "apply-concept-boosts",
        "buggy": """(define (apply-concept-boosts results boost-map)
  (if (zero? (hamt-size boost-map))
      results
      (map (lambda (result)
             (let* ([skill (result->skill-name result)]
                    [count (and skill (hamt-lookup skill boost-map))])
               (if count
                   (let* ([multiplier (min CONCEPT-BOOST-CAP
                                           (* CONCEPT-BOOST count))]
                          [old-score (cadr result)]
                          [new-score (* old-score multiplier)])
                     (list (car result) new-score (caddr result) (cadddr result)))
                   result)))
           results)))""",
        "note": "Multiplier must be 1 + boost*count before capping; omitting the base factor under-scores all matches.",
    },
    {
        "fn": "lattice-export-source",
        "buggy": """(define (lattice-export-source sym)
  (let loop ([skills (kg-skills)])
    (if (null? skills) #f
        (let* ([skill-name (car skills)]
               [data (kg-skill-data skill-name)]
               [exports-raw (if data
                                (let ([e (assq 'exports data)])
                                  (if e (cdr e) '()))
                                '())])
          (cond
           [(not (list? exports-raw)) (loop (cdr skills))]
           [(null? exports-raw) (loop (cdr skills))]
           [(symbol? (car exports-raw))
            (if (and (pair? exports-raw) (eq? sym (car exports-raw)))
                skill-name
                (loop (cdr skills)))]
           [else
            (let group-loop ([groups exports-raw])
              (if (null? groups)
                  (loop (cdr skills))
                  (let ([group (car groups)])
                    (if (and (pair? group) (memq sym group))
                        skill-name
                        (group-loop (cdr groups))))))])))))""",
        "note": "Flat export lists must check membership across the whole list, not only the first symbol.",
    },
    {
        "fn": "lattice-export-source",
        "buggy": """(define (lattice-export-source sym)
  (let loop ([skills (kg-skills)])
    (if (null? skills) #f
        (let* ([skill-name (car skills)]
               [data (kg-skill-data skill-name)]
               [exports-raw (if data
                                (let ([e (assq 'exports data)])
                                  (if e (cdr e) '()))
                                '())])
          (cond
           [(not (list? exports-raw)) (loop (cdr skills))]
           [(null? exports-raw) (loop (cdr skills))]
           [(symbol? (car exports-raw))
            (if (memq sym exports-raw)
                sym
                (loop (cdr skills)))]
           [else
            (let group-loop ([groups exports-raw])
              (if (null? groups)
                  (loop (cdr skills))
                  (let ([group (car groups)])
                    (if (and (pair? group) (memq sym group))
                        sym
                        (group-loop (cdr groups))))))])))))""",
        "note": "Return the owning skill name, not the export symbol itself.",
    },
    {
        "fn": "lattice-find-prefix",
        "buggy": """(define (lattice-find-prefix prefix-sym . options)
  (ensure-indexed!)
  (let* ([k (if (pair? options) (car options) 20)]
         [prefix-str (string-downcase (symbol->string prefix-sym))]
         [prefix-len (string-length prefix-str)])
    (let* ([export-matches
            (filter-map
             (lambda (export-entry)
               (let* ([export-name (car export-entry)]
                      [name-str (string-downcase (symbol->string export-name))])
                 (if (and (> (string-length name-str) prefix-len)
                          (string=? (substring name-str 0 prefix-len) prefix-str))
                     (let ([mod (hamt-lookup export-name *export-module-map*)])
                       (list export-name 0.9 'export
                             `((name . ,export-name)
                               ,@(if mod `((module . ,mod)) '()))))
                     #f)))
             (kg-exports))]
           [skill-matches
            (filter-map
             (lambda (skill-name)
               (let ([name-str (string-downcase (symbol->string skill-name))])
                 (if (and (> (string-length name-str) prefix-len)
                          (string=? (substring name-str 0 prefix-len) prefix-str))
                     (list skill-name 0.95 'skill (kg-skill-data skill-name))
                     #f)))
             (kg-skills))]
           [all-matches (append skill-matches export-matches)]
           [sorted (sort-by (lambda (a b) (> (cadr a) (cadr b))) all-matches)])
      (take-at-most k sorted))))""",
        "note": "Exact-length prefix matches should be included; use >= length guard, not >.",
    },
    {
        "fn": "lattice-find-prefix",
        "buggy": """(define (lattice-find-prefix prefix-sym . options)
  (ensure-indexed!)
  (let* ([k (if (pair? options) (car options) 20)]
         [prefix-str (string-downcase (symbol->string prefix-sym))]
         [prefix-len (string-length prefix-str)])
    (let* ([export-matches
            (filter-map
             (lambda (export-entry)
               (let* ([export-name (car export-entry)]
                      [name-str (string-downcase (symbol->string export-name))])
                 (if (and (>= (string-length name-str) prefix-len)
                          (string=? (substring name-str 0 prefix-len) prefix-str))
                     (let ([mod (hamt-lookup export-name *export-module-map*)])
                       (list export-name 0.9 'export
                             `((name . ,export-name)
                               ,@(if mod `((module . ,mod)) '()))))
                     #f)))
             (kg-exports))]
           [skill-matches
            (filter-map
             (lambda (skill-name)
               (let ([name-str (string-downcase (symbol->string skill-name))])
                 (if (and (>= (string-length name-str) prefix-len)
                          (string=? (substring name-str 0 prefix-len) prefix-str))
                     (list skill-name 0.95 'skill (kg-skill-data skill-name))
                     #f)))
             (kg-skills))]
           [all-matches (append skill-matches export-matches)]
           [sorted (sort-by (lambda (a b) (< (cadr a) (cadr b))) all-matches)])
      (take-at-most k sorted))))""",
        "note": "Results should be ranked from highest score to lowest, not ascending.",
    },
    {
        "fn": "lattice-find-substring",
        "buggy": """(define (lattice-find-substring substr-sym . options)
  (ensure-indexed!)
  (let* ([k (if (pair? options) (car options) 20)]
         [substr-str (string-downcase (symbol->string substr-sym))])
    (let* ([export-matches
            (filter-map
             (lambda (export-entry)
               (let* ([export-name (car export-entry)]
                      [name-str (string-downcase (symbol->string export-name))])
                 (if (and (>= (string-length name-str) (string-length substr-str))
                          (string=? (substring name-str 0 (string-length substr-str)) substr-str))
                     (let ([mod (hamt-lookup export-name *export-module-map*)])
                       (list export-name 0.8 'export
                             `((name . ,export-name)
                               ,@(if mod `((module . ,mod)) '()))))
                     #f)))
             (kg-exports))]
           [skill-matches
            (filter-map
             (lambda (skill-name)
               (let ([name-str (string-downcase (symbol->string skill-name))])
                 (if (and (>= (string-length name-str) (string-length substr-str))
                          (string=? (substring name-str 0 (string-length substr-str)) substr-str))
                     (list skill-name 0.85 'skill (kg-skill-data skill-name))
                     #f)))
             (kg-skills))]
           [all-matches (append skill-matches export-matches)]
           [sorted (sort-by (lambda (a b) (> (cadr a) (cadr b))) all-matches)])
      (take-at-most k sorted))))""",
        "note": "Substring search must match anywhere in the name, not only at the start.",
    },
    {
        "fn": "lattice-find-substring",
        "buggy": """(define (lattice-find-substring substr-sym . options)
  (ensure-indexed!)
  (let* ([k (if (pair? options) (car options) 20)]
         [substr-str (string-downcase (symbol->string substr-sym))])
    (let* ([export-matches
            (filter-map
             (lambda (export-entry)
               (let* ([export-name (car export-entry)]
                      [name-str (string-downcase (symbol->string export-name))])
                 (if (string-contains? name-str substr-str)
                     (let ([mod (hamt-lookup export-name *export-module-map*)])
                       (list export-name 0.8 'export
                             `((name . ,export-name)
                               ,@(if mod `((module . ,mod)) '()))))
                     #f)))
             (kg-exports))]
           [skill-matches
            (filter-map
             (lambda (skill-name)
               (let ([name-str (string-downcase (symbol->string skill-name))])
                 (if (string-contains? name-str substr-str)
                     (list skill-name 0.75 'skill (kg-skill-data skill-name))
                     #f)))
             (kg-skills))]
           [all-matches (append skill-matches export-matches)]
           [sorted (sort-by (lambda (a b) (> (cadr a) (cadr b))) all-matches)])
      (take-at-most k sorted))))""",
        "note": "Skills should retain higher base score than exports in substring search.",
    },
    {
        "fn": "lattice-find-by-tier",
        "buggy": """(define (lattice-find-by-tier query tier . options)
  (let* ([k (if (pair? options) (car options) 10)]
         [results (lattice-find query k 'skill)])
    (filter
     (lambda (result)
       (let ([data (cadddr result)])
         (and data
              (let ([t (assq 'level data)])
                (and t (= (cdr t) tier))))))
     results)))""",
        "note": "Tier metadata is stored under 'tier, not 'level.",
    },
    {
        "fn": "lattice-find-by-tier",
        "buggy": """(define (lattice-find-by-tier query tier . options)
  (let* ([k (if (pair? options) (car options) 10)]
         [results (lattice-find query k 'skill)])
    (filter
     (lambda (result)
       (let ([data (cadddr result)])
         (and data
              (let ([t (assq 'tier data)])
                (and t (not (= (cdr t) tier)))))))
     results)))""",
        "note": "Filter predicate must keep matching tiers, not reject them.",
    },
    {
        "fn": "lattice-find-by-purity",
        "buggy": """(define (lattice-find-by-purity query purity . options)
  (let* ([k (if (pair? options) (car options) 10)]
         [results (lattice-find query k 'skill)])
    (filter
     (lambda (result)
       (let ([data (cadddr result)])
         (and data
              (let ([p (assq 'tier data)])
                (and p (eq? (cdr p) purity))))))
     results)))""",
        "note": "Purity filter must read the 'purity field, not 'tier.",
    },
    {
        "fn": "lattice-find-by-purity",
        "buggy": """(define (lattice-find-by-purity query purity . options)
  (let* ([k (if (pair? options) (car options) 10)]
         [results (lattice-find query k 'skill)])
    (filter
     (lambda (result)
       (let ([data (cadddr result)])
         (and data
              (let ([p (assq 'purity data)])
                (and p (eq? (cdr p) 'purity))))))
     results)))""",
        "note": "Compare against the provided purity argument, not the literal symbol 'purity.",
    },
]

DIFFICULTY = {
    "concept-boost-for-query": "hard",
    "result->skill-name": "medium",
    "apply-concept-boosts": "hard",
    "lattice-export-source": "medium",
    "lattice-find-prefix": "medium",
    "lattice-find-substring": "medium",
    "lattice-find-by-tier": "easy",
    "lattice-find-by-purity": "easy",
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
    sid = f"meta_kg_usage_{family}_{family_counter[family]:03d}"
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
    parts = GLOBAL_DEFS + [DEFS[d] for d in dependency_closure(fn)] + [DEFS[fn], VERIFY_BY_FUNCTION[fn]]
    body = "\n  ".join(parts)
    return f"(let ()\n  {body})"


def wrap_verify_expr(expr: str) -> str:
    parts = GLOBAL_DEFS + [DEFS[name] for name in FUNCTION_ORDER] + [expr]
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
        tags=["tier1", "meta", "knowledge-graph", "usage", "spec-to-code", fn],
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
        tags=["tier1", "meta", "knowledge-graph", "usage", "spec-to-code", "skeleton", fn],
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
        tags=["tier1", "meta", "knowledge-graph", "usage", "translation", "python", fn],
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
        tags=["tier1", "meta", "knowledge-graph", "usage", "translation", "chez", fn],
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
        tags=["tier1", "meta", "knowledge-graph", "usage", "bugfix", fn],
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
        tags=["tier1", "meta", "knowledge-graph", "usage", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # concept-boost-for-query
    (
        "concept-boost-for-query",
        "Build concept boost counts for terms '(search optimization search)' and return counts for meta/query/linalg.",
        "(let ([m (concept-boost-for-query '(search optimization search))]) (list (hamt-lookup 'meta m) (hamt-lookup 'query m) (hamt-lookup 'linalg m)))",
        "(equal? (let ([m (concept-boost-for-query '(search optimization search))]) (list (hamt-lookup 'meta m) (hamt-lookup 'query m) (hamt-lookup 'linalg m))) '(2 3 1))",
        "hard",
        ["direct"],
    ),
    (
        "concept-boost-for-query",
        "Return the size of the boost map for an empty query term list.",
        "(hamt-size (concept-boost-for-query '()))",
        "(equal? (hamt-size (concept-boost-for-query '())) 0)",
        "easy",
        ["edge-case"],
    ),
    (
        "concept-boost-for-query",
        "Use concept boosts from '(crypto search)' to boost two results and return their boosted scores.",
        "(let* ([boost (concept-boost-for-query '(crypto search))] [rs (apply-concept-boosts (list '(sha256 1.0 export ((module . sha256))) '(meta 1.0 skill ((tier . 1)))) boost)]) (map cadr rs))",
        "(let* ([boost (concept-boost-for-query '(crypto search))] [rs (apply-concept-boosts (list '(sha256 1.0 export ((module . sha256))) '(meta 1.0 skill ((tier . 1)))) boost)] [scores (map cadr rs)]) (and (< (abs (- (list-ref scores 0) 1.25)) 0.00001) (< (abs (- (list-ref scores 1) 1.25)) 0.00001)))",
        "hard",
        ["integration"],
    ),
    (
        "concept-boost-for-query",
        "Check that repeating concept 'graph three times gives graphics a boost count of 3.",
        "(let ([m (concept-boost-for-query '(graph graph graph))]) (= (hamt-lookup 'graphics m) 3))",
        "(equal? (let ([m (concept-boost-for-query '(graph graph graph))]) (= (hamt-lookup 'graphics m) 3)) #t)",
        "medium",
        ["property"],
    ),

    # result->skill-name
    (
        "result->skill-name",
        "Extract parent skill from a direct skill result row.",
        "(result->skill-name '(query 0.9 skill ((tier . 1))))",
        "(equal? (result->skill-name '(query 0.9 skill ((tier . 1)))) 'query)",
        "easy",
        ["direct"],
    ),
    (
        "result->skill-name",
        "Extract parent skill from a module result row.",
        "(result->skill-name '(meta/search 0.8 module ((skill . meta))))",
        "(equal? (result->skill-name '(meta/search 0.8 module ((skill . meta)))) 'meta)",
        "medium",
        ["direct"],
    ),
    (
        "result->skill-name",
        "Extract parent skill from an export row using module->skill mapping.",
        "(result->skill-name '(sql-run 0.7 export ((module . sql))))",
        "(equal? (result->skill-name '(sql-run 0.7 export ((module . sql)))) 'query)",
        "medium",
        ["integration"],
    ),
    (
        "result->skill-name",
        "Return #f for an export row whose module has no skill mapping.",
        "(result->skill-name '(mystery 0.2 export ((module . unknown))))",
        "(equal? (result->skill-name '(mystery 0.2 export ((module . unknown)))) #f)",
        "medium",
        ["edge-case"],
    ),

    # apply-concept-boosts
    (
        "apply-concept-boosts",
        "Apply boosts with empty map and return the unchanged result list length.",
        "(length (apply-concept-boosts (list '(meta 1.0 skill ((tier . 1))) '(query 0.8 skill ((tier . 1)))) hamt-empty))",
        "(equal? (length (apply-concept-boosts (list '(meta 1.0 skill ((tier . 1))) '(query 0.8 skill ((tier . 1)))) hamt-empty)) 2)",
        "easy",
        ["edge-case"],
    ),
    (
        "apply-concept-boosts",
        "Boost one meta skill row with count=2 and return the resulting score.",
        "(cadr (car (apply-concept-boosts (list '(meta 1.0 skill ((tier . 1)))) '((meta . 2)))))",
        "(let ([s (cadr (car (apply-concept-boosts (list '(meta 1.0 skill ((tier . 1)))) '((meta . 2)))) )]) (< (abs (- s 1.5)) 0.00001))",
        "hard",
        ["direct"],
    ),
    (
        "apply-concept-boosts",
        "Boost rows with large count and ensure capping at CONCEPT-BOOST-CAP.",
        "(cadr (car (apply-concept-boosts (list '(meta 1.0 skill ((tier . 1)))) '((meta . 100)))))",
        "(let ([s (cadr (car (apply-concept-boosts (list '(meta 1.0 skill ((tier . 1)))) '((meta . 100)))) )]) (< (abs (- s 2.0)) 0.00001))",
        "hard",
        ["property"],
    ),
    (
        "apply-concept-boosts",
        "Compose concept-boost-for-query with apply-concept-boosts for query '(search)' and return boosted meta score.",
        "(let* ([boost (concept-boost-for-query '(search))] [rows (apply-concept-boosts (list '(meta 1.0 skill ((tier . 1)))) boost)]) (cadr (car rows)))",
        "(let ([s (let* ([boost (concept-boost-for-query '(search))] [rows (apply-concept-boosts (list '(meta 1.0 skill ((tier . 1)))) boost)]) (cadr (car rows)))]) (< (abs (- s 1.25)) 0.00001))",
        "hard",
        ["integration"],
    ),

    # lattice-export-source
    (
        "lattice-export-source",
        "Find the owning skill for export symbol lattice-find.",
        "(lattice-export-source 'lattice-find)",
        "(equal? (lattice-export-source 'lattice-find) 'meta)",
        "easy",
        ["direct"],
    ),
    (
        "lattice-export-source",
        "Find the owning skill for flat-export symbol sha256.",
        "(lattice-export-source 'sha256)",
        "(equal? (lattice-export-source 'sha256) 'crypto)",
        "medium",
        ["flat-format"],
    ),
    (
        "lattice-export-source",
        "Find the owning skill for grouped-export symbol sql-run.",
        "(lattice-export-source 'sql-run)",
        "(equal? (lattice-export-source 'sql-run) 'query)",
        "medium",
        ["grouped-format"],
    ),
    (
        "lattice-export-source",
        "Return #f for a symbol that is not exported by any skill.",
        "(lattice-export-source 'not-there)",
        "(equal? (lattice-export-source 'not-there) #f)",
        "easy",
        ["edge-case"],
    ),

    # lattice-find-prefix
    (
        "lattice-find-prefix",
        "Run prefix search for 'vec and return the first three ids.",
        "(map car (lattice-find-prefix 'vec 3))",
        "(equal? (map car (lattice-find-prefix 'vec 3)) '(vec-add vec-sub vec-dot))",
        "medium",
        ["direct"],
    ),
    (
        "lattice-find-prefix",
        "Run prefix search for 'q with k=2 and return ids.",
        "(map car (lattice-find-prefix 'q 2))",
        "(equal? (map car (lattice-find-prefix 'q 2)) '(query query-run))",
        "medium",
        ["ranking"],
    ),
    (
        "lattice-find-prefix",
        "Run prefix search for 'm and return matching ids.",
        "(map car (lattice-find-prefix 'm 5))",
        "(equal? (map car (lattice-find-prefix 'm 5)) '(meta))",
        "easy",
        ["skill-match"],
    ),
    (
        "lattice-find-prefix",
        "Verify that prefix search for 'z returns no matches.",
        "(null? (lattice-find-prefix 'z 5))",
        "(equal? (null? (lattice-find-prefix 'z 5)) #t)",
        "easy",
        ["edge-case"],
    ),

    # lattice-find-substring
    (
        "lattice-find-substring",
        "Run substring search for 'find and check presence of all find exports.",
        "(let ([ids (map car (lattice-find-substring 'find 10))]) (and (if (memq 'lattice-find ids) #t #f) (if (memq 'lattice-find-prefix ids) #t #f) (if (memq 'lattice-find-substring ids) #t #f)))",
        "(let ([ids (map car (lattice-find-substring 'find 10))]) (and (if (memq 'lattice-find ids) #t #f) (if (memq 'lattice-find-prefix ids) #t #f) (if (memq 'lattice-find-substring ids) #t #f)))",
        "medium",
        ["direct"],
    ),
    (
        "lattice-find-substring",
        "Run substring search for 'query and return top id.",
        "(car (car (lattice-find-substring 'query 3)))",
        "(equal? (car (car (lattice-find-substring 'query 3))) 'query)",
        "medium",
        ["ranking"],
    ),
    (
        "lattice-find-substring",
        "Run substring search for 'run with k=1 and return list length.",
        "(length (lattice-find-substring 'run 1))",
        "(equal? (length (lattice-find-substring 'run 1)) 1)",
        "easy",
        ["k-limit"],
    ),
    (
        "lattice-find-substring",
        "Verify substring search for 'xyz returns empty list.",
        "(null? (lattice-find-substring 'xyz 5))",
        "(equal? (null? (lattice-find-substring 'xyz 5)) #t)",
        "easy",
        ["edge-case"],
    ),

    # lattice-find-by-tier
    (
        "lattice-find-by-tier",
        "Filter query 'search to tier=1 skills and return ids.",
        "(map car (lattice-find-by-tier \"search\" 1 10))",
        "(equal? (map car (lattice-find-by-tier \"search\" 1 10)) '(meta query))",
        "easy",
        ["direct"],
    ),
    (
        "lattice-find-by-tier",
        "Filter query 'vector to tier=0 skills and return ids.",
        "(map car (lattice-find-by-tier \"vector\" 0 10))",
        "(equal? (map car (lattice-find-by-tier \"vector\" 0 10)) '(linalg))",
        "easy",
        ["direct"],
    ),
    (
        "lattice-find-by-tier",
        "Filter query 'vector to tier=2 skills and return ids.",
        "(map car (lattice-find-by-tier \"vector\" 2 10))",
        "(equal? (map car (lattice-find-by-tier \"vector\" 2 10)) '(graphics))",
        "medium",
        ["edge-case"],
    ),
    (
        "lattice-find-by-tier",
        "Filter query 'crypto to tier=1 skills with k=1.",
        "(map car (lattice-find-by-tier \"crypto\" 1 1))",
        "(equal? (map car (lattice-find-by-tier \"crypto\" 1 1)) '(crypto))",
        "easy",
        ["k-limit"],
    ),

    # lattice-find-by-purity
    (
        "lattice-find-by-purity",
        "Filter query 'search for purity 'partial and return ids.",
        "(map car (lattice-find-by-purity \"search\" 'partial 10))",
        "(equal? (map car (lattice-find-by-purity \"search\" 'partial 10)) '(meta))",
        "easy",
        ["direct"],
    ),
    (
        "lattice-find-by-purity",
        "Filter query 'search for purity 'total and return ids.",
        "(map car (lattice-find-by-purity \"search\" 'total 10))",
        "(equal? (map car (lattice-find-by-purity \"search\" 'total 10)) '(query))",
        "easy",
        ["direct"],
    ),
    (
        "lattice-find-by-purity",
        "Filter query 'vector for purity 'total and return ids.",
        "(map car (lattice-find-by-purity \"vector\" 'total 10))",
        "(equal? (map car (lattice-find-by-purity \"vector\" 'total 10)) '(linalg))",
        "easy",
        ["integration"],
    ),
    (
        "lattice-find-by-purity",
        "Filter query 'vector for purity 'partial and return ids.",
        "(map car (lattice-find-by-purity \"vector\" 'partial 10))",
        "(equal? (map car (lattice-find-by-purity \"vector\" 'partial 10)) '(graphics))",
        "medium",
        ["integration"],
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
