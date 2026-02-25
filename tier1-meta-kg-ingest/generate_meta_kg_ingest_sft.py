#!/usr/bin/env python3
"""Generate SFT samples for KG ingestion and storage workflows."""

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

SOURCE_MODULE = "lattice/meta/kg-ingest.ss"
SOURCE_TEST = "lattice/meta/test-kg-ingest.ss"

GLOBAL_DEFS = [
    """(define (fold-left f init lst)
  (let loop ([acc init] [xs lst])
    (if (null? xs)
        acc
        (loop (f acc (car xs)) (cdr xs)))))""",
    """(define (filter pred lst)
  (let loop ([xs lst] [acc '()])
    (if (null? xs)
        (reverse acc)
        (loop (cdr xs)
              (if (pred (car xs))
                  (cons (car xs) acc)
                  acc)))))""",
    """(define (string-join parts sep)
  (if (null? parts)
      ""
      (let loop ([rest (cdr parts)] [acc (car parts)])
        (if (null? rest)
            acc
            (loop (cdr rest) (string-append acc sep (car rest)))))))""",
    """(define (tokenize-simple x)
  (let* ([s (cond [(string? x) x]
                  [(symbol? x) (symbol->string x)]
                  [else ""])]
         [len (string-length s)])
    (let loop ([i 0] [cur '()] [acc '()])
      (if (= i len)
          (let ([acc2 (if (null? cur)
                          acc
                          (cons (list->string (reverse cur)) acc))])
            (reverse acc2))
          (let ([ch (char-downcase (string-ref s i))])
            (if (or (char-alphabetic? ch) (char-numeric? ch))
                (loop (+ i 1) (cons ch cur) acc)
                (if (null? cur)
                    (loop (+ i 1) '() acc)
                    (loop (+ i 1)
                          '()
                          (cons (list->string (reverse cur)) acc)))))))))""",
    """(define (list-prefix? prefix xs)
  (cond
    [(null? prefix) #t]
    [(null? xs) #f]
    [else
     (and (equal? (car prefix) (car xs))
          (list-prefix? (cdr prefix) (cdr xs)))]))""",
    """(define (drop n xs)
  (if (or (<= n 0) (null? xs))
      xs
      (drop (- n 1) (cdr xs))))""",
    """(define (split-on-pattern xs pat)
  (let loop ([rest xs] [prefix '()])
    (cond
      [(list-prefix? pat rest)
       (cons (reverse prefix)
             (drop (length pat) rest))]
      [(null? rest) #f]
      [else (loop (cdr rest) (cons (car rest) prefix))])))""",
    """(define (contains-equal? x xs)
  (cond
    [(null? xs) #f]
    [(equal? x (car xs)) #t]
    [else (contains-equal? x (cdr xs))]))""",
    """(define (alist-set alist key value)
  (let loop ([xs alist] [acc '()] [done #f])
    (cond
      [(null? xs)
       (let ([base (reverse acc)])
         (if done
             base
             (append base (list (cons key value)))))]
      [(and (pair? (car xs)) (eq? (caar xs) key))
       (loop (cdr xs) (cons (cons key value) acc) #t)]
      [else
       (loop (cdr xs) (cons (car xs) acc) done)])))""",
]

DEFS: Dict[str, str] = {
    "kg-normalize-entity": """(define (kg-normalize-entity x)
  (let ([tokens (tokenize-simple x)])
    (if (null? tokens)
        'unknown
        (string->symbol (string-join tokens "-")))))""",
    "kg-normalize-relation": """(define (kg-normalize-relation rel)
  (let ([tokens (tokenize-simple rel)])
    (cond
      [(or (equal? tokens '("works" "at"))
           (equal? tokens '("employed" "by"))
           (equal? tokens '("employee" "of")))
       'works_at]
      [(or (equal? tokens '("lives" "in"))
           (equal? tokens '("located" "in"))
           (equal? tokens '("in")))
       'located_in]
      [(or (equal? tokens '("is" "part" "of"))
           (equal? tokens '("part" "of"))
           (equal? tokens '("member" "of")))
       'part_of]
      [(equal? tokens '("founded")) 'founded]
      [(null? tokens) 'unknown_relation]
      [else
       (string->symbol
        (string-append
         "rel_"
         (symbol->string (kg-normalize-entity (string-join tokens "-")))))])))""",
    "kg-make-triple": """(define (kg-make-triple subject relation object)
  (list (kg-normalize-entity subject)
        (kg-normalize-relation relation)
        (kg-normalize-entity object)))""",
    "kg-valid-triple?": """(define (kg-valid-triple? triple)
  (and (list? triple)
       (= (length triple) 3)
       (let ([s (car triple)]
             [p (cadr triple)]
             [o (caddr triple)])
         (and (symbol? s)
              (symbol? p)
              (symbol? o)
              (not (eq? s 'unknown))
              (not (eq? o 'unknown))
              (not (eq? p 'unknown_relation))))))""",
    "kg-parse-simple-fact": """(define (kg-parse-simple-fact sentence)
  (let* ([tokens (tokenize-simple sentence)]
         [split
          (or (let ([x (split-on-pattern tokens '("works" "at"))])
                (and x (cons x "works at")))
              (let ([x (split-on-pattern tokens '("employed" "by"))])
                (and x (cons x "works at")))
              (let ([x (split-on-pattern tokens '("lives" "in"))])
                (and x (cons x "lives in")))
              (let ([x (split-on-pattern tokens '("is" "part" "of"))])
                (and x (cons x "part of")))
              (let ([x (split-on-pattern tokens '("part" "of"))])
                (and x (cons x "part of")))
              (let ([x (split-on-pattern tokens '("founded"))])
                (and x (cons x "founded"))))])
    (if (not split)
        #f
        (let* ([parts (car split)]
               [rel (cdr split)]
               [subject-tokens (car parts)]
               [object-tokens (cdr parts)])
          (if (or (null? subject-tokens)
                  (null? object-tokens))
              #f
              (kg-make-triple (string-join subject-tokens " ")
                              rel
                              (string-join object-tokens " ")))))))""",
    "kg-extract-triples": """(define (kg-extract-triples sentences)
  (let loop ([rest sentences] [acc '()])
    (if (null? rest)
        (reverse acc)
        (let ([triple (kg-parse-simple-fact (car rest))])
          (loop (cdr rest)
                (if (and triple
                         (kg-valid-triple? triple)
                         (not (contains-equal? triple acc)))
                    (cons triple acc)
                    acc))))))""",
    "kg-upsert-triple": """(define (kg-upsert-triple store triple)
  (if (not (kg-valid-triple? triple))
      store
      (let* ([s (car triple)]
             [p (cadr triple)]
             [o (caddr triple)]
             [subject-row (assq s store)]
             [preds (if subject-row (cdr subject-row) '())]
             [pred-row (assq p preds)]
             [objs (if pred-row (cdr pred-row) '())]
             [objs2 (if (memq o objs) objs (append objs (list o)))]
             [preds2 (alist-set preds p objs2)])
        (alist-set store s preds2))))""",
    "kg-upsert-triples": """(define (kg-upsert-triples store triples)
  (fold-left
   (lambda (acc triple)
     (kg-upsert-triple acc triple))
   store
   triples))""",
}

DEPENDS: Dict[str, List[str]] = {
    "kg-normalize-entity": [],
    "kg-normalize-relation": ["kg-normalize-entity"],
    "kg-make-triple": ["kg-normalize-entity", "kg-normalize-relation"],
    "kg-valid-triple?": [],
    "kg-parse-simple-fact": ["kg-make-triple"],
    "kg-extract-triples": ["kg-parse-simple-fact", "kg-valid-triple?"],
    "kg-upsert-triple": ["kg-valid-triple?"],
    "kg-upsert-triples": ["kg-upsert-triple"],
}

FUNCTION_ORDER = [
    "kg-normalize-entity",
    "kg-normalize-relation",
    "kg-make-triple",
    "kg-valid-triple?",
    "kg-parse-simple-fact",
    "kg-extract-triples",
    "kg-upsert-triple",
    "kg-upsert-triples",
]

FUNCTION_SPECS = {
    "kg-normalize-entity": "Canonicalize an entity token/string to a lowercase hyphenated symbol, returning 'unknown for empty input.",
    "kg-normalize-relation": "Map common relation phrases to canonical predicates (works_at, located_in, part_of, founded) with rel_* fallback.",
    "kg-make-triple": "Build a normalized (subject predicate object) triple from raw subject/relation/object strings or symbols.",
    "kg-valid-triple?": "Accept only well-formed 3-symbol triples that do not contain unknown placeholders.",
    "kg-parse-simple-fact": "Extract one triple from a simple natural-language fact sentence using controlled relation patterns.",
    "kg-extract-triples": "Parse a sentence list into deduplicated, valid triples while discarding non-facts.",
    "kg-upsert-triple": "Insert one triple into a nested alist KG store (subject->predicate->objects) without duplicating objects.",
    "kg-upsert-triples": "Batch-upsert triples into an existing KG store via left fold.",
}

SKELETONS = {
    "kg-normalize-entity": """(define (kg-normalize-entity x)
  ;; TODO: canonicalize entity text into a symbol
  <TODO>)""",
    "kg-normalize-relation": """(define (kg-normalize-relation rel)
  ;; TODO: map relation phrase aliases to canonical predicate symbols
  <TODO>)""",
    "kg-make-triple": """(define (kg-make-triple subject relation object)
  ;; TODO: construct normalized (subject predicate object)
  <TODO>)""",
    "kg-valid-triple?": """(define (kg-valid-triple? triple)
  ;; TODO: validate structure and reject unknown placeholders
  <TODO>)""",
    "kg-parse-simple-fact": """(define (kg-parse-simple-fact sentence)
  ;; TODO: parse one simple NL sentence into a normalized triple
  <TODO>)""",
    "kg-extract-triples": """(define (kg-extract-triples sentences)
  ;; TODO: parse all sentences, keep valid triples, and deduplicate
  <TODO>)""",
    "kg-upsert-triple": """(define (kg-upsert-triple store triple)
  ;; TODO: upsert a triple into nested subject/predicate/object store
  <TODO>)""",
    "kg-upsert-triples": """(define (kg-upsert-triples store triples)
  ;; TODO: fold kg-upsert-triple across triple list
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "kg-normalize-entity": """(and
  (eq? (kg-normalize-entity "Ada Lovelace") 'ada-lovelace)
  (eq? (kg-normalize-entity 'New_York) 'new-york)
  (eq? (kg-normalize-entity "  ") 'unknown)
  (eq? (kg-normalize-entity "C3PO") 'c3po))""",
    "kg-normalize-relation": """(and
  (eq? (kg-normalize-relation "works at") 'works_at)
  (eq? (kg-normalize-relation "employee of") 'works_at)
  (eq? (kg-normalize-relation "located in") 'located_in)
  (eq? (kg-normalize-relation "part of") 'part_of)
  (eq? (kg-normalize-relation "founded") 'founded)
  (eq? (kg-normalize-relation "contains") 'rel_contains))""",
    "kg-make-triple": """(and
  (equal? (kg-make-triple "Ada Lovelace" "works at" "Analytical Engine")
          '(ada-lovelace works_at analytical-engine))
  (equal? (kg-make-triple 'Paris "located in" 'France)
          '(paris located_in france)))""",
    "kg-valid-triple?": """(and
  (kg-valid-triple? '(ada works_at acme))
  (not (kg-valid-triple? '(unknown works_at acme)))
  (not (kg-valid-triple? '(ada unknown_relation acme)))
  (not (kg-valid-triple? '(ada works_at)))
  (not (kg-valid-triple? '(ada works_at 42))))""",
    "kg-parse-simple-fact": """(and
  (equal? (kg-parse-simple-fact "Ada works at Acme") '(ada works_at acme))
  (equal? (kg-parse-simple-fact "Paris is part of France") '(paris part_of france))
  (equal? (kg-parse-simple-fact "Linus founded Linux") '(linus founded linux))
  (equal? (kg-parse-simple-fact "Bob lives in Paris") '(bob located_in paris))
  (not (kg-parse-simple-fact "No relation sentence"))
  (not (kg-parse-simple-fact "works at Acme")))""",
    "kg-extract-triples": """(equal?
  (kg-extract-triples '("Ada works at Acme"
                        "Ada works at Acme"
                        "Bob lives in Paris"
                        "No relation sentence"))
  '((ada works_at acme)
    (bob located_in paris)))""",
    "kg-upsert-triple": """(let* ([s1 (kg-upsert-triple '() '(ada works_at acme))]
       [s2 (kg-upsert-triple s1 '(ada works_at acme))]
       [s3 (kg-upsert-triple s2 '(ada works_at babbage-labs))]
       [s4 (kg-upsert-triple s3 '(ada founded analytical-engine))])
  (and (equal? s1 '((ada (works_at acme))))
       (equal? s2 '((ada (works_at acme))))
       (equal? s3 '((ada (works_at acme babbage-labs))))
       (equal? s4 '((ada (works_at acme babbage-labs)
                         (founded analytical-engine))))))""",
    "kg-upsert-triples": """(equal?
  (kg-upsert-triples '()
                     '((ada works_at acme)
                       (bob located_in paris)
                       (ada works_at acme)
                       (ada founded analytical-engine)))
  '((ada (works_at acme)
         (founded analytical-engine))
    (bob (located_in paris))))""",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")
TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "kg-normalize-entity": """def kg_normalize_entity(x):
    s = x if isinstance(x, str) else str(x)
    tokens = tokenize_simple(s)
    if not tokens:
        return "unknown"
    return "-".join(tokens)""",
    "kg-normalize-relation": """def kg_normalize_relation(rel):
    tokens = tokenize_simple(rel)
    if tokens in (["works", "at"], ["employed", "by"], ["employee", "of"]):
        return "works_at"
    if tokens in (["lives", "in"], ["located", "in"], ["in"]):
        return "located_in"
    if tokens in (["is", "part", "of"], ["part", "of"], ["member", "of"]):
        return "part_of"
    if tokens == ["founded"]:
        return "founded"
    if not tokens:
        return "unknown_relation"
    return "rel_" + "-".join(tokens)""",
    "kg-make-triple": """def kg_make_triple(subject, relation, obj):
    return [kg_normalize_entity(subject),
            kg_normalize_relation(relation),
            kg_normalize_entity(obj)]""",
    "kg-valid-triple?": """def kg_valid_triple(triple):
    return (
        isinstance(triple, list)
        and len(triple) == 3
        and all(isinstance(x, str) for x in triple)
        and triple[0] != "unknown"
        and triple[2] != "unknown"
        and triple[1] != "unknown_relation"
    )""",
    "kg-parse-simple-fact": """def kg_parse_simple_fact(sentence):
    tokens = tokenize_simple(sentence)
    for pattern, rel in [
        (["works", "at"], "works at"),
        (["employed", "by"], "works at"),
        (["lives", "in"], "lives in"),
        (["is", "part", "of"], "part of"),
        (["part", "of"], "part of"),
        (["founded"], "founded"),
    ]:
        i = find_subseq(tokens, pattern)
        if i is not None:
            subj = tokens[:i]
            obj = tokens[i + len(pattern):]
            if subj and obj:
                return kg_make_triple(" ".join(subj), rel, " ".join(obj))
            return None
    return None""",
    "kg-extract-triples": """def kg_extract_triples(sentences):
    out = []
    for s in sentences:
        t = kg_parse_simple_fact(s)
        if t is not None and kg_valid_triple(t) and t not in out:
            out.append(t)
    return out""",
    "kg-upsert-triple": """def kg_upsert_triple(store, triple):
    if not kg_valid_triple(triple):
        return store
    s, p, o = triple
    preds = dict(store.get(s, {}))
    objs = list(preds.get(p, []))
    if o not in objs:
        objs.append(o)
    preds[p] = objs
    new_store = dict(store)
    new_store[s] = preds
    return new_store""",
    "kg-upsert-triples": """def kg_upsert_triples(store, triples):
    acc = store
    for triple in triples:
        acc = kg_upsert_triple(acc, triple)
    return acc""",
}

CHEZ_SNIPPETS = {
    "kg-normalize-entity": """(define (normalize-entity x)
  (let ([tokens (tokenize-simple x)])
    (if (null? tokens)
        'unknown
        (string->symbol (string-join tokens "-")))))""",
    "kg-normalize-relation": """(define (normalize-relation rel)
  (let ([tokens (tokenize-simple rel)])
    (cond
      [(or (equal? tokens '("works" "at"))
           (equal? tokens '("employed" "by"))
           (equal? tokens '("employee" "of")))
       'works_at]
      [(or (equal? tokens '("lives" "in"))
           (equal? tokens '("located" "in"))
           (equal? tokens '("in")))
       'located_in]
      [(or (equal? tokens '("is" "part" "of"))
           (equal? tokens '("part" "of"))
           (equal? tokens '("member" "of")))
       'part_of]
      [(equal? tokens '("founded")) 'founded]
      [(null? tokens) 'unknown_relation]
      [else
       (string->symbol (string-append "rel_" (string-join tokens "-")))])))""",
    "kg-make-triple": """(define (make-triple subject relation object)
  (list (kg-normalize-entity subject)
        (kg-normalize-relation relation)
        (kg-normalize-entity object)))""",
    "kg-valid-triple?": """(define (valid-triple? triple)
  (and (list? triple)
       (= (length triple) 3)
       (let ([s (car triple)]
             [p (cadr triple)]
             [o (caddr triple)])
         (and (symbol? s)
              (symbol? p)
              (symbol? o)
              (not (eq? s 'unknown))
              (not (eq? o 'unknown))
              (not (eq? p 'unknown_relation))))))""",
    "kg-parse-simple-fact": """(define (parse-simple-fact sentence)
  (let* ([tokens (tokenize-simple sentence)]
         [split (or (let ([x (split-on-pattern tokens '("works" "at"))]) (and x (cons x "works at")))
                    (let ([x (split-on-pattern tokens '("employed" "by"))]) (and x (cons x "works at")))
                    (let ([x (split-on-pattern tokens '("lives" "in"))]) (and x (cons x "lives in")))
                    (let ([x (split-on-pattern tokens '("is" "part" "of"))]) (and x (cons x "part of")))
                    (let ([x (split-on-pattern tokens '("part" "of"))]) (and x (cons x "part of")))
                    (let ([x (split-on-pattern tokens '("founded"))]) (and x (cons x "founded"))))])
    (if (not split)
        #f
        (let* ([parts (car split)]
               [rel (cdr split)]
               [subj (car parts)]
               [obj (cdr parts)])
          (if (or (null? subj) (null? obj))
              #f
              (kg-make-triple (string-join subj " ") rel (string-join obj " ")))))))""",
    "kg-extract-triples": """(define (extract-triples sentences)
  (let loop ([rest sentences] [acc '()])
    (if (null? rest)
        (reverse acc)
        (let ([t (kg-parse-simple-fact (car rest))])
          (loop (cdr rest)
                (if (and t (kg-valid-triple? t) (not (contains-equal? t acc)))
                    (cons t acc)
                    acc))))))""",
    "kg-upsert-triple": """(define (upsert-triple store triple)
  (if (not (kg-valid-triple? triple))
      store
      (let* ([s (car triple)]
             [p (cadr triple)]
             [o (caddr triple)]
             [subject-row (assq s store)]
             [preds (if subject-row (cdr subject-row) '())]
             [pred-row (assq p preds)]
             [objs (if pred-row (cdr pred-row) '())]
             [objs2 (if (memq o objs) objs (append objs (list o)))]
             [preds2 (alist-set preds p objs2)])
        (alist-set store s preds2))))""",
    "kg-upsert-triples": """(define (upsert-triples store triples)
  (fold-left (lambda (acc t) (kg-upsert-triple acc t))
             store
             triples))""",
}

BUGGY_CASES = [
    {
        "fn": "kg-normalize-entity",
        "buggy": """(define (kg-normalize-entity x)
  (let* ([s (if (symbol? x) (symbol->string x) x)])
    (if (or (not s) (= (string-length s) 0))
        'unknown
        (string->symbol (string-downcase s)))))""",
        "note": "Entity normalization must tokenize and hyphenate; simple downcasing leaves punctuation/spacing artifacts.",
    },
    {
        "fn": "kg-normalize-entity",
        "buggy": """(define (kg-normalize-entity x)
  (let ([tokens (tokenize-simple x)])
    (if (null? tokens)
        'unknown
        (string->symbol (car tokens)))))""",
        "note": "Multi-token entities must keep all tokens, not only the first token.",
    },
    {
        "fn": "kg-normalize-relation",
        "buggy": """(define (kg-normalize-relation rel)
  (let ([tokens (tokenize-simple rel)])
    (cond
      [(equal? tokens '("works" "at")) 'works_at]
      [(equal? tokens '("lives" "in")) 'located_in]
      [(equal? tokens '("part" "of")) 'part_of]
      [(equal? tokens '("founded")) 'founded]
      [else 'unknown_relation])))""",
        "note": "Known relation aliases (employee of / employed by / is part of) must map to canonical predicates.",
    },
    {
        "fn": "kg-normalize-relation",
        "buggy": """(define (kg-normalize-relation rel)
  (let ([tokens (tokenize-simple rel)])
    (cond
      [(or (equal? tokens '("works" "at"))
           (equal? tokens '("employed" "by"))
           (equal? tokens '("employee" "of")))
       'works_at]
      [(or (equal? tokens '("lives" "in"))
           (equal? tokens '("located" "in"))
           (equal? tokens '("in")))
       'located_in]
      [(or (equal? tokens '("is" "part" "of"))
           (equal? tokens '("part" "of"))
           (equal? tokens '("member" "of")))
       'part_of]
      [(equal? tokens '("founded")) 'founded]
      [(null? tokens) 'unknown_relation]
      [else 'unknown_relation])))""",
        "note": "Unknown relations should be preserved as rel_* predicates, not collapsed to unknown_relation.",
    },
    {
        "fn": "kg-make-triple",
        "buggy": """(define (kg-make-triple subject relation object)
  (list (kg-normalize-entity subject)
        relation
        (kg-normalize-entity object)))""",
        "note": "Relation field must be normalized with kg-normalize-relation.",
    },
    {
        "fn": "kg-make-triple",
        "buggy": """(define (kg-make-triple subject relation object)
  (list (kg-normalize-entity object)
        (kg-normalize-relation relation)
        (kg-normalize-entity subject)))""",
        "note": "Subject/object order is reversed; preserve (subject predicate object).",
    },
    {
        "fn": "kg-valid-triple?",
        "buggy": """(define (kg-valid-triple? triple)
  (and (list? triple)
       (= (length triple) 3)))""",
        "note": "Validation must enforce symbol fields and reject unknown placeholders.",
    },
    {
        "fn": "kg-valid-triple?",
        "buggy": """(define (kg-valid-triple? triple)
  (and (list? triple)
       (= (length triple) 3)
       (let ([s (car triple)]
             [p (cadr triple)]
             [o (caddr triple)])
         (and (symbol? s)
              (symbol? p)
              (symbol? o)))) )""",
        "note": "Triples containing unknown/unknown_relation should be rejected as invalid facts.",
    },
    {
        "fn": "kg-parse-simple-fact",
        "buggy": """(define (kg-parse-simple-fact sentence)
  (let* ([tokens (tokenize-simple sentence)]
         [split (let ([x (split-on-pattern tokens '("works" "at"))])
                  (and x (cons x "works at")))])
    (if (not split)
        #f
        (let* ([parts (car split)]
               [subject-tokens (car parts)]
               [object-tokens (cdr parts)])
          (kg-make-triple (string-join subject-tokens " ")
                          "works at"
                          (string-join object-tokens " "))))))""",
        "note": "Parser should support multiple relation patterns (lives in, part of, founded, employed by).",
    },
    {
        "fn": "kg-parse-simple-fact",
        "buggy": """(define (kg-parse-simple-fact sentence)
  (let* ([tokens (tokenize-simple sentence)]
         [split (or (let ([x (split-on-pattern tokens '("works" "at"))]) (and x (cons x "works at")))
                    (let ([x (split-on-pattern tokens '("founded"))]) (and x (cons x "founded"))))])
    (if (not split)
        #f
        (let* ([parts (car split)]
               [rel (cdr split)]
               [subject-tokens (car parts)]
               [object-tokens (cdr parts)])
          (kg-make-triple (string-join subject-tokens " ")
                          rel
                          (string-join object-tokens " "))))))""",
        "note": "Reject malformed facts with empty subject or object instead of emitting unknown placeholders.",
    },
    {
        "fn": "kg-extract-triples",
        "buggy": """(define (kg-extract-triples sentences)
  (let loop ([rest sentences] [acc '()])
    (if (null? rest)
        (reverse acc)
        (let ([triple (kg-parse-simple-fact (car rest))])
          (loop (cdr rest)
                (if triple
                    (cons triple acc)
                    acc))))))""",
        "note": "Extraction must deduplicate repeated facts and filter invalid triples.",
    },
    {
        "fn": "kg-extract-triples",
        "buggy": """(define (kg-extract-triples sentences)
  (map kg-parse-simple-fact sentences))""",
        "note": "Map-only implementation leaks #f values and duplicates; require parsed/valid/deduplicated output.",
    },
    {
        "fn": "kg-upsert-triple",
        "buggy": """(define (kg-upsert-triple store triple)
  (if (not (kg-valid-triple? triple))
      store
      (let* ([s (car triple)]
             [p (cadr triple)]
             [o (caddr triple)]
             [subject-row (assq s store)]
             [preds (if subject-row (cdr subject-row) '())]
             [pred-row (assq p preds)]
             [objs (if pred-row (cdr pred-row) '())]
             [objs2 (append objs (list o))]
             [preds2 (alist-set preds p objs2)])
        (alist-set store s preds2))))""",
        "note": "Object lists should be set-like per predicate; avoid inserting duplicates on repeated upsert.",
    },
    {
        "fn": "kg-upsert-triple",
        "buggy": """(define (kg-upsert-triple store triple)
  (if (not (kg-valid-triple? triple))
      store
      (let* ([s (car triple)]
             [p (cadr triple)]
             [o (caddr triple)]
             [preds2 (list (cons p (list o)))])
        (alist-set store s preds2))))""",
        "note": "Upsert should preserve existing predicates/objects for a subject instead of overwriting the entire row.",
    },
    {
        "fn": "kg-upsert-triples",
        "buggy": """(define (kg-upsert-triples store triples)
  (fold-left
   (lambda (_ triple)
     (kg-upsert-triple store triple))
   store
   triples))""",
        "note": "Batch upsert must thread the evolving accumulator, not repeatedly use the original store.",
    },
    {
        "fn": "kg-upsert-triples",
        "buggy": """(define (kg-upsert-triples store triples)
  (fold-left
   (lambda (acc triple)
     (kg-upsert-triple acc triple))
   '()
   triples))""",
        "note": "Batch upsert must start from input store; resetting to empty drops existing KG state.",
    },
]

DIFFICULTY = {
    "kg-normalize-entity": "easy",
    "kg-normalize-relation": "medium",
    "kg-make-triple": "easy",
    "kg-valid-triple?": "medium",
    "kg-parse-simple-fact": "hard",
    "kg-extract-triples": "hard",
    "kg-upsert-triple": "medium",
    "kg-upsert-triples": "medium",
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
    sid = f"meta_kg_ingest_{family}_{family_counter[family]:03d}"
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
        tags=["tier1", "meta", "knowledge-graph", "ingestion", "spec-to-code", fn],
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
        tags=["tier1", "meta", "knowledge-graph", "ingestion", "spec-to-code", "skeleton", fn],
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
        tags=["tier1", "meta", "knowledge-graph", "ingestion", "translation", "python", fn],
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
        tags=["tier1", "meta", "knowledge-graph", "ingestion", "translation", "chez", fn],
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
        tags=["tier1", "meta", "knowledge-graph", "ingestion", "bugfix", fn],
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
        tags=["tier1", "meta", "knowledge-graph", "ingestion", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # kg-normalize-entity
    (
        "kg-normalize-entity",
        "Normalize the entity text 'Ada Lovelace, PhD!' into a canonical symbol.",
        "(kg-normalize-entity \"Ada Lovelace, PhD!\")",
        "(equal? (kg-normalize-entity \"Ada Lovelace, PhD!\") 'ada-lovelace-phd)",
        "easy",
        ["direct"],
    ),
    (
        "kg-normalize-entity",
        "Normalize a symbol input 'New_York into canonical entity form.",
        "(kg-normalize-entity 'New_York)",
        "(equal? (kg-normalize-entity 'New_York) 'new-york)",
        "easy",
        ["symbol-input"],
    ),
    (
        "kg-normalize-entity",
        "Map normalization over three raw entities and return the resulting symbol list.",
        "(map kg-normalize-entity '(\"Alan Turing\" \"Bletchley Park\" \"C3PO\"))",
        "(equal? (map kg-normalize-entity '(\"Alan Turing\" \"Bletchley Park\" \"C3PO\")) '(alan-turing bletchley-park c3po))",
        "medium",
        ["map"],
    ),
    (
        "kg-normalize-entity",
        "Compose entity normalization with triple construction and return the subject symbol.",
        "(car (kg-make-triple \"Grace Hopper\" \"founded\" \"COBOL\"))",
        "(equal? (car (kg-make-triple \"Grace Hopper\" \"founded\" \"COBOL\")) 'grace-hopper)",
        "medium",
        ["integration"],
    ),

    # kg-normalize-relation
    (
        "kg-normalize-relation",
        "Normalize relation phrase 'employee of to the canonical predicate.",
        "(kg-normalize-relation \"employee of\")",
        "(equal? (kg-normalize-relation \"employee of\") 'works_at)",
        "easy",
        ["alias"],
    ),
    (
        "kg-normalize-relation",
        "Normalize relation phrase 'member of to the canonical predicate.",
        "(kg-normalize-relation \"member of\")",
        "(equal? (kg-normalize-relation \"member of\") 'part_of)",
        "easy",
        ["alias"],
    ),
    (
        "kg-normalize-relation",
        "Normalize unknown relation phrase 'contains while preserving information as rel_*.",
        "(kg-normalize-relation \"contains\")",
        "(equal? (kg-normalize-relation \"contains\") 'rel_contains)",
        "medium",
        ["fallback"],
    ),
    (
        "kg-normalize-relation",
        "Parse 'Ada employed by Acme' and return only the predicate from the extracted triple.",
        "(cadr (kg-parse-simple-fact \"Ada employed by Acme\"))",
        "(equal? (cadr (kg-parse-simple-fact \"Ada employed by Acme\")) 'works_at)",
        "hard",
        ["integration"],
    ),

    # kg-make-triple
    (
        "kg-make-triple",
        "Construct a normalized triple from raw text fields for a work relation.",
        "(kg-make-triple \"Ada Lovelace\" \"works at\" \"Analytical Engine\")",
        "(equal? (kg-make-triple \"Ada Lovelace\" \"works at\" \"Analytical Engine\") '(ada-lovelace works_at analytical-engine))",
        "easy",
        ["direct"],
    ),
    (
        "kg-make-triple",
        "Construct a triple with an unknown relation phrase and keep rel_* fallback predicate.",
        "(kg-make-triple \"Parser\" \"contains\" \"Grammar\")",
        "(equal? (kg-make-triple \"Parser\" \"contains\" \"Grammar\") '(parser rel_contains grammar))",
        "medium",
        ["fallback"],
    ),
    (
        "kg-make-triple",
        "Upsert one freshly constructed triple and return the resulting store.",
        "(kg-upsert-triple '() (kg-make-triple \"Ada\" \"works at\" \"Acme\"))",
        "(equal? (kg-upsert-triple '() (kg-make-triple \"Ada\" \"works at\" \"Acme\")) '((ada (works_at acme))))",
        "medium",
        ["integration"],
    ),
    (
        "kg-make-triple",
        "Build two triples and return whether their predicates are canonicalized to works_at and located_in.",
        "(let ([t1 (kg-make-triple \"Ada\" \"employee of\" \"Acme\")] [t2 (kg-make-triple \"Ada\" \"lives in\" \"Paris\")]) (list (cadr t1) (cadr t2)))",
        "(equal? (let ([t1 (kg-make-triple \"Ada\" \"employee of\" \"Acme\")] [t2 (kg-make-triple \"Ada\" \"lives in\" \"Paris\")]) (list (cadr t1) (cadr t2))) '(works_at located_in))",
        "hard",
        ["integration"],
    ),

    # kg-valid-triple?
    (
        "kg-valid-triple?",
        "Check validity of a normal triple '(ada works_at acme).",
        "(kg-valid-triple? '(ada works_at acme))",
        "(equal? (kg-valid-triple? '(ada works_at acme)) #t)",
        "easy",
        ["direct"],
    ),
    (
        "kg-valid-triple?",
        "Reject a triple with unknown subject placeholder.",
        "(kg-valid-triple? '(unknown works_at acme))",
        "(equal? (kg-valid-triple? '(unknown works_at acme)) #f)",
        "easy",
        ["edge-case"],
    ),
    (
        "kg-valid-triple?",
        "Parse 'Ada works at Acme' and validate the extracted triple.",
        "(kg-valid-triple? (kg-parse-simple-fact \"Ada works at Acme\"))",
        "(equal? (kg-valid-triple? (kg-parse-simple-fact \"Ada works at Acme\")) #t)",
        "medium",
        ["integration"],
    ),
    (
        "kg-valid-triple?",
        "Filter a mixed triple list by kg-valid-triple? and return the survivors.",
        "(filter kg-valid-triple? '((ada works_at acme) (unknown works_at acme) (ada unknown_relation acme)))",
        "(equal? (filter kg-valid-triple? '((ada works_at acme) (unknown works_at acme) (ada unknown_relation acme))) '((ada works_at acme)))",
        "medium",
        ["filter"],
    ),

    # kg-parse-simple-fact
    (
        "kg-parse-simple-fact",
        "Parse a basic work sentence into a triple.",
        "(kg-parse-simple-fact \"Ada works at Acme\")",
        "(equal? (kg-parse-simple-fact \"Ada works at Acme\") '(ada works_at acme))",
        "medium",
        ["direct"],
    ),
    (
        "kg-parse-simple-fact",
        "Parse a part-of sentence into a triple.",
        "(kg-parse-simple-fact \"Paris is part of France\")",
        "(equal? (kg-parse-simple-fact \"Paris is part of France\") '(paris part_of france))",
        "hard",
        ["direct"],
    ),
    (
        "kg-parse-simple-fact",
        "Parse a founded sentence into a triple.",
        "(kg-parse-simple-fact \"Linus founded Linux\")",
        "(equal? (kg-parse-simple-fact \"Linus founded Linux\") '(linus founded linux))",
        "hard",
        ["direct"],
    ),
    (
        "kg-parse-simple-fact",
        "Return #f for non-fact text that has no recognized relation pattern.",
        "(kg-parse-simple-fact \"This sentence has no supported relation\")",
        "(equal? (kg-parse-simple-fact \"This sentence has no supported relation\") #f)",
        "medium",
        ["edge-case"],
    ),

    # kg-extract-triples
    (
        "kg-extract-triples",
        "Extract triples from a sentence list with duplicates and return deduplicated results.",
        "(kg-extract-triples '(\"Ada works at Acme\" \"Ada works at Acme\" \"Bob lives in Paris\"))",
        "(equal? (kg-extract-triples '(\"Ada works at Acme\" \"Ada works at Acme\" \"Bob lives in Paris\")) '((ada works_at acme) (bob located_in paris)))",
        "medium",
        ["dedupe"],
    ),
    (
        "kg-extract-triples",
        "Extract triples while ignoring unsupported lines and preserving first-seen order.",
        "(kg-extract-triples '(\"Noise\" \"Linus founded Linux\" \"Ada employed by Acme\" \"Noise\"))",
        "(equal? (kg-extract-triples '(\"Noise\" \"Linus founded Linux\" \"Ada employed by Acme\" \"Noise\")) '((linus founded linux) (ada works_at acme)))",
        "hard",
        ["robustness"],
    ),
    (
        "kg-extract-triples",
        "Extract triples from mixed relation forms including part-of and lives-in.",
        "(kg-extract-triples '(\"Graph is part of Math\" \"Ada lives in London\"))",
        "(equal? (kg-extract-triples '(\"Graph is part of Math\" \"Ada lives in London\")) '((graph part_of math) (ada located_in london)))",
        "hard",
        ["coverage"],
    ),
    (
        "kg-extract-triples",
        "Compose extraction with upsert and return resulting store for three facts.",
        "(kg-upsert-triples '() (kg-extract-triples '(\"Ada works at Acme\" \"Ada founded Engine\" \"Bob lives in Paris\")))",
        "(equal? (kg-upsert-triples '() (kg-extract-triples '(\"Ada works at Acme\" \"Ada founded Engine\" \"Bob lives in Paris\"))) '((ada (works_at acme) (founded engine)) (bob (located_in paris))))",
        "hard",
        ["integration"],
    ),

    # kg-upsert-triple
    (
        "kg-upsert-triple",
        "Insert one triple into an empty store and return the store.",
        "(kg-upsert-triple '() '(ada works_at acme))",
        "(equal? (kg-upsert-triple '() '(ada works_at acme)) '((ada (works_at acme))))",
        "easy",
        ["direct"],
    ),
    (
        "kg-upsert-triple",
        "Upsert the same triple twice and verify no duplicate object is created.",
        "(let* ([s1 (kg-upsert-triple '() '(ada works_at acme))] [s2 (kg-upsert-triple s1 '(ada works_at acme))]) s2)",
        "(equal? (let* ([s1 (kg-upsert-triple '() '(ada works_at acme))] [s2 (kg-upsert-triple s1 '(ada works_at acme))]) s2) '((ada (works_at acme))))",
        "medium",
        ["dedupe"],
    ),
    (
        "kg-upsert-triple",
        "Upsert a second predicate for an existing subject and return resulting subject row.",
        "(cdr (assq 'ada (kg-upsert-triple (kg-upsert-triple '() '(ada works_at acme)) '(ada founded engine))))",
        "(equal? (cdr (assq 'ada (kg-upsert-triple (kg-upsert-triple '() '(ada works_at acme)) '(ada founded engine)))) '((works_at acme) (founded engine)))",
        "medium",
        ["integration"],
    ),
    (
        "kg-upsert-triple",
        "Attempt to upsert an invalid triple and ensure the store is unchanged.",
        "(kg-upsert-triple '((ada (works_at acme))) '(unknown works_at acme))",
        "(equal? (kg-upsert-triple '((ada (works_at acme))) '(unknown works_at acme)) '((ada (works_at acme))))",
        "easy",
        ["edge-case"],
    ),

    # kg-upsert-triples
    (
        "kg-upsert-triples",
        "Batch upsert three triples with one duplicate and return resulting store.",
        "(kg-upsert-triples '() '((ada works_at acme) (ada works_at acme) (bob located_in paris)))",
        "(equal? (kg-upsert-triples '() '((ada works_at acme) (ada works_at acme) (bob located_in paris))) '((ada (works_at acme)) (bob (located_in paris))))",
        "medium",
        ["dedupe"],
    ),
    (
        "kg-upsert-triples",
        "Batch upsert into a non-empty store and keep prior facts.",
        "(kg-upsert-triples '((ada (works_at acme))) '((ada founded engine) (carol located_in rome)))",
        "(equal? (kg-upsert-triples '((ada (works_at acme))) '((ada founded engine) (carol located_in rome))) '((ada (works_at acme) (founded engine)) (carol (located_in rome))))",
        "medium",
        ["existing-state"],
    ),
    (
        "kg-upsert-triples",
        "Extract triples from NL text, then batch upsert and return final store.",
        "(kg-upsert-triples '() (kg-extract-triples '(\"Ada employed by Acme\" \"Ada founded Engine\" \"Ada employed by Acme\")))",
        "(equal? (kg-upsert-triples '() (kg-extract-triples '(\"Ada employed by Acme\" \"Ada founded Engine\" \"Ada employed by Acme\"))) '((ada (works_at acme) (founded engine))))",
        "hard",
        ["integration"],
    ),
    (
        "kg-upsert-triples",
        "Batch upsert parsed facts and return object list for (ada, works_at).",
        "(let* ([store (kg-upsert-triples '() (kg-extract-triples '(\"Ada works at Acme\" \"Ada works at Babbage Labs\")))] [preds (cdr (assq 'ada store))] [row (assq 'works_at preds)]) (cdr row))",
        "(equal? (let* ([store (kg-upsert-triples '() (kg-extract-triples '(\"Ada works at Acme\" \"Ada works at Babbage Labs\")))] [preds (cdr (assq 'ada store))] [row (assq 'works_at preds)]) (cdr row)) '(acme babbage-labs))",
        "hard",
        ["query-shape"],
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
