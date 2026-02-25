(load "core/lang/module.ss")
(unless (equal? (let ()
  (define (egraph-uf eg) (vector-ref eg 1))
  (define (egraph-hashcons eg) (vector-ref eg 3))
  (define (egraph-classes eg) (vector-ref eg 2))
  (define (egraph-stats eg) (vector-ref eg 5))
  (define (egraph-inc-stat! eg idx)
  (let ([stats (egraph-stats eg)])
    (vector-set! stats idx (+ (vector-ref stats idx) 1))))
  (define (egraph-find eg id)
  (uf-find (egraph-uf eg) id))
  (define (egraph-add-enode! eg enode)
  (let* ([uf (egraph-uf eg)]
         [canonical (enode-canonicalize enode uf)]
         [hashcons (egraph-hashcons eg)]
         [existing (hashtable-ref hashcons canonical #f)])
    (if existing
        (begin
          (egraph-inc-stat! eg 3)
          (uf-find uf existing))
        (let* ([classes (egraph-classes eg)]
               [new-id (uf-make-set! uf)])
          (egraph-inc-stat! eg 0)
          (hashtable-set! hashcons canonical new-id)
          (eclass-add-node! classes new-id canonical)
          (let ([children (enode-children canonical)])
            (do ([i 0 (+ i 1)])
                ((>= i (vector-length children)))
              (let ([child-id (egraph-find eg (vector-ref children i))])
                (eclass-add-parent! classes child-id new-id))))
          new-id))))
  (define (egraph-stat-adds eg) (vector-ref (egraph-stats eg) 0))
  (define (egraph-stat-hits eg) (vector-ref (egraph-stats eg) 3))
  (define (egraph-class-count eg)
  (uf-count (egraph-uf eg)))
  (define egraph-tag 'egraph)
  (define (make-egraph)
  (vector egraph-tag
          (make-uf)
          (make-eclass-store)
          (make-hashtable enode-hash enode-equal?)
          hamt-empty
          (vector 0 0 0 0)))
  (define (egraph-add-term! eg term)
  (cond
    [(pair? term)
     (let* ([op (car term)]
            [args (cdr term)]
            [child-ids (map (lambda (arg) (egraph-add-term! eg arg)) args)]
            [enode (make-enode op (list->vector child-ids))])
       (egraph-add-enode! eg enode))]
    [else
     (egraph-add-enode! eg (make-enode term (vector)))]))
  (load "lattice/egraph/union-find.ss")
  (load "lattice/egraph/eclass.ss")
  (and
  (let ([eg (make-egraph)]
        [n (make-enode 'x (vector))])
    (let ([id1 (egraph-add-enode! eg n)]
          [id2 (egraph-add-enode! eg n)])
      (and (= id1 id2)
           (= (egraph-class-count eg) 1)
           (= (egraph-stat-adds eg) 1)
           (= (egraph-stat-hits eg) 1))))
  (let ([eg (make-egraph)])
    (let ([x (egraph-add-term! eg 'x)])
      (let ([parent-id (egraph-add-enode! eg (make-enode 'f (vector x)))]
            [parents (eclass-get-parents (egraph-classes eg) x)])
        (and (not (null? parents))
             (not (not (memv parent-id parents))))))))) #t) (begin (display "FAIL") (newline) (write (let ()
  (define (egraph-uf eg) (vector-ref eg 1))
  (define (egraph-hashcons eg) (vector-ref eg 3))
  (define (egraph-classes eg) (vector-ref eg 2))
  (define (egraph-stats eg) (vector-ref eg 5))
  (define (egraph-inc-stat! eg idx)
  (let ([stats (egraph-stats eg)])
    (vector-set! stats idx (+ (vector-ref stats idx) 1))))
  (define (egraph-find eg id)
  (uf-find (egraph-uf eg) id))
  (define (egraph-add-enode! eg enode)
  (let* ([uf (egraph-uf eg)]
         [canonical (enode-canonicalize enode uf)]
         [hashcons (egraph-hashcons eg)]
         [existing (hashtable-ref hashcons canonical #f)])
    (if existing
        (begin
          (egraph-inc-stat! eg 3)
          (uf-find uf existing))
        (let* ([classes (egraph-classes eg)]
               [new-id (uf-make-set! uf)])
          (egraph-inc-stat! eg 0)
          (hashtable-set! hashcons canonical new-id)
          (eclass-add-node! classes new-id canonical)
          (let ([children (enode-children canonical)])
            (do ([i 0 (+ i 1)])
                ((>= i (vector-length children)))
              (let ([child-id (egraph-find eg (vector-ref children i))])
                (eclass-add-parent! classes child-id new-id))))
          new-id))))
  (define (egraph-stat-adds eg) (vector-ref (egraph-stats eg) 0))
  (define (egraph-stat-hits eg) (vector-ref (egraph-stats eg) 3))
  (define (egraph-class-count eg)
  (uf-count (egraph-uf eg)))
  (define egraph-tag 'egraph)
  (define (make-egraph)
  (vector egraph-tag
          (make-uf)
          (make-eclass-store)
          (make-hashtable enode-hash enode-equal?)
          hamt-empty
          (vector 0 0 0 0)))
  (define (egraph-add-term! eg term)
  (cond
    [(pair? term)
     (let* ([op (car term)]
            [args (cdr term)]
            [child-ids (map (lambda (arg) (egraph-add-term! eg arg)) args)]
            [enode (make-enode op (list->vector child-ids))])
       (egraph-add-enode! eg enode))]
    [else
     (egraph-add-enode! eg (make-enode term (vector)))]))
  (load "lattice/egraph/union-find.ss")
  (load "lattice/egraph/eclass.ss")
  (and
  (let ([eg (make-egraph)]
        [n (make-enode 'x (vector))])
    (let ([id1 (egraph-add-enode! eg n)]
          [id2 (egraph-add-enode! eg n)])
      (and (= id1 id2)
           (= (egraph-class-count eg) 1)
           (= (egraph-stat-adds eg) 1)
           (= (egraph-stat-hits eg) 1))))
  (let ([eg (make-egraph)])
    (let ([x (egraph-add-term! eg 'x)])
      (let ([parent-id (egraph-add-enode! eg (make-enode 'f (vector x)))]
            [parents (eclass-get-parents (egraph-classes eg) x)])
        (and (not (null? parents))
             (not (not (memv parent-id parents))))))))))(newline) (exit 1)))
(display "OK\n")
