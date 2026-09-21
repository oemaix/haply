;; Operator macros. Require this module; do not import it.
;; Known operands expand to a helper in haply._dispatch that picks the
;; torch or NumPy kernel at runtime. Anything else uses the cell loop.
(import hy.models [Symbol Integer Expression])

(eval-and-compile
  (setv _REDUCE {"+" "sum"  "×" "prod"  "⌈" "amax"  "⌊" "amin"}
        _SCAN {"+" "cumsum"  "×" "cumprod"  "⌈" "cummax"  "⌊" "cummin"}
        _SCALAR #{"+" "-" "×" "÷" "**" "⍟" "||" "⌊" "⌈"
                  "<" "≤" "==" "≥" ">" "≁" "∧" "∨" "⍲" "⍱" "○" "!"}
        _FN (set.union _SCALAR #{"⌽" "⊖" "⍴" "++" "⊢" "⊣" "↑" "↓" "⍉"
                                 "≠" "≡" "≢" "⍪"}))

  (defn _sym-str [x]
    (if (isinstance x Symbol) (str x) None))

  (defn _int-lit [x]
    (if (isinstance x Integer) (int x) None))

  (defn _known-fn? [x]
    (setv s (_sym-str x))
    (or (and s (in s _FN))
        (and (isinstance x Expression)
             (isinstance (get x 0) Symbol)
             (= (str (get x 0)) "."))))

  (defn _dispatch-call [fn-sym #* args]
    `(do
       (import haply._dispatch [~fn-sym])
       (~fn-sym ~@args)))

  (defn _slash [axis forms]
    (setv n (len forms))
    (when (< n 2)
      (raise (TypeError "⌿ needs an operand and an argument")))
    (setv k (_int-lit (get forms 0)))
    (when (and (is-not k None) (>= n 3))
      (setv op (get forms 1)
            y (get forms 2)
            kn (.get _REDUCE (_sym-str op)))
      (return (if kn
                (_dispatch-call 'nwise-known axis k kn y)
                (_dispatch-call 'nwise-reduce axis k op y))))
    (setv op (get forms 0)
          y (get forms 1)
          kn (.get _REDUCE (_sym-str op)))
    (if kn
      (_dispatch-call 'reduce-known axis kn y)
      (_dispatch-call 'reduce-or-replicate axis op y)))

  (defn _backslash [axis forms]
    (when (!= (len forms) 2)
      (raise (TypeError "⍀ needs an operand and an argument")))
    (setv op (get forms 0)
          y (get forms 1)
          kn (.get _SCAN (_sym-str op)))
    (if kn
      (_dispatch-call 'scan-known axis kn y)
      (_dispatch-call 'scan-or-expand axis op y))))

(defmacro ⌿ [#* forms]
  (_slash 0 forms))

(defmacro ⌿_ [#* forms]
  (_slash -1 forms))

(defmacro ⍀ [#* forms]
  (_backslash 0 forms))

(defmacro ⍀_ [#* forms]
  (_backslash -1 forms))

(defmacro ⍨ [op #* args]
  (setv n (len args)
        s (_sym-str op))
  (cond
    (and (is-not s None) (= n 1) (in s _FN))
      `(~op ~(get args 0) ~(get args 0))
    (and (is-not s None) (= n 2) (in s _FN))
      `(~op ~(get args 1) ~(get args 0))
    (or (= n 1) (= n 2))
      (_dispatch-call 'commute op #* args)
    True
      (raise (TypeError "⍨ takes 2 or 3 forms"))))

(defmacro · [f g x y]
  (if (and (= (_sym-str f) "+") (= (_sym-str g) "×"))
    (_dispatch-call 'matmul-known x y)
    (_dispatch-call 'inner-product f g x y)))

;; Hy cannot parse Dyalog ∘. (ASCII dot). Writable name: ∘· (decision 61).
(defmacro ∘· [g x y]
  (setv s (_sym-str g))
  (cond
    (= s "×") (_dispatch-call 'outer-mul x y)
    (= s "+") (_dispatch-call 'outer-add x y)
    True `(~g ~(_dispatch-call 'outer-left x y) ~y)))

(defmacro ¨ [f #* args]
  (if (in (_sym-str f) _SCALAR)
    `(~f ~@args)
    (_dispatch-call 'each-cells f #* args)))

;; --- Phase 5 composition --------------------------------------------------

(defmacro ∘ [#* forms]
  (setv n (len forms))
  (cond
    (= n 3)
      (do
        (setv a (get forms 0)
              b (get forms 1)
              y (get forms 2))
        (if (and (_known-fn? a) (_known-fn? b))
          `(~a (~b ~y))
          (_dispatch-call 'beside-or-bind a b y)))
    (= n 4)
      `(~(get forms 0) ~(get forms 2) (~(get forms 1) ~(get forms 3)))
    True
      `(raise (TypeError "∘ takes 3 or 4 forms"))))

(defmacro ⍤ [#* forms]
  (setv n (len forms))
  (cond
    (and (>= n 2) (is-not (_int-lit (get forms 1)) None))
      `(raise (TypeError "⍤ rank (array operand) waits on item 48"))
    (= n 3)
      `(~(get forms 0) (~(get forms 1) ~(get forms 2)))
    (= n 4)
      `(~(get forms 0) (~(get forms 1) ~(get forms 2) ~(get forms 3)))
    True
      `(raise (TypeError "⍤ takes 3 or 4 forms"))))

(defmacro ⍥ [#* forms]
  (setv n (len forms))
  (cond
    (= n 3)
      `(~(get forms 0) (~(get forms 1) ~(get forms 2)))
    (= n 4)
      `(~(get forms 0) (~(get forms 1) ~(get forms 2))
                       (~(get forms 1) ~(get forms 3)))
    True
      `(raise (TypeError "⍥ takes 3 or 4 forms"))))

(defmacro ⍛ [#* forms]
  ;; Haply (not Dyalog): monad is (f (g Y)); dyad is ((g X) f Y).
  (setv n (len forms))
  (cond
    (= n 3)
      `(~(get forms 0) (~(get forms 1) ~(get forms 2)))
    (= n 4)
      `(~(get forms 0) (~(get forms 1) ~(get forms 2)) ~(get forms 3))
    True
      `(raise (TypeError "⍛ takes 3 or 4 forms"))))
