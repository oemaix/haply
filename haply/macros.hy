;; Operator macros. Require this module; do not import it.
;; Known operands expand to torch kernels. Anything else calls a helper
;; in haply._dispatch (import injected so the caller need not bind haply).
(import hy.models [Symbol Integer])

(eval-and-compile
  (setv _REDUCE {"+" "sum"  "×" "prod"  "⌈" "amax"  "⌊" "amin"}
        _SCAN {"+" "cumsum"  "×" "cumprod"  "⌈" "cummax"  "⌊" "cummin"}
        _SCALAR #{"+" "-" "×" "÷" "**" "⍟" "||" "⌊" "⌈"
                  "<" "≤" "==" "≥" ">" "≁" "∧" "∨" "⍲" "⍱" "○" "!"}
        _FN (set.union _SCALAR #{"⌽" "⊖" "⍴" "++"}))

  (defn _sym-str [x]
    (if (isinstance x Symbol) (str x) None))

  (defn _int-lit [x]
    (if (isinstance x Integer) (int x) None))

  (defn _dispatch-call [fn-sym #* args]
    `(do
       (import haply._dispatch [~fn-sym])
       (~fn-sym ~@args)))

  (defn _reduce-kernel [axis op y]
    (setv name (get _REDUCE (_sym-str op)))
    (cond
      (= name "sum") `(torch.sum ~y :dim ~axis)
      (= name "prod") `(torch.prod ~y :dim ~axis)
      (= name "amax") `(torch.amax ~y :dim ~axis)
      True `(torch.amin ~y :dim ~axis)))

  (defn _scan-kernel [axis op y]
    (setv name (get _SCAN (_sym-str op)))
    (cond
      (= name "cumsum") `(torch.cumsum ~y :dim ~axis)
      (= name "cumprod") `(torch.cumprod ~y :dim ~axis)
      (= name "cummax") `(.values (torch.cummax ~y ~axis))
      True `(.values (torch.cummin ~y ~axis))))

  (defn _nwise-kernel [axis k op y]
    (setv kn (get _REDUCE (_sym-str op))
          w `(.unfold ~y ~axis ~k 1))
    (cond
      (= kn "sum") `(torch.sum ~w :dim -1)
      (= kn "prod") `(torch.prod ~w :dim -1)
      (= kn "amax") `(torch.amax ~w :dim -1)
      True `(torch.amin ~w :dim -1)))

  (defn _slash [axis forms]
    (setv n (len forms))
    (when (< n 2)
      (raise (TypeError "⌿ needs an operand and an argument")))
    (setv k (_int-lit (get forms 0)))
    (when (and (is-not k None) (>= n 3))
      (setv op (get forms 1)
            y (get forms 2))
      (return (if (in (_sym-str op) _REDUCE)
                (_nwise-kernel axis k op y)
                (_dispatch-call 'nwise-reduce axis k op y))))
    (setv op (get forms 0)
          y (get forms 1))
    (if (in (_sym-str op) _REDUCE)
      (_reduce-kernel axis op y)
      (_dispatch-call 'reduce-or-replicate axis op y)))

  (defn _backslash [axis forms]
    (when (!= (len forms) 2)
      (raise (TypeError "⍀ needs an operand and an argument")))
    (setv op (get forms 0)
          y (get forms 1))
    (if (in (_sym-str op) _SCAN)
      (_scan-kernel axis op y)
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
    `(torch.matmul ~x ~y)
    (_dispatch-call 'inner-product f g x y)))

;; Hy cannot parse the catalog name ∘. (item 61). Writable name: outer.
(defmacro outer [g x y]
  (setv s (_sym-str g)
        left (_dispatch-call 'outer-left x y))
  (cond
    (= s "×") `(torch.mul ~left ~y)
    (= s "+") `(torch.add ~left ~y)
    True `(~g ~left ~y)))

(defmacro ¨ [f #* args]
  (if (in (_sym-str f) _SCALAR)
    `(~f ~@args)
    (_dispatch-call 'each-cells f #* args)))
