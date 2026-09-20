;; Named 3-train. Require this module; do not import it.
(import hy.models [Symbol Integer Expression])

(eval-and-compile
  (setv _FN #{"+" "-" "×" "÷" "**" "⍟" "||" "⌊" "⌈"
              "<" "≤" "==" "≥" ">" "≁" "∧" "∨" "⍲" "⍱" "○" "!"
              "⌽" "⊖" "⍴" "++" "⊢" "⊣" "↑" "↓" "⍉" "≠" "≡" "≢" "⍪"})

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

  (defn _wing [op #* args]
    (cond
      (_known-fn? op) `(~op ~@args)
      (is-not (_int-lit op) None) op
      True (_dispatch-call 'fork-wing op #* args))))

(defmacro fork [f g h #* rest]
  (setv n (len rest))
  (cond
    (= n 1)
      `(~g ~(_wing f (get rest 0)) ~(_wing h (get rest 0)))
    (= n 2)
      `(~g ~(_wing f (get rest 0) (get rest 1))
           ~(_wing h (get rest 0) (get rest 1)))
    True
      '(raise (TypeError "fork is the 3-train only"))))
