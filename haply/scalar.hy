(import torch)
(import numpy :as np)
(import haply._backend [is-torch is-numpy is-complex is-floating is-bool is-integer])
(import haply._dispatch [tprim tdyad tfold u1 u2 flatten])

;; --- G001 + ---------------------------------------------------------------

(defn plus-monad [y]
  "Conjugate on complex, identity on reals (decision 31). Do not copy reals."
  (if (is-complex y)
    (if (is-torch y) (torch.conj y) (np.conjugate y))
    y))

(setv + (tfold "+" plus-monad (u2 torch.add np.add)))

;; --- G002 - ---------------------------------------------------------------

(setv - (tprim "-" (u1 torch.neg np.negative) (u2 torch.sub np.subtract)))

;; --- G003 × ---------------------------------------------------------------

(defn times-monad [y]
  (if (is-torch y)
    (torch.sgn y)
    (if (is-complex y)
      (do
        (setv a (np.abs y)
              out (np.zeros-like y))
        (np.divide y a :out out :where (!= a 0))
        out)
      (np.sign y))))

(setv × (tfold "×" times-monad (u2 torch.mul np.multiply)))

;; --- G004 ÷ ---------------------------------------------------------------

(defn rec-monad [y]
  (if (is-torch y)
    (torch.true-divide 1 y)
    (np.true-divide 1 y)))

(setv ÷ (tprim "÷" rec-monad (u2 torch.true-divide np.true-divide)))

;; --- G005 ** --------------------------------------------------------------
;; Hy's core ** macro requires two or more arguments, so (** Y) is a syntax
;; error. The function object still implements monadic exp.

(setv ** (tprim "**" (u1 torch.exp np.exp) (u2 torch.pow np.power)))

;; --- G006 ⍟ ---------------------------------------------------------------

(defn log-dyad [x y]
  (if (is-torch x)
    (torch.div (torch.log y) (torch.log x))
    (np.true-divide (np.log y) (np.log x))))

(setv ⍟ (tprim "⍟" (u1 torch.log np.log) log-dyad))

;; --- G007 || --------------------------------------------------------------
;; Residue is Dyalog X|Y: remainder of Y by X. remainder has the sign of
;; the divisor, which matches the Dyalog table.

(defn residue-dyad [x y]
  (if (is-torch x)
    (torch.remainder y x)
    (np.remainder y x)))

(setv || (tprim "||" (u1 torch.abs np.abs) residue-dyad))

;; --- G008 ⌊ / G009 ⌈ ------------------------------------------------------
;; Integers are already integral. floor / ceil on integer dtypes stay as-is.

(defn floor-monad [y]
  (if (is-floating y)
    (if (is-torch y) (torch.floor y) (np.floor y))
    y))

(defn ceil-monad [y]
  (if (is-floating y)
    (if (is-torch y) (torch.ceil y) (np.ceil y))
    y))

(setv ⌊ (tfold "⌊" floor-monad (u2 torch.minimum np.minimum)))
(setv ⌈ (tfold "⌈" ceil-monad (u2 torch.maximum np.maximum)))

;; --- G013–G017 comparisons ------------------------------------------------
;; Exact backend comparison (item 29 default A). Result dtype is the backend
;; boolean (item 35 default A). No monadic APL meaning.

(setv < (tdyad "<" (u2 torch.lt np.less)))
(setv ≤ (tdyad "≤" (u2 torch.le np.less-equal)))
(setv == (tdyad "==" (u2 torch.eq np.equal)))
(setv ≥ (tdyad "≥" (u2 torch.ge np.greater-equal)))
(setv > (tdyad ">" (u2 torch.gt np.greater)))

;; --- G021 ≁ ---------------------------------------------------------------
;; Monadic: logical not. Dyadic: values of X not in Y (ravel, stable).

(defn without-dyad [x y]
  (setv xf (flatten x))
  (if (is-torch x)
    (get xf (torch.logical-not (torch.isin xf y)))
    (get xf (np.logical-not (np.isin xf y)))))

(setv ≁ (tprim "≁" (u1 torch.logical-not np.logical-not) without-dyad))

;; --- G022 ∧ / G023 ∨ ------------------------------------------------------
;; Boolean AND/OR; integer LCM/GCD (decision 46). Floats are ValueError.
;; Python bool is an int subclass: check bool first.

(defn _kind [x]
  (cond
    (is-bool x) 'bool
    (is-floating x) 'float
    (is-complex x) 'complex
    (is-integer x) 'int
    (isinstance x bool) 'bool
    (isinstance x int) 'int
    True 'float))

(defn and-dyad [x y]
  (setv kx (_kind x) ky (_kind y))
  (cond
    (and (= kx 'bool) (= ky 'bool))
      (if (is-torch x) (torch.logical-and x y) (np.logical-and x y))
    (and (= kx 'int) (= ky 'int))
      (if (is-torch x) (torch.lcm x y) (np.lcm x y))
    True (raise (ValueError "∧ expects boolean or integer tensors"))))

(defn or-dyad [x y]
  (setv kx (_kind x) ky (_kind y))
  (cond
    (and (= kx 'bool) (= ky 'bool))
      (if (is-torch x) (torch.logical-or x y) (np.logical-or x y))
    (and (= kx 'int) (= ky 'int))
      (if (is-torch x) (torch.gcd x y) (np.gcd x y))
    True (raise (ValueError "∨ expects boolean or integer tensors"))))

(defn and-monad [y]
  (raise (TypeError "∧ takes 2 arguments, got 1")))

(defn or-monad [y]
  (raise (TypeError "∨ takes 2 arguments, got 1")))

(setv ∧ (tfold "∧" and-monad and-dyad))
(setv ∨ (tfold "∨" or-monad or-dyad))

;; --- G024 ⍲ / G025 ⍱ ------------------------------------------------------

(defn nand-dyad [x y]
  (if (is-torch x)
    (torch.logical-not (torch.logical-and x y))
    (np.logical-not (np.logical-and x y))))

(defn nor-dyad [x y]
  (if (is-torch x)
    (torch.logical-not (torch.logical-or x y))
    (np.logical-not (np.logical-or x y))))

(setv ⍲ (tdyad "⍲" nand-dyad))
(setv ⍱ (tdyad "⍱" nor-dyad))

(setv __all__ ["+" "-" "×" "÷" "**" "⍟" "||" "⌊" "⌈"
               "<" "≤" "==" "≥" ">" "≁" "∧" "∨" "⍲" "⍱"])
