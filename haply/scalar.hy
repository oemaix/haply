(import torch)
(import haply._dispatch [tprim tmonad tdyad tfold])

;; --- G001 + ---------------------------------------------------------------

(defn plus-monad [y]
  "Conjugate on complex, identity on reals (decision 31). Do not copy reals."
  (if (.is-complex y) (torch.conj y) y))

(defn plus-dyad [x y]
  (torch.add x y))

(setv + (tfold "+" plus-monad plus-dyad))

;; --- G002 - ---------------------------------------------------------------

(setv - (tprim "-" torch.neg torch.sub))

;; --- G003 × ---------------------------------------------------------------

(setv × (tfold "×" torch.sgn torch.mul))

;; --- G004 ÷ ---------------------------------------------------------------

(defn rec-monad [y]
  (torch.true-divide 1 y))

(setv ÷ (tprim "÷" rec-monad torch.true-divide))

;; --- G005 ** --------------------------------------------------------------
;; Hy's core ** macro requires two or more arguments, so (** Y) is a syntax
;; error. The function object still implements monadic exp.

(setv ** (tprim "**" torch.exp torch.pow))

;; --- G006 ⍟ ---------------------------------------------------------------

(defn log-dyad [x y]
  (torch.div (torch.log y) (torch.log x)))

(setv ⍟ (tprim "⍟" torch.log log-dyad))

;; --- G007 || --------------------------------------------------------------
;; Residue is Dyalog X|Y: remainder of Y by X. torch.remainder(Y, X) has the
;; sign of the divisor, which matches the Dyalog table.

(defn residue-dyad [x y]
  (torch.remainder y x))

(setv || (tprim "||" torch.abs residue-dyad))

;; --- G008 ⌊ / G009 ⌈ ------------------------------------------------------
;; Integers are already integral. torch.floor / ceil reject integer dtypes.

(defn floor-monad [y]
  (if (.is-floating-point y) (torch.floor y) y))

(defn ceil-monad [y]
  (if (.is-floating-point y) (torch.ceil y) y))

(setv ⌊ (tfold "⌊" floor-monad torch.minimum))
(setv ⌈ (tfold "⌈" ceil-monad torch.maximum))

;; --- G013–G017 comparisons ------------------------------------------------
;; Exact backend comparison (item 29 default A). Result dtype is the backend
;; boolean (item 35 default A). No monadic APL meaning.

(setv < (tdyad "<" torch.lt))
(setv ≤ (tdyad "≤" torch.le))
(setv == (tdyad "==" torch.eq))
(setv ≥ (tdyad "≥" torch.ge))
(setv > (tdyad ">" torch.gt))

;; --- G021 ≁ ---------------------------------------------------------------
;; Monadic logical not is Phase 1. Dyadic without waits for Phase 3.

(setv ≁ (tmonad "≁" torch.logical-not))

;; --- G022 ∧ / G023 ∨ ------------------------------------------------------
;; Boolean AND/OR; integer LCM/GCD (decision 46). Floats are ValueError.
;; Python bool is an int subclass: check bool first.

(defn _kind [x]
  (if (isinstance x torch.Tensor)
    (cond
      (= x.dtype torch.bool) 'bool
      (.is-floating-point x) 'float
      (.is-complex x) 'complex
      True 'int)
    (cond
      (isinstance x bool) 'bool
      (isinstance x int) 'int
      True 'float)))

(defn and-dyad [x y]
  (setv kx (_kind x) ky (_kind y))
  (cond
    (and (= kx 'bool) (= ky 'bool)) (torch.logical-and x y)
    (and (= kx 'int) (= ky 'int)) (torch.lcm x y)
    True (raise (ValueError "∧ expects boolean or integer tensors"))))

(defn or-dyad [x y]
  (setv kx (_kind x) ky (_kind y))
  (cond
    (and (= kx 'bool) (= ky 'bool)) (torch.logical-or x y)
    (and (= kx 'int) (= ky 'int)) (torch.gcd x y)
    True (raise (ValueError "∨ expects boolean or integer tensors"))))

(defn and-monad [y]
  (raise (TypeError "∧ takes 2 arguments, got 1")))

(defn or-monad [y]
  (raise (TypeError "∨ takes 2 arguments, got 1")))

(setv ∧ (tfold "∧" and-monad and-dyad))
(setv ∨ (tfold "∨" or-monad or-dyad))

;; --- G024 ⍲ / G025 ⍱ ------------------------------------------------------

(defn nand-dyad [x y]
  (torch.logical-not (torch.logical-and x y)))

(defn nor-dyad [x y]
  (torch.logical-not (torch.logical-or x y)))

(setv ⍲ (tdyad "⍲" nand-dyad))
(setv ⍱ (tdyad "⍱" nor-dyad))

(setv __all__ ["+" "-" "×" "÷" "**" "⍟" "||" "⌊" "⌈"
               "<" "≤" "==" "≥" ">" "≁" "∧" "∨" "⍲" "⍱"])
