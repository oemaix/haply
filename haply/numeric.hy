(import torch)
(import haply._backend [is-torch])
(import haply._dispatch [tprim])

(defn _as-t [x]
  (if (is-torch x) x (torch.as-tensor x)))

(defn _prim [name monad dyad]
  (fn [#* args]
    (match (len args)
      1 (monad (_as-t (get args 0)))
      2 (dyad (_as-t (get args 0)) (_as-t (get args 1)))
      _ (raise (TypeError (.format "{} takes 1 or 2 arguments, got {}"
                                   name
                                   (len args)))))))

;; --- G010 ○ ---------------------------------------------------------------
;; Monadic: π * Y. Dyadic: working subset of the circular table (item 47 B).

(defn pi-monad [y]
  (torch.mul y torch.pi))

(defn _circ-code [n y]
  (match n
    1 (torch.sin y)
    2 (torch.cos y)
    3 (torch.tan y)
    -1 (torch.asin y)
    -2 (torch.acos y)
    -3 (torch.atan y)
    5 (torch.sinh y)
    6 (torch.cosh y)
    7 (torch.tanh y)
    _ (raise (ValueError "○ X is not a supported circular code"))))

(defn circ-dyad [x y]
  (when (> (.numel x) 1)
    (raise (ValueError "○ X must be a single circular code")))
  (setv raw (.item x)
        n (int raw))
  (when (!= raw n)
    (raise (ValueError "○ X must be an integer circular code")))
  (_circ-code n y))

(setv ○ (tprim "○" pi-monad circ-dyad))

;; --- G011 ! ---------------------------------------------------------------
;; gamma(Y+1); binomial via gammaln: X!Y is C(Y, X).

(defn _to-float [y]
  (if (or (.is-floating-point y) (.is-complex y))
    y
    (.to y torch.float64)))

(defn fact-monad [y]
  ;; torch.lgamma is log|Γ|; restore the sign of Γ(Y+1) on the real line.
  (setv z (torch.add (_to-float y) 1)
        mag (torch.exp (torch.lgamma z)))
  (if (.is-complex z)
    mag
    (torch.mul mag (torch.where (torch.gt z 0)
                                (torch.ones-like mag)
                                (torch.pow (torch.as-tensor -1.0 :dtype z.dtype :device z.device)
                                           (torch.floor z))))))

(defn binom-dyad [x y]
  (setv xf (_to-float x)
        yf (_to-float y))
  (torch.exp (torch.sub (torch.sub (torch.lgamma (torch.add yf 1))
                                   (torch.lgamma (torch.add xf 1)))
                        (torch.lgamma (torch.add (torch.sub yf xf) 1)))))

(setv ! (tprim "!" fact-monad binom-dyad))

;; --- G012 ? ---------------------------------------------------------------
;; Roll: each bound is an exclusive upper integer (0-based). Bound ≤ 0
;; is an error (no Dyalog ?0 float-in-(0,1) special case).
;; Deal: k distinct draws from [0, n). Host RNG is torch.

(defn roll-monad [y]
  (when (torch.any (torch.le y 0))
    (raise (ValueError "? roll bounds must be positive")))
  (setv u (torch.rand (list y.shape) :device y.device))
  (.to (torch.floor (torch.mul u (.to y torch.float64))) torch.int64))

(defn deal-dyad [x y]
  (when (or (> (.numel x) 1) (> (.numel y) 1))
    (raise (ValueError "? deal X and Y must be scalars")))
  (setv k (int (.item x))
        n (int (.item y)))
  (when (or (< k 0) (< n 0) (> k n))
    (raise (ValueError "? deal needs 0 ≤ X ≤ Y")))
  (get (torch.randperm n :device y.device) (slice None k)))

(setv ? (_prim "?" roll-monad deal-dyad))

;; --- G045 ⌹ ---------------------------------------------------------------
;; Monad: inv if square 2-d, else pinv (0-d is reciprocal).
;; Dyad: (⌹ X Y) is Dyalog X⌹Y — solve Y B = X (Y⁻¹X when Y is square).

(defn domino-monad [y]
  (cond
    (= y.ndim 0)
      (torch.reciprocal y)
    (= y.ndim 1)
      (torch.linalg.pinv y)
    (= y.ndim 2)
      (if (= (get y.shape 0) (get y.shape 1))
        (torch.linalg.inv y)
        (torch.linalg.pinv y))
    True
      (raise (ValueError "⌹ Y must have rank ≤ 2"))))

(defn domino-dyad [x y]
  (when (> x.ndim 2)
    (raise (ValueError "⌹ X must have rank ≤ 2")))
  (setv a (cond
            (= y.ndim 0) (torch.reshape y [1 1])
            (= y.ndim 1) (torch.reshape y [(.numel y) 1])
            (= y.ndim 2) y
            True (raise (ValueError "⌹ Y must have rank ≤ 2"))))
  (when (!= (get x.shape 0) (get a.shape 0))
    (raise (ValueError "⌹ X and Y must have the same number of rows")))
  (if (= (get a.shape 0) (get a.shape 1))
    (torch.linalg.solve a x)
    (.solution (torch.linalg.lstsq a x))))

(setv ⌹ (_prim "⌹" domino-monad domino-dyad))

;; --- G046 ⊤ / G047 ⊥ ------------------------------------------------------
;; Encode: digits of Y in mixed radix X. Result shape is shape(X)+shape(Y).
;; A leading 0 radix keeps the remaining value. Decode: Horner along
;; axis 0 of Y; a scalar X is a repeated radix. First vector radix is unused.

(defn encode-dyad [x y]
  (when (> x.ndim 1)
    (raise (ValueError "⊤ X must have rank 0 or 1")))
  (if (= x.ndim 0)
    (torch.remainder y x)
    (do
      (setv cur y
            digits [])
      (for [i (range (- (get x.shape 0) 1) -1 -1)]
        (setv r (get x i))
        (if (torch.eq r 0)
          (do
            (.append digits cur)
            (setv cur (torch.zeros-like cur)))
          (do
            (.append digits (torch.remainder cur r))
            (setv cur (torch.div cur r :rounding-mode "floor")))))
      (torch.stack (list (reversed digits))))))

(defn decode-dyad [x y]
  (when (> x.ndim 1)
    (raise (ValueError "⊥ X must have rank 0 or 1")))
  (when (= y.ndim 0)
    (raise (ValueError "⊥ Y needs rank ≥ 1")))
  (setv n (get y.shape 0)
        acc (get y 0))
  (when (= n 0)
    (raise (ValueError "⊥ Y first axis must be non-empty")))
  (cond
    (= x.ndim 0)
      (for [i (range 1 n)]
        (setv acc (torch.add (torch.mul acc x) (get y i))))
    True
      (do
        (when (!= (get x.shape 0) n)
          (raise (ValueError "⊥ X length must match Y's first axis")))
        (for [i (range 1 n)]
          (setv acc (torch.add (torch.mul acc (get x i)) (get y i))))))
  acc)

(setv ⊤ (_prim "⊤" (fn [y] (raise (TypeError "⊤ is dyadic only"))) encode-dyad))
(setv ⊥ (_prim "⊥" (fn [y] (raise (TypeError "⊥ is dyadic only"))) decode-dyad))

(setv __all__ ["○" "!" "?" "⌹" "⊤" "⊥"])
