(import torch)
(import numpy :as np)
(import haply._backend [is-torch is-numpy is-array backend nelem])
(import haply._dispatch [tprim])

(defn _prim [name monad dyad]
  "Constructors lift Python to torch (item 30). Array args keep their backend."
  (fn [#* args]
    (match (len args)
      1 (do
          (setv y (get args 0))
          (monad (if (is-array y) y (torch.as-tensor y))))
      2 (do
          (setv x (get args 0)
                y (get args 1)
                tx (is-torch x)
                ty (is-torch y)
                nx (is-numpy x)
                ny (is-numpy y))
          (cond
            (and (or tx nx) (or ty ny) (not (= (backend x) (backend y))))
              (raise (TypeError "haply mixed tensor backends"))
            (or nx ny)
              (dyad (if nx x (np.asarray x)) (if ny y (np.asarray y)))
            True
              (dyad (if tx x (torch.as-tensor x))
                    (if ty y (torch.as-tensor y)))))
      _ (raise (TypeError (.format "{} takes 1 or 2 arguments, got {}"
                                   name
                                   (len args)))))))

;; --- G010 ○ ---------------------------------------------------------------
;; Monadic: π * Y. Dyadic: working subset of the circular table (item 47 B).

(defn pi-monad [y]
  (if (is-torch y)
    (torch.mul y torch.pi)
    (np.multiply y np.pi)))

(defn _circ-code [n y]
  (if (is-torch y)
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
      _ (raise (ValueError "○ X is not a supported circular code")))
    (match n
      1 (np.sin y)
      2 (np.cos y)
      3 (np.tan y)
      -1 (np.arcsin y)
      -2 (np.arccos y)
      -3 (np.arctan y)
      5 (np.sinh y)
      6 (np.cosh y)
      7 (np.tanh y)
      _ (raise (ValueError "○ X is not a supported circular code")))))

(defn circ-dyad [x y]
  (when (> (nelem x) 1)
    (raise (ValueError "○ X must be a single circular code")))
  (setv raw (.item x)
        n (int raw))
  (when (!= raw n)
    (raise (ValueError "○ X must be an integer circular code")))
  (_circ-code n y))

(setv ○ (tprim "○" pi-monad circ-dyad))

;; --- G011 ! ---------------------------------------------------------------
;; gamma(Y+1); binomial via gammaln: X!Y is C(Y, X).
;; NumPy has no lgamma; the ndarray path uses the torch kernel and wraps
;; the result so the user's backend is preserved.

(defn _to-float-t [y]
  (if (or (.is-floating-point y) (.is-complex y))
    y
    (.to y torch.float64)))

(defn fact-monad-t [y]
  (setv z (torch.add (_to-float-t y) 1)
        mag (torch.exp (torch.lgamma z)))
  (if (.is-complex z)
    mag
    (torch.mul mag (torch.where (torch.gt z 0)
                                (torch.ones-like mag)
                                (torch.pow (torch.as-tensor -1.0 :dtype z.dtype :device z.device)
                                           (torch.floor z))))))

(defn binom-dyad-t [x y]
  (setv xf (_to-float-t x)
        yf (_to-float-t y))
  (torch.exp (torch.sub (torch.sub (torch.lgamma (torch.add yf 1))
                                   (torch.lgamma (torch.add xf 1)))
                        (torch.lgamma (torch.add (torch.sub yf xf) 1)))))

(defn fact-monad [y]
  (if (is-torch y)
    (fact-monad-t y)
    (np.asarray (fact-monad-t (torch.as-tensor y)))))

(defn binom-dyad [x y]
  (if (is-torch x)
    (binom-dyad-t x y)
    (np.asarray (binom-dyad-t (torch.as-tensor x) (torch.as-tensor y)))))

(setv ! (tprim "!" fact-monad binom-dyad))

;; --- G012 ? ---------------------------------------------------------------
;; Roll: each bound is an exclusive upper integer (0-based). Bound ≤ 0
;; is an error (no Dyalog ?0 float-in-(0,1) special case).
;; Deal: k distinct draws from [0, n). Host RNG follows the array backend.

(defn roll-monad [y]
  (if (is-torch y)
    (do
      (when (torch.any (torch.le y 0))
        (raise (ValueError "? roll bounds must be positive")))
      (setv u (torch.rand (list y.shape) :device y.device))
      (.to (torch.floor (torch.mul u (.to y torch.float64))) torch.int64))
    (do
      (when (np.any (np.less-equal y 0))
        (raise (ValueError "? roll bounds must be positive")))
          (setv u (np.random.random (tuple y.shape)))
          (.astype (np.floor (np.multiply u (np.asarray y :dtype np.float64)))
                   np.int64))))

(defn deal-dyad [x y]
  (when (or (> (nelem x) 1) (> (nelem y) 1))
    (raise (ValueError "? deal X and Y must be scalars")))
  (setv k (int (.item x))
        n (int (.item y)))
  (when (or (< k 0) (< n 0) (> k n))
    (raise (ValueError "? deal needs 0 ≤ X ≤ Y")))
  (if (is-numpy y)
    (get (np.random.permutation n) (slice None k))
    (get (torch.randperm n :device (if (is-torch y) y.device None)) (slice None k))))

(setv ? (_prim "?" roll-monad deal-dyad))

;; --- G045 ⌹ ---------------------------------------------------------------
;; Monad: inv if square 2-d, else pinv (0-d is reciprocal).
;; Dyad: (⌹ X Y) is Dyalog X⌹Y — solve Y B = X (Y⁻¹X when Y is square).

(defn domino-monad [y]
  (if (is-torch y)
    (cond
      (= y.ndim 0) (torch.reciprocal y)
      (= y.ndim 1) (torch.linalg.pinv y)
      (= y.ndim 2)
        (if (= (get y.shape 0) (get y.shape 1))
          (torch.linalg.inv y)
          (torch.linalg.pinv y))
      True (raise (ValueError "⌹ Y must have rank ≤ 2")))
    (cond
      (= y.ndim 0) (np.reciprocal y)
      (= y.ndim 1) (np.linalg.pinv y)
      (= y.ndim 2)
        (if (= (get y.shape 0) (get y.shape 1))
          (np.linalg.inv y)
          (np.linalg.pinv y))
      True (raise (ValueError "⌹ Y must have rank ≤ 2")))))

(defn domino-dyad [x y]
  (when (> x.ndim 2)
    (raise (ValueError "⌹ X must have rank ≤ 2")))
  (if (is-torch x)
    (do
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
    (do
      (setv a (cond
                (= y.ndim 0) (np.reshape y [1 1])
                (= y.ndim 1) (np.reshape y [y.size 1])
                (= y.ndim 2) y
                True (raise (ValueError "⌹ Y must have rank ≤ 2"))))
      (when (!= (get x.shape 0) (get a.shape 0))
        (raise (ValueError "⌹ X and Y must have the same number of rows")))
      (if (= (get a.shape 0) (get a.shape 1))
        (np.linalg.solve a x)
        (get (np.linalg.lstsq a x) 0)))))

(setv ⌹ (_prim "⌹" domino-monad domino-dyad))

;; --- G046 ⊤ / G047 ⊥ ------------------------------------------------------
;; Encode: digits of Y in mixed radix X. Result shape is shape(X)+shape(Y).
;; A leading 0 radix keeps the remaining value. Decode: Horner along
;; axis 0 of Y; a scalar X is a repeated radix. First vector radix is unused.

(defn encode-dyad [x y]
  (when (> x.ndim 1)
    (raise (ValueError "⊤ X must have rank 0 or 1")))
  (if (is-torch x)
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
        (torch.stack (list (reversed digits)))))
    (if (= x.ndim 0)
      (np.remainder y x)
      (do
        (setv cur y
              digits [])
        (for [i (range (- (get x.shape 0) 1) -1 -1)]
          (setv r (get x i))
          (if (= r 0)
            (do
              (.append digits cur)
              (setv cur (np.zeros-like cur)))
            (do
              (.append digits (np.remainder cur r))
              (setv cur (np.floor-divide cur r)))))
        (np.stack (list (reversed digits)))))))

(defn decode-dyad [x y]
  (when (> x.ndim 1)
    (raise (ValueError "⊥ X must have rank 0 or 1")))
  (when (= y.ndim 0)
    (raise (ValueError "⊥ Y needs rank ≥ 1")))
  (setv n (get y.shape 0)
        acc (get y 0))
  (when (= n 0)
    (raise (ValueError "⊥ Y first axis must be non-empty")))
  (if (is-torch y)
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
    (cond
      (= x.ndim 0)
        (for [i (range 1 n)]
          (setv acc (np.add (np.multiply acc x) (get y i))))
      True
        (do
          (when (!= (get x.shape 0) n)
            (raise (ValueError "⊥ X length must match Y's first axis")))
          (for [i (range 1 n)]
            (setv acc (np.add (np.multiply acc (get x i)) (get y i)))))))
  acc)

(setv ⊤ (_prim "⊤" (fn [y] (raise (TypeError "⊤ is dyadic only"))) encode-dyad))
(setv ⊥ (_prim "⊥" (fn [y] (raise (TypeError "⊥ is dyadic only"))) decode-dyad))

(setv __all__ ["○" "!" "?" "⌹" "⊤" "⊥"])
