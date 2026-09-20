(import torch)
(import haply._dispatch [tprim])

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

(setv __all__ ["○" "!"])
