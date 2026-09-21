(import torch)
(import numpy :as np)
(import haply._backend [is-torch bool-zeros])
(import haply._dispatch [tprim tdyad flatten reshape])

;; --- G018 ≠ ---------------------------------------------------------------
;; Dyadic: elementwise != (exact, item 29). Monadic: first-occurrence mask
;; of raveled values, same shape as Y (decision 45 default A).

(defn unique-mask [y]
  (setv flat (flatten y)
        n (if (is-torch y) (.numel flat) flat.size))
  (if (= n 0)
    (bool-zeros y (list y.shape))
    (if (is-torch y)
      (do
        (setv pair (torch.unique flat :return-inverse True)
              inverse (get pair 1)
              idx (torch.arange n :device y.device)
              first (torch.full [(.numel (get pair 0))]
                                n
                                :dtype torch.long
                                :device y.device))
        (.scatter-reduce_ first 0 inverse idx "amin")
        (reshape (torch.eq (get first inverse) idx) y.shape))
      (do
        (setv pair (np.unique flat :return-inverse True)
              inverse (get pair 1)
              idx (np.arange n)
              first (np.full (len (get pair 0)) n :dtype np.int64))
        (np.minimum.at first inverse idx)
        (reshape (np.equal (get first inverse) idx) y.shape)))))

(setv ≠ (tprim "≠" unique-mask (fn [x y]
                                 (if (is-torch x) (torch.ne x y) (np.not-equal x y)))))

;; --- G019 ≡ ---------------------------------------------------------------
;; Dyadic match only. Monadic depth is dropped (decision 56 default A).

(defn match-dyad [x y]
  "Value match via the backend equal. NaN does not match NaN (item 56)."
  (if (is-torch x)
    (torch.equal x y)
    (np.array-equal x y)))

(setv ≡ (tdyad "≡" match-dyad))

;; --- G020 ≢ ---------------------------------------------------------------
;; Tally is a Python int (item 57): shape[0], or 1 if scalar. Not numel.

(defn tally-monad [y]
  (if (= y.ndim 0)
    1
    (get y.shape 0)))

(defn not-match-dyad [x y]
  (not (match-dyad x y)))

(setv ≢ (tprim "≢" tally-monad not-match-dyad))

(setv __all__ ["≠" "≡" "≢"])
