(import torch)
(import haply._dispatch [tprim tdyad])

;; --- G018 ≠ ---------------------------------------------------------------
;; Dyadic: elementwise != (exact, item 29). Monadic: first-occurrence mask
;; of raveled values, same shape as Y (decision 45 default A).

(defn unique-mask [y]
  (setv flat (torch.flatten y)
        n (.numel flat))
  (if (= n 0)
    (torch.empty y.shape :dtype torch.bool :device y.device)
    (do
      (setv pair (torch.unique flat :return-inverse True)
            inverse (get pair 1)
            idx (torch.arange n :device y.device)
            first (torch.full [(.numel (get pair 0))]
                              n
                              :dtype torch.long
                              :device y.device))
      (.scatter-reduce_ first 0 inverse idx "amin")
      (torch.reshape (torch.eq (get first inverse) idx) y.shape))))

(setv ≠ (tprim "≠" unique-mask torch.ne))

;; --- G019 ≡ ---------------------------------------------------------------
;; Dyadic match only. Monadic depth is dropped (decision 56 default A).

(defn match-dyad [x y]
  (torch.equal x y))

(setv ≡ (tdyad "≡" match-dyad))

;; --- G020 ≢ ---------------------------------------------------------------
;; Tally is a Python int (item 57): shape[0], or 1 if scalar. Not numel.

(defn tally-monad [y]
  (if (= y.ndim 0)
    1
    (get y.shape 0)))

(defn not-match-dyad [x y]
  (not (torch.equal x y)))

(setv ≢ (tprim "≢" tally-monad not-match-dyad))

(setv __all__ ["≠" "≡" "≢"])
