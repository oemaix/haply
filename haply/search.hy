(import torch)
(import haply._backend [require-torch])
(import haply._dispatch [tprim tmonad tdyad])
(import haply.structural [++])

;; --- G040 ∪ ---------------------------------------------------------------
;; Unique values in ravel order of first occurrence. Dyadic: unique of
;; catenated ravels.

(defn _unique-stable [flat]
  (if (= (.numel flat) 0)
    flat
    (do
      (setv pair (torch.unique flat :return-inverse True)
            uniques (get pair 0)
            inverse (get pair 1)
            n (.numel flat)
            first (torch.full [(.numel uniques)] n :dtype torch.long :device flat.device)
            idx (torch.arange n :device flat.device))
      (.scatter-reduce_ first 0 inverse idx "amin")
      (get uniques (torch.argsort first)))))

(defn unique-monad [y]
  (_unique-stable (torch.flatten y)))

(defn union-dyad [x y]
  (_unique-stable (torch.cat [(torch.flatten x) (torch.flatten y)])))

(setv ∪ (tprim "∪" unique-monad union-dyad))

;; --- G041 ∩ ---------------------------------------------------------------

(defn intersect-dyad [x y]
  (setv xf (torch.flatten x))
  (get xf (torch.isin xf y)))

(setv ∩ (tdyad "∩" intersect-dyad))

;; --- G042 ⍋ / G043 ⍒ ------------------------------------------------------
;; Numeric only. Higher rank grades major cells lexicographically.
;; Dyadic collation is dropped (decision 32).

(defn _need-numeric [name y]
  (when (.is-complex y)
    (raise (ValueError (.format "{} is numeric only" name)))))

(defn _grade [y descending]
  (cond
    (= y.ndim 0)
      (torch.tensor [0] :dtype torch.int64 :device y.device)
    (= y.ndim 1)
      (torch.argsort y :descending descending :stable True)
    True
      (do
        (setv keys (torch.flatten y 1)
              n (get keys.shape 0)
              idx (torch.arange n :device y.device))
        (for [c (range (- (get keys.shape 1) 1) -1 -1)]
          (setv col (.select (get keys idx) 1 c))
          (setv idx (get idx (torch.argsort col :descending descending :stable True))))
        idx)))

(defn grade-up [y]
  (_need-numeric "⍋" y)
  (_grade y False))

(defn grade-down [y]
  (_need-numeric "⍒" y)
  (_grade y True))

(setv ⍋ (tmonad "⍋" grade-up))
(setv ⍒ (tmonad "⍒" grade-down))

(setv __all__ ["∪" "∩" "⍋" "⍒"])
