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

;; --- G039 ⍷ ---------------------------------------------------------------
;; Boolean mask, same shape as Y, True where X begins as a sub-array.
;; If rank(X) < rank(Y), X is left-padded with 1s. Rank(X) > rank(Y)
;; finds nothing. A 0-size axis in X finds nothing.

(defn find-dyad [x y]
  (when (> x.ndim y.ndim)
    (return (torch.zeros (list y.shape) :dtype torch.bool :device y.device)))
  (when (and (= x.ndim 0) (= y.ndim 0))
    (return (torch.eq x y)))
  (setv x (if (< x.ndim y.ndim)
            (torch.reshape x (+ (lfor _ (range (- y.ndim x.ndim)) 1)
                                (list x.shape)))
            x)
        out (torch.zeros (list y.shape) :dtype torch.bool :device y.device))
  (for [[xs ys] (zip x.shape y.shape)]
    (when (> xs ys)
      (return out)))
  (when (in 0 (list x.shape))
    (return out))
  (setv w y)
  (for [ax (range y.ndim)]
    (setv w (.unfold w ax (get x.shape ax) 1)))
  (setv hit (torch.eq w x))
  (for [_ (range y.ndim)]
    (setv hit (torch.all hit :dim -1)))
  (setv sl (tuple (lfor s hit.shape (slice None s))))
  (setv (get out sl) hit)
  out)

(setv ⍷ (tdyad "⍷" find-dyad))

(setv __all__ ["∪" "∩" "⍋" "⍒" "⍷"])
