(import torch)
(import numpy :as np)
(import haply._backend [is-torch is-complex bool-zeros])
(import haply._dispatch [tprim tmonad tdyad flatten reshape])

;; --- G040 ∪ ---------------------------------------------------------------
;; Unique values in ravel order of first occurrence. Dyadic: unique of
;; catenated ravels.

(defn _unique-stable [flat]
  (setv n (if (is-torch flat) (.numel flat) flat.size))
  (if (= n 0)
    flat
    (if (is-torch flat)
      (do
        (setv pair (torch.unique flat :return-inverse True)
              uniques (get pair 0)
              inverse (get pair 1)
              first (torch.full [(.numel uniques)] n :dtype torch.long :device flat.device)
              idx (torch.arange n :device flat.device))
        (.scatter-reduce_ first 0 inverse idx "amin")
        (get uniques (torch.argsort first)))
      (do
        (setv pair (np.unique flat :return-inverse True)
              uniques (get pair 0)
              inverse (get pair 1)
              first (np.full (len uniques) n :dtype np.int64)
              idx (np.arange n))
        (np.minimum.at first inverse idx)
        (get uniques (np.argsort first))))))

(defn unique-monad [y]
  (_unique-stable (flatten y)))

(defn union-dyad [x y]
  (if (is-torch x)
    (_unique-stable (torch.cat [(torch.flatten x) (torch.flatten y)]))
    (_unique-stable (np.concatenate [(np.reshape x -1) (np.reshape y -1)]))))

(setv ∪ (tprim "∪" unique-monad union-dyad))

;; --- G041 ∩ ---------------------------------------------------------------

(defn intersect-dyad [x y]
  (setv xf (flatten x))
  (if (is-torch x)
    (get xf (torch.isin xf y))
    (get xf (np.isin xf y))))

(setv ∩ (tdyad "∩" intersect-dyad))

;; --- G042 ⍋ / G043 ⍒ ------------------------------------------------------
;; Numeric only. Higher rank grades major cells lexicographically.
;; Dyadic collation is dropped (decision 32).

(defn _need-numeric [name y]
  (when (is-complex y)
    (raise (ValueError (.format "{} is numeric only" name)))))

(defn _argsort-desc [col]
  "Stable descending argsort. Rank-then-negate so int64 min does not overflow."
  (setv inv (get (np.unique col :return-inverse True) 1))
  (np.argsort (- inv) :kind "stable"))

(defn _grade [y descending]
  (cond
    (= y.ndim 0)
      (if (is-torch y)
        (torch.tensor [0] :dtype torch.int64 :device y.device)
        (np.array [0] :dtype np.int64))
    (= y.ndim 1)
      (if (is-torch y)
        (torch.argsort y :descending descending :stable True)
        (if descending
          (_argsort-desc y)
          (np.argsort y :kind "stable")))
    True
      (if (is-torch y)
        (do
          (setv keys (torch.flatten y 1)
                n (get keys.shape 0)
                idx (torch.arange n :device y.device))
          (for [c (range (- (get keys.shape 1) 1) -1 -1)]
            (setv col (.select (get keys idx) 1 c))
            (setv idx (get idx (torch.argsort col :descending descending :stable True))))
          idx)
        (do
          (setv keys (np.reshape y [(get y.shape 0) -1])
                n (get keys.shape 0)
                idx (np.arange n))
          (for [c (range (- (get keys.shape 1) 1) -1 -1)]
            (setv col (get keys #(idx c)))
            (setv order (if descending
                          (_argsort-desc col)
                          (np.argsort col :kind "stable")))
            (setv idx (get idx order)))
          idx))))

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
    (return (bool-zeros y (list y.shape))))
  (when (and (= x.ndim 0) (= y.ndim 0))
    (return (if (is-torch x) (torch.eq x y) (np.equal x y))))
  (setv x (if (< x.ndim y.ndim)
            (reshape x (+ (lfor _ (range (- y.ndim x.ndim)) 1) (list x.shape)))
            x)
        out (bool-zeros y (list y.shape)))
  (for [[xs ys] (zip x.shape y.shape)]
    (when (> xs ys)
      (return out)))
  (when (in 0 (list x.shape))
    (return out))
  (if (is-torch y)
    (do
      (setv w y)
      (for [ax (range y.ndim)]
        (setv w (.unfold w ax (get x.shape ax) 1)))
      (setv hit (torch.eq w x))
      (for [_ (range y.ndim)]
        (setv hit (torch.all hit :dim -1)))
      (setv sl (tuple (lfor s hit.shape (slice None s))))
      (setv (get out sl) hit)
      out)
    (do
      (setv w y)
      (for [ax (range y.ndim)]
        (setv w (np.lib.stride-tricks.sliding-window-view w (get x.shape ax) :axis ax)))
      (setv hit (np.equal w x))
      (for [_ (range y.ndim)]
        (setv hit (np.all hit :axis -1)))
      (setv sl (tuple (lfor s hit.shape (slice None s))))
      (setv (get out sl) hit)
      out)))

(setv ⍷ (tdyad "⍷" find-dyad))

(setv __all__ ["∪" "∩" "⍋" "⍒" "⍷"])
