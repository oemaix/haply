(import torch)
(import numpy :as np)
(import haply._backend [is-torch is-numpy is-array require-array
                        item-int int64-vector])
(import haply._dispatch [tprim reshape flatten cat-axis unsqueeze
                         flip-axis roll-axis permute zeros-like-shape])

(defn ->ints [x [name "axis spec"]]
  "0-d/1-d array, Python int, or list/tuple of ints."
  (cond
    (is-array x)
      (do
        (when (> x.ndim 1)
          (raise (ValueError (.format "{} must have rank 0 or 1" name))))
        (if (= x.ndim 0)
          #((item-int x))
          (tuple (lfor i x (item-int i)))))
    (isinstance x bool)
      (raise (TypeError (.format "{} must be a 1-d tensor or a sequence of ints" name)))
    (isinstance x int)
      #(x)
    (isinstance x #(list tuple))
      (tuple x)
    True
      (raise (TypeError (.format "{} must be a 1-d tensor or a sequence of ints" name)))))

(defn ->shape [x]
  (->ints x "⍴ shape"))

;; --- G026 ⍴ ---------------------------------------------------------------

(defn shape-monad [y]
  "1-d int64 shape vector (G026 working default). Same backend as Y."
  (int64-vector (list y.shape) y))

(defn reshape-dyad [x y]
  "Backend reshape: element count must match. No Dyalog recycle."
  (reshape y (->shape x)))

(defn ⍴ [#* args]
  (match (len args)
    1 (shape-monad (require-array "⍴" (get args 0)))
    2 (reshape-dyad (get args 0) (require-array "⍴" (get args 1)))
    _ (raise (TypeError (.format "⍴ takes 1 or 2 arguments, got {}"
                                 (len args))))))

;; --- G027 ++ --------------------------------------------------------------
;; Ravel is C-order flatten. Catenate joins the last axis. Rank difference
;; 0 or 1 (unsqueeze the shorter on the join axis). No implicit laminate:
;; same-shape arrays still cat, they do not stack on a new last axis.

(defn ravel-monad [y]
  (flatten y))

(defn _catenate [x y axis]
  (setv rx x.ndim
        ry y.ndim)
  (cond
    (and (= rx 0) (= ry 0))
      (if (is-torch x)
        (torch.stack [x y] :dim 0)
        (np.stack [x y] :axis 0))
    (= rx ry)
      (cat-axis [x y] axis)
    (= rx (+ ry 1))
      (cat-axis [x (unsqueeze y axis)] axis)
    (= ry (+ rx 1))
      (cat-axis [(unsqueeze x axis) y] axis)
    True
      (raise (ValueError "++ / ⍪ rank difference must be 0 or 1"))))

(defn catenate-last [x y]
  (_catenate x y -1))

(setv ++ (tprim "++" ravel-monad catenate-last))

;; --- G028 ⍪ ---------------------------------------------------------------

(defn table-monad [y]
  (cond
    (= y.ndim 0) (reshape y #(1 1))
    (= y.ndim 1) (reshape y #((get y.shape 0) 1))
    True (reshape y #((get y.shape 0) -1))))

(defn catenate-first [x y]
  (_catenate x y 0))

(setv ⍪ (tprim "⍪" table-monad catenate-first))

;; --- G029 ⌽ / G030 ⊖ ------------------------------------------------------

(defn _reverse [y dim]
  (if (= y.ndim 0)
    y
    (flip-axis y dim)))

(defn _scalar-shift [x]
  (if (is-array x)
    (do
      (when (> (if (is-torch x) (.numel x) x.size) 1)
        (raise (ValueError "⌽ / ⊖ rotation X must be a scalar")))
      (item-int x))
    (int x)))

(defn _rotate [x y dim]
  (if (= y.ndim 0)
    y
    (roll-axis y (- (_scalar-shift x)) dim)))

(setv ⌽ (tprim "⌽" (fn [y] (_reverse y -1)) (fn [x y] (_rotate x y -1))))
(setv ⊖ (tprim "⊖" (fn [y] (_reverse y 0)) (fn [x y] (_rotate x y 0))))

;; --- G031 ⍉ ---------------------------------------------------------------

(defn transpose-monad [y]
  (if (< y.ndim 2)
    y
    (permute y (list (range (- y.ndim 1) -1 -1)))))

(defn transpose-dyad [x y]
  (setv spec (->ints x "⍉ axes")
        n y.ndim)
  (when (!= (len spec) n)
    (raise (ValueError "⍉ X length must equal the rank of Y")))
  (when (!= (sorted spec) (list (range n)))
    (raise (ValueError "⍉ X must be a permutation of axes")))
  (setv inv (lfor _ (range n) 0))
  (for [i (range n)]
    (setv (get inv (get spec i)) i))
  (if (< n 2)
    y
    (permute y inv)))

(defn ⍉ [#* args]
  (match (len args)
    1 (transpose-monad (require-array "⍉" (get args 0)))
    2 (transpose-dyad (get args 0) (require-array "⍉" (get args 1)))
    _ (raise (TypeError (.format "⍉ takes 1 or 2 arguments, got {}"
                                 (len args))))))

;; --- G032 ↑ / G033 ↓ ------------------------------------------------------
;; Monadic mix/split are dropped. Pad/overtake fill is backend zero (item 33).

(defn _promote-scalar [y n-axes]
  (if (and (= y.ndim 0) (> n-axes 0))
    (reshape y (list (lfor _ (range n-axes) 1)))
    y))

(defn _narrow [t axis start length]
  (if (is-torch t)
    (torch.narrow t axis start length)
    (do
      (setv sl (lfor _ (range t.ndim) (slice None)))
      (setv (get sl axis) (slice start (+ start length)))
      (get t (tuple sl)))))

(defn _take-axis [t n axis]
  (setv size (get t.shape axis))
  (cond
    (>= n 0)
      (if (<= n size)
        (_narrow t axis 0 n)
        (do
          (setv pad-shape (list t.shape))
          (setv (get pad-shape axis) (- n size))
          (cat-axis [t (zeros-like-shape t pad-shape)] axis)))
    True
      (do
        (setv n (- n))
        (if (<= n size)
          (_narrow t axis (- size n) n)
          (do
            (setv pad-shape (list t.shape))
            (setv (get pad-shape axis) (- n size))
            (cat-axis [(zeros-like-shape t pad-shape) t] axis))))))

(defn _drop-axis [t n axis]
  (setv size (get t.shape axis))
  (cond
    (>= n 0)
      (if (>= n size)
        (_narrow t axis 0 0)
        (_narrow t axis n (- size n)))
    True
      (do
        (setv n (- n))
        (if (>= n size)
          (_narrow t axis 0 0)
          (_narrow t axis 0 (- size n))))))

(defn take-dyad [x y]
  (setv spec (->ints x "↑ take")
        t (_promote-scalar y (len spec)))
  (when (> (len spec) t.ndim)
    (raise (ValueError "↑ X is longer than the rank of Y")))
  (for [[i n] (enumerate spec)]
    (setv t (_take-axis t n i)))
  t)

(defn drop-dyad [x y]
  (setv spec (->ints x "↓ drop")
        t (_promote-scalar y (len spec)))
  (when (> (len spec) t.ndim)
    (raise (ValueError "↓ X is longer than the rank of Y")))
  (for [[i n] (enumerate spec)]
    (setv t (_drop-axis t n i)))
  t)

(defn ↑ [#* args]
  (if (= (len args) 2)
    (take-dyad (get args 0) (require-array "↑" (get args 1)))
    (raise (TypeError (.format "↑ takes 2 arguments, got {}" (len args))))))

(defn ↓ [#* args]
  (if (= (len args) 2)
    (drop-dyad (get args 0) (require-array "↓" (get args 1)))
    (raise (TypeError (.format "↓ takes 2 arguments, got {}" (len args))))))

;; --- G034 ⊢ / G035 ⊣ ------------------------------------------------------
;; Identity / right / left. Any host value; no backend kernel.

(defn ⊢ [#* args]
  (match (len args)
    1 (get args 0)
    2 (get args 1)
    _ (raise (TypeError (.format "⊢ takes 1 or 2 arguments, got {}"
                                 (len args))))))

(defn ⊣ [#* args]
  (match (len args)
    1 (get args 0)
    2 (get args 0)
    _ (raise (TypeError (.format "⊣ takes 1 or 2 arguments, got {}"
                                 (len args))))))

(setv __all__ ["⍴" "++" "⍪" "⌽" "⊖" "⍉" "↑" "↓" "⊢" "⊣"])
