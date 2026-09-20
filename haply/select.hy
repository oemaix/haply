(import torch)
(import haply._backend [is-torch require-torch])
(import haply._dispatch [tprim tmonad])
(import haply.structural [->ints ++])

;; --- G036 ⍳ ---------------------------------------------------------------
;; Monadic: CPU int64 arange / stacked meshgrid (item 30). Dyadic: first
;; index of each Y in 1-d X; not-found is n (length of X).

(defn iota-monad [x]
  (setv shp (->ints x "⍳"))
  (cond
    (= (len shp) 0)
      (raise (ValueError "⍳ needs a length or a shape"))
    (= (len shp) 1)
      (torch.arange (get shp 0) :dtype torch.int64)
    True
      (do
        (setv grids (torch.meshgrid
                      #* (lfor n shp (torch.arange n :dtype torch.int64))
                      :indexing "ij"))
        (torch.stack (list grids) :dim -1))))

(defn index-of [x y]
  (setv x (require-torch "⍳" x)
        y (if (is-torch y) y (torch.as-tensor y :device x.device)))
  (when (!= x.ndim 1)
    (raise (ValueError "⍳ index-of searches a 1-d X")))
  (setv n (.numel x)
        eq (torch.eq (torch.reshape x (+ (lfor _ (range y.ndim) 1) [n]))
                     (torch.unsqueeze y -1))
        found (torch.any eq :dim -1)
        idx (torch.argmax (eq.to torch.int64) :dim -1))
  (torch.where found idx (torch.full-like idx n)))

(defn ⍳ [#* args]
  (match (len args)
    1 (iota-monad (get args 0))
    2 (index-of (get args 0) (get args 1))
    _ (raise (TypeError (.format "⍳ takes 1 or 2 arguments, got {}"
                                 (len args))))))

;; --- G037 ⍸ ---------------------------------------------------------------

(defn where-monad [y]
  (setv nz (torch.nonzero y :as-tuple False))
  (if (= y.ndim 1)
    (torch.flatten nz)
    nz))

(defn interval-index [x y]
  "Bucket `Y` by sorted breakpoints `X` (0-based)."
  (torch.bucketize y x))

(setv ⍸ (tprim "⍸" where-monad interval-index))

;; --- G038 ∊ ---------------------------------------------------------------
;; Monadic enlist is flatten (same as ++). Dyadic: isin.

(defn membership-dyad [x y]
  (torch.isin x y))

(setv ∊ (tprim "∊" (fn [y] (++ y)) membership-dyad))

;; --- G044 ⌷ ---------------------------------------------------------------

(defn squad-dyad [x y]
  (cond
    (isinstance x int)
      (get y x)
    (is-torch x)
      (if (= x.ndim 0)
        (get y (int (.item x)))
        (get y (tuple (lfor i x (int (.item i))))))
    (isinstance x #(list tuple))
      (get y (tuple x))
    True
      (raise (TypeError "⌷ X must be an index or a sequence of indices"))))

(defn ⌷ [#* args]
  (match (len args)
    1 (require-torch "⌷" (get args 0))
    2 (squad-dyad (get args 0) (require-torch "⌷" (get args 1)))
    _ (raise (TypeError (.format "⌷ takes 1 or 2 arguments, got {}"
                                 (len args))))))

;; --- G050 ⊃ ---------------------------------------------------------------
;; First major cell. Dyadic pick is dropped.

(defn first-cell [y]
  (if (= y.ndim 0)
    y
    (do
      (when (= (get y.shape 0) 0)
        (raise (ValueError "⊃ first axis must be non-empty")))
      (get y 0))))

(setv ⊃ (tmonad "⊃" first-cell))

(setv __all__ ["⍳" "⍸" "∊" "⌷" "⊃"])
