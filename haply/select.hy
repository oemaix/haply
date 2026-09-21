(import torch)
(import numpy :as np)
(import haply._backend [is-torch is-array backend
                        require-array as-like parse-backend item-int])
(import haply._dispatch [tprim tmonad])
(import haply.structural [->ints ++])

;; --- G036 ⍳ ---------------------------------------------------------------
;; Monadic: arange / stacked meshgrid (item 30). Default CPU torch; Phase 7
;; adds `(⍳ n :backend 'numpy)`. Dyadic: first index of each Y in 1-d X;
;; not-found is n (length of X).

(defn iota-monad [x [be 'torch]]
  (setv shp (->ints x "⍳"))
  (cond
    (= (len shp) 0)
      (raise (ValueError "⍳ needs a length or a shape"))
    (= (len shp) 1)
      (if (= be 'numpy)
        (np.arange (get shp 0) :dtype np.int64)
        (torch.arange (get shp 0) :dtype torch.int64))
    True
      (if (= be 'numpy)
        (do
          (setv grids (np.meshgrid
                        #* (lfor n shp (np.arange n :dtype np.int64))
                        :indexing "ij"))
          (np.stack (list grids) :axis -1))
        (do
          (setv grids (torch.meshgrid
                        #* (lfor n shp (torch.arange n :dtype torch.int64))
                        :indexing "ij"))
          (torch.stack (list grids) :dim -1)))))

(defn index-of [x y]
  (setv x (require-array "⍳" x))
  (cond
    (is-array y)
      (when (!= (backend x) (backend y))
        (raise (TypeError "haply mixed tensor backends")))
    True
      (setv y (as-like y x)))
  (when (!= x.ndim 1)
    (raise (ValueError "⍳ index-of searches a 1-d X")))
  (if (is-torch x)
    (do
      (setv n (.numel x)
            eq (torch.eq (torch.reshape x (+ (lfor _ (range y.ndim) 1) [n]))
                         (torch.unsqueeze y -1))
            found (torch.any eq :dim -1)
            idx (torch.argmax (eq.to torch.int64) :dim -1))
      (torch.where found idx (torch.full-like idx n)))
    (do
      (setv n x.size
            eq (np.equal (np.reshape x (+ (lfor _ (range y.ndim) 1) [n]))
                         (np.expand-dims y -1))
            found (np.any eq :axis -1)
            idx (np.argmax eq :axis -1))
      (np.where found idx n))))

(defn ⍳ [#* args #** kwargs]
  (setv be (parse-backend (.get kwargs "backend" None)))
  (match (len args)
    1 (iota-monad (get args 0) be)
    2 (index-of (get args 0) (get args 1))
    _ (raise (TypeError (.format "⍳ takes 1 or 2 arguments, got {}"
                                 (len args))))))

;; --- G037 ⍸ ---------------------------------------------------------------

(defn where-monad [y]
  (if (is-torch y)
    (do
      (setv nz (torch.nonzero y :as-tuple False))
      (if (= y.ndim 1)
        (torch.flatten nz)
        nz))
    (do
      (setv nz (np.argwhere y))
      (if (= y.ndim 1)
        (np.reshape nz -1)
        nz))))

(defn interval-index [x y]
  "Bucket `Y` by sorted breakpoints `X` (0-based)."
  (if (is-torch x)
    (torch.bucketize y x)
    (np.searchsorted x y :side "left")))

(setv ⍸ (tprim "⍸" where-monad interval-index))

;; --- G038 ∊ ---------------------------------------------------------------
;; Monadic enlist is flatten (same as ++). Dyadic: isin.

(defn membership-dyad [x y]
  (if (is-torch x)
    (torch.isin x y)
    (np.isin x y)))

(setv ∊ (tprim "∊" (fn [y] (++ y)) membership-dyad))

;; --- G044 ⌷ ---------------------------------------------------------------

(defn squad-dyad [x y]
  (cond
    (isinstance x int)
      (get y x)
    (is-array x)
      (if (= x.ndim 0)
        (get y (item-int x))
        (get y (tuple (lfor i x (item-int i)))))
    (isinstance x #(list tuple))
      (get y (tuple x))
    True
      (raise (TypeError "⌷ X must be an index or a sequence of indices"))))

(defn ⌷ [#* args]
  (match (len args)
    1 (require-array "⌷" (get args 0))
    2 (squad-dyad (get args 0) (require-array "⌷" (get args 1)))
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
