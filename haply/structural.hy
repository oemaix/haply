(import torch)
(import haply._backend [is-torch require-torch])
(import haply._dispatch [tprim])

(defn ->ints [x [name "axis spec"]]
  "0-d/1-d torch tensor, Python int, or list/tuple of ints."
  (cond
    (is-torch x)
      (do
        (when (> x.ndim 1)
          (raise (ValueError (.format "{} must have rank 0 or 1" name))))
        (if (= x.ndim 0)
          #((int (.item x)))
          (tuple (lfor i x (int (.item i))))))
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
  "1-d int64 shape vector (G026 working default). CPU, like constructors."
  (torch.tensor (list y.shape) :dtype torch.int64))

(defn reshape-dyad [x y]
  "Backend reshape: element count must match. No Dyalog recycle."
  (torch.reshape y (->shape x)))

(defn ⍴ [#* args]
  (match (len args)
    1 (shape-monad (require-torch "⍴" (get args 0)))
    2 (reshape-dyad (get args 0) (require-torch "⍴" (get args 1)))
    _ (raise (TypeError (.format "⍴ takes 1 or 2 arguments, got {}"
                                 (len args))))))

;; --- G027 ++ --------------------------------------------------------------
;; Ravel is C-order flatten. Catenate joins the last axis. Rank difference
;; 0 or 1 (unsqueeze the shorter on the join axis). No implicit laminate:
;; same-shape arrays still cat, they do not stack on a new last axis.

(defn ravel-monad [y]
  (torch.flatten y))

(defn _catenate [x y axis]
  (setv rx x.ndim
        ry y.ndim)
  (cond
    (and (= rx 0) (= ry 0))
      (torch.stack [x y] :dim 0)
    (= rx ry)
      (torch.cat [x y] :dim axis)
    (= rx (+ ry 1))
      (torch.cat [x (torch.unsqueeze y axis)] :dim axis)
    (= ry (+ rx 1))
      (torch.cat [(torch.unsqueeze x axis) y] :dim axis)
    True
      (raise (ValueError "++ / ⍪ rank difference must be 0 or 1"))))

(defn catenate-last [x y]
  (_catenate x y -1))

(setv ++ (tprim "++" ravel-monad catenate-last))

;; --- G028 ⍪ ---------------------------------------------------------------

(defn table-monad [y]
  (cond
    (= y.ndim 0) (torch.reshape y #(1 1))
    (= y.ndim 1) (torch.reshape y #((get y.shape 0) 1))
    True (torch.reshape y #((get y.shape 0) -1))))

(defn catenate-first [x y]
  (_catenate x y 0))

(setv ⍪ (tprim "⍪" table-monad catenate-first))

;; --- G029 ⌽ / G030 ⊖ ------------------------------------------------------

(defn _reverse [y dim]
  (if (= y.ndim 0)
    y
    (torch.flip y [dim])))

(defn _scalar-shift [x]
  (if (is-torch x)
    (do
      (when (> (.numel x) 1)
        (raise (ValueError "⌽ / ⊖ Phase 2 rotation X must be a scalar")))
      (int (.item x)))
    (int x)))

(defn _rotate [x y dim]
  (if (= y.ndim 0)
    y
    (torch.roll y (- (_scalar-shift x)) dim)))

(defn reverse-last [y]
  (_reverse y -1))

(defn rotate-last [x y]
  (_rotate x y -1))

(defn reverse-first [y]
  (_reverse y 0))

(defn rotate-first [x y]
  (_rotate x y 0))

(setv ⌽ (tprim "⌽" reverse-last rotate-last))
(setv ⊖ (tprim "⊖" reverse-first rotate-first))

;; --- G031 ⍉ ---------------------------------------------------------------

(defn transpose-monad [y]
  (if (< y.ndim 2)
    y
    (torch.permute y (list (range (- y.ndim 1) -1 -1)))))

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
    (torch.permute y inv)))

(defn ⍉ [#* args]
  (match (len args)
    1 (transpose-monad (require-torch "⍉" (get args 0)))
    2 (transpose-dyad (get args 0) (require-torch "⍉" (get args 1)))
    _ (raise (TypeError (.format "⍉ takes 1 or 2 arguments, got {}"
                                 (len args))))))

;; --- G032 ↑ / G033 ↓ ------------------------------------------------------
;; Monadic mix/split are dropped. Pad/overtake fill is backend zero (item 33).

(defn _promote-scalar [y n-axes]
  (if (and (= y.ndim 0) (> n-axes 0))
    (torch.reshape y (list (lfor _ (range n-axes) 1)))
    y))

(defn _take-axis [t n axis]
  (setv size (get t.shape axis))
  (cond
    (>= n 0)
      (if (<= n size)
        (torch.narrow t axis 0 n)
        (do
          (setv pad-shape (list t.shape))
          (setv (get pad-shape axis) (- n size))
          (torch.cat [t (torch.zeros pad-shape :dtype t.dtype :device t.device)]
                     :dim axis)))
    True
      (do
        (setv n (- n))
        (if (<= n size)
          (torch.narrow t axis (- size n) n)
          (do
            (setv pad-shape (list t.shape))
            (setv (get pad-shape axis) (- n size))
            (torch.cat [(torch.zeros pad-shape :dtype t.dtype :device t.device) t]
                       :dim axis))))))

(defn _drop-axis [t n axis]
  (setv size (get t.shape axis))
  (cond
    (>= n 0)
      (if (>= n size)
        (torch.narrow t axis 0 0)
        (torch.narrow t axis n (- size n)))
    True
      (do
        (setv n (- n))
        (if (>= n size)
          (torch.narrow t axis 0 0)
          (torch.narrow t axis 0 (- size n))))))

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
    (take-dyad (get args 0) (require-torch "↑" (get args 1)))
    (raise (TypeError (.format "↑ takes 2 arguments, got {}" (len args))))))

(defn ↓ [#* args]
  (if (= (len args) 2)
    (drop-dyad (get args 0) (require-torch "↓" (get args 1)))
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
