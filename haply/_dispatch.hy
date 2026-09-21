(import functools)
(import torch)
(import numpy :as np)
(import haply._backend [is-torch is-numpy is-array is-bool is-floating
                        is-complex is-integer backend
                        require-array require-same
                        as-like array-monad array-dyad
                        nelem zeros-like-shape empty-like-shape bool-zeros])

(defn defprim [name monad dyad]
  "Strict monadic / dyadic wrapper (architecture helper)."
  (fn [#* args]
    (match (len args)
      1 (monad (get args 0))
      2 (dyad (get args 0) (get args 1))
      _ (raise (TypeError (.format "{} takes 1 or 2 arguments, got {}"
                                   name
                                   (len args)))))))

(defn defmonad [name monad]
  "Strict monadic wrapper."
  (fn [#* args]
    (if (= (len args) 1)
      (monad (get args 0))
      (raise (TypeError (.format "{} takes 1 argument, got {}"
                                 name
                                 (len args)))))))

(defn defdyad [name dyad]
  "Strict dyadic wrapper."
  (fn [#* args]
    (if (= (len args) 2)
      (dyad (get args 0) (get args 1))
      (raise (TypeError (.format "{} takes 2 arguments, got {}"
                                 name
                                 (len args)))))))

(defn deffold [name monad dyad]
  "Hy-style fold for associative scalar ops (decision 37)."
  (fn [#* args]
    (setv n (len args))
    (cond
      (< n 1)
        (raise (TypeError (.format "{} takes at least 1 argument, got {}"
                                   name
                                   n)))
      (= n 1) (monad (get args 0))
      (= n 2) (dyad (get args 0) (get args 1))
      True (functools.reduce dyad args))))

(defn tprim [name monad dyad]
  (defprim name (array-monad name monad) (array-dyad name dyad)))

(defn tmonad [name monad]
  (defmonad name (array-monad name monad)))

(defn tdyad [name dyad]
  (defdyad name (array-dyad name dyad)))

(defn tfold [name monad dyad]
  (deffold name (array-monad name monad) (array-dyad name dyad)))

(defn u1 [torch-f numpy-f]
  "Monadic kernel that picks the backend function."
  (fn [y]
    (if (is-torch y) (torch-f y) (numpy-f y))))

(defn u2 [torch-f numpy-f]
  "Dyadic kernel that picks the backend function."
  (fn [x y]
    (if (is-torch x) (torch-f x y) (numpy-f x y))))

;; --- axis helpers ---------------------------------------------------------

(defn unbind [y axis]
  (if (is-torch y)
    (torch.unbind y axis)
    (np.moveaxis y axis 0)))

(defn stack-axis [parts axis [like None]]
  (setv sample (if (is like None) (get parts 0) like))
  (if (is-torch sample)
    (torch.stack (list parts) :dim axis)
    (np.stack (list parts) :axis axis)))

(defn cat-axis [parts axis]
  (if (is-torch (get parts 0))
    (torch.cat (list parts) :dim axis)
    (np.concatenate (list parts) :axis axis)))

(defn unsqueeze [y axis]
  (if (is-torch y)
    (torch.unsqueeze y axis)
    (np.expand-dims y axis)))

(defn flip-axis [y axis]
  (if (is-torch y)
    (torch.flip y [axis])
    (np.flip y :axis axis)))

(defn roll-axis [y shift axis]
  (if (is-torch y)
    (torch.roll y shift axis)
    (np.roll y shift :axis axis)))

(defn reshape [y shp]
  (if (is-torch y)
    (torch.reshape y (list shp))
    (np.reshape y (tuple shp))))

(defn flatten [y]
  (if (is-torch y)
    (torch.flatten y)
    (np.reshape y -1)))

(defn permute [y axes]
  (if (is-torch y)
    (torch.permute y (list axes))
    (np.transpose y (tuple axes))))

;; --- Phase 3 operator runtimes -------------------------------------------

(defn reduce-cells [axis f y]
  "Reduce slices along `axis` by calling dyadic `f`. Empty is an error (41 B)."
  (setv y (require-array "⌿" y))
  (when (= y.ndim 0)
    (return y))
  (setv n (get y.shape axis))
  (when (= n 0)
    (raise (ValueError "⌿ empty reduce needs a known identity")))
  (functools.reduce f (unbind y axis)))

(defn scan-cells [axis f y]
  "Prefix scan along `axis` by calling dyadic `f`."
  (setv y (require-array "⍀" y))
  (when (or (= y.ndim 0) (= (get y.shape axis) 0))
    (return y))
  (setv parts (list (unbind y axis))
        acc (get parts 0)
        out [acc])
  (for [p (cut parts 1 None)]
    (setv acc (f acc p))
    (.append out acc))
  (stack-axis out axis y))

(defn _axis [y axis]
  (if (< axis 0) (+ y.ndim axis) axis))

(defn replicate [axis x y]
  "Boolean compress or integer repeat along `axis`. Negative repeats error."
  (setv y (require-array "⌿" y)
        pair (cond
               (is-array x)
                 (if (= (backend x) (backend y))
                   #(x y)
                   (raise (TypeError "haply mixed tensor backends")))
               True
                 #((as-like x y) y))
        x (get pair 0))
  (when (= y.ndim 0)
    (raise (ValueError "⌿ replicate needs rank ≥ 1")))
  (when (> x.ndim 1)
    (raise (ValueError "⌿ replicate X must have rank 0 or 1")))
  (when (or (is-floating x) (is-complex x))
    (raise (ValueError "⌿ replicate X must be boolean or integer")))
  (setv x (if (= x.ndim 0) (reshape x [1]) x)
        n (get y.shape axis)
        ax (_axis y axis))
  (when (!= (nelem x) n)
    (raise (ValueError "⌿ replicate X length must match the axis")))
  (cond
    (is-bool x)
      (if (is-torch y)
        (do
          (setv idx (lfor _ (range y.ndim) (slice None)))
          (setv (get idx ax) x)
          (get y (tuple idx)))
        (np.compress x y :axis ax))
    (if (is-torch x) (torch.any (torch.lt x 0)) (np.any (np.less x 0)))
      (raise (ValueError "⌿ replicate X must be non-negative"))
    True
      (if (is-torch y)
        (torch.repeat-interleave y x :dim ax)
        (np.repeat y x :axis ax))))

(defn reduce-or-replicate [axis op y]
  (if (callable op)
    (reduce-cells axis op y)
    (replicate axis op y)))

(defn expand-along [axis x y]
  "Insert backend-zero fill along `axis`. 0/False inserts; 1/True takes the next item."
  (setv y (require-array "⍀" y)
        pair (cond
               (is-array x)
                 (if (= (backend x) (backend y))
                   #(x y)
                   (raise (TypeError "haply mixed tensor backends")))
               True
                 #((as-like x y) y))
        x (get pair 0))
  (when (= y.ndim 0)
    (raise (ValueError "⍀ expand needs rank ≥ 1")))
  (when (> x.ndim 1)
    (raise (ValueError "⍀ expand X must have rank 0 or 1")))
  (when (or (is-floating x) (is-complex x))
    (raise (ValueError "⍀ expand X must be boolean or integer")))
  (setv x (if (= x.ndim 0) (reshape x [1]) x)
        ax (_axis y axis)
        n (get y.shape ax)
        xb (is-bool x))
  (when (and (not xb)
             (if (is-torch x) (torch.any (torch.lt x 0)) (np.any (np.less x 0))))
    (raise (ValueError "⍀ expand X must be non-negative")))
  (when (and (not xb)
             (if (is-torch x) (torch.any (torch.gt x 1)) (np.any (np.greater x 1))))
    (raise (ValueError "⍀ expand X must be 0 or 1")))
  (setv ones (if xb
               (int (if (is-torch x) (torch.sum x) (np.sum x)))
               (int (if (is-torch x)
                      (torch.sum (torch.ne x 0))
                      (np.sum (np.not-equal x 0))))))
  (when (!= ones n)
    (raise (ValueError "⍀ expand number of 1s must match the axis")))
  (setv shp (list y.shape))
  (setv (get shp ax) (int (nelem x)))
  (setv out (zeros-like-shape y shp)
        dest (if (is-torch x)
               (.reshape (torch.nonzero x) [-1])
               (get (np.nonzero x) 0)))
  (if (is-torch y)
    (torch.index-copy out ax dest y)
    (do
      (setv sl (lfor _ (range y.ndim) (slice None)))
      (setv (get sl ax) dest)
      (setv (get out (tuple sl)) y)
      out)))

(defn scan-or-expand [axis op y]
  (if (callable op)
    (scan-cells axis op y)
    (expand-along axis op y)))

(defn _windows [y axis n]
  (if (is-torch y)
    (.unfold y axis n 1)
    (np.lib.stride-tricks.sliding-window-view y n :axis axis)))

(defn nwise-reduce [axis n f y]
  "Positive n-wise reduce. Unknown `f` folds each window."
  (setv y (require-array "⌿" y)
        n (int n))
  (when (< n 1)
    (raise (ValueError "⌿ n-wise n must be positive")))
  (when (= y.ndim 0)
    (raise (ValueError "⌿ n-wise needs rank ≥ 1")))
  (setv size (get y.shape axis))
  (when (> n size)
    (setv shp (list y.shape))
    (setv (get shp axis) 0)
    (return (empty-like-shape y shp)))
  (setv windows (_windows y axis n))
  (if (callable f)
    (if (is-torch y)
      (functools.reduce f (torch.unbind windows -1))
      (functools.reduce f (np.moveaxis windows -1 0)))
    (raise (TypeError "⌿ n-wise operand must be callable"))))

(defn commute [op #* args]
  (setv n (len args))
  (cond
    (= n 1)
      (if (callable op)
        (op (get args 0) (get args 0))
        op)
    (= n 2)
      (if (callable op)
        (op (get args 1) (get args 0))
        op)
    True
      (raise (TypeError (.format "⍨ takes 2 or 3 arguments, got {}" (+ n 1))))))

(defn outer-left [x y]
  "Reshape `x` so an elementwise `g` with `y` is an outer product."
  (setv pair (require-same "∘·" x y)
        x (get pair 0)
        y (get pair 1))
  (reshape x (+ (list x.shape) (lfor _ (range y.ndim) 1))))

(defn inner-product [f g x y]
  "Contract last axis of `x` with first axis of `y`, then reduce with `f`."
  (setv pair (require-same "·" x y)
        x (get pair 0)
        y (get pair 1))
  (when (or (= x.ndim 0) (= y.ndim 0))
    (raise (ValueError "· needs rank ≥ 1")))
  (when (!= (get x.shape -1) (get y.shape 0))
    (raise (ValueError "· contracted axes must match")))
  (setv xs (reshape x (+ (list x.shape) (lfor _ (range (- y.ndim 1)) 1)))
        ys (reshape y (+ (lfor _ (range (- x.ndim 1)) 1) (list y.shape)))
        z (g xs ys)
        dim (- x.ndim 1))
  (if (callable f)
    (reduce-cells dim f z)
    (raise (TypeError "· left operand must be callable"))))

(defn each-cells [f #* args]
  "Map `f` over major cells and stack (item 42 default B)."
  (setv n (len args))
  (cond
    (= n 1)
      (do
        (setv y (require-array "¨" (get args 0)))
        (when (= y.ndim 0)
          (return (f y)))
        (stack-axis (list (map f (unbind y 0))) 0 y))
    (= n 2)
      (do
        (setv pair (require-same "¨" (get args 0) (get args 1))
              x (get pair 0)
              y (get pair 1))
        (when (!= (get x.shape 0) (get y.shape 0))
          (raise (ValueError "¨ major-cell counts must match")))
        (stack-axis (list (map f (unbind x 0) (unbind y 0))) 0 x))
    True
      (raise (TypeError (.format "¨ takes 2 or 3 arguments, got {}" (+ n 1))))))

(defn fork-wing [op #* args]
  "Call `op` if it is a function; otherwise it is a constant fork wing."
  (if (callable op)
    (op #* args)
    op))

(defn beside-or-bind [a b y]
  "Runtime ∘ with three forms: compose, or bind one array operand."
  (cond
    (and (callable a) (callable b))
      (a (b y))
    (and (not (callable a)) (callable b))
      (b a y)
    (and (callable a) (not (callable b)))
      (a y b)
    True
      (raise (TypeError "∘ needs two functions, or one function and one array"))))

;; --- known-operand kernels (macro targets; pick torch vs NumPy) ----------

(defn reduce-known [axis name y]
  (setv y (require-array "⌿" y))
  (when (= y.ndim 0)
    (return y))
  (if (is-torch y)
    (cond
      (= name "sum") (torch.sum y :dim axis)
      (= name "prod") (torch.prod y :dim axis)
      (= name "amax") (torch.amax y :dim axis)
      True (torch.amin y :dim axis))
    (cond
      (= name "sum") (np.sum y :axis axis)
      (= name "prod") (np.prod y :axis axis)
      (= name "amax") (np.amax y :axis axis)
      True (np.amin y :axis axis))))

(defn scan-known [axis name y]
  (setv y (require-array "⍀" y))
  (when (= y.ndim 0)
    (return y))
  (if (is-torch y)
    (cond
      (= name "cumsum") (torch.cumsum y :dim axis)
      (= name "cumprod") (torch.cumprod y :dim axis)
      (= name "cummax") (.values (torch.cummax y axis))
      True (.values (torch.cummin y axis)))
    (cond
      (= name "cumsum") (np.cumsum y :axis axis)
      (= name "cumprod") (np.cumprod y :axis axis)
      (= name "cummax") (np.maximum.accumulate y :axis axis)
      True (np.minimum.accumulate y :axis axis))))

(defn nwise-known [axis k name y]
  (setv y (require-array "⌿" y)
        k (int k))
  (when (or (< k 1) (= y.ndim 0))
    (raise (ValueError "⌿ n-wise needs rank ≥ 1 and positive n")))
  (setv size (get y.shape axis))
  (when (> k size)
    (setv shp (list y.shape))
    (setv (get shp axis) 0)
    (return (empty-like-shape y shp)))
  (setv w (_windows y axis k))
  (if (is-torch y)
    (cond
      (= name "sum") (torch.sum w :dim -1)
      (= name "prod") (torch.prod w :dim -1)
      (= name "amax") (torch.amax w :dim -1)
      True (torch.amin w :dim -1))
    (cond
      (= name "sum") (np.sum w :axis -1)
      (= name "prod") (np.prod w :axis -1)
      (= name "amax") (np.amax w :axis -1)
      True (np.amin w :axis -1))))

(defn matmul-known [x y]
  (setv pair (require-same "·" x y))
  (if (is-torch (get pair 0))
    (torch.matmul (get pair 0) (get pair 1))
    (np.matmul (get pair 0) (get pair 1))))

(defn outer-mul [x y]
  (setv left (outer-left x y)
        y (require-array "∘·" y))
  (if (is-torch y) (torch.mul left y) (np.multiply left y)))

(defn outer-add [x y]
  (setv left (outer-left x y)
        y (require-array "∘·" y))
  (if (is-torch y) (torch.add left y) (np.add left y)))
