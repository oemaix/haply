(import functools)
(import torch)
(import haply._backend [is-torch require-torch torch-monad torch-dyad])

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
  (defprim name (torch-monad name monad) (torch-dyad name dyad)))

(defn tmonad [name monad]
  (defmonad name (torch-monad name monad)))

(defn tdyad [name dyad]
  (defdyad name (torch-dyad name dyad)))

(defn tfold [name monad dyad]
  (deffold name (torch-monad name monad) (torch-dyad name dyad)))

;; --- Phase 3 operator runtimes (unknown operands / mixed cases) -----------

(defn reduce-cells [axis f y]
  "Reduce slices along `axis` by calling dyadic `f`. Empty is an error (41 B)."
  (setv y (require-torch "⌿" y))
  (when (= y.ndim 0)
    (return y))
  (setv n (get y.shape axis))
  (when (= n 0)
    (raise (ValueError "⌿ empty reduce needs a known identity")))
  (functools.reduce f (torch.unbind y axis)))

(defn scan-cells [axis f y]
  "Prefix scan along `axis` by calling dyadic `f`."
  (setv y (require-torch "⍀" y))
  (when (or (= y.ndim 0) (= (get y.shape axis) 0))
    (return y))
  (setv parts (list (torch.unbind y axis))
        acc (get parts 0)
        out [acc])
  (for [p (cut parts 1 None)]
    (setv acc (f acc p))
    (.append out acc))
  (torch.stack out :dim axis))

(defn replicate [axis x y]
  "Boolean compress or integer repeat along `axis`. Negative repeats error."
  (setv y (require-torch "⌿" y)
        x (if (is-torch x) x (torch.as-tensor x :device y.device)))
  (when (> x.ndim 1)
    (raise (ValueError "⌿ replicate X must have rank 0 or 1")))
  (setv x (if (= x.ndim 0) (torch.reshape x [1]) x)
        n (get y.shape axis))
  (when (!= (.numel x) n)
    (raise (ValueError "⌿ replicate X length must match the axis")))
  (cond
    (= x.dtype torch.bool)
      (do
        (setv idx (lfor _ (range y.ndim) (slice None))
              ax (if (< axis 0) (+ y.ndim axis) axis))
        (setv (get idx ax) x)
        (get y (tuple idx)))
    (torch.any (torch.lt x 0))
      (raise (ValueError "⌿ replicate X must be non-negative"))
    True
      (torch.repeat-interleave y x :dim axis)))

(defn reduce-or-replicate [axis op y]
  (if (callable op)
    (reduce-cells axis op y)
    (replicate axis op y)))

(defn expand-along [axis x y]
  "Insert backend-zero fill along `axis`. 0/False inserts; 1/True takes the next item."
  (setv y (require-torch "⍀" y)
        x (if (is-torch x) x (torch.as-tensor x :device y.device)))
  (when (= y.ndim 0)
    (raise (ValueError "⍀ expand needs rank ≥ 1")))
  (when (> x.ndim 1)
    (raise (ValueError "⍀ expand X must have rank 0 or 1")))
  (when (or (.is-floating-point x) (.is-complex x))
    (raise (ValueError "⍀ expand X must be boolean or integer")))
  (setv x (if (= x.ndim 0) (torch.reshape x [1]) x)
        ax (if (< axis 0) (+ y.ndim axis) axis)
        n (get y.shape ax))
  (when (and (!= x.dtype torch.bool) (torch.any (torch.lt x 0)))
    (raise (ValueError "⍀ expand X must be non-negative")))
  (when (and (!= x.dtype torch.bool) (torch.any (torch.gt x 1)))
    (raise (ValueError "⍀ expand X must be 0 or 1")))
  (setv ones (if (= x.dtype torch.bool)
               (int (torch.sum x))
               (int (torch.sum (torch.ne x 0)))))
  (when (!= ones n)
    (raise (ValueError "⍀ expand number of 1s must match the axis")))
  (setv shp (list y.shape))
  (setv (get shp ax) (int (.numel x)))
  (setv out (torch.zeros shp :dtype y.dtype :device y.device)
        dest (.reshape (torch.nonzero x) [-1]))
  (torch.index-copy out ax dest y))

(defn scan-or-expand [axis op y]
  (if (callable op)
    (scan-cells axis op y)
    (expand-along axis op y)))

(defn nwise-reduce [axis n f y]
  "Positive n-wise reduce. Unknown `f` folds each window."
  (setv y (require-torch "⌿" y)
        n (int n))
  (when (< n 1)
    (raise (ValueError "⌿ n-wise n must be positive")))
  (when (= y.ndim 0)
    (raise (ValueError "⌿ n-wise needs rank ≥ 1")))
  (setv size (get y.shape axis))
  (when (> n size)
    (setv shp (list y.shape))
    (setv (get shp axis) 0)
    (return (torch.empty shp :dtype y.dtype :device y.device)))
  (setv windows (.unfold y axis n 1))
  (if (callable f)
    (functools.reduce f (torch.unbind windows -1))
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
  (setv x (require-torch "outer" x)
        y (require-torch "outer" y))
  (torch.reshape x (+ (list x.shape) (lfor _ (range y.ndim) 1))))

(defn inner-product [f g x y]
  "Contract last axis of `x` with first axis of `y`, then reduce with `f`."
  (setv x (require-torch "·" x)
        y (require-torch "·" y))
  (when (or (= x.ndim 0) (= y.ndim 0))
    (raise (ValueError "· needs rank ≥ 1")))
  (when (!= (get x.shape -1) (get y.shape 0))
    (raise (ValueError "· contracted axes must match")))
  (setv xs (torch.reshape x (+ (list x.shape) (lfor _ (range (- y.ndim 1)) 1)))
        ys (torch.reshape y (+ (lfor _ (range (- x.ndim 1)) 1) (list y.shape)))
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
        (setv y (require-torch "¨" (get args 0)))
        (when (= y.ndim 0)
          (return (f y)))
        (torch.stack (list (map f (torch.unbind y 0)))))
    (= n 2)
      (do
        (setv x (require-torch "¨" (get args 0))
              y (require-torch "¨" (get args 1)))
        (when (!= (get x.shape 0) (get y.shape 0))
          (raise (ValueError "¨ major-cell counts must match")))
        (torch.stack (list (map f (torch.unbind x 0) (torch.unbind y 0)))))
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
