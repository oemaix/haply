(import torch)
(import haply._backend [is-torch require-torch])

(defn ->shape [x]
  "Shape specifier: 0-d/1-d torch tensor, Python int, or list/tuple of ints."
  (cond
    (is-torch x)
      (do
        (when (> x.ndim 1)
          (raise (ValueError "⍴ shape must have rank 0 or 1")))
        (if (= x.ndim 0)
          #((int (.item x)))
          (tuple (lfor i x (int (.item i))))))
    (isinstance x int)
      #(x)
    (isinstance x #(list tuple))
      (tuple x)
    True
      (raise (TypeError "⍴ shape must be a 1-d tensor or a sequence of ints"))))

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

(setv __all__ ["⍴"])
