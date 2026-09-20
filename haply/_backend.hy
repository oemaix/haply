;; Type detection uses the type's module name so a missing optional backend
;; stays optional later. Torch is required in the Nix shell; some kernels
;; need a 0-d tensor for a Python scalar. NumPy waits on Phase 7 (item 21).
(import torch)

(defn is-torch [x]
  (setv mod (getattr (type x) "__module__" ""))
  (and (or (= mod "torch") (.startswith mod "torch."))
       (hasattr x "ndim")))

(defn is-numpy [x]
  (setv mod (getattr (type x) "__module__" ""))
  (and (or (= mod "numpy") (.startswith mod "numpy."))
       (hasattr x "ndim")))

(defn backend [x]
  (cond
    (is-torch x) 'torch
    (is-numpy x) 'numpy
    True 'python))

(defn is-python-number [x]
  (isinstance x #(int float complex bool)))

(defn require-torch [name x]
  (if (is-torch x)
    x
    (raise (TypeError (.format "{} accepts torch.Tensor, got {}"
                               name
                               (type x))))))

(defn torch-monad [name f]
  "PyTorch-only monadic path (item 21; NumPy is Phase 7)."
  (fn [y]
    (f (require-torch name y))))

(defn as-torch-like [x other]
  "0-d tensor on other's device. Do not move other (decision 49)."
  (if (is-torch x)
    x
    (torch.as-tensor x :device other.device)))

(defn torch-dyad [name f]
  "PyTorch dyadic path: torch+torch or Python number + tensor.
  Mixed tensor backends error. NumPy and Python+Python wait on item 21."
  (fn [x y]
    (setv tx (is-torch x)
          ty (is-torch y)
          nx (is-numpy x)
          ny (is-numpy y))
    (cond
      (and tx ty) (f x y)
      (and tx (is-python-number y)) (f x (as-torch-like y x))
      (and (is-python-number x) ty) (f (as-torch-like x y) y)
      (and (or tx nx) (or ty ny) (not (= (backend x) (backend y))))
        (raise (TypeError "haply mixed tensor backends"))
      True
        (raise (TypeError (.format "{} accepts torch.Tensor, got {} and {}"
                                   name
                                   (type x)
                                   (type y)))))))
