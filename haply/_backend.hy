;; Type detection without importing torch or numpy. A missing backend then
;; stays optional later (architecture Phase 0 still ships both in the shell).

(defn is-torch [x]
  (= (getattr (type x) "__module__" "") "torch"))

(defn is-numpy [x]
  (= (getattr (type x) "__module__" "") "numpy"))

(defn backend [x]
  (cond
    (is-torch x) 'torch
    (is-numpy x) 'numpy
    True 'python))

(defn route-dyad [x y torch-f numpy-f py-f]
  "Same backend or Python scalar + tensor. Mixed tensors raise TypeError."
  (setv bx (backend x)
        by (backend y))
  (cond
    (= bx by)
      (cond
        (= bx 'torch) (torch-f x y)
        (= bx 'numpy) (numpy-f x y)
        True (py-f x y))
    (and (= bx 'python) (in by ['torch 'numpy]))
      (if (= by 'torch) (torch-f x y) (numpy-f x y))
    (and (= by 'python) (in bx ['torch 'numpy]))
      (if (= bx 'torch) (torch-f x y) (numpy-f x y))
    True
      (raise (TypeError "haply mixed tensor backends"))))
