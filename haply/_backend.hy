;; Phase 7 requires both backends at import (item 21 notes). NumPy
;; scalars (`np.int64`) are not arrays (item 36). Python natives wait
;; on item 21.
(import torch)
(import numpy :as np)

(defn is-torch [x]
  (setv mod (getattr (type x) "__module__" ""))
  (and (or (= mod "torch") (.startswith mod "torch."))
       (hasattr x "ndim")))

(defn is-numpy [x]
  (isinstance x np.ndarray))

(defn is-array [x]
  (or (is-torch x) (is-numpy x)))

(defn backend [x]
  (cond
    (is-torch x) 'torch
    (is-numpy x) 'numpy
    True 'python))

(defn is-python-number [x]
  (isinstance x #(int float complex bool)))

(defn require-array [name x]
  (if (is-array x)
    x
    (raise (TypeError (.format "{} accepts torch.Tensor or numpy.ndarray, got {}"
                               name
                               (type x))))))

(defn require-same [name x y]
  "Two arrays of the same backend. Mixed tensor backends error."
  (setv x (require-array name x)
        y (require-array name y))
  (when (!= (backend x) (backend y))
    (raise (TypeError "haply mixed tensor backends")))
  #(x y))

(defn as-like [x other]
  "0-d array on other's backend. Do not move other (decision 49)."
  (cond
    (is-torch other)
      (if (is-torch x) x (torch.as-tensor x :device other.device))
    (is-numpy other)
      (if (is-numpy x) x (np.asarray x))
    True x))

(defn array-monad [name f]
  "Monadic path: torch or NumPy array (item 21; Python natives wait)."
  (fn [y]
    (f (require-array name y))))

(defn array-dyad [name f]
  "Dyadic path: same tensor backend, or a Python number next to an array.
  Mixed tensor backends error. Python+Python waits on item 21."
  (fn [x y]
    (setv tx (is-torch x)
          ty (is-torch y)
          nx (is-numpy x)
          ny (is-numpy y))
    (cond
      (and tx ty) (f x y)
      (and nx ny) (f x y)
      (and tx (is-python-number y)) (f x (as-like y x))
      (and (is-python-number x) ty) (f (as-like x y) y)
      (and nx (is-python-number y)) (f x (as-like y x))
      (and (is-python-number x) ny) (f (as-like x y) y)
      (and (or tx nx) (or ty ny) (not (= (backend x) (backend y))))
        (raise (TypeError "haply mixed tensor backends"))
      True
        (raise (TypeError (.format "{} accepts torch.Tensor or numpy.ndarray, got {} and {}"
                                   name
                                   (type x)
                                   (type y)))))))

;; --- per-array queries ----------------------------------------------------

(defn is-complex [x]
  (if (is-torch x)
    (.is-complex x)
    (np.issubdtype x.dtype np.complexfloating)))

(defn is-floating [x]
  (if (is-torch x)
    (.is-floating-point x)
    (np.issubdtype x.dtype np.floating)))

(defn is-bool [x]
  (if (is-torch x)
    (= x.dtype torch.bool)
    (np.issubdtype x.dtype np.bool_)))

(defn is-integer [x]
  (if (is-torch x)
    (and (not (.is-floating-point x))
         (not (.is-complex x))
         (!= x.dtype torch.bool))
    (np.issubdtype x.dtype np.integer)))

(defn nelem [x]
  (if (is-torch x) (.numel x) x.size))

(defn item-int [x]
  (int (if (is-array x) (.item x) x)))

(defn zeros-like-shape [y shp]
  (if (is-torch y)
    (torch.zeros shp :dtype y.dtype :device y.device)
    (np.zeros (tuple shp) :dtype y.dtype)))

(defn empty-like-shape [y shp]
  (if (is-torch y)
    (torch.empty shp :dtype y.dtype :device y.device)
    (np.empty (tuple shp) :dtype y.dtype)))

(defn bool-zeros [y shp]
  (if (is-torch y)
    (torch.zeros shp :dtype torch.bool :device y.device)
    (np.zeros (tuple shp) :dtype np.bool_)))

(defn int64-vector [xs [like None]]
  "1-d int64 vector. Same backend as `like`, else CPU torch (item 30).
  Torch device follows `like` (item 68 working default A)."
  (cond
    (is-numpy like) (np.asarray xs :dtype np.int64)
    (is-torch like) (torch.tensor xs :dtype torch.int64 :device like.device)
    True (torch.tensor xs :dtype torch.int64)))

(defn parse-backend [value]
  "Constructor :backend. Symbols or strings; default is torch (item 30)."
  (if (is value None)
    'torch
    (do
      (setv s (str value))
      (cond
        (= s "torch") 'torch
        (= s "numpy") 'numpy
        True (raise (ValueError "backend must be 'torch or 'numpy"))))))
