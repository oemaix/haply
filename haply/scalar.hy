(import operator)
(import torch)
(import numpy :as np)
(import haply._backend [backend route-dyad])
(import haply._dispatch [deffold])

(defn plus-monad [y]
  "G001 monadic: conjugate on complex, identity on reals (decision 31)."
  (setv b (backend y))
  (cond
    (= b 'torch) (if (.is-complex y) (torch.conj y) y)
    (= b 'numpy) (if (np.iscomplexobj y) (np.conjugate y) y)
    True (if (isinstance y complex) (.conjugate y) y)))

(defn plus-dyad [x y]
  "G001 dyadic: addition. Broadcasting is the backend's."
  (route-dyad x y torch.add np.add operator.add))

(setv + (deffold "+" plus-monad plus-dyad))
(setv __all__ ["+"])
