(import haply [+])
(import haply.scalar :as sc)
(import torch)
(import numpy :as np)
(import pytest)

(defn test-python-add-stays-hy []
  "Hy's + macro still owns call position (decision 8, 53)."
  (assert (= (+ 1 2) 3)))

(defn test-torch-add []
  (setv t (torch.tensor [1 2]))
  (assert (torch.equal (+ t t) (torch.tensor [2 4]))))

(defn test-haply-plus-python-waits []
  "Item 21: the Haply function is PyTorch-only in Phase 1."
  (with [(pytest.raises TypeError)]
    (sc.+ 1 2)))

(defn test-mixed-backends []
  (with [(pytest.raises TypeError)]
    (sc.+ (torch.tensor [1]) (np.array [2]))))

(defn test-numpy-waits []
  (with [(pytest.raises TypeError)]
    (sc.+ (np.array [1]) (np.array [2]))))
