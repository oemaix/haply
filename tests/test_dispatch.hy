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
  "Item 21: Python natives stay out through Phase 7."
  (with [(pytest.raises TypeError)]
    (sc.+ 1 2)))

(defn test-mixed-backends []
  (with [(pytest.raises TypeError)]
    (sc.+ (torch.tensor [1]) (np.array [2]))))

(defn test-numpy-add []
  (setv a (np.array [1 2]))
  (setv r (sc.+ a a))
  (assert (isinstance r np.ndarray))
  (assert (np.array-equal r (np.array [2 4]))))

(defn test-numpy-scalar-is-not-an-array []
  (with [(pytest.raises TypeError)]
    (sc.+ (np.int64 1) (np.int64 2)))
  (with [(pytest.raises TypeError)]
    (sc.+ (np.array [1 2]) (np.int64 3))))
