(import haply [⍴])
(import torch)
(import pytest)

(defn test-shape-matrix []
  (setv t (torch.tensor [[1 2 3] [4 5 6]]))
  (assert (torch.equal (⍴ t) (torch.tensor [2 3]))))

(defn test-shape-scalar []
  (setv s (torch.tensor 7))
  (assert (torch.equal (⍴ s) (torch.tensor [] :dtype torch.int64))))

(defn test-reshape-list []
  (setv t (torch.tensor [1 2 3 4 5 6]))
  (assert (torch.equal (⍴ [2 3] t) (torch.tensor [[1 2 3] [4 5 6]]))))

(defn test-reshape-tensor-shape []
  (setv t (torch.tensor [1 2 3 4 5 6]))
  (assert (torch.equal (⍴ (torch.tensor [3 2]) t)
                       (torch.tensor [[1 2] [3 4] [5 6]]))))

(defn test-reshape-int []
  (setv t (torch.tensor [[1 2] [3 4]])
        r (⍴ 4 t))
  (assert (torch.equal r (torch.tensor [1 2 3 4])))
  (assert (= (list r.shape) [4])))

(defn test-reshape-no-cycle []
  (with [(pytest.raises RuntimeError)]
    (⍴ [2 2] (torch.tensor [1 2 3]))))

(defn test-shape-error []
  (with [(pytest.raises TypeError)]
    (⍴))
  (with [(pytest.raises TypeError)]
    (⍴ [2 2] 1)))
