(import haply [+])
(import torch)

(defn test-python-add []
  (assert (= (+ 1 2) 3)))

(defn test-torch-add []
  (setv t (torch.tensor [1 2]))
  (assert (torch.equal (+ t t) (torch.tensor [2 4]))))
