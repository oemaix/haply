(import haply [⍳ ⍸ ∊ ⌷ ⊃ ∪ ∩ ⍋ ⍒ ≁])
(import torch)
(import pytest)

(defn T [#* xs]
  (torch.tensor (list xs)))

;; --- G036 ⍳ ---------------------------------------------------------------

(defn test-iota-vector []
  (assert (torch.equal (⍳ 5) (torch.arange 5))))

(defn test-iota-shape []
  (setv r (⍳ [2 3]))
  (assert (= (list r.shape) [2 3 2]))
  (assert (torch.equal (get r #(0 0)) (T 0 0)))
  (assert (torch.equal (get r #(1 2)) (T 1 2))))

(defn test-index-of []
  (assert (torch.equal (⍳ (T 10 20 30) (T 20 40 10)) (T 1 3 0))))

(defn test-index-of-broadcast []
  (assert (torch.equal (⍳ (T 1 2 3) 2) (torch.tensor 1))))

(defn test-iota-error []
  (with [(pytest.raises TypeError)]
    (⍳)))

;; --- G037 ⍸ ---------------------------------------------------------------

(defn test-where-vector []
  (assert (torch.equal (⍸ (torch.tensor [False True False True]))
                       (T 1 3))))

(defn test-where-matrix []
  (setv r (⍸ (torch.tensor [[0 1] [1 0]])))
  (assert (torch.equal r (torch.tensor [[0 1] [1 0]]))))

(defn test-interval-index []
  ;; torch.bucketize, left: insertion index in sorted X (0-based).
  (assert (torch.equal (⍸ (T 0 2 5) (T -1 0 1 2 4 5 9))
                       (T 0 0 1 1 2 2 3))))

(defn test-where-error []
  (with [(pytest.raises TypeError)]
    (⍸)))

;; --- G038 ∊ ---------------------------------------------------------------

(defn test-enlist []
  (assert (torch.equal (∊ (torch.tensor [[1 2] [3 4]])) (T 1 2 3 4))))

(defn test-membership []
  (assert (torch.equal (∊ (T 1 2 3) (T 2 4))
                       (torch.tensor [False True False]))))

(defn test-membership-broadcast []
  (assert (torch.equal (∊ (T 1 2 3) 2)
                       (torch.tensor [False True False]))))

;; --- G044 ⌷ / G050 ⊃ ------------------------------------------------------

(defn test-squad-identity []
  (setv t (T 1 2 3))
  (assert (is (⌷ t) t)))

(defn test-squad-index []
  (setv m (torch.tensor [[1 2 3] [4 5 6]]))
  (assert (torch.equal (⌷ 1 m) (T 4 5 6)))
  (assert (= (int (.item (⌷ [1 2] m))) 6)))

(defn test-first-cell []
  (setv m (torch.tensor [[1 2] [3 4]]))
  (assert (torch.equal (⊃ m) (T 1 2))))

(defn test-first-scalar []
  (setv s (torch.tensor 7))
  (assert (is (⊃ s) s)))

(defn test-pick-dropped []
  (with [(pytest.raises TypeError)]
    (⊃ 0 (T 1 2))))

;; --- G040 ∪ / G041 ∩ ------------------------------------------------------

(defn test-unique []
  (assert (torch.equal (∪ (T 22 10 22 21 10)) (T 22 10 21))))

(defn test-union []
  (assert (torch.equal (∪ (T 1 2 1) (T 2 3)) (T 1 2 3))))

(defn test-intersect []
  (assert (torch.equal (∩ (T 1 2 3 2) (T 2 4)) (T 2 2))))

(defn test-unique-error []
  (with [(pytest.raises TypeError)]
    (∪)))

;; --- G042 ⍋ / G043 ⍒ ------------------------------------------------------

(defn test-grade-up []
  (assert (torch.equal (⍋ (T 30 10 20)) (T 1 2 0))))

(defn test-grade-down []
  (assert (torch.equal (⍒ (T 30 10 20)) (T 0 2 1))))

(defn test-grade-rows []
  (setv m (torch.tensor [[2 1] [1 9] [2 0]]))
  (assert (torch.equal (⍋ m) (T 1 2 0))))

(defn test-grade-dyad-dropped []
  (with [(pytest.raises TypeError)]
    (⍋ (T 1) (T 3 2 1))))

(defn test-grade-complex-error []
  (with [(pytest.raises ValueError)]
    (⍋ (torch.tensor [1+2j]))))

;; --- G021 ≁ without -------------------------------------------------------

(defn test-without []
  (assert (torch.equal (≁ (T 1 2 3 2) (T 2 4)) (T 1 3))))
