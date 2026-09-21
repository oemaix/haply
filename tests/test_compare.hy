(import haply [≠ ≡ ≢])
(import torch)
(import pytest)

(defn T [#* xs]
  (torch.tensor (list xs)))

;; --- G018 ≠ ---------------------------------------------------------------

(defn test-unique-mask-vector []
  (assert (torch.equal (≠ (T 22 10 22 22 21 10 5 10))
                       (torch.tensor [True True False False True False True False]))))

(defn test-unique-mask-matrix-ravel []
  ;; Haply: first occurrence in C-order ravel, same shape as Y.
  (setv m (torch.tensor [[1 2] [1 3]]))
  (assert (torch.equal (≠ m) (torch.tensor [[True True] [False True]]))))

(defn test-unique-mask-scalar []
  (assert (torch.equal (≠ (torch.tensor 3)) (torch.tensor True))))

(defn test-nqe-dyad []
  (assert (torch.equal (≠ (T 1 2 3) (T 1 0 3))
                       (torch.tensor [False True False]))))

(defn test-nqe-broadcast []
  (assert (torch.equal (≠ (T 1 2 3) 2)
                       (torch.tensor [True False True]))))

(defn test-nqe-error []
  (with [(pytest.raises TypeError)]
    (≠ (T 1) (T 2) (T 3))))

;; --- G019 ≡ ---------------------------------------------------------------

(defn test-match-true []
  (setv t (T 1 2 3))
  (assert (≡ t (T 1 2 3)))
  (assert (≡ t t)))

(defn test-match-false []
  (assert (not (≡ (T 1 2) (T 1 3))))
  (assert (not (≡ (T 1 2) (T 1 2 3))))
  (setv nan (float "nan"))
  (assert (not (≡ (torch.tensor [nan]) (torch.tensor [nan]))))
  (assert (≡ (T 1 2) (torch.tensor [1.0 2.0]))))

(defn test-match-monad-dropped []
  (with [(pytest.raises TypeError)]
    (≡ (T 1 2))))

;; --- G020 ≢ ---------------------------------------------------------------

(defn test-tally-matrix []
  (assert (= (≢ (torch.tensor [[1 2 3] [4 5 6]])) 2)))

(defn test-tally-vector []
  (assert (= (≢ (T 1 2 3 4)) 4)))

(defn test-tally-scalar []
  (assert (= (≢ (torch.tensor 7)) 1)))

(defn test-tally-empty []
  (assert (= (≢ (torch.tensor [] :dtype torch.int64)) 0)))

(defn test-not-match []
  (assert (≢ (T 1 2) (T 1 3)))
  (assert (not (≢ (T 1 2) (T 1 2)))))

(defn test-tally-error []
  (with [(pytest.raises TypeError)]
    (≢)))
