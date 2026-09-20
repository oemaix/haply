(require haply.macros [⌿ ⌿_ ⍀ ⍀_ ⍨ · outer ¨])
(import haply [× ⌽])
(import haply.scalar :as sc)
(import torch)
(import pytest)

(defn T [#* xs]
  (torch.tensor (list xs)))

;; --- done criterion -------------------------------------------------------

(defn test-reduce-plus-after-times []
  (setv A (torch.tensor [[1 2 3] [4 5 6]] :dtype torch.float32)
        B A)
  (assert (torch.equal (⌿ + (× A B)) (torch.sum (× A B) :dim 0))))

(defn test-inner-plus-times-is-matmul []
  (setv A (torch.tensor [[1 2] [3 4]] :dtype torch.float32)
        B (torch.tensor [[5 6] [7 8]] :dtype torch.float32))
  (assert (torch.equal (· + × A B) (torch.matmul A B))))

;; --- reduce / replicate ---------------------------------------------------

(defn test-reduce-plus-last []
  (setv m (torch.tensor [[1 2 3] [4 5 6]]))
  (assert (torch.equal (⌿_ + m) (T 6 15))))

(defn test-reduce-times []
  (assert (torch.equal (⌿ × (T 2 3 4)) (torch.tensor 24))))

(defn test-replicate-bool []
  (setv m (torch.tensor [[1 2] [3 4] [5 6]])
        mask (torch.tensor [True False True]))
  (assert (torch.equal (⌿ mask m) (torch.tensor [[1 2] [5 6]]))))

(defn test-replicate-int []
  (assert (torch.equal (⌿ (T 1 0 2) (T 7 8 9)) (T 7 9 9))))

(defn test-replicate-last []
  (setv m (torch.tensor [[1 2 3] [4 5 6]]))
  (assert (torch.equal (⌿_ (T 1 0 1) m) (torch.tensor [[1 3] [4 6]]))))

(defn test-replicate-negative-error []
  (with [(pytest.raises ValueError)]
    (⌿ (T 1 -1) (T 3 4))))

(defn test-replicate-scalar-error []
  (with [(pytest.raises ValueError)]
    (⌿ (T 1) (torch.tensor 3))))

(defn test-replicate-float-error []
  (with [(pytest.raises ValueError)]
    (⌿ (torch.tensor [1.0 0.0]) (T 7 8))))

(defn test-nwise-plus []
  (assert (torch.equal (⌿ 2 + (T 1 2 3 4)) (T 3 5 7))))

(defn test-reduce-empty-plus []
  (assert (torch.equal (⌿ + (torch.tensor [] :dtype torch.float32))
                       (torch.tensor 0.0))))

(defn test-reduce-empty-unknown []
  (with [(pytest.raises ValueError)]
    (⌿ sc.+ (torch.tensor [] :dtype torch.float32))))

(defn test-reduce-user-plus []
  (assert (torch.equal (⌿ sc.+ (T 1 2 3)) (torch.tensor 6))))

;; --- scan -----------------------------------------------------------------

(defn test-scan-plus []
  (assert (torch.equal (⍀ + (T 1 2 3 4)) (T 1 3 6 10))))

(defn test-scan-plus-last []
  (setv m (torch.tensor [[1 2 3] [4 5 6]]))
  (assert (torch.equal (⍀_ + m) (torch.tensor [[1 3 6] [4 9 15]]))))

(defn test-expand-first []
  (assert (torch.equal (⍀ (T 1 0 1) (T 7 8)) (T 7 0 8))))

(defn test-expand-bool []
  (assert (torch.equal (⍀ (torch.tensor [True False True]) (T 7 8))
                       (T 7 0 8))))

(defn test-expand-last []
  (setv m (torch.tensor [[1 2] [3 4]]))
  (assert (torch.equal (⍀_ (T 1 0 1) m) (torch.tensor [[1 0 2] [3 0 4]]))))

(defn test-expand-matrix-first []
  (setv m (torch.tensor [[1 2] [3 4]]))
  (assert (torch.equal (⍀ (T 1 0 1) m) (torch.tensor [[1 2] [0 0] [3 4]]))))

(defn test-expand-length-error []
  (with [(pytest.raises ValueError)]
    (⍀ (T 1 0 1) (T 1 2 3))))

(defn test-expand-negative-error []
  (with [(pytest.raises ValueError)]
    (⍀ (T 1 -1) (T 3))))

;; --- commute --------------------------------------------------------------

(defn test-selfie []
  (assert (torch.equal (⍨ × (T 2 3)) (T 4 9))))

(defn test-commute []
  (assert (torch.equal (⍨ sc.- (T 1 2) (T 10 20)) (T 9 18))))

(defn test-constant []
  (setv a (T 9))
  (assert (is (⍨ a (T 1 2 3)) a)))

;; --- outer ----------------------------------------------------------------

(defn test-outer-times []
  (assert (torch.equal (outer × (T 1 2) (T 3 4 5))
                       (torch.tensor [[3 4 5] [6 8 10]]))))

(defn test-outer-plus []
  (assert (torch.equal (outer + (T 1 2) (T 10 20))
                       (torch.tensor [[11 21] [12 22]]))))

;; --- each -----------------------------------------------------------------

(defn test-each-scalar-is-elementwise []
  (assert (torch.equal (¨ × (T -2 0 4)) (T -1 0 1))))

(defn test-each-major-cells []
  (setv m (torch.tensor [[1 2 3] [4 5 6]]))
  (assert (torch.equal (¨ ⌽ m) (torch.tensor [[3 2 1] [6 5 4]]))))
