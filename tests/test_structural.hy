(import haply [⍴ ++ ⍪ ⌽ ⊖ ⍉ ↑ ↓ ⊢ ⊣])
(import torch)
(import pytest)

(defn T [#* xs]
  (torch.tensor (list xs)))

(defn cube []
  (.reshape (torch.arange 24) 2 3 4))

;; --- G026 ⍴: 1-d / 2-d / 3-d ----------------------------------------------

(defn test-shape-vector []
  (assert (torch.equal (⍴ (T 1 2 3 4)) (torch.tensor [4]))))

(defn test-shape-matrix []
  (setv t (torch.tensor [[1 2 3] [4 5 6]]))
  (assert (torch.equal (⍴ t) (torch.tensor [2 3]))))

(defn test-shape-cube []
  (assert (torch.equal (⍴ (cube)) (torch.tensor [2 3 4]))))

(defn test-shape-scalar []
  (setv s (torch.tensor 7))
  (assert (torch.equal (⍴ s) (torch.tensor [] :dtype torch.int64))))

(defn test-reshape-1d []
  (setv t (torch.tensor [[1 2] [3 4]]))
  (assert (torch.equal (⍴ 4 t) (T 1 2 3 4))))

(defn test-reshape-2d []
  (setv t (T 1 2 3 4 5 6))
  (assert (torch.equal (⍴ [2 3] t) (torch.tensor [[1 2 3] [4 5 6]]))))

(defn test-reshape-3d []
  (setv t (torch.arange 24)
        r (⍴ [2 3 4] t))
  (assert (torch.equal r (cube)))
  (assert (= (list r.shape) [2 3 4])))

(defn test-reshape-tensor-shape []
  (setv t (T 1 2 3 4 5 6))
  (assert (torch.equal (⍴ (torch.tensor [3 2]) t)
                       (torch.tensor [[1 2] [3 4] [5 6]]))))

(defn test-reshape-no-cycle []
  (with [(pytest.raises RuntimeError)]
    (⍴ [2 2] (T 1 2 3))))

(defn test-shape-error []
  (with [(pytest.raises TypeError)]
    (⍴))
  (with [(pytest.raises TypeError)]
    (⍴ [2 2] 1)))

;; --- G027 ++ --------------------------------------------------------------

(defn test-ravel-1d []
  (setv t (T 1 2 3))
  (assert (torch.equal (++ t) t)))

(defn test-ravel-2d []
  (assert (torch.equal (++ (torch.tensor [[1 2 3] [4 5 6]])) (T 1 2 3 4 5 6))))

(defn test-ravel-3d []
  (assert (torch.equal (++ (cube)) (torch.arange 24))))

(defn test-ravel-scalar []
  (setv r (++ (torch.tensor 7)))
  (assert (torch.equal r (T 7)))
  (assert (= (list r.shape) [1])))

(defn test-catenate-vectors []
  (assert (torch.equal (++ (T 1 2) (T 3 4)) (T 1 2 3 4))))

(defn test-catenate-matrices-last []
  (setv a (torch.tensor [[1 2] [3 4]])
        b (torch.tensor [[5] [6]]))
  (assert (torch.equal (++ a b) (torch.tensor [[1 2 5] [3 4 6]]))))

(defn test-catenate-scalars []
  (assert (torch.equal (++ (torch.tensor 1) (torch.tensor 2)) (T 1 2))))

(defn test-catenate-rank-diff []
  (setv m (torch.tensor [[1 2] [3 4]])
        v (T 5 6))
  (assert (torch.equal (++ m v) (torch.tensor [[1 2 5] [3 4 6]]))))

(defn test-catenate-python-scalar []
  (assert (torch.equal (++ (T 1 2) 3) (T 1 2 3))))

(defn test-catenate-rank-error []
  (with [(pytest.raises ValueError)]
    (++ (cube) (T 1 2))))

(defn test-catenate-arity-error []
  (with [(pytest.raises TypeError)]
    (++ (T 1) (T 2) (T 3))))

;; --- G028 ⍪ ---------------------------------------------------------------

(defn test-table-vector []
  (assert (torch.equal (⍪ (T 1 2 3)) (torch.tensor [[1] [2] [3]]))))

(defn test-table-scalar []
  (assert (torch.equal (⍪ (torch.tensor 7)) (torch.tensor [[7]]))))

(defn test-table-cube []
  (setv r (⍪ (cube)))
  (assert (= (list r.shape) [2 12]))
  (assert (torch.equal (get r 0) (torch.arange 12)))
  (assert (torch.equal (get r 1) (torch.arange 12 24))))

(defn test-catenate-first []
  (setv a (torch.tensor [[1 2] [3 4]])
        b (torch.tensor [[5 6]]))
  (assert (torch.equal (⍪ a b) (torch.tensor [[1 2] [3 4] [5 6]]))))

(defn test-catenate-first-broadcast []
  (assert (torch.equal (⍪ (T 1 2) 3) (T 1 2 3))))

(defn test-table-error []
  (with [(pytest.raises TypeError)]
    (⍪)))

;; --- G029 ⌽ / G030 ⊖ ------------------------------------------------------

(defn test-reverse-1d []
  (assert (torch.equal (⌽ (T 1 2 3 4 5)) (T 5 4 3 2 1))))

(defn test-reverse-2d-last []
  (setv m (torch.tensor [[1 2 3] [4 5 6]]))
  (assert (torch.equal (⌽ m) (torch.tensor [[3 2 1] [6 5 4]]))))

(defn test-reverse-3d-last []
  (setv c (cube)
        r (⌽ c))
  (assert (= (list r.shape) [2 3 4]))
  (assert (torch.equal (get (get r 0) 0) (T 3 2 1 0))))

(defn test-reverse-first-2d []
  (setv m (torch.tensor [[1 2 3] [4 5 6]]))
  (assert (torch.equal (⊖ m) (torch.tensor [[4 5 6] [1 2 3]]))))

(defn test-reverse-first-3d []
  (assert (torch.equal (get (⊖ (cube)) 0) (get (cube) 1))))

(defn test-rotate-last []
  (assert (torch.equal (⌽ 1 (T 1 2 3 4)) (T 2 3 4 1)))
  (assert (torch.equal (⌽ -1 (T 1 2 3 4)) (T 4 1 2 3))))

(defn test-rotate-last-3d []
  (setv r (⌽ 1 (cube)))
  (assert (torch.equal (get (get r 0) 0) (T 1 2 3 0))))

(defn test-rotate-first []
  (setv m (torch.tensor [[1 2] [3 4] [5 6]]))
  (assert (torch.equal (⊖ 1 m) (torch.tensor [[3 4] [5 6] [1 2]]))))

(defn test-reverse-scalar []
  (setv s (torch.tensor 3))
  (assert (torch.equal (⌽ s) s))
  (assert (torch.equal (⌽ 1 s) s)))

(defn test-rotate-error []
  (with [(pytest.raises TypeError)]
    (⌽ (T 1) (T 2) (T 3))))

;; --- G031 ⍉ ---------------------------------------------------------------

(defn test-transpose-1d []
  (setv t (T 1 2 3))
  (assert (torch.equal (⍉ t) t)))

(defn test-transpose-2d []
  (setv m (torch.tensor [[1 2 3] [4 5 6]]))
  (assert (torch.equal (⍉ m) (torch.tensor [[1 4] [2 5] [3 6]]))))

(defn test-transpose-3d []
  (setv r (⍉ (cube)))
  (assert (= (list r.shape) [4 3 2]))
  (assert (= (int (.item (get r #(0 0 0)))) 0))
  (assert (= (int (.item (get r #(0 0 1)))) 12)))

(defn test-transpose-dyadic-2d []
  (setv m (torch.tensor [[1 2 3] [4 5 6]]))
  (assert (torch.equal (⍉ [1 0] m) (⍉ m))))

(defn test-transpose-dyadic-3d []
  (setv r (⍉ [1 0 2] (cube)))
  (assert (= (list r.shape) [3 2 4])))

(defn test-transpose-bad-perm []
  (with [(pytest.raises ValueError)]
    (⍉ [0 0] (torch.tensor [[1 2] [3 4]]))))

(defn test-transpose-error []
  (with [(pytest.raises TypeError)]
    (⍉)))

;; --- G032 ↑ / G033 ↓ ------------------------------------------------------

(defn test-take-vector []
  (assert (torch.equal (↑ 3 (T 1 2 3 4 5)) (T 1 2 3)))
  (assert (torch.equal (↑ -2 (T 1 2 3 4 5)) (T 4 5))))

(defn test-take-overtake []
  (assert (torch.equal (↑ 5 (T 1 2 3)) (T 1 2 3 0 0)))
  (assert (torch.equal (↑ -5 (T 1 2 3)) (T 0 0 1 2 3))))

(defn test-take-matrix []
  (setv m (torch.tensor [[1 2 3 4] [5 6 7 8]]))
  (assert (torch.equal (↑ [2 3] m) (torch.tensor [[1 2 3] [5 6 7]])))
  (assert (torch.equal (↑ [-1 -2] m) (torch.tensor [[7 8]]))))

(defn test-take-leading-only []
  (setv m (torch.tensor [[1 2 3] [4 5 6] [7 8 9]]))
  (assert (torch.equal (↑ 2 m) (torch.tensor [[1 2 3] [4 5 6]]))))

(defn test-take-scalar []
  (assert (torch.equal (↑ 3 (torch.tensor 9)) (T 9 0 0))))

(defn test-drop-vector []
  (assert (torch.equal (↓ 3 (T 1 2 3 4 5)) (T 4 5)))
  (assert (torch.equal (↓ -2 (T 1 2 3 4 5)) (T 1 2 3))))

(defn test-drop-empty []
  (setv r (↓ 10 (T 1 2 3)))
  (assert (= (list r.shape) [0])))

(defn test-drop-matrix []
  (setv m (torch.tensor [[1 2 3 4] [5 6 7 8] [9 10 11 12]]))
  (assert (torch.equal (↓ [1 1] m) (torch.tensor [[6 7 8] [10 11 12]]))))

(defn test-take-monad-dropped []
  (with [(pytest.raises TypeError)]
    (↑ (T 1 2 3))))

(defn test-drop-monad-dropped []
  (with [(pytest.raises TypeError)]
    (↓ (T 1 2 3))))

;; --- G034 ⊢ / G035 ⊣ ------------------------------------------------------

(defn test-right []
  (setv t (T 1 2))
  (assert (is (⊢ t) t))
  (assert (is (⊢ (T 0) t) t)))

(defn test-left []
  (setv t (T 1 2)
        u (T 3 4))
  (assert (is (⊣ t) t))
  (assert (is (⊣ t u) t)))

(defn test-identity-python []
  (assert (= (⊢ 3) 3))
  (assert (= (⊣ "a" "b") "a"))
  (assert (= (⊢ "a" "b") "b")))

(defn test-right-error []
  (with [(pytest.raises TypeError)]
    (⊢)))
