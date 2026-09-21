(require haply.macros [⌿ ⌿_ ⍀ ⍀_ ⍨ · ∘· ¨ ∘])
(require haply.trains [⋔])
(import haply [× ÷ ⍴ ++ ⍪ ⌽ ⊖ ⍉ ↑ ↓ ⊢ ⊣ ≠ ≡ ≢ ⍟ || ⌊ ⌈ ∧])
(import haply [⍳ ⍸ ∊ ⌷ ⊃ ∪ ∩ ⍋ ⍒ ≁ ⍷ ○ ! ? ⌹ ⊤ ⊥])
(import haply.scalar :as sc)
(import numpy :as np)
(import torch)
(import pytest)

(defn N [#* xs]
  (np.array (list xs)))

(defn F [#* xs]
  (np.array (list xs) :dtype np.float64))

;; --- done criterion -------------------------------------------------------

(defn test-plus-returns-ndarray []
  (setv a (N 1 2)
        r (sc.+ a a))
  (assert (isinstance r np.ndarray))
  (assert (np.array-equal r (N 2 4))))

(defn test-shape-returns-ndarray []
  (setv a (np.array [[1 2 3] [4 5 6]])
        r (⍴ a))
  (assert (isinstance r np.ndarray))
  (assert (np.array-equal r (N 2 3))))

(defn test-mixed-still-errors []
  (with [(pytest.raises TypeError)]
    (sc.+ (torch.tensor [1]) (N 2))))

(defn test-python-natives-still-wait []
  (with [(pytest.raises TypeError)]
    (sc.+ 1 2)))

;; --- scalars --------------------------------------------------------------

(defn test-plus-monad-complex []
  (setv z (np.array [1+2j]))
  (assert (np.array-equal (sc.+ z) (np.array [1-2j]))))

(defn test-minus-times-div []
  (assert (np.array-equal (sc.- (N 1 -2)) (N -1 2)))
  (assert (np.array-equal (× (F -2.0 0.0 4.0)) (F -1.0 0.0 1.0)))
  (assert (np.array-equal (× (N 2 3) (N 4 5)) (N 8 15)))
  (assert (np.allclose (÷ (F 2.0 4.0)) (F 0.5 0.25))))

(defn test-plus-broadcast-fold []
  (assert (np.array-equal (sc.+ (N 1 2 3) 10) (N 11 12 13)))
  (assert (np.array-equal (sc.+ (N 1) (N 2) (N 3)) (N 6))))

(defn test-exp-log-residue-floor []
  (assert (np.allclose (sc.** (F 0.0 1.0)) (F 1.0 np.e)))
  (assert (np.allclose (⍟ 2.0 (F 2.0 4.0 8.0)) (F 1.0 2.0 3.0)))
  (assert (np.array-equal (|| 3 (N 5 7)) (N 2 1)))
  (setv t (N 3 4))
  (assert (is (⌊ t) t))
  (assert (np.array-equal (⌈ (N 1 8) (N 4 3)) (N 4 8))))

(defn test-compare-logic []
  (assert (np.array-equal (sc.< (N 1 2 3) 2) (np.array [True False False])))
  (assert (np.array-equal (≁ (np.array [False True])) (np.array [True False])))
  (assert (np.array-equal (∧ (np.array [True False True])
                            (np.array [True True False]))
                         (np.array [True False False])))
  (assert (np.array-equal (∧ (N 4 6) (N 6 9)) (N 12 18))))

;; --- structure ------------------------------------------------------------

(defn test-reshape-cat-reverse []
  (setv t (N 1 2 3 4 5 6))
  (assert (np.array-equal (⍴ [2 3] t) (np.array [[1 2 3] [4 5 6]])))
  (assert (np.array-equal (++ (N 1 2) (N 3 4)) (N 1 2 3 4)))
  (assert (np.array-equal (⌽ (N 1 2 3)) (N 3 2 1)))
  (assert (np.array-equal (⊖ (np.array [[1 2] [3 4]])) (np.array [[3 4] [1 2]]))))

(defn test-transpose-take-drop []
  (setv m (np.array [[1 2 3] [4 5 6]]))
  (assert (np.array-equal (⍉ m) (np.array [[1 4] [2 5] [3 6]])))
  (assert (np.array-equal (↑ 2 (N 1 2 3 4)) (N 1 2)))
  (assert (np.array-equal (↓ 1 (N 1 2 3)) (N 2 3)))
  (assert (is (⊢ m) m))
  (assert (= (⊣ 1 m) 1)))

(defn test-compare-match-tally []
  (assert (np.array-equal (≠ (N 22 10 22 21 10))
                         (np.array [True True False True False])))
  (assert (≡ (N 1 2 3) (N 1 2 3)))
  (assert (not (≡ (N 1 2) (N 1 3))))
  (assert (= (≢ (np.array [[1 2 3] [4 5 6]])) 2)))

;; --- search ---------------------------------------------------------------

(defn is-torch-result [x]
  (setv mod (getattr (type x) "__module__" ""))
  (or (= mod "torch") (.startswith mod "torch.")))

(defn test-iota-numpy-backend []
  (setv r (⍳ 5 :backend 'numpy))
  (assert (isinstance r np.ndarray))
  (assert (np.array-equal r (np.arange 5))))

(defn test-iota-default-still-torch []
  (setv r (⍳ 3))
  (assert (is-torch-result r)))

(defn test-index-of-where-member []
  (assert (np.array-equal (⍳ (N 10 20 30) (N 20 40 10)) (N 1 3 0)))
  (assert (np.array-equal (⍸ (np.array [False True False True])) (N 1 3)))
  (assert (np.array-equal (∊ (N 1 2 3) (N 2 4)) (np.array [False True False]))))

(defn test-unique-grade-find []
  (assert (np.array-equal (∪ (N 22 10 22 21 10)) (N 22 10 21)))
  (assert (np.array-equal (∩ (N 1 2 3 2) (N 2 4)) (N 2 2)))
  (assert (np.array-equal (⍋ (N 30 10 20)) (N 1 2 0)))
  (assert (np.array-equal (⍒ (N 30 10 20)) (N 0 2 1)))
  (assert (np.array-equal (⍷ (N 1 2) (N 0 1 2 1 2 9))
                         (np.array [False True False True False False])))
  (assert (np.array-equal (≁ (N 1 2 3 2) (N 2 4)) (N 1 3))))

(defn test-squad-first []
  (setv m (np.array [[1 2 3] [4 5 6]]))
  (assert (np.array-equal (⌷ 1 m) (N 4 5 6)))
  (assert (np.array-equal (⊃ m) (N 1 2 3))))

;; --- numeric --------------------------------------------------------------

(defn test-circ-fact []
  (assert (np.allclose (○ (F 1.0)) (F np.pi)))
  (assert (np.allclose (○ 1 (F 0.0)) (F 0.0)))
  (assert (np.allclose (! (F 0.0 1.0 2.0 3.0)) (F 1.0 1.0 2.0 6.0))))

(defn test-roll-deal-range []
  (np.random.seed 0)
  (setv r (? (N 6 6 6 6)))
  (assert (isinstance r np.ndarray))
  (assert (np.all (np.greater-equal r 0)))
  (assert (np.all (np.less r 6)))
  (np.random.seed 0)
  (setv d (? (np.array 4) (np.array 10)))
  (assert (= (len d) 4))
  (assert (= (len (np.unique d)) 4)))

(defn test-domino-encode []
  (setv m (np.array [[2.0 0.0] [0.0 4.0]]))
  (assert (np.allclose (⌹ m) (np.array [[0.5 0.0] [0.0 0.25]])))
  (assert (np.array-equal (⊤ (N 2 2 2) 5) (N 1 0 1)))
  (assert (= (int (⊥ 2 (N 1 0 1 0))) 10)))

;; --- operators / trains ---------------------------------------------------

(defn test-reduce-inner-expand []
  (setv A (np.array [[1 2 3] [4 5 6]] :dtype np.float32)
        B A)
  (assert (np.array-equal (⌿ + (× A B)) (np.sum (× A B) :axis 0)))
  (setv X (np.array [[1 2] [3 4]] :dtype np.float32)
        Y (np.array [[5 6] [7 8]] :dtype np.float32))
  (assert (np.array-equal (· + × X Y) (np.matmul X Y)))
  (assert (np.array-equal (⍀ (N 1 0 1) (N 7 8)) (N 7 0 8))))

(defn test-scan-outer-each []
  (assert (np.array-equal (⍀ + (N 1 2 3 4)) (N 1 3 6 10)))
  (assert (np.array-equal (∘· × (N 1 2) (N 3 4 5))
                         (np.array [[3 4 5] [6 8 10]])))
  (assert (np.array-equal (¨ ⌽ (np.array [[1 2 3] [4 5 6]]))
                         (np.array [[3 2 1] [6 5 4]]))))

(defn test-fork-beside []
  (setv X (N 1 2 3)
        Y (N 10 20 30))
  (assert (np.array-equal (⋔ ⊣ + ⊢ X Y) (sc.+ X Y)))
  (assert (np.array-equal (∘ × + (N -2 0 4)) (× (sc.+ (N -2 0 4))))))

(defn test-reduce-scan-0d []
  (setv s (np.array 3))
  (assert (= (int (.item (⌿ + s))) 3))
  (assert (= (int (.item (⍀ + s))) 3)))

(defn test-reduce-last-and-empty []
  (setv m (np.array [[1 2 3] [4 5 6]]))
  (assert (np.array-equal (⌿_ + m) (N 6 15)))
  (assert (np.array-equal (⌿ + (np.array [] :dtype np.float32))
                         (np.array 0.0 :dtype np.float32))))

(defn test-replicate-bool-int-last []
  (setv m (np.array [[1 2] [3 4] [5 6]])
        mask (np.array [True False True]))
  (assert (np.array-equal (⌿ mask m) (np.array [[1 2] [5 6]])))
  (assert (np.array-equal (⌿ (N 1 0 2) (N 7 8 9)) (N 7 9 9)))
  (setv a (np.array [[1 2 3] [4 5 6]]))
  (assert (np.array-equal (⌿_ (N 1 0 1) a) (np.array [[1 3] [4 6]]))))

(defn test-nwise-plus-and-overflow []
  (assert (np.array-equal (⌿ 2 + (N 1 2 3 4)) (N 3 5 7)))
  (setv y (N 1 2 3)
        r (⌿ 5 + y))
  (assert (= (list r.shape) [0]))
  (assert (= r.dtype y.dtype)))

(defn test-match-values-not-dtype []
  (assert (≡ (N 1 2 3) (N 1 2 3)))
  (assert (≡ (N 1 2) (F 1.0 2.0)))
  (assert (not (≡ (np.array [np.nan]) (np.array [np.nan])))))

(defn test-grade-rows-and-int64-min []
  (setv m (np.array [[2 1] [1 9] [2 0]]))
  (assert (np.array-equal (⍋ m) (N 1 2 0)))
  (setv info (np.iinfo np.int64)
        y (np.array [info.min 0 1] :dtype np.int64))
  (assert (np.array-equal (⍒ y) (N 2 1 0))))

(defn test-find-matrix []
  (setv y (np.array [[1 0 1 0]
                     [0 1 0 1]])
        x (np.array [[1 0]
                     [0 1]]))
  (assert (np.array-equal (⍷ x y)
                         (np.array [[True False True False]
                                    [False False False False]]))))

(defn test-solve-and-binomial []
  (setv y (np.array [[2.0 0.0] [0.0 2.0]])
        x (np.array [[2.0 0.0] [0.0 4.0]]))
  (assert (np.allclose (⌹ x y) (np.array [[1.0 0.0] [0.0 2.0]])))
  (assert (np.allclose (! (F 2.0) (F 5.0)) (F 10.0))))
