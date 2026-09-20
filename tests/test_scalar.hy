(import haply [× ÷ ⍟ || ⌊ ⌈ ≤ == ≥ ≁ ∧ ∨ ⍲ ⍱])
(import haply.scalar :as sc)
(import math)
(import torch)
(import pytest)

(defn T [#* xs]
  (torch.tensor (list xs)))

(defn F [#* xs]
  (torch.tensor (list xs) :dtype torch.float32))

;; --- G001 + ---------------------------------------------------------------

(defn test-plus-monad-real []
  (setv t (T 1 2))
  (assert (is (sc.+ t) t)))

(defn test-plus-monad-complex []
  (setv z (torch.tensor [1+2j]))
  (assert (torch.equal (sc.+ z) (torch.tensor [1-2j]))))

(defn test-plus-dyad []
  (assert (torch.equal (sc.+ (T 1 2) (T 3 4)) (T 4 6))))

(defn test-plus-broadcast []
  (assert (torch.equal (sc.+ (T 1 2 3) 10) (T 11 12 13))))

(defn test-plus-fold []
  (assert (torch.equal (sc.+ (T 1) (T 2) (T 3)) (T 6))))

(defn test-plus-error []
  (with [(pytest.raises TypeError)]
    (sc.+)))

;; --- G002 - ---------------------------------------------------------------

(defn test-minus-monad []
  (assert (torch.equal (sc.- (T 1 -2)) (T -1 2))))

(defn test-minus-dyad []
  (assert (torch.equal (sc.- (T 5 5) (T 1 3)) (T 4 2))))

(defn test-minus-broadcast []
  (assert (torch.equal (sc.- (T 5 6) 1) (T 4 5))))

(defn test-minus-error []
  (with [(pytest.raises TypeError)]
    (sc.- (T 1) (T 2) (T 3))))

;; --- G003 × ---------------------------------------------------------------

(defn test-times-monad []
  (assert (torch.equal (× (F -2.0 0.0 4.0)) (F -1.0 0.0 1.0))))

(defn test-times-monad-complex []
  (setv z (torch.tensor [3+4j]))
  (assert (torch.allclose (× z) (torch.tensor [0.6+0.8j]))))

(defn test-times-dyad []
  (assert (torch.equal (× (T 2 3) (T 4 5)) (T 8 15))))

(defn test-times-broadcast []
  (assert (torch.equal (× (T 1 2 3) 2) (T 2 4 6))))

(defn test-times-error []
  (with [(pytest.raises TypeError)]
    (×)))

;; --- G004 ÷ ---------------------------------------------------------------

(defn test-div-monad []
  (assert (torch.allclose (÷ (F 2.0 4.0)) (F 0.5 0.25))))

(defn test-div-dyad []
  (assert (torch.allclose (÷ (F 6.0 8.0) (F 3.0 2.0)) (F 2.0 4.0))))

(defn test-div-broadcast []
  (assert (torch.allclose (÷ (F 2.0 4.0) 2.0) (F 1.0 2.0))))

(defn test-div-error []
  (with [(pytest.raises TypeError)]
    (÷ (F 1.0) (F 2.0) (F 3.0))))

;; --- G005 ** --------------------------------------------------------------

(defn test-starstar-monad []
  (assert (torch.allclose (sc.** (F 0.0 1.0)) (F 1.0 math.e))))

(defn test-starstar-dyad []
  (assert (torch.allclose (sc.** (F 2.0 3.0) (F 3.0 2.0)) (F 8.0 9.0))))

(defn test-starstar-broadcast []
  (assert (torch.allclose (sc.** (F 2.0 3.0) 2.0) (F 4.0 9.0))))

(defn test-starstar-error []
  (with [(pytest.raises TypeError)]
    (sc.** (F 1.0) (F 2.0) (F 3.0))))

;; --- G006 ⍟ ---------------------------------------------------------------

(defn test-log-monad []
  (assert (torch.allclose (⍟ (F 1.0 math.e)) (F 0.0 1.0))))

(defn test-log-dyad []
  (assert (torch.allclose (⍟ (F 10.0) (F 100.0)) (F 2.0))))

(defn test-log-broadcast []
  (assert (torch.allclose (⍟ 2.0 (F 2.0 4.0 8.0)) (F 1.0 2.0 3.0))))

(defn test-log-error []
  (with [(pytest.raises TypeError)]
    (⍟)))

;; --- G007 || --------------------------------------------------------------

(defn test-stile-monad []
  (assert (torch.allclose (|| (F -3.0 4.0)) (F 3.0 4.0))))

(defn test-stile-monad-complex []
  (assert (torch.allclose (|| (torch.tensor [3+4j])) (torch.tensor [5.0]))))

(defn test-stile-residue []
  ;; Dyalog 3 3 ¯3 ¯3 | ¯5 5 ¯4 4  →  1 2 ¯1 ¯2
  (assert (torch.equal (|| (T 3 3 -3 -3) (T -5 5 -4 4)) (T 1 2 -1 -2))))

(defn test-stile-broadcast []
  (assert (torch.equal (|| 3 (T 5 7)) (T 2 1))))

(defn test-stile-error []
  (with [(pytest.raises TypeError)]
    (|| (T 1) (T 2) (T 3))))

;; --- G008 ⌊ ---------------------------------------------------------------

(defn test-floor-monad []
  (assert (torch.allclose (⌊ (F 1.9 -1.1)) (F 1.0 -2.0))))

(defn test-floor-monad-int []
  (setv t (T 3 4))
  (assert (is (⌊ t) t)))

(defn test-floor-min []
  (assert (torch.equal (⌊ (T 1 8) (T 4 3)) (T 1 3))))

(defn test-floor-broadcast []
  (assert (torch.equal (⌊ (T 1 8 3) 2) (T 1 2 2))))

(defn test-floor-error []
  (with [(pytest.raises TypeError)]
    (⌊)))

;; --- G009 ⌈ ---------------------------------------------------------------

(defn test-ceil-monad []
  (assert (torch.allclose (⌈ (F 1.1 -1.9)) (F 2.0 -1.0))))

(defn test-ceil-max []
  (assert (torch.equal (⌈ (T 1 8) (T 4 3)) (T 4 8))))

(defn test-ceil-broadcast []
  (assert (torch.equal (⌈ (T 1 8 3) 2) (T 2 8 3))))

(defn test-ceil-error []
  (with [(pytest.raises TypeError)]
    (⌈)))

;; --- G013–G017 ------------------------------------------------------------

(defn test-lt []
  (assert (torch.equal (sc.< (T 1 3) (T 2 2)) (torch.tensor [True False]))))

(defn test-le []
  (assert (torch.equal (≤ (T 1 2) (T 2 2)) (torch.tensor [True True]))))

(defn test-eq []
  (assert (torch.equal (== (T 1 2) (T 1 3)) (torch.tensor [True False]))))

(defn test-ge []
  (assert (torch.equal (≥ (T 1 3) (T 2 2)) (torch.tensor [False True]))))

(defn test-gt []
  (assert (torch.equal (sc.> (T 1 3) (T 2 2)) (torch.tensor [False True]))))

(defn test-lt-broadcast []
  (assert (torch.equal (sc.< (T 1 2 3) 2) (torch.tensor [True False False]))))

(defn test-eq-error []
  (with [(pytest.raises TypeError)]
    (== (T 1)))
  (with [(pytest.raises TypeError)]
    (sc.< (T 1))))

;; --- G021 ≁ ---------------------------------------------------------------

(defn test-not-monad []
  (assert (torch.equal (≁ (torch.tensor [False True]))
                       (torch.tensor [True False]))))

(defn test-not-error []
  (with [(pytest.raises TypeError)]
    (≁)))

;; --- G022 ∧ / G023 ∨ ------------------------------------------------------

(defn test-and-bool []
  (assert (torch.equal (∧ (torch.tensor [True False True])
                          (torch.tensor [True True False]))
                       (torch.tensor [True False False]))))

(defn test-or-bool []
  (assert (torch.equal (∨ (torch.tensor [True False True])
                          (torch.tensor [False False True]))
                       (torch.tensor [True False True]))))

(defn test-and-lcm []
  (assert (torch.equal (∧ (T 4 6) (T 6 9)) (T 12 18))))

(defn test-or-gcd []
  (assert (torch.equal (∨ (T 4 6) (T 6 9)) (T 2 3))))

(defn test-and-broadcast []
  (assert (torch.equal (∧ (torch.tensor [True False]) True)
                       (torch.tensor [True False]))))

(defn test-and-float-error []
  (with [(pytest.raises ValueError)]
    (∧ (F 1.0) (F 2.0))))

(defn test-or-float-error []
  (with [(pytest.raises ValueError)]
    (∨ (F 1.0) (F 2.0))))

(defn test-and-monad-error []
  (with [(pytest.raises TypeError)]
    (∧ (torch.tensor [True]))))

;; --- G024 ⍲ / G025 ⍱ ------------------------------------------------------

(defn test-nand []
  (assert (torch.equal (⍲ (torch.tensor [True False])
                          (torch.tensor [True True]))
                       (torch.tensor [False True]))))

(defn test-nor []
  (assert (torch.equal (⍱ (torch.tensor [True False])
                          (torch.tensor [False False]))
                       (torch.tensor [False True]))))

(defn test-nand-broadcast []
  (assert (torch.equal (⍲ (torch.tensor [True False]) False)
                       (torch.tensor [True True]))))

(defn test-nand-error []
  (with [(pytest.raises TypeError)]
    (⍲ (torch.tensor [True]))))
