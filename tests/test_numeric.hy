(import haply [○ ! ? ⌹ ⊤ ⊥])
(import math)
(import torch)
(import pytest)

(defn F [#* xs]
  (torch.tensor (list xs) :dtype torch.float64))

(defn T [#* xs]
  (torch.tensor (list xs)))

;; --- G010 ○ ---------------------------------------------------------------

(defn test-pi-times []
  (assert (torch.allclose (○ (F 0.5 1.0 2.0))
                          (F (* 0.5 math.pi) math.pi (* 2.0 math.pi)))))

(defn test-circ-sin-cos []
  (setv y (F 0.0 (/ math.pi 2)))
  (assert (torch.allclose (○ 1 y) (F 0.0 1.0)))
  (assert (torch.allclose (○ 2 (F 0.0)) (F 1.0))))

(defn test-circ-tan []
  (assert (torch.allclose (○ 3 (F 0.0)) (F 0.0))))

(defn test-circ-inverse []
  (assert (torch.allclose (○ -1 (F 0.0)) (F 0.0)))
  (assert (torch.allclose (○ -2 (F 1.0)) (F 0.0)))
  (assert (torch.allclose (○ -3 (F 0.0)) (F 0.0))))

(defn test-circ-hyperbolic []
  (assert (torch.allclose (○ 5 (F 0.0)) (F 0.0)))
  (assert (torch.allclose (○ 6 (F 0.0)) (F 1.0)))
  (assert (torch.allclose (○ 7 (F 0.0)) (F 0.0))))

(defn test-circ-broadcast []
  (assert (torch.allclose (○ 1 (F 0.0 0.0)) (F 0.0 0.0))))

(defn test-circ-unknown-code []
  (with [(pytest.raises ValueError)]
    (○ 4 (F 0.5))))

(defn test-circ-error []
  (with [(pytest.raises TypeError)]
    (○)))

;; --- G011 ! ---------------------------------------------------------------

(defn test-factorial []
  (assert (torch.allclose (! (F 0.0 1.0 2.0 3.0 4.0 5.0))
                          (F 1.0 1.0 2.0 6.0 24.0 120.0))))

(defn test-factorial-half []
  (assert (torch.allclose (! (F -1.5)) (F (math.gamma -0.5)))))

(defn test-binomial []
  (assert (torch.allclose (! (F 2.0) (F 5.0)) (F 10.0))))

(defn test-binomial-broadcast []
  (assert (torch.allclose (! 2.0 (F 3.0 4.0 5.0)) (F 3.0 6.0 10.0))))

(defn test-factorial-error []
  (with [(pytest.raises TypeError)]
    (! (F 1.0) (F 2.0) (F 3.0))))

;; --- G012 ? ---------------------------------------------------------------

(defn test-roll-range []
  (torch.manual-seed 0)
  (setv r (? (torch.tensor [6 6 6 6 6 6])))
  (assert (= (list r.shape) [6]))
  (assert (torch.all (torch.ge r 0)))
  (assert (torch.all (torch.lt r 6))))

(defn test-roll-scalar []
  (torch.manual-seed 0)
  (setv r (? 6))
  (assert (= r.ndim 0))
  (assert (and (>= (int (.item r)) 0) (< (int (.item r)) 6))))

(defn test-deal []
  (torch.manual-seed 0)
  (setv r (? 4 10))
  (assert (= (list r.shape) [4]))
  (assert (= (.numel (torch.unique r)) 4))
  (assert (torch.all (torch.ge r 0)))
  (assert (torch.all (torch.lt r 10))))

(defn test-deal-empty []
  (assert (torch.equal (? 0 5) (torch.tensor [] :dtype torch.int64))))

(defn test-deal-too-many []
  (with [(pytest.raises ValueError)]
    (? 5 3)))

(defn test-roll-nonpositive []
  (with [(pytest.raises ValueError)]
    (? 0)))

(defn test-roll-error []
  (with [(pytest.raises TypeError)]
    (?)))

;; --- G045 ⌹ ---------------------------------------------------------------

(defn test-inv-square []
  (setv m (torch.tensor [[2.0 0.0] [0.0 4.0]]))
  (assert (torch.allclose (⌹ m) (torch.tensor [[0.5 0.0] [0.0 0.25]]))))

(defn test-inv-scalar []
  (assert (torch.allclose (⌹ (torch.tensor 4.0)) (torch.tensor 0.25))))

(defn test-solve []
  (setv y (torch.tensor [[2.0 0.0] [0.0 2.0]])
        x (torch.tensor [[2.0 0.0] [0.0 4.0]]))
  (assert (torch.allclose (⌹ x y) (torch.tensor [[1.0 0.0] [0.0 2.0]]))))

(defn test-solve-vector []
  (setv y (torch.tensor [[2.0 0.0] [0.0 2.0]])
        x (torch.tensor [2.0 4.0]))
  (assert (torch.allclose (⌹ x y) (torch.tensor [1.0 2.0]))))

(defn test-domino-error []
  (with [(pytest.raises TypeError)]
    (⌹)))

;; --- G046 ⊤ / G047 ⊥ ------------------------------------------------------

(defn test-encode-binary []
  (assert (torch.equal (⊤ (T 2 2 2) 5) (T 1 0 1))))

(defn test-encode-scalar-radix []
  (assert (torch.equal (⊤ 10 (T 5 15 125)) (T 5 5 5))))

(defn test-encode-leading-zero []
  (assert (torch.equal (⊤ (T 0 10) (T 5 15 125))
                       (torch.tensor [[0 1 12] [5 5 5]]))))

(defn test-decode-binary []
  (assert (torch.equal (⊥ 2 (T 1 0 1 0)) (torch.tensor 10))))

(defn test-decode-mixed []
  (assert (torch.equal (⊥ (T 60 60) (T 3 13)) (torch.tensor 193))))

(defn test-decode-matrix []
  (setv bits (torch.tensor [[0 0 1 1] [0 1 0 1]]))
  (assert (torch.equal (⊥ 2 bits) (T 0 1 2 3))))

(defn test-encode-decode-roundtrip []
  (setv rad (T 2 2 2 2)
        y (torch.tensor 10))
  (assert (torch.equal (⊥ rad (⊤ rad y)) y)))

(defn test-encode-error []
  (with [(pytest.raises TypeError)]
    (⊤ (T 2 2 2))))

(defn test-decode-error []
  (with [(pytest.raises TypeError)]
    (⊥ 2)))
