(import haply [○ !])
(import math)
(import torch)
(import pytest)

(defn F [#* xs]
  (torch.tensor (list xs) :dtype torch.float64))

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
