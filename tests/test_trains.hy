(require haply.macros [∘ ⍤ ⍥ ⍛])
(require haply.trains [fork])
(import haply [× ⊣ ⊢])
(import haply.scalar :as sc)
(import torch)
(import pytest)

(defn T [#* xs]
  (torch.tensor (list xs)))

;; --- done criterion -------------------------------------------------------

(defn test-fork-plus-wings []
  (setv X (T 1 2 3)
        Y (T 10 20 30))
  (assert (torch.equal (fork ⊣ + ⊢ X Y) (+ X Y))))

(defn test-beside-times-plus []
  (setv Y (T -2 0 4))
  (assert (torch.equal (∘ × + Y) (× (+ Y)))))

;; --- fork -----------------------------------------------------------------

(defn test-fork-monad []
  (assert (torch.equal (fork ⊢ + ⊢ (T 1 2 3)) (T 2 4 6))))

(defn test-fork-constant-wing []
  (assert (torch.equal (fork 0 + ⊢ (T 1 2 3)) (T 1 2 3))))

(defn test-fork-too-long []
  (with [(pytest.raises TypeError)]
    (fork ⊣ + ⊢ ⊣ (T 1) (T 2))))

;; --- beside / bind --------------------------------------------------------

(defn test-beside-dyad []
  (setv X (T 10 20)
        Y (T -2 4))
  (assert (torch.equal (∘ sc.- × X Y) (T 11 19))))

(defn test-bind-left []
  (assert (torch.equal (∘ 2 × (T 3 4)) (T 6 8))))

(defn test-bind-right []
  (assert (torch.equal (∘ × 2 (T 3 4)) (T 6 8))))

(defn test-beside-user-fn []
  (setv sign ×)
  (assert (torch.equal (∘ sign × (T -2 0 4)) (× (× (T -2 0 4))))))

(defn test-beside-error []
  (with [(pytest.raises TypeError)]
    (∘ ×)))

;; --- atop / over / behind -------------------------------------------------

(defn test-atop-monad []
  (assert (torch.equal (⍤ × + (T -2 0 4)) (× (+ (T -2 0 4))))))

(defn test-atop-dyad []
  (assert (torch.equal (⍤ sc.- × (T 3 4) (T 1 1)) (T -3 -4))))

(defn test-atop-rank-later []
  (with [(pytest.raises TypeError)]
    (⍤ × 1 (T 1 2))))

(defn test-over-dyad []
  (assert (torch.equal (⍥ + × (T -2 3) (T 4 0)) (T 0 1))))

(defn test-behind-monad []
  (assert (torch.equal (⍛ × + (T -2 0 4)) (× (+ (T -2 0 4))))))

(defn test-behind-dyad []
  (assert (torch.equal (⍛ + ⊢ (T 2 3) (T 10 20)) (T 12 23))))
