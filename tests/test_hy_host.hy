;; This module does not import Haply names. Hy must keep its own meaning
;; (design principle 6, decision 8, Phase 1 done criterion).

(defn test-hy-plus-without-haply []
  (assert (= (+ 1 2 3) 6)))

(defn test-hy-minus-without-haply []
  (assert (= (- 5) -5))
  (assert (= (- 5 2) 3)))

(defn test-hy-compare-without-haply []
  (assert (< 1 2))
  (assert (> 2 1)))
