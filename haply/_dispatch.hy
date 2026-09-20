(import functools)

(defn defprim [name monad dyad]
  "Strict monadic / dyadic wrapper (architecture helper)."
  (fn [#* args]
    (match (len args)
      1 (monad (get args 0))
      2 (dyad (get args 0) (get args 1))
      _ (raise (TypeError (.format "{} takes 1 or 2 arguments, got {}"
                                   name
                                   (len args)))))))

(defn deffold [name monad dyad]
  "Hy-style fold for associative scalar ops (decision 37 default B)."
  (fn [#* args]
    (setv n (len args))
    (cond
      (< n 1)
        (raise (TypeError (.format "{} takes at least 1 argument, got {}"
                                   name
                                   n)))
      (= n 1) (monad (get args 0))
      (= n 2) (dyad (get args 0) (get args 1))
      True (functools.reduce dyad args))))
