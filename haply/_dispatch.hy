(import functools)
(import haply._backend [torch-monad torch-dyad])

(defn defprim [name monad dyad]
  "Strict monadic / dyadic wrapper (architecture helper)."
  (fn [#* args]
    (match (len args)
      1 (monad (get args 0))
      2 (dyad (get args 0) (get args 1))
      _ (raise (TypeError (.format "{} takes 1 or 2 arguments, got {}"
                                   name
                                   (len args)))))))

(defn defmonad [name monad]
  "Strict monadic wrapper."
  (fn [#* args]
    (if (= (len args) 1)
      (monad (get args 0))
      (raise (TypeError (.format "{} takes 1 argument, got {}"
                                 name
                                 (len args)))))))

(defn defdyad [name dyad]
  "Strict dyadic wrapper."
  (fn [#* args]
    (if (= (len args) 2)
      (dyad (get args 0) (get args 1))
      (raise (TypeError (.format "{} takes 2 arguments, got {}"
                                 name
                                 (len args)))))))

(defn deffold [name monad dyad]
  "Hy-style fold for associative scalar ops (decision 37)."
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

(defn tprim [name monad dyad]
  (defprim name (torch-monad name monad) (torch-dyad name dyad)))

(defn tmonad [name monad]
  (defmonad name (torch-monad name monad)))

(defn tdyad [name dyad]
  (defdyad name (torch-dyad name dyad)))

(defn tfold [name monad dyad]
  (deffold name (torch-monad name monad) (torch-dyad name dyad)))
