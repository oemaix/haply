# Operators

Operators are **macros**. Require them; do not `import` them.

```hy
(import torch)
(import haply [× ⌽])
(require haply.macros [⌿ ⌿_ ⍀ ⍀_ ⍨ · outer ¨])
```

Known operands (`+` `×` `⌈` `⌊` for reduce/scan; `+ ×` for inner
product) expand to `torch` kernels. Other operands run a cell loop.
Kernel expansions use the name `torch`, so import it in the same file.

## Axis

| Form | Axis |
| --- | --- |
| `⌿` `⍀` | 0 (first) |
| `⌿_` `⍀_` | −1 (last) |

## Reduce and replicate — `⌿` / `⌿_`

A function operand reduces. An array operand replicates (boolean
compress, or integer repeat). Negative repeats are a `ValueError`.

```hy
(setv A (torch.tensor [[1 2 3] [4 5 6]]))
(⌿ + A)                          ; sum axis 0 → [5 7 9]
(⌿_ + A)                         ; sum last → [6 15]
(⌿ × (torch.tensor [2 3 4]))     ; 24
(⌿ (torch.tensor [True False True]) A)   ; rows 0 and 2
(⌿ (torch.tensor [1 0 2]) (torch.tensor [7 8 9]))  ; [7 9 9]
(⌿ 2 + (torch.tensor [1 2 3 4])) ; n-wise windows of 2 → [3 5 7]
```

Empty reduce of `+` / `×` follows the `torch` identity (0 / 1). Empty
reduce of an unknown function is a `ValueError`. n-wise `n` must be
positive.

`(⌿ + (× A B))` is `torch.sum` of the product on axis 0.

## Scan — `⍀` / `⍀_`

Cumulative `+` `×` `⌈` `⌊` along the axis. An array operand (expand) is
not shipped.

```hy
(⍀ + (torch.tensor [1 2 3 4]))   ; [1 3 6 10]
```

## Commute — `⍨`

```hy
(⍨ × Y)        ; (× Y Y)  selfie
(⍨ f X Y)      ; (f Y X)  commute
(⍨ A Y)        ; A        constant (A is not a function name)
```

Hy-macro operands such as `sc.-` go through the commute helper, not a
kernel rewrite.

## Inner and outer product

```hy
(· + × X Y)        ; torch.matmul
(outer × X Y)      ; all-pairs times; shape shape(X)+shape(Y)
(outer + X Y)      ; all-pairs plus
```

The catalog name for outer is `∘.`. Hy cannot parse that identifier; the
macro is `outer`.

## Each — `¨`

A Haply scalar primitive expands to the elementwise call (no loop).
Anything else maps over major cells and `stack`s.

```hy
(¨ × (torch.tensor [-2 0 4]))    ; signum, elementwise
(¨ ⌽ (torch.tensor [[1 2 3] [4 5 6]]))
; [[3 2 1] [6 5 4]]
```
