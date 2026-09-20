# Operators

Operators are **macros**. Require them; do not `import` them.

```hy
(import torch)
(import haply [× ⌽])
(require haply.macros [⌿ ⌿_ ⍀ ⍀_ ⍨ · outer ¨ ∘ ⍤ ⍥ ⍛])
(require haply.trains [fork])
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

Cumulative `+` `×` `⌈` `⌊` along the axis.

```hy
(⍀ + (torch.tensor [1 2 3 4]))   ; [1 3 6 10]
```

## Expand — array operand on `⍀` / `⍀_`

A boolean or 0/1 integer mask inserts backend-zero fill where the mask
is 0. The number of 1s must match the length of `Y` along the axis.

```hy
(⍀ (torch.tensor [1 0 1]) (torch.tensor [7 8]))
; [7 0 8]
(⍀_ (torch.tensor [1 0 1]) (torch.tensor [[1 2] [3 4]]))
; [[1 0 2]
;  [3 0 4]]
```

Negative values and integers other than 0 or 1 are a `ValueError`.

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

## Beside and bind — `∘`

Two functions compose. One function and one array bind.

```hy
(∘ × + Y)        ; (× (+ Y))
(∘ 2 × Y)        ; (× 2 Y)
(∘ × 2 Y)        ; (× Y 2)
(∘ sc.- × X Y)   ; (sc.- X (× Y))
```

Hy call-position `+` is identity on reals, so `(∘ × + Y)` is signum of `Y`.
A function stored in a name still composes: `(∘ sign × Y)` if `sign` is
callable.

## Atop — `⍤`

```hy
(⍤ × + Y)              ; (× (+ Y))
(⍤ sc.- × X Y)         ; (sc.- (× X Y))
```

An integer right operand is rank, and is a `TypeError` in this phase.

## Over — `⍥`

```hy
(⍥ + × X Y)            ; (+ (× X) (× Y))
```

## Behind — `⍛`

Haply, not Dyalog: the monad is `(f (g Y))`; the dyad is `((g X) f Y)`.

```hy
(⍛ × + Y)              ; (× (+ Y))
(⍛ + ⊢ X Y)            ; (+ X Y)
```

## Fork — `(fork …)`

The 3-train only. Require `haply.trains`, not `haply.macros`.

```hy
(fork ⊣ + ⊢ X Y)       ; (+ X Y)
(fork ⊢ + ⊢ Y)         ; (+ Y Y)
(fork 0 + ⊢ Y)         ; (+ 0 Y)
```

A longer fork is a `TypeError`. Wings that are known glyphs or integer
literals expand in place; anything else is a constant unless it is
callable.

Call-position `+` in a fork is Hy `+`, which already adds tensors.
