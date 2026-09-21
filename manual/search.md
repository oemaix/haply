# Search and grade

Index origin is 0. Results stay on the argument’s backend except where
a row says otherwise.

```hy
(import haply [⍳ ⍸ ∊ ⌷ ⊃ ∪ ∩ ⍋ ⍒ ≁ ⍷])
```

## Iota and where

| Glyph | Monad | Dyad |
| --- | --- | --- |
| `⍳` | index generator: `arange` for a length, stacked `meshgrid` for a shape | first index of each `Y` in a **1-d** `X`; not-found is `n` (length of `X`) |
| `⍸` | indices of truthy values (`nonzero`); 1-d flattens to a vector | interval index: bucketize / `searchsorted` of `Y` into sorted breakpoints `X` (left) |

Monadic `⍳` is a CPU `int64` torch vector by default. Pass
`:backend 'numpy` for a NumPy result. `X` may be a Python `int`, a
list/tuple of ints, or a 0-d/1-d integer array.

```hy
(⍳ 5)                    ; torch [0 1 2 3 4]
(⍳ 5 :backend 'numpy)    ; ndarray [0 1 2 3 4]
(⍳ [2 3])                ; shape [2 3 2]; last axis is the index pair
(⍳ (torch.tensor [10 20 30]) (torch.tensor [20 40 10]))
; [1 3 0]
(⍸ (torch.tensor [False True False True]))   ; [1 3]
(⍸ (torch.tensor [0 2 5]) (torch.tensor [-1 0 1 2 4 5 9]))
; [0 0 1 1 2 2 3]
```

## Membership, unique, grade

| Glyph | Monad | Dyad |
| --- | --- | --- |
| `∊` | flatten, C-order (same as `++`) | `isin`: which elements of `X` appear in `Y` |
| `∪` | unique values, first-occurrence ravel order | unique of the catenated ravels |
| `∩` | — | values of `X` that appear in `Y`, order and repeats from `X` |
| `⍋` | grade up (indices that sort `Y`) | *not shipped* (no collation alphabet) |
| `⍒` | grade down | *not shipped* |
| `≁` | logical not (see [scalars.md](scalars.md)) | without: ravel of `X` whose values are not in `Y` |

Grade is **numeric only**. Rank ≥ 2 grades major cells lexicographically.
Complex input is a `ValueError`.

```hy
(∊ (torch.tensor [1 2 3]) (torch.tensor [2 4]))   ; [False True False]
(∪ (torch.tensor [22 10 22 21 10]))               ; [22 10 21]
(∩ (torch.tensor [1 2 3 2]) (torch.tensor [2 4])) ; [2 2]
(⍋ (torch.tensor [30 10 20]))                     ; [1 2 0]
(≁ (torch.tensor [1 2 3 2]) (torch.tensor [2 4])) ; [1 3]
```

## Find

| Glyph | Dyad |
| --- | --- |
| `⍷` | boolean mask, same shape as `Y`, `True` where `X` begins as a sub-array |

If rank(`X`) is smaller than rank(`Y`), `X` is left-padded with 1s. If
rank(`X`) is larger, the mask is all `False`. Overlaps are reported at
every start index.

```hy
(⍷ (torch.tensor [1 2]) (torch.tensor [0 1 2 1 2 9]))
; [False True False True False False]
```

## Index and first

| Glyph | Monad | Dyad |
| --- | --- | --- |
| `⌷` | identity on a tensor | index `Y` by `X` (int, 0-d tensor, or a sequence of ints) |
| `⊃` | first major cell (the scalar itself if rank 0; empty first axis is `ValueError`) | *not shipped* (pick needs boxes) |

```hy
(setv m (torch.tensor [[1 2 3] [4 5 6]]))
(⌷ 1 m)          ; [4 5 6]
(⌷ [1 2] m)      ; 6
(⊃ m)            ; [1 2 3]
```
