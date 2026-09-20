# Structure

Shape and layout on `torch.Tensor`. Rank is `tensor.ndim`. Axis `0` is
the first (major) axis; `-1` is the last.

```hy
(import haply [⍴ ++ ⍪ ⌽ ⊖ ⍉ ↑ ↓ ⊢ ⊣ ≠ ≡ ≢])
```

## Shape and ravel

| Glyph | Monad | Dyad |
| --- | --- | --- |
| `⍴` | shape as a 1-d `int64` tensor (empty for a 0-d scalar) | reshape `Y` to `X`; **element count must match** |
| `++` | ravel, C-order flatten | catenate on the last axis |
| `⍪` | table: matrix with axis 0 preserved | catenate on axis 0 |

`X` for reshape may be a Python `int`, a list/tuple of ints, or a 0-d/1-d
integer tensor.

**Deviation:** Dyalog reshape recycles `Y` to fill `X`. Haply does not.
A size mismatch raises a backend `RuntimeError`.

```hy
(setv A (torch.tensor [[1 2 3] [4 5 6]]))
(⍴ A)                 ; tensor([2, 3])
(⍴ [3 2] (++ A))      ; [[1 2] [3 4] [5 6]]
(⍴ [2 2] (torch.tensor [1 2 3]))   ; error — no recycle
```

Catenate (`++` last, `⍪` first) allows rank difference 0 or 1 (the
shorter side is unsqueezed on the join axis). Two 0-d values stack to a
length-2 vector. Same-shape arrays **catenate**; they do not laminate
onto a new axis.

```hy
(++ (torch.tensor [1 2]) (torch.tensor [3 4]))     ; [1 2 3 4]
(++ (torch.tensor [[1 2] [3 4]]) (torch.tensor [5 6]))
; [[1 2 5] [3 4 6]]
(⍪ (torch.tensor [1 2 3]))                         ; column [[1] [2] [3]]
```

## Reverse, rotate, transpose

| Glyph | Monad | Dyad |
| --- | --- | --- |
| `⌽` | reverse last axis | rotate last axis by scalar `X` |
| `⊖` | reverse axis 0 | rotate axis 0 by scalar `X` |
| `⍉` | reverse the axis order | permute axes; `X` is a 0-based permutation of `range(rank)` |

Positive rotate shifts toward the start of the axis (`1 ⌽ 1 2 3 4` →
`2 3 4 1`). A 0-d argument is returned unchanged. A non-scalar rotate
`X` is a `ValueError` in this phase.

```hy
(⌽ (torch.tensor [[1 2 3] [4 5 6]]))     ; [[3 2 1] [6 5 4]]
(⌽ 1 (torch.tensor [1 2 3 4]))           ; [2 3 4 1]
(⊖ (torch.tensor [[1 2 3] [4 5 6]]))     ; [[4 5 6] [1 2 3]]
(⍉ (torch.tensor [[1 2 3] [4 5 6]]))     ; [[1 4] [2 5] [3 6]]
(⍉ [1 0 2] cube)                         ; swap the first two axes
```

## Take and drop

Monadic mix/split are **not** implemented. Both glyphs are dyadic only.

| Glyph | Dyad |
| --- | --- |
| `↑` | take along leading axes given by `X` |
| `↓` | drop along leading axes given by `X` |

`X` is an int or a sequence of ints, one per leading axis. Negative
counts take or drop from the end. Overtake (`↑` longer than the axis)
pads with **backend zero**. Overdrop yields an empty axis.

```hy
(↑ 3 (torch.tensor [1 2 3 4 5]))         ; [1 2 3]
(↑ -2 (torch.tensor [1 2 3 4 5]))        ; [4 5]
(↑ 5 (torch.tensor [1 2 3]))             ; [1 2 3 0 0]
(↓ [1 1] (torch.tensor [[1 2 3 4] [5 6 7 8] [9 10 11 12]]))
; [[6 7 8] [10 11 12]]
```

## Identity, match, tally

| Glyph | Monad | Dyad |
| --- | --- | --- |
| `⊢` | identity | right (`Y`) |
| `⊣` | identity | left (`X`) |
| `≡` | *dropped* (no depth) | whole-array match → Python `bool` |
| `≢` | tally: `shape[0]`, or `1` if 0-d → Python `int` | not-match → Python `bool` |
| `≠` | first-occurrence mask of the C-order ravel, **same shape as `Y`** | elementwise `!=` |

`⊢` and `⊣` accept any host value, not only tensors.

**Deviation:** monadic `≠` is a ravel unique-mask, not Dyalog’s
major-cell unique mask.

```hy
(≢ (torch.tensor [[1 2 3] [4 5 6]]))     ; 2  — not 6
(≡ (torch.tensor [1 2]) (torch.tensor [1 2]))   ; True
(≠ (torch.tensor [22 10 22 21 10]))
; [True True False True False]
(≠ (torch.tensor [[1 2] [1 3]]))
; [[True True] [False True]]
```
