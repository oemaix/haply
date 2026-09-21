# Random, inverse, encode

```hy
(import haply [? ⌹ ⊤ ⊥])
```

## Roll and deal — `?`

Host RNG follows the array backend: `torch.manual_seed` for torch,
`numpy.random.seed` for NumPy. Python scalars still construct a torch
result. Index origin 0.

| Glyph | Monad | Dyad |
| --- | --- | --- |
| `?` | roll: each element of `Y` is an exclusive upper bound | deal: `X` distinct draws from `[0, Y)` |

`X` and `Y` for deal are scalars (Python `int` or 0-d tensor). `X > Y`
or a non-positive roll bound is a `ValueError`. There is no Dyalog
`?0` uniform-in-(0, 1) reading.

```hy
(? 6)                    ; one int in [0, 6)
(? (torch.tensor [6 6 6]))
(? 4 10)                 ; four distinct ints in [0, 10)
```

## Matrix inverse and divide — `⌹`

Rank at most 2.

| Glyph | Monad | Dyad |
| --- | --- | --- |
| `⌹` | inverse: `inv` if square 2-d, else `pinv`; 0-d is reciprocal | `(⌹ X Y)` solves `Y B = X` |

That dyad is Dyalog `X⌹Y` (`Y⁻¹X` when `Y` is square; `lstsq` when
tall). `X` and `Y` must share the number of rows.

```hy
(setv M (torch.tensor [[2.0 0.0] [0.0 4.0]]))
(⌹ M)                    ; [[0.5 0.0] [0.0 0.25]]
(⌹ (torch.tensor [2.0 4.0]) (torch.tensor [[2.0 0.0] [0.0 2.0]]))
; [1.0 2.0]
```

## Encode and decode — `⊤` `⊥`

Dyadic only. `X` is rank 0 or 1.

| Glyph | Dyad |
| --- | --- |
| `⊤` | digits of `Y` in mixed radix `X`; result shape `shape(X)+shape(Y)` |
| `⊥` | value of digits `Y` in radix `X` (Horner along axis 0 of `Y`) |

A leading 0 in `X` for encode keeps the remaining value. A scalar radix
for decode repeats along that axis. The first element of a radix
*vector* does not change decode.

```hy
(⊤ (torch.tensor [2 2 2]) 5)     ; [1 0 1]
(⊤ 10 (torch.tensor [5 15 125])) ; [5 5 5]
(⊥ 2 (torch.tensor [1 0 1 0]))   ; 10
(⊥ (torch.tensor [60 60]) (torch.tensor [3 13]))  ; 193
```
