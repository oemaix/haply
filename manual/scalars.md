# Scalars

Elementwise on `torch.Tensor`. Dyadic shapes follow PyTorch broadcast.
A Python number on one side is accepted and promoted.

Import the glyph, except `+` `-` `<` `>` `**` — those need the function
object (`haply.scalar` as `sc`). See [README](README.md#import).

## Arithmetic

| Glyph | Monad | Dyad | Fold |
| --- | --- | --- | --- |
| `+` | conjugate on complex; identity on reals (no copy) | add | yes |
| `-` | negate | subtract | |
| `×` | signum (`sgn`) | multiply | yes |
| `÷` | reciprocal | true divide | |
| `**` | `exp` | power | |
| `⍟` | natural log | log of `Y` in base `X` | |
| `\|\|` | magnitude | residue: remainder of `Y` by `X` (Dyalog argument order) | |
| `⌊` | floor (ints unchanged) | minimum | yes |
| `⌈` | ceiling (ints unchanged) | maximum | yes |

```hy
(import haply [× ÷ ⍟ || ⌊ ⌈])
(import haply.scalar :as sc)
(import torch)

(setv t (torch.tensor [1.0 4.0]))
(÷ t)                 ; [1.0, 0.25]
(× t 2)               ; [2.0, 8.0]
(sc.+ t t t)          ; fold → [3.0, 12.0]
(⍟ 2.0 (torch.tensor [2.0 4.0 8.0]))   ; [1, 2, 3]
(|| 3 (torch.tensor [5 7]))            ; [2, 1]  — Y rem X
```

`(sc.+ z)` on a real tensor returns that same object. On complex it
conjugates.

Residue follows Dyalog `X|Y`: `||` of `X=3` and `Y=-5` is `1`. The
sign matches `torch.remainder`.

## Compare

Exact. Result dtype is the backend boolean. No monadic form.

| Glyph | Dyad |
| --- | --- |
| `<` `≤` `==` `≥` `>` | elementwise relation |
| `≠` | elementwise `!=` (monadic unique-mask is in [structure.md](structure.md)) |

```hy
(import haply [≤ ==])
(import haply.scalar :as sc)

(== (torch.tensor [1 2]) (torch.tensor [1 3]))  ; [True, False]
(sc.< (torch.tensor [1 2 3]) 2)                 ; [True, False, False]
```

Use `==`, not Hy `=`. Use `sc.<` / `sc.>` when you need the Haply
function rather than Hy’s comparison.

## Logic

| Glyph | Monad | Dyad | Fold |
| --- | --- | --- | --- |
| `≁` | logical not | without: ravel of `X` not in `Y` | |
| `∧` | — | bool and; integer LCM | yes |
| `∨` | — | bool or; integer GCD | yes |
| `⍲` | — | nand | |
| `⍱` | — | nor | |

Floats and mixed bool/int raise `ValueError` on `∧` `∨`.

```hy
(import haply [≁ ∧ ∨])

(≁ (torch.tensor [False True]))          ; [True, False]
(≁ (torch.tensor [1 2 3 2]) (torch.tensor [2 4]))  ; [1 3]
(∧ (torch.tensor [True False]) True)     ; [True, False]
(∧ (torch.tensor [4 6]) (torch.tensor [6 9]))  ; LCM [12, 18]
```

## Circular and factorial

| Glyph | Monad | Dyad |
| --- | --- | --- |
| `○` | `π * Y` | circular table on a **single** integer code `X` |
| `!` | `Γ(Y+1)` (factorial on integers) | binomial `C(Y, X)` |

Shipped circular codes: `1` sin, `2` cos, `3` tan, `-1` asin, `-2`
acos, `-3` atan, `5` sinh, `6` cosh, `7` tanh. Other Dyalog codes
raise `ValueError`.

```hy
(import haply [○ !])
(import math)

(○ (torch.tensor [1.0]))                 ; [π]
(○ 1 (torch.tensor [0.0 (/ math.pi 2)])) ; [0, 1]
(! (torch.tensor [0.0 1.0 2.0 5.0]))     ; [1, 1, 2, 120]
(! 2.0 (torch.tensor [5.0]))             ; C(5, 2) → [10]
```
