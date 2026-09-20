# Recipes

Programs that run on today’s public API.

```hy
(import torch)
(import haply [⍴ × ÷ ⌽ ++ ⍪ ↑ ⊖ ≢ ⍳ ⊤ ⊥ ⊣ ⊢])
(import haply.scalar :as sc)
(require haply.macros [⌿ · ∘ ⍀])
(require haply.trains [fork])
```

## Square, then reverse last

Dyalog sketch `⌽ A × A` becomes prefix:

```hy
(setv A (torch.tensor [[1 2 3] [4 5 6]]))
(⌽ (× A A))
; [[ 9  4  1]
;  [36 25 16]]
```

## Reshape a ravel (no recycle)

```hy
(setv v (torch.arange 6))
(⍴ [2 3] v)
; [[0 1 2]
;  [3 4 5]]
```

`(⍴ [2 2] v)` fails: six elements do not fill four. There is no Dyalog
cycle.

## Glue a column, then take the first row

```hy
(setv M (torch.tensor [[1 2] [3 4]]))
(↑ 1 (++ M (torch.tensor [5 6])))
; [[1 2 5]]
```

## How many major cells?

```hy
(≢ (torch.tensor [[1 2 3] [4 5 6]]))   ; 2
```

`≢` is `shape[0]`, not `numel`.

## Haply plus as a function

Hy owns call-position `+`. Conjugate and a plus-fold use the object:

```hy
(sc.+ (torch.tensor [1+2j]))           ; [1-2j]
(sc.+ (torch.tensor [1]) (torch.tensor [2]) (torch.tensor [3]))
; [6]
```

Elementwise add of two tensors can stay `(+ t u)` — that is Python/Hy
`+`, which already works on tensors. Use `sc.+` when you mean the Haply
glyph (conjugate, fold, or passing the function as a value).

## Reduce a product; matrix product

Dyalog sketches `+⌿ A × B` and `A +.× B`:

```hy
(setv A (torch.tensor [[1 2] [3 4]] :dtype torch.float32)
      B A)
(⌿ + (× A B))          ; sum of squares per column
(· + × A B)            ; matmul
```

## Iota, then take

```hy
(↑ 3 (⍳ 8))            ; [0 1 2]
```

## Encode, then decode

```hy
(setv rad (torch.tensor [2 2 2 2])
      y (torch.tensor 10))
(⊥ rad (⊤ rad y))      ; 10
```

## Fork a plus; compose signum

```hy
(setv X (torch.tensor [1 2 3])
      Y (torch.tensor [10 20 30]))
(fork ⊣ + ⊢ X Y)       ; [11 22 33]
(∘ × + (torch.tensor [-2 0 4]))
; [-1 0 1]
```

## Expand a gap

```hy
(⍀ (torch.tensor [1 0 1]) (torch.tensor [7 8]))
; [7 0 8]
```
