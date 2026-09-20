# Operators and trains

APL *operators* take functions (and sometimes arrays) and produce a derived
operation. APL *trains* combine functions without a dedicated operator
glyph.

In Haply both families are **Hy macros** (decision 7). They should expand
to backend code when the operand is a known Haply primitive.

## Calling convention

Dyalog writes the operand to the left of a monadic operator:

```apl
+/Y
+⌿Y
+.×
```

Haply writes the operator as the head (Lisp prefix):

```hy
(⌿_ + Y)      ; reduce last  — Dyalog +/Y
(⌿  + Y)      ; reduce first — Dyalog +⌿Y
(·  + × X Y)  ; inner product
```

The first operand(s) after the operator are functions (or arrays, for
replicate/expand and bind). The remaining arguments are the derived
function’s arguments.

## Axis

v1 provides two axes only (decisions 15, 20, 34):

| Form | Axis |
| --- | --- |
| `⌿` `⍀` | 0 (first) |
| `⌿_` `⍀_` | −1 (last) |

Do not implement Dyalog `f[k]` until decision 34 is closed. Leave room in
the macro for an optional axis keyword.

---

## Reduce — `⌿` / `⌿_`

| | |
| --- | --- |
| Dyalog | `f/Y`, `f⌿Y`, n-wise `X f/Y` |
| Haply | `(⌿ f Y)`, `(⌿_ f Y)`, `(⌿ n f Y)` |
| Phase | 3 |
| Status | implement |

**Intent.** Collapse one axis by repeating dyadic `f`.

- `(⌿ + Y)` → `Y.sum(dim=0)` when `f` is Haply `+`; otherwise a documented
  reduction loop over major cells.
- `(⌿_ + Y)` → `Y.sum(dim=-1)`.
- Empty axis: decision 41 (identities for known scalar `f`; error for
  unknown `f`).

**N-wise reduce.** `(⌿ n + Y)` reduces windows of length `n` along the
axis. Negative `n` follows Dyalog’s reverse-window convention only if
tests pin it down in Phase 3; otherwise implement positive `n` first.

**Not Dyalog.** No prototype identity for empty nested arrays. No `/`.

## Replicate — same glyphs, array operand

| | |
| --- | --- |
| Dyalog | `X/Y`, `X⌿Y` |
| Haply | `(⌿ X Y)`, `(⌿_ X Y)` when `X` is an array |
| Phase | 3 |
| Status | implement (decision 60 default) |

**Intent.** If `X` is boolean, compress (keep along the axis). If `X` is
integer, repeat each slice `X[i]` times (negative integers are an error in
the working default, unless Phase 3 adopts Dyalog’s reverse-and-repeat).

The macro distinguishes function operand vs array operand at expansion
time when it can; otherwise at runtime.

## Scan — `⍀` / `⍀_`

| | |
| --- | --- |
| Dyalog | `f\Y`, `f⍀Y` |
| Haply | `(⍀ f Y)`, `(⍀_ f Y)` |
| Phase | 3 |
| Status | implement |

**Intent.** Cumulative application of `f` along the axis, keeping rank.

- `(⍀_ + Y)` → `cumsum` on the last dim when `f` is `+`.
- Known operands (`+`, `×`, `⌈`, `⌊`) should expand to `cumprod` /
  `cummax` / `cummin` when those kernels exist.
- Unknown operands: explicit prefix loop.

## Expand — array operand on `⍀` / `⍀_`

| | |
| --- | --- |
| Dyalog | `X\Y`, `X⍀Y` |
| Haply | `(⍀ X Y)`, `(⍀_ X Y)` |
| Phase | 3 |
| Status | later |

**Intent.** Insert fill along the axis where `X` is 0. Fill is the backend
zero (decision 33), not an APL prototype.

---

## Each — `¨`

| | |
| --- | --- |
| Dyalog | `f¨Y`, `X f¨Y` |
| Haply | `(¨ f Y)`, `(¨ f X Y)` |
| Phase | 3 |
| Status | undecided (42) |

**Working intent.** Map `f` over major cells and stack the results. If `f`
is a Haply scalar function, the macro should refuse to loop and should
emit the elementwise call instead.

Dyalog Each over boxes is not implementable: there are no boxes.

---

## Commute / selfie / constant — `⍨`

| | |
| --- | --- |
| Dyalog | `f⍨Y` (selfie), `X f⍨Y` (commute), `A⍨` (constant) |
| Haply | `(⍨ f Y)` → `(f Y Y)`; `(⍨ f X Y)` → `(f Y X)`; `(⍨ A Y)` → `A` |
| Phase | 3 |
| Status | implement |

**Intent.** Swap arguments, or pair the single argument with itself. Array
operand makes a constant function.

---

## Inner product — `·`

| | |
| --- | --- |
| Dyalog | `X f.g Y` |
| Haply | `(· f g X Y)` |
| Phase | 3 |
| Status | implement |

**Intent.** The Dyalog inner product: apply `g` between cells, then reduce
with `f`.

Special case that must expand to a kernel:

```hy
(· + × X Y)   ; matrix product / batched matmul
```

That form is `torch.matmul` / `np.matmul` (broadcasting, not APL inner
product conformability).

General `f`/`g` may be slower and should still have a defined cell rule:
contract the last axis of `X` with the first axis of `Y` (working default,
matching common APL `+.×` on matrices). Confirm in Phase 3 tests.

Hy `.` is never this operator. `@` is not this operator either; it stays
Hy/Python matmul. Function combination is jot `∘` (decisions 13 and 23).

---

## Outer product — `∘.`

| | |
| --- | --- |
| Dyalog | `X ∘.g Y` |
| Haply | `(∘. g X Y)` (decision 59) |
| Phase | 3 |
| Status | implement |

**Intent.** All-pairs `g` on elements (or on cells, if we document cells).
Working default: elementwise outer — result shape `shape(X) + shape(Y)`,
like `np.multiply.outer` when `g` is `×`.

Not Dyalog’s nested-array result structure; the result is a tensor.

---

## Function combination

Decisions 13, 23, and 58. These are Haply macros with the Dyalog glyphs.
They are specified now; shipping order is item 48.

J-style hook is not a separate form. It is already beside, behind, atop,
or over.

### Beside — `∘`

Jot. This is Haply’s function-combination operator (decision 23).

Dyalog `f∘g` is composition (beside); `A∘g` / `g∘B` bind an argument.

```hy
(∘ f g Y)      ; (f (g Y))
(∘ f g X Y)    ; (X f (g Y))
(∘ A g Y)      ; (g A Y)
(∘ g B Y)      ; (g Y B)     ; bind form must be unambiguous in the macro
```

Bind vs two function operands must be obvious from whether an operand is
an array. Write examples before coding the bind cases.

### Atop — `⍤`

Dyalog overloads `⍤`: rank when the right operand is an array, atop when
it is a function.

```hy
(⍤ f g Y)      ; (f (g Y))
(⍤ f g X Y)    ; (f (X g Y))
(⍤ f r Y)      ; apply f to rank-r cells (array operand)
```

Atop is the glyph, not `(atop …)`.

### Behind — `⍛`

```hy
(⍛ f g Y)      ; (f (g Y))          ; confirm against Dyalog monadic behind
(⍛ f g X Y)    ; ((g X) f Y)
```

### Over — `⍥`

```hy
(⍥ f g Y)      ; (f (g Y))
(⍥ f g X Y)    ; (f (g X) (g Y))
```

## Later operators

Specified so names stay stable. Not Phase 1–3 work unless item 48 pulls
one forward.

### Power — `⍣`

```hy
(⍣ f n Y)      ; apply f, n times
(⍣ f pred Y)   ; apply until pred (later, if ever)
```

Working default: integer `n` only, including `0` (identity) and negative
`n` only when `f` has a defined inverse (otherwise error).

### At

Dyalog `@` is selective replacement. Haply does not take `@` (decision
23: `@` stays Hy/Python matmul). If At ships, it needs another spelling.
Later (item 48).

### Key — `⌸`

Group major cells by key and apply `f` to each group. Tensor analogue of
`groupby`. Later.

### Stencil — `⌺`

Sliding windows plus `f`. Natural fit for conv-like work. Later. Prefer
expanding known stencils to `unfold` / convolution rather than Python
loops.

### Dropped operators

`⍠` variant, `⌶` I-beam, `&` spawn, assignment operators, axis brackets.
See decision 44 and [glyphs.md](glyphs.md) section 11.

---

## Trains

Dyalog trains are juxtaposition. Haply does not parse juxtaposition as a
train (decision 2). The only named train is fork (decisions 13 and 58).
Atop is `⍤`, not a 2-train parser.

### Fork — `(fork …)`

Dyalog `(f g h)` on `Y` is `(f Y) g (h Y)`; on `X Y` is `(X f Y) g (X h Y)`.

```hy
(fork f g h Y)
(fork f g h X Y)
```

**Intent.** Same as Dyalog 3-trains on functions. Longer forks
`(fork a b c d e)` follow Dyalog’s even/odd train rules if we implement
them; v1 working default is the 3-train only.

`f` and `h` may be arrays (Dyalog constant forks). Working default: allow
array wings.

---

## Macro expansion rules

1. If the operand is a known Haply primitive, emit the backend kernel.
2. If the operand is a user `fn`, emit a correct but possibly slow loop
   over the chosen cells.
3. Do not silently change the user’s function into a vectorised form
   unless you can prove it is elementwise (Haply scalar primitives).
4. Preserve backend and device (decisions 9, 49).
5. The derived call remains a Hy expression. Users should be able to nest:

```hy
(⌿ + (· + × A B))
```

## What Haply is not doing

- No tacit parser.
- No Dyalog operator binding strengths.
- No derived-function arrays (operators return expansion, not APL
  function values, except as ordinary Hy callables when that falls out).
- No `⎕IO` in axis numbers.
