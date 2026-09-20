# Glyphs and primitive functions

This is the official list of Dyalog primitives and the Haply forms we will
implement. Operators are summarised here and specified in
[operators.md](operators.md).

Haply intent always wins over Dyalog when the two differ. Deviations are
called out in the **Haply intent** column, not hidden.

Status values:

| Status | Meaning |
| --- | --- |
| **implement** | Specified; build it |
| **adapt** | Build it with the stated Dyalog deviation |
| **later** | Specified; not yet shipped |
| **drop** | Will not be implemented |
| **undecided** | See the linked decision number |

Phase numbers follow [architecture.md](architecture.md).

Closed names: inner product `·` (23), magnitude/residue `||` (24),
AND/OR `∧` `∨` (25), trains and combinations (13, 58). Glyph ids do not
change when a decision lands.

## How a Haply call looks

```hy
(g Y)       ; monadic
(g X Y)     ; dyadic
(op f Y)    ; monadic operator producing a monadic derived call
(op f X Y)  ; monadic operator producing a dyadic derived call
(op f g Y)  ; dyadic operator (two function operands)
```

There is no APL infix and no axis bracket in v1.

## Hy conflicts (resolved names)

These Dyalog glyphs are **not** Haply exports. The Haply name is the one
to implement.

| Dyalog | Why not | Haply | Decision |
| --- | --- | --- | --- |
| `*` | Hy multiply | `**` | 16 |
| `/` `\` | Hy division; last-axis slash | `⌿_` `⍀_` | 15, 20 |
| `~` | Hy/Python bitwise not confusion | `≁` | 19 |
| `,` | awkward as a Hy symbol | `++` | 18 |
| `=` | Hy equality | `==` | 17 |
| `|` | Hy name reserved for the host | `||` | 24 |
| `.` | Hy attribute access | `·` | 23 |
| `¯` | Hy already has `-` | `-` | 14 |

Hy names that Haply *may* export, with opt-in shadowing (decision 53):
`+`, `-`, `<`, `>`. `<=` is **not** used; Haply uses `≤` and `≥`.
English aliases, if any, follow decision 39 (hard-to-type glyphs only)
and are listed in this catalog when named.

---

## 1. Scalar arithmetic

Elementwise. Broadcasting is the backend’s. No APL conformability.

| Id | Dyalog | Haply | Monadic Dyalog | Dyadic Dyalog | Haply intent | Phase | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G001 | `+` | `+` | Conjugate | Plus | Monadic: conjugate on complex values (decision 31), identity on reals. Dyadic: addition. Variadic fold (decision 37). Hy's core `+` macro still owns call position `(+ Y)` (unary plus / identity); the Haply function object conjugates. | 1 | implement |
| G002 | `-` | `-` | Negate | Minus | Negation; subtraction. Negative literals use Hy `-`. No `¯`. | 1 | implement |
| G003 | `×` | `×` | Direction (signum) | Times | Signum (`sign` / `np.sign`); multiplication. Variadic fold (decision 37). | 1 | implement |
| G004 | `÷` | `÷` | Reciprocal | Divide | `1/x`; true division. Do not export Hy `/` as divide. | 1 | implement |
| G005 | `*` | `**` | Exponential | Power | Monadic: `exp`. Dyadic: `x ** y`. Hy's core `**` macro requires two or more arguments, so `(** Y)` is a syntax error; call the function object for monadic exp. | 1 | implement |
| G006 | `⍟` | `⍟` | Natural log | Logarithm | Monadic: `ln`. Dyadic: log of `Y` in base `X` (`log(Y)/log(X)`). | 1 | implement |
| G007 | `|` | `||` | Magnitude | Residue | Haply `||` is both valences (decision 24). Residue follows Dyalog sign convention *if* easy on the backend, otherwise document the backend `remainder`/`fmod` choice. | 1 | adapt |
| G008 | `⌊` | `⌊` | Floor | Minimum | Floor; elementwise min. Variadic fold (decision 37). | 1 | implement |
| G009 | `⌈` | `⌈` | Ceiling | Maximum | Ceiling; elementwise max. Variadic fold (decision 37). | 1 | implement |
| G010 | `○` | `○` | π times | Circular | Monadic: `π * Y`. Dyadic: circular table (extent still decision 47). Complex-valued inputs are in scope (decision 31). | 2 | adapt |
| G011 | `!` | `!` | Factorial | Binomial | `gamma(Y+1)`; binomial via gammaln or an integer path. | 2 | implement |
| G012 | `?` | `?` | Roll | Deal | Monadic: random integers in `[0, n)` per element (decision 28). Bound ≤ 0 is an error — no Dyalog `?0` float-in-(0,1). Dyadic: `k` distinct draws from `[0, n)`. Host RNG is `torch`. | 4 | adapt |

### Circular left arguments (G010), working subset

| `X` | Intent |
| --- | --- |
| `1` | `sin Y` |
| `2` | `cos Y` |
| `3` | `tan Y` |
| `-1` | `asin Y` |
| `-2` | `acos Y` |
| `-3` | `atan Y` |
| `5` | `sinh Y` |
| `6` | `cosh Y` |
| `7` | `tanh Y` |

Other Dyalog `X` values (complex parts, `sqrt(1-Y²)`, …) wait on decision 47.

---

## 2. Comparisons

Results follow decision 35 (working default: backend boolean dtype).
Comparisons are exact until decision 29 says otherwise.

| Id | Dyalog | Haply | Monadic Dyalog | Dyadic Dyalog | Haply intent | Phase | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G013 | `<` | `<` | — | Less than | Elementwise `<` | 1 | implement |
| G014 | `≤` | `≤` | — | Less or equal | Elementwise `<=` | 1 | implement |
| G015 | `=` | `==` | — | Equal to | Elementwise equality. Not Hy `=`. | 1 | implement |
| G016 | `≥` | `≥` | — | Greater or equal | Elementwise `>=` | 1 | implement |
| G017 | `>` | `>` | — | Greater than | Elementwise `>` | 1 | implement |
| G018 | `≠` | `≠` | Unique mask | Not equal | Dyadic: elementwise `!=`. Monadic: first-occurrence mask of raveled values in C-order, **same shape as `Y`** (decision 45 default). Not Dyalog’s major-cell unique mask. | 2 | implement |
| G019 | `≡` | `≡` | Depth | Match | Dyadic: whole-array equality (`torch.equal` / `np.array_equal`). Monadic depth is dropped by default (decision 56). | 2 | adapt |
| G020 | `≢` | `≢` | Tally | Not match | Monadic: major-cell count (`shape[0]`, or `1` if scalar). Dyadic: `not match`. | 2 | implement |

---

## 3. Logic and set-like scalar ops

| Id | Dyalog | Haply | Monadic Dyalog | Dyadic Dyalog | Haply intent | Phase | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G021 | `~` | `≁` | Not | Without | Monadic: logical not. Dyadic: keep values of `X` that are not in `Y` (ravel membership), tensor analogue of without. | 1 / 3 | adapt |
| G022 | `∧` | `∧` | — | And / LCM | Boolean and; integer LCM (decision 46). Variadic fold (decision 37). | 1 | adapt |
| G023 | `∨` | `∨` | — | Or / GCD | Boolean or; integer GCD (decision 46). Variadic fold (decision 37). | 1 | adapt |
| G024 | `⍲` | `⍲` | — | Nand | `not (X and Y)`. Integer/float domain is item 66. | 1 | implement |
| G025 | `⍱` | `⍱` | — | Nor | `not (X or Y)`. Integer/float domain is item 66. | 1 | implement |

---

## 4. Structural

These are the core tensor vocabulary.

| Id | Dyalog | Haply | Monadic Dyalog | Dyadic Dyalog | Haply intent | Phase | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G026 | `⍴` | `⍴` | Shape | Reshape | Monadic: shape vector (1-d tensor or tuple — Phase 0 should pick one and stay with it; working default: 1-d tensor of the same backend). Dyadic: reshape `Y` to shape `X`, row-major, cycling `Y` if needed *only if* we choose Dyalog reshape; working default: **no cycle**, size must match, like `view`/`reshape`. Document this as a deviation. | 1 | adapt |
| G027 | `,` | `++` | Ravel | Catenate / laminate | Monadic: flatten to 1-d (C order). Dyadic: `torch.cat` on the last dim. Rank difference 0 or 1 (unsqueeze the shorter on the join axis). 0-d ++ 0-d stacks to a 1-d pair. No implicit laminate: same-shape arrays still cat, they do not stack on a new last axis. | 2 | adapt |
| G028 | `⍪` | `⍪` | Table | Catenate first | Monadic: reshape to a matrix, preserving axis 0. Dyadic: concatenate on axis 0. | 2 | implement |
| G029 | `⌽` | `⌽` | Reverse last | Rotate last | Reverse or roll the last dimension. | 2 | implement |
| G030 | `⊖` | `⊖` | Reverse first | Rotate first | Reverse or roll axis 0. | 2 | implement |
| G031 | `⍉` | `⍉` | Transpose | Dyadic transpose | Monadic: reverse axes (`T` / `permute(range(ndim-1,-1,-1))`). Dyadic: permute axes by `X` (0-based under decision 28 default). | 2 | implement |
| G032 | `↑` | `↑` | Mix | Take | **Monadic mix is dropped** (needs nested arrays). Dyadic: take first/last items along leading axes given by `X` (negative = from the end). Pad policy: decision 33. | 2 | adapt |
| G033 | `↓` | `↓` | Split | Drop | **Monadic split is dropped**. Dyadic: drop along leading axes given by `X`. | 2 | adapt |
| G034 | `⊢` | `⊢` | Same | Right | Monadic: identity. Dyadic: return `Y`. Useful in trains. | 2 | implement |
| G035 | `⊣` | `⊣` | Same | Left | Monadic: identity. Dyadic: return `X`. | 2 | implement |

### Reshape deviation (G026)

Dyalog `(⍴ shape A)` recycles elements of `A` to fill `shape`. Haply’s
working intent is **NumPy/PyTorch reshape**: the number of elements must
match. Recycling, if wanted later, needs an explicit decision and a
different name or flag. This is a deliberate, documented break.

---

## 5. Selection, search, grade

Index origin is 0 (decision 28). Translate Dyalog examples accordingly.

| Id | Dyalog | Haply | Monadic Dyalog | Dyadic Dyalog | Haply intent | Phase | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G036 | `⍳` | `⍳` | Index generator | Index of | Monadic: `arange` / `meshgrid` for a shape vector. Dyadic: first index of each `Y` in `X` along the search axis; not-found sentinel is `-1` or `n` (working default: `n`, length of the search vector — closer to Dyalog’s `⎕IO+≢X` after origin 0). | 3 | adapt |
| G037 | `⍸` | `⍸` | Where | Interval index | Monadic: indices of truthy values (`nonzero`). Dyadic: `torch.bucketize` of `Y` into sorted `X` (left / insertion index, 0-based). | 3 | implement |
| G038 | `∊` | `∊` | Enlist | Membership | **Monadic enlist** becomes flatten (same as `++` on a simple tensor) or is dropped as redundant. Dyadic: `isin`. | 3 | adapt |
| G039 | `⍷` | `⍷` | — | Find | Boolean mask of occurrences of array `X` as a sub-array of `Y`. Same shape as `Y`. Rank(`X`) < rank(`Y`) left-pads `X` with 1s; rank(`X`) > rank(`Y`) finds nothing. | 4 | implement |
| G040 | `∪` | `∪` | Unique | Union | Monadic: unique values in ravel order of first occurrence. Dyadic union waits on how “set” we want tensors to be (working: unique of catenated ravels). | 3 | adapt |
| G041 | `∩` | `∩` | — | Intersection | Values of `X` that appear in `Y`, stable order from `X`, duplicates kept (item 65). | 3 | adapt |
| G042 | `⍋` | `⍋` | Grade up | Dyadic grade up | Indices that sort `Y` ascending. Numeric only (decision 32); no collation alphabet. | 3 | adapt |
| G043 | `⍒` | `⍒` | Grade down | Dyadic grade down | Same, descending. Numeric only (decision 32). | 3 | adapt |
| G044 | `⌷` | `⌷` | Materialise | Index | Monadic materialise is identity on a tensor. Dyadic: index by `X` (list of index arrays / integers), 0-based. Prefer mapping to `tensor[…]` / `np.ix_` rather than APL squad idiosyncrasies. | 3 | adapt |

---

## 6. Linear algebra and encode

| Id | Dyalog | Haply | Monadic Dyalog | Dyadic Dyalog | Haply intent | Phase | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G045 | `⌹` | `⌹` | Matrix inverse | Matrix divide | Monad: `inv` if square 2-d, else `pinv` (0-d is reciprocal). Dyad: Haply `(⌹ X Y)` is Dyalog `X⌹Y` — solve `Y B = X` (`torch.linalg.solve` / `lstsq`), i.e. `Y⁻¹X` when `Y` is square. Rank ≤ 2. | 4 | adapt |
| G046 | `⊤` | `⊤` | — | Encode | Represent `Y` in mixed radix `X`. `X` rank 0 or 1; result shape `shape(X)+shape(Y)`. A leading 0 radix keeps the remaining value. Dyadic only. | 4 | implement |
| G047 | `⊥` | `⊥` | — | Decode | Evaluate `Y` in mixed radix `X` by Horner along axis 0 of `Y`. Scalar `X` is a repeated radix. `X` rank 0 or 1. Dyadic only. | 4 | implement |

---

## 7. Nested-adjacent functions

Decision 43 default: keep only a clean tensor analogue.

| Id | Dyalog | Haply | Dyalog meaning | Haply intent | Phase | Status |
| --- | --- | --- | --- | --- | --- | --- |
| G048 | `⊂` | — | Enclose / partitioned enclose | No boxes. Drop, or later `unsqueeze` analogue if 43 chooses B. | — | drop |
| G049 | `⊆` | — | Nest / partition | Drop. | — | drop |
| G050 | `⊃` | `⊃` | First / pick | Working: monadic first major cell (or first element of a 1-d tensor). Empty first axis is `ValueError`. Dyadic pick is dropped (needs paths into boxes). | 3 | adapt |
| G051 | `↑` monadic | — | Mix | Drop. Dyadic take is G032. | — | drop |
| G052 | `↓` monadic | — | Split | Drop. Dyadic drop is G033. | — | drop |
| G053 | `≡` monadic | — | Depth | Drop by default (G019, decision 56). | — | drop |

---

## 8. Replicate and expand (functions)

In Dyalog these share glyphs with reduce/scan. Haply uses the first-axis
glyphs plus `_` (decisions 15, 20, 60).

| Id | Dyalog | Haply | Role | Haply intent | Phase | Status |
| --- | --- | --- | --- | --- | --- | --- |
| G054 | `/` | `⌿_` | Replicate last | Repeat/select along the last axis by integer mask `X`. | 3 | implement |
| G055 | `⌿` | `⌿` | Replicate first | Same on axis 0. | 3 | implement |
| G056 | `\` | `⍀_` | Expand last | Insert zeros/fill along the last axis by boolean/int mask. | 6 | implement |
| G057 | `⍀` | `⍀` | Expand first | Same on axis 0. | 6 | implement |

When the left operand is a *function* rather than an array, the same
glyphs are operators (reduce/scan). See decision 60 and
[operators.md](operators.md).

---

## 9. Operators (index)

Full specification: [operators.md](operators.md).

| Id | Dyalog | Haply | Role | Phase | Status |
| --- | --- | --- | --- | --- | --- |
| G058 | `/` | `⌿_` | Reduce last; n-wise reduce last | 3 | implement |
| G059 | `⌿` | `⌿` | Reduce first; n-wise reduce first | 3 | implement |
| G060 | `\` | `⍀_` | Scan last | 3 | implement |
| G061 | `⍀` | `⍀` | Scan first | 3 | implement |
| G062 | `¨` | `¨` | Each | 3 | implement (42 default) |
| G063 | `⍨` | `⍨` | Commute / selfie / constant | 3 | implement |
| G064 | `.` | `·` | Inner product | 3 | implement |
| G065 | `∘.` | `outer` | Outer product | 3 | implement |
| G066 | `∘` | `∘` | Beside (jot); function combination | 5 | implement |
| G067 | `⍤` | `⍤` | Atop (function operand); rank (array operand) | 5 / 8 | implement (atop); rank later (48) |
| G068 | `⍥` | `⍥` | Over | 5 | implement |
| G069 | `⍣` | `⍣` | Power operator | 8 | later (48) |
| G070 | `@` | — | At | 8 | later (48) |
| G071 | `⌸` | `⌸` | Key | 8 | later (48) |
| G072 | `⌺` | `⌺` | Stencil | 8 | later (48) |
| G073 | `⍛` | `⍛` | Behind | 5 | implement |
| G074 | `⍠` | — | Variant | — | drop (44) |
| G075 | `⌶` | — | I-beam | — | drop (44) |
| G076 | `&` | — | Spawn | — | drop (44) |

---

## 10. Named combinators (not glyphs)

Decision 13 and 58. The only named train is fork. Atop, beside, behind,
and over are the glyphs in section 9. There is no `(hook …)`.

| Id | Concept | Haply | Phase | Status |
| --- | --- | --- | --- | --- |
| G077 | Fork | `(fork f g h …)` | 5 | implement (3-train); longer later (62) |
| G078 | Atop | `⍤` (see G067) | 5 | implement |
| G079 | Hook | — (covered by `⍤` `∘` `⍛` `⍥`) | — | drop |

---

## 11. Dropped syntax and runtime glyphs

Hy or Python already provide these, or they belong to an APL session.
They are **not** Haply exports.

| Dyalog | Dyalog use | Why dropped |
| --- | --- | --- |
| `←` `[]←` `()←` | Assignment | Hy `setv` / `setx` |
| `→` | Branch / abort | Hy control flow |
| `⍝` | Comment | Hy `;` |
| `⋄` | Statement separator | Separate Hy forms |
| `¯` | High minus | Decision 14 |
| `'` `''` | Character arrays | Decisions 26 and 32: no Haply binding. Strings stay Hy. |
| `` ` `` | Hy/Python syntax | Decision 26: no Haply binding |
| `⍺` `⍵` `⍺⍺` `⍵⍵` | Dfn arguments | Hy function parameters |
| `∇` `∇∇` | Self-reference | Hy recursion |
| `∆` `⍙` `_` | Name characters | `_` is reserved for last-axis suffix |
| `{}` | Dfns | `defn` / `fn` |
| `[]` | Index / axis / rank-2 literals | Hy `get` / Python indexing; axis open (34) |
| `:` `::` | Labels, guards | Hy |
| `;` | Index separator / locals | Hy |
| `⎕` `⍞` | Session I/O, system names | Decision 44 |
| `#` `##` | Namespaces | Python modules |
| `⍬` | Empty numeric vector | Decision 33 (constructor optional, not a syntax glyph) |
| `⍎` | Execute | Hy/Python eval — not provided |
| `⍕` | Format | `str` / format — not provided in v1 |

---

## 12. Implementation checklist (what we will build)

Build these names. This is the concrete v1-or-specified set.

### Phase 1 — scalars

`+` `-` `×` `÷` `**` `⍟` `||` `⌊` `⌈` `<` `≤` `==` `≥` `>` `≁` `∧` `∨` `⍲` `⍱` `⍴`

### Phase 2 — structure and compare

`++` `⍪` `⌽` `⊖` `⍉` `↑` `↓` `⊢` `⊣` `≠` `≡` `≢` `○` `!`

### Phase 3 — search and operators

`⍳` `⍸` `∊` `∪` `∩` `⍋` `⍒` `⌷` `⊃` `⌿` `⌿_` `⍀` `⍀_` `⍨` `·` `outer` `¨`

### Phase 4 — remaining functions

`?` `⍷` `⌹` `⊤` `⊥`

### Phase 5 — fork and jot family

`fork` `∘` `⍤` (atop only) `⍥` `⍛`

### Phase 6 — specified leftovers

`⍀` `⍀_` expand (array operand)

### Phase 7 — NumPy backend

No new glyphs. Repeat shipped names on `numpy.ndarray` (item 21).

### Phase 8 — item 48 operators

`⍣` `⌸` `⌺` ; `⍤` rank (array operand); At (not `@`)

### Never (unless a new decision says so)

`/` `\` `*` `=` `,` `~` `|` `.` `¯` `←` `→` `⍝` `⍎` `⍕` `⌶` `&` `⍠` `⎕` `⍞` `⊂` `⊆` mix, split, depth, I/O, system space

---

## 13. Full Dyalog primitive-function inventory

Every Dyalog primitive function, with the Haply disposition in one line.

| Dyalog | Monadic | Dyadic | Haply |
| --- | --- | --- | --- |
| `+` | Conjugate | Plus | `+` G001 |
| `-` | Negate | Minus | `-` G002 |
| `×` | Direction | Times | `×` G003 |
| `÷` | Reciprocal | Divide | `÷` G004 |
| `\|` | Magnitude | Residue | `\|\|` G007 |
| `*` | Exponential | Power | `**` G005 |
| `⍟` | Natural log | Log | `⍟` G006 |
| `○` | Pi times | Circular | `○` G010 |
| `⌈` | Ceiling | Maximum | `⌈` G009 |
| `⌊` | Floor | Minimum | `⌊` G008 |
| `!` | Factorial | Binomial | `!` G011 |
| `?` | Roll | Deal | `?` G012 |
| `∧` | — | And / LCM | `∧` G022 |
| `∨` | — | Or / GCD | `∨` G023 |
| `⍲` | — | Nand | `⍲` G024 |
| `⍱` | — | Nor | `⍱` G025 |
| `<` | — | Less | `<` G013 |
| `≤` | — | Less or equal | `≤` G014 |
| `=` | — | Equal | `==` G015 |
| `≥` | — | Greater or equal | `≥` G016 |
| `>` | — | Greater | `>` G017 |
| `≠` | Unique mask | Not equal | `≠` G018 |
| `~` | Not | Without | `≁` G021 |
| `∩` | — | Intersection | `∩` G041 |
| `∪` | Unique | Union | `∪` G040 |
| `≡` | Depth | Match | `≡` G019 |
| `≢` | Tally | Not match | `≢` G020 |
| `∊` | Enlist | Membership | `∊` G038 |
| `⍷` | — | Find | `⍷` G039 |
| `⍳` | Index generator | Index of | `⍳` G036 |
| `⍸` | Where | Interval index | `⍸` G037 |
| `⍋` | Grade up | Dyadic grade up | `⍋` G042 |
| `⍒` | Grade down | Dyadic grade down | `⍒` G043 |
| `⌷` | Materialise | Index | `⌷` G044 |
| `⌽` | Reverse | Rotate | `⌽` G029 |
| `⍉` | Transpose | Dyadic transpose | `⍉` G031 |
| `⊖` | Reverse first | Rotate first | `⊖` G030 |
| `⍴` | Shape | Reshape | `⍴` G026 |
| `,` | Ravel | Catenate | `++` G027 |
| `⍪` | Table | Catenate first | `⍪` G028 |
| `↑` | Mix | Take | take only G032 |
| `↓` | Split | Drop | drop only G033 |
| `⊂` | Enclose | Partitioned enclose | drop G048 |
| `⊆` | Nest | Partition | drop G049 |
| `⊃` | First | Pick | first only G050 |
| `⊤` | — | Encode | `⊤` G046 |
| `⊥` | — | Decode | `⊥` G047 |
| `⌹` | Matrix inverse | Matrix divide | `⌹` G045 |
| `⊢` | Same | Right | `⊢` G034 |
| `⊣` | Same | Left | `⊣` G035 |
| `⍎` | Execute | Dyadic execute | drop |
| `⍕` | Format | Format by spec | drop |
| `/` | — | Replicate | `⌿_` G054 |
| `⌿` | — | Replicate first | `⌿` G055 |
| `\` | — | Expand | `⍀_` G056 |
| `⍀` | — | Expand first | `⍀` G057 |

Dyalog operators are the G058–G076 rows in section 9.
