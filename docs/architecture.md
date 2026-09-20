# Architecture

This is the implementation map. It is a working default for
[undecided 55](undecided.md#55-package-layout). Change the tree only by
closing that item.

## Goals

- Hy-first public API of glyphs and operator macros.
- Thin dispatch over `torch` and `numpy`.
- No Haply array type.
- Macros for operators and trains; functions for primitives (decision 38).
- Tests that pin **Haply intent**, not Dyalog identity.

## Proposed tree

```
haply/                      repository root
  README.md
  flake.nix                 required (decision 11)
  flake.lock                required (decision 11)
  pyproject.toml            install metadata (decision 52)
  docs/                     this suite
  haply/                    the package
    __init__.py             package marker; optional Python re-export
    __init__.hy             public glyphs and helpers
    _backend.hy             type detection, backend routing
    _dispatch.hy            arity dispatch, shared wrappers
    scalar.hy               G001–G011, G013–G017, G021–G025
    structural.hy           G026–G035
    compare.hy              G018–G020
    select.hy               G032–G033, G036–G038, G044, G050
    search.hy               G039–G043
    numeric.hy              G010–G012, G045–G047
    macros.hy               operators (require this)
    trains.hy               fork only
  tests/
    test_scalar.hy
    test_structural.hy
    test_operators.hy
    test_trains.hy
    test_dispatch.hy
```

Phase 0 creates the package, `flake.nix`, and one passing test. It does
not invent extra folders.

## Public import

Functions (selective, opt-in shadowing — decision 53):

```hy
(import haply [+ - × ÷ ⍴])
```

Macros:

```hy
(require haply.macros [⌿ ⌿_ ⍀ ⍀_ ⍨ · outer ¨ ∘ ⍤ ⍥ ⍛])
(require haply.trains [fork])
```

Phase 5 adds `∘` `⍤` `⍥` `⍛` here and `(require haply.trains [fork])`.
`outer` is the Hy-legal head for catalog `∘.` (item 61). `⍤` in Phase 5
is atop (function operand) only; rank (array operand) waits on item 48.

A convenience star-import or star-require should not be the documented
path. Document the two-step import/require.

Never export: `*` `/` `=` `|` `.` `,` `~` `^` `¯`.

## Backend detection

```hy
(defn backend [x]
  (cond
    (is-torch x) 'torch
    (is-numpy x) 'numpy
    True 'python))
```

`is-torch` / `is-numpy` must not import a missing optional dependency at
runtime if we later allow NumPy-only or PyTorch-only installs. Phase 0
working default: both are required in the Nix shell.

### Routing a dyadic function

Phase 1 implements the PyTorch branch only (item 21). The later steps are
the intended shape of dispatch, not Phase 1 work.

1. Count arguments → monadic or dyadic (or variadic, decision 37).
2. Read backends of the arguments.
3. Same backend → call that kernel.
4. Mixed backends → error (do not silently promote).
5. Python + Python → host operator (`+`, `abs`, …), when that path exists.
6. Python scalar + tensor → let the backend broadcast the scalar if it
   already does; do not first convert the tensor to a list.

### Constructors

`(⍳ n)` and friends have no tensor argument. Decision 30 default: produce
a CPU `torch.Tensor`. Provide `(⍳ n :backend 'numpy)` only when Phase 7
needs it; do not add kwargs casually.

## Dispatch helper

A single helper should wrap most primitives:

```hy
(defn defprim [name monad dyad]
  (fn [#* args]
    (match (len args)
      1 (monad (get args 0))
      2 (dyad (get args 0) (get args 1))
      _ (raise (TypeError "…")))))
```

Variadic associative ops (`+`, `×`, …) get a different helper that folds
with the dyadic kernel.

Keep the helper boring. The interesting work is the per-glyph intent in
[glyphs.md](glyphs.md).

## Macros

`macros.hy` is the only place that should pattern-match operand names.

Example target expansion:

```hy
(⌿ + Y)      ⇒  (torch.sum Y :dim 0)     ; if Y is a tensor path
(⌿_ + Y)     ⇒  (torch.sum Y :dim -1)
(· + × X Y)  ⇒  (torch.matmul X Y)
```

If the operand is not a recognised symbol, expand to a cell loop helper
in `_dispatch.hy`, not to a nested tree of closures.

Trains expand to nested calls of existing primitives:

```hy
(fork ⊣ + ⊢ X Y)  ⇒  (+ (⊣ X Y) (⊢ X Y))
```

## Errors

Phase 1: raise `TypeError`, `ValueError`, `IndexError`, or the backend
exception. No `DOMAIN ERROR` type (decision 40).

## Implementation phases

### Phase 0 — scaffolding

- `flake.nix` with Hy, Python, PyTorch, NumPy, pytest (or Hy’s test
  runner), and `uv`.
- Package imports.
- One dispatch test: tensor `(+ t t)` works. Python `(+ 1 2)` waits on
  item 21.
- No glyph soup yet.

### Phase 1 — scalar core

Implement the Phase 1 checklist in [glyphs.md](glyphs.md) section 12.

Done when:

- arity dispatch works;
- a PyTorch path exists for `+` and `⍴` (item 21: Phase 1 is PyTorch
  only; NumPy and Python natives wait);
- broadcasting follows the backend;
- Hy programs that do not import Haply are unaffected.

### Phase 2 — structure

`++` `⍪` `⌽` `⊖` `⍉` take/drop (dyadic only) `⊢` `⊣` `≢` `≡` (dyadic)
`≠` `○` `!`.

Done when reshape, reverse, and transpose are tested on 1-d, 2-d, and
3-d PyTorch tensors. Repeat on NumPy when item 21 adds that backend.

### Phase 3 — operators and search

Reduce, scan, replicate, commute, inner/outer product, iota, where,
membership, unique, grade, index.

Done when `(⌿ + (× A B))` matches `sum(A * B, dim=0)` and
`(· + × A B)` matches `matmul`. **Met** for PyTorch.

### Phase 4 — remaining functions

Random, find, matrix divide, encode/decode.

Done when `(? k n)` is a deal of `k` distinct integers in `[0, n)`,
`(⌹ X Y)` solves `Y B = X`, and `(⊥ X (⊤ X Y))` recovers a small
integer `Y`. **Met** for PyTorch.

### Phase 5 — fork and jot family

Composition macros and the named 3-train. Item 48 is **not** this phase.

| Ship | Leave |
| --- | --- |
| `(fork f g h …)` — 3-train only | longer forks |
| `∘` beside (two functions, then bind) | |
| `⍤` atop (function operand) | `⍤` rank (array operand) — item 48 |
| `⍥` over | `⍣` `⌸` `⌺` At — item 48 |
| `⍛` behind | |

Done when `(fork ⊣ + ⊢ X Y)` matches `(+ X Y)` and `(∘ × + Y)` matches
`(× (+ Y))` on PyTorch tensors. **Met** for PyTorch.

### Phase 6 — specified leftovers

Work already in the catalog that is not item 48 and not a new backend.
Finish the PyTorch surface before the NumPy sweep.

- Expand: array operand on `⍀` / `⍀_` (G056, G057). Fill is backend
  zero (decision 33).

Done when `(⍀ mask Y)` inserts fill along axis 0 and `(⍀_ mask Y)`
does the same on the last axis.

### Phase 7 — NumPy backend

Item 21 working default C: repeat the **already shipped** glyphs on
`numpy.ndarray`. Mixed torch/NumPy still errors. Python natives stay
out. This phase does not close item 21.

Done when `(+ a a)` and `(⍴ a)` on a NumPy array return a NumPy array,
and a torch/NumPy mix raises `TypeError`.

### Phase 8 — item 48 operators

Do not start until item 48 closes, or an explicit pull-forward names
which of these ship:

- Power `⍣`
- Key `⌸`
- Stencil `⌺`
- Rank (array operand of `⍤`)
- At (not `@`)

No done-when until that item moves.

### After Phase 8

New phase numbers only when a decision pulls work forward. Still
decision-gated, not a phase:

- Python natives and whether every glyph must take all backends (21)
- Python-first API (22)
- Comparison tolerance (29), constructor backend (30), fill/zilde (33)
- General axis (34), circular-table extent (47)
- Aliases, oracles, package-layout close (39, 54, 55)

## Per-glyph implementation recipe

1. Read the row in [glyphs.md](glyphs.md).
2. Write tests for Haply intent (monadic, dyadic, one broadcast case,
   one error case).
3. Implement the function or macro.
4. Map known operator operands to kernels.
5. Note any new deviation in the glyph row — do not “just match Dyalog”
   if the catalog already chose a tensor meaning.

## What not to build

- A nested-array runtime.
- `⎕` system space.
- An APL parser.
- A custom tensor class.
- In-place glyphs (decision 50).
- A tacit parser for juxtaposition.

## Performance budget

A Haply scalar call on two large tensors should be a thin Python/Hy
function around one backend kernel. If a glyph needs a Python loop over
elements, it is either a later operator with a user operand, or the
implementation is not done.

`(· + × X Y)` looping in Python is a bug.
