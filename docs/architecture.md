# Architecture

This is the implementation map. It is a working default for
[undecided 55](undecided.md#55-package-layout). Change the tree only by
closing that item.

## Goals

- Hy-first public API of glyphs and operator macros.
- Thin dispatch over `torch` and `numpy`.
- No Haply array type.
- Macros for operators and trains; functions for primitives (decision 38
  default).
- Tests that pin **Haply intent**, not Dyalog identity.

## Proposed tree

```
haply/                      repository root
  README.md
  flake.nix                 required (decision 11)
  flake.lock                required (decision 11)
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
(require haply.macros [⌿ ⌿_ ⍀ ⍀_ ⍨ · ∘ ⍤ ⍥ ⍛])
(require haply.trains [fork])
```

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

1. Count arguments → monadic or dyadic (or variadic, decision 37).
2. Read backends of the arguments.
3. Same backend → call that kernel.
4. Mixed backends → error in Phase 1 (do not silently promote).
5. Python + Python → host operator (`+`, `abs`, …).
6. Python scalar + tensor → let the backend broadcast the scalar if it
   already does; do not first convert the tensor to a list.

### Constructors

`(⍳ n)` and friends have no tensor argument. Decision 30 default: produce
a CPU `torch.Tensor`. Provide `(⍳ n :backend 'numpy)` only if Phase 3
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
exception. No `DOMAIN ERROR` type (decision 40 default).

## Implementation phases

### Phase 0 — scaffolding

- `flake.nix` with Hy, Python, PyTorch, NumPy, pytest (or Hy’s test
  runner), and `uv`.
- Package imports.
- One dispatch test: Python `(+ 1 2)` and tensor `(+ t t)` both work.
- No glyph soup yet.

### Phase 1 — scalar core

Implement the Phase 1 checklist in [glyphs.md](glyphs.md) section 12.

Done when:

- arity dispatch works;
- Python / NumPy / PyTorch paths exist for `+` and `⍴`;
- broadcasting follows the backend;
- Hy programs that do not import Haply are unaffected.

### Phase 2 — structure

`++` `⍪` `⌽` `⊖` `⍉` take/drop (dyadic only) `⊢` `⊣` `≢` `≡` (dyadic)
`≠` `○` `!`.

Done when reshape, reverse, and transpose are tested on 1-d, 2-d, and
3-d tensors for both backends.

### Phase 3 — operators and search

Reduce, scan, replicate, commute, inner/outer product, iota, where,
membership, unique, grade, index.

Done when `(⌿ + (× A B))` matches `sum(A * B, dim=0)` and
`(· + × A B)` matches `matmul`.

### Phase 4 — remaining functions

Random, find, matrix divide, encode/decode.

### Phase 5 — trains and later operators

`fork` plus `∘` `⍤` `⍥` `⍛`. Then the later glyphs in item 48.

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
- In-place glyphs (decision 50 default).
- A tacit parser for juxtaposition.

## Performance budget

A Haply scalar call on two large tensors should be a thin Python/Hy
function around one backend kernel. If a glyph needs a Python loop over
elements, it is either a later operator with a user operand, or the
implementation is not done.

`(· + × X Y)` looping in Python is a bug.
