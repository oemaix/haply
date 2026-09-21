# Undecided

Open entries from the decision register. Numbers are stable.

When an item is closed:

1. Do not change its number.
2. Move the entire heading and body to [decided.md](decided.md).
3. Leave a one-line stub here only if you need a pointer; prefer a clean
   move so this file contains only open items.
4. Record the chosen option and the date in the moved entry.

**Working defaults** below are scaffolding for implementation. They are not
decisions. Code that relies on a default must be easy to change when the
item moves.

---

## 21. Backend set

**Question.** Must every operation accept Python natives, NumPy arrays, and
PyTorch tensors?

**Context.** Decision 9 says “preserve the backend you are given”. The
compatibility note still asks whether all three are in scope.

Left open on 2026-09-20. Phases 1–6 implemented **PyTorch tensors only**.
Phase 7 shipped NumPy (working default C). Whether every glyph must later
accept all three remains the question. Python natives still wait.

**Options.**

- A. All three, for every implemented glyph.
- B. PyTorch and NumPy only; Python scalars only where the backend already
  accepts them.
- C. PyTorch first; NumPy later; Python natives only as a convenience.

**Working default.** C for shipping order. Priority if others land:
PyTorch, then NumPy, then Python natives (scalars before lists). Option A
is not a Phase 1 requirement.

**Notes.** Python scalars are not a third tensor backend. A `3` has no
shape, device, or kernel; a Python list is a sequence, not an ndarray.
Implementing a glyph on lists means reject, walk in Python (slow, and it
starts to look like nested APL), or lift to a tensor (against decision 9
unless the lift is documented). A Python number next to a tensor is
usually free: the backend already broadcasts it.

Phase 7 imports `numpy` at package load (`pyproject` requires it). A
torch-only install of `haply` therefore fails on import. Lazy detection
can return if this item later drops NumPy as a required backend.

**Impact.** Dispatch tables, tests, and mixed-type errors.

---

## 22. Python import surface

**Question.** Haply is a Hy library. Should plain Python `import haply` be
supported, documented, or explicitly rejected?

Hy compiles to Python, so `import haply` can exist as a side effect of
packaging. That is not the same as supporting a Python API.

**Options.**

- A. Hy is the only supported surface. Python import may exist as an
  implementation detail but is not a product.
- B. Official Python API with the same glyphs as Unicode identifiers.
- C. Official Python API with ASCII aliases only.

**A vs B.** Implementation in Hy does **not** automatically give Python
users the same product.

| | A (Hy only) | B (official Python, Unicode glyphs) |
| --- | --- | --- |
| Functions such as `⍴` `×` `∧` | Hy names | Also legal Python identifiers |
| `+` `-` `<` `>` | Hy names; opt-in shadow | **Not** legal Python identifiers; need `getattr` or a mapping |
| Operators / trains (`⌿`, `⋔`, `∘`) | Hy `require` macros | Macros do not run from Python; B needs function fallbacks or drops them |
| Docs and tests | Hy examples | A second surface to specify and test |

C is B with ASCII names instead of Unicode. Decision 39 already limits
English aliases; C would make those aliases the Python product.

**Working default.** A. Implement in Hy; do not advertise a Python API yet.

**Impact.** Packaging, `__init__.py`, docs, and name mangling.

---

## 29. Comparison tolerance

**Question.** Dyalog uses `⎕CT`. Haply has no system variables.

Left open on 2026-09-20. None of A–C is a Dyalog equivalent: Dyalog’s
tolerance is a session-global `⎕CT` that affects `=` `≠` `<` `≤` `≥` `>`
and, through those, match, membership, unique, and index-of.

**Options.**

- A. Exact comparison only.
- B. Explicit tolerant functions (`≈`, `match-at`, …).
- C. Optional keyword on `==` / `≡`.

**Notes.** B and C as written cover equal and match (and the not-forms).
They do not reconstruct tolerant `<` `≤` `≥` `>`. Full `⎕CT` on order
comparisons is not a PyTorch primitive; it is a scalar formula on top of
`abs` / `max` and is a real cost if applied to every compare.

The honest tensor analogue of *tolerant equality* is `torch.isclose` /
`torch.allclose` (and the NumPy pair). Exact `torch.eq` / `torch.equal`
is the analogue of A. That part is easy. A hidden global like `⎕CT` is
neither easy nor wanted (no system variables).

**Working default.** A.

**Impact.** `==`, `≠`, `≡`, `≢`, membership, unique.

---

## 30. Default backend for constructors

**Question.** What does `(⍳ 5)` return if no tensor argument is present?

**Options.**

- A. `torch.Tensor` (CPU, default dtype).
- B. `numpy.ndarray`.
- C. Python list / range.
- D. Require an explicit prototype or backend argument.

**Working default.** A.

**Impact.** All constructors: `⍳`, `?`, `⍴` when building from Python data.

---

## 33. Empty tensors, zilde, and fill

**Question.** Dyalog `⍬` is the empty numeric vector. Expand, take, and
reshape may insert fill/prototypes.

**Options.**

- A. No `⍬`. Empty is `torch.empty` / `np.empty` with an explicit shape.
- B. `⍬` as a convenience empty 1-d numeric tensor.
- C. Implement fill values as an extra argument, still no prototypes.

**Working default.** B as a constructor; no prototypes. Pad/take fill is
backend default (often zero) until specified per glyph.

**Impact.** `↑`, `⍴`, expand, replicate.

---

## 34. General axis selection

**Question.** Dyalog’s axis operator is `f[k]`. Haply has first-axis glyphs
and `_` for last axis. What about an arbitrary axis?

**Options.**

- A. v1: first and last only.
- B. Extra axis argument: `(⌿ + A :axis 2)` or `(⌿[2] + A)`.
- C. A general `on-axis` macro.

**Working default.** A. Design the macros so an axis argument can be added
without renaming glyphs.

**Impact.** Reduce, scan, reverse, rotate, catenate, replicate, expand.

---

## 35. Boolean representation

**Question.** Dyalog Booleans are 0/1 arrays. PyTorch comparisons often
return `dtype=torch.bool`.

**Options.**

- A. Keep backend boolean dtype.
- B. Always 0/1 with the argument’s numeric dtype.
- C. Always 0/1 `int8` / `uint8`.

**Working default.** A.

**Impact.** Logic, compress/replicate, `⍸`.

---

## 36. Scalars and 0-d tensors

**Question.** Is a Python `3`, a 0-d tensor, and a 1-element vector the
same in Haply?

They are three different host values. APL has one “scalar” (empty shape).
PyTorch has two tensor shapes that people casually call scalar.

```hy
(setv py 3)                       ; int, no .shape
(setv s  (torch.tensor 3))        ; 0-d, shape (), ndim 0
(setv v  (torch.tensor [3]))      ; 1-d, shape (1,), ndim 1
```

| Form | `type` | `shape` / `size` | `(⍴ ·)` under typical tensor intent | `(≢ ·)` tally |
| --- | --- | --- | --- | --- |
| Python `3` | `int` | none | not a tensor; error, or no shape | not specified |
| `np.int64(3)` | NumPy scalar | `()` on the scalar, not an `ndarray` | not an array | not specified |
| `torch.tensor(3)` | 0-d tensor | `()` | empty 1-d shape vector | `1` |
| `torch.tensor([3])` | vector | `(1,)` | `[1]` | `1` |

Broadcasting also differs: `s` acts like a true scalar against any
shape; `v` is a length-1 vector and only lines up with a trailing `1` or
an equal length. Reductions: `s.sum()` stays 0-d; `v.sum()` becomes 0-d.
Indexing: `s` cannot be indexed; `v[0]` is 0-d.

**Options.**

- A. Python scalars stay Python; 0-d tensors stay 0-d.
- B. Promote Python numbers to 0-d tensors of the default backend.
- C. Follow the backend’s own treatment and document it per function.

**Working default.** A, plus C when both arguments are already tensors.
NumPy scalars (`np.int64`, `np.float64`, …) are not arrays and not
Python numbers: they raise `TypeError` until this item says otherwise.

**Impact.** Shape of results, `⍴`, `≢`, reductions.

---

## 41. Reduce on empty

**Question.** `(⌿ + (⍳ 0))` needs an identity or an error. Dyalog uses
function identities (0 for `+`, 1 for `×`, …).

**Options.**

- A. Use known identities for a listed set; error otherwise.
- B. Always error on empty reduction.
- C. Follow `torch.sum` / `np.sum` defaults (0 for sum, …) even when the
  operand is a user function.

**Working default.** A for built-in scalar operands; B for unknown
operands.

**Impact.** `⌿`, `⌿_`, n-wise reduce.

---

## 42. Each (`¨`) on tensors

**Question.** Dyalog `¨` maps over boxes/items. Tensors have no boxes.

**PyTorch / NumPy analogues** (this is also item 43):

| Wanted Each | Host analogue | Cost |
| --- | --- | --- |
| Scalar `f` on every element | just call `f` (already elementwise) | one kernel; do not loop |
| Same `f` on every major cell | `torch.vmap` / `torch.func.vmap` | compiled batch; closest to “each cell” |
| Split, map, restack | `unbind` + Python map + `stack` | works; Python-level; shapes must agree |
| Ragged / boxed items | `torch.nested` or a list of tensors | not the Haply model (decision 5) |
| NumPy object array of arrays | `dtype=object` | slow; rejected |

Dyalog Each is cheap because items are already boxed. On a tensor, Each
that is not elementwise or `vmap` is a Python loop and fails principle 8
unless the operand is a user function we cannot vectorise.

**Options.**

- A. Drop `¨` in v1; users write Python/Hy loops or `vmap`.
- B. Map over major cells and `stack` the results.
- C. Expand to `torch.vmap` / `numpy.vectorize` when possible.

**Working default.** B as a macro, C when the operand is a Haply scalar
function (then prefer a true elementwise call instead of Each).

**Impact.** Operator catalog, performance story.

---

## 43. Nested-adjacent primitives

**Question.** Mix, split, enclose, disclose, nest, pick, partition, depth,
enlist: Dyalog needs nested arrays. Some have tensor analogues (`stack`,
`unbind`, `unsqueeze`, `flatten`).

Boxes are not unique to Dyalog (J boxes, K/Lisp lists), but they are
almost absent from the tensor stack. The cheap analogue is **another
dimension**, not a nest. NumPy `dtype=object` and `torch.nested` are
ragged containers, not Dyalog arrays, and they fail the cost rule.

| Dyalog | Role | Closest host analogue | Haply |
| --- | --- | --- | --- |
| `⊂` enclose | add a nest layer | `unsqueeze` / wrap in a list | drop (G048) |
| `⊆` nest | enclose if simple | none | drop (G049) |
| `⊃` first / pick | first item; path into boxes | `[0]` / `unbind` / `select` | first cell only (G050) |
| `↑` mix | nest → higher rank | `torch.stack` | drop monadic (G051) |
| `↓` split | rank → nest | `unbind` / `split` | drop monadic (G052) |
| `≡` depth | nest depth | none (`ndim` is rank) | drop monadic (G053) |
| `∊` enlist | flatten nest | `flatten` / ravel | flatten (G038) |
| partitioned enclose | group by mask | `split` / groupby | drop |
| `¨` each | map over items | see item 42 | open |

A Python list of tensors is the honest ragged leftover. It is a host
value, not a Haply array (decision 4).

**Options.**

- A. Drop the glyphs.
- B. Rebind each to a documented tensor analogue.
- C. Keep only those with a clean analogue (`⊃` first cell, `∊` flatten /
  membership).

**Working default.** C. See the “adapted / dropped” rows in
[glyphs.md](glyphs.md).

**Impact.** A large part of the mixed-function set.

---

## 44. APL runtime primitives

**Question.** Execute (`⍎`), format (`⍕`), I-beam (`⌶`), spawn (`&`),
variant (`⍠`), session I/O, system functions.

**Working default.** Drop all of them. Hy and Python already cover eval,
formatting, and threads.

**Impact.** Catalog “dropped” section.

---

## 45. Monadic `≠` (unique mask)

**Question.** Dyalog monadic `≠` is unique mask (nub sieve), not “not
equal”.

**Options.**

- A. Implement unique mask.
- B. Monadic `≠` is an error; unique mask gets another name.
- C. Drop.

**Working default.** A.

**Impact.** Compare catalog, unique (`∪`).

---

## 47. Circular table `○`

**Question.** Dyalog `X○Y` is a large family (trig, hyperbolic, complex
parts) selected by integer `X`.

**Options.**

- A. Full Dyalog table where the backend can.
- B. A useful subset: `1 2 3 ¯1 ¯2 ¯3 5 6 7` (sin/cos/tan and inverses,
  hyperbolic).
- C. Drop `○`; tell people to call `torch.sin`.

**Working default.** B in Phase 2; extend toward A without renaming.

**Impact.** Numeric catalog.

---

## 48. Later operators in v1

**Question.** Which of Key, Stencil, Rank, Power, and At belong in the
first shipped version?

Names for atop, beside, behind, over, and the 3-train are already closed
(decisions 13 and 58: `⋔`). At’s glyph is `⊡` (decision 23). This item
is only about *when* the remaining operators ship.

**Working default.** Composition glyphs and `⋔` are Phase 5, not this
item. Longer trains are item 62. Expand (array `⍀`) is Phase 6. Key,
Stencil, Rank-as-array, Power, and At `⊡` are Phase 8 — do not start
that phase until this item closes or an explicit pull-forward names
which of them ship.

**Impact.** Roadmap in [architecture.md](architecture.md).

---

## 51. Version pins

**Question.** Which Hy, Python, PyTorch, and NumPy versions are supported?

**Working default.** Whatever `flake.lock` pins. Document the resolved
versions in [development.md](development.md) once the shell evaluates.

**Impact.** CI, Nix, packaging.

---

## 54. Testing and oracles

**Question.** Do tests compare against a Dyalog interpreter, against
hand-written tensors, or both?

**Working default.** Hand-written tensor fixtures for all glyphs. Optional
Dyalog oracle later for numeric cases where intent is “same as Dyalog on
simple arrays”.

**Impact.** `tests/`, CI, Nix dependencies.

---

## 55. Package layout

**Question.** Accept the layout in [architecture.md](architecture.md), or
change it before Phase 0?

**Working default.** Accept it until this item is closed with a different
tree.

**Notes.** Phase 7 added `tests/test_numpy.hy`. The file is real; the
proposed tree in [architecture.md](architecture.md) was not updated
because this item is still open.

**Impact.** Imports, macros, tests.

---

## 56. Depth and match (`≡`)

**Question.** Monadic `≡` is depth of a nested array. Without boxes it is
almost always 0 or 1.

**Options.**

- A. Drop monadic `≡`. Keep dyadic match (`torch.equal` / `array_equal`).
- B. Monadic `≡` returns 0 for a Python scalar, 1 for a tensor.
- C. Monadic `≡` returns tensor rank.

**Working default.** A for monadic; implement dyadic match.

**Notes.** `torch.equal` and `np.array_equal` both compare values, not
dtype (`int` 1 matches `float` 1.0; `True` matches `1`). Both treat
NaN as not equal. A stricter “same dtype” match would be a new option
on this item; the working default stays the backend equal.

**Impact.** Compare catalog.

---

## 57. Tally vs first-dimension vs numel

**Question.** `≢Y` is the count of major cells (1 for a scalar).

**Working default.** Dyalog tally on tensors: `shape[0]` if `ndim > 0`,
else `1`. Not `numel`.

**Impact.** `≢`, reductions over major cells.

---

## 62. Longer forks

**Question.** Phase 5 ships the 3-train only: `(⋔ f g h Y)` and
`(⋔ f g h X Y)`. Dyalog also has longer odd-length trains
(`(a b c d e)` is `a b (c d e)`). Does Haply ever accept
`(⋔ a b c d e …)`?

This is not item 48. Rank, Power, Key, Stencil, and At stay there.
Decision 13 already forbids a second train form; even length would be
atop (`⍤`), not a 2-train `⋔`.

**Options.**

- A. Never. The 3-train is the whole product.
- B. Odd length, nested like Dyalog: `(⋔ a b c d e Y)` is
  `(⋔ a b (⋔ c d e) Y)`.
- C. Later, but not in v1. Keep the Phase 5 `TypeError` until a
  decision pulls B forward.

**Working default.** C. No phase number until this item closes or an
explicit pull-forward names one.

**Impact.** `haply/trains.hy`, G077, [operators.md](operators.md).

---

## 63. Python lift on numeric glyphs

**Question.** `?` `⌹` `⊤` `⊥` use `torch.as-tensor` on every argument.
`○` `!` and the rest of the public glyphs use `require-torch`. Tests pin
`(? 6)` and `(? 4 10)`. Should lists and Python numbers be arrays?

**Options.**

- A. Constructors and scalar specs may lift (`?`, `⍳`, shape codes).
  Array operands (`⌹` `⊤` `⊥` `Y`, and every other glyph) stay
  `require-torch`. Lists are not tensors (item 21 notes).
- B. Lift on every glyph (`as-tensor` everywhere).
- C. Keep the current split: only `?` `⌹` `⊤` `⊥` lift.

**Working default.** A. `?` stays a constructor (item 30). `⌹` `⊤` `⊥`
should move to `require-array` when this item closes.

**Notes.** NumPy has no `lgamma`. Monadic and dyadic `!` run the torch
kernel and wrap the result as `ndarray`. That is a lift of the *kernel*,
not of the user’s value type. A native NumPy path can wait.

**Impact.** `haply/numeric.hy`, item 21, tests for `?`.

---

## 64. Replicate mask vs expand mask

**Question.** Expand rejects rank 0, float/complex, and integers other
than 0/1. Replicate accepts any non-negative integer count, does not
check float, and indexes `shape[axis]` on a 0-d `Y` (`IndexError`).

**Options.**

- A. Align replicate’s *checks* with expand: rank ≥ 1, boolean or
  integer dtype. Integer *values* stay counts (0 drops, 2 repeats).
  0-d `Y` is `ValueError`.
- B. Leave replicate torch-ish (`repeat-interleave`, backend errors).
- C. Loosen expand to match replicate (counts > 1 become copies).

**Working default.** A. Applied on the PyTorch path: 0-d `Y` and
float/complex masks are `ValueError`; integer counts stay. The item
stays open until A is chosen.

**Impact.** `replicate` in `_dispatch.hy`, G054, G055.

---

## 65. Intersection duplicates

**Question.** Dyalog `∩` is a set. Haply `(∩ [1 2 3 2] [2 4])` is
`[2 2]` (membership filter, stable from `X`). Tests pin that.

**Options.**

- A. Keep duplicates (current).
- B. Unique like Dyalog (first occurrences from `X` that appear in `Y`).

**Working default.** A. Catalog G041 is `adapt`.

**Impact.** G041, `haply/search.hy`.

---

## 66. Nand and nor on non-booleans

**Question.** Decision 46 splits `∧` `∨`: boolean logic, integer LCM/GCD,
float `ValueError`. `⍲` `⍱` always call `logical_and` / `logical_or`
(integers become truthy).

**Options.**

- A. Boolean only; integer and float are `ValueError`.
- B. Always logical (current).
- C. Same dtype split as `∧` `∨` (nand-of-LCM has no meaning).

**Working default.** B until this closes. A matches the `∧` `∨` domain
if we want one rule for the four glyphs.

**Impact.** G024, G025, `haply/scalar.hy`.

---

## 67. Operator arity errors: expand-time vs runtime

**Question.** `⌿` `⍀` `⍨` raise `TypeError` while the macro expands.
`∘` `⍤` `⍥` `⍛` `⋔` emit `(raise …)` so `pytest.raises` can catch
them. Same class of mistake, two moments.

**Options.**

- A. Always emit a runtime `raise` (testable; one rule).
- B. Expand-time for missing or extra forms (Hy-macro style).
- C. Keep the mix (Phase 3 expand-time, Phase 5+ runtime).

**Working default.** A for new macros. Migrate `⌿` `⍀` `⍨` when those
heads are touched.

**Impact.** `haply/macros.hy`, `haply/trains.hy`, operator tests.

---

## 68. Device of derived index and shape vectors

**Question.** Decision 49 says do not move devices. Item 30 says
constructors with no tensor argument are CPU torch. `(⍴ Y)` has a
tensor argument. Before Phase 7 the shape vector was always CPU; now
`int64-vector` follows `Y.device`. Grade, where, and other index
results already followed `Y`.

**Options.**

- A. Same device as `Y` (decision 49; current).
- B. Always CPU, even when `Y` is CUDA (old monadic `⍴`).
- C. CPU only for constructors (`⍳`, `?`); derived vectors follow `Y`.

**Working default.** A.

**Impact.** G026, `int64-vector`, grade, where, any later index result.

---

## 69. n-wise window longer than the axis

**Question.** `(⌿ n + Y)` when `n` is larger than the reduced axis.
The unknown-operand helper already returned an empty array (axis
length 0, other shape kept). The old known-operand torch path used
`.unfold` and raised `RuntimeError`. Phase 7 uses one helper for both.

**Options.**

- A. Empty result, same dtype, axis length 0 (current).
- B. Backend error (`RuntimeError` / `ValueError`).
- C. Dyalog n-wise (identity or empty depending on `n` vs length).

**Working default.** A. Item 41 still covers empty *reduce* identities;
this is the window that never starts.

**Impact.** `nwise-known`, `nwise-reduce`, operator tests.
