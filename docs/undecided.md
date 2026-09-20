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

Left open on 2026-09-20. Phase 1 implements **PyTorch tensors only**.
Whether every glyph must later accept all three remains the question.

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
| Operators / trains (`⌿`, `fork`, `∘`) | Hy `require` macros | Macros do not run from Python; B needs function fallbacks or drops them |
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

Names for atop, beside, behind, over, and fork are already closed
(decisions 13 and 58). This item is only about *when* the remaining
operators ship.

**Working default.** v1 operators: reduce, scan, replicate, expand, inner
product, outer product, commute, each (if 42 allows). Composition glyphs
and `fork` are specified; Key, Stencil, Rank-as-array, Power, and At are
later.

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

**Impact.** Compare catalog.

---

## 57. Tally vs first-dimension vs numel

**Question.** `≢Y` is the count of major cells (1 for a scalar).

**Working default.** Dyalog tally on tensors: `shape[0]` if `ndim > 0`,
else `1`. Not `numel`.

**Impact.** `≢`, reductions over major cells.
