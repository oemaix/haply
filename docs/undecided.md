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

**Options.**

- A. All three, for every implemented glyph.
- B. PyTorch and NumPy only; Python scalars only where the backend already
  accepts them.
- C. PyTorch first; NumPy later; Python natives only as a convenience.

**Working default.** A for scalar arithmetic; C is acceptable for Phase 1
if a glyph is tensor-only and documented as such.

**Impact.** Dispatch tables, tests, and mixed-type errors.

---

## 22. Python import surface

**Question.** Haply is a Hy library. Should plain Python `import haply` be
supported, documented, or explicitly rejected?

**Options.**

- A. Hy is the only supported surface. Python import may exist as an
  implementation detail but is not a product.
- B. Official Python API with the same glyphs as Unicode identifiers.
- C. Official Python API with ASCII aliases only.

**Working default.** A. Implement in Hy; do not advertise a Python API yet.

**Impact.** Packaging, `__init__.py`, docs, and name mangling.

---

## 29. Comparison tolerance

**Question.** Dyalog uses `⎕CT`. Haply has no system variables.

**Options.**

- A. Exact comparison only.
- B. Explicit tolerant functions (`≈`, `match-at`, …).
- C. Optional keyword on `==` / `≡`.

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

## 32. Character and string arrays

**Question.** Dyalog is rich in character arrays. Tensors are numeric.

**Options.**

- A. Drop character arrays. Use Python strings outside Haply.
- B. Support NumPy/PyTorch string or bytes dtypes where they exist.
- C. Treat Python `str` as a vector of characters.

**Working default.** A for v1.

**Impact.** `⍕`, find, membership, grade on characters.

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

**Options.**

- A. Python scalars stay Python; 0-d tensors stay 0-d.
- B. Promote Python numbers to 0-d tensors of the default backend.
- C. Follow the backend’s own treatment and document it per function.

**Working default.** A, plus C when both arguments are already tensors.

**Impact.** Shape of results, `⍴`, `≢`, reductions.

---

## 37. Variadic scalar functions

**Question.** Hy `(+ 1 2 3)` is legal. Dyalog `+` is strictly monadic or
dyadic.

**Options.**

- A. Strict APL valence: 1 or 2 arguments only.
- B. Hy-style variadic folds for associative scalar ops.
- C. Variadic only for `+` `×` `⌈` `⌊` `∧` `∨`.

**Working default.** B for associative arithmetic and logic; A for the
rest.

**Impact.** Function signatures, tests, documentation examples.

---

## 38. Function vs macro for scalar primitives

**Question.** Decision 7 forces macros for operators. Should `+` itself be
a function or a macro?

**Options.**

- A. Functions. Only operators/trains are macros.
- B. Macros that inline backend ops.
- C. Functions with optional compiler helpers later.

**Working default.** A.

**Impact.** `require` vs `import`, inlining, first-class use of `+` as an
operand: `(⌿ + A)` needs `+` to be a resolvable name.

---

## 39. Keyword aliases

**Question.** Should every glyph also have an English name (`shape`,
`reduce`, `iota`)?

**Options.**

- A. Glyphs only.
- B. Official aliases for every implemented glyph.
- C. Aliases only where a glyph is hard to type.

**Working default.** C later; A for Phase 1 (glyphs plus the forced ASCII
forms `**`, `==`, `++`, `||`).

**Impact.** Public API size, docs, Python surface (item 22).

---

## 40. Error model

**Question.** APL domain/length/rank errors vs Python exceptions.

**Options.**

- A. Native Python / backend exceptions only.
- B. A small Haply exception hierarchy that wraps backend errors.
- C. Dyalog-like error names.

**Working default.** A.

**Impact.** Tests, user code, macros.

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

## 46. `∧` / `∨` as LCM / GCD

**Question.** On Booleans, Dyalog `∧` / `∨` are AND / OR. On integers they
are LCM / GCD.

**Options.**

- A. Boolean only.
- B. Full Dyalog (LCM/GCD on integers, AND/OR on 0/1).
- C. Split names: `∧` boolean, `lcm` / `gcd` separate.

**Working default.** B if both arguments are integral; A if boolean.

**Impact.** Logic vs number theory, tests.

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

## 49. Device and dtype policy

**Question.** GPU devices, dtype promotion, mixing CUDA and CPU.

**Working default.** Do not move devices. Follow the backend’s promotion.
If devices differ, let PyTorch raise.

**Impact.** Every binary function.

---

## 50. In-place operations

**Question.** Should Haply expose in-place forms (`+=`, `add_`)?

**Working default.** No in-place Haply glyphs in v1. Users call backend
in-place methods themselves.

**Impact.** API surface, aliasing bugs.

---

## 51. Version pins

**Question.** Which Hy, Python, PyTorch, and NumPy versions are supported?

**Working default.** Whatever `flake.lock` pins. Document the resolved
versions in [development.md](development.md) once the shell evaluates.

**Impact.** CI, Nix, packaging.

---

## 53. Import and shadowing policy

**Question.** `(import haply [+])` shadows Hy `+`. Decision 8 forbids
breaking Hy, but the design notes show importing `+`.

**Options.**

- A. Recommend selective import. Shadowing is opt-in and must preserve
  Python-scalar behaviour (decision 9).
- B. Never export Hy-colliding names; use only APL letters (`×` not a
  shadowed `*`).
- C. A `haply.strict` vs `haply.hy` namespace split.

**Working default.** A. `+`, `-`, `<`, `>` may be imported; `*`, `/`, `=`,
`|`, `.`, `,`, `~` are never Haply exports.

**Impact.** Public names, tutorials, decision 8.

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

---

## 60. Shared glyphs for function vs operator

**Question.** In Dyalog, `/` is both replicate (function) and reduce
(operator). Haply already split `/` away. Do `⌿` / `⌿_` stay overloaded
(function when the “operand” is an array, operator when it is a function)?

**Options.**

- A. Keep Dyalog overload: `(⌿ mask A)` replicate; `(⌿ + A)` reduce.
- B. Split: `⌿` reduce only; `rep` / `⌿*` or similar for replicate.
- C. Split the other way.

**Working default.** A, using macro head inspection (name vs value).

**Impact.** Macro implementation complexity, error messages.
