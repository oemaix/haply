# Design Principles

These rules are binding. They come from the design notes and from the decision
to treat Dyalog as a semantic reference rather than a clone. Related numbers
are in [decided.md](decided.md).

## 1. Lisp syntax over APL syntax

Every Haply form is a Hy S-expression. There is no APL parser, no strand
notation, and no right-to-left APL evaluation.

APL’s “right-to-left” rule is replaced by ordinary Lisp evaluation:
arguments are evaluated, then the head is applied.

```apl
+⌿ A × B
```

```hy
(⌿ + (× A B))
```

Assignment, comments, control flow, and function definition stay with Hy
(`setv`, `;`, `if`, `defn`, and so on). Haply does not reimplement APL syntax
glyphs such as `←`, `⍝`, `⋄`, or `∇`.

## 2. Arity-based dispatch

In Dyalog, a glyph’s meaning depends on whether it is used monadically or
dyadically. Haply keeps that split. A Haply function inspects the number of
arguments and dispatches.

```hy
(⍴ A)        ; monadic: shape of A
(⍴ shape A)  ; dyadic: reshape A to shape
```

There is no separate monadic-only name unless a Dyalog glyph has only one
valence, or a Hy conflict forces a split (see [glyphs.md](glyphs.md)).

## 3. Tensor-native data

Do not build a Haply array class.

The atomic unit is `torch.Tensor` or `numpy.ndarray`. Python scalars and
built-in sequences are allowed where they are the natural host value.

APL nested arrays (boxes) are dropped. Rank is tensor rank. Cells are tensor
slices along leading or trailing dimensions, not enclosed arrays.

The cheap host analogue of a box is **another dimension**, not a nest.
`stack`, `unbind`, `unsqueeze`, and `flatten` stand in for mix, split,
enclose, and enlist where the catalog keeps a glyph. NumPy object arrays
and `torch.nested` are ragged containers, not Dyalog arrays, and are not
the Haply model. See [undecided 43](undecided.md#43-nested-adjacent-primitives).

## 4. Broadcasting over conformability

Dyalog has strict conformability and a specific scalar-extension rule. Haply
does not implement that rule.

If PyTorch or NumPy broadcasting accepts the shapes, Haply accepts them.
If the backend rejects the shapes, Haply fails with that backend’s error
(unless a later decision defines a thinner wrapper).

This is intentional: Haply should sit in existing AI and numeric workflows
without a second shape calculus.

## 5. Macros for operators and trains

APL *operators* take functions as operands (`⌿`, `¨`, `·`, `∘`, …). The
only train is `⋔`; atop, beside, behind, and over use `⍤`
`∘` `⍛` `⍥` (decisions 13 and 58).

In Haply these are **Hy macros**, not higher-order functions that close over
callables at runtime. The macro should expand to straight PyTorch or NumPy
calls wherever that is possible, so the abstraction can stay close to zero
cost.

Primitive scalar and mixed *functions* are ordinary Hy/Python functions
(decision 38). They are not macros.

## 6. Do not break Hy, NumPy, or PyTorch

Haply must remain a library, not a fork of the host stack.

- Hy programs that never import a conflicting Haply name keep Hy’s meaning.
- NumPy and PyTorch objects returned by Haply are still NumPy and PyTorch
  objects.
- Haply should not replace the language, monkey-patch the backends, or
  require a custom REPL.

Glyphs that collide with Hy (`*`, `/`, `=`, `|`, `.`, `,`, `~`, `^`, …) are
renamed or avoided. See [glyphs.md](glyphs.md) and decisions 14–20.

## 7. Backend preservation

The type that goes in should be the type that comes out, when that is
honest:

| Input | Result |
| --- | --- |
| Python native | Python native |
| `numpy.ndarray` | `numpy.ndarray` |
| `torch.Tensor` | `torch.Tensor` |

Do not silently lift a Python list to a tensor, or a NumPy array to a torch
tensor, unless a later decision defines mixed-backend promotion.

How far this rule extends (Python-only usage, mixed tensor arguments) is
still open; see [undecided.md](undecided.md#21-backend-set) and
[undecided.md](undecided.md#22-python-import-surface). Device and dtype
policy is decision 49.

## 8. Low-cost abstraction

Haply is syntactic sugar over Python, NumPy, and PyTorch.

It should not add an extra array model, avoidable memory copies, or a
heavy runtime interpreter for trains. Prefer:

- backend kernels (`torch.sum`, `np.matmul`, …);
- compile-time macro expansion;
- thin wrappers that only dispatch on type and arity.

A glyph that needs a Python loop over elements, an object-array walk, or
a box runtime is not accepted by default. It must earn its place against
this rule, or be adapted to a kernel, deferred, or dropped.

## 9. Dyalog as reference, Haply as intent

When a glyph is implemented, the Dyalog name and valence are the starting
point. The documentation then states **what Haply intends to do on tensors**.

Impossible or undesirable Dyalog behaviour (nested prototypes, `⎕IO`,
`⎕ML`, session I/O, I-beams, …) is adapted or dropped, never silently
approximated.

Haply must stay coherent with itself even when that differs from APL,
Hy, PyTorch, NumPy, or Python lists. Those hosts already disagree with
each other; copying any one of them at the cost of the tensor model is
not a goal.

APL session and runtime machinery (`⍎`, `⍕`, `⌶`, `⎕`, spawn, system
space) is dropped (decision 44). Character arrays and string processing
are dropped (decision 32); use Hy and Python for that work.

## 10. Development environment

Build and test happen in Nix. The repo root should contain `flake.nix`
and `flake.lock`. Details are in [development.md](development.md).

## 11. Implementation order

PyTorch is the first implementation target. NumPy is Phase 7, after the
PyTorch glyph set is stable. Python natives (scalars, then lists) come
last, and only where they stay honest under decision 9.

Phases 1–6 may ship PyTorch tensors only. Item 21 stays open for whether
every glyph must eventually accept all three backends.
