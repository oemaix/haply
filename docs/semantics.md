# Semantics

Haply uses **Dyalog APL** as its semantic reference. It does not promise
Dyalog compatibility, source portability, or identical edge cases.

This document states the contract: what “Dyalog-inspired” means, which Dyalog
rules Haply keeps, and which rules Haply replaces.

Authoritative Dyalog pages used while writing this suite:

- [Primitive functions](https://docs.dyalog.com/20.0/language-reference-guide/primitive-functions/)
- [Glyphs](https://docs.dyalog.com/20.0/language-reference-guide/glyphs/)
- [Operators summarised](https://docs.dyalog.com/20.0/language-reference-guide/primitive-operators/operators-summarised/)

## How to read an entry

Every implemented glyph in [glyphs.md](glyphs.md) has four layers:

1. **Dyalog name and valence** — the reference meaning (monadic / dyadic).
2. **Haply intent** — what the Haply form does to tensors or native values.
3. **Backend mapping** — the intended NumPy/PyTorch operation.
4. **Deviation** — any deliberate difference from Dyalog.

If layer 2 and layer 1 disagree, layer 2 wins for Haply code. The deviation
must stay visible in the catalog.

## Evaluation

| Topic | Dyalog | Haply |
| --- | --- | --- |
| Syntax | APL infix / juxtaposition | Hy prefix S-expressions |
| Order | Right-to-left | Lisp inside-out |
| Functions | Primitive or defined functions | Hy functions |
| Operators | Primitive operators producing derived functions | Hy macros |
| Trains | Tacit forks / atops | `(fork …)`; atop/beside/behind/over are `⍤` `∘` `⍛` `⍥` |
| Assignment | `←` and modified assignment | Hy `setv` / `setx` / … |
| Comments | `⍝` | Hy `;` |
| Control | `:If`, guards, `→` | Hy `if`, `cond`, `while`, … |

Haply implements a **vocabulary**, not an APL machine.

## Data model

| Topic | Dyalog | Haply |
| --- | --- | --- |
| Atom | Simple scalar (number, character, …) | Python scalar or 0-d tensor (open: [undecided 36](undecided.md#36-scalars-and-0-d-tensors)) |
| Array | Rectangular nested array | `torch.Tensor` or `numpy.ndarray` |
| Nested boxes | First-class | Dropped |
| Rank | Array rank | Tensor `ndim` / `.ndim` |
| Empty arrays | Prototypes and fill elements | Backend empty tensors; no APL prototypes |
| Characters | Character arrays | Dropped (decision 32). Strings stay Hy/Python. |
| Complex | Native complex | In scope where the backend has it (decision 31) |

There is no Haply “box”, no prototype, and no fill element derived from
`⊂∊⊃Y`. Operations that Dyalog fills (reshape, take, expand, replicate) use
the backend’s empty/pad behaviour or an explicit fill argument if one is
added later.

## Shape rules

Dyalog *conformability* (including singleton extension that is not the same
as NumPy broadcasting) is **not** implemented.

Haply uses the broadcasting of the values’ backend:

- two `torch.Tensor` values → `torch` broadcasting;
- two `numpy.ndarray` values → `numpy` broadcasting;
- mixed or Python-native cases follow [undecided 21](undecided.md#21-backend-set)
  and decision 49 (device and dtype: do not move devices; follow the
  backend’s promotion).

## Index origin

Dyalog indexes are controlled by `⎕IO` (default 1). Python, NumPy, and
PyTorch are 0-based.

Decision 28: Haply indexes are **0-based**. There is no `⎕IO`. Dyalog
examples in this suite that use `⍳` or `⌷` are translated with origin 0
when they describe Haply intent.

A later switch to 1-based indexing would be a reversal of decision 28,
not a silent catalog edit.

## Comparison and tolerance

Dyalog comparisons use `⎕CT` / `⎕DCT`. Haply has no system variables.

Working default ([undecided 29](undecided.md#29-comparison-tolerance)):
exact backend comparison (`==`, `torch.eq`, `np.equal`). Tolerant match, if
wanted, will be an explicit function or an extra argument, not a hidden
global.

## Axis convention

Dyalog functions pick a default axis (often first or last) and accept an
axis operator `[k]`.

Haply working convention:

| Marker | Meaning |
| --- | --- |
| Unmarked first-axis glyph (`⌿`, `⊖`, `⍪`, …) | Operate on dimension 0 |
| Trailing `_` (`⌿_`, `⍀_`) | Operate on the last dimension |
| No `/` or `\` | Those glyphs stay with Hy (`/` is division) |

A general axis selector (the Dyalog `[k]` operator) is open
([undecided 34](undecided.md#34-general-axis-selection)).

`_` as “last dimension” is a Haply convention, not a Dyalog glyph meaning.

## Pervasiveness

Dyalog scalar functions are pervasive: they dive into nested arrays.

Haply scalar functions are **elementwise on tensors**. They do not walk
Python lists as if they were nested APL arrays. A Python list is either
rejected or treated as a host sequence, depending on the backend decision.

## Errors

Dyalog has DOMAIN ERROR, LENGTH ERROR, RANK ERROR, INDEX ERROR, and others.

Decision 40: raise ordinary Python exceptions (`TypeError`, `ValueError`,
`IndexError`, or the exception the backend raises). Do not invent an APL
error hierarchy.

## System space

Haply does not implement:

- system variables (`⎕IO`, `⎕ML`, `⎕CT`, `⎕WX`, …);
- system functions (`⎕FMT`, `⎕TS`, …);
- session I/O (`⎕`, `⍞`);
- I-beam (`⌶`);
- namespaces (`#`, `##`);
- component files, GUI, or Dyalog objects.

If a Dyalog primitive *depends* on those (for example Unique Mask under a
particular `⎕ML`), Haply documents a single intended behaviour instead.

## Identity of “same result”

A Haply result is correct when it matches the **Haply intent** in the
catalog, not when it matches Dyalog byte-for-byte.

Tests may use Dyalog as an oracle for simple numeric cases where the intent
is “same as Dyalog on simple numeric arrays, after origin and broadcasting
adjustments”. That is a test tool, not the definition of correctness.
See [undecided 54](undecided.md#54-testing-and-oracles).
