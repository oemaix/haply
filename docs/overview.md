# Overview

## Name

The project is **Haply**. The package name is `haply`.

The name fuses **Hy** and **APL**. It is short, lowercase, and reads as a
normal Python import:

```hy
(import haply [⍴ +])
(require haply.macros [⌿])
```

In English, *haply* means “by chance” or “perhaps”. The name is an intentional
joke: a precise, deterministic tensor library named after accident.

In prose the name is **Haply**. In code and imports it is `haply`
(decision 27).

## What Haply is

Haply is a **Hy library of APL-shaped tensor operations**.

- The host language is **Hy** (Lisp syntax on the Python runtime).
- The reference vocabulary is **Dyalog APL**.
- The data model is **tensors**, not APL arrays.
- The evaluation rule is **Lisp evaluation** (inside-out), not APL
  right-to-left.

A Dyalog expression such as:

```apl
+⌿ A × B
```

is written in Haply as:

```hy
(⌿ + (× A B))
```

The glyphs are Hy functions or macros. They are meant to compile down to
ordinary PyTorch or NumPy work.

## What Haply is not

- Not an APL interpreter.
- Not a source-compatible Dyalog implementation.
- Not a new array language with its own parser.
- Not a custom nested-array runtime.
- Not a replacement for Hy, NumPy, or PyTorch.

Dyalog is the **semantic reference**, not a contract of identity. Where Haply
differs, this documentation states the **intended operation** in tensor terms.
See [semantics.md](semantics.md).

## Who it is for

People who want APL’s compact array vocabulary while writing Hy that sits on
the same tensors used in existing Python, NumPy, and PyTorch code.

Whether a plain-Python import surface is supported, or rejected, is open; see
[undecided.md](undecided.md#22-python-import-surface).

## Host and backends

| Layer | Role |
| --- | --- |
| Hy | Syntax, macros, user programs |
| Python runtime | Host |
| `torch.Tensor` | Primary tensor type (working default) |
| `numpy.ndarray` | Second tensor type |
| Python scalars and sequences | Passed through when that is the honest result |

Haply does not introduce its own array class. The atomic unit is a tensor (or
a native Python value). APL nested arrays and boxes are dropped.

## Design in one page

1. Lisp prefix notation only.
2. Monadic vs dyadic meaning is selected by argument count.
3. Broadcasting is PyTorch/NumPy broadcasting, not APL conformability.
4. Operators and trains are macros.
5. Existing Hy, NumPy, and PyTorch behaviour must remain usable.
6. Abstraction cost stays low: prefer sugar over copies and wrappers.
7. Haply stays coherent with itself even when that differs from APL.
8. Implement PyTorch first, then NumPy, then Python natives.

The binding rules are in [design-principles.md](design-principles.md).

## Documentation map

- Using Haply: [`manual/`](../manual/README.md)
- Vocabulary to implement: [glyphs.md](glyphs.md) and [operators.md](operators.md)
- How to build it: [architecture.md](architecture.md)
- How to work on it: [development.md](development.md)
- Closed vs open questions: [decided.md](decided.md), [undecided.md](undecided.md)
