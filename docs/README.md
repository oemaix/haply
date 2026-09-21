# Haply Documentation

Haply is a Hy library that brings Dyalog-inspired APL glyphs and operators to
PyTorch and NumPy tensors. It is not an APL interpreter and not a Dyalog clone.

This suite is the official specification for implementation. Code should follow
these documents. Open questions live in the undecided register and keep their
numbers when they move to the decided register.

To *use* Haply, start at [`manual/`](../manual/README.md).

## Read in this order

| Document | Purpose |
| --- | --- |
| [overview.md](overview.md) | What Haply is, what it is not, and the intended user |
| [design-principles.md](design-principles.md) | Binding design rules |
| [semantics.md](semantics.md) | Dyalog as reference; Haply intent; documented deviations |
| [glyphs.md](glyphs.md) | Every Dyalog primitive and what Haply will implement |
| [operators.md](operators.md) | Operators, trains, and combinators |
| [architecture.md](architecture.md) | Package layout, dispatch, macros, and implementation phases |
| [development.md](development.md) | Nix environment, tests, and how to add a glyph |
| [glossary.md](glossary.md) | Terms used in this suite |
| [decided.md](decided.md) | Numbered decisions that are closed |
| [undecided.md](undecided.md) | Numbered questions still open |

## Decision register

Decisions use one number space across both lists.

- New questions get the next free number and start in [undecided.md](undecided.md).
- When a question is closed, **move the whole entry** to [decided.md](decided.md).
- **Never renumber.** Number 23 stays 23 after it is decided.
- If a decision is later reversed, keep the number and record the reversal in
  place. Open a new number only for a new question.

Working defaults in the undecided list are temporary so development can start.
They are not decisions.

## Status

Phases 0–7 are implemented. Continue with Phase 8 (item 48 operators)
in [architecture.md](architecture.md).
