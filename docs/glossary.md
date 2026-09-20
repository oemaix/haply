# Glossary

| Term | Meaning in Haply |
| --- | --- |
| **APL** | Array Programming Language. Haply borrows vocabulary from it; it does not host it. |
| **Arity** | Number of arguments. One argument is monadic; two is dyadic. |
| **Backend** | The native implementation that actually runs the work: Python, NumPy, or PyTorch. |
| **Broadcasting** | NumPy/PyTorch shape extension. Replaces Dyalog conformability. |
| **Cell** | A sub-tensor along a chosen axis (first or last, unless a later axis selector exists). Not an APL box. |
| **Conformability** | Dyalog’s shape-agreement rule. Haply does not implement it. |
| **Decision register** | The numbered lists in `decided.md` and `undecided.md`. Numbers never change. |
| **Deviation** | A documented place where Haply intent differs from Dyalog. |
| **Dyalog** | The APL implementation used as Haply’s semantic reference. |
| **Function** | An operation on arrays/tensors. Haply functions are Hy/Python callables. |
| **Glyph** | A symbol such as `⍴` or `⌿`. In Haply it is a Hy name. |
| **Hy** | A Lisp that compiles to Python. Haply’s host language. |
| **Intent** | What a Haply operation is specified to do, even if Dyalog differs. |
| **Macro** | A Hy compile-time form. Required for operators and trains. |
| **Major cells** | Sub-tensors along axis 0. Tally (`≢`) counts them. |
| **Monadic** | Called with one argument: `(⍴ A)`. |
| **Dyadic** | Called with two arguments: `(⍴ shape A)`. |
| **Operand** | A function (or array) given to an operator. In `(⌿ + A)`, `+` is the operand. |
| **Operator** | A form that takes functions and produces a derived operation (`⌿`, `¨`, `·`, …). |
| **Pervasive** | Dyalog term: scalar functions descend into nested arrays. Haply uses elementwise tensor ops instead. |
| **Prototype / fill** | Dyalog empty-array ghosts. Haply does not have them. |
| **Rank** | Number of dimensions. Tensor `ndim`, not APL nest depth. |
| **Reduce** | Collapse one axis by repeating a dyadic function (`(⌿ + A)`). |
| **Scan** | Prefix (or axis-wise) cumulative application of a dyadic function. |
| **Tally** | Number of major cells (`≢`). |
| **Train** | Tacit combination of functions. Haply’s only named train is `(fork …)`. Atop, beside, behind, and over are `⍤` `∘` `⍛` `⍥`. |
| **Working default** | Temporary choice on an undecided item so coding can start. Not a decision. |
