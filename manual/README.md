# Haply user manual

This directory is the book for **people and assistants who write programs
with Haply**. It describes only names that are exported and tested.

The specification for *building* Haply stays in [`docs/`](../docs/README.md).
When the two disagree, the catalog in `docs/glyphs.md` wins and this
manual must be fixed.

Prose here is English.

## Status

Phases 0–4 are available on `torch.Tensor`. NumPy and a Python-first
import surface are not shipped. Operators are macros:
`(require haply.macros [⌿ · outer …])`. Fork and jot (`∘` `⍤` `⍥` `⍛`)
are Phase 5. Expand is Phase 6. NumPy is Phase 7. Power, key, stencil,
rank, and At are Phase 8.

## What Haply is

A Hy library of APL-shaped tensor functions. Prefix Lisp calls, backend
broadcasting, no APL parser, no Haply array type.

```hy
(import torch)
(import haply [⍴ × ⌽])

(setv A (torch.tensor [[1 2 3] [4 5 6]]))
(⍴ A)           ; tensor([2, 3])
(⌽ (× A A))     ; reverse last after times
```

## Install

There is no PyPI release yet.

To **develop Haply**, use the Nix shell. `PYTHONPATH` is the repository
root, so `(import haply)` works there.

```sh
nix develop
```

To **use Haply from another project**, depend on this repo. A local
editable path is the usual choice while the phases are still landing:

```sh
uv add --editable /path/to/haply
```

Pin a Git revision when you need a frozen tree:

```sh
uv add "haply @ git+https://github.com/oemaix/haply.git@<commit>"
```

The other project must provide Hy 1.2.0 and a `torch` that Haply can
import. Nix remains the version pin for developing Haply itself.

## Import

Select the glyphs you need. Do not star-import.

```hy
(import haply [⍴ × ÷ ⌽ ++ ⍳])
(require haply.macros [⌿ · outer])
```

Hy macros and functions do **not** share a namespace. After
`(import haply [+])`, a call `(+ x y)` is still Hy’s `+`. Use the
function object when you need Haply’s meaning (conjugate, monadic
`exp`, …):

```hy
(import haply.scalar :as sc)

(sc.+ z)        ; conjugate on complex; identity on reals
(sc.** y)       ; exp — (`**` y) is a Hy syntax error
(sc.< a b)      ; Haply comparison; (< a b) stays Hy
```

Glyphs that are not Hy core macros (`×` `÷` `⍴` `||` `++` …) are
ordinary calls after `import`.

## How a call looks

| Form | Meaning |
| --- | --- |
| `(g Y)` | monadic |
| `(g X Y)` | dyadic |
| `(g A B C)` | fold, only for `+` `×` `⌊` `⌈` `∧` `∨` |
| `(op f Y)` / `(op f X Y)` | operator macro after `require haply.macros` |

No APL infix. No axis bracket. Wrong arity raises `TypeError`.

Index origin is 0. Comparisons are exact. Results stay
`torch.Tensor` except where a page says otherwise (`≢` tally is a
Python `int`; `≡` match is a Python `bool`).

## Renamed glyphs

These Dyalog characters are **not** Haply names.

| Dyalog | Haply | Why |
| --- | --- | --- |
| `*` | `**` | Hy multiply |
| `=` | `==` | Hy `=` |
| `,` | `++` | awkward as a Hy symbol |
| `~` | `≁` | bitwise-not confusion |
| `\|` | `\|\|` | reserved by the host |
| `.` | `·` | attribute access (inner product) |
| `/` `\` | `⌿_` `⍀_` | Hy division; last-axis slash |
| `∘.` | `outer` | Hy cannot parse `∘.` |

## Pages

| Page | Contents |
| --- | --- |
| [scalars.md](scalars.md) | Arithmetic, compare, logic, circular, factorial |
| [structure.md](structure.md) | Shape, ravel, reverse, take/drop, match, tally |
| [search.md](search.md) | Iota, where, membership, unique, grade, find, index |
| [operators.md](operators.md) | Reduce, scan, replicate, commute, inner/outer, each |
| [numeric.md](numeric.md) | Roll/deal, matrix inverse/divide, encode/decode |
| [recipes.md](recipes.md) | Short programs that run today |
| [ai.md](ai.md) | Compact sheet for assistants |

## Not in this manual yet

- Fork and jot (`fork` `∘` `⍤` `⍥` `⍛`) — Phase 5
- Expand (`⍀` with an array operand) — Phase 6
- NumPy backend — Phase 7
- Power, key, stencil, rank, At — Phase 8
- English aliases (glyphs are the names)
- Runtime `__doc__` on every export
