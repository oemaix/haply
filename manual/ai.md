# Haply for assistants

Read this first when writing or reviewing **user** Hy that imports
`haply`. For implementing Haply itself, use `docs/` and `.cursor/rules`,
not this file.

## Facts

- Install: no PyPI. Path or git dependency. Nix shell to develop Haply.
- Host: Hy 1.2.0, prefix S-expressions, Lisp evaluation.
- Values: `torch.Tensor` (only backend shipped). No Haply array class.
- Broadcast: PyTorch. Not APL conformability.
- Index origin: 0. Compare: exact. No prototypes.
- Public import: `(import haply [⍴ × ⌽ ⍳ …])` — selective, never `*`.
- Operators: `(require haply.macros [⌿ ⌿_ ⍀ ⍀_ ⍨ · ∘· ¨ ∘ ⍤ ⍥ ⍛])`.
- Train: `(require haply.trains [⋔])`. 3-train only.
- No Python-first API.

## Call-position traps

Hy macros and functions do **not** share a namespace. Importing the name
does not shadow a core macro.

| You write | What runs |
| --- | --- |
| `(+ t t)` | Hy `+` (fine for tensor add) |
| `(sc.+ z)` | Haply conjugate / plus / fold |
| `(** y)` | syntax error |
| `(sc.** y)` | Haply `exp` |
| `(< a b)` | Hy compare |
| `(sc.< a b)` | Haply elementwise `<` |
| `(⌿ + Y)` | Haply reduce (macro; `+` is the operand symbol) |

Safe ordinary calls: `×` `÷` `⍟` `||` `⌊` `⌈` `≤` `==` `≥` `≁` `∧` `∨`
`⍲` `⍱` `⍴` `++` `⍪` `⌽` `⊖` `⍉` `↑` `↓` `⊢` `⊣` `≠` `≡` `≢` `○` `!`
`⍳` `⍸` `∊` `⌷` `⊃` `∪` `∩` `⍋` `⍒` `⍷` `?` `⌹` `⊤` `⊥`.

## Shipped names

**Scalar.** `+` `-` `×` `÷` `**` `⍟` `||` `⌊` `⌈` `<` `≤` `==` `≥` `>`
`≁` `∧` `∨` `⍲` `⍱` `○` `!`

Folds (`+` `×` `⌊` `⌈` `∧` `∨`): 1 = monad, 2 = dyad, 3+ = reduce over
*arguments*, not over an axis. Axis reduce is `(⌿ + Y)`.

**Structure.** `⍴` `++` `⍪` `⌽` `⊖` `⍉` `↑` `↓` `⊢` `⊣` `≠` `≡` `≢`

**Search.** `⍳` `⍸` `∊` `⌷` `⊃` `∪` `∩` `⍋` `⍒` `⍷`

**Numeric.** `?` `⌹` `⊤` `⊥`

**Operators** (`require haply.macros`). `⌿` `⌿_` `⍀` `⍀_` `⍨` `·`
`∘·` `¨` `∘` `⍤` `⍥` `⍛`. Not `∘.` (write `∘·`). `⍤` is
atop only; an integer right operand is a `TypeError`.

**Train** (`require haply.trains`). `⋔` — 3-train only.

## Do / do not

- Do write `(g Y)` / `(g X Y)`. Do not write infix `X g Y` or `g⌿Y`.
- Do use Haply names from the tables. Do not invent `*`, `/`, `=`, `,`,
  `~`, `|`, `.`, `¯`, `∘.`, `@`, or English aliases.
- Do document reshape as “size must match”. Do not recycle like Dyalog.
- Do treat `↑` `↓` as dyadic only; pad is zero. Do not call them monadically.
- Do treat `≡` as dyadic match (Python `bool`). Do not ask for depth.
- Do treat `≢` monad as `shape[0]` (Python `int`), not `numel`.
- Do treat monadic `≠` as a same-shape ravel unique-mask.
- Do use `||` as magnitude / residue of `Y` by `X`.
- Do treat `⍳` not-found as `n`. Do treat `⍋` as numeric only.
- Do treat `(⌹ X Y)` as solve `Y B = X`. Do treat `?` bounds as `[0, n)`.
- Do `require` operators and `⋔`; kernel expansions need `torch` in the file.
- Do treat `(⍀ mask Y)` as expand (0/1 or boolean; fill is zero).
- Do treat `(⌿ mask Y)` replicate as rank ≥ 1 and boolean/integer mask.
- Do treat `⊃` of an empty first axis as `ValueError`.
- Do treat Haply `⍛` as `(f (g Y))` / `((g X) f Y)`, not Dyalog behind.
- Do not import NumPy through Haply.

Pages: [README](README.md), [scalars](scalars.md),
[structure](structure.md), [search](search.md),
[operators](operators.md), [numeric](numeric.md), [recipes](recipes.md).
