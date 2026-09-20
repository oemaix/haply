# Haply for assistants

Read this first when writing or reviewing **user** Hy that imports
`haply`. For implementing Haply itself, use `docs/` and `.cursor/rules`,
not this file.

## Facts

- Host: Hy 1.2.0, prefix S-expressions, Lisp evaluation.
- Values: `torch.Tensor` (only backend shipped). No Haply array class.
- Broadcast: PyTorch. Not APL conformability.
- Index origin: 0. Compare: exact. No prototypes.
- Public import: `(import haply [⍴ × ⌽ …])` — selective, never `*`.
- No `require haply.macros` yet. No trains. No Python-first API.

## Call-position traps

Hy macros keep the call. Importing the name does not shadow it.

| You write | What runs |
| --- | --- |
| `(+ t t)` | Hy `+` (fine for tensor add) |
| `(sc.+ z)` | Haply conjugate / plus / fold |
| `(** y)` | syntax error |
| `(sc.** y)` | Haply `exp` |
| `(< a b)` | Hy compare |
| `(sc.< a b)` | Haply elementwise `<` |

Safe ordinary calls: `×` `÷` `⍟` `||` `⌊` `⌈` `≤` `==` `≥` `≁` `∧` `∨`
`⍲` `⍱` `⍴` `++` `⍪` `⌽` `⊖` `⍉` `↑` `↓` `⊢` `⊣` `≠` `≡` `≢` `○` `!`.

## Shipped names

**Scalar.** `+` `-` `×` `÷` `**` `⍟` `||` `⌊` `⌈` `<` `≤` `==` `≥` `>`
`≁` (monad only) `∧` `∨` `⍲` `⍱` `○` `!`

Folds (`+` `×` `⌊` `⌈` `∧` `∨`): 1 = monad, 2 = dyad, 3+ = reduce over
*arguments*, not over an axis.

**Structure.** `⍴` `++` `⍪` `⌽` `⊖` `⍉` `↑` `↓` `⊢` `⊣` `≠` `≡` `≢`

## Do / do not

- Do write `(g Y)` / `(g X Y)`. Do not write infix `X g Y` or `g⌿Y`.
- Do use Haply names from the tables. Do not invent `*`, `/`, `=`, `,`,
  `~`, `|`, `.`, `¯`, or English aliases.
- Do document reshape as “size must match”. Do not recycle like Dyalog.
- Do treat `↑` `↓` as dyadic only; pad is zero. Do not call them monadically.
- Do treat `≡` as dyadic match (Python `bool`). Do not ask for depth.
- Do treat `≢` monad as `shape[0]` (Python `int`), not `numel`.
- Do treat monadic `≠` as a same-shape ravel unique-mask.
- Do use `||` as magnitude / residue of `Y` by `X`.
- Do not `require` operators. Use `torch.sum` / `torch.matmul` until Phase 3.
- Do not import NumPy through Haply.

Pages: [README](README.md), [scalars](scalars.md),
[structure](structure.md), [recipes](recipes.md).
