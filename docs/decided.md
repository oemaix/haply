# Decided

Closed entries from the decision register. Numbers are stable. When you close
an item from [undecided.md](undecided.md), move the whole entry here and
**keep the same number**.

---

## 1. Project name

The project is called Haply. The importable package is `haply`.

The name fuses Hy and APL, is short, and is a deliberate English joke
(“by chance”) for a deterministic library.

---

## 2. Lisp syntax, not APL syntax

All Haply code is Hy S-expressions with prefix heads. Haply does not parse
APL source and does not use APL right-to-left evaluation.

---

## 3. Arity-based dispatch

A glyph chooses monadic or dyadic meaning from the number of arguments.

---

## 4. Tensor-native values

No custom Haply array class. Values are `torch.Tensor`, `numpy.ndarray`, or
honest Python natives.

---

## 5. Nested APL arrays are dropped

Boxes, nest, enclose-as-scalar, and prototypes are not part of the model.
Structure is tensor dimensions.

---

## 6. Broadcasting replaces conformability

Shape agreement is whatever the backend broadcasts. Dyalog conformability
and Dyalog singleton extension are not implemented.

---

## 7. Operators and trains are macros

APL operators and trains are Hy macros so they can expand to backend calls
instead of building runtime closures by default.

---

## 8. Do not break Hy, NumPy, or PyTorch

Haply is a library. It must not replace the host language or wrap tensors
in a type that breaks existing code.

---

## 9. Backend preservation

Python in, Python out; NumPy in, NumPy out; PyTorch in, PyTorch out — when
that is the honest result. No silent cross-backend lift in the core rule.

---

## 10. Low-cost abstraction

Haply is sugar over host operations. Avoid extra copies, extra array types,
and a heavy train interpreter.

---

## 11. Nix development environment

Revised 2026-09-20.

Build and test use Nix. The repository root should contain `flake.nix` and
`flake.lock`. Enter the official shell with `nix develop`.

A second shell, `nix develop .#oracle`, may add Dyalog for a later live
oracle (decision 54). It is not required for Phase 0 tests.

---

## 12. Dyalog is the reference, not the product

Haply documents Dyalog names and valences, then states Haply intent. Full
replication of Dyalog is neither required nor possible.

---

## 13. Fork is the only named train; combinations use APL glyphs

Revised 2026-09-17 (with decision 58).

The only APL train without a dedicated operator glyph is the **fork**.
Haply writes it as `(fork …)`.

Function combination uses the Dyalog operator glyphs, not English names
and not a separate hook form:

| Role | Haply |
| --- | --- |
| Atop | `⍤` |
| Beside (jot) | `∘` |
| Behind | `⍛` |
| Over | `⍥` |

There are no macros named `atop`, `hook`, `beside`, or `over`. J-style
hook is already those four operators (typically beside / behind).

---

## 14. No high minus

Dyalog `¯` is not used. Negation and negative literals use Hy/Python `-`.

---

## 15. No slash or backslash glyphs

Dyalog `/` and `\` are not Haply names. They collide with Hy (`/` is
division). Last-axis reduce, scan, replicate, and expand use the `_`
suffix on the first-axis glyphs: `⌿_`, `⍀_`.

---

## 16. Power is `**`

Dyalog `*` (power / exponential) is Haply `**`. Hy already uses `*` for
multiplication and `**` for power.

---

## 17. Equal-to is `==`

Dyalog `=` is Haply `==`. This avoids shadowing Hy `=`.

---

## 18. Ravel and catenate use `++`

Dyalog `,` is Haply `++`.

---

## 19. Not and without use `≁`

Dyalog `~` is Haply `≁`.

---

## 20. Underscore means last dimension

A trailing `_` on an axis-sensitive glyph means “last dimension”. The
unmarked first-axis glyph keeps Dyalog’s first-axis sense (`⌿`, `⊖`, `⍪`).

---

## 23. Inner product is `·`; combination is jot

Chosen 2026-09-17: option A, plus jot for combination.

Dyalog `.` is Haply `·`. Hy/Python `@` stays matmul and is not a Haply
export. Function combination uses APL jot `∘` (decision 13), not `@` and
not `.`.

---

## 24. Magnitude and residue are `||`

Chosen 2026-09-17: Haply defines `||` for both valences (magnitude and
residue).

Hy and Python keep `abs` and `%`. Haply does not shadow or re-export
them.

---

## 25. AND and OR are `∧` and `∨`

Chosen 2026-09-17: Haply defines `∧` and `∨`.

---

## 26. No backtick or quote glyphs

Chosen 2026-09-17: Haply does nothing with `` ` `` or `'`. They remain
Hy/Python syntax.

---

## 27. Display form

Chosen 2026-09-17: **Haply** in prose; `haply` in code and imports.

---

## 28. Index origin is 0

Chosen 2026-09-17: option A.

Indexes are always 0-based. There is no `⎕IO`. Dyalog examples are
translated when they describe Haply intent.

---

## 31. Complex numbers

Chosen 2026-09-17: option A.

Complex dtypes are in scope wherever the backend has them. Monadic `+`
is conjugate on complex values (identity on reals). No extra promotion
of reals to complex.

---

## 58. Train and combination names

Chosen 2026-09-17.

| Form | Haply |
| --- | --- |
| Fork | `(fork …)` |
| Atop | `⍤` |
| Behind | `⍛` |
| Beside | `∘` |
| Over | `⍥` |

No `(hook …)`. See decision 13.

---

## 52. License

Chosen 2026-09-17: Apache License 2.0 (the Apache Commons license).

The repo `LICENSE` file is that text. Package metadata must match.

---

## 59. Outer product is `∘.`

Chosen 2026-09-17: option A.

```hy
(∘. × A B)
```

Beside remains `∘` (decisions 13, 23, 58). Outer product is the two-character
name `∘.`, not a special case of jot.

Option C would have overloaded `∘` so one head did both beside and outer
product (for example by treating a missing second function, or a lone
dot, as outer). That collides with jot and is rejected.

---

## 39. Keyword aliases

Chosen 2026-09-20: option C.

Glyphs stay the primary names. Official English aliases exist only where
a glyph is hard to type, or where a forced ASCII form already exists
(`**`, `==`, `++`, `||`). Aliases are a didactic layer, not a second
catalog. Spellings are added in [glyphs.md](glyphs.md) when they are
named; they are not invented outside the catalog.

Phase 1 may ship glyphs first and attach aliases later. Do not rename a
glyph to English inside the implementation.

Rejected: A (no teaching names) and B (an English name for every glyph).

---

## 40. Error model

Chosen 2026-09-20: option A.

Haply raises native Python and backend exceptions (`TypeError`,
`ValueError`, `IndexError`, or whatever PyTorch or NumPy raise). There is
no Haply exception hierarchy and no Dyalog-style `DOMAIN ERROR` types.

---

## 46. `∧` / `∨` as LCM / GCD

Chosen 2026-09-20: option B.

Both readings: AND / OR on booleans, LCM / GCD on integers. On `{0,1}`
integers the two coincide (`lcm` is AND, `gcd` is OR), which is why
Dyalog unified them.

Map boolean dtypes to `logical_and` / `logical_or` (or equivalent) and
integer dtypes to the backend `lcm` / `gcd`. Floats are a domain error
(`ValueError`, or the backend’s error). Sign, empty, and `gcd(0, 0)`
follow the backend, not a reconstructed Dyalog table.

---

## 49. Device and dtype policy

Chosen 2026-09-20.

Do not move devices. Follow the backend’s dtype promotion. If devices
differ, let PyTorch raise. Haply does not insert `.to(device)` or a
private promotion table.

---

## 32. Character and string arrays

Chosen 2026-09-20: option A.

Drop character arrays. Strings stay ordinary Hy/Python values and are
not Haply arrays. No character dtype, no `str` treated as a vector of
cells, and no collation alphabet for grade.

`⍕` remains dropped (decision 44). Find, membership, unique, and grade
are numeric (or boolean) tensor operations. Use Hy and Python for text.

---

## 37. Variadic scalar functions

Chosen 2026-09-20: option B.

Hy-style variadic folds for associative scalar ops. Everything else
stays strict APL valence: one or two arguments only.

The associative set is:

| Glyph | Fold | Identity (empty reduce, item 41) |
| --- | --- | --- |
| `+` | sum | `0` |
| `×` | product | `1` |
| `⌈` | maximum | −∞ / dtype min |
| `⌊` | minimum | +∞ / dtype max |
| `∧` | AND / LCM (46) | `1` |
| `∨` | OR / GCD (46) | `0` |

A later associative scalar may join this list only by a catalog edit.
Boolean `≠` (XOR) and `==` (XNOR) are associative in a narrow sense but
are not variadic: monadic `≠` is unique mask.

Not associative, therefore never variadic: `-` `÷` `**` `⍟` `||`
(residue) `<` `≤` `==` `≥` `>` `⍲` `⍱` and all mixed functions.

Rejected: A (no folds) and C (freeze the list at these six with no
later additions).

---

## 53. Import and shadowing policy

Chosen 2026-09-20: option A.

Recommend selective import. Shadowing is opt-in and must preserve
Python-scalar behaviour (decision 9). Modules that do not import a
Haply name keep Hy’s meaning (decision 8).

**Never export** (already closed by 14–19, 23, 24):

| Hy / host | Why it stays with Hy | Haply name |
| --- | --- | --- |
| `*` | multiply | `**` (16) |
| `/` | divide | `÷` and `⌿_` (15) |
| `=` | Hy equality / name | `==` (17) |
| `\|` | reserved | `\|\|` (24) |
| `.` | attribute access | `·` (23) |
| `,` | awkward as a Hy symbol | `++` (18) |
| `~` | bitwise-not confusion | `≁` (19) |
| `^` | Hy XOR / host name | `∧` (25) |
| `¯` | Hy already has `-` | `-` (14) |

**May export** (only after `import haply`):

| Name | Hy meaning | Haply meaning |
| --- | --- | --- |
| `+` | variadic host add | conjugate / add (31, 37) |
| `-` | host sub / negate | negate / sub |
| `<` `>` | host compare | elementwise compare |
| `**` | host power | exp / power (16) |

Not shadowed: `<=` `>=` (Haply uses `≤` `≥`), `and` `or` `not` (`∧` `∨`
`≁`), `%` (`||`), `in` (`∊`).

Rejected: B (never export `+` `-` `<` `>`) and C (a `haply.strict` /
`haply.hy` split).

---

## 38. Function vs macro for scalar primitives

Chosen 2026-09-20: option A.

Primitives are functions. Only operators and trains are macros
(decision 7). A function name is enough for `(⌿ + A)`: the operator
macro pattern-matches the symbol and emits a kernel.

**Same situation as `+`** — Hy already owns the name (decision 53):

| Name | Hy | Haply extra |
| --- | --- | --- |
| `+` | variadic add; monadic identity | monadic conjugate (31); tensor add |
| `**` | power | monadic `exp` (16) |
| `-` | negate / sub | tensor elementwise; monadic already matches Hy |
| `<` `>` | host compare | tensor elementwise; no monadic APL meaning |

`**` is the closest twin of `+`: the monadic meaning changes. `-` `<` `>`
are milder.

**Same operand need, different names.** Any primitive used as an operator
operand must stay a resolvable function: `×` `÷` `⌈` `⌊` `∧` `∨` `==`
and the rest. `(⌿ × A)` and `(· + × X Y)` would be painful if those
heads were macros.

**Not this situation.** `*` `/` `=` `|` `.` `,` `~` are never Haply
exports. Hy macros such as `and` `or` `if` are not shadowed (`∧` `∨`).

Rejected: B (primitives as macros) and C (leave a compiler hook open in
the decision). Inlining later does not require reversing A.

---

## 50. In-place operations

Chosen 2026-09-20.

No in-place Haply glyphs. Haply does not export `+=`, `add_`, or any
other mutating form. Users who want mutation call backend methods
themselves (`tensor.add_`, `copy_`, …).

Haply functions return a result; they do not write through an argument.
This avoids aliasing surprises and matches the low-cost, library-not-fork
rules (decisions 8 and 10).

---

## 60. Shared glyphs for function vs operator

Chosen 2026-09-20: option A.

Keep the Dyalog overload on the first-axis glyphs and their `_` last-axis
forms. The same head is replicate or expand when the operand is an
array, and reduce or scan when the operand is a function:

```hy
(⌿ mask A)    ; replicate along axis 0
(⌿ + A)       ; reduce + along axis 0
(⌿_ mask A)   ; replicate last
(⌿_ + A)      ; reduce last
(⍀ mask A)    ; expand first
(⍀ + A)       ; scan first
```

The macro inspects the operand: a known function name or a callable is
the operator reading; an array value is the function reading. Prefer
expansion-time inspection; fall back to runtime if the operand is not
obvious.

Haply still does not export `/` or `\` (decisions 15 and 20).

Rejected: B (split replicate to another name) and C (the other split).
