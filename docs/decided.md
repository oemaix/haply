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
