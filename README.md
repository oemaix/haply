# Haply

Haply is a Hy library of Dyalog-inspired APL glyphs for PyTorch and NumPy
tensors.

It is not an APL interpreter and not a Dyalog clone. The host language is
Hy. The data model is tensors. The evaluation rule is Lisp prefix
evaluation, not APL right-to-left.

```hy
(import haply [⍴ × +])
(require haply.macros [⌿])

(setv A (torch.tensor [[1 2 3] [4 5 6]]))
(⍴ A)              ; shape
(⌿ + (× A A))      ; +⌿ A × A  — reduce-plus after times
```

The name fuses **Hy** and **APL**. In English, *haply* means “by chance”,
which is a joke for a deterministic numeric library.

## Status

The project is specified and ready for implementation. Start with
[docs/architecture.md](docs/architecture.md) (Phase 0, then Phase 1).

The official specification lives in [`docs/`](docs/README.md).

## Design in brief

- **Lisp syntax.** `(⌿ + (× A B))`, never `+⌿ A × B` as source.
- **Arity dispatch.** `(⍴ A)` is shape; `(⍴ shape A)` is reshape.
- **Tensors, not APL arrays.** `torch.Tensor` or `numpy.ndarray`. No boxes.
- **Broadcasting, not conformability.** If the backend allows the shapes,
  Haply allows them.
- **Macros for operators and trains.** `⌿`, `·`, `∘`, `(fork …)`, and
  friends expand toward backend kernels.
- **Do not break Hy, NumPy, or PyTorch.** Conflicting glyphs are renamed
  (`**`, `==`, `++`, `≁`, `⌿_`, `||`, `·`).
- **Low-cost sugar.** Prefer a thin call to `torch` / `numpy` over a new
  runtime.

Dyalog is the **semantic reference**. Each operation documents what Haply
intends to do on tensors, including deliberate deviations (0-based
indexes, no prototypes, reshape that does not recycle, and others). See [docs/semantics.md](docs/semantics.md) and
[docs/glyphs.md](docs/glyphs.md).

## Documentation

| Document | Contents |
| --- | --- |
| [docs/README.md](docs/README.md) | Index and how the decision register works |
| [docs/overview.md](docs/overview.md) | What Haply is |
| [docs/design-principles.md](docs/design-principles.md) | Binding rules |
| [docs/glyphs.md](docs/glyphs.md) | Every glyph and operator we will implement |
| [docs/operators.md](docs/operators.md) | Reduce, inner product, trains, … |
| [docs/architecture.md](docs/architecture.md) | Package layout and phases |
| [docs/development.md](docs/development.md) | Nix, tests, how to add a glyph |
| [docs/decided.md](docs/decided.md) | Closed decisions (stable numbers) |
| [docs/undecided.md](docs/undecided.md) | Open questions (same numbers when moved) |

Open questions keep their number when they are decided. Do not renumber.

## Development

Build and test in Nix. A `flake.nix` at the repository root is required
(see [docs/development.md](docs/development.md)).

```sh
nix develop
```

Implementation order: Phase 0 scaffolding, then the Phase 1 scalar
checklist in [docs/glyphs.md](docs/glyphs.md#12-implementation-checklist-what-we-will-build).

## License

Apache License 2.0. See [LICENSE](LICENSE) and
[decision 52](docs/decided.md#52-license).
