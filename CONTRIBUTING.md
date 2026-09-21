# Contributing

Haply is specified first, then implemented. Read this page, then the
docs it points to. Do not invent glyphs, trains, or APIs that
[`docs/glyphs.md`](docs/glyphs.md) does not list.

## Where to start

| You want to | Open |
| --- | --- |
| Use shipped names | [`manual/`](manual/README.md) |
| Change behaviour or add a glyph | [`docs/`](docs/README.md), then [`docs/development.md`](docs/development.md) |
| Ask a design question | an issue with the spec template, then a number in [`docs/undecided.md`](docs/undecided.md) |

Dyalog is the semantic reference, not the product. Haply intent on
`torch.Tensor` and `numpy.ndarray` wins. Documented deviations stay.

## Environment

Nix is the official pin (decision 11). Inside the repo:

```sh
nix develop
hy -m pytest tests
```

`PYTHONPATH` is the repository root. Do not require a global `pip install`.
`flake.lock` pins Hy 1.2.0 and the rest of the toolchain.

## Pull requests

1. Find the glyph id in the catalog. If the meaning is open, stop and
   use or add an undecided number. Do not guess a Dyalog edge into
   existence.
2. Tests pin **Haply intent**, not a Dyalog transcript. Minimum: one
   case per valence that exists, one broadcast case for scalar dyads,
   one error case.
3. Keep wrappers thin. Do not mutate inputs. Do not add a Haply array
   class.
4. Operators and trains are Hy macros (`require haply.macros`,
   `require haply.trains`). Do not `import` those modules.
5. Never export Hy-colliding names: `*` `/` `=` `|` `.` `,` `~` `^` `¯` `@`.
6. If the name is public, update the matching [`manual/`](manual/README.md)
   page in the same work.
7. Prose in `docs/` and `manual/` is English.

Closing a design question means **moving** the whole entry from
`undecided.md` to `decided.md` with the same number. Never renumber.

## Out of scope until decided

Python-first API (item 22), Python natives as a third backend
(item 21), and item 48 operators (`⍣` `⌸` `⌺`, rank-`⍤`, At `⊡`).

## License

Contributions land under Apache License 2.0. See [LICENSE](LICENSE).
