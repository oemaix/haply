# Development

Haply is specified first, then implemented against this suite. This page
is the working agreement for how to build and test.

## Nix

Decision 11: build and test in Nix. The repository root should contain
`flake.nix` and `flake.lock`.

The official `devShells.default` provides:

- a pinned Python;
- Hy;
- PyTorch;
- NumPy;
- a test runner (`pytest` with a Hy-friendly setup, or the equivalent);
- `uv` for ad-hoc extras. Nix remains the official pin.

`devShells.oracle` adds Dyalog (unfree; entering it accepts the Dyalog
license). A live interpreter is optional until [decision 54](undecided.md#54-testing-and-oracles)
asks for it.

Version pins are [undecided 51](undecided.md#51-version-pins). They live in
`flake.lock` (`nixos-26.05` tarball). Working default from that lock:

| Tool | Version |
| --- | --- |
| Python | 3.13.15 |
| Hy | 1.2.0 |
| NumPy | 2.4.4 |
| PyTorch | 2.11.0 |
| pytest | 9.0.3 |
| uv | 0.11.21 |

Typical loop:

```sh
nix develop
# inside the shell:
hy -m pytest tests
```

Do not require a global `pip install` as the official path.

## Layout

Follow [architecture.md](architecture.md). New modules need a reason; do
not add a parallel package.

## Adding a primitive

1. Find its id in [glyphs.md](glyphs.md).
2. If the meaning is still open, stop and add or use an undecided number.
   Do not guess a Dyalog edge case into existence.
3. Write tests for the **Haply intent** (not a Dyalog transcript dump).
4. Implement the function in the module named by the architecture.
5. If it is an operator, implement a macro and at least one kernel
   expansion (`+` for reduce, `+ ×` for inner product).
6. Update the glyph row only if you discovered a real deviation that the
   catalog missed. Do not silently “fix” intent to match Dyalog.

## Adding an operator or train

See [operators.md](operators.md).

- Operators live in `haply/macros.hy` and are `require`d.
- Trains live in `haply/trains.hy`.
- Users must be able to pass Haply `+` as an operand: `(⌿ + Y)`.

## Tests

Decision 54 default: hand-written tensor fixtures. A live Dyalog oracle
is optional and later.

Minimum for each implemented glyph:

- one monadic case, if the valence exists;
- one dyadic case, if the valence exists;
- one broadcast case for scalar dyadics;
- one error case (wrong arity or impossible shape).

Cover both `torch.Tensor` and `numpy.ndarray` once dispatch exists.
Python natives at least for `+` `-` `×` `÷` `**` `||`.

## Style

- Glyphs are first-class names. Do not rename `⍴` to `rho` in the
  implementation unless decision 39 says so.
- Keep wrappers thin.
- Do not mutate input tensors.
- Do not import Haply names in Hy modules that still need Hy `+` / `=`
  unless you mean to shadow them.
- Comments in code are for intent and deviations, not for retelling
  Dyalog’s manual.

## Decision hygiene

- New design questions get the next free number in
  [undecided.md](undecided.md).
- Closing a question means **moving** the entry to
  [decided.md](decided.md) with the same number.
- Never reuse a number.
- Working defaults may be used in code. When the decision lands, update
  code and the glyph row together.

## Documentation

If implementation changes intended behaviour, change the docs in the same
work. The catalog is the spec.

Prose in `docs/` is English, matching the design notes and the public
identifiers.

## Out of scope until decided

- Python-first API (item 22)
- GPU policy beyond “don’t move devices” (item 49)
- In-place glyphs (item 50)
