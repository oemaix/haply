# Known issues

Notes from projects that use Haply. The maintainer reads this file and
follows up.

Each entry has:

- **Kind.** Missing feature or glyph, or a bug against the manual.
- **Symptom.**
- **Example.** A minimal call.
- **Blocked.** Where the caller was stuck.
- **Local edit.** Only for a bug. The smallest change already made in
  this repository. Omit when nothing was edited.

Open design questions stay in [undecided.md](undecided.md).

---

## Package import dies on `+`

- **Kind.** Bug against the manual (public import `(import haply [⍴ × …])`).
- **Symptom.** `import haply` raises `AttributeError: module 'haply.scalar' has no attribute '+'`.
- **Example.** `python -c "import haply"`.
- **Blocked.** `stateful-ssm-engine` StateBank (WP-2.3) could not import a glyph.
- **Local edit.** `haply/__init__.py` no longer star-imports the glyph modules. It runs `__init__.hy`, which binds the names Hy can import.
