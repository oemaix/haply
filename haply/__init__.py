"""Package marker. Public glyphs are Hy names loaded from ``__init__.hy``.

A Python ``import *`` cannot bind glyph names such as ``+``: the name is
not an attribute of the compiled module. The Hy package body binds them
for ``(import haply [⍴ ×])``.
"""

import sys
from pathlib import Path

import hy

_path = Path(__file__).with_suffix(".hy")
hy.eval(
    hy.read_many(_path.read_text(encoding="utf-8")),
    module=sys.modules[__name__],
)
