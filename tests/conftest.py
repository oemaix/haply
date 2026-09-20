"""Collect `test_*.hy` without rewriting Hy as Python AST."""

import hy
import pytest


def pytest_collect_file(file_path, parent):
    if file_path.suffix == ".hy" and file_path.name.startswith("test_"):
        return pytest.Module.from_parent(parent, path=file_path)
