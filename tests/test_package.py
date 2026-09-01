"""Smoke test: the package is importable and the toolchain is wired up correctly.

This exists mainly to prove that the editable install and the registered Jupyter kernel
point at the same environment - the most likely source of early confusion.
"""

import dsa


def test_package_exposes_version() -> None:
    assert dsa.__version__ == "0.1.0"
