"""Reading a local tabular file into a DataFrame.

Kaggle hands back a directory, not a table, so fetching and reading are separate
concerns: :mod:`dsa.io.kaggle` gets bytes onto disk, and this module turns whatever
landed there into a frame.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Callable

import pandas as pd

# Compression suffixes pandas handles transparently. Stripped before dispatch so that
# "train.csv.gz" is recognised as a CSV rather than an unknown ".gz".
_COMPRESSION = {".gz", ".bz2", ".zip", ".xz", ".zst"}

_READERS: dict[str, Callable[..., pd.DataFrame]] = {
    ".csv": pd.read_csv,
    ".tsv": partial(pd.read_csv, sep="\t"),
    ".txt": pd.read_csv,
    ".parquet": pd.read_parquet,
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
    ".json": pd.read_json,
}

# Directories that archives commonly carry and that never contain the data of interest.
_SKIP_DIRS = {"__MACOSX", ".git", ".ipynb_checkpoints"}


def table_format(path: Path | str) -> str | None:
    """Return the reader suffix for ``path``, ignoring compression, or None."""
    suffixes = [s.lower() for s in Path(path).suffixes]
    while suffixes and suffixes[-1] in _COMPRESSION:
        suffixes.pop()
    return suffixes[-1] if suffixes and suffixes[-1] in _READERS else None


def find_tables(root: Path | str) -> list[Path]:
    """Every readable table under ``root``, sorted by name for a stable presentation."""
    root = Path(root)
    if root.is_file():
        return [root] if table_format(root) else []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and table_format(p)
        and not any(part in _SKIP_DIRS or part.startswith(".") for part in p.relative_to(root).parts[:-1])
    )


def read_table(path: Path | str, **kwargs: Any) -> pd.DataFrame:
    """Read one table, dispatching on its extension."""
    path = Path(path)
    fmt = table_format(path)
    if fmt is None:
        raise ValueError(
            f"don't know how to read {path.name!r}; supported: {', '.join(sorted(_READERS))}"
        )
    return _READERS[fmt](path, **kwargs)


def describe_candidates(paths: list[Path], root: Path) -> str:
    """Render a file list with sizes, so a user choosing between them has the context."""
    lines = []
    for p in paths:
        size_mb = p.stat().st_size / 1_048_576
        lines.append(f'    "{p.relative_to(root).as_posix()}"  ({size_mb:,.1f} MB)')
    return "\n".join(lines)
