"""Step 1: loading a dataset.

Fetching from Kaggle and reading a table are separate steps, because Kaggle returns a
directory that may hold several files and choosing between them is the user's call.

Credentials are never read into a variable that could be logged: this module only ever
asks *which* mechanism is present, and leaves the value to kagglehub.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from dsa.gates import DECISION, open_gate
from dsa.io.readers import describe_candidates, find_tables, read_table
from dsa.session import Session

STEP = 1


class CredentialsMissing(RuntimeError):
    """Raised when no Kaggle credential mechanism is configured."""


def credential_source() -> str | None:
    """Name the credential mechanism kagglehub will use, without reading its value.

    Mirrors kagglehub's own precedence order. Returns a description for display, or None
    if nothing is configured.
    """
    if os.getenv("KAGGLE_API_TOKEN"):
        return "KAGGLE_API_TOKEN environment variable"
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return "KAGGLE_USERNAME / KAGGLE_KEY environment variables"

    config_dir = Path(os.getenv("KAGGLE_CONFIG_DIR") or (Path.home() / ".kaggle"))
    credentials_file = config_dir / "kaggle.json"
    if credentials_file.exists():
        return f"credentials file at {credentials_file}"
    return None


def require_credentials() -> str:
    """Return the credential mechanism in use, or explain how to configure one.

    The message is the entire user experience when this fails, so it names the three
    mechanisms and the one that suits a modern ``KGAT_`` token.
    """
    source = credential_source()
    if source:
        return source
    raise CredentialsMissing(
        "No Kaggle credentials found. kagglehub accepts, in order of precedence:\n"
        "  1. KAGGLE_API_TOKEN        - a single modern token (starts with 'KGAT_')\n"
        "  2. KAGGLE_USERNAME + KAGGLE_KEY  - the classic pair\n"
        "  3. ~/.kaggle/kaggle.json   - the classic pair as a file\n"
        "\n"
        "For a modern token, set it once as a user environment variable, keeping the\n"
        "value out of your shell history:\n"
        '  $t = Read-Host "token"; '
        "[Environment]::SetEnvironmentVariable('KAGGLE_API_TOKEN', $t, 'User')\n"
        "\n"
        "Then restart the kernel so it inherits the new environment."
    )


def fetch(slug: str) -> Path:
    """Download a Kaggle dataset and return the local directory holding it.

    kagglehub caches, so re-running this after answering a gate costs nothing.
    """
    import kagglehub  # imported lazily: loading the toolkit should not require network code

    return Path(kagglehub.dataset_download(slug))


def load_kaggle(session: Session, slug: str, file: str | None = None) -> Session:
    """Fetch a Kaggle dataset into the session (step 1).

    If the dataset contains more than one table and ``file`` was not given, a decision
    gate is opened and :class:`~dsa.gates.GateRequired` is raised. Answer it and re-run;
    the download is cached, so the second call is immediate.
    """
    source = require_credentials()

    with session.log.record(STEP, "load.fetch", {"slug": slug, "auth_source": source}) as rec:
        directory = fetch(slug)
        rec.notes = f"cached at {directory}"

    candidates = find_tables(directory)
    if not candidates:
        raise FileNotFoundError(f"no readable table found in {directory}")

    chosen = _choose_file(session, directory, candidates, file)
    return _ingest(session, chosen, source=f"kaggle:{slug}", root=directory)


def load_file(session: Session, path: Path | str) -> Session:
    """Load a local table directly.

    Kaggle is the v1 data source, but a Kaggle download *is* a local file by the time it
    is read, so this is the same code path exposed rather than a second one.
    """
    path = Path(path)
    return _ingest(session, path, source=f"file:{path.name}", root=path.parent)


def _choose_file(
    session: Session, directory: Path, candidates: list[Path], file: str | None
) -> Path:
    """Resolve which table to read, opening a gate when the choice is genuinely ambiguous."""
    if file is not None:
        match = directory / file
        if not match.exists():
            available = ", ".join(p.relative_to(directory).as_posix() for p in candidates)
            raise FileNotFoundError(f"{file!r} is not in the dataset; available: {available}")
        return match

    if len(candidates) == 1:
        return candidates[0]

    # More than one table: the user picks. Sizes are shown because the choice is usually
    # between a full table and a small sample or submission template.
    from dsa.gates import require

    relative = tuple(p.relative_to(directory).as_posix() for p in candidates)
    open_gate(
        session,
        key="source_file",
        kind=DECISION,
        question=f"This dataset contains {len(candidates)} tables. Which one should be loaded?",
        step=STEP,
        options=relative,
        context=describe_candidates(candidates, directory),
    )
    return directory / require(session, "source_file")


def _ingest(session: Session, path: Path, source: str, root: Path) -> Session:
    """Read a table into the session and open the target-column gate.

    The target gate is opened but not required: you need to look at the data before you
    can sensibly name a target, so step 1 completes and the gate waits.
    """
    with session.log.record(STEP, "load.read", {"file": path.name, "source": source}) as rec:
        frame = read_table(path)
        rec.output_shape = frame.shape
        rec.artifacts = [str(path)]
        rec.notes = f"{frame.shape[0]:,} rows x {frame.shape[1]} columns"

    session.raw = frame
    session.source = source
    session.repairs = []  # a new dataset invalidates any repairs approved for the old one
    session.rebuild()

    open_gate(
        session,
        key="target",
        kind=DECISION,
        question="Which column is the target?",
        step=STEP,
        options=tuple(frame.columns),
        context=_column_preview(frame),
    )
    return session


def _column_preview(frame: pd.DataFrame, limit: int = 25) -> str:
    """Compact column listing to help name a target without leaving the traceback."""
    lines = []
    for name in list(frame.columns)[:limit]:
        column = frame[name]
        lines.append(f"    {name:<28} {str(column.dtype):<12} {column.nunique(dropna=True):>6} unique")
    if frame.shape[1] > limit:
        lines.append(f"    ... and {frame.shape[1] - limit} more columns")
    return "\n".join(lines)
