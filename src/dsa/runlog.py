"""Append-only run log.

Every operation records what it did here before returning (CLAUDE.md rule 6), and the
step-7 write-up is rendered from this file rather than from recollection. That makes the
log the single source of truth about what actually happened during a session.

Entries are one JSON object per line, flushed and fsync'd on write, so a kernel crash or
a closed browser tab loses nothing.
"""

from __future__ import annotations

import json
import os
import platform
import re
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Mapping

# Parameter names whose values must never reach disk (rule 7). Matched case-insensitively
# as a substring. Deliberately does NOT include a bare "key", which would redact ordinary
# parameters like "key_columns" and make the log misleading in a different way.
_SECRET_HINTS = re.compile(r"token|secret|password|credential|api_key", re.IGNORECASE)
_REDACTED = "<redacted>"

Shape = tuple[int, int]


def _redact(params: Mapping[str, Any]) -> dict[str, Any]:
    """Replace values whose parameter *name* suggests a credential."""
    return {k: (_REDACTED if _SECRET_HINTS.search(k) else v) for k, v in params.items()}


def _utc_now() -> str:
    """Timestamps are UTC and ISO-8601 so logs from different sessions sort correctly."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class RunEntry:
    """One recorded operation.

    ``status`` is retained for failures as well as successes: a step-7 write-up that
    silently omits the things that went wrong would not be an honest account of the work.
    """

    ts: str
    step: int  # pipeline step 1-7; 0 for session-level events
    op: str  # dotted operation name, e.g. "load.kaggle" or "clean.approve"
    params: dict[str, Any] = field(default_factory=dict)
    input_shape: Shape | None = None
    output_shape: Shape | None = None
    artifacts: list[str] = field(default_factory=list)
    notes: str | None = None
    duration_s: float | None = None
    status: str = "ok"  # "ok" | "error"

    def to_json(self) -> str:
        # default=str lets NumPy scalars, Paths and similar serialise without every
        # caller having to pre-convert them.
        return json.dumps(asdict(self), default=str)


class _Recording:
    """Mutable handle yielded by :meth:`RunLog.record` so a caller can attach results
    that are only known once the operation has finished."""

    def __init__(self) -> None:
        self.output_shape: Shape | None = None
        self.artifacts: list[str] = []
        self.notes: str | None = None


class RunLog:
    """Append-only JSONL log backed by a file on disk."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[RunEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[RunEntry]:
        """Entries appended during this session, in order."""
        return list(self._entries)

    def append(self, entry: RunEntry) -> RunEntry:
        """Write one entry and force it to disk before returning.

        The file is opened and closed per entry rather than held open. At human
        interaction rates the cost is irrelevant, and it means the log survives a kernel
        restart, a moved directory, or a second Session writing to the same file.
        """
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(entry.to_json() + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._entries.append(entry)
        return entry

    @contextmanager
    def record(
        self,
        step: int,
        op: str,
        params: Mapping[str, Any] | None = None,
        input_shape: Shape | None = None,
    ) -> Iterator[_Recording]:
        """Time an operation and log it, whether it succeeds or raises.

        Usage::

            with s.log.record(2, "clean.apply", {"n_repairs": 3}, df.shape) as rec:
                out = do_work(df)
                rec.output_shape = out.shape
        """
        rec = _Recording()
        started = perf_counter()
        status, notes = "ok", None
        try:
            yield rec
        except Exception as exc:  # noqa: BLE001 - re-raised immediately below
            status = "error"
            notes = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.append(
                RunEntry(
                    ts=_utc_now(),
                    step=step,
                    op=op,
                    params=_redact(params or {}),
                    input_shape=input_shape,
                    output_shape=rec.output_shape,
                    artifacts=rec.artifacts,
                    # A failure message must not be silently overwritten by a note the
                    # caller set before the exception was raised.
                    notes=notes or rec.notes,
                    duration_s=round(perf_counter() - started, 4),
                    status=status,
                )
            )

    @staticmethod
    def read(path: str | os.PathLike[str]) -> list[RunEntry]:
        """Load a log written by a previous session, for rendering a write-up later."""
        entries: list[RunEntry] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(RunEntry(**json.loads(line)))
        return entries


def environment_snapshot(seed: int) -> dict[str, Any]:
    """Capture what would be needed to reproduce a run.

    Package versions are read from installed metadata rather than by importing each
    library, so building the snapshot never triggers a slow import or fails on a package
    that happens to be broken.
    """
    import importlib.metadata as md

    packages: dict[str, str] = {}
    for name in (
        "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn",
        "xgboost", "lightgbm", "kagglehub",
    ):
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:
            packages[name] = "not installed"

    return {
        "captured_at": _utc_now(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": seed,
        "packages": packages,
    }
