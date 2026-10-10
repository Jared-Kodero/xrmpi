"""Small helpers shared by the MPI-aware I/O routines."""

from __future__ import annotations

import fcntl
import sys
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path


@contextmanager
def _locked(lockfile: Path | str | None) -> Iterator[None]:
    """Hold an exclusive advisory lock on ``lockfile`` if one is given."""
    if lockfile is None:
        yield
        return
    with open(lockfile, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class SerialProgressBar:
    """Wrap an iterable and report its progress on a text stream.

    Parameters
    ----------
    iterable : iterable
        Items to yield. Its length is used for the total when available.
    description : str, optional
        Text printed before the counter.
    file : file-like, optional
        Destination stream. Defaults to ``sys.stdout``.
    lockfile : path-like, optional
        Advisory lock taken around each write so concurrent processes do not
        interleave output.
    tmpdir : path-like, optional
        Accepted for interface compatibility; unused.
    min_interval : float, default 0.2
        Minimum seconds between redraws. The final state is always drawn.
    """

    def __init__(
        self,
        iterable: Iterable[Any],
        *,
        description: str = "",
        file: TextIO | None = None,
        lockfile: Path | str | None = None,
        tmpdir: Path | str | None = None,
        min_interval: float = 0.2,
    ) -> None:
        self._iterable = iterable
        self._description = description
        self._file = file if file is not None else sys.stdout
        self._lockfile = lockfile
        self._tmpdir = tmpdir
        self._min_interval = min_interval
        try:
            self._total: int | None = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            self._total = None

    def _draw(self, count: int, *, final: bool) -> None:
        total = "?" if self._total is None else str(self._total)
        text = f"\r{self._description}: {count}/{total}"
        with _locked(self._lockfile):
            self._file.write(text + ("\n" if final else ""))
            self._file.flush()

    def __iter__(self) -> Iterator[Any]:
        count = 0
        last = 0.0
        for item in self._iterable:
            yield item
            count += 1
            now = time.monotonic()
            if now - last >= self._min_interval:
                self._draw(count, final=False)
                last = now
        self._draw(count, final=True)

    def __len__(self) -> int:
        if self._total is None:
            raise TypeError("length is unknown")
        return self._total
