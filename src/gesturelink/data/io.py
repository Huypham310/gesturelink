"""Read / write the raw landmark CSV files."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from pathlib import Path

from gesturelink.data.schema import CSV_COLUMNS, Sample, SchemaError, check_header


class SampleWriter:
    """Append samples to a CSV, writing the header only when the file is new or empty."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        need_header = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        if need_header:
            self._writer.writerow(CSV_COLUMNS)
        self.count = 0

    def write(self, sample: Sample) -> None:
        self._writer.writerow(sample.to_row())
        self._fh.flush()  # flush mỗi dòng để lỡ crash khi thu vẫn không mất data
        self.count += 1

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> SampleWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_samples(path: str | Path, samples: Iterable[Sample]) -> int:
    """Append many samples to ``path``; returns how many were written."""
    with SampleWriter(path) as writer:
        for s in samples:
            writer.write(s)
        return writer.count


def iter_samples(path: str | Path) -> Iterator[Sample]:
    """Yield validated samples from one CSV. Raises SchemaError with the line number on bad rows."""
    path = Path(path)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise SchemaError(f"{path.name}: empty file")
        check_header(list(reader.fieldnames))
        for line_no, row in enumerate(reader, start=2):  # line 1 is the header
            try:
                yield Sample.from_row(row)
            except SchemaError as exc:
                raise SchemaError(f"{path.name}:{line_no}: {exc}") from exc


def read_samples(path: str | Path) -> list[Sample]:
    return list(iter_samples(path))


def find_csv_files(path: str | Path) -> list[Path]:
    """A single CSV, or every ``*.csv`` below a directory (sorted so runs are reproducible)."""
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("*.csv"))
        if not files:
            raise FileNotFoundError(f"no .csv files under {path}")
        return files
    raise FileNotFoundError(f"input path not found: {path}")
