#!/usr/bin/env python3
"""Collect a time series of processed Bruker ASCII spectra.

Expected input layout::

    input_folder/
        10/
            pdata/1/ascii-spec.txt
            pdata/1/auditp.txt
        11/
            pdata/1/ascii-spec.txt
            pdata/1/auditp.txt

The immediate child folders do not have to be numeric.  Measurements are
ordered by the ``started at`` timestamp in ``auditp`` rather than by folder
name.  The first measurement is written as ``ascii-spec_0.txt`` and later
measurements use their rounded start-to-start delay in minutes.

For normal IDE use, edit ``CONFIG`` near the top of this file and press Run.
Terminal paths can optionally override the configured paths::

    python Brucker_batch_extractor.py INPUT_FOLDER OUTPUT_FOLDER

"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


# ===========================================================================
# CONFIG -- edit these values, then press Run in your IDE
# ===========================================================================
CONFIG = {
    # Folder containing measurement subfolders such as 10, 11, 12, ...
    "input_folder": r"D:\Frederik_Data\EGDA_amb15_70C_0,4M_run1",

    # Folder in which ascii-spec_0.txt, ascii-spec_4.txt, ... will be saved.
    # It will be created automatically if it does not exist. Point the rest of
    # the pipeline (plot_nmr.py / analyze_real_nmr.py / main.py) at this folder.
    "output_folder": r"C:\Users\vt4ho\Simulations\NMR\EGDA\NMR-EGDA_data",

    # Text written after "Sample id:" in every output file.
    # Use None to automatically use the input folder's name.
    "sample_id": None,

    # False protects existing output files; True replaces them on a rerun.
    "overwrite": False,
}


AUDIT_TIMESTAMP_RE = re.compile(
    r"\b(?P<event>started|completed)\s+at\s+"
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?\s+(?:[+-]\d{4}|[+-]\d{2}:\d{2}|Z))",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class Measurement:
    """Files and acquisition times belonging to one measurement folder."""

    folder: Path
    spectrum_path: Path
    audit_path: Path
    started_at: datetime
    completed_at: datetime | None


def parse_audit_timestamp(timestamp: str) -> datetime:
    """Parse a Bruker audit timestamp, including its UTC offset."""
    value = " ".join(timestamp.split())
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    elif re.search(r"[+-]\d{4}$", value):
        # Convert +0000 to the ISO form +00:00.
        value = value[:-5] + value[-5:-2] + ":" + value[-2:]
    # Python 3.8's datetime parser requires the UTC offset to touch the time;
    # Bruker writes a space before it ("...22.415 +0000").
    value = re.sub(r"\s+([+-]\d{2}:\d{2})$", r"\1", value)
    return datetime.fromisoformat(value)


def read_audit_times(audit_path: Path) -> tuple[datetime, datetime | None]:
    """Return the start time and, when present, completion time from auditp."""
    text = audit_path.read_text(encoding="utf-8", errors="replace")
    events: dict[str, datetime] = {}
    for match in AUDIT_TIMESTAMP_RE.finditer(text):
        event = match.group("event").lower()
        # The first occurrence describes this processing run if an audit file
        # happens to contain repeated history entries.
        events.setdefault(event, parse_audit_timestamp(match.group("timestamp")))

    if "started" not in events:
        raise ValueError("no 'started at' timestamp was found")

    started = events["started"]
    completed = events.get("completed")
    if completed is not None and completed < started:
        raise ValueError("the 'completed at' timestamp is earlier than 'started at'")
    return started, completed


def discover_measurements(input_folder: Path) -> tuple[list[Measurement], list[str]]:
    """Find valid ``<child>/pdata/1`` measurements under *input_folder*.

    Child folders with no complete spectrum/audit pair are skipped and
    returned as warnings.  A malformed audit file is also skipped so one bad
    measurement does not prevent extraction of the remaining batch.
    """
    measurements: list[Measurement] = []
    warnings: list[str] = []

    def find_file(folder: Path, names: tuple[str, ...]) -> Path | None:
        """Return the first existing filename from the supported variants."""
        return next((folder / name for name in names if (folder / name).is_file()), None)

    for child in sorted(
        (path for path in input_folder.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    ):
        processed = child / "pdata" / "1"

        if not processed.is_dir():
            warnings.append(f"Skipping {child.name!r}: pdata/1 was not found.")
            continue

        # TopSpin commonly exports these as .txt, although some versions and
        # workflows use the same names without an extension.
        spectrum_path = find_file(processed, ("ascii-spec.txt", "ascii-spec"))
        audit_path = find_file(processed, ("auditp.txt", "auditp"))
        missing = []
        if spectrum_path is None:
            missing.append("ascii-spec.txt/ascii-spec")
        if audit_path is None:
            missing.append("auditp.txt/auditp")
        if missing:
            warnings.append(
                f"Skipping {child.name!r}: missing {', '.join(missing)} in pdata/1."
            )
            continue

        try:
            started_at, completed_at = read_audit_times(audit_path)
        except (OSError, UnicodeError, ValueError) as exc:
            warnings.append(f"Skipping {child.name!r}: cannot read auditp ({exc}).")
            continue

        measurements.append(
            Measurement(
                folder=child,
                spectrum_path=spectrum_path,
                audit_path=audit_path,
                started_at=started_at,
                completed_at=completed_at,
            )
        )

    measurements.sort(key=lambda item: (item.started_at, item.folder.name.casefold()))
    return measurements, warnings


def rounded_elapsed_minutes(started_at: datetime, time_zero: datetime) -> int:
    """Return a non-negative elapsed time rounded to the nearest minute."""
    elapsed_minutes = (started_at - time_zero).total_seconds() / 60.0
    # All values are non-negative.  Adding 0.5 gives conventional half-up
    # rounding instead of Python's round-to-even behaviour.
    return int(elapsed_minutes + 0.5)


def add_sample_header(spectrum_text: str, sample_id: str) -> str:
    """Add the metadata line used by ``test_with_name.txt``.

    Existing ``Sample id:`` lines are replaced, which also makes processing an
    already-labelled input idempotent.
    """
    text = spectrum_text.lstrip("\ufeff")
    lines = text.splitlines()
    # TopSpin's ascii-spec.txt starts with a blank line, while the labelled
    # reference format starts immediately with "Sample id:".
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].strip().lower().startswith("sample id"):
        lines = lines[1:]
    body = "\n".join(lines)
    if body and not body.endswith("\n"):
        body += "\n"
    return f"Sample id: {sample_id}\n{body}"


def extract_batch(
    input_folder: str | Path,
    output_folder: str | Path,
    *,
    sample_id: str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Extract all valid spectra and return their output paths.

    ``sample_id`` defaults to the name of the input folder.  Existing output
    files are protected unless ``overwrite=True`` is supplied.
    """
    source = Path(input_folder).expanduser().resolve()
    destination = Path(output_folder).expanduser().resolve()

    if not source.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {source}")

    measurements, warnings = discover_measurements(source)
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    if not measurements:
        raise ValueError(
            f"No valid measurements were found below {source}. Expected "
            "<measurement>/pdata/1/ascii-spec.txt and auditp.txt "
            "(extensionless names are also supported)."
        )

    effective_sample_id = sample_id if sample_id is not None else source.name
    if not effective_sample_id.strip():
        raise ValueError("Sample ID cannot be empty.")

    time_zero = measurements[0].started_at
    output_plan: list[tuple[Measurement, int, Path]] = []
    paths_by_minute: dict[int, list[str]] = {}
    for measurement in measurements:
        minute = rounded_elapsed_minutes(measurement.started_at, time_zero)
        output_path = destination / f"ascii-spec_{minute}.txt"
        output_plan.append((measurement, minute, output_path))
        paths_by_minute.setdefault(minute, []).append(measurement.folder.name)

    collisions = {
        minute: names for minute, names in paths_by_minute.items() if len(names) > 1
    }
    if collisions:
        details = "; ".join(
            f"minute {minute}: {', '.join(repr(name) for name in names)}"
            for minute, names in sorted(collisions.items())
        )
        raise ValueError(
            "Multiple measurements round to the same output minute; refusing "
            f"to overwrite one spectrum with another ({details})."
        )

    existing = [path for _, _, path in output_plan if path.exists()]
    if existing and not overwrite:
        shown = ", ".join(path.name for path in existing[:5])
        if len(existing) > 5:
            shown += f", and {len(existing) - 5} more"
        raise FileExistsError(
            f"Output file(s) already exist ({shown}). Use --overwrite to replace them."
        )

    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for measurement, minute, output_path in output_plan:
        spectrum_text = measurement.spectrum_path.read_text(
            encoding="utf-8", errors="replace"
        )
        with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
            output_file.write(add_sample_header(spectrum_text, effective_sample_id))
        written.append(output_path)
        print(
            f"{measurement.folder.name}: {measurement.started_at.isoformat()} "
            f"-> t={minute} min -> {output_path.name}"
        )

    return written


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_folder",
        nargs="?",
        help="Folder containing the measurement subfolders.",
    )
    parser.add_argument(
        "output_folder",
        nargs="?",
        help="Folder in which the labelled spectra will be saved.",
    )
    parser.add_argument(
        "--sample-id",
        help="Value for the 'Sample id:' line (default: input folder name).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing ascii-spec_<minute>.txt files to be replaced.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    # IDE runs normally have no positional arguments, so CONFIG is used.
    # Explicit terminal arguments still take precedence when supplied.
    input_folder = args.input_folder or CONFIG["input_folder"]
    output_folder = args.output_folder or CONFIG["output_folder"]
    sample_id = args.sample_id if args.sample_id is not None else CONFIG["sample_id"]
    overwrite = args.overwrite or CONFIG["overwrite"]

    if not input_folder or not output_folder:
        parser.error("set input_folder and output_folder in CONFIG")

    try:
        written = extract_batch(
            input_folder,
            output_folder,
            sample_id=sample_id,
            overwrite=overwrite,
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    print(f"Done: wrote {len(written)} spectrum file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
