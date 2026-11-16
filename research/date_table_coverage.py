"""TRAIN-only oracle coverage of dates rendered by the native planner.

Counts scalar gold argument occurrences, including nested mappings/lists. This
reads only the allowlisted TRAIN JSONL and the local TRAIN split manifest; it
never constructs ToolEnv or accesses environment files. Only gold call arguments
and context.current_time are analyzed.
Run from the repository root with python -B -m research.date_table_coverage.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from datetime import date, datetime
import json
from pathlib import Path
import re

from src.agent.native_planner import render_date_table


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_JSONL = _REPO_ROOT.parent / "datalife" / "train+public_test" / "train" / "train.jsonl"
_VALIDATION_SPLIT = _REPO_ROOT / "research" / "validation_split.json"
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_DATETIME_PREFIX = re.compile(r"\d{4}-\d{2}-\d{2}[Tt ]\d{2}")
_TABLE_DATE = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")


def _scalars(value: object) -> Iterator[object]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _scalars(child)
    elif isinstance(value, list):
        for child in value:
            yield from _scalars(child)
    else:
        yield value


def _iso_date(value: object) -> tuple[str, str] | None:
    """Return a validated calendar date and kind for a whole ISO scalar value."""
    if not isinstance(value, str):
        return None
    try:
        if _DATE.fullmatch(value):
            return date.fromisoformat(value).isoformat(), "date"
        if _DATETIME_PREFIX.match(value):
            return datetime.fromisoformat(value).date().isoformat(), "datetime"
    except ValueError:
        pass
    return None


def measure(train_jsonl: Path) -> dict[str, int]:
    """Measure argument-date membership without exposing sample text or IDs."""
    if train_jsonl.resolve() != _TRAIN_JSONL.resolve():
        raise ValueError("Only the canonical allowlisted train/train.jsonl is permitted")
    manifest = json.loads(_VALIDATION_SPLIT.read_text(encoding="utf-8"))
    validation_ids = set(manifest["validation_ids"])
    expected_train_ids = set(manifest["train_ids"])
    if expected_train_ids & validation_ids:
        raise ValueError("TRAIN and validation IDs overlap in the split manifest")

    counts = dict.fromkeys(
        ("rows_read", "validation_samples_excluded", "train_samples_selected",
         "invalid_or_missing_anchors", "gold_calls", "iso_date_arguments",
         "covered_iso_date_arguments", "iso_datetime_arguments",
         "covered_iso_datetime_arguments", "samples_with_iso_arguments",
         "samples_with_all_iso_dates_covered"), 0)
    seen_ids: set[str] = set()
    selected_ids: set[str] = set()
    with train_jsonl.open(encoding="utf-8") as stream:
        for line in stream:
            sample = json.loads(line)
            sample_id = sample["id"]
            if sample_id in seen_ids:
                raise ValueError("Duplicate sample ID in TRAIN JSONL")
            seen_ids.add(sample_id)
            counts["rows_read"] += 1
            if sample_id in validation_ids:
                counts["validation_samples_excluded"] += 1
                continue
            if sample_id not in expected_train_ids:
                raise ValueError("TRAIN sample ID absent from the split manifest")
            selected_ids.add(sample_id)
            counts["train_samples_selected"] += 1

            table = render_date_table((sample.get("context") or {}).get("current_time"))
            dates = set(_TABLE_DATE.findall(table))
            counts["invalid_or_missing_anchors"] += not bool(table)
            calls = sample["label"]["tool_calls"]
            if not isinstance(calls, list):
                raise ValueError("Gold tool_calls must be a list")
            sample_total = sample_covered = 0
            for call in calls:
                arguments = call["arguments"]
                if not isinstance(arguments, dict):
                    raise ValueError("Gold call arguments must be an object")
                counts["gold_calls"] += 1
                for scalar in _scalars(arguments):
                    parsed = _iso_date(scalar)
                    if parsed is None:
                        continue
                    day, kind = parsed
                    covered = day in dates
                    counts[f"iso_{kind}_arguments"] += 1
                    counts[f"covered_iso_{kind}_arguments"] += covered
                    sample_total += 1
                    sample_covered += covered
            if sample_total:
                counts["samples_with_iso_arguments"] += 1
                counts["samples_with_all_iso_dates_covered"] += sample_total == sample_covered

    if selected_ids != expected_train_ids:
        raise ValueError("Not every manifest TRAIN ID was found in the JSONL")
    if not validation_ids <= seen_ids:
        raise ValueError("Not every manifest validation ID was found in the JSONL")
    return counts


def _coverage(covered: int, total: int) -> str:
    return f"{covered}/{total} ({100.0 * covered / total:.2f}%)" if total else "0/0 (N/A)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=_TRAIN_JSONL,
                        help="Canonical allowlisted train/train.jsonl only")
    args = parser.parse_args()
    counts = measure(args.train)
    print("TRAIN date-table oracle coverage (argument occurrences; no model score)")
    for key in ("rows_read", "validation_samples_excluded", "train_samples_selected",
                "invalid_or_missing_anchors", "gold_calls", "samples_with_iso_arguments"):
        print(f"{key}: {counts[key]}")
    for kind in ("date", "datetime"):
        print(f"covered_iso_{kind}_arguments: " + _coverage(
            counts[f"covered_iso_{kind}_arguments"], counts[f"iso_{kind}_arguments"]))
    total = counts["iso_date_arguments"] + counts["iso_datetime_arguments"]
    covered = counts["covered_iso_date_arguments"] + counts["covered_iso_datetime_arguments"]
    print("covered_all_iso_arguments: " + _coverage(covered, total))
    print("samples_with_all_iso_dates_covered: " + _coverage(
        counts["samples_with_all_iso_dates_covered"], counts["samples_with_iso_arguments"]))


if __name__ == "__main__":
    main()
