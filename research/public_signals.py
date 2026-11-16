"""Label-free signals of a public submission (docs/experiments/PUBLIC_GAP.md), from public inputs and our own predictions only.

    python -B -m research.public_signals <submission.jsonl> [--compare other.jsonl] [--first N] [--data-root <kit>]

No public label exists or is inferred: every number is a count over the calls we produced, grouped by what the
public inputs show (installed apps, history) and by whether a tool exists in the TRAIN catalog. `--first N` keeps
the first N rows of the main file (a partially re-run submission); `--compare` prints a second column on the same IDs.
"""

import argparse
import json
from pathlib import Path

# TRAIN apps a request for a public-only app could be pulled toward (app ids only; not tuned on any output).
SIBLINGS = {"grocery": {"shopping_list", "food_delivery"}, "ecommerce": {"shopping_list", "food_delivery"},
            "pantry": {"shopping_list", "notes"}, "event_tickets": {"calendar"},
            "train_bus_tickets": {"flights", "ride_hailing"}, "public_transit": {"maps", "ride_hailing"},
            "parking": {"maps"}, "recipes": {"notes", "food_delivery"},
            "habit_tracker": {"fitness", "tasks", "reminders"}, "stocks": {"banking", "gold_price"},
            "translator": {"notes"}}
ASK = "sys.ask_missing_info"


def _app(name: str) -> str:
    return name.split(".")[0]


def load(data_root: Path) -> tuple[dict, dict, set]:
    """Public inputs by ID, tool kind by name, and the tools that exist only in the public catalog."""
    def tools(split):
        catalog = json.loads((data_root / split / f"{split}_tools_catalog.json").read_text(encoding="utf-8"))
        return {tool["name"]: tool["kind"] for app in catalog["apps"] for tool in app["tools"]}

    train, public = tools("train"), tools("public_test")
    with (data_root / "public_test" / "public_test.jsonl").open(encoding="utf-8") as lines:
        inputs = {row["id"]: row for row in map(json.loads, filter(str.strip, lines))}
    return inputs, {**train, **public}, set(public) - set(train)


def read(path: Path) -> dict[str, list]:
    with path.open(encoding="utf-8") as lines:
        return {row["id"]: row["tool_calls"] for row in map(json.loads, filter(str.strip, lines))}


def signals(inputs: dict, kind: dict, unseen: set, calls_by_id: dict, ids: list[str]) -> dict[str, str]:
    """Counts as "n/total (share)" strings, keyed by signal name, over `ids`."""
    new_apps = {_app(name) for name in unseen}
    ratio = lambda part, whole: f"{part}/{whole} ({part / whole:.1%})" if whole else "0/0"
    asked = {"public-only": [0, 0], "train": [0, 0]}
    ends_read = {"public-only app, history did not use it": [0, 0], "history used a public-only tool": [0, 0],
                 "no public-only app": [0, 0]}
    empty = total_calls = calls_new = uncalled = sibling = with_new = 0
    for sample_id in ids:
        row, calls = inputs[sample_id], calls_by_id[sample_id]
        installed = set(row["context"]["installed_apps"])
        history = {call["name"] for turn in row["history"] if turn.get("role") == "assistant"
                   for call in turn.get("tool_calls") or ()}
        called = {_app(call["name"]) for call in calls} - {"sys"}
        asked_about = {_app(str(call["arguments"].get("tool_name", ""))) for call in calls if call["name"] == ASK}
        empty += not calls
        total_calls += len(calls)
        calls_new += sum(call["name"] in unseen for call in calls)
        with_new += any(call["name"] in unseen for call in calls)
        for app in (called | asked_about) & installed:
            group = asked["public-only" if app in new_apps else "train"]
            group[0] += app in asked_about
            group[1] += 1
        group = ends_read["no public-only app" if not installed & new_apps
                          else "history used a public-only tool" if history & unseen
                          else "public-only app, history did not use it"]
        group[0] += bool(calls) and kind.get(calls[-1]["name"]) == "read"
        group[1] += 1
        for app in (installed & new_apps) - called:
            uncalled += 1
            sibling += bool(SIBLINGS.get(app, set()) & installed & called)
    count = len(ids)
    return {
        "samples": str(count),
        "empty outputs": ratio(empty, count),
        "calls per sample": f"{total_calls / count:.2f}" if count else "0",
        "samples calling a public-only tool": ratio(with_new, count),
        "calls to public-only tools": ratio(calls_new, total_calls),
        "ask_missing_info, public-only apps acted on or asked about": ratio(*asked["public-only"]),
        "ask_missing_info, TRAIN apps acted on or asked about": ratio(*asked["train"]),
        **{f"last call is a read: {name}": ratio(*value) for name, value in ends_read.items()},
        "installed public-only app not called, a TRAIN sibling app called instead": ratio(sibling, uncalled),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--first", type=int, help="only the first N rows of the submission")
    parser.add_argument("--data-root", type=Path, default=Path("../datalife/train+public_test"))
    args = parser.parse_args()
    if any(part.endswith("_environment") for part in args.data_root.resolve().parts):
        raise ValueError("Forbidden root")
    inputs, kind, unseen = load(args.data_root)
    files = [args.submission] + ([args.compare] if args.compare else [])
    runs = [read(path) for path in files]
    ids = list(runs[0])[:args.first]
    if any(sample_id not in inputs or any(sample_id not in run for run in runs) for sample_id in ids):
        raise ValueError("Every compared ID must be a public ID present in all files")
    columns = [signals(inputs, kind, unseen, run, ids) for run in runs]
    print("| Signal | " + " | ".join(path.parent.name + "/" + path.name for path in files) + " |")
    print("|---|" + "---|" * len(files))
    for name in columns[0]:
        print(f"| {name} | " + " | ".join(column[name] for column in columns) + " |")


if __name__ == "__main__":
    main()
