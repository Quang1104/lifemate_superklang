"""Deterministic leave-app-out split of the TRAIN pool (never the 309 validation IDs).

Holds out whole apps: a sample goes to the holdout when its gold calls or its history calls name a held-out
app (including sys.ask_missing_info targets), so the remaining pool never shows those tools. This estimates
generalisation to the public test's unseen tools. TRAIN only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from research.data_audit import call_name, calls_for, read_jsonl, safe_path


def apps_used(row: dict) -> set[str]:
    calls = list(calls_for(row))
    for turn in row.get("history", []):
        calls.extend(turn.get("tool_calls") or [])
    apps = set()
    for call in calls:
        name = call_name(call)
        if name == "sys.ask_missing_info":
            name = str(call.get("arguments", {}).get("tool_name", ""))
        if "." in name and not name.startswith("sys."):
            apps.add(name.partition(".")[0])
    return apps


def make_split(rows: list[dict], excluded: set[str], tools_per_app: dict[str, int],
               fraction: float = 0.15, seed: int = 20261113, min_samples: int = 5,
               max_samples: int = 40) -> dict:
    if not 0 < fraction < 1:
        raise ValueError("fraction must be strictly between 0 and 1")
    pool = sorted((row for row in rows if row["id"] not in excluded), key=lambda row: row["id"])
    if len({row["id"] for row in pool}) != len(pool):
        raise ValueError("Duplicate IDs are ambiguous")
    touching = defaultdict(set)
    for row in pool:
        for app in apps_used(row):
            touching[app].add(row["id"])
    budget = round(len(pool) * fraction)
    order = sorted(touching, key=lambda app: hashlib.sha256(f"{seed}:{app}".encode()).hexdigest())
    held_apps, held_ids = [], set()
    for app in order:
        # Small apps first-come by hash: many unseen tools, like the public test, not one dominant app.
        if (min_samples <= len(touching[app]) <= max_samples
                and len(held_ids | touching[app]) <= budget):
            held_apps.append(app)
            held_ids |= touching[app]
    train_ids = [row["id"] for row in pool if row["id"] not in held_ids]
    holdout_ids = [row["id"] for row in pool if row["id"] in held_ids]
    assert not any(apps_used(row) & set(held_apps) for row in pool if row["id"] in set(train_ids))
    return {
        "source_split": "train", "seed": seed, "requested_fraction": fraction,
        "strategy": "leave-app-out v1: hash-ordered apps with 5..40 pool samples each, greedily held out "
                    "while the holdout stays within the budget; any gold or history call naming a held-out app "
                    "puts the sample in the holdout",
        "excluded_validation_ids": len(excluded), "pool_count": len(pool),
        "holdout_apps": sorted(held_apps),
        "holdout_tool_count": sum(tools_per_app.get(app, 0) for app in held_apps),
        "holdout_app_samples": {app: len(touching[app]) for app in sorted(held_apps)},
        "train_ids": train_ids, "validation_ids": holdout_ids,
        "train_count": len(train_ids), "validation_count": len(holdout_ids),
        "warning": "Held-out apps are still listed in installed_apps context; only their calls are unseen. "
                   "validation_ids here means the leave-app-out holdout, not the 309-ID validation split.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--validation", type=Path, default=Path("research/validation_split.json"))
    parser.add_argument("--output", type=Path, default=Path("research/leave_app_out_split.json"))
    parser.add_argument("--fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    rows = read_jsonl(args.input, train_only=True)
    excluded = set(json.loads(safe_path(args.validation).read_text(encoding="utf-8"))["validation_ids"])
    catalog = json.loads(safe_path(args.catalog).read_text(encoding="utf-8"))
    tools_per_app = {app["id"]: len(app["tools"]) for app in catalog["apps"]}
    result = make_split(rows, excluded, tools_per_app, args.fraction, args.seed)
    result["source_sha256"] = hashlib.sha256(safe_path(args.input).read_bytes()).hexdigest()
    target = safe_path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("pool_count", "train_count", "validation_count",
                                                     "holdout_apps", "holdout_tool_count")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
