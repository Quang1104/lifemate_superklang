"""Deterministic group-safe approximate stratification, for TRAIN only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata

from research.data_audit import call_name, calls_for, installed_apps, read_jsonl, safe_path


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def stratum(row: dict) -> str:
    names = [call_name(c) for c in calls_for(row)]
    active_apps = sorted({name.split(".")[0] for name in names if not name.startswith("sys.")})
    primary_app = active_apps[0] if active_apps else "none"
    return json.dumps([primary_app, "cross_app" if len(active_apps) > 1 else "one_app",
                       "empty" if not names else "single" if len(names) == 1 else "multi",
                       "missing" if "sys.ask_missing_info" in names else "normal"], ensure_ascii=False)


def make_split(rows: list[dict], fraction: float = 0.2, seed: int = 20261112) -> dict:
    if not 0 < fraction < 1:
        raise ValueError("fraction must be strictly between 0 and 1")
    if len(rows) < 2:
        raise ValueError("At least two TRAIN samples are required")
    if any("id" not in row or row["id"] is None for row in rows):
        raise ValueError("Every row must have an ID")
    rows = sorted(rows, key=lambda r: str(r["id"]))
    ids = [str(row["id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate IDs are ambiguous")
    parents = list(range(len(rows)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, row in enumerate(rows):
        audio = row.get("audio")
        text = row.get("audio_text")
        keys = []
        if isinstance(audio, str) and audio.strip():
            # Windows paths are case-insensitive; normalize slash aliases.
            keys.append(("audio", audio.replace("\\", "/").casefold()))
        if isinstance(text, str) and normalize(text):
            keys.append(("transcript", normalize(text)))
        for key in keys:
            if key in seen:
                parents[find(index)] = find(seen[key])
            else:
                seen[key] = index
    groups = defaultdict(list)
    for index in range(len(rows)):
        groups[find(index)].append(index)
    if len(groups) < 2:
        raise ValueError("All samples belong to one duplicate group; cannot split without leakage")
    strata = [stratum(row) for row in rows]
    totals = Counter(strata)
    targets = {key: count * fraction for key, count in totals.items()}
    target_count = max(1, min(len(rows) - 1, round(len(rows) * fraction)))
    stable_groups = sorted(groups.values(), key=lambda group: hashlib.sha256(
        (str(seed) + ":" + "|".join(ids[i] for i in group)).encode()).hexdigest())
    selected: set[int] = set()
    selected_counts: Counter = Counter()

    def loss(counts: Counter, total: int) -> float:
        return sum((counts[key] - target) ** 2 / max(1, totals[key])
                   for key, target in targets.items()) + (total - target_count) ** 2 / len(rows)

    remaining = list(stable_groups)
    group_counts = {tuple(group): Counter(strata[i] for i in group) for group in remaining}
    while remaining:
        base_loss = loss(selected_counts, len(selected))
        candidates = []
        for order, group in enumerate(remaining):
            if len(selected) + len(group) == len(rows):
                continue
            additions = group_counts[tuple(group)]
            delta = sum(((selected_counts[key] + added - targets[key]) ** 2 -
                         (selected_counts[key] - targets[key]) ** 2) / max(1, totals[key])
                        for key, added in additions.items())
            delta += ((len(selected) + len(group) - target_count) ** 2 -
                      (len(selected) - target_count) ** 2) / len(rows)
            candidates.append((base_loss + delta, order, group, additions))
        if not candidates:
            break
        best_loss, order, group, additions = min(candidates, key=lambda item: (item[0], item[1]))
        if len(selected) >= target_count and best_loss >= base_loss:
            break
        selected.update(group)
        selected_counts.update(additions)
        remaining.pop(order)
    train_ids = [ids[i] for i in range(len(rows)) if i not in selected]
    validation_ids = [ids[i] for i in range(len(rows)) if i in selected]
    return {
        "source_split": "train", "seed": seed, "requested_fraction": fraction,
        "strategy": "duplicate audio/transcript union groups + deterministic greedy joint stratification v1 (primary action app, cross-app, call bucket, clarification)",
        "sample_count": len(rows), "group_count": len(groups),
        "train_ids": train_ids, "validation_ids": validation_ids,
        "train_count": len(train_ids), "validation_count": len(validation_ids),
        "strata": {key: {"total": totals[key], "validation": selected_counts[key],
                         "train": totals[key] - selected_counts[key]} for key in sorted(totals)},
        "uncovered_validation_strata": sorted(key for key in totals if not selected_counts[key]),
        "uncovered_training_strata": sorted(key for key in totals if selected_counts[key] == totals[key]),
        "warning": "Approximate stratification; no guarantee for small strata. Never select few-shot examples from validation IDs or duplicate groups.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--fraction", type=float, default=0.2)
    args = parser.parse_args()
    rows = read_jsonl(args.input, train_only=True)
    result = make_split(rows, args.fraction, args.seed)
    result["source_sha256"] = hashlib.sha256(safe_path(args.input).read_bytes()).hexdigest()
    target = safe_path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("sample_count", "group_count", "train_count", "validation_count")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
