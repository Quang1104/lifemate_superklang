"""Build the ID-clean TRAIN manifest and audit held-out tool exposure.

Only explicit TRAIN metadata/catalog paths are opened. No ToolEnv, runtime state,
audio, model, or test split is used. This measures names in installed tool lists
and all history strings, plus direct current/history calls; it does not score a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "datalife" / "train+public_test"


def checked_path(value: str | Path) -> Path:
    candidate = Path(value)
    for path in (candidate, candidate.resolve()):
        if any(part.lower().endswith("_environment") or part.lower() == "state.json"
               for part in path.parts):
            raise ValueError("Forbidden path")
    return candidate


def read_json(path: Path) -> dict:
    value = json.loads(checked_path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def manifest_ids(manifest: dict, key: str = "validation_ids") -> set[str]:
    if manifest.get("source_split") != "train":
        raise ValueError("Only TRAIN manifests are accepted")
    ids = manifest.get(key)
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise ValueError("Expected a list of TRAIN IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate manifest IDs")
    return set(ids)


def strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def history_call_names(history: list[dict]) -> set[str]:
    return {call["name"] for turn in history for call in turn.get("tool_calls", [])
            if isinstance(call, dict) and isinstance(call.get("name"), str)}


def build_manifest(rows: list[dict], catalog: dict, validation: dict, leave_app: dict,
                   source_sha256: str) -> dict:
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate TRAIN IDs")
    all_ids = set(ids)
    validation_ids, holdout_ids = manifest_ids(validation), manifest_ids(leave_app)
    if validation_ids & holdout_ids:
        raise ValueError("Validation and leave-app holdout must be disjoint")
    excluded = validation_ids | holdout_ids
    if not excluded <= all_ids:
        raise ValueError("Excluded IDs are absent from TRAIN")
    train_ids = sorted(all_ids - excluded)
    apps = leave_app.get("holdout_apps")
    if not isinstance(apps, list) or not all(isinstance(app, str) for app in apps):
        raise ValueError("Expected holdout app names")
    app_set = set(apps)
    heldout_tools = sorted({tool["name"] for app in catalog["apps"] if app["id"] in app_set
                            for tool in app["tools"]})
    if not heldout_tools:
        raise ValueError("No holdout tools found in TRAIN catalog")
    patterns = {tool: re.compile(r"(?<![\w.])" + re.escape(tool) + r"(?![\w.])")
                for tool in heldout_tools}
    counts = {key: Counter() for key in
              ("installed_tool_lists", "history_anywhere", "history_calls", "current_gold_calls")}
    exposed_samples = set()
    retained = set(train_ids)
    for row in rows:
        if row["id"] not in retained:
            continue
        installed = {tool for spec in row["context"]["installed_apps"].values()
                     for tool in spec["tools"]}
        history = row.get("history", [])
        history_text = list(strings(history))
        history_calls = history_call_names(history)
        gold_calls = {call["name"] for call in row["label"]["tool_calls"]}
        for tool in heldout_tools:
            present = {
                "installed_tool_lists": tool in installed,
                "history_anywhere": any(patterns[tool].search(text) for text in history_text),
                "history_calls": tool in history_calls,
                "current_gold_calls": tool in gold_calls,
            }
            for key, found in present.items():
                counts[key][tool] += int(found)
            if present["installed_tool_lists"] or present["history_anywhere"]:
                exposed_samples.add(row["id"])
    details = {tool: {key + "_samples": counts[key][tool] for key in counts}
               for tool in heldout_tools}
    exposed = [tool for tool in heldout_tools
               if counts["installed_tool_lists"][tool] or counts["history_anywhere"][tool]]
    return {
        "source_split": "train",
        "strategy": "TRAIN minus validation IDs minus leave-app-out holdout IDs; no further filtering",
        "source_sha256": source_sha256,
        "sample_count": len(ids),
        "train_count": len(train_ids),
        "validation_count": len(excluded),
        "original_validation_count": len(validation_ids),
        "leave_app_holdout_count": len(holdout_ids),
        "train_ids": train_ids,
        "validation_ids": sorted(excluded),
        "original_validation_ids": sorted(validation_ids),
        "leave_app_holdout_ids": sorted(holdout_ids),
        "holdout_apps": sorted(app_set),
        "holdout_tools": heldout_tools,
        "holdout_tool_count": len(heldout_tools),
        "exposure_audit": {
            "unit": "distinct retained TRAIN conversations containing each tool name",
            "exposed_tool_count": len(exposed),
            "exposed_tools": exposed,
            "exposed_conversation_count": len(exposed_samples),
            "installed_tool_list_exposed_tool_count": sum(bool(counts["installed_tool_lists"][t]) for t in heldout_tools),
            "history_anywhere_exposed_tool_count": sum(bool(counts["history_anywhere"][t]) for t in heldout_tools),
            "history_call_exposed_tool_count": sum(bool(counts["history_calls"][t]) for t in heldout_tools),
            "current_gold_call_exposed_tool_count": sum(bool(counts["current_gold_calls"][t]) for t in heldout_tools),
            "per_tool": details,
        },
        "warning": "ID-clean and call-unseen only: held-out tool names and schemas remain in installed tool lists. validation_ids is the union of both exclusions for build_sft.py compatibility. No clean unseen-tool model result has been measured.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--validation", type=Path, default=Path("research/validation_split.json"))
    parser.add_argument("--leave-app", type=Path, default=Path("research/leave_app_out_split.json"))
    parser.add_argument("--output", type=Path, default=Path("research/unseen_clean_split.json"))
    args = parser.parse_args()
    root = checked_path(args.data_root)
    source = checked_path(root / "train" / "train.jsonl").read_bytes()
    rows = [json.loads(line) for line in source.decode("utf-8-sig").splitlines() if line.strip()]
    catalog = read_json(root / "train" / "train_tools_catalog.json")
    result = build_manifest(rows, catalog, read_json(args.validation), read_json(args.leave_app),
                            hashlib.sha256(source).hexdigest())
    checked_path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = result["exposure_audit"]
    print(json.dumps({"TRAIN": result["sample_count"], "validation": result["original_validation_count"],
                      "holdout": result["leave_app_holdout_count"], "clean_train": result["train_count"],
                      "excluded_union": result["validation_count"],
                      "holdout_tools": result["holdout_tool_count"], **audit,
                      "output": str(args.output), "model_score": "NOT MEASURED"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
