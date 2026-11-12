"""Allowlisted data audit. Never visits environment directories.

Run: python -m research.data_audit --output docs/DATASET_ANALYSIS.md
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct
from typing import Any

APPLEDOUBLE_MAGIC = b"\x00\x05\x16\x07"
SPLITS = ("train", "public_test")


def safe_path(path: Path) -> Path:
    """Reject prohibited names before resolution/stat/open, and resolved aliases."""
    for candidate in (path,):
        if any(p.lower().endswith("_environment") or p.lower() == "state.json"
               for p in candidate.parts):
            raise ValueError("Prohibited environment/state path")
    resolved = path.resolve()
    if any(p.lower().endswith("_environment") or p.lower() == "state.json" for p in resolved.parts):
        raise ValueError("Prohibited environment/state path")
    return path


def metadata_info(path: Path) -> dict[str, Any]:
    path = safe_path(path)
    if not path.is_file():
        return {"present": False}
    with path.open("rb") as handle:
        header = handle.read(26)
    is_appledouble = header[:4] == APPLEDOUBLE_MAGIC
    return {"present": True, "bytes": path.stat().st_size,
            "appledouble": is_appledouble,
            "entry_count": struct.unpack(">H", header[24:26])[0]
            if is_appledouble and len(header) == 26 else None}


def file_status(path: Path) -> dict[str, Any]:
    path = safe_path(path)
    return {"path": path.as_posix(), "present": path.is_file(),
            "metadata": metadata_info(path.with_name("._" + path.name))}


def read_jsonl(path: Path, *, train_only: bool = False) -> list[dict[str, Any]]:
    path = safe_path(path)
    if train_only and path.name != "train.jsonl":
        raise ValueError("TRAIN analysis requires an explicitly named train.jsonl")
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path.name}:{line_no}: expected object")
                rows.append(row)
    return rows


def calls_for(row: dict[str, Any]) -> list[dict[str, Any]]:
    label = row.get("label")
    if not isinstance(label, dict) or not isinstance(label.get("tool_calls"), list):
        raise ValueError("TRAIN row lacks label.tool_calls; schema requires review")
    calls = label["tool_calls"]
    if not all(isinstance(c, dict) for c in calls):
        raise ValueError("Invalid call object")
    return calls


def call_name(call: dict[str, Any]) -> str:
    name = call.get("name", call.get("tool", call.get("tool_name")))
    if name is None and isinstance(call.get("function"), dict):
        name = call["function"].get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Unknown tool-call name schema")
    return name


def installed_apps(row: dict[str, Any]) -> list[str]:
    apps = row.get("context", {}).get("installed_apps", [])
    if isinstance(apps, dict):
        return sorted(apps)
    if isinstance(apps, list):
        names = [a if isinstance(a, str) else a.get("name", a.get("app_id", a.get("app")))
                 for a in apps if isinstance(a, (str, dict))]
        return sorted(a for a in names if isinstance(a, str))
    return []


def extract_tools(catalog: Any) -> dict[str, dict[str, Any]]:
    """Recognize explicit tool definitions, preserving their schemas verbatim.

    Unknown formats are an error. This is analysis only, not an inference parser.
    """
    result: dict[str, dict[str, Any]] = {}

    def visit(node: Any, app: str | None = None, key: str | None = None) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item, app)
            return
        if not isinstance(node, dict):
            return
        declared_app = node.get("app", node.get("app_id", app))
        if not isinstance(declared_app, str):
            declared_app = app
        name = node.get("name", node.get("tool_name", key))
        schema = node.get("parameters", node.get("input_schema", node.get("args_schema")))
        if isinstance(name, str) and isinstance(schema, dict):
            full = name if "." in name or not declared_app else f"{declared_app}.{name}"
            if full in result and result[full] != node:
                raise ValueError(f"Ambiguous duplicate tool: {full}")
            result[full] = node
            return
        for child_key, child in node.items():
            if child_key in ("parameters", "input_schema", "args_schema"):
                continue
            next_app = declared_app
            if child_key == "tools" and isinstance(node.get("name"), str):
                next_app = node["name"]
            if child_key not in ("apps", "tools", "functions", "metadata") and isinstance(child, dict) and "tools" in child:
                next_app = child_key
            visit(child, next_app, child_key)

    visit(catalog)
    if not result:
        raise ValueError("No recognizable tool definitions; inspect official catalog schema")
    return result


def extensions(node: Any, prefix: str = "") -> list[dict[str, Any]]:
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            location = f"{prefix}/{key}"
            if key in ("x-ref", "x-one-of", "x-match"):
                found.append({"path": location, "extension": key, "value": value})
            found.extend(extensions(value, location))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(extensions(value, f"{prefix}/{index}"))
    return found


def operation_class(name: str, tool: dict[str, Any]) -> tuple[str, str]:
    for key in ("operation_type", "tool_type", "kind", "type", "x-operation"):
        if tool.get(key) in ("read", "write", "meta"):
            return tool[key], "declared"
    verb = name.rsplit(".", 1)[-1].split("_", 1)[0]
    if name.startswith("sys."):
        return "meta", "name_heuristic"
    if verb in {"get", "list", "search", "find", "lookup", "check", "query", "read"}:
        return "read", "name_heuristic"
    if verb in {"create", "add", "delete", "remove", "update", "set", "send", "book", "cancel", "pay", "transfer", "turn"}:
        return "write", "name_heuristic"
    return "unknown", "unknown"


TEMPORAL = ("hôm nay", "ngày mai", "ngày kia", "hôm qua", "tối nay", "sáng mai",
            "chiều mai", "tuần sau", "cuối tuần", "tháng sau", "nửa tiếng", "phút nữa", "giờ nữa")
SENSITIVE = {
    "contacts": ("contact", "recipient", "beneficiary", "phone", "email"),
    "amounts": ("amount", "price", "money", "cost"),
    "addresses": ("address", "destination", "origin", "pickup", "location"),
    "times": ("date", "time", "start", "end", "duration"),
    "device_names": ("device", "room"),
}


def analyze_rows(rows: list[dict[str, Any]], *, split: str, split_dir: Path) -> dict[str, Any]:
    ids = [str(row.get("id")) for row in rows]
    apps: Counter[str] = Counter()
    names: Counter[str] = Counter()
    lengths: Counter[int] = Counter()
    temporal: Counter[str] = Counter()
    multi = cross = empty = missing = histories = 0
    audio_exists = audio_missing = audio_rejected = 0
    for row in rows:
        apps.update(installed_apps(row))
        histories += bool(row.get("history"))
        audio = row.get("audio")
        if isinstance(audio, str):
            rel = Path(audio)
            try:
                candidate = safe_path(split_dir / rel)
                resolved_audio_root = safe_path(split_dir / "audio").resolve()
                if rel.is_absolute() or not candidate.resolve().is_relative_to(resolved_audio_root):
                    raise ValueError("Audio outside split/audio")
                if candidate.is_file():
                    audio_exists += 1
                else:
                    audio_missing += 1
            except ValueError:
                audio_rejected += 1
        else:
            audio_missing += 1
        if split != "train":
            continue  # Do not inspect public-test labels/transcripts even if present.
        calls = calls_for(row)
        row_names = [call_name(c) for c in calls]
        names.update(row_names)
        lengths[len(calls)] += 1
        multi += len(calls) > 1
        cross += len({name.split(".")[0] for name in row_names if not name.startswith("sys.")}) > 1
        empty += len(calls) == 0
        missing += "sys.ask_missing_info" in row_names
        transcript = str(row.get("audio_text", "")).casefold()
        temporal.update(phrase for phrase in TEMPORAL if phrase in transcript)
    return {
        "sample_count": len(rows), "unique_id_count": len(set(ids)),
        "missing_id_count": sum("id" not in row for row in rows),
        "installed_app_sample_counts": dict(sorted(apps.items())),
        "samples_with_history": histories,
        "referenced_audio": {"present": audio_exists, "missing": audio_missing, "rejected_path": audio_rejected},
        "train_labels": {"action_counts": dict(sorted(names.items())),
                         "call_count_distribution": dict(sorted(lengths.items())),
                         "multi_step_samples": multi, "cross_app_samples": cross,
                         "gold_empty_samples": empty, "missing_info_samples": missing,
                         "temporal_expression_sample_counts": dict(sorted(temporal.items()))}
        if split == "train" else "NOT INSPECTED (public test)",
    }


def audit(root: Path) -> dict[str, Any]:
    root = safe_path(root)
    report: dict[str, Any] = {"root": root.as_posix(), "status": "READY", "splits": {}, "errors": []}
    catalogs: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        split_dir = root / split
        jsonl = safe_path(split_dir / f"{split}.jsonl")
        catalog = safe_path(split_dir / f"{split}_tools_catalog.json")
        entry: dict[str, Any] = {"jsonl": file_status(jsonl), "catalog": file_status(catalog)}
        if jsonl.is_file():
            try:
                entry["statistics"] = analyze_rows(read_jsonl(jsonl), split=split, split_dir=split_dir)
                entry["jsonl_sha256"] = hashlib.sha256(jsonl.read_bytes()).hexdigest()
            except (ValueError, OSError) as exc:
                report["errors"].append(f"{split} JSONL: {exc}")
        else:
            entry["statistics"] = "NOT MEASURED: original JSONL missing"
            report["status"] = "BLOCKED_MISSING_DATA"
        if catalog.is_file():
            try:
                tools = extract_tools(json.loads(catalog.read_text(encoding="utf-8-sig")))
                catalogs[split] = tools
                ext = {name: extensions(tool) for name, tool in tools.items()}
                sensitive = {category: sorted(name for name, tool in tools.items()
                             if any(token in json.dumps(tool.get("parameters", tool.get("input_schema", tool.get("args_schema", {})))).lower()
                                    for token in tokens)) for category, tokens in SENSITIVE.items()}
                entry["tool_statistics"] = {
                    "tool_count": len(tools), "names": sorted(tools),
                    "operation_class_counts": dict(Counter("/".join(operation_class(n, t)) for n, t in tools.items())),
                    "extensions": {name: value for name, value in ext.items() if value},
                    "asr_sensitive_schema_heuristics": sensitive,
                }
                if split == "train" and isinstance(entry.get("statistics"), dict):
                    action_counts = entry["statistics"]["train_labels"]["action_counts"]
                    kinds: Counter[str] = Counter()
                    for name, count in action_counts.items():
                        kinds["/".join(operation_class(name, tools.get(name, {})))] += count
                    entry["tool_statistics"]["labeled_action_operation_counts"] = dict(sorted(kinds.items()))
            except (ValueError, OSError) as exc:
                report["errors"].append(f"{split} catalog: {exc}")
        else:
            entry["tool_statistics"] = "NOT MEASURED: original catalog missing"
            report["status"] = "BLOCKED_MISSING_DATA"
        report["splits"][split] = entry
    if set(catalogs) == set(SPLITS):
        train, public = catalogs["train"], catalogs["public_test"]
        report["generalization"] = {
            "public_only_tools": sorted(public.keys() - train.keys()),
            "public_only_apps": sorted({n.split(".")[0] for n in public} - {n.split(".")[0] for n in train}),
            "shared_tools_changed_definition": sorted(n for n in train.keys() & public.keys() if train[n] != public[n]),
        }
    else:
        report["generalization"] = "NOT MEASURED: both catalogs required"
    if report["errors"]:
        report["status"] = "BLOCKED_SCHEMA_OR_DATA_ERROR"
    return report


def render(reports: list[dict[str, Any]]) -> str:
    text = "# Dataset analysis\n\nGenerated from explicit permitted paths by `research.data_audit`; no environment traversal/state reads.\n\n"
    for report in reports:
        text += f"## `{report['root']}`\n\nStatus: **{report['status']}**.\n\n"
        if report["status"] != "READY":
            text += "Original JSONL/catalog data are absent or unsupported. Counts are **NOT MEASURED**. AppleDouble `._*` metadata files do not substitute for originals.\n\n"
            continue
        text += "| Metric | TRAIN | PUBLIC_TEST |\n|---|---:|---:|\n"
        for label, getter in (
            ("Samples", lambda e: e["statistics"]["sample_count"]),
            ("Unique IDs", lambda e: e["statistics"]["unique_id_count"]),
            ("Installed apps (union)", lambda e: len(e["statistics"]["installed_app_sample_counts"])),
            ("Catalog tools", lambda e: e["tool_statistics"]["tool_count"]),
            ("Samples with history", lambda e: e["statistics"]["samples_with_history"]),
            ("Referenced audio present", lambda e: e["statistics"]["referenced_audio"]["present"]),
            ("Referenced audio missing", lambda e: e["statistics"]["referenced_audio"]["missing"]),
        ):
            text += f"| {label} | {getter(report['splits']['train'])} | {getter(report['splits']['public_test'])} |\n"
        tr = report["splits"]["train"]["statistics"]
        labels = tr["train_labels"]
        text += "\n### TRAIN task complexity\n\n"
        for key in ("multi_step_samples", "cross_app_samples", "gold_empty_samples", "missing_info_samples"):
            text += f"- {key}: **{labels[key]} / {tr['sample_count']} ({100 * labels[key] / tr['sample_count']:.2f}%)**.\n"
        text += f"- Calls per sample: `{json.dumps(labels['call_count_distribution'])}`.\n"
        text += f"- Labeled calls: **{sum(labels['action_counts'].values())}**; distinct labeled tools: **{len(labels['action_counts'])}**.\n\n"
        text += "Most frequent labeled actions: " + "; ".join(f"`{n}` {c}" for n, c in sorted(labels["action_counts"].items(), key=lambda x: (-x[1], x[0]))[:15]) + ".\n\n"
        text += "### Catalog constraints\n\n"
        for split in SPLITS:
            ts = report["splits"][split]["tool_statistics"]
            text += f"- {split}: operation types `{json.dumps(ts['operation_class_counts'])}`.\n"
            if "labeled_action_operation_counts" in ts:
                text += f"- TRAIN labeled action types: `{json.dumps(ts['labeled_action_operation_counts'])}`.\n"
            for extension in ("x-ref", "x-one-of", "x-match"):
                hits = {name: [x for x in exts if x["extension"] == extension] for name, exts in ts["extensions"].items()}
                hits = {name: values for name, values in hits.items() if values}
                text += f"  - `{extension}`: {len(hits)} tools, {sum(len(x) for x in hits.values())} occurrences.\n"
        gen = report["generalization"]
        text += "\n### Generalization\n\n"
        text += f"Public-only: **{len(gen['public_only_tools'])} tools / {len(gen['public_only_apps'])} apps**. "
        text += ", ".join(f"`{x}`" for x in gen["public_only_apps"]) + ".\n\n"
        text += f"Shared tools with changed definitions: **{len(gen['shared_tools_changed_definition'])}**. Dynamic schema routing is required; memorizing TRAIN names cannot cover public-test tools.\n\n"
        text += "### Vietnamese time and ASR risk\n\n"
        text += "; ".join(f"{phrase}: {count}" for phrase, count in sorted(labels["temporal_expression_sample_counts"].items(), key=lambda x: -x[1])) + ". Counts mean samples containing each phrase; phrases may overlap.\n\n"
        text += "Schema heuristics identify contacts, amounts, addresses/locations, times, and device names as sensitive fields. See detailed JSON for tool lists. English code-switching and entity-level ASR errors have **not** been measured. No speech quality is inferred from file presence.\n\n"
    text += """## Method, validation split, and limitations

- Complete app/action distributions, schema extension paths/values, sensitive-field heuristics, and unseen tool names are in `research/data_audit.json`. No sample transcripts or state are copied there.
- Read/write/meta classification uses declared catalog `kind` where available. Fuzzy constraints preserve `x-match`; `x-ref` and `x-one-of` semantics require the official SDK.
- Multi-step means more than one labeled call; cross-app excludes `sys`. Gold-empty is a label category, not an independent semantic judgment.
- Public test uses only IDs, context/app metadata, audio paths, and catalog definitions. Public labels and transcripts are never analyzed.
- `research/validation_split.json`: fixed seed 20261112, 20% target; duplicate audio paths and normalized identical transcripts grouped transitively. Strata use primary action app, cross-app status, call-count bucket, clarification. Manifest records exact coverage gaps; rare groups cannot guarantee coverage. Renamed/paraphrased duplicates are not detected.
- `research/smoke_split.json`: 40 samples chosen from held-out IDs by the same stratification routine. Never use held-out IDs/groups as in-context examples or training rows.
- `research.metrics`: diagnostic exact-call F1 and WER/CER. `research.published_scorer`: published state-gated fuzzy-call metric with maximum one-to-one matching; reference implementation parity is unverified, especially tokenization. Neither synthetic tests nor the audit measure planner accuracy.

## Highest-value follow-up

1. Run all 40 held-out samples through the official ToolEnv; evaluate with separately replayed TRAIN gold hashes.
2. Compare one local planner on the exact same oracle transcripts and ASR predictions, with checkpoint/config and latency recorded.
3. Target real errors in entity references, temporal conversion, or clarification; test generalization on held-out app/tool groups before training.

```powershell
python -m research.data_audit --root '<actual-kit-root>' --output docs/DATASET_ANALYSIS.md --json-output research/data_audit.json
python -m research.validation_split --input '<actual-kit-root>/train/train.jsonl' --output research/validation_split.json --seed 20261112 --fraction 0.2
python -m pytest tests_audit -q
```
"""
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=Path("docs/DATASET_ANALYSIS.md"))
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    reports = [audit(root) for root in (args.root or [Path("."), Path("train+public_test")])]
    output = safe_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(reports), encoding="utf-8")
    if args.json_output:
        target = safe_path(args.json_output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"roots": [{"root": r["root"], "status": r["status"]} for r in reports], "report": str(output)}))
    return 0 if all(r["status"] == "READY" for r in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
