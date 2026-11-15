"""CPU measurements for the LIFEMATE_DUP_GUARD / LIFEMATE_CONTEXT_V4 / LIFEMATE_SMALL_FIXES flags.

TRAIN oracle analysis only: reads train.jsonl (gold calls and gold tool results), the tools catalog and the
split manifests; never a ToolEnv. The 309 validation IDs are left out of every TRAIN count.

    python -B -m scripts.orca_v4_measure dup --data-root <kit>
    python -B -m scripts.orca_v4_measure prompt --data-root <kit>
    python -B -m scripts.orca_v4_measure small --data-root <kit> [--transcripts outputs/asr_train.jsonl]
"""

import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
from statistics import median
from time import monotonic

from src.agent.runner import _record_lists, same_name_ask
from src.tooling.lifemate_catalog import build_registry


def _train(data_root: Path, manifest: str = "research/validation_split.json", *, keep_validation: bool = False):
    ids = set(json.loads(Path(manifest).read_text(encoding="utf-8"))["validation_ids"])
    with (data_root / "train" / "train.jsonl").open(encoding="utf-8") as lines:
        return [row for row in map(json.loads, lines) if (row["id"] in ids) == keep_validation]


def _schemas(data_root: Path) -> dict[str, dict]:
    catalog = json.loads((data_root / "train" / "train_tools_catalog.json").read_text(encoding="utf-8"))
    apps = frozenset(app["id"] for app in catalog["apps"])
    return {tool["name"]: tool["parameters"] for tool in build_registry(catalog).available(apps)}


def dup(data_root: Path) -> None:
    """How many TRAIN gold calls the same-name guard would rewrite, and how many gold asks it reproduces."""
    schemas, total = _schemas(data_root), Counter()
    for row in _train(data_root):
        calls, results = row["label"]["tool_calls"], row["label"]["tool_results"]
        for index, call in enumerate(calls):
            name, arguments, seen = call["name"], call["arguments"], results[:index]
            if name != "sys.ask_missing_info":
                properties = schemas[name].get("properties", {})
                uses_id = any("x-ref" in properties.get(key, {}) and isinstance(value, str)
                              for key, value in arguments.items())
                total["gold calls that use an ID"] += uses_id
                total["  rewritten by the guard"] += same_name_ask(name, arguments, schemas[name], seen) is not None
                continue
            target, params = arguments.get("tool_name"), arguments.get("params", [])
            properties = schemas.get(target, {}).get("properties", {})
            if not any("x-ref" in properties.get(key, {}) for key in params):
                total["gold asks for a value (guard never applies)"] += 1
                continue
            total["gold asks for an ID"] += 1
            # The model's mistake the guard corrects: calling the tool with any ID of a same-name group.
            ids = {record["id"] for result in seen for records in _record_lists(result) for record in records}
            total["  reproduced exactly from a direct call"] += any(
                same_name_ask(target, {params[0]: value}, schemas[target], seen) == arguments for value in ids)
    for key, value in total.items():
        print(f"{key}: {value}")


def prompt(data_root: Path) -> None:
    """Characters LIFEMATE_CONTEXT_V4 adds to the system prompt and tool definitions on smoke40 (K=2)."""
    from src.agent import fewshot
    from src.agent.native_planner import build_messages, tool_definitions
    from src.agent.planner import PlannerInput
    from src.inference.official import OfficialKit
    from src.inference.pipeline import _sample_from_row

    kit = OfficialKit(data_root, "train")
    smoke = set(json.loads(Path("research/smoke_split.json").read_text(encoding="utf-8"))["validation_ids"])
    os.environ["LIFEMATE_NATIVE_FEWSHOT_K"] = "2"
    sizes = {}
    for flag in ("0", "1"):
        os.environ["LIFEMATE_CONTEXT_V4"] = flag
        fewshot.configured.cache_clear()
        for row in kit.rows(smoke):
            sample = _sample_from_row(kit, row, None, monotonic() + 60, control_noop=False)
            tools = kit.registry.available(sample.installed_apps, sample.allowed_tools)
            system = build_messages(PlannerInput(sample, tools, (), None))[0]["content"]
            definitions = json.dumps(tool_definitions(tools), ensure_ascii=False)
            sizes.setdefault(row["id"], []).append((len(system), len(definitions), system))
    def spread(values):
        ordered = sorted(values)
        return f"median {median(ordered):.0f}, p95 {ordered[int(0.95 * len(ordered))]}, max {ordered[-1]}"
    print(f"smoke40 samples: {len(sizes)}")
    print("system prompt chars, flag off:", spread(off[0] for off, _ in sizes.values()))
    print("system prompt chars, flag on: ", spread(on[0] for _, on in sizes.values()))
    print("system prompt chars added:    ", spread(on[0] - off[0] for off, on in sizes.values()))
    print("tool definition chars added:  ", spread(on[1] - off[1] for off, on in sizes.values()))
    print("total chars added:            ", spread(on[0] + on[1] - off[0] - off[1] for off, on in sizes.values()))
    print("--- flag-on system prompt tail of", min(sizes), "---")
    print(sizes[min(sizes)][1][2][-1500:])


def small(data_root: Path, transcripts: Path | None) -> None:
    """TRAIN grounding of the two small fixes: update_event end times and spelled-out e-mail addresses."""
    from src.agent.asr_fix import spoken_email

    total = Counter()
    rows = _train(data_root)
    for row in rows:
        calls, results = row["label"]["tool_calls"], row["label"]["tool_results"]
        events = {}
        for turn in row["history"]:
            for result in turn.get("tool_results") or ():
                events.update({item["id"]: item for records in _record_lists(result) for item in records})
        for index, call in enumerate(calls):
            for records in _record_lists(results[index - 1] if index else {}):
                events.update({item["id"]: item for item in records})
            arguments = call["arguments"]
            if call["name"] != "calendar.update_event" or "start" not in arguments:
                continue
            total["gold update_event calls that move the start"] += 1
            total["  with an end"] += "end" in arguments
            old = events.get(arguments.get("event_id"), {})
            if "end" in arguments and old.get("start") and old.get("end"):
                total["  old event visible in earlier results"] += 1
                moved = datetime.fromisoformat(arguments["end"]) - datetime.fromisoformat(arguments["start"])
                kept = datetime.fromisoformat(old["end"]) - datetime.fromisoformat(old["start"])
                total["  end = new start + old duration"] += moved == kept
    heard = {}
    if transcripts is not None and transcripts.is_file():
        with transcripts.open(encoding="utf-8") as lines:
            heard = {item["audio"]: item["transcript"] for item in map(json.loads, filter(str.strip, lines))}
    for row in rows:
        emails = {value for call in row["label"]["tool_calls"] for item in call["arguments"].values()
                  for value in (item if isinstance(item, list) else [item]) if isinstance(value, str) and "@" in value}
        fixed = spoken_email(row["audio_text"])
        total["gold requests changed by the e-mail rule"] += fixed != row["audio_text"]
        if emails:
            total["gold requests with an e-mail argument"] += 1
            total["  address spelled out in the gold transcript"] += "a còng" in row["audio_text"].lower()
            total["  rule output contains the gold address (gold transcript)"] += any(item in fixed for item in emails)
            if row["audio"] in heard:
                total["  rule output contains the gold address (ASR transcript)"] += any(
                    item in spoken_email(heard[row["audio"]]) for item in emails)
        elif fixed != row["audio_text"]:
            total["  changed although gold has no e-mail argument"] += 1
    for key, value in total.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=("dup", "prompt", "small"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--transcripts", type=Path, default=Path("outputs/asr_train.jsonl"))
    options = parser.parse_args()
    if any(part.endswith("_environment") for part in options.data_root.resolve().parts):
        raise ValueError("Forbidden root")
    if options.what == "dup":
        dup(options.data_root)
    elif options.what == "prompt":
        prompt(options.data_root)
    else:
        small(options.data_root, options.transcripts)
