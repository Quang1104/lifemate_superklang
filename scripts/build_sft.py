"""Build QLoRA training conversations from TRAIN gold calls in the native planner's chat format.

Each sample is rendered with the same build_messages/tool_definitions used at inference: the
gold calls are replayed through one official ToolEnv, each becomes an assistant tool call
followed by its real tool result, and a short closing reply ends the turn. The 309 validation
IDs (and any other --exclude manifest) never enter the training file.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import monotonic

from src.agent.native_planner import build_messages, tool_definitions
from src.agent.planner import Observation, PlannerInput
from src.inference.official import OfficialKit
from src.inference.pipeline import _sample_from_row
from src.tooling.registry import ToolRejected

DONE = "Đã xong."
NOTHING = "Mình không thực hiện thao tác nào cho yêu cầu này."


def conversation(sample, tools: list[dict], calls: list[dict], payloads: list[dict]) -> dict:
    observations = tuple(Observation("untrusted_tool_result", call["name"], payload, call["arguments"])
                         for call, payload in zip(calls, payloads))
    messages = build_messages(PlannerInput(sample, tools, observations, None))
    target_start = max(i for i, message in enumerate(build_messages(PlannerInput(sample, tools, (), None)))
                       if message["role"] == "user") + 1
    messages.append({"role": "assistant", "content": DONE if calls else NOTHING})
    return {"id": sample.sample_id, "messages": messages, "tools": tool_definitions(tools),
            "target_start": target_start}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/sft_native.jsonl"))
    parser.add_argument("--exclude", type=Path, nargs="+", default=[Path("research/validation_split.json")])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if any(part.endswith("_environment") for part in args.data_root.resolve().parts):
        raise ValueError("Forbidden root")
    excluded = set()
    for manifest in args.exclude:
        split = json.loads(manifest.read_text(encoding="utf-8"))
        if split.get("source_split") != "train":
            raise ValueError(f"Expected a TRAIN manifest: {manifest}")
        excluded |= set(split["validation_ids"])
    kit = OfficialKit(args.data_root, "train")
    gold = {}
    with (kit.directory / "train.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] not in excluded:
                gold[row["id"]] = row["label"]["tool_calls"]
    selected = set(sorted(gold)[:args.limit] if args.limit else gold)
    totals, lengths = Counter(), []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as sink:
        for row in kit.rows(selected):
            sample = _sample_from_row(kit, row, None, monotonic() + 3_600, control_noop=False)
            tools = kit.registry.available(sample.installed_apps, sample.allowed_tools)
            environment = kit.open_environment(sample)  # the one gold-replay ToolEnv for this sample
            calls, payloads = gold[row["id"]], []
            try:
                for call in calls:
                    payloads.append(environment.call(call["name"], call["arguments"]))
            except ToolRejected:
                totals["skipped_preflight_rejected"] += 1
                continue
            item = conversation(sample, tools, calls, payloads)
            sink.write(json.dumps(item, ensure_ascii=False) + "\n")
            totals["written"] += 1
            totals["assistant_calls"] += len(calls)
            totals["empty_gold"] += not calls
            lengths.append(len(json.dumps(item, ensure_ascii=False)))
    lengths.sort()
    pick = lambda q: lengths[min(len(lengths) - 1, int(q * len(lengths)))] if lengths else None
    print(json.dumps({**totals, "excluded_ids": len(excluded), "chars_p50": pick(0.5), "chars_p95": pick(0.95),
                      "chars_max": lengths[-1] if lengths else None, "tokens": "NOT MEASURED (needs tokenizer)",
                      "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
