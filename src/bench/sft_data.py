"""Planner SFT examples from TRAIN manifest `train_ids` only (never the held-out validation IDs).

Each gold call becomes one example: the exact inference prompt for that step -> the gold JSON action.
Tool results come from replaying gold calls through one official ToolEnv per sample, as at inference.
"""

import json
from pathlib import Path

from ..agent.local_model import build_messages
from ..agent.planner import Observation, PlannerInput
from ..inference.official import OfficialKit
from ..inference.paths import safe_path
from ..inference.pipeline import _sample_from_row
from .evaluate import VALIDATION

_ASK = "sys.ask_missing_info"


def _compact(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sample_examples(kit: OfficialKit, row: dict, gold_calls: list[dict]):
    sample = _sample_from_row(kit, row, None, 0.0, control_noop=False)
    tools = kit.registry.available(sample.installed_apps, sample.allowed_tools)
    environment = kit.open_environment(sample)
    observations: list[Observation] = []
    for step, call in enumerate([*gold_calls, None]):
        target = ({"action": "finish", "reply": ""} if call is None
                  else {"action": "call", "tool": call["name"], "arguments": call["arguments"]})
        yield {"id": row["id"], "step": step, "completion": _compact(target),
               "messages": build_messages(PlannerInput(sample, tools, tuple(observations)))}
        if call is None:
            return
        payload = environment.call(call["name"], call["arguments"])
        observations.append(Observation("untrusted_tool_result", call["name"], payload, call["arguments"]))
        if call["name"] == _ASK and payload.get("waiting_for_user") is True:
            return  # the runner stops here, so there is no finish step to learn


def build(data_root, output, *, manifest=VALIDATION, limit=None) -> dict:
    split = json.loads(Path(manifest).read_text(encoding="utf-8"))
    train_ids = sorted(split["train_ids"])
    if set(train_ids) & set(split["validation_ids"]):
        raise ValueError("Train and validation IDs overlap")
    ids = set(train_ids[:limit] if limit else train_ids)
    labels = {}
    with safe_path(Path(data_root) / "train" / "train.jsonl").open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                item = json.loads(line)
                if item["id"] in ids:
                    labels[item["id"]] = item["label"]["tool_calls"]
    kit = OfficialKit(data_root, "train")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"samples": 0, "examples": 0, "skipped": []}
    with output.open("w", encoding="utf-8") as sink:
        for row in kit.rows(ids):
            try:
                examples = list(sample_examples(kit, row, labels[row["id"]]))
            except Exception as exc:  # e.g. official preflight rejects a gold call
                counts["skipped"].append(f'{row["id"]}: {type(exc).__name__}')
                continue
            sink.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in examples)
            counts["samples"] += 1
            counts["examples"] += len(examples)
    return counts
