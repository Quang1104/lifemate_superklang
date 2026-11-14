"""Build QLoRA training conversations from TRAIN gold calls in the native planner's chat format.

Each sample is rendered with the same build_messages/tool_definitions used at inference: the
gold calls are replayed through one official ToolEnv, each becomes an assistant tool call
followed by its real tool result, and a short closing reply ends the turn. The 309 validation
IDs (and any other --exclude manifest) never enter the training file.

--transcripts replaces the released TRAIN transcripts of the user turns with recorded ASR text
(scripts/asr_transcripts.py), so training sees what inference sees. --fewshot-k adds retrieved
TRAIN examples to every prompt through the same code and pool file inference uses.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import monotonic

from src.agent.fewshot import NATIVE_POOL, configured
from src.agent.native_planner import build_messages, tool_definitions
from src.agent.planner import Observation, PlannerInput
from src.agent.runner import CallbackError
from src.inference.official import OfficialKit, installed_permissions
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


def excluded_ids(manifests: list[Path]) -> set[str]:
    excluded = set()
    for manifest in manifests:
        split = json.loads(manifest.read_text(encoding="utf-8"))
        if split.get("source_split") != "train":
            raise ValueError(f"Expected a TRAIN manifest: {manifest}")
        excluded |= set(split["validation_ids"])
    return excluded


def transcript_lookup(path: Path):
    """Stand-in for ASR: the recorded transcript of each audio file (scripts/asr_transcripts.py output)."""
    with path.open(encoding="utf-8") as lines:
        known = {row["audio"]: row["transcript"] for row in map(json.loads, filter(str.strip, lines))}
    return lambda audio: known[Path(*audio.parts[-3:]).as_posix()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/sft_native.jsonl"))
    parser.add_argument("--exclude", type=Path, nargs="+", default=[Path("research/validation_split.json")])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--transcripts", type=Path,
                        help="JSONL from scripts/asr_transcripts.py; user turns use this ASR text")
    parser.add_argument("--fewshot-k", type=int, default=0,
                        help="retrieved TRAIN examples per prompt; rewrites the native example pool")
    args = parser.parse_args()
    if any(part.endswith("_environment") for part in args.data_root.resolve().parts):
        raise ValueError("Forbidden root")
    excluded = excluded_ids(args.exclude)
    kit = OfficialKit(args.data_root, "train")
    gold, pool = {}, []
    with (kit.directory / "train.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] not in excluded:
                gold[row["id"]] = row["label"]["tool_calls"]
                pool.append({"id": row["id"], "transcript": row["audio_text"],
                             "apps": sorted(installed_permissions(row["context"])[0]), "calls": gold[row["id"]]})
    if args.fewshot_k > 0:
        # Inference reads this same file behind LIFEMATE_NATIVE_FEWSHOT_K; it holds no excluded ID.
        with NATIVE_POOL.open("w", encoding="utf-8", newline='\n') as sink:
            sink.writelines(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + '\n' for row in pool)
        configured.cache_clear()
    os.environ["LIFEMATE_NATIVE_FEWSHOT_K"] = str(args.fewshot_k)  # the CLI decides, not the caller's shell
    transcribe = transcript_lookup(args.transcripts) if args.transcripts else None
    selected = set(sorted(gold)[:args.limit] if args.limit else gold)
    totals, lengths = Counter(), []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as sink:
        for row in kit.rows(selected):
            try:
                sample = _sample_from_row(kit, row, transcribe, monotonic() + 3_600, control_noop=False)
            except CallbackError:  # an audio file with no usable text in --transcripts
                totals["skipped_no_transcript"] += 1
                continue
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
                      "transcripts": str(args.transcripts) if args.transcripts else "train", "fewshot_k": args.fewshot_k,
                      "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
