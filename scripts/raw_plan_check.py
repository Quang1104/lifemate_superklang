"""TRAIN oracle check of raw-literal plans (task 4b) and few-shot prompt growth (task 4). Offline analysis only.

For each selected TRAIN sample, the gold calls are replayed on one official ToolEnv (public API) for the gold
call log and state hash. Then plans shaped like the logged model output are compiled and run with the real
CompiledPlanner/run_sample against a fresh ToolEnv:
  raw:      every argument is the gold literal (no typed SOURCEs), anchors from gold_plans.jsonl or derived.
  fake_ids: as raw, but x-ref IDs that are not in the request/context/history are replaced by an invented
            "<ns>_123"; when the planner asks for a repair of a focused step, the simulated model copies the
            gold value (as a model would from visible_results). Measures routing plus one-repair recovery.
Exact = same tool_calls and state hash as the gold replay. The model itself is NOT run here.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from statistics import median
from time import monotonic

from scripts.gold_to_plan import Converter, Unexpressible
from src.agent.compiled_plan import _SYSTEM, CompiledPlanner, _compile, _json
from src.agent.fewshot import configured
from src.agent.plan_protocol import Grounding, InvalidPlan
from src.agent.plan_values import Unresolved
from src.agent.planner import PlannerInput
from src.agent.runner import Limits, run_sample
from src.inference.official import OfficialKit
from src.inference.pipeline import _sample_from_row
from src.tooling.json_data import InvalidJSON
from src.tooling.registry import ToolRejected


def _ids(args) -> dict[str, set[str]]:
    validation = json.loads(Path("research/validation_split.json").read_text(encoding="utf-8"))
    smoke = json.loads(Path("research/smoke_split.json").read_text(encoding="utf-8"))
    leave = json.loads(Path("research/leave_app_out_split.json").read_text(encoding="utf-8"))
    with args.plans.open(encoding="utf-8") as lines:
        plans = [json.loads(line) for line in lines]
    return {"smoke40": set(smoke["validation_ids"]), "leave_app_out_holdout": set(leave["validation_ids"]),
            "verified530": {row["id"] for row in plans if row["verified"]},
            "train1236": {row["id"] for row in plans} - set(validation["validation_ids"])}


def _fake(plan: dict, sample, tools: dict) -> tuple[dict, int]:
    grounding, fakes, steps = Grounding(sample), 0, []
    for step in plan["steps"]:
        fields = tools[step["tool"]]["parameters"].get("properties", {})
        args = dict(step["args"])
        for key, value in args.items():
            kind = fields.get(key, {}).get("x-ref")
            if isinstance(kind, str) and isinstance(value, str) and grounding.source(value, fields[key]) is None:
                args[key] = f"{kind.split('.')[-1].rstrip('s')}_123"
                fakes += 1
        steps.append({**step, "args": args})
    return {"steps": steps}, fakes


def _run(kit, row, plan, gold_plan, gold_calls, gold_hash) -> tuple[str, int]:
    sample = _sample_from_row(kit, row, None, monotonic() + 3_600, control_noop=False)
    tools = kit.registry.available(sample.installed_apps, sample.allowed_tools)
    raw = json.dumps(plan, ensure_ascii=False)
    try:
        _compile(raw, PlannerInput(sample, tools, ()), {tool["name"]: tool for tool in tools}, [], 7)
    except (InvalidPlan, Unresolved, InvalidJSON) as exc:
        return f"compile:{re.sub(r'[0-9]+', 'N', str(exc))[:70]}", 0
    repairs = []

    def generate(system, user, *, deadline=None):
        repair = json.loads(user).get("repair")
        if not repair:
            return raw
        repairs.append(repair["reason"])
        focus = repair.get("focus")
        if focus:  # the simulated model copies the real value from visible results
            gold = next(step for step in gold_plan["steps"] if step["id"] == focus["id"])
            return json.dumps({"steps": [gold]}, ensure_ascii=False)
        return raw

    environment = kit.open_environment(sample)
    run_sample(sample, kit.registry, CompiledPlanner(generate, fewshot=0).for_sample(), lambda _: environment,
               limits=Limits(sample_seconds=60, planner_seconds=30))
    calls, state_hash = environment.read_output()
    if calls == gold_calls and state_hash == gold_hash:
        return "exact", len(repairs)
    return ("mismatch:state_only" if calls == gold_calls else "mismatch:calls"), len(repairs)


def check(args) -> dict:
    sets = _ids(args)
    wanted = set().union(*(sets[name] for name in args.sets))
    kit = OfficialKit(args.data_root, "train")
    with args.plans.open(encoding="utf-8") as lines:
        anchors = {row["id"]: [step["anchor"] for step in row["plan"]["steps"]] for row in map(json.loads, lines)}
    gold = {}
    with (kit.directory / "train.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item["id"] in wanted:
                gold[item["id"]] = item["label"]["tool_calls"]
    outcome: dict[str, dict] = {}
    started = monotonic()
    for row in kit.rows(wanted):
        sample = _sample_from_row(kit, row, None, monotonic() + 3_600, control_noop=False)
        tools = {tool["name"]: tool for tool in kit.registry.available(sample.installed_apps, sample.allowed_tools)}
        environment = kit.open_environment(sample)  # the one gold-replay ToolEnv for this sample
        try:
            for call in gold[row["id"]]:
                environment.call(call["name"], call["arguments"])
        except ToolRejected:
            outcome[row["id"]] = {"raw": "replay_rejected", "fake_ids": "replay_rejected"}
            continue
        calls, gold_hash = environment.read_output()
        steps, converter = [], Converter(sample, tools)
        for index, call in enumerate(gold[row["id"]], 1):
            if call["name"] not in tools:
                break
            known = anchors.get(row["id"], [])
            anchor = known[index - 1] if len(known) >= index else None
            if not anchor:
                try:
                    anchor = converter.anchor({"args": {}, "tool": call["name"]}, tools[call["name"]])
                except Unexpressible:
                    anchor = ""
            steps.append({"id": f"s{index}", "tool": call["name"], "anchor": anchor, "args": call["arguments"]})
        if len(steps) != len(gold[row["id"]]) or any(not step["anchor"] for step in steps):
            outcome[row["id"]] = {"raw": "no_anchor_or_tool", "fake_ids": "no_anchor_or_tool"}
            continue
        plan = {"steps": steps}
        raw_result, _ = _run(kit, row, plan, plan, gold[row["id"]], gold_hash)
        fake_plan, fakes = _fake(plan, sample, tools)
        fake_result, repairs = (_run(kit, row, fake_plan, plan, gold[row["id"]], gold_hash) if fakes
                                else (raw_result, 0))
        outcome[row["id"]] = {"raw": raw_result, "fake_ids": fake_result, "fakes": fakes, "repairs": repairs}
    report = {"note": "TRAIN oracle literals; not a model score", "seconds": round(monotonic() - started, 1)}
    for name in args.sets:
        rows = [outcome[sample_id] for sample_id in sorted(sets[name])]
        faked = [item for item in rows if item.get("fakes")]
        report[name] = {
            "samples": len(rows),
            "raw_exact": sum(item["raw"] == "exact" for item in rows),
            "raw_exact_pct": round(100 * sum(item["raw"] == "exact" for item in rows) / len(rows), 2),
            "raw_failures": dict(Counter(item["raw"] for item in rows if item["raw"] != "exact").most_common(12)),
            "samples_with_invented_ids": len(faked),
            "invented_ids_exact": sum(item["fake_ids"] == "exact" for item in faked),
            "invented_ids_exact_without_repair": sum(item["fake_ids"] == "exact" and not item["repairs"]
                                                     for item in faked),
            "invented_ids_failures": dict(Counter(item["fake_ids"] for item in faked
                                                  if item["fake_ids"] != "exact").most_common(8)),
        }
    return report


def prompt_size(args) -> dict:
    """Characters of system + user payload on smoke40, with and without few-shot examples."""
    kit = OfficialKit(args.data_root, "train")
    smoke = set(json.loads(Path("research/smoke_split.json").read_text(encoding="utf-8"))["validation_ids"])
    sizes: dict[int, list[int]] = {k: [] for k in (0, 2, 3)}
    for row in kit.rows(smoke):
        sample = _sample_from_row(kit, row, None, monotonic() + 3_600, control_noop=False)
        request = PlannerInput(sample, kit.registry.available(sample.installed_apps, sample.allowed_tools), ())
        for k in sizes:
            planner = CompiledPlanner(lambda *a, **kw: "", fewshot=k, examples=configured(None))
            sizes[k].append(len(_SYSTEM) + len(_json(planner.payload(request))))
    base = sizes[0]
    return {f"k={k}": {"sum": sum(values), "median": median(values), "max": max(values),
                       "added_median": median(b - a for a, b in zip(base, values)),
                       "added_max": max(b - a for a, b in zip(base, values)),
                       "growth_pct_of_sum": round(100 * (sum(values) - sum(base)) / sum(base), 2)}
            for k, values in sizes.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--plans", type=Path, default=Path("outputs/gold_plans.jsonl"))
    parser.add_argument("--sets", nargs="+", default=["smoke40", "leave_app_out_holdout", "verified530"],
                        choices=["smoke40", "leave_app_out_holdout", "verified530", "train1236"])
    parser.add_argument("--prompt-size", action="store_true")
    args = parser.parse_args()
    if any(part.endswith("_environment") for part in args.data_root.resolve().parts):
        raise ValueError("Forbidden root")
    report = prompt_size(args) if args.prompt_size else check(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
