"""TRAIN oracle tool: convert gold tool_calls into compiled-plan DSL steps and verify them by execution.

For every TRAIN sample outside the 309 validation IDs: replay the gold calls through one official ToolEnv
(public API only) to observe each result, express every gold argument as a typed plan SOURCE, then compile
and run the plan with the real CompiledPlanner/run_sample against a fresh ToolEnv and compare the produced
tool_calls and state hash with the gold replay. Offline analysis only; never runs during inference.
"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import re
from time import monotonic
from types import SimpleNamespace

from src.agent.compiled_plan import CompiledPlanner, _blocked_write, _compile
from src.agent.plan_protocol import InvalidPlan, enum_values, exact_span, leaves, matches, scalar_paths, spans
from src.agent.plan_values import Unresolved, convert_span, shift_time
from src.agent.planner import PlannerInput
from src.agent.runner import Limits, run_sample
from src.inference.official import OfficialKit
from src.inference.pipeline import _sample_from_row
from src.tooling.json_data import InvalidJSON, decode_json
from src.tooling.registry import ToolRejected

_CATEGORY_FIELDS = {"type", "kind", "category", "status", "state", "mode", "relation"}
_FORMS = (("datetime", r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d"), ("date", r"\d{4}-\d\d-\d\d"),
          ("time", r"\d\d:\d\d"))
_EDGE = ".,;:!?…\"'()“”"


class Unexpressible(ValueError):
    """A gold argument has no plan SOURCE that reproduces it; the message is the grouped reason."""


def find_span(value, mode: str, text: str, context: dict) -> str | None:
    """The longest transcript span whose conversion equals the gold value exactly."""
    hits = []
    for span in spans(text):
        try:
            converted = convert_span(exact_span(text, span), mode, context)
        except (Unresolved, ValueError):
            continue
        if type(converted) is type(value) and converted == value:
            hits.append(span)
    return max(hits, key=len) if hits else None


def text_span(value: str, text: str) -> str | None:
    """A verbatim (case-exact after evaluation) transcript span for a free-text value."""
    try:
        return value if value.strip() and exact_span(text, value) == value else None
    except Unresolved:
        return None


def record_lists(root, path=()):
    """(dict-only path, list of dict records) pairs."""
    if isinstance(root, dict):
        for key, child in root.items():
            yield from record_lists(child, path + (key,))
    elif isinstance(root, list) and root and all(isinstance(item, dict) for item in root):
        yield path, root


def same(first, second) -> bool:
    return type(first) is type(second) and first == second


def where_for(records: list[dict], record: dict, text: str, context: dict) -> dict | None:
    """A grounded where-filter that selects exactly this record, or None."""
    options = []
    for key, value in record.items():
        if type(value) is bool:
            options.append((key, {"value": value}, value))
        elif type(value) is int and (span := find_span(value, "integer", text, context)) is not None:
            options.append((key, {"text": span, "as": "integer"}, value))
        elif isinstance(value, str) and value.strip():
            try:
                options.append((key, {"text": exact_span(text, value)}, value))
            except Unresolved:
                pass
            if key in _CATEGORY_FIELDS:
                options.append((key, {"value": value}, value))
    for size in (1, 2):
        for chosen in combinations(options, size):
            if len({key for key, _, _ in chosen}) != size:
                continue
            hits = [item for item in records if all(key in item and matches(item[key], expected)
                                                    for key, _, expected in chosen)]
            if hits == [record]:
                return {key: source for key, source, _ in chosen}
    return None


def selectors(value, payload: dict, step_id: str, text: str, context: dict, evidence: list | None = None):
    """Result SOURCES inside one earlier result that evaluate to value: direct fields, then grounded rows."""
    if evidence is not None:  # x-ref: only the producer's declared evidence paths compile.
        layouts = []
        for path in evidence:
            if "*" not in path:
                layouts.append((tuple(path), None))
            elif path.count("*") == 1:
                star = path.index("*")
                layouts.append((tuple(path[:star]), tuple(path[star + 1:])))
    else:
        layouts = [(path, None) for path, _ in scalar_paths(payload)]
        layouts += [(path, (key,)) for path, rows in record_lists(payload) for key in
                    {key for row in rows for key in row}]
    for path, field in layouts:
        if field is None:
            current = payload
            for key in path:
                current = current.get(key) if isinstance(current, dict) else None
            if same(current, value):
                yield {"result": step_id, "path": list(path)}
            continue
        current = payload
        for key in path:
            current = current.get(key) if isinstance(current, dict) else None
        if not isinstance(current, list) or not all(isinstance(row, dict) for row in current):
            continue
        for row in current:
            target = row
            for key in field:
                target = target.get(key) if isinstance(target, dict) else None
            if not same(target, value):
                continue
            where = where_for(current, row, text, context)
            if where is not None:
                yield {"result": step_id, "path": list(path), "where": where, "field": list(field)}
            elif len(current) == 1:
                yield {"result": step_id, "path": list(path), "field": list(field)}


class Converter:
    """Express gold arguments for one sample as typed plan SOURCES."""

    def __init__(self, sample, tools: dict[str, dict]):
        self.sample = sample
        self.tools = tools
        self.text = sample.user_request
        self.results: list[tuple[str, str, dict]] = []  # (step id, tool, payload), newest last
        self.missing: list[dict] = []  # per-argument detail of unexpressible gold arguments
        self.history_user = []
        for index, raw in enumerate(sample.history, 1):
            try:
                turn = decode_json(raw, max_bytes=64_000)
            except InvalidJSON:
                continue
            if isinstance(turn, dict) and turn.get("role") == "user" and isinstance(turn.get("text"), str):
                self.history_user.append((index, turn["text"]))

    def context_source(self, value):
        for path, found in scalar_paths(self.sample.context):
            if same(found, value):
                return {"context": list(path)}
        return None

    def history_source(self, value):
        for index, prior in enumerate(self.sample.prior_tool_results, 1):
            if prior.payload.get("ok") is not True or prior.payload.get("done") is False:
                continue
            for path, found in scalar_paths(prior.payload):
                if same(found, value):
                    return {"history": index, "path": list(path)}
        return None

    def in_history_list(self, value) -> bool:
        return any(same(row.get(key), value) for prior in self.sample.prior_tool_results
                   for _, rows in record_lists(prior.payload) for row in rows for key in row)

    def in_results(self, value) -> bool:
        return any(same(found, value) for _, _, payload in self.results for found in leaves(payload))

    def result_source(self, value, xref: str | None = None):
        for step_id, tool, payload in reversed(self.results):
            evidence = None
            if xref is not None:
                evidence = [item["path"] for item in self.tools[tool].get("evidence_paths", [])
                            if item["namespace"] == xref]
                if not evidence:
                    continue
            for source in selectors(value, payload, step_id, self.text, self.sample.context, evidence):
                return source
        return None

    def observed(self, value, xref: str | None = None):
        return self.result_source(value, xref) or self.history_source(value) or self.context_source(value)

    def shift_source(self, value: str, kind: str):
        for by in spans(self.text):
            try:
                minutes = convert_span(exact_span(self.text, by), "duration", self.sample.context)
            except (Unresolved, ValueError):
                continue
            for direction, sign in (("later", -1), ("earlier", 1)):
                try:
                    base = shift_time(value, sign * minutes)
                except Unresolved:
                    continue
                source = self.observed(base)
                if source is None and (span := find_span(base, kind, self.text, self.sample.context)):
                    source = {"text": span, "as": kind}
                if source is not None:
                    return {"shift": source, "by": {"text": by, "as": "duration"}, "dir": direction}
        return None

    def history_text_source(self, value, modes):
        for index, said in self.history_user:
            for mode in modes:
                if mode == "text" and isinstance(value, str):
                    span = text_span(value, said)
                else:
                    span = find_span(value, mode, said, self.sample.context)
                if span is not None:
                    return {"history_text": index, "text": span, "as": mode}
        return None

    def unexpressible(self, kind: str, value) -> Unexpressible:
        """Group why no SOURCE reproduces the value: where it was seen and why that cannot be used."""
        if self.in_results(value):
            reason = "in_result_no_grounded_selector"
        elif self.in_history_list(value):
            reason = "in_history_list"
        elif kind == "text" and _found_casefold(value, self.text):
            reason = "case_differs"
        else:
            reason = "not_verbatim" if kind == "text" else "not_observed" if kind == "xref" else "no_span_or_field"
        return Unexpressible(f"{kind}:{reason}")

    def source(self, value, schema: dict, tool: str):
        """One SOURCE for a gold value, or raise Unexpressible(kind:reason)."""
        schema = schema if isinstance(schema, dict) else {}
        xref = schema.get("x-ref")
        if tool == "sys.ask_missing_info":
            if isinstance(value, list):
                return {"array": [{"value": item} for item in value]}
            return {"value": value}
        if xref:
            found = self.observed(value, xref)
            if found is None and (span := text_span(value, self.text) if isinstance(value, str) else None):
                found = {"text": span}
            if found is None:
                raise self.unexpressible("xref", value)
            return found
        if isinstance(value, dict):
            fields = schema.get("properties", {})
            return {"object": {key: self.source(item, fields.get(key, {}), tool) for key, item in value.items()}}
        if isinstance(value, list):
            return {"array": [self.source(item, schema.get("items", {}), tool) for item in value]}
        if type(value) is bool or value in enum_values(schema) and value is not None:
            return {"value": value}
        if isinstance(value, str):
            kind = next((mode for mode, form in _FORMS if re.fullmatch(form, value)), "text")
            if kind == "text":
                span = text_span(value, self.text)
                if span is not None:
                    return {"text": span}
            else:
                span = find_span(value, kind, self.text, self.sample.context)
                if span is not None:
                    return {"text": span, "as": kind}
            found = self.observed(value) or self.history_text_source(value, ("text",))
            if found is None and kind in ("datetime", "time"):
                found = self.shift_source(value, kind)
            if found is None:
                raise self.unexpressible(kind, value)
            return found
        if type(value) in (int, float):
            modes = ("integer", "money", "duration", "number") if type(value) is int else ("number",)
            for mode in modes:
                span = find_span(value, mode, self.text, self.sample.context)
                if span is not None:
                    return {"text": span, "as": mode}
            found = self.observed(value) or self.history_text_source(value, modes)
            if found is None:
                raise self.unexpressible(type(value).__name__, value)
            return found
        raise Unexpressible(f"{type(value).__name__}:unsupported_value")

    def anchor(self, step: dict, tool: dict) -> str:
        """A verbatim span for the step: around a span it uses, else the clause words that best match the
        tool description, else a clause; never a negated span for a write."""
        used = [item for item in _texts(step["args"]) if _found_casefold(item, self.text)]
        words = self.text.split()
        candidates = []
        for span in sorted(used, key=lambda item: self.text.casefold().find(item.casefold())):
            found = exact_span(self.text, span)
            start = len(self.text[:self.text.find(found)].split())
            for extra in (2, 1, 0):
                if start - extra >= 0:
                    candidate = " ".join(words[start - extra:start] + [found]).strip(_EDGE)
                    if candidate and not re.search(r"[.,;!?]\s", candidate):
                        candidates.append(candidate)
        clauses = [clause.split() for clause in re.split(r"[.,;!?…]", self.text) if clause.strip()]
        target = step["args"].get("tool_name", {}).get("value")  # a clarification anchors on its target action
        description = self.tools[target]["description"] if target in self.tools else tool.get("description", "")
        described = set(re.findall(r"\w+", description.casefold()))
        windows = [(sum(word.casefold().strip(_EDGE) in described for word in clause[i:i + size]), -size,
                    " ".join(clause[i:i + size]))
                   for clause in clauses for size in (5, 4, 3, 2) for i in range(max(1, len(clause) - size + 1))]
        candidates += [window for score, _, window in sorted(windows, key=lambda item: (-item[0], item[1]))
                       if score >= 1]
        candidates += [" ".join(clause[:8]) for clause in clauses]
        for candidate in candidates:
            try:
                candidate = exact_span(self.text, candidate)
            except Unresolved:
                continue
            if not tool.get("writes") or not _blocked_write(SimpleNamespace(anchor=candidate), self.sample):
                return candidate
        raise Unexpressible("anchor:negated_context")

    def convert(self, calls: list[dict], payloads: list[dict]) -> tuple[dict, list[str]]:
        """A plan for the gold calls plus the grouped reasons of unexpressible arguments."""
        steps, reasons = [], []
        for index, (call, payload) in enumerate(zip(calls, payloads), 1):
            tool = self.tools.get(call["name"])
            step_id = f"s{index}"
            if tool is None:
                reasons.append("tool:not_installed")
                continue
            fields = tool["parameters"].get("properties", {})
            args = {}
            for key, value in call["arguments"].items():
                try:
                    args[key] = self.source(value, fields.get(key, {}), call["name"])
                except Unexpressible as exc:
                    reasons.append(str(exc))
                    self.missing.append({"tool": call["name"], "param": key, "reason": str(exc)})
            step = {"id": step_id, "tool": call["name"], "anchor": "", "args": args}
            try:
                step["anchor"] = self.anchor(step, tool)
            except Unexpressible as exc:
                reasons.append(str(exc))
            steps.append(step)
            self.results.append((step_id, call["name"], payload))
        if len(calls) > 7:
            reasons.append("plan:over_tool_limit")
        return {"steps": steps}, reasons


def _found_casefold(value: str, text: str) -> bool:
    try:
        exact_span(text, value)
        return True
    except Unresolved:
        return False


def _texts(source) -> list[str]:
    """Current-utterance spans used anywhere inside a step's sources."""
    if isinstance(source, dict):
        found = [source["text"]] if "text" in source and "history_text" not in source else []
        for key, value in source.items():
            if key not in ("text", "history_text") and isinstance(value, (dict, list)):
                found += _texts(value)
        return found
    if isinstance(source, list):
        return [item for value in source for item in _texts(value)]
    return []


def verify(kit: OfficialKit, row: dict, plan: dict, gold_calls: list, gold_hash: str) -> tuple[bool, str | None]:
    """Compile and execute the plan with the real planner against a fresh ToolEnv."""
    sample = _sample_from_row(kit, row, None, monotonic() + 3_600, control_noop=False)
    tools = kit.registry.available(sample.installed_apps, sample.allowed_tools)
    raw = json.dumps(plan, ensure_ascii=False)
    try:
        _compile(raw, PlannerInput(sample, tools, ()), {tool["name"]: tool for tool in tools}, [], 7)
    except (InvalidPlan, Unresolved, InvalidJSON) as exc:
        return False, f"compile:{exc}"
    repairs = []

    def generate(system, user, *, deadline=None):
        repair = json.loads(user).get("repair")
        if repair:
            repairs.append(repair["reason"])
        return raw

    environment = kit.open_environment(sample)
    result = run_sample(sample, kit.registry, CompiledPlanner(generate).for_sample(), lambda _: environment,
                        limits=Limits(sample_seconds=60, planner_seconds=30))
    calls, state_hash = environment.read_output()
    if calls == gold_calls and state_hash == gold_hash:
        return True, None
    if repairs:
        return False, f"runtime:{repairs[0]}"
    if state_hash != gold_hash and calls == gold_calls:
        return False, "mismatch:state_only"
    return False, f"mismatch:calls ({result.status})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--validation", type=Path, default=Path("research/validation_split.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/gold_plans.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="convert only the first N eligible samples")
    args = parser.parse_args()
    if any(part.endswith("_environment") for part in args.data_root.resolve().parts):
        raise ValueError("Forbidden root")
    split = json.loads(args.validation.read_text(encoding="utf-8"))
    if split.get("source_split") != "train" or len(split["validation_ids"]) != 309:
        raise ValueError("Expected the 309-ID TRAIN validation manifest")
    held_out = set(split["validation_ids"])
    kit = OfficialKit(args.data_root, "train")
    gold = {}
    with (kit.directory / "train.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item["id"] not in held_out:
                gold[item["id"]] = item["label"]["tool_calls"]
    eligible = sorted(gold)[:args.limit] if args.limit else set(gold)
    totals, argument_reasons, sample_reasons, failures = Counter(), Counter(), Counter(), Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    with args.output.open("w", encoding="utf-8") as sink:
        for row in kit.rows(set(eligible)):
            totals["samples"] += 1
            expected = gold[row["id"]]
            sample = _sample_from_row(kit, row, None, monotonic() + 3_600, control_noop=False)
            tools = {tool["name"]: tool for tool in kit.registry.available(sample.installed_apps,
                                                                           sample.allowed_tools)}
            environment = kit.open_environment(sample)  # the one gold-replay ToolEnv for this sample
            payloads, replay_error = [], None
            for call in expected:
                try:
                    payloads.append(environment.call(call["name"], call["arguments"]))
                except ToolRejected:
                    replay_error = "replay:preflight_rejected"
                    break
            calls, gold_hash = environment.read_output()
            if replay_error is None and calls != expected:
                replay_error = "replay:log_differs"
            converter = Converter(sample, tools)
            plan, reasons = ({"steps": []}, [replay_error]) if replay_error else \
                converter.convert(expected, payloads)
            argument_reasons.update(reasons)
            sample_reasons.update(set(reasons))
            converted = not reasons
            verified, failure = False, None
            if converted:
                totals["converted"] += 1
                verified, failure = verify(kit, row, plan, expected, gold_hash)
                if verified:
                    totals["verified"] += 1
                    totals["verified_nonempty"] += bool(expected)
                else:
                    failures[re.sub(r"\d+", "N", failure)[:120]] += 1
            totals["nonempty"] += bool(expected)
            sink.write(json.dumps({"id": row["id"], "transcript": row["audio_text"], "plan": plan,
                                   "converted": converted, "verified": verified, "reasons": reasons,
                                   "unexpressed": converter.missing,
                                   "verify_failure": failure}, ensure_ascii=False) + "\n")
    count = totals["samples"]
    report = {
        "dataset": "TRAIN v3.2.0 excluding the 309 validation IDs", "samples": count,
        "nonempty_gold_samples": totals["nonempty"],
        "fully_converted": totals["converted"], "fully_converted_pct": round(100 * totals["converted"] / count, 2),
        "verified_exact": totals["verified"], "verified_exact_pct": round(100 * totals["verified"] / count, 2),
        "verified_exact_nonempty": totals["verified_nonempty"],
        "unexpressible_argument_reasons": dict(argument_reasons.most_common()),
        "samples_with_reason": dict(sample_reasons.most_common()),
        "verify_failures_after_conversion": dict(failures.most_common(15)),
        "seconds": round(monotonic() - started, 1), "output": str(args.output),
    }
    args.output.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
