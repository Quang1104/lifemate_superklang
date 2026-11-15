"""Run one sample without replaying context or repeating tool attempts."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
import os
from queue import Empty, Queue
from threading import Thread
from time import monotonic
from typing import Any, Callable, Protocol, TypeVar

from ..inference.types import Sample
from ..tooling.evidence import EvidenceLedger
from ..tooling.json_data import InvalidJSON, decode_json, encode_json
from ..tooling.registry import ToolRegistry, ToolRejected
from .arg_normalize import normalize_arguments
from .planner import Finish, Observation, Planner, PlannerInput, ToolCall, parse_action


class Environment(Protocol):
    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Limits:
    max_steps: int = 8
    sample_seconds: float = 30.0
    planner_seconds: float = 22.0
    tool_seconds: float = 5.0
    planner_output_bytes: int = 8_192
    result_bytes: int = 48_000

    def __post_init__(self) -> None:
        for value in (self.max_steps, self.planner_output_bytes, self.result_bytes):
            if type(value) is not int or value < 1:
                raise ValueError("Step and byte budgets must be positive integers")
        for value in (self.sample_seconds, self.planner_seconds, self.tool_seconds):
            if isinstance(value, bool) or not isfinite(value) or value <= 0:
                raise ValueError("Timeouts must be finite and positive")


@dataclass(frozen=True)
class RunResult:
    status: str
    reply: str | None
    steps: int
    dispatched_calls: int
    rejections: int
    session_safe_to_export: bool
    guard_rejections: tuple[dict[str, str], ...] = ()
    last_planner_output: str | None = None


class CallTimeout(TimeoutError):
    """The callback might still be running. Its session must not be reused."""


class CallbackError(RuntimeError):
    """An injected callback failed."""


_T = TypeVar("_T")


def _bounded_call(callback: Callable[[], _T], seconds: float) -> _T:
    if seconds <= 0:
        raise CallTimeout("Sample deadline elapsed")
    outcome: Queue[tuple[bool, Any]] = Queue(maxsize=1)

    def invoke() -> None:
        try:
            outcome.put((True, callback()))
        except BaseException as exc:
            outcome.put((False, exc))

    worker = Thread(target=invoke, daemon=True)
    worker.start()
    try:
        completed, value = outcome.get(timeout=seconds)
    except Empty as exc:
        raise CallTimeout("Callback deadline elapsed") from exc
    if not completed:
        raise CallbackError("Callback raised") from value
    return value


def _record_lists(value: Any):
    """Every list of records with an id, anywhere inside a tool result."""
    if isinstance(value, list):
        if value and all(isinstance(item, dict) and "id" in item for item in value):
            yield value
        for item in value:
            yield from _record_lists(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _record_lists(item)


def same_name_ask(name: str, arguments: dict[str, Any], parameters: dict[str, Any],
                  results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Arguments of the clarification gold asks when an ID was picked among records that share a name.

    In TRAIN gold a lookup that returns two records with one name is followed by sys.ask_missing_info for the
    ID parameter, not by a choice (docs/CONTEXT_PLAN.md 3.1). `results` are this turn's tool results only.
    """
    def label(record: dict[str, Any]) -> str:
        return str(record.get("name") or record.get("title") or "").casefold()

    properties = parameters.get("properties", {})
    for parameter, value in arguments.items():
        if not isinstance(value, str) or "x-ref" not in properties.get(parameter, {}):
            continue
        for records in (found for result in results for found in _record_lists(result)):
            chosen = next((record for record in records if record["id"] == value and label(record)), None)
            if chosen is not None and sum(label(record) == label(chosen) for record in records) > 1:
                return {"tool_name": name, "params": [parameter]}
    return None


def moved_event_end(arguments: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    """calendar.update_event arguments with the end TRAIN gold always sends when the start moves.

    Gold keeps the event's length: end = new start + old duration (27 of 27 moves; docs/CONTEXT_PLAN.md).
    Nothing changes unless the old event, with its start and end, is in the visible results.
    """
    if "start" not in arguments or "end" in arguments:
        return arguments
    old = None
    for records in (found for result in results for found in _record_lists(result)):
        old = next((record for record in records if record["id"] == arguments.get("event_id")
                    and "start" in record and "end" in record), old)
    try:
        length = datetime.fromisoformat(old["end"]) - datetime.fromisoformat(old["start"])
        return {**arguments, "end": (datetime.fromisoformat(arguments["start"]) + length).isoformat()}
    except (TypeError, ValueError):  # no such event, or timestamps this cannot read: leave the call alone
        return arguments


# Kinship and role words that open a spoken name. In TRAIN contacts are stored with one (sếp 33, bác 27, cô 23,
# anh 15, thằng 4, thầy 3, bà 3, vợ 3, chị 2, ông 2, mẹ 1, chú 1, bé 1) or without it although the user says it
# (anh 6, chị 3, thầy 1, ông 1); only the search result tells which.
_NAME_TITLES = frozenset("anh chị thầy ông sếp bác cô thằng bà vợ mẹ chú bé".split())


def name_only_search(arguments: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any] | None:
    """contacts.search arguments without the leading title, when the search as asked found nobody."""
    query = arguments.get("query")
    if payload.get("ok") is not True or payload.get("contacts") != [] or not isinstance(query, str):
        return None
    title, _, name = query.strip().partition(" ")
    return {**arguments, "query": name.strip()} if title.casefold() in _NAME_TITLES and name.strip() else None


def run_sample(sample: Sample, registry: ToolRegistry, planner: Planner,
               environment_factory: Callable[[Sample], Environment], *,
               limits: Limits = Limits()) -> RunResult:
    deadline = monotonic() + limits.sample_seconds

    def remaining(operation_limit: float) -> float:
        return min(operation_limit, deadline - monotonic())

    try:
        environment = _bounded_call(lambda: environment_factory(sample), remaining(limits.tool_seconds))
    except CallTimeout:
        return RunResult("environment_timeout", None, 0, 0, 0, False)
    except CallbackError:
        return RunResult("environment_error", None, 0, 0, 0, False)

    tools = registry.available(sample.installed_apps, sample.allowed_tools)
    observations: list[Observation] = []
    ledger = EvidenceLedger()
    for prior in sample.prior_tool_results:
        if prior.payload.get("ok") is True and prior.payload.get("done") is not False:
            ledger.record(prior.payload, registry.evidence_paths(prior.name), prior.arguments)
    nonrepeatable: set[tuple[str, str]] = set()
    recent_reads: set[tuple[str, str]] = set()
    dispatched_calls = 0
    rejections = 0
    step = 0
    last_planner_output: str | None = None
    follow_up: list[str] = []  # an action the runner itself queues; it passes every guard like a planner action

    def result(status: str, reply: str | None = None, *, safe: bool = True) -> RunResult:
        # Guard reasons already exclude submitted values; never include arguments.
        guard_rejections = tuple({"tool": item.tool, "reason": item.content["reason"]}
                                 for item in observations if item.kind == "rejected" and item.tool is not None)
        return RunResult(status, reply, step, dispatched_calls, rejections, safe,
                         guard_rejections, last_planner_output if dispatched_calls == 0 else None)

    for step in range(1, limits.max_steps + 1):
        if monotonic() >= deadline:
            return result("sample_timeout")
        planner_deadline = min(deadline, monotonic() + limits.planner_seconds)
        request = PlannerInput(deepcopy(sample), deepcopy(tools), tuple(deepcopy(observations)),
                               planner_deadline - 0.1)
        try:
            output = (follow_up.pop() if follow_up
                      else _bounded_call(lambda: planner(request), remaining(limits.planner_seconds)))
            # TRAIN native diagnostics expose the raw chat text before call normalization.
            diagnostic_output = getattr(planner, "last_planner_output", None)
            last_planner_output = (diagnostic_output[:400] if isinstance(diagnostic_output, str)
                                   else output[:400] if isinstance(output, str) else None)
            action = parse_action(output, max_bytes=limits.planner_output_bytes)
        except CallTimeout:
            # Only a tool call can be in flight against the environment.
            return result("planner_timeout")
        except CallbackError as exc:
            if isinstance(exc.__cause__, TimeoutError):
                return result("planner_timeout")
            return result("planner_error")
        except InvalidJSON:
            observations.append(Observation("rejected", None, {"reason": "Invalid planner JSON action"}))
            rejections += 1
            continue
        if isinstance(action, Finish):
            if action.reply in {"plan_invalid", "plan_incomplete", "plan_timeout"}:
                return result(action.reply)
            return result("finished", action.reply)

        if os.environ.get("LIFEMATE_ARG_NORMALIZE") == "1":
            parameters = next((tool["parameters"] for tool in tools if tool["name"] == action.name), None)
            if parameters is not None:
                action = ToolCall(action.name, normalize_arguments(action.arguments, parameters, sample.context))

        if os.environ.get("LIFEMATE_SMALL_FIXES") == "1" and action.name == "calendar.update_event":
            action = ToolCall(action.name, moved_event_end(action.arguments, [
                *(prior.payload for prior in sample.prior_tool_results),
                *(item.content for item in observations if item.kind == "untrusted_tool_result")]))

        if os.environ.get("LIFEMATE_DUP_GUARD") == "1" and not action.name.startswith("sys."):
            parameters = next((tool["parameters"] for tool in tools if tool["name"] == action.name), None)
            if parameters is not None and any(tool["name"] == "sys.ask_missing_info" for tool in tools):
                ask = same_name_ask(action.name, action.arguments, parameters, [
                    item.content for item in observations if item.kind == "untrusted_tool_result"])
                if ask is not None:
                    action = ToolCall("sys.ask_missing_info", ask)

        try:
            spec = registry.authorize(action.name, action.arguments, sample.installed_apps,
                                      ledger, sample.allowed_tools)
            key = (action.name, encode_json(action.arguments, max_bytes=limits.planner_output_bytes))
            if key in nonrepeatable or key in recent_reads:
                raise ToolRejected("Identical action already attempted in this sample")
        except (ToolRejected, InvalidJSON) as exc:
            observations.append(Observation("rejected", action.name, {"reason": str(exc)}, action.arguments))
            rejections += 1
            continue

        # Record before dispatch so an error or no-op never enables a retry.
        nonrepeatable.add(key)
        dispatched_calls += 1
        try:
            payload = _bounded_call(lambda: environment.call(action.name, deepcopy(action.arguments)),
                                    remaining(limits.tool_seconds))
        except CallTimeout:
            return result("tool_timeout", safe=False)
        except CallbackError as exc:
            if isinstance(exc.__cause__, ToolRejected):
                observations.append(Observation("rejected", action.name,
                                                {"reason": "Environment preflight rejected the action"},
                                                action.arguments))
                rejections += 1
                dispatched_calls -= 1
                continue
            return result("tool_error", safe=False)
        try:
            encoded = encode_json(payload, max_bytes=limits.result_bytes)
            payload = decode_json(encoded, max_bytes=limits.result_bytes)
        except InvalidJSON:
            return result("invalid_tool_result", safe=False)
        if not isinstance(payload, dict) or type(payload.get("ok")) is not bool:
            return result("invalid_tool_result", safe=False)
        observations.append(Observation("untrusted_tool_result", action.name, payload,
                                        action.arguments))
        if os.environ.get("LIFEMATE_CONTACT_RETRY") == "1" and action.name == "contacts.search":
            again = name_only_search(action.arguments, payload)
            if again is not None:  # "anh Tuấn" found nobody: look for "Tuấn" before the planner gives up
                follow_up.append(encode_json({"action": "call", "tool": action.name, "arguments": again},
                                             max_bytes=limits.planner_output_bytes))
        if payload["ok"] and payload.get("done") is not False:
            if spec.writes:
                recent_reads.clear()
            else:
                nonrepeatable.remove(key)
                recent_reads.add(key)
            ledger.record(payload, spec.evidence_paths, action.arguments)
            if payload.get("waiting_for_user") is True and action.name == "sys.ask_missing_info":
                return result("waiting_for_user")
    return result("step_limit")
