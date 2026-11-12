"""Run one sample without replaying context or repeating tool attempts."""

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from queue import Empty, Queue
from threading import Thread
from time import monotonic
from typing import Any, Callable, Protocol, TypeVar

from ..inference.types import Sample
from ..tooling.evidence import EvidenceLedger
from ..tooling.json_data import InvalidJSON, decode_json, encode_json
from ..tooling.registry import ToolRegistry, ToolRejected
from .planner import Finish, Observation, Planner, PlannerInput, parse_action


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

    def result(status: str, reply: str | None = None, *, safe: bool = True) -> RunResult:
        return RunResult(status, reply, step, dispatched_calls, rejections, safe)

    for step in range(1, limits.max_steps + 1):
        if monotonic() >= deadline:
            return result("sample_timeout")
        request = PlannerInput(deepcopy(sample), deepcopy(tools), tuple(deepcopy(observations)))
        try:
            action = parse_action(_bounded_call(lambda: planner(request), remaining(limits.planner_seconds)),
                                  max_bytes=limits.planner_output_bytes)
        except CallTimeout:
            return result("planner_timeout", safe=False)
        except CallbackError:
            return result("planner_error")
        except InvalidJSON:
            observations.append(Observation("rejected", None, {"reason": "Invalid planner JSON action"}))
            rejections += 1
            continue
        if isinstance(action, Finish):
            return result("finished", action.reply)

        try:
            spec = registry.authorize(action.name, action.arguments, sample.installed_apps,
                                      ledger, sample.allowed_tools)
            key = (action.name, encode_json(action.arguments, max_bytes=limits.planner_output_bytes))
            if key in nonrepeatable or key in recent_reads:
                raise ToolRejected("Identical action already attempted in this sample")
        except (ToolRejected, InvalidJSON) as exc:
            observations.append(Observation("rejected", None, {"reason": str(exc)}))
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
                                                {"reason": "Environment preflight rejected the action"}))
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
