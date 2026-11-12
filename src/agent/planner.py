"""Strict project-local protocol between a planner and the bounded runner."""

from dataclasses import dataclass
from typing import Any, Protocol

from ..inference.types import Sample
from ..tooling.json_data import InvalidJSON, decode_json


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Finish:
    reply: str


@dataclass(frozen=True)
class Observation:
    kind: str
    tool: str | None
    content: dict[str, Any]
    arguments: dict[str, Any] | None = None


@dataclass(frozen=True)
class PlannerInput:
    sample: Sample
    tools: list[dict[str, Any]]
    observations: tuple[Observation, ...]


class Planner(Protocol):
    def __call__(self, request: PlannerInput) -> str: ...


def parse_action(output: str, *, max_bytes: int) -> ToolCall | Finish:
    value = decode_json(output, max_bytes=max_bytes)
    if not isinstance(value, dict):
        raise InvalidJSON("Planner output must be one JSON object")
    if (value.get("action") == "call" and set(value) == {"action", "tool", "arguments"}
            and isinstance(value["tool"], str) and value["tool"]
            and isinstance(value["arguments"], dict)):
        return ToolCall(value["tool"], value["arguments"])
    if (value.get("action") == "finish" and set(value) == {"action", "reply"}
            and isinstance(value["reply"], str)):
        return Finish(value["reply"])
    raise InvalidJSON("Planner output does not match the call or finish shape")
