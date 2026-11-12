"""Deterministic offline exercise of the same planner and tool loop."""

import json
from typing import Any

from ..agent.planner import PlannerInput
from ..agent.runner import run_sample
from ..tooling.evidence import EvidencePath
from ..tooling.registry import ToolRegistry, ToolSpec
from .types import Sample


_TOOLS = (
    ToolSpec("mock_notes.list", "mock_notes", "List synthetic notes",
             {"type": "object", "properties": {}, "additionalProperties": False}, False,
             (EvidencePath("mock_notes.notes", ("notes", "*", "id")),)),
    ToolSpec("mock_notes.append", "mock_notes", "Append to a synthetic note", {
        "type": "object", "additionalProperties": False, "required": ["note_id", "text"],
        "properties": {"note_id": {"type": "string", "x-ref": "mock_notes.notes"},
                       "text": {"type": "string", "minLength": 1}},
    }, True),
)


class MockNotes:
    def __init__(self, note_id: str):
        self.note_id = note_id
        self.text = "Original text"
        self.writes = 0

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "mock_notes.list":
            return {"ok": True, "notes": [{"id": self.note_id, "text": self.text}]}
        if name == "mock_notes.append" and arguments["note_id"] == self.note_id:
            self.text += arguments["text"]
            self.writes += 1
            return {"ok": True, "updated": True}
        return {"ok": False, "error": "Unexpected synthetic tool or identifier"}


def _mock_planner(request: PlannerInput) -> str:
    observations = request.observations
    mode = request.sample.user_request
    if mode == "No applicable action" or (observations and mode in {"Uninstalled tool", "Unknown ID"}):
        return '{"action":"finish","reply":"No action taken"}'
    if mode == "Uninstalled tool":
        return '{"action":"call","tool":"mock_notes.list","arguments":{}}'
    if mode == "Unknown ID":
        return ('{"action":"call","tool":"mock_notes.append",'
                '"arguments":{"note_id":"unobserved","text":" extra"}}')
    if not observations:
        return '{"action":"call","tool":"mock_notes.list","arguments":{}}'
    if len(observations) == 1 or (len(observations) == 2 and mode == "Duplicate attempt"):
        note_id = observations[0].content["notes"][0]["id"]
        return json.dumps({"action": "call", "tool": "mock_notes.append",
                           "arguments": {"note_id": note_id, "text": " extra"}})
    return '{"action":"finish","reply":"Done"}'


def run_mock(count: int) -> dict[str, Any]:
    if not 1 <= count <= 1_000:
        raise ValueError("Mock sample count must be in 1..1000")
    registry = ToolRegistry(_TOOLS)
    modes = ("Append", "Duplicate attempt", "Uninstalled tool", "Unknown ID", "No applicable action")
    verified = 0
    rejections = 0
    calls = 0
    for index in range(count):
        mode = modes[index % len(modes)]
        apps = frozenset() if mode == "Uninstalled tool" else frozenset({"mock_notes"})
        sample_id = f"synthetic-{index + 1:04d}"
        sample = Sample(sample_id, mode, apps, ("Previous tool actions were already applied",))
        environment = MockNotes(f"synthetic-note-{index + 1:04d}")
        result = run_sample(sample, registry, _mock_planner, lambda _: environment)
        expected_writes = int(mode in {"Append", "Duplicate attempt"})
        expected_rejections = int(mode in {"Duplicate attempt", "Uninstalled tool", "Unknown ID"})
        verified += (result.status == "finished" and environment.writes == expected_writes
                     and result.rejections == expected_rejections and result.dispatched_calls == 2 * expected_writes)
        calls += result.dispatched_calls
        rejections += result.rejections
    return {"mode": "synthetic_mock", "samples": count, "checks_passed": verified,
            "dispatched_calls": calls, "guard_rejections": rejections,
            "competition_score": "NOT MEASURED"}
