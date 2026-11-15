"""A name collision alone must not override a planner choice grounded in other record fields."""

import json

import pytest

from src.agent.runner import run_sample, same_name_ask
from src.inference.types import Sample
from src.tooling.evidence import EvidencePath
from src.tooling.registry import ToolRegistry, ToolSpec


def schema(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


CALL = schema({"contact_id": {"type": "string", "x-ref": "contacts.contacts"}, "note": {"type": "string"}},
              ["contact_id"])
TWO_LANS = {"ok": True, "contacts": [{"id": "c_05", "name": "Lan", "relation": "vợ"},
                                     {"id": "c_10", "name": "lan", "relation": "đồng nghiệp"},
                                     {"id": "c_11", "name": "Minh"}]}
ASK = {"tool_name": "phone.call", "params": ["contact_id"]}


def test_only_an_id_that_shares_its_name_in_this_turns_results_is_replaced():
    assert same_name_ask("phone.call", {"contact_id": "c_05"}, CALL, [TWO_LANS]) == ASK
    assert same_name_ask("phone.call", {"contact_id": "c_10", "note": "c_05"}, CALL, [TWO_LANS]) == ASK
    assert same_name_ask("phone.call", {"contact_id": "c_11"}, CALL, [TWO_LANS]) is None  # the only Minh
    assert same_name_ask("phone.call", {"contact_id": "c_99"}, CALL, [TWO_LANS]) is None  # not from these results
    assert same_name_ask("phone.call", {"note": "c_05"}, CALL, [TWO_LANS]) is None  # not an ID parameter
    assert same_name_ask("phone.call", {"contact_id": "c_05"}, CALL, []) is None
    unnamed = {"ok": True, "items": [{"id": "a"}, {"id": "b"}]}
    assert same_name_ask("phone.call", {"contact_id": "a"}, CALL, [unnamed]) is None
    titled = {"ok": True, "data": {"events": [{"id": "e1", "title": "Họp"}, {"id": "e2", "title": "họp"}]}}
    assert same_name_ask("phone.call", {"contact_id": "e2"}, CALL, [titled]) == ASK


def registry():
    path = (EvidencePath("contacts.contacts", ("contacts", "*", "id")),)
    return ToolRegistry([
        ToolSpec("contacts.search", "contacts", "Tìm", schema({"query": {"type": "string"}}), False, path),
        ToolSpec("phone.call", "phone", "Gọi", CALL, True),
        ToolSpec("sys.ask_missing_info", "sys", "Hỏi", schema({"tool_name": {"type": "string"},
                 "params": {"type": "array", "items": {"type": "string"}}}, ["tool_name", "params"]), False),
    ])


def run(monkeypatch, flag, apps=("contacts", "phone", "sys"), text="Gọi cho Lan"):
    monkeypatch.delenv("LIFEMATE_DUP_GUARD", raising=False)
    if flag is not None:
        monkeypatch.setenv("LIFEMATE_DUP_GUARD", flag)
    planned = iter([{"action": "call", "tool": "contacts.search", "arguments": {"query": "Lan"}},
                    {"action": "call", "tool": "phone.call", "arguments": {"contact_id": "c_05"}},
                    {"action": "finish", "reply": ""}])
    calls = []

    class Environment:
        def call(self, name, arguments):
            calls.append((name, arguments))
            if name == "contacts.search":
                return TWO_LANS
            return {"ok": True, "waiting_for_user": True} if name == "sys.ask_missing_info" else {"ok": True}

    sample = Sample("s", text, frozenset(apps))
    result = run_sample(sample, registry(), lambda request: json.dumps(next(planned)), lambda sample: Environment())
    return result, calls


@pytest.mark.parametrize("flag", [None, "0", "true"])
def test_off_by_default_the_model_choice_is_dispatched(monkeypatch, flag):
    result, calls = run(monkeypatch, flag)
    assert calls == [("contacts.search", {"query": "Lan"}), ("phone.call", {"contact_id": "c_05"})]
    assert result.status == "finished"


def test_flag_on_preserves_a_choice_disambiguated_by_relation(monkeypatch):
    result, calls = run(monkeypatch, "1", text="Gọi cho Lan, vợ tôi")
    assert calls == [("contacts.search", {"query": "Lan"}), ("phone.call", {"contact_id": "c_05"})]
    assert result.status == "finished" and result.rejections == 0


def test_flag_on_without_the_clarification_tool_changes_nothing(monkeypatch):
    result, calls = run(monkeypatch, "1", apps=("contacts", "phone"))
    assert calls[-1] == ("phone.call", {"contact_id": "c_05"}) and result.status == "finished"
