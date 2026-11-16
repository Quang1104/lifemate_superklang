"""LIFEMATE_CONTACT_RETRY: a contact search that found nobody is repeated without the leading title."""

import json

import pytest

from src.agent.runner import name_only_search, run_sample
from src.inference.types import Sample
from src.tooling.evidence import EvidencePath
from src.tooling.registry import ToolRegistry, ToolSpec

NOBODY = {"ok": True, "contacts": []}
TUAN = {"ok": True, "contacts": [{"id": "c_04", "name": "Tuấn"}]}


@pytest.mark.parametrize("query,again", [
    ("Anh Tuấn", "Tuấn"), ("anh Tuấn", "Tuấn"), ("Sếp  Dũng", "Dũng"), ("chị Mai Anh", "Mai Anh"), ("Thầy Phong", "Phong"),
    ("Tuấn", None), ("Trần Văn Tuấn", None), ("anh", None), ("Mai Anh", None), ("", None),
])
def test_only_a_title_followed_by_a_name_is_shortened(query, again):
    expected = None if again is None else {"query": again}
    assert name_only_search({"query": query}, NOBODY) == expected


def test_nothing_is_retried_when_the_search_found_someone_or_failed():
    assert name_only_search({"query": "Anh Tuấn"}, TUAN) is None
    assert name_only_search({"query": "Anh Tuấn"}, {"ok": False, "contacts": []}) is None
    assert name_only_search({"query": "Anh Tuấn"}, {"ok": True}) is None
    assert name_only_search({}, NOBODY) is None


def run(monkeypatch, flag, *, first=NOBODY, query="Anh Tuấn"):
    monkeypatch.delenv("LIFEMATE_CONTACT_RETRY", raising=False)
    if flag is not None:
        monkeypatch.setenv("LIFEMATE_CONTACT_RETRY", flag)

    def schema(properties):
        return {"type": "object", "properties": properties, "additionalProperties": False}

    registry = ToolRegistry([
        ToolSpec("contacts.search", "contacts", "Tìm", schema({"query": {"type": "string"}}), False,
                 (EvidencePath("contacts.contacts", ("contacts", "*", "id")),)),
        ToolSpec("sms.send", "sms", "Gửi", schema({"contact_id": {"type": "string", "x-ref": "contacts.contacts"}}), True)])
    seen, calls = [], []

    def planner(request):
        seen.append([item.arguments for item in request.observations])
        if not request.observations:
            return json.dumps({"action": "call", "tool": "contacts.search", "arguments": {"query": query}})
        found = [item for item in request.observations if item.content.get("contacts")]
        if found and not any(item.tool == "sms.send" for item in request.observations):
            return json.dumps({"action": "call", "tool": "sms.send", "arguments": {"contact_id": "c_04"}})
        return json.dumps({"action": "finish", "reply": ""})

    class Environment:
        def call(self, name, arguments):
            calls.append((name, arguments))
            if name == "contacts.search":
                return first if arguments["query"] == query else TUAN
            return {"ok": True}

    sample = Sample("s", "Nhắn anh Tuấn", frozenset({"contacts", "sms"}))
    result = run_sample(sample, registry, planner, lambda sample: Environment())
    return result, calls, seen


@pytest.mark.parametrize("flag", [None, "0", "true"])
def test_flag_off_the_planner_is_asked_right_after_the_empty_search(monkeypatch, flag):
    result, calls, seen = run(monkeypatch, flag)
    assert calls == [("contacts.search", {"query": "Anh Tuấn"})]
    assert result.status == "finished" and len(seen) == 2


def test_flag_on_searches_the_name_alone_then_lets_the_planner_continue(monkeypatch):
    result, calls, seen = run(monkeypatch, "1")
    assert calls == [("contacts.search", {"query": "Anh Tuấn"}), ("contacts.search", {"query": "Tuấn"}),
                     ("sms.send", {"contact_id": "c_04"})]
    assert result.status == "finished" and result.rejections == 0 and result.dispatched_calls == 3
    assert seen[1] == [{"query": "Anh Tuấn"}, {"query": "Tuấn"}]  # the planner sees both searches and their results


def test_flag_on_does_not_retry_when_results_exist_or_the_query_has_no_title(monkeypatch):
    _, calls, _ = run(monkeypatch, "1", first=TUAN)
    assert calls == [("contacts.search", {"query": "Anh Tuấn"}), ("sms.send", {"contact_id": "c_04"})]
    _, calls, _ = run(monkeypatch, "1", query="Tuấn Anh")
    assert calls == [("contacts.search", {"query": "Tuấn Anh"})]
