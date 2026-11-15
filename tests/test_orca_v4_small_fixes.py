"""LIFEMATE_SMALL_FIXES: a moved calendar event keeps its length; dictated e-mail addresses are joined."""

import json
from pathlib import Path

import pytest

from src.agent.asr_fix import spoken_email, wrapped
from src.agent.runner import moved_event_end, run_sample
from src.inference.types import PriorToolResult, Sample
from src.tooling.evidence import EvidencePath
from src.tooling.registry import ToolRegistry, ToolSpec

EVENTS = {"ok": True, "events": [
    {"id": "evt_03", "title": "đi chơi", "start": "2026-10-02T15:00:00+07:00", "end": "2026-10-02T16:30:00+07:00"},
    {"id": "evt_04", "title": "Họp điều phối kho", "start": "2026-10-20T09:00:00+07:00", "end": "2026-10-20T10:00:00+07:00"}]}
MOVE = {"event_id": "evt_04", "start": "2026-10-22T15:00:00+07:00"}


def test_a_moved_event_gets_the_end_that_keeps_its_old_length():
    assert moved_event_end(MOVE, [EVENTS]) == {**MOVE, "end": "2026-10-22T16:00:00+07:00"}
    assert moved_event_end({**MOVE, "event_id": "evt_03"}, [EVENTS])["end"] == "2026-10-22T16:30:00+07:00"
    later = {"ok": True, "item": [{"id": "evt_04", "start": "2026-10-20T09:00:00+07:00", "end": "2026-10-20T11:00:00+07:00"}]}
    assert moved_event_end(MOVE, [EVENTS, later])["end"] == "2026-10-22T17:00:00+07:00"  # the latest record wins


@pytest.mark.parametrize("arguments,results", [
    ({**MOVE, "end": "2026-10-22T15:30:00+07:00"}, [EVENTS]),  # the model gave an end
    ({"event_id": "evt_04", "title": "Họp"}, [EVENTS]),  # the start does not move
    ({"event_id": "evt_99", "start": MOVE["start"]}, [EVENTS]),  # the old event is not visible
    (MOVE, []),
    ({**MOVE, "start": "9999-12-31T23:59:59+07:00"}, [EVENTS]),  # duration would overflow
    ({"event_id": "evt_04", "start": "ba giờ chiều"}, [EVENTS]),  # not a timestamp
    (MOVE, [{"ok": True, "events": [{"id": "evt_04", "start": "x", "end": "y"}]}]),
])
def test_everything_else_is_left_exactly_as_the_model_wrote_it(arguments, results):
    assert moved_event_end(arguments, results) is arguments


def run(monkeypatch, flag, *, history=False):
    monkeypatch.delenv("LIFEMATE_SMALL_FIXES", raising=False)
    if flag is not None:
        monkeypatch.setenv("LIFEMATE_SMALL_FIXES", flag)
    def schema(properties):
        return {"type": "object", "properties": properties, "additionalProperties": False}
    registry = ToolRegistry([
        ToolSpec("calendar.list_events", "calendar", "Lịch", schema({}), False,
                 (EvidencePath("calendar.events", ("events", "*", "id")),)),
        ToolSpec("calendar.update_event", "calendar", "Sửa", schema({
            "event_id": {"type": "string", "x-ref": "calendar.events"}, "start": {"type": "string"},
            "end": {"type": "string"}}), True)])
    planned = iter(([] if history else [{"action": "call", "tool": "calendar.list_events", "arguments": {}}])
                   + [{"action": "call", "tool": "calendar.update_event", "arguments": MOVE},
                      {"action": "finish", "reply": ""}])
    calls = []

    class Environment:
        def call(self, name, arguments):
            calls.append((name, arguments))
            return EVENTS if name == "calendar.list_events" else {"ok": True}

    prior = (PriorToolResult("calendar.list_events", {}, EVENTS),) if history else ()
    sample = Sample("s", "Dời họp", frozenset({"calendar"}), prior_tool_results=prior)
    run_sample(sample, registry, lambda request: json.dumps(next(planned)), lambda sample: Environment())
    return calls[-1]


@pytest.mark.parametrize("flag", [None, "0"])
def test_flag_off_dispatches_the_call_unchanged(monkeypatch, flag):
    assert run(monkeypatch, flag) == ("calendar.update_event", MOVE)


@pytest.mark.parametrize("history", [False, True])
def test_flag_on_adds_the_end_from_this_turn_or_from_history_results(monkeypatch, history):
    assert run(monkeypatch, "1", history=history) == (
        "calendar.update_event", {**MOVE, "end": "2026-10-22T16:00:00+07:00"})


@pytest.mark.parametrize("heard,written", [
    ("gửi team chấm data a còng example chấm test, báo là xong.", "gửi team.data@example.test, báo là xong."),
    ("sửa thành mai chấm anh mười bốn a còng example chấm test nha", "sửa thành mai.anh14@example.test nha"),
    ("Ghi là mai chấm anh năm mốt, a còng example chấm test nhé.", "Ghi là mai.anh51@example.test nhé."),
    ("gửi Nhóm hai hai A còng Example chấm Test", "gửi nhom22@example.test"),
    ("mai chấm anh năm mốt a còng Example Chấm Test", "mai.anh51@example.test"),
    ("mai Chấm anh năm mốt A Còng Example Chấm Test", "mai.anh51@example.test"),
    ("gửi nhóm bảy tháng ba a còng excelen bồ chẳng tét", "gửi nhóm bảy tháng ba a còng excelen bồ chẳng tét"),
    ("a còng là ký hiệu gì", "a còng là ký hiệu gì"),
    ("chấm điểm giúp tôi rồi gửi family@example.test", "chấm điểm giúp tôi rồi gửi family@example.test"),
])
def test_only_a_complete_dictated_address_is_joined(heard, written):
    assert spoken_email(heard) == written


def test_the_transcriber_wrapper_applies_each_flag_on_its_own(monkeypatch):
    heard = "nhắn zalo cho lan rồi gửi family a còng example chấm test."
    whisper = lambda path: heard
    row = {"context": {}, "history": []}
    monkeypatch.delenv("LIFEMATE_ASR_FIX", raising=False)
    monkeypatch.delenv("LIFEMATE_SMALL_FIXES", raising=False)
    assert wrapped(whisper, row) is whisper
    monkeypatch.setenv("LIFEMATE_SMALL_FIXES", "1")
    assert wrapped(whisper, row)(Path("a.wav")) == "nhắn zalo cho lan rồi gửi family@example.test."
    assert wrapped(None, row) is None
    monkeypatch.setenv("LIFEMATE_ASR_FIX", "1")
    assert wrapped(whisper, row)(Path("a.wav")) == "nhắn Zalo cho Lan rồi gửi family@example.test."
    monkeypatch.delenv("LIFEMATE_SMALL_FIXES")
    assert wrapped(whisper, row)(Path("a.wav")) == "nhắn Zalo cho Lan rồi gửi family a còng example chấm test."
