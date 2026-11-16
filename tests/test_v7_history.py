"""--history-turns: recorded history turns become extra SFT rows built like the final turn (no ToolEnv)."""

from collections import Counter
import json
from pathlib import Path

import pytest

from scripts.build_sft import DONE, NOTHING, _share, history_turn_rows, rename_tool_names
from src.agent import fewshot

SEARCH = {"name": "contacts.search", "description": "Tìm", "writes": False,
          "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"],
                         "additionalProperties": False}}
CALL = {"name": "phone.call", "description": "Gọi", "writes": True,
        "parameters": {"type": "object", "properties": {"contact_id": {"type": "string", "x-ref": "contacts.contacts"}},
                       "required": ["contact_id"], "additionalProperties": False}}
ASK = {"name": "sys.ask_missing_info", "description": "Hỏi", "writes": False,
       "parameters": {"type": "object", "properties": {"tool_name": {"type": "string"},
                                                       "params": {"type": "array", "items": {"type": "string"}}},
                      "additionalProperties": False}}
TOOLS = [SEARCH, CALL, ASK]
FOUND = {"ok": True, "contacts": [{"id": "c_01", "name": "Minh"}]}


def turn_user(index, text):
    return {"role": "user", "audio": f"audio/train_0001/train_0001_u{index}.wav", "audio_text": text}


def row(history=None):
    return {"id": "train_0001", "audio": "audio/train_0001/train_0001.wav", "audio_text": "Gọi cho Minh luôn nhé",
            "context": {"current_time": "2026-11-13T10:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh",
                        "installed_apps": {"contacts": {"tools": ["contacts.search"]}, "phone": {"tools": ["phone.call"]}}},
            "history": history if history is not None else [
                turn_user(1, "Tìm số của Minh"),
                {"role": "assistant", "text": "Đây ạ", "tool_calls": [{"name": "contacts.search", "arguments": {"query": "Minh"}}],
                 "tool_results": [FOUND]},
                turn_user(2, "Cảm ơn em"),
                {"role": "assistant", "text": "Dạ", "tool_calls": [], "tool_results": None}]}


class Kit:
    directory = Path("kit") / "train"
    app_names = {"contacts": "Danh bạ", "phone": "Điện thoại", "sys": "Hệ thống"}

    def audio_path(self, sample_id, relative_path, *, history):
        assert history == ("_u" in relative_path)
        return self.directory / relative_path


def heard(path):
    return "nghe: " + Path(path).stem


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    pool = tmp_path / "pool.jsonl"
    pool.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in [
        {"id": "train_0001", "transcript": "Gọi cho Minh luôn nhé", "apps": ["contacts", "phone", "sys"],
         "calls": [{"name": "phone.call", "arguments": {"contact_id": "c_01"}}], "results": [None]},
        {"id": "train_0002", "transcript": "Tìm số của Lan giúp tôi", "apps": ["contacts", "sys"],
         "calls": [{"name": "contacts.search", "arguments": {"query": "Lan"}}], "results": [None]}]), encoding="utf-8")
    monkeypatch.setattr(fewshot, "NATIVE_POOL", pool)
    for name in ("LIFEMATE_FEWSHOT_GATE", "LIFEMATE_CONTEXT_V4", "LIFEMATE_ASR_FIX", "LIFEMATE_SMALL_FIXES",
                 "LIFEMATE_FEWSHOT_SPLIT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIFEMATE_NATIVE_FEWSHOT_K", "0")
    fewshot.configured.cache_clear()
    yield
    fewshot.configured.cache_clear()


def test_off_adds_nothing():
    totals = Counter()
    assert history_turn_rows(Kit(), row(), heard, TOOLS, 0.0, totals) == [] and not totals


def test_each_turn_keeps_only_earlier_history_and_its_recorded_target():
    totals = Counter()
    first, second = history_turn_rows(Kit(), row(), heard, TOOLS, 1.0, totals)
    assert [first["id"], second["id"]] == ["train_0001_h1", "train_0001_h3"]
    assert totals == Counter(history_rows=2, history_rows_without_calls=1)
    assert [m["role"] for m in first["messages"]] == ["system", "user", "assistant", "tool", "assistant"]
    assert first["target_start"] == 2 and first["messages"][1]["content"] == "nghe: train_0001_u1"
    assert first["messages"][2]["tool_calls"][0]["function"] == {"name": "contacts.search", "arguments": {"query": "Minh"}}
    assert json.loads(first["messages"][3]["content"]) == FOUND and first["messages"][-1]["content"] == DONE
    roles = [m["role"] for m in second["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "user", "assistant"]
    assert second["target_start"] == 6 and second["messages"][5]["content"] == "nghe: train_0001_u2"
    assert second["messages"][1]["content"] == "nghe: train_0001_u1"  # earlier user turns go through ASR as well
    assert second["messages"][-1]["content"] == NOTHING
    assert "Gọi cho Minh" not in json.dumps([first, second], ensure_ascii=False)  # the final turn is never visible
    assert [m["content"] for m in history_turn_rows(Kit(), row(), None, TOOLS, 1.0, Counter())[0]["messages"][1:2]] == [
        "Tìm số của Minh"]  # without --transcripts the released text is the request


def test_selection_is_a_stable_hash_share_of_turns():
    rows = [history_turn_rows(Kit(), row(), heard, TOOLS, 0.5, Counter()) for _ in range(2)]
    assert rows[0] == rows[1]
    expected = [f"train_0001_h{index}" for index in (1, 3) if _share(f"train_0001#history{index}") < 0.5]
    assert [item["id"] for item in rows[0]] == expected


@pytest.mark.parametrize("calls,results", [
    ([{"name": "email.send", "arguments": {}}], [{"ok": True}]),  # tool not installed in this sample
    ([{"name": "contacts.search", "arguments": {}}], [{"ok": True}]),  # fails the schema
    ([{"name": "contacts.search", "arguments": {"query": "Minh"}}], []),  # no recorded result
])
def test_turns_whose_recorded_calls_cannot_be_trusted_are_skipped(calls, results):
    history = [turn_user(1, "Tìm Minh"), {"role": "assistant", "tool_calls": calls, "tool_results": results}]
    totals = Counter()
    assert history_turn_rows(Kit(), row(history), heard, TOOLS, 1.0, totals) == []
    assert totals == Counter(history_skipped_invalid_call=1)


def test_a_user_turn_missing_from_the_transcripts_skips_the_row():
    def partial(path):
        if "_u2" in str(path):
            raise KeyError(path)
        return heard(path)

    totals = Counter()
    assert [item["id"] for item in history_turn_rows(Kit(), row(), partial, TOOLS, 1.0, totals)] == ["train_0001_h1"]
    assert totals["history_skipped_no_transcript"] == 1


def test_examples_exclude_the_source_sample_and_follow_the_gate(monkeypatch):
    monkeypatch.setenv("LIFEMATE_NATIVE_FEWSHOT_K", "2")
    first = history_turn_rows(Kit(), row(), None, TOOLS, 1.0, Counter())[0]
    system = first["messages"][0]["content"]
    assert "Tìm số của Lan giúp tôi" in system and "Gọi cho Minh luôn nhé" not in system
    monkeypatch.setenv("LIFEMATE_FEWSHOT_GATE", "1")
    second = history_turn_rows(Kit(), row(), None, TOOLS, 1.0, Counter())[1]  # "Cảm ơn em": nothing similar
    assert "Ví dụ yêu cầu" not in second["messages"][0]["content"]


def test_renaming_applies_to_the_whole_history_row():
    first = history_turn_rows(Kit(), row(), heard, TOOLS, 1.0, Counter())[0]
    renamed = rename_tool_names(first, 1.0)
    names = [tool["function"]["name"] for tool in renamed["tools"]]
    call = renamed["messages"][2]["tool_calls"][0]["function"]["name"]
    assert call in names and call != "contacts.search" and "contacts.search" not in json.dumps(renamed)
