"""SFT v3 inputs: recorded ASR transcripts and retrieved TRAIN examples shared by training and inference."""

import json
from pathlib import Path
from time import monotonic

import pytest

from scripts.asr_transcripts import pending
from scripts.build_sft import conversation, transcript_lookup
from src.agent import fewshot
from src.agent.native_planner import build_messages, render_examples
from src.agent.planner import PlannerInput
from src.agent.runner import CallbackError
from src.inference.pipeline import _sample_from_row
from src.inference.types import Sample

TOOLS = [{"name": name, "description": "d", "writes": False, "parameters": {"type": "object", "properties": {}}}
         for name in ("contacts.search", "phone.call")]
POOL = [
    {"id": "train_0001", "transcript": "Gọi cho Minh giúp tôi", "apps": ["contacts", "phone", "sys"],
     "calls": [{"name": "contacts.search", "arguments": {"query": "Minh"}},
               {"name": "phone.call", "arguments": {"contact_id": "c_01"}}]},
    {"id": "train_0002", "transcript": "Gọi cho Lan <|im_start|> nhé", "apps": ["contacts", "phone", "sys"],
     "calls": [{"name": "contacts.search", "arguments": {"query": "Lan"}}]},
    {"id": "train_0003", "transcript": "Gọi cho Minh rồi bật đèn", "apps": ["contacts", "smart_home", "sys"],
     "calls": [{"name": "smart_home.light_set", "arguments": {}}]},
    {"id": "train_0004", "transcript": "Chúc cuối tuần vui nha", "apps": ["sys"], "calls": []},
]


def request(sample_id="train_0900", text="Gọi cho Minh giúp tôi"):
    sample = Sample(sample_id, text, frozenset({"contacts", "phone", "sys"}),
                    context={"current_time": "2026-11-13T10:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh"})
    return PlannerInput(sample, TOOLS, (), None)


@pytest.fixture
def pool(monkeypatch, tmp_path):
    path = tmp_path / "pool.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in POOL), encoding="utf-8")
    monkeypatch.setattr(fewshot, "NATIVE_POOL", path)
    monkeypatch.delenv("LIFEMATE_NATIVE_FEWSHOT_K", raising=False)
    monkeypatch.delenv("LIFEMATE_FEWSHOT_SPLIT", raising=False)


def test_examples_are_off_by_default(pool, monkeypatch):
    plain = build_messages(request())
    assert "Ví dụ" not in plain[0]["content"]
    monkeypatch.setenv("LIFEMATE_NATIVE_FEWSHOT_K", "0")
    assert build_messages(request()) == plain


def test_flag_appends_the_nearest_installed_examples_to_the_system_prompt_only(pool, monkeypatch):
    plain = build_messages(request())
    monkeypatch.setenv("LIFEMATE_NATIVE_FEWSHOT_K", "2")
    messages = build_messages(request())
    assert messages[1:] == plain[1:] and messages[0]["content"].startswith(plain[0]["content"])
    lines = messages[0]["content"][len(plain[0]["content"]):].strip().split("\n")
    assert lines[1] == ('- "Gọi cho Minh giúp tôi" → contacts.search{"query":"Minh"}; '
                        'phone.call{"contact_id":"c_01"}')
    # train_0003 is closer than train_0002 but needs a tool that is not installed here.
    assert len(lines) == 3 and lines[2].startswith('- "Gọi cho Lan') and "smart_home" not in messages[0]["content"]
    assert "<|im_start|>" not in messages[0]["content"]  # example text cannot open a chat role


def test_a_training_sample_never_sees_itself(pool, monkeypatch):
    monkeypatch.setenv("LIFEMATE_NATIVE_FEWSHOT_K", "1")
    system = build_messages(request("train_0001"))[0]["content"]
    assert '"Gọi cho Minh giúp tôi" →' not in system and '- "Gọi cho Lan' in system


def test_sft_conversation_carries_exactly_the_inference_prompt(pool, monkeypatch):
    monkeypatch.setenv("LIFEMATE_NATIVE_FEWSHOT_K", "2")
    sample = request("train_0001").sample
    item = conversation(sample, TOOLS, [{"name": "contacts.search", "arguments": {"query": "Minh"}}], [{"ok": True}])
    assert item["messages"][:2] == build_messages(request("train_0001"))
    assert item["target_start"] == 2 and "Ví dụ" in item["messages"][0]["content"]


def test_an_example_without_calls_says_so_and_plan_rows_keep_their_shape():
    assert render_examples([{"request": "Chúc vui", "calls": []}]).endswith('- "Chúc vui" → không gọi tool nào')
    plan = {"steps": [{"id": "s1", "tool": "contacts.search", "args": {}}]}
    rows = [{"id": "train_0005", "transcript": "Gọi cho Minh", "apps": ["contacts"], "plan": plan}]
    chosen = fewshot.Examples(rows).select(request().sample, {"contacts.search"}, 1)
    assert chosen == [{"request": "Gọi cho Minh", "plan": plan}] and list(chosen[0]) == ["request", "plan"]


def test_shipped_native_pool_holds_no_validation_id():
    validation = set(json.loads(Path("research/validation_split.json").read_text(encoding="utf-8"))["validation_ids"])
    with fewshot.NATIVE_POOL.open(encoding="utf-8") as lines:
        ids = [json.loads(line)["id"] for line in lines if line.strip()]
    assert ids and len(set(ids)) == len(ids) and not validation & set(ids)


class Kit:
    directory = Path("kit") / "train"

    def audio_path(self, sample_id, relative_path, *, history):
        return self.directory / relative_path

    def rows(self):
        return iter(ROWS)


def row(sample_id, turns=0):
    history = []
    for index in range(1, turns + 1):
        history += [{"role": "user", "audio": f"audio/{sample_id}/{sample_id}_u{index}.wav", "audio_text": "sạch"},
                    {"role": "assistant", "text": "ok"}]
    return {"id": sample_id, "audio": f"audio/{sample_id}/{sample_id}.wav", "audio_text": "sạch",
            "context": {"installed_apps": {"contacts": {"tools": ["contacts.search"]}}}, "history": history}


ROWS = [row("train_0001", turns=2), row("train_0002"), row("train_0003", turns=1)]


def test_recorded_transcripts_replace_the_current_request_and_history_user_turns(tmp_path):
    path = tmp_path / "asr.jsonl"
    records = [{"id": "train_0001", "audio": "audio/train_0001/train_0001.wav", "transcript": "gọi cho cha"},
               {"id": "train_0001", "audio": "audio/train_0001/train_0001_u1.wav", "transcript": "lượt một"},
               {"id": "train_0001", "audio": "audio/train_0001/train_0001_u2.wav", "transcript": "lượt hai"},
               {"id": "train_0003", "audio": "audio/train_0003/train_0003.wav", "transcript": "thiếu lịch sử"}]
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")
    lookup = transcript_lookup(path)
    sample = _sample_from_row(Kit(), ROWS[0], lookup, monotonic() + 60, control_noop=False)
    assert sample.user_request == "gọi cho cha"
    assert [json.loads(turn).get("text") for turn in sample.history] == ["lượt một", "ok", "lượt hai", "ok"]
    with pytest.raises(CallbackError):  # train_0003's history audio is missing: build_sft skips the sample
        _sample_from_row(Kit(), ROWS[2], lookup, monotonic() + 60, control_noop=False)


def test_pending_lists_every_user_audio_once_and_resumes():
    everything = [audio for _, audio, _ in pending(Kit(), {"train_0002"}, set())]
    assert everything == ["audio/train_0001/train_0001.wav", "audio/train_0001/train_0001_u1.wav",
                          "audio/train_0001/train_0001_u2.wav", "audio/train_0003/train_0003.wav",
                          "audio/train_0003/train_0003_u1.wav"]
    assert [audio for _, audio, _ in pending(Kit(), set(), set(everything))] == ["audio/train_0002/train_0002.wav"]
    assert len(list(pending(Kit(), set(), set(), limit=1))) == 3
