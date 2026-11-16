"""LIFEMATE_FEWSHOT_GATE: no retrieved examples when the best one is not similar, the same at build and inference."""

from copy import deepcopy
import json
from time import monotonic

import pytest

from scripts.build_sft import conversation
from src.agent import fewshot
from src.agent.native_planner import build_messages
from src.agent.planner import PlannerInput
from src.inference.types import Sample

TOOLS = [{"name": "contacts.search", "description": "Tìm", "writes": False,
          "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}]
POOL = [{"id": "train_0001", "transcript": "gọi cho Minh giúp tôi", "apps": ["contacts", "sys"],
         "calls": [{"name": "contacts.search", "arguments": {"query": "Minh"}}], "results": [None]},
        {"id": "train_0002", "transcript": "tìm số của Lan", "apps": ["contacts", "sys"],
         "calls": [{"name": "contacts.search", "arguments": {"query": "Lan"}}], "results": [None]}]


@pytest.fixture
def pool(monkeypatch, tmp_path):
    path = tmp_path / "pool.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in POOL), encoding="utf-8")
    monkeypatch.setattr(fewshot, "NATIVE_POOL", path)
    for name in ("LIFEMATE_FEWSHOT_GATE", "LIFEMATE_FEWSHOT_SPLIT", "LIFEMATE_CONTEXT_V4"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIFEMATE_NATIVE_FEWSHOT_K", "2")
    fewshot.configured.cache_clear()
    yield
    fewshot.configured.cache_clear()


def request(text, sample_id="train_0900"):
    sample = Sample(sample_id, text, frozenset({"contacts", "sys"}),
                    context={"current_time": "2026-11-13T10:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh"})
    return PlannerInput(sample, deepcopy(TOOLS), (), None)


def top_score(text):
    words = fewshot._words(text)
    return max(fewshot._jaccard(words, fewshot._words(row["transcript"])) + 0.2 for row in POOL)


SIMILAR, UNRELATED = "gọi cho Minh giúp tôi nhé", "đặt vé xem phim tối nay"


def test_the_two_requests_sit_on_either_side_of_the_threshold():
    assert top_score(SIMILAR) >= fewshot.GATE_TAU > top_score(UNRELATED)


@pytest.mark.parametrize("flag", [None, "0", "true"])
def test_flag_off_keeps_examples_for_every_request(pool, monkeypatch, flag):
    if flag is not None:
        monkeypatch.setenv("LIFEMATE_FEWSHOT_GATE", flag)
    for text in (SIMILAR, UNRELATED):
        assert len(fewshot.native_examples().select(request(text).sample, {"contacts.search"}, 2)) == 2


def test_flag_on_drops_examples_only_below_the_threshold(pool, monkeypatch):
    names = {"contacts.search"}
    kept = fewshot.native_examples().select(request(SIMILAR).sample, names, 2)
    monkeypatch.setenv("LIFEMATE_FEWSHOT_GATE", "1")
    assert fewshot.native_examples().select(request(SIMILAR).sample, names, 2) == kept
    assert fewshot.native_examples().select(request(UNRELATED).sample, names, 2) == []
    assert fewshot.native_examples().select(request(SIMILAR).sample, set(), 2) == []  # nothing usable at all
    assert "Ví dụ yêu cầu" in build_messages(request(SIMILAR))[0]["content"]
    assert "Ví dụ yêu cầu" not in build_messages(request(UNRELATED))[0]["content"]


@pytest.mark.parametrize("text", [SIMILAR, UNRELATED])
def test_training_rows_and_inference_build_the_same_gated_prompt(pool, monkeypatch, text):
    monkeypatch.setenv("LIFEMATE_FEWSHOT_GATE", "1")
    planned = request(text, "train_0001")  # a TRAIN sample never sees itself, at build or at inference
    row = conversation(planned.sample, planned.tools, [], [])
    assert row["messages"][:2] == build_messages(planned)
