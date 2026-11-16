"""LIFEMATE_EMBED_RETRIEVAL: examples ranked by sentence-embedding cosine; filters, gate, budget and format unchanged."""

from copy import deepcopy
import json
from pathlib import Path
import re

import numpy
import pytest

from scripts.build_sft import conversation
from src.agent import embed_retrieval, fewshot
from src.agent.native_planner import build_messages
from src.agent.planner import PlannerInput
from src.inference.types import Sample

ROOT = Path(__file__).resolve().parents[1]
TOOLS = [{"name": name, "description": "d", "writes": False, "parameters": {"type": "object", "properties": {}}}
         for name in ("contacts.search", "phone.call")]
POOL = [
    {"id": "train_0001", "transcript": "gọi cho Minh giúp tôi", "apps": ["contacts", "phone", "sys"],
     "calls": [{"name": "phone.call", "arguments": {"contact_id": "c1"}}], "results": [None]},
    {"id": "train_0002", "transcript": "alo cho Lan cái", "apps": ["contacts", "phone", "sys"],
     "calls": [{"name": "contacts.search", "arguments": {"query": "Lan"}}], "results": [None]},
    {"id": "train_0003", "transcript": "gọi cho Minh rồi bật đèn giúp tôi", "apps": ["contacts", "smart_home", "sys"],
     "calls": [{"name": "smart_home.light_set", "arguments": {}}], "results": [None]},
    {"id": "train_0004", "transcript": "alo cho Lan cái", "apps": ["contacts", "sys"],
     "calls": [{"name": "contacts.search", "arguments": {"query": "Lan"}}], "results": [None]},
]
# Meaning as the fake embedder sees it: "điện cho Minh" is the same request as "alo cho Lan cái", in other words.
MEANING = {"điện cho Minh nhé": (1.0, 0.0), "alo cho Lan cái": (0.99, 0.14), "gọi cho Minh giúp tôi": (0.6, 0.8),
           "gọi cho Minh rồi bật đèn giúp tôi": (0.0, 1.0), "đặt vé xem phim tối nay": (0.0, 1.0)}


@pytest.fixture
def pool(monkeypatch, tmp_path):
    path = tmp_path / "pool.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in POOL), encoding="utf-8")
    monkeypatch.setattr(fewshot, "NATIVE_POOL", path)
    calls = []

    def fake(texts):
        calls.append(list(texts))
        return numpy.array([MEANING[text] for text in texts], dtype="float32")

    monkeypatch.setattr(fewshot, "_embed", fake)
    for name in ("LIFEMATE_EMBED_RETRIEVAL", "LIFEMATE_FEWSHOT_GATE", "LIFEMATE_FEWSHOT_SPLIT", "LIFEMATE_CONTEXT_V4"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIFEMATE_NATIVE_FEWSHOT_K", "2")
    fewshot.configured.cache_clear()
    yield calls
    fewshot.configured.cache_clear()


def request(text="điện cho Minh nhé", sample_id="train_0900"):
    sample = Sample(sample_id, text, frozenset({"contacts", "phone", "sys"}),
                    context={"current_time": "2026-11-13T10:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh"})
    return PlannerInput(sample, deepcopy(TOOLS), (), None)


def chosen(planned):
    return [item["request"] for item in fewshot.native_examples().select(
        planned.sample, {tool["name"] for tool in planned.tools}, 2)]


@pytest.mark.parametrize("flag", [None, "0", "true"])
def test_flag_off_ranks_by_word_overlap_and_never_embeds(pool, monkeypatch, flag):
    if flag is not None:
        monkeypatch.setenv("LIFEMATE_EMBED_RETRIEVAL", flag)
    assert chosen(request()) == ["gọi cho Minh giúp tôi", "alo cho Lan cái"] and pool == []


def test_flag_on_ranks_the_same_candidates_by_cosine(pool, monkeypatch):
    monkeypatch.setenv("LIFEMATE_EMBED_RETRIEVAL", "1")
    assert chosen(request()) == ["alo cho Lan cái", "gọi cho Minh giúp tôi"]  # same K, duplicate transcript once
    assert "gọi cho Minh rồi bật đèn giúp tôi" not in chosen(request())  # its tool is not installed
    assert chosen(request(sample_id="train_0002")) == ["alo cho Lan cái", "gọi cho Minh giúp tôi"]  # 0004, not itself
    # The pool is embedded once and the request once, however many planner steps ask for examples.
    assert [len(texts) for texts in pool] == [4, 1]


def test_only_the_order_of_examples_changes_in_the_prompt(pool, monkeypatch):
    plain = build_messages(request())
    monkeypatch.setenv("LIFEMATE_EMBED_RETRIEVAL", "1")
    embedded = build_messages(request())
    lines = lambda messages: messages[0]["content"].split("\n")
    assert embedded[1:] == plain[1:] and sorted(lines(embedded)) == sorted(lines(plain)) and embedded != plain
    row = conversation(request().sample, request().tools, [], [])
    assert row["messages"][:2] == embedded  # training rows would be built by the same code


def test_the_gate_still_decides_on_word_overlap(pool, monkeypatch):
    monkeypatch.setenv("LIFEMATE_EMBED_RETRIEVAL", "1")
    monkeypatch.setenv("LIFEMATE_FEWSHOT_GATE", "1")
    assert chosen(request("đặt vé xem phim tối nay")) == [] and pool == []  # gated before any embedding
    assert chosen(request("gọi cho Minh giúp tôi")) != []


def test_warm_loads_once_embeds_the_pool_and_reports_parameters(pool, monkeypatch):
    class Fake:
        parameter_count = 123

        def __init__(self, directory, device):
            self.where = (str(directory), device)

        def __call__(self, texts):
            return fewshot._embed(texts)

    monkeypatch.setattr(embed_retrieval, "Embedder", Fake)
    monkeypatch.setattr(embed_retrieval, "_loaded", None)
    monkeypatch.setenv("LIFEMATE_EMBEDDER_DIR", "some/dir")
    assert embed_retrieval.warm("cpu") == 123 and embed_retrieval.embedder().where == (str(Path("some/dir")), "cpu")
    assert [len(texts) for texts in pool] == [4]


def test_embedder_is_pinned_fetchable_and_wired_for_release():
    pin = json.loads((ROOT / "configs" / "colab_models.json").read_text(encoding="utf-8"))["embedder"]
    assert pin["repo_id"] == "intfloat/multilingual-e5-small" and re.fullmatch(r"[0-9a-f]{40}", pin["revision"])
    colab = (ROOT / "scripts" / "fetch_models.py").read_text(encoding="utf-8")
    docker = (ROOT / "docker" / "fetch_models.py").read_text(encoding="utf-8")
    for text in (colab, docker):
        assert '"model.safetensors"' in text and "EMBEDDER_FILES" in text
    assert '"embedder")' in colab and 'checked_pin(pins, "embedder")' in docker
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "fetch_models.py --embedder /models/embedder" in dockerfile
    assert dockerfile.index("fetch_models.py /tmp/adapter") < dockerfile.index("ARG LIFEMATE_EMBED_RETRIEVAL=")
    api = (ROOT / "src" / "inference" / "api.py").read_text(encoding="utf-8")
    assert api.index('parameter_counts["embedder"] = warm(device)') < api.index("six billion")
    script = (ROOT / "scripts" / "colab_val_embed.sh").read_text(encoding="utf-8")
    assert "LIFEMATE_EMBED_RETRIEVAL=1" in script and "colab_val.sh" in script and "v7_embed" in script
