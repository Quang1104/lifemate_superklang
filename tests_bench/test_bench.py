"""CPU-only checks for the Colab benchmark/SFT modules. Kit tests need LIFEMATE_DATA=<kit root>."""

import json
import os
from pathlib import Path

import pytest

from src.agent.local_model import RESULT_CHARS, build_messages, extract_json
from src.agent.planner import Observation, PlannerInput, parse_action
from src.bench import colab_setup, evaluate, finetune, sft_data
from src.inference.types import Sample

DATA = os.environ.get("LIFEMATE_DATA")
needs_kit = pytest.mark.skipif(not DATA, reason="set LIFEMATE_DATA to the official kit root")
NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


def test_messages_shape_and_truncation():
    sample = Sample("s1", "bật đèn", frozenset({"sys"}), context={"timezone": "Asia/Ho_Chi_Minh"})
    tool = {"name": "a.b", "description": "d", "parameters": {"type": "object"}, "writes": True}
    seen = (Observation("untrusted_tool_result", "a.b", {"ok": True, "x": "y" * 9000}, {"q": 1}),
            Observation("rejected", None, {"reason": "bad"}))
    messages = build_messages(PlannerInput(sample, [tool], seen))
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "user"]
    assert "writes" not in messages[1]["content"] and "bật đèn" in messages[1]["content"]
    assert len(messages[3]["content"]) <= RESULT_CHARS + 8 and messages[4]["content"].startswith("REJECTED")
    parse_action(messages[2]["content"], max_bytes=8192)


def test_extract_json():
    assert extract_json('```json\n{"action":"finish","reply":""}\n```') == '{"action":"finish","reply":""}'
    assert extract_json('<think>{x}</think>\n{"a":{"b":1}}') == '{"a":{"b":1}}'
    assert extract_json("no json") == "no json"


def test_wer():
    assert evaluate.wer("Bật đèn, phòng khách!", "bật đèn phòng khách") == 0
    assert evaluate.wer("a b c d", "a x c") == 0.5


def test_kit_prefix():
    assert colab_setup.kit_prefix(["k/train/train.jsonl", "k/public_test/x", "__MACOSX/k/train/._train.jsonl"]) == "k/"
    assert colab_setup.kit_prefix(["train/train.jsonl"]) == ""
    with pytest.raises(ValueError):
        colab_setup.kit_prefix(["a/train/train.jsonl", "b/train/train.jsonl"])


def test_classify():
    gold = {"tool_calls": [{"name": "a.x", "arguments": {}}]}
    detail = {"score": 0.0, "gold_empty": False, "matched": 0}
    assert evaluate.classify(detail, gold, {"tool_calls": []}) == "no_calls"
    assert evaluate.classify(detail, gold, {"tool_calls": [{"name": "a.y", "arguments": {}}]}) == "tool_selection"
    assert evaluate.classify(detail, gold, {"tool_calls": [{"name": "a.x", "arguments": {"k": 1}}]}) == "arguments"
    assert evaluate.classify({"failure_override": "timeout"}, gold, {"tool_calls": []}) == "timeout"


def test_tokenize_masks_prompt_and_drops_long():
    class Tok:
        def convert_tokens_to_ids(self, token):
            return 99

        def apply_chat_template(self, messages, **_):
            return [1] * len(messages[-1]["content"])

        def __call__(self, text, add_special_tokens):
            return {"input_ids": [7] * len(text)}

    examples = [{"messages": [{"role": "user", "content": "abc"}], "completion": "xy"},
                {"messages": [{"role": "user", "content": "a" * 50}], "completion": "xy"}]
    kept, dropped = finetune.tokenize_examples(Tok(), examples, max_len=10)
    assert dropped == 1 and kept == [{"input_ids": [1, 1, 1, 7, 7, 99], "labels": [-100] * 3 + [7, 7, 99]}]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_cells_compile(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(line for line in "".join(cell["source"]).splitlines(True)
                             if not line.lstrip().startswith(("!", "%")))
            compile(source, path.name, "exec")


@needs_kit
def test_oracle_planner_scores_one(tmp_path):
    """Replaying gold actions through the real runner + ToolEnv must score 1.0: checks the whole bench path."""
    ids = evaluate.select_ids(limit=4)
    gold_path = evaluate.ensure_gold(DATA, tmp_path / "gold.jsonl")
    gold = {r["id"]: r["tool_calls"] for r in map(json.loads, gold_path.read_text(encoding="utf-8").splitlines())}

    def oracle(request):
        calls, done = gold[request.sample.sample_id], len(request.observations)
        if done < len(calls):
            return json.dumps({"action": "call", "tool": calls[done]["name"], "arguments": calls[done]["arguments"]})
        return '{"action":"finish","reply":""}'

    summary = evaluate.run(DATA, ids, gold_path, oracle, tmp_path / "run", verbose=False)
    assert summary["samples"] == 4 and summary["score_official"] == 1.0, summary
    assert evaluate.run(DATA, ids, gold_path, oracle, tmp_path / "run", verbose=False) == summary  # resume


@needs_kit
def test_sft_examples(tmp_path):
    counts = sft_data.build(DATA, tmp_path / "sft.jsonl", limit=3)
    rows = [json.loads(line) for line in (tmp_path / "sft.jsonl").read_text(encoding="utf-8").splitlines()]
    assert counts["samples"] == 3 and len(rows) == counts["examples"] > 3
    validation = set(json.loads(evaluate.VALIDATION.read_text(encoding="utf-8"))["validation_ids"])
    assert not {r["id"] for r in rows} & validation
    for row in rows:
        parse_action(row["completion"], max_bytes=8192)
        assert row["messages"][0]["role"] == "system"
