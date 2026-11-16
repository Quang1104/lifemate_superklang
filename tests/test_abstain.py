"""Opt-in step-1 abstain probe: off by default, records a probability, forces a call above a threshold."""

from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest
import torch

from src.agent.local_model import LocalPlanner, _FirstTokenProbe, _PrefixCache
from src.agent.native_planner import NativePlanner
from src.agent.planner import Observation, PlannerInput
from src.inference.__main__ import _sample_model_timings
from src.inference.types import Sample

NOTHING = "Mình không thực hiện thao tác nào cho yêu cầu này."
CALL = '<tool_call>\n{"name": "contacts.search", "arguments": {"query": "Minh"}}\n</tool_call>'
TOOLS = [{"name": "contacts.search", "description": "Tìm", "writes": False,
          "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}]


def request(observations=()):
    sample = Sample("s", "Gọi cho Minh", frozenset({"contacts"}),
                    context={"current_time": "2026-11-13T10:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh"})
    return PlannerInput(sample, TOOLS, tuple(observations), None)


def scripted(first, call_prob, seen):
    def chat(messages, tools, **options):
        seen.append(options)
        if "probe" in options:
            options["probe"]["call_prob"] = call_prob
        return CALL if options.get("prefix") else first
    return chat


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("LIFEMATE_ABSTAIN_PROBE", raising=False)
    monkeypatch.delenv("LIFEMATE_ABSTAIN_THRESHOLD", raising=False)


def test_off_by_default_even_with_a_threshold_set(monkeypatch):
    monkeypatch.setenv("LIFEMATE_ABSTAIN_THRESHOLD", "0.0")
    seen = []
    action = json.loads(NativePlanner(scripted(NOTHING, 0.99, seen)).for_sample()(request()))
    assert action == {"action": "finish", "reply": ""}
    assert seen == [{"max_tokens": 384, "deadline": None}]  # the exact call made before this feature


def test_probe_alone_records_but_never_changes_the_answer(monkeypatch):
    monkeypatch.setenv("LIFEMATE_ABSTAIN_PROBE", "1")
    seen = []
    action = json.loads(NativePlanner(scripted(NOTHING, 0.99, seen)).for_sample()(request()))
    assert action == {"action": "finish", "reply": ""}
    assert len(seen) == 1 and seen[0]["probe"] == {"call_prob": 0.99}


@pytest.mark.parametrize("call_prob,forced", [(0.31, True), (0.30, True), (0.29, False)])
def test_threshold_forces_one_tool_call_only_when_the_call_was_likely_enough(monkeypatch, call_prob, forced):
    monkeypatch.setenv("LIFEMATE_ABSTAIN_PROBE", "1")
    monkeypatch.setenv("LIFEMATE_ABSTAIN_THRESHOLD", "0.3")
    seen = []
    action = json.loads(NativePlanner(scripted(NOTHING, call_prob, seen)).for_sample()(request()))
    assert (action["action"] == "call") is forced
    assert [options.get("prefix") for options in seen] == ([None, "<tool_call>\n"] if forced else [None])


def test_a_model_that_already_calls_a_tool_and_later_steps_are_left_alone(monkeypatch):
    monkeypatch.setenv("LIFEMATE_ABSTAIN_PROBE", "1")
    monkeypatch.setenv("LIFEMATE_ABSTAIN_THRESHOLD", "0.0")
    seen = []
    assert json.loads(NativePlanner(scripted(CALL, 0.9, seen)).for_sample()(request()))["action"] == "call"
    assert len(seen) == 1
    done = [Observation("untrusted_tool_result", "contacts.search", {"ok": True}, {"query": "Minh"})]
    seen.clear()
    assert json.loads(NativePlanner(scripted(NOTHING, 0.9, seen)).for_sample()(request(done)))["action"] == "finish"
    assert seen == [{"max_tokens": 384, "deadline": None}]  # finishing after an action is never second-guessed


def test_a_timeout_in_the_forced_retry_keeps_the_first_answer(monkeypatch):
    monkeypatch.setenv("LIFEMATE_ABSTAIN_PROBE", "1")
    monkeypatch.setenv("LIFEMATE_ABSTAIN_THRESHOLD", "0.0")

    def chat(messages, tools, **options):
        if options.get("prefix"):
            raise TimeoutError
        options["probe"]["call_prob"] = 0.9
        return NOTHING

    assert json.loads(NativePlanner(chat).for_sample()(request())) == {"action": "finish", "reply": ""}


def test_first_token_probe_reads_only_the_first_step_and_passes_scores_through():
    probe = _FirstTokenProbe(2)
    first = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float16)
    assert probe(None, first) is first and probe(None, torch.tensor([[9.0, 0.0, 0.0]])) is not None
    assert probe.probability == pytest.approx(torch.tensor([0.0, 0.0, 1.0]).softmax(-1)[2].item(), abs=1e-3)


class _Tokenizer:
    all_special_tokens = []
    CALL_ID = 7

    def apply_chat_template(self, messages, **kwargs):
        ids = torch.tensor([[1, 3, 4, 2]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def __call__(self, text, **kwargs):
        assert text == "<tool_call>\n" and kwargs["add_special_tokens"] is False
        return {"input_ids": torch.tensor([[self.CALL_ID, 5]])}

    def convert_tokens_to_ids(self, token):
        return self.CALL_ID

    def convert_ids_to_tokens(self, token_id):
        return "<tool_call>"

    def decode(self, tokens, **kwargs):
        return " " + ",".join(str(int(token)) for token in tokens) + " "


class _Model:
    device, dtype = "cpu", "float32"

    def generate(self, input_ids, attention_mask, **options):
        self.prompt = input_ids[0].tolist()
        assert attention_mask.shape == input_ids.shape and bool(attention_mask.all())
        scores = torch.zeros((1, 10))
        scores[0, _Tokenizer.CALL_ID] = 2.0
        for processor in options.get("logits_processor", ()):
            scores = processor(input_ids, scores)
        return torch.cat([input_ids, torch.tensor([[8, 9]])], dim=-1)


class _Batch(dict):
    def to(self, device):
        return self


def local_planner():
    instance = LocalPlanner.__new__(LocalPlanner)
    instance._torch = SimpleNamespace(inference_mode=nullcontext, cat=torch.cat, ones_like=torch.ones_like)
    instance._device = SimpleNamespace(type="cpu")
    instance._tokenizer = _Tokenizer()
    instance._tokenizer.apply_chat_template = lambda messages, **kwargs: _Batch(
        _Tokenizer.apply_chat_template(instance._tokenizer, messages))
    instance._model = _Model()
    instance._context_tokens = 4096
    instance._reuse_prefix_cache = False
    instance._prefix_cache = _PrefixCache()
    instance._grammars = {}
    return instance


def test_local_chat_reports_the_call_probability_without_changing_the_text():
    plain, probed, probe = local_planner(), local_planner(), {}
    messages = [{"role": "user", "content": "x"}]
    assert probed.chat(messages, [], probe=probe) == plain.chat(messages, []) == "8,9"
    expected = torch.zeros(10).index_fill(0, torch.tensor(_Tokenizer.CALL_ID), 2.0).softmax(-1)[_Tokenizer.CALL_ID]
    assert probe["call_prob"] == pytest.approx(expected.item())
    assert "call_prob" not in plain.generation_metrics[0] and "forced_prefix" not in plain.generation_metrics[0]
    assert probed.generation_metrics[0]["call_prob"] == probe["call_prob"]


def test_local_chat_prefix_extends_the_prompt_and_is_returned_with_the_text():
    planner = local_planner()
    assert planner.chat([{"role": "user", "content": "x"}], [], prefix="<tool_call>\n") == "<tool_call>\n8,9"
    assert planner._model.prompt == [1, 3, 4, 2, _Tokenizer.CALL_ID, 5]
    row = planner.generation_metrics[0]
    assert row["forced_prefix"] is True and row["prompt_tokens"] == 6 and row["generated_tokens"] == 2


def _snapshot(rows):
    base = {"prompt_tokens": 1, "generated_tokens": 1, "reused_prefix_tokens": 0, "grammar_compile_ms": 0.0,
            "generate_ms": 1.0, "grammar_status": "unconstrained"}
    return SimpleNamespace(timing_snapshot=lambda: [{**base, **row} for row in rows])


def test_train_report_gains_the_probability_only_when_it_was_probed():
    assert not {"abstain_call_prob", "abstain_forced"} & set(_sample_model_timings(_snapshot([{}, {}]), None))
    logged = _sample_model_timings(_snapshot([{"call_prob": 0.4}, {}]), None)
    assert (logged["abstain_call_prob"], logged["abstain_forced"]) == (0.4, False)
    forced = _sample_model_timings(_snapshot([{"call_prob": 0.4}, {"forced_prefix": True}]), None)
    assert (forced["abstain_call_prob"], forced["abstain_forced"]) == (0.4, True)
