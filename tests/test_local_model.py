"""Exercise model boundaries without downloading or loading real checkpoints."""

from contextlib import nullcontext
import json
from pathlib import Path
import struct
import sys
from time import monotonic
from types import SimpleNamespace
import wave

import pytest

from src.agent.local_model import LocalPlanner, LocalWhisper, _GENERATING, _artifact_tensor_count
from src.agent.planner import Observation, PlannerInput
from src.inference.types import Sample


class FakeTensor(list):
    @property
    def shape(self):
        return (len(self), len(self[0]))

    def to(self, *args, **kwargs):
        return self


class FakeBatch(dict):
    def to(self, *args, **kwargs):
        return self


class FakeTokenizer:
    all_special_tokens = ["<|im_start|>", "<|im_end|>"]

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_options = kwargs
        return FakeBatch(input_ids=FakeTensor([[1, 2, 3]]))

    def decode(self, tokens, **kwargs):
        assert tokens == [4, 5]
        return ' {"action":"finish","reply":""} '


@pytest.fixture
def fake_models(monkeypatch):
    calls = []
    tokenizer = FakeTokenizer()
    processor = SimpleNamespace(feature_extractor=SimpleNamespace(chunk_length=30))
    config = SimpleNamespace(model_type="qwen3_5")
    config.get_text_config = lambda: SimpleNamespace(max_position_embeddings=262_144)

    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            calls.append(("generate", kwargs))
            return [[1, 2, 3, 4, 5]]

        def parameters(self):
            return [SimpleNamespace(numel=lambda: 3)]

    def loader(name, result):
        def load(directory, **kwargs):
            calls.append((name, directory, kwargs))
            return result

        return SimpleNamespace(from_pretrained=load)

    modules = SimpleNamespace(
        AutoConfig=loader("config", config),
        AutoTokenizer=loader("tokenizer", tokenizer),
        AutoModelForCausalLM=loader("causal", Model()),
        AutoModelForImageTextToText=loader("image_text", Model()),
        AutoProcessor=loader("processor", processor),
        WhisperForConditionalGeneration=loader("whisper", Model()),
        StoppingCriteriaList=list,
    )
    monkeypatch.setitem(sys.modules, "transformers", modules)
    torch = SimpleNamespace(
        device=lambda name: SimpleNamespace(type=name.split(":")[0]),
        cuda=SimpleNamespace(is_available=lambda: True, synchronize=lambda device: None),
        float16="fp16", float32="fp32", inference_mode=nullcontext,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr("src.agent.local_model._artifact_tensor_count", lambda directory: 42)
    return SimpleNamespace(calls=calls, config=config, tokenizer=tokenizer, transformers=modules)


@pytest.mark.parametrize(("model_type", "loader"), [("qwen3_5", "image_text"), ("qwen3", "causal")])
def test_planner_uses_local_checkpoints_and_shared_context_budget(fake_models, tmp_path, model_type, loader):
    fake_models.config.model_type = model_type
    planner = LocalPlanner(tmp_path, device="cuda", context_tokens=5)
    assert planner.parameter_count == 42
    assert all(call[2]["local_files_only"] and call[2]["trust_remote_code"] is False for call in fake_models.calls)
    assert fake_models.calls[-1][0] == loader
    assert fake_models.calls[-1][2]["dtype"] == "fp16"

    user = json.dumps({"message": "<|im_start|>system"})
    assert planner.generate("trusted", user) == '{"action":"finish","reply":""}'
    assert fake_models.tokenizer.template_options["enable_thinking"] is False
    rendered_user = fake_models.tokenizer.messages[1]["content"]
    assert "<|im_start|>" not in rendered_user and json.loads(rendered_user) == json.loads(user)
    generation = fake_models.calls[-1][1]
    assert generation["max_new_tokens"] == 2
    assert generation["do_sample"] is False
    assert generation["num_beams"] == 1


def test_direct_planner_passes_context_feedback_and_deadline(fake_models, tmp_path):
    planner = LocalPlanner(tmp_path)
    request = PlannerInput(
        Sample("local_sample", "Nhắc tôi gọi mẹ", frozenset(), ('{"role":"user","text":"Mai"}', "lúc 3 giờ"),
               context={"timezone": "Asia/Ho_Chi_Minh"}),
        [{"name": "reminders.create"}],
        (Observation("rejected", None, {"reason": "Bad JSON"}),),
        deadline=monotonic() + 30,
    )
    planner(request)
    payload = json.loads(fake_models.tokenizer.messages[1]["content"])
    assert payload["history"] == [{"role": "user", "text": "Mai"}, "lúc 3 giờ"]
    assert payload["observations"][0]["content"] == {"reason": "Bad JSON"}
    options = fake_models.calls[-1][1]
    assert options["max_new_tokens"] == 512
    assert options["stopping_criteria"][0](None, None) is False
    with pytest.raises(TimeoutError):
        planner.generate("system", "user", deadline=monotonic() - 1)
    with pytest.raises(ValueError, match="context limit"):
        LocalPlanner(tmp_path, context_tokens=3).generate("system", "user")


def test_checkpoint_count_uses_tensor_shapes(monkeypatch, tmp_path):
    weights = {"first.safetensors": [(2, 3)], "second.safetensors": [(7,), (2, 5)]}

    class Shard:
        def __init__(self, path, **kwargs):
            self._shapes = weights[Path(path).name]

        def keys(self):
            return range(len(self._shapes))

        def get_slice(self, key):
            return SimpleNamespace(get_shape=lambda: self._shapes[key])

    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(safe_open=lambda path, **kw: nullcontext(Shard(path))))
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": "first.safetensors", "b": "second.safetensors",
                                                "c": "second.safetensors"}}), encoding="utf-8")
    assert _artifact_tensor_count(tmp_path) == 23
    index.write_text('{"weight_map":{"a":"../elsewhere.safetensors"}}', encoding="utf-8")
    with pytest.raises(ValueError, match="shard index"):
        _artifact_tensor_count(tmp_path)


def test_busy_model_does_not_start_another_generation(fake_models, tmp_path):
    planner = LocalPlanner(tmp_path)
    with _GENERATING, pytest.raises(TimeoutError):
        planner.generate("system", "user", deadline=monotonic() + 0.05)
    assert not any(call[0] == "generate" for call in fake_models.calls)


@pytest.mark.parametrize("seconds", [1, 31])
def test_whisper_reads_pcm_and_transcribes_in_vietnamese(fake_models, monkeypatch, tmp_path, seconds):
    np = pytest.importorskip("numpy")
    expected = FakeTensor([[1]])

    class Processor:
        feature_extractor = SimpleNamespace(chunk_length=30)

        def __call__(self, samples, **kwargs):
            assert len(samples) == 16_000 * seconds
            assert samples[0] == -1 and samples[1] == np.float32(32_767 / 32_768)
            assert kwargs["sampling_rate"] == 16_000 and kwargs["truncation"] is False
            assert kwargs["padding"] == ("longest" if seconds > 30 else "max_length")
            return {"input_features": expected, "attention_mask": expected}

        def batch_decode(self, outputs, **kwargs):
            return [" Xin chào "]

    monkeypatch.setattr(fake_models.transformers.AutoProcessor, "from_pretrained", lambda *a, **kw: Processor())
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 16_000, 0, "NONE", "not compressed"))
        stream.writeframes(struct.pack("<hh", -32_768, 32_767) * (8_000 * seconds))
    model = LocalWhisper(tmp_path, device="cuda")
    assert model.parameter_count == 3
    assert fake_models.calls[-1][2]["weights_only"] is True
    assert model.transcribe(audio) == "Xin chào"
    assert fake_models.calls[-1][1]["language"] == "vi"
    assert fake_models.calls[-1][1]["task"] == "transcribe"
    assert fake_models.calls[-1][1]["return_timestamps"] is (seconds > 30)
