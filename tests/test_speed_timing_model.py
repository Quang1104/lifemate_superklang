"""CPU-only timing counters and TRAIN-report boundary checks."""

from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from threading import Event, Thread
from types import SimpleNamespace

import math
import tempfile
import unittest
from unittest.mock import patch

from src.agent import local_model
from src.agent.local_model import LocalPlanner, LocalWhisper, _PrefixCache


def assert_close(actual, expected):
    if isinstance(expected, list):
        assert len(actual) == len(expected)
        for value, target in zip(actual, expected):
            assert_close(value, target)
    else:
        assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6), (actual, expected)


class Patcher:
    def __init__(self, test):
        self.test = test

    def setattr(self, target, name, value):
        active = patch.object(target, name, value)
        active.start()
        self.test.addCleanup(active.stop)

    def setitem(self, target, name, value):
        active = patch.dict(target, {name: value})
        active.start()
        self.test.addCleanup(active.stop)


class Clock:
    def __init__(self):
        self.now = 0.0

    def advance(self, seconds):
        self.now += seconds


class Tensor(list):
    device, dtype = "cpu", "int64"

    @property
    def shape(self):
        return (len(self), len(self[0]))

    def to(self, *args, **kwargs):
        return self


class Batch(dict):
    def to(self, *args, **kwargs):
        return self


class Cache:
    is_croppable = True

    def __init__(self, length):
        self.length = length
        self.layers = [SimpleNamespace(keys=self._kv(), values=self._kv(), is_croppable=True)]

    def _kv(self):
        return SimpleNamespace(shape=(1, 2, self.length, 8), device="cpu", dtype="float32")

    def get_seq_length(self):
        return self.length

    def crop(self, remove):
        assert remove < 0
        self.length += remove
        self.layers[0].keys.shape = self.layers[0].values.shape = (1, 2, self.length, 8)


def tiny_planner(monkeypatch, clock):
    instance = LocalPlanner.__new__(LocalPlanner)
    syncs = []

    def synchronize(device):
        syncs.append(device)
        clock.advance(0.003)

    class Tokenizer:
        all_special_tokens = []

        def apply_chat_template(self, messages, **kwargs):
            ids = [1, 2] + [ord(character) for character in messages[1]["content"]]
            return Batch(input_ids=Tensor([ids]), attention_mask=Tensor([[1] * len(ids)]))

        def decode(self, tokens, **kwargs):
            assert tokens == [8, 9]
            return "done"

    class Model:
        device, dtype = "cpu", "float32"

        def generate(self, input_ids, **options):
            clock.advance(0.020)
            for processor in options.get("logits_processor", []):
                processor(input_ids, SimpleNamespace(device="cpu"))
            sequences = [list(input_ids[0]) + [8, 9]]
            return SimpleNamespace(sequences=sequences, past_key_values=Cache(len(sequences[0]) - 1))

    instance._torch = SimpleNamespace(inference_mode=nullcontext,
                                      cuda=SimpleNamespace(synchronize=synchronize))
    instance._device = SimpleNamespace(type="cuda")  # A fake device; no torch/GPU import.
    instance._tokenizer = Tokenizer()
    instance._model = Model()
    instance._context_tokens = 4096
    instance._reuse_prefix_cache = True
    instance._prefix_cache = _PrefixCache()
    instance._config = SimpleNamespace(get_text_config=lambda: SimpleNamespace(vocab_size=256))
    instance._grammars = {}
    instance.grammar_stats = Counter()
    instance.reset_metrics()
    monkeypatch.setattr(local_model, "monotonic", lambda: clock.now)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(LogitsProcessorList=list, StoppingCriteriaList=list))
    return instance, syncs


def fake_xgrammar(clock, *, compile_failed=False, runtime_fallback=False):
    compilations = []

    class Compiler:
        def __init__(self, info):
            pass

        def compile_json_schema(self, schema):
            compilations.append(schema)
            clock.advance(0.007)
            if compile_failed:
                raise ValueError("synthetic compile failure")
            return object()

    class Matcher:
        def __init__(self, compiled):
            pass

        def is_terminated(self):
            return False

        def fill_next_token_bitmask(self, bitmask):
            if runtime_fallback:
                raise ValueError("synthetic mask failure")

    class Bitmask:
        def to(self, device):
            return self

    module = SimpleNamespace(
        TokenizerInfo=SimpleNamespace(from_huggingface=lambda *args, **kwargs: object()),
        GrammarCompiler=Compiler, GrammarMatcher=Matcher,
        allocate_token_bitmask=lambda *args: Bitmask(),
        apply_token_bitmask_inplace=lambda *args, **kwargs: None,
    )
    return module, compilations


def check_grammar_status(monkeypatch, status):
    clock = Clock()
    planner, _ = tiny_planner(monkeypatch, clock)
    module, _ = fake_xgrammar(clock, compile_failed=status == "compile_failed",
                             runtime_fallback=status == "runtime_fallback")
    monkeypatch.setitem(sys.modules, "xgrammar", None if status == "unavailable" else module)
    planner.generate("tools", "abc", json_schema={"type": "object"})
    assert planner.timing_snapshot()[0]["grammar_status"] == status
    assert planner.timing_snapshot()[0]["generated_tokens"] == 2




class ModelTimingTests(unittest.TestCase):
    def setUp(self):
        self.monkeypatch = Patcher(self)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)

    def test_local_planner_tokens_prefix_and_reset_without_extra_sync(self):
        monkeypatch = self.monkeypatch
        clock = Clock()
        planner, syncs = tiny_planner(monkeypatch, clock)
        assert planner.generate("tools", "abc") == "done"
        assert planner.generate("tools", "abc") == "done"
        snapshot = planner.timing_snapshot()
        assert [row["prompt_tokens"] for row in snapshot] == [5, 5]
        assert [row["generated_tokens"] for row in snapshot] == [2, 2]
        assert [row["reused_prefix_tokens"] for row in snapshot] == [0, 4]
        assert [row["grammar_compile_ms"] for row in snapshot] == [0, 0]
        assert_close([row["generate_ms"] for row in snapshot], [23, 23])
        assert [row["grammar_status"] for row in snapshot] == ["unconstrained", "unconstrained"]
        assert all(row["timing_complete"] for row in snapshot)
        assert len(syncs) == 2  # The existing one synchronization per generation is retained.
        snapshot[0]["prompt_tokens"] = -1
        assert planner.timing_snapshot()[0]["prompt_tokens"] == 5
        previous = planner.generation_metrics
        planner.reset_metrics()
        assert planner.timing_snapshot() == []
        assert planner.generation_metrics is not previous and len(previous) == 2

    def test_grammar_compile_time_is_separate_and_zero_when_cached(self):
        monkeypatch = self.monkeypatch
        clock = Clock()
        planner, syncs = tiny_planner(monkeypatch, clock)
        xgrammar, compilations = fake_xgrammar(clock)
        monkeypatch.setitem(sys.modules, "xgrammar", xgrammar)
        for _ in range(2):
            planner.generate("tools", "abc", json_schema={"type": "object"})
        snapshot = planner.timing_snapshot()
        assert len(compilations) == 1
        assert_close([row["grammar_compile_ms"] for row in snapshot], [7, 0])
        assert_close([row["generate_ms"] for row in snapshot], [23, 23])
        assert [row["grammar_status"] for row in snapshot] == ["constrained", "constrained"]
        assert len(syncs) == 2

    def test_reset_detaches_inflight_planner_metrics(self):
        monkeypatch = self.monkeypatch
        clock = Clock()
        planner, _ = tiny_planner(monkeypatch, clock)
        started, finish = Event(), Event()
        original = planner._model.generate
        errors = []

        def blocked_generate(*args, **kwargs):
            # HF calls this after each decoded token; input shape requires no device read.
            stop = kwargs["stopping_criteria"][0]
            partial = Tensor([list(kwargs["input_ids"][0]) + [8]])
            assert stop(partial, None) is False
            started.set()
            if not finish.wait(2):
                raise RuntimeError("test did not release fake generation")
            return original(*args, **kwargs)

        def generate():
            try:
                planner.generate("tools", "abc", deadline=1)
            except BaseException as error:
                errors.append(error)

        planner._model.generate = blocked_generate
        worker = Thread(target=generate, daemon=True)
        worker.start()
        try:
            assert started.wait(1)
            clock.advance(0.012)
            snapshot = planner.timing_snapshot()
            assert_close(snapshot[0]["generate_ms"], 12)
            assert snapshot[0]["generated_tokens"] == 1
            assert snapshot[0]["timing_complete"] is False
            assert not any(key.startswith("_") for key in snapshot[0])
            planner.reset_metrics()
        finally:
            finish.set()
            worker.join(2)
        assert not worker.is_alive() and not errors
        assert planner.timing_snapshot() == []
        assert_close(snapshot[0]["generate_ms"], 12)


    def test_inflight_grammar_compile_is_distinct_from_generation(self):
        clock = Clock()
        planner, _ = tiny_planner(self.monkeypatch, clock)
        module, _ = fake_xgrammar(clock)
        original = module.GrammarCompiler
        started, finish = Event(), Event()
        errors = []

        class BlockingCompiler(original):
            def compile_json_schema(self, schema):
                started.set()
                if not finish.wait(2):
                    raise RuntimeError("test did not release fake compiler")
                return super().compile_json_schema(schema)

        module.GrammarCompiler = BlockingCompiler
        self.monkeypatch.setitem(sys.modules, "xgrammar", module)

        def generate():
            try:
                planner.generate("tools", "abc", json_schema={"type": "object"})
            except BaseException as error:
                errors.append(error)

        worker = Thread(target=generate, daemon=True)
        worker.start()
        try:
            assert started.wait(1)
            clock.advance(0.011)
            snapshot = planner.timing_snapshot()
            assert snapshot[0]["grammar_status"] == "compiling"
            assert snapshot[0]["timing_complete"] is False
            assert_close(snapshot[0]["grammar_compile_ms"], 11)
            assert snapshot[0]["generate_ms"] == snapshot[0]["generated_tokens"] == 0
            planner.reset_metrics()
        finally:
            finish.set()
            worker.join(2)
        assert not worker.is_alive() and not errors
        assert planner.timing_snapshot() == []
        assert_close(snapshot[0]["grammar_compile_ms"], 11)

    def test_local_whisper_records_each_file_and_resets(self):
        monkeypatch = self.monkeypatch
        tmp_path = self.tmp_path
        clock = Clock()
        syncs = []
        whisper = LocalWhisper.__new__(LocalWhisper)
        monkeypatch.setattr(local_model, "monotonic", lambda: clock.now)

        def audio(path):
            clock.advance(0.004)
            return [0] * 100

        class Processor:
            def __call__(self, samples, **kwargs):
                clock.advance(0.002)
                return {"input_features": Tensor([[1]]), "attention_mask": Tensor([[1]])}

            def batch_decode(self, outputs, **kwargs):
                clock.advance(0.001)
                return [" transcript "]

        def generate(**kwargs):
            clock.advance(0.010)
            return [[1, 2]]

        def synchronize(device):
            syncs.append(device)
            clock.advance(0.003)

        monkeypatch.setattr(local_model, "_read_audio", audio)
        whisper._torch = SimpleNamespace(inference_mode=nullcontext,
                                         cuda=SimpleNamespace(synchronize=synchronize))
        whisper._device, whisper._dtype = SimpleNamespace(type="cuda"), "float16"
        whisper._processor = Processor()
        whisper._model = SimpleNamespace(generate=generate)
        whisper._chunk_seconds = 30
        whisper.reset_metrics()
        paths = [tmp_path / "current.wav", tmp_path / "history.wav"]
        assert [whisper(path) for path in paths] == ["transcript", "transcript"]
        snapshot = whisper.timing_snapshot()
        assert [row["path"] for row in snapshot] == [str(path) for path in paths]
        assert_close([row["asr_ms"] for row in snapshot], [20, 20])
        assert all(row["timing_complete"] for row in snapshot)
        assert len(syncs) == 2
        snapshot[0]["asr_ms"] = -1
        assert_close(whisper.timing_snapshot()[0]["asr_ms"], 20)
        previous = whisper.asr_metrics
        whisper.reset_metrics()
        assert whisper.timing_snapshot() == []
        assert whisper.asr_metrics is not previous and len(previous) == 2


    def test_grammar_fallback_status_is_recorded_per_call(self):
        for status in ("unavailable", "compile_failed", "runtime_fallback"):
            with self.subTest(status=status):
                check_grammar_status(self.monkeypatch, status)


if __name__ == "__main__":
    unittest.main()
