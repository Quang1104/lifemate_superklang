from pathlib import Path

from training_code.train_asr import utterances, wer


def test_wer_ignores_case_and_punctuation():
    assert wer([("Gọi Minh, nhé.", "gọi minh nhé")]) == 0
    assert wer([("một hai ba bốn", "một ba bốn")]) == 0.25  # one deletion over four reference words
    assert wer([("a b", "a b c d")]) == 1.0  # insertions count


class FakeKit:
    def rows(self):
        return [
            {"id": "t1", "audio": "a/t1.wav", "audio_text": "câu một",
             "history": [{"role": "user", "audio": "a/t1_u1.wav", "audio_text": "lượt cũ"},
                         {"role": "assistant", "content": "ok"}]},
            {"id": "v1", "audio": "a/v1.wav", "audio_text": "câu kiểm", "history": []},
        ]

    def audio_path(self, sample_id, audio, *, history):
        return Path(("H:" if history else "C:") + audio)


def test_utterances_split_and_history():
    train = utterances(FakeKit(), lambda i: i != "v1")
    assert train == [(Path("C:a/t1.wav"), "câu một"), (Path("H:a/t1_u1.wav"), "lượt cũ")]
    assert utterances(FakeKit(), lambda i: i == "v1", history=False) == [(Path("C:a/v1.wav"), "câu kiểm")]


# Exercise the complete training control flow without torch, transformers, audio or weights.
import contextlib
import io
import json
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class TrainingBoundaryTests(unittest.TestCase):
    def run_training(self, *, overlong=False, improved=True):
        from training_code import train_asr
        from src.agent import local_model

        events = {"labels": [], "saves": [], "loads": []}

        class Tensor(list):
            def cuda(self):
                return self

            def detach(self):
                return self

            def to(self, **kwargs):
                events["copy"] = kwargs
                return "cpu-fp16-copy"

        class Tokenizer:
            def set_prefix_tokens(self, **kwargs):
                events["prefix"] = kwargs

            def __call__(self, text):
                ids = {"short": [99, 1, 2, 3, 10, 0], "long": [99, 1, 2, 3, 11, 12, 0]}[text]
                return SimpleNamespace(input_ids=ids + ([5] * 10 if overlong else []))

        class Processor:
            tokenizer = Tokenizer()

            def feature_extractor(self, audio, **kwargs):
                return SimpleNamespace(input_features=Tensor([[0]]))

            def batch_decode(self, out, **kwargs):
                return [next(hypotheses)]

            def save_pretrained(self, path):
                events["processor_save"] = path

        class Model:
            config = SimpleNamespace(forced_decoder_ids=[(1, 1)], use_cache=True)
            generation_config = SimpleNamespace(forced_decoder_ids=[(1, 1)])

            def cuda(self):
                return self

            def gradient_checkpointing_enable(self, **kwargs):
                events["checkpointing"] = kwargs

            def eval(self):
                return self

            def train(self):
                return self

            def parameters(self):
                return []

            def generate(self, **kwargs):
                assert kwargs["language"] == "vi" and kwargs["task"] == "transcribe"
                assert kwargs["use_cache"] is True
                assert self.config.forced_decoder_ids is None
                assert self.generation_config.forced_decoder_ids is None
                return [[1]]

            def __call__(self, **kwargs):
                events["labels"].append(kwargs["labels"])
                assert self.config.use_cache is False
                return SimpleNamespace(loss=SimpleNamespace(backward=lambda: None))

            def state_dict(self):
                return {"weight": Tensor([1])}

            def save_pretrained(self, path, **kwargs):
                assert kwargs == {"state_dict": {"weight": "cpu-fp16-copy"}, "safe_serialization": True}
                events["saves"].append(path)

        rows = [{"id": "synthetic-train-short", "audio": "short.wav", "audio_text": "short", "history": []},
                {"id": "synthetic-train-long", "audio": "long.wav", "audio_text": "long", "history": []},
                {"id": "synthetic-validation", "audio": "val.wav", "audio_text": "correct", "history": []}]
        kit = SimpleNamespace(rows=lambda: rows, audio_path=lambda sample_id, audio, **kw: Path(audio))
        hypotheses = iter(["wrong", "correct" if improved else "wrong", "wrong"])
        optimizer = SimpleNamespace(step=lambda: None, zero_grad=lambda **kw: None)

        def adam(parameters, **kwargs):
            events["optimizer"] = kwargs
            return optimizer

        def autocast(device, **kwargs):
            assert device == "cuda" and kwargs == {"dtype": "bf16"}
            return contextlib.nullcontext()

        def load_model(path, **kwargs):
            events["loads"].append(kwargs)
            return Model()

        modules = {"torch": SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True),
            manual_seed=lambda seed: None, float32="fp32", float16="fp16", bfloat16="bf16",
            no_grad=contextlib.nullcontext, autocast=autocast, tensor=Tensor,
            optim=SimpleNamespace(AdamW=adam), nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=lambda *a: None))),
            "transformers": SimpleNamespace(
                AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: Processor()),
                AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **kw:
                                           SimpleNamespace(decoder_start_token_id=99, max_target_positions=8)),
                WhisperForConditionalGeneration=SimpleNamespace(from_pretrained=load_model),
                get_linear_schedule_with_warmup=lambda *a: SimpleNamespace(step=lambda: None)),
            "src.inference.official": SimpleNamespace(OfficialKit=lambda *a: kit)}
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / "validation.json"
            manifest.write_text(json.dumps({"validation_ids": ["synthetic-validation"]}), encoding="utf-8")
            argv = ["train_asr", "--data-root", directory, "--model-dir", directory,
                    "--output-dir", str(Path(directory) / "out"), "--validation", str(manifest), "--epochs", "2"]
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv), \
                    patch.object(local_model, "_read_audio", return_value=[0]), contextlib.redirect_stdout(io.StringIO()):
                if overlong:
                    with self.assertRaisesRegex(ValueError, "Invalid label length"):
                        train_asr.main()
                else:
                    train_asr.main()
        return events

    def test_prefix_padding_mixed_precision_and_best_checkpoint(self):
        events = self.run_training()
        self.assertEqual(events["prefix"], {"language": "vietnamese", "task": "transcribe"})
        self.assertEqual(sorted(events["labels"][0]), sorted([[1, 2, 3, 10, 0, -100], [1, 2, 3, 11, 12, 0]]))
        self.assertEqual(events["loads"][0]["torch_dtype"], "fp32")
        self.assertEqual(events["loads"][0]["attn_implementation"], "sdpa")
        self.assertEqual(events["checkpointing"], {"gradient_checkpointing_kwargs": {"use_reentrant": False}})
        self.assertFalse(events["optimizer"]["foreach"])
        self.assertEqual(events["copy"], {"device": "cpu", "dtype": "fp16"})
        self.assertEqual(len(events["saves"]), 1)
        self.assertEqual(events["processor_save"], events["saves"][0])

    def test_overlong_labels_rejected_before_weights_load(self):
        self.assertEqual(self.run_training(overlong=True)["loads"], [])

    def test_no_improvement_does_not_save(self):
        self.assertEqual(self.run_training(improved=False)["saves"], [])


import shutil
import subprocess


@unittest.skipUnless(shutil.which("bash"), "bash unavailable")
class ASRShellTests(unittest.TestCase):
    def run_script(self, *, stale=False):
        source = (Path(__file__).resolve().parents[1] / "scripts" / "colab_asr.sh").read_text(encoding="utf-8")
        with TemporaryDirectory() as directory:
            # Replace only the shared-library source in this test: never run its pip or ToolEnv commands.
            setup = "\n".join([
                'R="$PWD/results"', 'M="$PWD/models"',
                "K=unused", "OFF=", 'mkdir -p "$R" "$M"',
                'env() { echo SYNTHETIC_TRAIN; mkdir -p "$FT"; echo fixture > "$FT/config.json"; }',
                'merge() { return 0; }',
                'smoke() { echo SYNTHETIC_SMOKE_FAILURE; return 7; }',
                'mkdir -p "$M/asr_ft"' if stale else ":",
            ])
            source = source.replace('source "$(dirname "$0")/colab_lib.sh"', setup, 1)
            return subprocess.run([shutil.which("bash"), "-s"], input=source, text=True,
                                  capture_output=True, cwd=directory)

    def test_failed_smoke_is_not_hidden_by_tail_or_echo(self):
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("SYNTHETIC_TRAIN", result.stdout)
        self.assertNotIn("ASR_DONE", result.stdout)
        self.assertNotIn("SMOKE w_asrft", result.stdout)

    def test_stale_directory_stops_before_training(self):
        result = self.run_script(stale=True)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("output already exists", result.stdout)
        self.assertNotIn("SYNTHETIC_TRAIN", result.stdout)
