"""CPU-only coverage; API/model files below are explicit synthetic test fixtures.

Can also run without pytest: python -B -m unittest discover -s tests -p test_asr_large.py -v
"""

import contextlib
from datetime import datetime, timezone
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.asr_large_report import build_report, latency_summary
from scripts.check_asr_params import parameter_budget, qwen3_parameters, whisper_parameters
from scripts.fetch_models import fetch_models
from training_code.model_pins import CUTOFF, UNRESOLVED, checked_pin, eligible_revision, verify_revision

ROOT = Path(__file__).resolve().parents[1]
# Real public commit listed by HF (2023-12-20); dates in rejection tests are synthetic metadata.
LARGE_SHA = "6b62f92bcadde8d9f0fa1d293328134af8866c19"
PLANNER = {"model_type": "qwen3", "hidden_size": 2560, "intermediate_size": 9728,
           "num_hidden_layers": 36, "vocab_size": 151936, "head_dim": 128,
           "num_attention_heads": 32, "num_key_value_heads": 8, "tie_word_embeddings": True,
           "attention_bias": False}
LARGE = {"model_type": "whisper", "d_model": 1280, "encoder_ffn_dim": 5120,
         "decoder_ffn_dim": 5120, "encoder_layers": 32, "decoder_layers": 32,
         "vocab_size": 51865, "num_mel_bins": 80, "max_source_positions": 1500,
         "max_target_positions": 448}


def commit(date="2023-12-20T09:16:00+00:00", sha=LARGE_SHA):
    return SimpleNamespace(commit_id=sha, created_at=datetime.fromisoformat(date))


class PinTests(unittest.TestCase):
    def test_config_has_large_role_and_full_pin_or_explicit_placeholder(self):
        pins = json.loads((ROOT / "configs/colab_models.json").read_text(encoding="utf-8"))
        self.assertEqual(pins["asr_large"]["repo_id"], "vinai/PhoWhisper-large")
        if pins["asr_large"]["revision"] != UNRESOLVED:
            self.assertEqual(checked_pin(pins, "asr_large"), pins["asr_large"])
        self.assertNotEqual(pins["asr_large"]["repo_id"], pins["asr"]["repo_id"])

    def test_mutable_short_and_unresolved_revisions_are_rejected(self):
        for revision in ("main", LARGE_SHA[:7], UNRESOLVED, "", "../main"):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                checked_pin({"asr_large": {"repo_id": "vinai/PhoWhisper-large", "revision": revision}}, "asr_large")

    def test_cutoff_is_strict_and_timezone_aware(self):
        pin = {"repo_id": "vinai/PhoWhisper-large", "revision": LARGE_SHA}
        self.assertEqual(eligible_revision([commit("2026-10-26T00:00:00+00:00"), commit()]), LARGE_SHA)
        self.assertIn("2023-12-20", verify_revision(pin, commit()))
        for value in (CUTOFF, datetime(2026, 10, 27, tzinfo=timezone.utc), datetime(2023, 1, 1)):
            with self.subTest(date=value), self.assertRaises(ValueError):
                verify_revision(pin, SimpleNamespace(commit_id=LARGE_SHA, created_at=value))
        with self.assertRaises(ValueError):
            eligible_revision([commit("2026-10-26T00:00:00+00:00")])
        real_medium_sha = json.loads((ROOT / "configs/colab_models.json").read_text())["asr"]["revision"]
        with self.assertRaises(ValueError):
            verify_revision(pin, commit(sha=real_medium_sha))

    def exercise_fetch(self, *, revision=UNRESOLVED, resolve=False, date=None):
        events = []
        def commits(**kwargs):
            events.append(("metadata", kwargs))
            return [commit(date)] if date else [commit()]
        hub = SimpleNamespace(HfApi=lambda: SimpleNamespace(list_repo_commits=commits),
                              snapshot_download=lambda **kw: events.append(("download", kw)))
        with TemporaryDirectory() as directory, patch.dict(sys.modules, {"huggingface_hub": hub}):
            root = Path(directory)
            pin_file = root / "pins.json"
            pin_file.write_text(json.dumps({"asr_large": {"repo_id": "vinai/PhoWhisper-large",
                                                          "revision": revision}}), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    fetch_models(root / "models", ["asr_large"], resolve_asr_large=resolve, pin_file=pin_file)
                    error = None
                except ValueError as exc:
                    error = exc
            saved = json.loads(pin_file.read_text(encoding="utf-8"))
        return events, saved, error

    def test_resolver_persists_real_api_sha_and_checks_date_before_download(self):
        events, saved, error = self.exercise_fetch(resolve=True)
        self.assertIsNone(error)
        self.assertEqual(saved["asr_large"]["revision"], LARGE_SHA)
        self.assertEqual([e[0] for e in events], ["metadata", "metadata", "download"])
        download = events[-1][1]
        self.assertEqual(download["revision"], LARGE_SHA)
        self.assertEqual(download["local_dir"].name, "asr_large")
        self.assertIn("*.bin", download["allow_patterns"])

    def test_resolving_an_existing_pin_does_not_move_it(self):
        events, saved, error = self.exercise_fetch(revision=LARGE_SHA, resolve=True)
        self.assertIsNone(error)
        self.assertEqual([e[0] for e in events], ["metadata", "download"])
        self.assertEqual(events[0][1]["revision"], saved["asr_large"]["revision"])

    def test_unresolved_or_ineligible_metadata_cannot_download_weights(self):
        for kwargs in ({}, {"revision": LARGE_SHA, "date": "2026-10-26T00:00:00+00:00"},
                       {"resolve": True, "date": "2026-10-27T00:00:00+00:00"}):
            events, _, error = self.exercise_fetch(**kwargs)
            self.assertIsInstance(error, ValueError)
            self.assertFalse(any(e[0] == "download" for e in events))


class ParameterTests(unittest.TestCase):
    def test_published_config_dimensions_include_frozen_parameters(self):
        result = parameter_budget(PLANNER, LARGE)
        self.assertEqual(result["planner"], 4_022_468_096)
        self.assertEqual(result["asr_large"], 1_543_304_960)
        self.assertEqual(result["total"], 5_565_773_056)
        self.assertTrue(result["planner_below_5b"] and result["total_below_6b"])

    def test_tied_weights_and_strict_limits(self):
        self.assertEqual(qwen3_parameters({**PLANNER, "tie_word_embeddings": False}) - qwen3_parameters(PLANNER),
                         PLANNER["vocab_size"] * PLANNER["hidden_size"])
        with patch("scripts.check_asr_params.qwen3_parameters", return_value=5_000_000_000), \
                patch("scripts.check_asr_params.whisper_parameters", return_value=1_000_000_000):
            result = parameter_budget({}, {})
            self.assertFalse(result["planner_below_5b"])
            self.assertFalse(result["total_below_6b"])

    def test_unsupported_architectures_and_invalid_dimensions_fail(self):
        for config in ({**PLANNER, "model_type": "qwen3_5"}, {**PLANNER, "attention_bias": True},
                       {**PLANNER, "hidden_size": 0}, {**PLANNER, "hidden_size": True}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                qwen3_parameters(config)
        with self.assertRaises(ValueError):
            whisper_parameters({**LARGE, "model_type": "other"})

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                         "optional torch/transformers unavailable; no weights needed")
    def test_counts_match_transformers_meta_models(self):
        import torch
        try:
            from transformers import Qwen3Config, Qwen3ForCausalLM, WhisperConfig, WhisperForConditionalGeneration
        except ImportError:  # local transformers predates Qwen3
            self.skipTest("installed transformers has no Qwen3")
        with torch.device("meta"):
            asr_config = WhisperConfig(d_model=16, encoder_layers=2, decoder_layers=2, encoder_ffn_dim=32,
                                       decoder_ffn_dim=32, encoder_attention_heads=2, decoder_attention_heads=2,
                                       vocab_size=64, max_source_positions=30, max_target_positions=20,
                                       pad_token_id=0, bos_token_id=1, eos_token_id=2)
            model = WhisperForConditionalGeneration(asr_config)
            self.assertEqual(whisper_parameters(asr_config.to_dict()), sum(p.numel() for p in model.parameters()))
            for tied in (True, False):
                config = Qwen3Config(hidden_size=64, head_dim=16, num_hidden_layers=2, intermediate_size=128,
                                     num_attention_heads=4, num_key_value_heads=2, vocab_size=80,
                                     tie_word_embeddings=tied)
                model = Qwen3ForCausalLM(config)
                self.assertEqual(qwen3_parameters(config.to_dict()), sum(p.numel() for p in model.parameters()))


class LatencyTests(unittest.TestCase):
    def fixture(self):
        rows = [{"id": f"synthetic-sample-{i}", "time": i * 100} for i in range(1, 41)]
        report = {"samples": [dict(row) for row in rows], "model_load_ms": 1234, "time_scale": 1, "statuses": {"ok": 40}}
        expected = [row["id"] for row in rows]
        scored = {"local_proxy_0_to_100": 61.29}
        log = 'warning text\n{"epoch":0,"wer":0.28}\n{"epoch":1,"wer":0.08}\n'
        return rows, report, expected, scored, log

    def test_nearest_rank_ms_quantiles_and_separate_model_load(self):
        result = build_report(*self.fixture())
        self.assertEqual(result["official_time"], {"count": 40, "unit": "ms", "quantiles": "nearest rank",
                                                 "p50": 2000, "p95": 3800, "max": 4000, "over_30000_ms": 0})
        self.assertEqual(result["model_load_ms"], 1234)
        self.assertTrue(result["candidate_passes_numeric_gates"])

    def test_missing_export_and_failed_sample_time_cannot_pass(self):
        rows, report, expected, scored, log = self.fixture()
        report["samples"][-1]["time"] = 31_000
        result = build_report(rows[:-1], report, expected, scored, log)
        self.assertFalse(result["candidate_passes_numeric_gates"])
        self.assertFalse(result["numeric_gates"]["all_smoke_rows_exported"])
        self.assertEqual(result["all_sample_time"]["max"], 31_000)

    def test_acceptance_thresholds_and_invalid_time(self):
        rows, report, expected, scored, log = self.fixture()
        self.assertFalse(build_report(rows, report, expected, scored, log.replace('0.08', '0.099'))
                         ["candidate_passes_numeric_gates"])
        self.assertFalse(build_report(rows, report, expected, {"local_proxy_0_to_100": 61.28}, log)
                         ["candidate_passes_numeric_gates"])
        for value in (None, -1, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                latency_summary([{"time": value}])
        self.assertIsNone(latency_summary([])["p95"])
        self.assertFalse(build_report([], report, expected, scored, log)["candidate_passes_numeric_gates"])
        report["time_scale"] = 3
        self.assertFalse(build_report(rows, report, expected, scored, log)["candidate_passes_numeric_gates"])


class DockerTests(unittest.TestCase):
    def exercise(self, *, tuned, revision=LARGE_SHA):
        calls = []
        hub = SimpleNamespace(snapshot_download=lambda **kw: calls.append(kw))
        merge = SimpleNamespace(merge=lambda *args: 0)
        with TemporaryDirectory() as directory, patch.dict(sys.modules, {"huggingface_hub": hub,
                "training_code.stream_merge": merge}), patch.object(sys, "path", sys.path.copy()):
            root = Path(directory)
            (root / "configs").mkdir()
            pins = json.loads((ROOT / "configs/colab_models.json").read_text(encoding="utf-8"))
            pins["asr_large"]["revision"] = revision
            (root / "configs/colab_models.json").write_text(json.dumps(pins), encoding="utf-8")
            spec = importlib.util.spec_from_file_location("synthetic_docker_fetch", ROOT / "docker/fetch_models.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.__file__ = str(root / "fetch_models.py")
            base, adapter, target = root / "base", root / "adapter", root / "asr"
            base.mkdir()
            if tuned:
                (adapter / "asr").mkdir(parents=True)
                (adapter / "asr/config.json").write_text(json.dumps(LARGE), encoding="utf-8")
                (adapter / "asr/model.safetensors").write_bytes(b"synthetic serialization; no model weights")
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    module.main(str(adapter), base=str(base), planner=str(root / "planner"),
                                asr=str(target), asr_role="asr_large")
                    error = None
                except ValueError as exc:
                    error = exc
            if tuned:
                self.assertEqual((target / "model.safetensors").read_bytes(),
                                 (adapter / "asr/model.safetensors").read_bytes())
                self.assertEqual((target / "config.json").read_bytes(), (adapter / "asr/config.json").read_bytes())
        return calls, error

    def test_large_fine_tune_is_copied_unchanged_and_bypasses_base_asr_pin(self):
        calls, error = self.exercise(tuned=True, revision=UNRESOLVED)
        self.assertIsNone(error)
        self.assertEqual(len(calls), 1)  # only planner download; no ASR download

    def test_large_fallback_uses_exact_pin(self):
        calls, error = self.exercise(tuned=False)
        self.assertIsNone(error)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[-1]["repo_id"], "vinai/PhoWhisper-large")
        self.assertEqual(calls[-1]["revision"], LARGE_SHA)

    def test_unresolved_large_fails_before_any_download(self):
        calls, error = self.exercise(tuned=False, revision=UNRESOLVED)
        self.assertIsInstance(error, ValueError)
        self.assertEqual(calls, [])


@unittest.skipUnless(shutil.which("bash"), "bash unavailable")
class ShellTests(unittest.TestCase):
    def exercise(self, stale=False):
        source = (ROOT / "scripts/colab_asr_large.sh").read_text(encoding="utf-8")
        with TemporaryDirectory() as directory:
            setup = '\n'.join([
                'R="$PWD/results"; M="$PWD/models"; K=unused; OFF=; SMOKE=unused',
                'mkdir -p "$R" "$M" configs; echo fixture > configs/colab_models.json',
                'python() { echo SYNTHETIC_SETUP; }',
                'env() { echo SYNTHETIC_TRAIN; mkdir -p "$FT_L"; echo fixture > "$FT_L/config.json"; }',
                'merge() { return 0; }',
                'smoke() { echo SYNTHETIC_SMOKE_FAILURE; return 7; }',
                'mkdir -p "$M/asr_large_ft"' if stale else ':',
            ])
            source = source.replace('source "$(dirname "$0")/colab_lib.sh"', setup, 1)
            return subprocess.run([shutil.which("bash"), "-s"], input=source, text=True, capture_output=True,
                                  cwd=directory)

    def test_stale_output_refused_before_fetch_or_train(self):
        result = self.exercise(stale=True)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("output already exists", result.stdout)
        self.assertNotIn("SYNTHETIC_SETUP", result.stdout)

    def test_smoke_failure_survives_tail_pipeline(self):
        result = self.exercise()
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        self.assertIn("SYNTHETIC_TRAIN", result.stdout)
        self.assertNotIn("ASR_LARGE_DONE", result.stdout)
        self.assertNotIn("SMOKE v4_asrlarge", result.stdout)


if __name__ == "__main__":
    unittest.main()
