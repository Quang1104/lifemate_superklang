"""TRAIN-only timing reports with synthetic kit and model fakes."""

from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent import local_model
from src.inference import __main__ as inference_main
from src.inference.pipeline import predict_records
from src.tooling.registry import ToolRegistry


SAMPLE_FIELDS = {"asr_ms", "planner_calls", "prompt_tokens", "generated_tokens",
                 "reused_prefix_tokens", "grammar_compile_ms", "generate_ms", "grammar_status",
                 "timing_complete", "asr_timing_complete"}
SUMMARY_FIELDS = {"mean_planner_generate_ms", "p95_planner_generate_ms", "mean_prompt_tokens",
                  "prompts_above_12000", "total_grammar_compile_ms"}


def metric(prompt_tokens, generated_tokens, reused_prefix_tokens, grammar_compile_ms,
           generate_ms, grammar_status):
    return dict(zip(("prompt_tokens", "generated_tokens", "reused_prefix_tokens",
                     "grammar_compile_ms", "generate_ms", "grammar_status"),
                    (prompt_tokens, generated_tokens, reused_prefix_tokens,
                     grammar_compile_ms, generate_ms, grammar_status)))


class TimedPlanner:
    parameter_count = 1

    def __init__(self, *args, **kwargs):
        self.generation_metrics = [metric(-1, -1, -1, -1, -1, "stale")]
        self.grammar_stats = Counter()
        self.resets = 0
        self.snapshots = 0

    def reset_metrics(self):
        self.resets += 1
        self.generation_metrics = []

    def timing_snapshot(self):
        self.snapshots += 1
        return [dict(row) for row in self.generation_metrics]

    def __call__(self, request):
        if request.sample.sample_id == "synthetic-1":
            calls = [metric(12000, 4, 0, 7.25, 10, "constrained"),
                     metric(12001, 5, 100, 0, 30, "runtime_fallback")]
            index = len(self.generation_metrics)
            self.generation_metrics.append(calls[index])
            if index == 0:
                return "synthetic invalid JSON forces a second planner call"
        else:
            self.generation_metrics.append(metric(13000, 6, 0, 1.75, 20, "compile_failed"))
        return '{"action":"finish","reply":""}'


class TimedWhisper:
    parameter_count = 1

    def __init__(self, *args, fail_first=False, **kwargs):
        self.asr_metrics = [{"path": "stale", "asr_ms": -1}]
        self.fail_first = fail_first
        self.resets = 0
        self.snapshots = 0

    def reset_metrics(self):
        self.resets += 1
        self.asr_metrics = []

    def timing_snapshot(self):
        self.snapshots += 1
        return [dict(row) for row in self.asr_metrics]

    def transcribe(self, path, *, deadline=None):
        name = str(path)
        milliseconds = {"synthetic-1": 2, "synthetic-1-history": 3, "synthetic-2": 7.5}[name]
        self.asr_metrics.append({"path": name, "asr_ms": milliseconds})
        if self.fail_first and name == "synthetic-1":
            raise TimeoutError("synthetic preprocessing timeout")
        return "test request"

    __call__ = transcribe


class Kit:
    split = "train"
    registry = ToolRegistry(())

    def __init__(self, *args):
        self.opened = []

    def rows(self, sample_ids):
        for sample_id in ("synthetic-1", "synthetic-2"):
            if sample_ids is not None and sample_id not in sample_ids:
                continue
            history = ([{"role": "user", "audio": "synthetic-1-history", "audio_text": "history"}]
                       if sample_id == "synthetic-1" else [])
            yield {"id": sample_id, "audio": sample_id, "audio_text": "test request", "history": history,
                   "context": {"installed_apps": {}}}

    def audio_path(self, sample_id, relative_path, *, history):
        return Path(relative_path)

    def open_environment(self, sample):
        self.opened.append(sample.sample_id)
        return SimpleNamespace(read_output=lambda: ([], "test-only-official-value"))


def test_pipeline_resets_before_preprocessing_including_failed_sample():
    kit = Kit()
    planner, transcribe = TimedPlanner(), TimedWhisper(fail_first=True)
    details = []

    def track(sample_id, status, elapsed_ms, result, exported):
        details.append((sample_id, status, planner.timing_snapshot(), transcribe.timing_snapshot()))

    records = list(predict_records(kit, planner, transcribe, on_sample=track,
                                   continue_train_failures=True))
    assert len(records) == 2 and kit.opened == ["synthetic-1", "synthetic-2"]
    assert planner.resets == transcribe.resets == 2
    assert details[0][1] == "preprocessing_timeout"
    assert details[0][2] == []
    assert [row["asr_ms"] for row in details[0][3]] == [2]
    assert details[1][1] == "finished"
    assert [row["prompt_tokens"] for row in details[1][2]] == [13000]
    assert [row["asr_ms"] for row in details[1][3]] == [7.5]


@pytest.mark.parametrize("with_report", [True, False])
def test_timing_fields_only_appear_in_train_report(monkeypatch, tmp_path, with_report):
    kit = Kit()
    planner, transcribe = TimedPlanner(), TimedWhisper()
    monkeypatch.setattr(inference_main, "OfficialKit", lambda *args: kit)
    monkeypatch.setattr(local_model, "initialize_runtime", lambda: None)
    monkeypatch.setattr(local_model, "LocalPlanner", lambda *args, **kwargs: planner)
    monkeypatch.setattr(local_model, "LocalWhisper", lambda *args, **kwargs: transcribe)
    output = tmp_path / "official.jsonl"
    report_path = tmp_path / "train-report.json"
    argv = ["official", "--split", "train", "--data-root", str(tmp_path), "--output", str(output),
            "--model-dir", str(tmp_path / "model"), "--asr-dir", str(tmp_path / "asr")]
    if with_report:
        argv += ["--train-report", str(report_path)]
    summary = inference_main._run_official(inference_main._parser().parse_args(argv))
    assert not (SAMPLE_FIELDS | SUMMARY_FIELDS | {"samples"}) & summary.keys()
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert all(set(row) == {"id", "tool_calls", "state_hash"} for row in records)
    assert planner.resets == transcribe.resets == 2
    if not with_report:
        assert not report_path.exists()
        assert planner.snapshots == transcribe.snapshots == 0
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    first, second = report["samples"]
    assert SAMPLE_FIELDS <= first.keys() and SAMPLE_FIELDS <= second.keys()
    assert [row["asr_ms"] for row in (first, second)] == [5, 7.5]
    assert [row["planner_calls"] for row in (first, second)] == [2, 1]
    assert first["prompt_tokens"] == [12000, 12001]
    assert first["generated_tokens"] == [4, 5]
    assert first["reused_prefix_tokens"] == [0, 100]
    assert first["grammar_compile_ms"] == [7.25, 0]
    assert first["generate_ms"] == [10, 30]
    assert first["grammar_status"] == ["constrained", "runtime_fallback"]
    assert first["timing_complete"] == [True, True]
    assert second["timing_complete"] == [True]
    assert first["asr_timing_complete"] is second["asr_timing_complete"] is True
    assert second["prompt_tokens"] == [13000]
    assert second["grammar_status"] == ["compile_failed"]
    assert report["mean_planner_generate_ms"] == 20
    assert report["p95_planner_generate_ms"] == 30
    assert report["mean_prompt_tokens"] == pytest.approx((12000 + 12001 + 13000) / 3)
    assert report["prompts_above_12000"] == 2
    assert report["total_grammar_compile_ms"] == 9


def test_noop_train_report_contains_zero_model_timing_aggregates(monkeypatch, tmp_path):
    monkeypatch.setattr(inference_main, "OfficialKit", Kit)
    manifest = tmp_path / "train-manifest.json"
    manifest.write_text(json.dumps({"source_split": "train", "validation_ids": ["synthetic-1", "synthetic-2"]}),
                        encoding="utf-8")
    report_path = tmp_path / "report.json"
    args = inference_main._parser().parse_args([
        "official", "--split", "train", "--data-root", str(tmp_path),
        "--output", str(tmp_path / "official.jsonl"), "--train-report", str(report_path),
        "--control-noop", "--train-manifest", str(manifest),
    ])
    inference_main._run_official(args)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert {field: report[field] for field in SUMMARY_FIELDS} == dict.fromkeys(SUMMARY_FIELDS, 0)
    for sample in report["samples"]:
        assert sample["asr_ms"] == sample["planner_calls"] == 0
        assert sample["asr_timing_complete"] is True
        assert all(sample[field] == [] for field in SAMPLE_FIELDS - {"asr_ms", "planner_calls",
                                                                   "asr_timing_complete"})
