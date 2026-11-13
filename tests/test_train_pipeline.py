import json
from types import SimpleNamespace

import pytest

from scripts.compare_train import evaluate, locked_ids
from src.agent.runner import Limits
from src.inference.official import KitError
from src.inference.pipeline import predict_records, write_official_records
from src.tooling.registry import ToolRegistry


class Kit:
    split = "train"
    registry = ToolRegistry(())

    def __init__(self):
        self.opened = []

    def rows(self, _):
        for sample_id in ("synthetic-1", "synthetic-2"):
            yield {"id": sample_id, "audio": sample_id, "history": [],
                   "context": {"installed_apps": {}}}

    def audio_path(self, sample_id, path, *, history):
        return path

    def open_environment(self, sample):
        self.opened.append(sample.sample_id)
        return SimpleNamespace(read_output=lambda: ([], "test-only-official-value"))


def test_training_reports_asr_timeout_and_exports_real_log(tmp_path):
    kit = Kit()
    diagnostics = []
    sessions = []

    def transcribe(path):
        if path == "synthetic-1":
            raise TimeoutError()
        return "Unsupported request"

    def factory():
        sessions.append(object())
        return lambda request: '{"action":"finish","reply":""}'

    predictions = tmp_path / "predictions.jsonl"
    records = predict_records(
        kit, None, transcribe, planner_factory=factory, continue_train_failures=True,
        on_sample=lambda sample, status, time, result, exported: diagnostics.append(
            {"id": sample, "status": status, "time": time, "exported": exported}),
    )
    assert write_official_records(predictions, records) == 2
    assert [row["status"] for row in diagnostics] == ["preprocessing_timeout", "finished"]
    assert kit.opened == ["synthetic-1", "synthetic-2"]
    assert len(sessions) == 1

    report = tmp_path / "report.json"
    report.write_text(json.dumps({"split": "train", "planner_mode": "direct", "samples": diagnostics,
                                  "p95_sample_time_ms": 1, "max_sample_time_ms": 1}), encoding="utf-8")
    gold = {f"synthetic-{number}": {"tool_calls": []} for number in (1, 2)}
    scores, summary = evaluate(gold, predictions, report, {})
    # The internal ASR deadline fired within 30 s and an empty record was exported, so the
    # official scorer grades it like any record: empty prediction vs empty gold scores 1.
    assert scores == {"synthetic-1": 1.0, "synthetic-2": 1.0}
    assert summary["local_proxy_0_to_100"] == 100
    assert summary["timeout_count"] == 1


def test_over_limit_and_missing_records_score_zero(tmp_path):
    predictions = tmp_path / "p.jsonl"
    predictions.write_text(json.dumps({"id": "a", "tool_calls": [], "state_hash": "h"}) + "\n", encoding="utf-8")
    report = tmp_path / "r.json"
    report.write_text(json.dumps({"split": "train", "planner_mode": "direct", "p95_sample_time_ms": 1,
                                  "max_sample_time_ms": 1, "samples": [
                                      {"id": "a", "status": "finished", "time": 30_001, "exported": True},
                                      {"id": "b", "status": "planner_error", "time": 10, "exported": False}]}),
                      encoding="utf-8")
    gold = {"a": {"tool_calls": []}, "b": {"tool_calls": []}}
    scores, summary = evaluate(gold, predictions, report, {})
    assert scores == {"a": 0.0, "b": 0.0}
    assert summary["lost_proxy_points_by_outcome"] == {"not_exported": 50.0, "over_time_limit": 50.0}


def test_planner_deadline_leaves_time_to_export_real_log():
    kit = Kit()
    diagnostics = []

    def planner(request):
        assert request.deadline is not None
        raise TimeoutError()

    records = list(predict_records(
        kit, planner, lambda path: "test", limits=Limits(sample_seconds=2),
        export_reserve_seconds=0.5, on_sample=lambda *sample: diagnostics.append(sample),
    ))
    assert [row.sample_id for row in records] == kit.opened
    assert len(records) == 2
    assert all(row.tool_calls == [] for row in records)
    assert [row[1] for row in diagnostics] == ["planner_timeout", "planner_timeout"]


def test_submission_failure_does_not_replace_existing_output(tmp_path):
    output = tmp_path / "submission.jsonl"
    output.write_text("previous", encoding="utf-8")
    kit = Kit()
    kit.split = "private_test"

    def broken_environment(_):
        raise RuntimeError("ToolEnv unavailable")

    kit.open_environment = broken_environment

    with pytest.raises(KitError, match="could not export official output"):
        write_official_records(output, predict_records(
            kit, lambda request: '{"action":"finish","reply":""}', lambda path: "test"))
    assert output.read_text(encoding="utf-8") == "previous"


def test_locked_manifest_excludes_smoke_ids(tmp_path):
    validation = tmp_path / "validation.json"
    smoke = tmp_path / "smoke.json"
    validation.write_text('{"source_split":"train","validation_ids":["a","b","c"]}', encoding="utf-8")
    smoke.write_text('{"source_split":"train","validation_ids":["b"]}', encoding="utf-8")
    assert locked_ids(validation, smoke) == ["a", "c"]


import pytest


@pytest.mark.parametrize("split, scale", [("public_test", "3"), ("private_test", "2"), ("train", "0.5"), ("train", "nan")])
def test_time_scale_is_train_only_and_never_tightens(tmp_path, split, scale, capsys):
    from src.inference.__main__ import main

    code = main(["official", "--split", split, "--data-root", str(tmp_path), "--output", str(tmp_path / "o.jsonl"),
                 "--time-scale", scale])
    assert code == 2
    assert "Inference stopped" in capsys.readouterr().err
