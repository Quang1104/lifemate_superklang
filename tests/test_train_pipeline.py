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
    assert scores == {"synthetic-1": 0.0, "synthetic-2": 1.0}
    assert summary["local_proxy_0_to_100"] == 50
    assert summary["timeout_count"] == 1


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
