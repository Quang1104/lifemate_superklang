"""Resumable official runs with a synthetic kit (TRAIN no-op control, no dataset or state access)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.inference import __main__ as inference_main
from src.inference.official import KitError, OfficialRecord
from src.inference.pipeline import read_resume_log
from src.tooling.registry import ToolRegistry


IDS = [f"synthetic-{index}" for index in range(1, 5)]


class Kit:
    split = "train"
    registry = ToolRegistry(())
    crash_on = None

    def __init__(self, *args):
        self.opened = []

    def rows(self, sample_ids):
        for sample_id in IDS:
            if sample_ids is None or sample_id in sample_ids:
                yield {"id": sample_id, "audio": sample_id, "audio_text": "test request", "history": [],
                       "context": {"installed_apps": {}}}

    def audio_path(self, sample_id, relative_path, *, history):
        return Path(relative_path)

    def open_environment(self, sample):
        if sample.sample_id == self.crash_on:
            raise KitError("simulated disconnect")
        self.opened.append(sample.sample_id)
        return SimpleNamespace(read_output=lambda: ([], f"test-only-{sample.sample_id}"))


def _run(tmp_path, monkeypatch, name, *, resume=None, crash_on=None):
    kit = Kit()
    kit.crash_on = crash_on
    monkeypatch.setattr(inference_main, "OfficialKit", lambda *args: kit)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"source_split": "train", "validation_ids": IDS}), encoding="utf-8")
    argv = ["official", "--split", "train", "--data-root", str(tmp_path), "--control-noop",
            "--train-manifest", str(manifest), "--output", str(tmp_path / name)]
    if resume is not None:
        argv += ["--resume-log", str(resume)]
    return kit, inference_main._run_official(inference_main._parser().parse_args(argv))


def test_resume_after_crash_and_truncated_line_equals_full_run(tmp_path, monkeypatch):
    _, _ = _run(tmp_path, monkeypatch, "full.jsonl")
    full = (tmp_path / "full.jsonl").read_bytes()
    log = tmp_path / "logs" / "progress.jsonl"
    with pytest.raises(KitError):
        _run(tmp_path, monkeypatch, "resumed.jsonl", resume=log, crash_on="synthetic-3")
    assert not (tmp_path / "resumed.jsonl").exists()
    assert log.read_bytes() == b"".join(full.splitlines(keepends=True)[:2])
    with log.open("ab") as sink:
        sink.write(b'{"id": "synthetic-3", "tool_')  # a write cut by the disconnect
    kit, summary = _run(tmp_path, monkeypatch, "resumed.jsonl", resume=log)
    assert kit.opened == ["synthetic-3", "synthetic-4"]
    assert summary["records"] == 4 and summary["attempted"] == 2
    assert (tmp_path / "resumed.jsonl").read_bytes() == full
    assert log.read_bytes() == full


def test_flag_off_writes_no_log(tmp_path, monkeypatch):
    kit, summary = _run(tmp_path, monkeypatch, "out.jsonl")
    assert kit.opened == IDS and summary["records"] == 4
    assert sorted(path.name for path in tmp_path.iterdir()) == ["manifest.json", "out.jsonl"]


def test_log_rejects_foreign_or_repeated_ids_and_keeps_time(tmp_path):
    log = tmp_path / "progress.jsonl"
    log.write_text('{"id": "a", "tool_calls": [], "state_hash": "h", "time": 1234}\n', encoding="utf-8")
    assert read_resume_log(log, {"a"}) == {"a": OfficialRecord("a", [], "h", 1234)}
    with pytest.raises(KitError):
        read_resume_log(log, {"b"})
    log.write_text(log.read_text(encoding="utf-8") * 2, encoding="utf-8")
    with pytest.raises(KitError):
        read_resume_log(log, {"a"})
