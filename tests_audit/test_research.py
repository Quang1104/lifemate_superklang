"""Synthetic fixtures only. No competition data or environment access."""
import json
from pathlib import Path
import random
import struct

import pytest

from research.data_audit import audit, analyze_rows, extract_tools, metadata_info, read_jsonl, safe_path
from research.metrics import asr_metrics, exact_call_metrics
from research.validation_split import make_split


def row(index, app="calendar", text=None, audio=None, names=None):
    return {"id": f"s{index}", "audio": audio or f"audio/{index}.wav",
            "audio_text": text or f"synthetic request {index}",
            "context": {"installed_apps": [app]},
            "label": {"tool_calls": [{"name": name, "arguments": {}} for name in
                                      (names if names is not None else [f"{app}.create"])]}}


def test_forbidden_paths_rejected_before_file_io(monkeypatch):
    def no_stat(*args, **kwargs):
        raise AssertionError("Must reject before file IO")
    monkeypatch.setattr(Path, "is_file", no_stat)
    for path in ["train/train_environment/x", "a/state.json", "a/PUBLIC_TEST_ENVIRONMENT/x"]:
        with pytest.raises(ValueError, match="Prohibited"):
            safe_path(Path(path))


def test_appledouble_recognized_without_parsing_as_data(tmp_path):
    path = tmp_path / "._train.jsonl"
    path.write_bytes(b"\x00\x05\x16\x07" + b"\x00" * 20 + struct.pack(">H", 2))
    info = metadata_info(path)
    assert info["appledouble"] and info["entry_count"] == 2
    report = audit(tmp_path)
    assert report["status"] == "BLOCKED_MISSING_DATA"
    assert isinstance(report["splits"]["train"]["statistics"], str)


def test_audit_never_enumerates_directories(tmp_path, monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("No directory traversal allowed")
    monkeypatch.setattr(Path, "iterdir", deny)
    monkeypatch.setattr(Path, "glob", deny)
    monkeypatch.setattr(Path, "rglob", deny)
    assert audit(tmp_path)["status"] == "BLOCKED_MISSING_DATA"


def test_public_labels_not_used(tmp_path):
    sample = row(1)
    sample["label"] = "unparseable and must be ignored"
    stats = analyze_rows([sample], split="public_test", split_dir=tmp_path)
    assert stats["train_labels"].startswith("NOT INSPECTED")


def test_train_statistics_and_audio_escape(tmp_path):
    samples = [row(1, names=[]), row(2, names=["sys.ask_missing_info"]),
               row(3, text="ngày mai", names=["calendar.create", "email.send"], audio="../outside.wav")]
    stats = analyze_rows(samples, split="train", split_dir=tmp_path)
    assert stats["train_labels"]["cross_app_samples"] == 1
    assert stats["train_labels"]["gold_empty_samples"] == 1
    assert stats["train_labels"]["missing_info_samples"] == 1
    assert stats["referenced_audio"]["rejected_path"] == 1


def test_catalog_detection_and_unknown_rejection():
    catalog = {"apps": [{"name": "calendar", "tools": [{"name": "create", "parameters": {"type": "object"}}]}]}
    assert list(extract_tools(catalog)) == ["calendar.create"]
    with pytest.raises(ValueError, match="No recognizable"):
        extract_tools({"unrecognized": "format"})


def test_split_stable_group_safe_and_stratified():
    samples = [row(i, app="email" if i % 2 else "calendar") for i in range(30)]
    samples[0]["audio_text"] = samples[1]["audio_text"]
    samples[1]["audio"] = samples[2]["audio"]  # transitive duplicate grouping
    result = make_split(samples)
    shuffled = samples[:]
    random.Random(99).shuffle(shuffled)
    assert result == make_split(shuffled)
    validation = set(result["validation_ids"])
    assert len(validation & {"s0", "s1", "s2"}) in (0, 3)
    assert validation and set(result["train_ids"])
    assert not validation & set(result["train_ids"])
    assert not result["uncovered_validation_strata"]


def test_split_rejects_all_duplicates_and_public_path(tmp_path):
    with pytest.raises(ValueError, match="one duplicate group"):
        make_split([row(1, text="same"), row(2, text="same")])
    with pytest.raises(ValueError, match="TRAIN"):
        read_jsonl(tmp_path / "public_test.jsonl", train_only=True)


def test_exact_metric_counts_duplicate_calls_and_empty():
    call = {"name": "calendar.create", "arguments": {"title": "meeting"}}
    result = exact_call_metrics([call], [call, call])
    assert result["precision"] == 0.5 and result["recall"] == 1
    assert result["official_competition_score"] == "NOT MEASURED"
    assert exact_call_metrics([], [call])["f1"] == 0
    assert exact_call_metrics([], [])["f1"] == 1


def test_asr_metrics_and_empty_reference():
    assert asr_metrics(["xin chào"], ["xin cháo"])["wer"] == 0.5
    assert asr_metrics(["Xin   chào"], ["xin chào"])["cer"] == 0
    assert asr_metrics([""], ["word"])["wer"] is None
    with pytest.raises(ValueError):
        asr_metrics(["a"], [])
