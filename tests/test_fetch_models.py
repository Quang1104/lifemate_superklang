from pathlib import Path

import pytest

fetch = pytest.importorskip("docker.fetch_models")  # needs huggingface_hub


def _pins(tmp_path, monkeypatch):
    # In the image fetch_models.py sits next to configs/; point it at a stub pin file.
    (tmp_path / "configs").mkdir()
    source = Path(__file__).resolve().parents[1] / "configs" / "colab_models.json"
    (tmp_path / "configs" / "colab_models.json").write_text(source.read_text(encoding="utf-8"))
    monkeypatch.setattr(fetch, "__file__", str(tmp_path / "fetch_models.py"))


def test_fine_tuned_asr_in_adapter_folder_replaces_download(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fetch, "snapshot_download", lambda **kw: calls.append(kw["local_dir"]))
    monkeypatch.setattr(fetch, "merge", lambda *a: 0)
    _pins(tmp_path, monkeypatch)
    adapter = tmp_path / "adapter"
    (adapter / "asr").mkdir(parents=True)
    (adapter / "asr" / "config.json").write_text("{}")
    base = tmp_path / "base"
    base.mkdir()
    fetch.main(str(adapter), base=str(base), planner=str(tmp_path / "planner"), asr=str(tmp_path / "asr"))
    assert (tmp_path / "asr" / "config.json").is_file()
    assert calls == [str(base)]  # only the base planner is downloaded


def test_without_tuned_asr_the_pinned_asr_is_downloaded(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fetch, "snapshot_download", lambda **kw: calls.append(kw["local_dir"]))
    monkeypatch.setattr(fetch, "merge", lambda *a: 0)
    _pins(tmp_path, monkeypatch)
    base = tmp_path / "base"
    base.mkdir()
    fetch.main(str(tmp_path / "adapter"), base=str(base), planner=str(tmp_path / "p"), asr=str(tmp_path / "asr"))
    assert calls == [str(base), str(tmp_path / "asr")]
