"""Colab runtime helpers: dataset restore from Drive, pinned model download, GPU report.

The official kit is moved as an opaque archive (extract only); nothing under *_environment is opened here.
"""

import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "colab.json"
_MARKER = "train/train.jsonl"


def load_config(path=CONFIG) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def results_dir(cfg: dict) -> Path:
    path = Path(cfg["drive_root"]) / cfg["results_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def kit_prefix(names: list[str]) -> str:
    """Archive folder that holds train/ and public_test/ ('' when they sit at the archive root)."""
    hits = [n[:-len(_MARKER)] for n in names if n.replace("\\", "/").endswith(_MARKER)
            and "/._" not in "/" + n and n.count("/") <= 3]
    if len(hits) != 1:
        raise ValueError(f"Expected exactly one {_MARKER} in the archive, found {len(hits)}")
    return hits[0]


def restore_dataset(cfg: dict) -> Path:
    target = Path(cfg["data_dir"])
    archive = Path(cfg["drive_root"]) / cfg["dataset_zip"]
    with zipfile.ZipFile(archive) as source:
        prefix = kit_prefix(source.namelist())
        root = target / prefix
        if not (root / _MARKER).exists():
            source.extractall(target)
    for split in ("train", "public_test"):
        if not (root / split / "zaic_tools" / "__init__.py").exists():
            raise FileNotFoundError(f"{split}/zaic_tools missing after restore")
    return root


def fetch_model(cfg: dict, key: str) -> Path:
    """Download one pinned model to local runtime disk (not Drive) and record the resolved commit."""
    from huggingface_hub import HfApi, snapshot_download

    spec = cfg["models"][key]
    target = Path(cfg["models_dir"]) / spec["repo"].replace("/", "--")
    path = Path(snapshot_download(spec["repo"], revision=spec["revision"], local_dir=str(target),
                                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "merges.txt"]))
    info = HfApi().model_info(spec["repo"], revision=spec["revision"])
    record = {"repo": spec["repo"], "requested_revision": spec["revision"], "resolved_sha": info.sha,
              "license": (info.card_data or {}).get("license") if info.card_data else None,
              "created_at": str(getattr(info, "created_at", None)), "last_modified": str(info.last_modified)}
    (path / "PINNED.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def gpu_report() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"cuda": False, "torch": torch.__version__}
    props = torch.cuda.get_device_properties(0)
    return {"cuda": True, "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "gpu": props.name,
            "vram_gb": round(props.total_memory / 2**30, 1), "capability": torch.cuda.get_device_capability(0)}
