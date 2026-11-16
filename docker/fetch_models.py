"""Docker build step only: fetch the pinned base planner and ASR, merge the LoRA adapter, drop the base.

Running all of it in one RUN keeps the 8 GB base checkpoint and the adapter out of every image layer.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from training_code.stream_merge import merge  # noqa: E402
from training_code.model_pins import checked_pin  # noqa: E402

# Same file set the Colab notebook downloads for the measured runs.
PATTERNS = ["*.json", "*.jinja", "*.safetensors", "*.bin", "merges.txt", "LICENSE", "README.md"]
# The embedder repo also ships ONNX/OpenVINO exports and a second copy of the weights; take only what is loaded.
EMBEDDER_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                  "sentencepiece.bpe.model", "model.safetensors"]


def fetch_embedder(directory: str) -> None:
    """The pinned sentence embedder for LIFEMATE_EMBED_RETRIEVAL (configs/colab_models.json role `embedder`)."""
    pins = json.loads((Path(__file__).resolve().parent / "configs" / "colab_models.json").read_text(encoding="utf-8"))
    snapshot_download(**checked_pin(pins, "embedder"), local_dir=directory, allow_patterns=EMBEDDER_FILES)
    shutil.rmtree(Path(directory) / ".cache", ignore_errors=True)


def main(adapter: str, base: str = "/tmp/base", planner: str = "/models/planner", asr: str = "/models/asr",
         asr_role: str = "asr") -> None:
    pins = json.loads((Path(__file__).resolve().parent / "configs" / "colab_models.json").read_text(encoding="utf-8"))
    tuned = Path(adapter) / "asr"  # optional fine-tuned ASR (medium or large) shipped with the adapter
    asr_pin = None if (tuned / "config.json").is_file() else checked_pin(pins, asr_role)
    snapshot_download(**checked_pin(pins, "fallback_planner"), local_dir=base, allow_patterns=PATTERNS)
    if asr_pin is None:
        shutil.copytree(tuned, asr)
    else:
        snapshot_download(**asr_pin, local_dir=asr, allow_patterns=PATTERNS)
    print("merged", merge(Path(base), Path(adapter), Path(planner)), "LoRA modules into", planner, flush=True)
    shutil.rmtree(base)
    for directory in (Path(asr), Path(planner)):
        shutil.rmtree(directory / ".cache", ignore_errors=True)  # hub download metadata, not model files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", nargs="?")
    parser.add_argument("--asr-role", choices=("asr", "asr_large"), default="asr")
    parser.add_argument("--embedder", metavar="DIR", help="fetch only the pinned sentence embedder into DIR")
    args = parser.parse_args()
    if args.embedder:
        fetch_embedder(args.embedder)
    elif args.adapter:
        main(args.adapter, asr_role=args.asr_role)
    else:
        parser.error("pass the adapter folder, or --embedder DIR")
