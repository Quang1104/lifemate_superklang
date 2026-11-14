"""Merge a QLoRA adapter into the fp16 base so `src.inference --model-dir <merged>` works unchanged."""

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float16)
    count = sum(p.numel() for p in base.parameters())
    merged = PeftModel.from_pretrained(base, args.adapter_dir).merge_and_unload()
    assert sum(p.numel() for p in merged.parameters()) == count, "merge changed the parameter count"
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.model_dir).save_pretrained(args.output_dir)
    for extra in ("generation_config.json", "LICENSE", "README.md"):
        if (Path(args.model_dir) / extra).is_file():
            shutil.copy(Path(args.model_dir) / extra, Path(args.output_dir) / extra)
    print("merged", count, "parameters into", args.output_dir)


if __name__ == "__main__":
    main()
