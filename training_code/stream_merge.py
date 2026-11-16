"""Merge a LoRA adapter into base safetensors one shard at a time.

Gives the same fp16 weights as merge_lora.py (fp16 base + PEFT merge_and_unload on CPU) while
holding only one shard in RAM, so it fits the Docker build's memory limit.
"""

import argparse
import json
from pathlib import Path
import shutil


def merge(base: Path, adapter: Path, output: Path) -> int:
    import torch
    from safetensors.torch import load_file, save_file

    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    scale = config["lora_alpha"] / config["r"]
    pairs: dict[str, dict[str, "torch.Tensor"]] = {}
    for key, tensor in load_file(adapter / "adapter_model.safetensors").items():
        # base_model.model.<module>.lora_A.weight -> <module>.weight
        module, part = key.removeprefix("base_model.model.").rsplit(".lora_", 1)
        pairs.setdefault(module + ".weight", {})[part.split(".")[0]] = tensor
    output.mkdir(parents=True, exist_ok=True)
    for path in base.iterdir():
        if path.is_file() and path.suffix != ".safetensors":
            shutil.copy(path, output / path.name)
    merged = set()
    for shard in sorted(base.glob("*.safetensors")):
        # merge_lora.py loads the base as fp16, so every floating tensor is cast first.
        tensors = {name: value.to(torch.float16) if value.is_floating_point() else value
                   for name, value in load_file(shard).items()}
        for name, lora in pairs.items():
            if name in tensors:
                delta = (lora["B"] @ lora["A"]) * scale  # PEFT get_delta_weight, adapter dtype
                weight = tensors[name]
                weight += delta  # same in-place promotion as PEFT's `weight.data += delta`
                merged.add(name)
        save_file(tensors, str(output / shard.name), metadata={"format": "pt"})
    missing = sorted(set(pairs) - merged)
    if missing:
        raise SystemExit(f"Adapter modules not found in the base checkpoint: {missing[:3]}")
    return len(merged)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print("merged", merge(args.base, args.adapter, args.output), "LoRA modules into", args.output)


if __name__ == "__main__":
    main()
