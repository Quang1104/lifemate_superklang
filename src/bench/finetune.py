"""LoRA / QLoRA SFT of the planner on `sft_data` output. Resumes from the last checkpoint in out_dir."""

import hashlib
import json
from pathlib import Path

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
END_OF_TURN = "<|im_end|>"  # Qwen chat-template turn terminator


def vram_profile() -> dict:
    import torch

    gb = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0.0
    return {"vram_gb": round(gb, 1), "bf16": bool(gb) and torch.cuda.get_device_capability()[0] >= 8,
            # shortcut: fixed tiers (T4 16 GB / L4 24 GB / A100 40+ GB); tune if OOM or under-used.
            "load_in_4bit": gb < 35, "max_len": 8192 if gb >= 20 else 6144, "batch": 1, "grad_accum": 16}


def tokenize_examples(tokenizer, examples, max_len: int) -> tuple[list[dict], int]:
    """Loss only on the gold action tokens; examples longer than max_len are dropped and counted."""
    end = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    kept, dropped = [], 0
    for example in examples:
        prompt = list(tokenizer.apply_chat_template(example["messages"], add_generation_prompt=True,
                                                    enable_thinking=False, tokenize=True))
        answer = list(tokenizer(example["completion"], add_special_tokens=False)["input_ids"]) + [end]
        if len(prompt) + len(answer) > max_len:
            dropped += 1
            continue
        kept.append({"input_ids": prompt + answer, "labels": [-100] * len(prompt) + answer})
    return kept, dropped


def collate(batch: list[dict], pad_id: int) -> dict:
    import torch

    width = max(len(item["input_ids"]) for item in batch)
    pad = lambda seq, value: seq + [value] * (width - len(seq))
    return {"input_ids": torch.tensor([pad(i["input_ids"], pad_id) for i in batch]),
            "attention_mask": torch.tensor([pad([1] * len(i["input_ids"]), 0) for i in batch]),
            "labels": torch.tensor([pad(i["labels"], -100) for i in batch])}


def train(model_dir, data_path, out_dir, *, epochs: float = 1.0, max_steps: int = -1, lr: float = 1e-4,
          rank: int = 16, seed: int = 20261112, profile: dict | None = None) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    profile = profile or vram_profile()
    out_dir, data_path = Path(out_dir), Path(data_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    with data_path.open(encoding="utf-8") as source:
        data, dropped = tokenize_examples(tokenizer, (json.loads(line) for line in source if line.strip()),
                                          profile["max_len"])
    dtype = torch.bfloat16 if profile["bf16"] else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=dtype) if profile["load_in_4bit"] else None
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=dtype, quantization_config=quant,
                                                 device_map={"": 0}, attn_implementation="sdpa")
    if quant is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    args = TrainingArguments(
        output_dir=str(out_dir), per_device_train_batch_size=profile["batch"],
        gradient_accumulation_steps=profile["grad_accum"], learning_rate=lr, num_train_epochs=epochs,
        max_steps=max_steps, lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=10,
        save_steps=50, save_total_limit=2, bf16=profile["bf16"], fp16=not profile["bf16"],
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none", seed=seed, data_seed=seed, remove_unused_columns=False, dataloader_num_workers=0)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    trainer = Trainer(model=model, args=args, train_dataset=data, data_collator=lambda b: collate(b, pad_id))
    last = get_last_checkpoint(str(out_dir)) if out_dir.exists() else None
    trainer.train(resume_from_checkpoint=last)
    adapter = out_dir / "adapter"
    model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    meta = {"base_model": str(model_dir), "profile": profile, "examples": len(data), "dropped_too_long": dropped,
            "epochs": epochs, "max_steps": max_steps, "lr": lr, "rank": rank, "seed": seed, "resumed_from": last,
            "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest()}
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return adapter
