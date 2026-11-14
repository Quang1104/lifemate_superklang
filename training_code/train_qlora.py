"""QLoRA fine-tune of the planner on scripts/build_sft.py conversations (native tool-calling format).

Loss covers only the assistant turns of the current request (tool calls and the closing reply);
system prompt, history, user text and tool results are context. Logits are computed only at
those positions, so long prompts fit on a 16 GB T4.
"""

import argparse
import json
from pathlib import Path
import random
import re
import time

ASSISTANT = re.compile(r"<\|im_start\|>assistant\n(.*?<\|im_end\|>)", re.S)


def assistant_spans(text: str, start: int) -> list[tuple[int, int]]:
    """Character spans of assistant replies (including <|im_end|>) that begin at or after start."""
    return [(m.start(1), m.end(1)) for m in ASSISTANT.finditer(text) if m.start() >= start]


def label_tokens(offsets: list[tuple[int, int]], ids: list[int], spans: list[tuple[int, int]]) -> list[int]:
    labels = [-100] * len(ids)
    for index, (begin, end) in enumerate(offsets):
        if end > begin and any(s <= begin and end <= e for s, e in spans):
            labels[index] = ids[index]
    return labels


def encode(tokenizer, row: dict, max_tokens: int):
    render = lambda messages: tokenizer.apply_chat_template(
        messages, tools=row["tools"], tokenize=False, add_generation_prompt=False, enable_thinking=False)
    full = render(row["messages"])
    prefix = render(row["messages"][:row["target_start"]])
    if not full.startswith(prefix):
        return None  # template not prefix-stable for this row; skip rather than mislabel
    enc = tokenizer(full, return_offsets_mapping=True, add_special_tokens=False)
    if len(enc["input_ids"]) > max_tokens:
        return None
    labels = label_tokens(enc["offset_mapping"], enc["input_ids"], assistant_spans(full, len(prefix)))
    return (enc["input_ids"], labels) if any(label != -100 for label in labels) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=12_000)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=None, help="optimizer steps; overrides --epochs")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    examples, skipped = [], 0
    for row in rows:
        item = encode(tokenizer, row, args.max_tokens)
        skipped += item is None
        if item is not None:
            examples.append(item)
    lengths = sorted(len(ids) for ids, _ in examples)
    print(json.dumps({"rows": len(rows), "used": len(examples), "skipped": skipped,
                      "tokens_p50": lengths[len(lengths) // 2], "tokens_max": lengths[-1]}), flush=True)

    # T4 reports emulated bf16 as supported; emulated bf16 SDPA falls back to the quadratic math kernel and OOMs.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=dtype, device_map={"": 0}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    state_file = out / "trainer_state.json"
    done_steps = json.loads(state_file.read_text())["step"] if state_file.is_file() else 0
    if done_steps:
        model = PeftModel.from_pretrained(model, out, is_trainable=True)
        print("resumed from step", done_steps, flush=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    base = model.get_base_model()

    per_epoch = max(1, len(examples) // args.grad_accum)
    total = args.max_steps or int(per_epoch * args.epochs)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    schedule = get_cosine_schedule_with_warmup(optimizer, max(1, total // 20), total)
    for _ in range(done_steps):
        schedule.step()  # shortcut: optimizer moments restart on resume, only the LR schedule is restored
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    order = []
    while len(order) < total * args.grad_accum:
        epoch = list(range(len(examples)))
        random.shuffle(epoch)
        order += epoch
    order = order[done_steps * args.grad_accum: total * args.grad_accum]
    model.train()
    running, step, started = 0.0, done_steps, time.monotonic()
    for index, example_index in enumerate(order, start=1):
        ids, labels = examples[example_index]
        input_ids = torch.tensor([ids], device="cuda")
        targets = torch.tensor(labels[1:], device="cuda")
        positions = (targets != -100).nonzero().squeeze(-1)
        with torch.autocast("cuda", dtype=dtype):
            hidden = base.model(input_ids=input_ids).last_hidden_state[0, :-1]
            logits = base.lm_head(hidden[positions]).float()
        loss = F.cross_entropy(logits, targets[positions]) / args.grad_accum
        scaler.scale(loss).backward()
        running += loss.item()
        if index % args.grad_accum:
            continue
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        schedule.step()
        step += 1
        if step % 10 == 0 or step == total:
            print(json.dumps({"step": step, "of": total, "loss": round(running / 10 if step % 10 == 0 else running, 4),
                              "lr": schedule.get_last_lr()[0], "elapsed_s": round(time.monotonic() - started)}),
                  flush=True)
            running = 0.0
        if step % args.save_every == 0 or step == total:
            model.save_pretrained(out)
            state_file.write_text(json.dumps({"step": step, "total": total}))
    print("adapter saved to", out, flush=True)


if __name__ == "__main__":
    main()
