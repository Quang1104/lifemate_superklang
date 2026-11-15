"""Full fine-tune of the Whisper ASR (PhoWhisper) on TRAIN audio with the released transcripts.

Uses the current request and every user turn of the history of the TRAIN IDs outside the validation
split. Each epoch reports word error rate on the current-request audio of the validation IDs (epoch 0 is
the untouched model) and keeps the best epoch in --output-dir, loadable by LocalWhisper (--asr-dir).
Needs a bf16 GPU (L4/A100).
"""

import argparse
import json
from pathlib import Path
import random
import re
import time
import unicodedata


def words(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text.lower()))


def word_errors(reference: list[str], hypothesis: list[str]) -> int:
    row = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        diagonal, row[0] = row[0], i
        for j, hyp in enumerate(hypothesis, 1):
            diagonal, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, diagonal + (ref != hyp))
    return row[-1]


def wer(pairs: list[tuple[str, str]]) -> float:
    """Corpus WER over (reference, hypothesis) pairs."""
    errors = sum(word_errors(words(ref), words(hyp)) for ref, hyp in pairs)
    return errors / max(1, sum(len(words(ref)) for ref, _ in pairs))


def utterances(kit, keep, *, history: bool = True) -> list[tuple[Path, str]]:
    """(audio path, transcript) for the rows whose ID passes keep; history adds the earlier user turns."""
    items = []
    for row in kit.rows():
        if not keep(row["id"]):
            continue
        items.append((kit.audio_path(row["id"], row["audio"], history=False), row["audio_text"]))
        if history:
            items += [(kit.audio_path(row["id"], turn["audio"], history=True), turn["audio_text"])
                      for turn in row["history"] if turn.get("role") == "user" and turn.get("audio_text")]
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation", type=Path, default=Path("research/validation_split.json"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261115)
    args = parser.parse_args()
    if any(part.endswith("_environment") for part in args.data_root.resolve().parts):
        raise ValueError("Forbidden root")

    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration, get_linear_schedule_with_warmup

    from src.agent.local_model import _read_audio
    from src.inference.official import OfficialKit

    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "needs a bf16 GPU (L4/A100)"
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    validation = set(json.loads(args.validation.read_text(encoding="utf-8"))["validation_ids"])
    kit = OfficialKit(args.data_root, "train")
    train = utterances(kit, lambda i: i not in validation)
    evaluation = utterances(kit, lambda i: i in validation, history=False)
    print(json.dumps({"train_utterances": len(train), "eval_utterances": len(evaluation)}), flush=True)

    processor = AutoProcessor.from_pretrained(args.model_dir)
    processor.tokenizer.set_prefix_tokens(language="vietnamese", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir, torch_dtype=torch.float32).cuda()
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    start_id = model.config.decoder_start_token_id

    def features(paths):
        audio = [_read_audio(path) for path in paths]
        return processor.feature_extractor(audio, sampling_rate=16_000, return_tensors="pt").input_features.cuda()

    def evaluate() -> float:
        model.eval()
        pairs = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for begin in range(0, len(evaluation), args.batch):
                chunk = evaluation[begin:begin + args.batch]
                out = model.generate(input_features=features([p for p, _ in chunk]), language="vi",
                                     task="transcribe", max_new_tokens=256)
                texts = processor.batch_decode(out, skip_special_tokens=True)
                pairs += [(ref, hyp) for (_, ref), hyp in zip(chunk, texts)]
        model.train()
        return wer(pairs)

    best = evaluate()
    print(json.dumps({"epoch": 0, "wer": round(best, 4)}), flush=True)
    steps = args.epochs * ((len(train) + args.batch - 1) // args.batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    schedule = get_linear_schedule_with_warmup(optimizer, min(50, steps // 10), steps)
    started, step = time.time(), 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train)
        for begin in range(0, len(train), args.batch):
            chunk = train[begin:begin + args.batch]
            labels = [processor.tokenizer(text).input_ids for _, text in chunk]
            labels = [ids[1:] if ids and ids[0] == start_id else ids for ids in labels]  # the model prepends it
            width = max(map(len, labels))
            labels = torch.tensor([ids + [-100] * (width - len(ids)) for ids in labels]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_features=features([p for p, _ in chunk]), labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(), schedule.step(), optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0:
                print(json.dumps({"step": step, "of": steps, "loss": round(loss.item(), 4),
                                  "elapsed_s": round(time.time() - started)}), flush=True)
        score = evaluate()
        print(json.dumps({"epoch": epoch, "wer": round(score, 4), "elapsed_s": round(time.time() - started)}),
              flush=True)
        if score < best:
            best = score
            args.output_dir.mkdir(parents=True, exist_ok=True)
            # fp16 copy on disk; the fp32 weights the optimizer holds stay untouched.
            half = {name: tensor.half() for name, tensor in model.state_dict().items()}
            model.save_pretrained(args.output_dir, state_dict=half, safe_serialization=True)
            processor.save_pretrained(args.output_dir)
            print(json.dumps({"saved": str(args.output_dir), "epoch": epoch}), flush=True)
    print(json.dumps({"best_wer": round(best, 4)}), flush=True)


if __name__ == "__main__":
    main()
