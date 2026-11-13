"""Offline Hugging Face planner and Whisper ASR backends (the names `src.inference` already imports).

Heavy libraries are imported lazily so CPU-only checks never need torch/transformers.
The prompt format here is shared by inference and SFT data, so train and test see the same text.
"""

import json
from pathlib import Path
from time import monotonic

SYSTEM = """You are LifeMate, a Vietnamese phone assistant. Fulfil the user's LATEST request by calling tools.
Reply with exactly ONE JSON object per turn and nothing else:
{"action":"call","tool":"<tool name>","arguments":{...}}
{"action":"finish","reply":"<short Vietnamese reply>"}
Rules:
- Use only the listed tools and their exact argument names, types and enums. Omit optional arguments the user did not specify.
- IDs must be copied from tool results or history; never invent one. Look it up with a read tool first.
- If required information is missing and no tool can look it up, call sys.ask_missing_info.
- One call per turn. Never repeat a call. Avoid reads that the request does not need.
- Resolve relative dates/times from context.current_time; write ISO 8601 with the context UTC offset.
- History actions are already done; do not redo them.
- When the request is fully handled, or no installed tool can handle it, finish."""

RESULT_CHARS = 4000  # shortcut: hard truncation of tool results, upgrade to field-aware trimming if IDs get cut
FALLBACK = '{"action":"finish","reply":""}'


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_messages(request) -> list[dict]:
    """Chat messages for one planner step: tools, context, history, request, then prior steps."""
    sample = request.sample
    tools = [{k: t[k] for k in ("name", "description", "parameters")} for t in request.tools]
    user = "\n".join(["TOOLS:", *map(_dump, tools), "CONTEXT: " + _dump(sample.context),
                      "HISTORY:", *sample.history, "REQUEST: " + sample.user_request])
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    for seen in request.observations:
        if seen.kind == "rejected":
            messages.append({"role": "user", "content": "REJECTED: " + _dump(seen.content)})
            continue
        messages.append({"role": "assistant", "content": _dump(
            {"action": "call", "tool": seen.tool, "arguments": seen.arguments})})
        messages.append({"role": "user", "content": "RESULT: " + _dump(seen.content)[:RESULT_CHARS]})
    return messages


def extract_json(text: str) -> str:
    """Keep the outermost {...} so code fences or a stray think block do not fail strict parsing."""
    text = text.split("</think>")[-1]
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else text.strip()


def initialize_runtime(seed: int = 20261112) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _dtype(device: str):
    import torch

    if not device.startswith("cuda"):
        return torch.float32
    # T4 (sm75) only emulates bf16; use fp16 there.
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


class LocalPlanner:
    """Greedy, non-thinking chat model; one JSON action per call."""

    def __init__(self, model_dir, *, device: str = "cuda", adapter_dir=None, load_in_4bit: bool = False,
                 max_context: int = 16_384, max_new_tokens: int = 512) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = _dtype(device)
        kwargs = {"torch_dtype": dtype, "attn_implementation": "sdpa", "device_map": {"": device}}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)
        if adapter_dir:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, str(adapter_dir))
            if not load_in_4bit:
                self.model = self.model.merge_and_unload()
        self.model.eval()
        self.parameter_count = self.model.num_parameters()
        self.device, self.max_context, self.max_new_tokens = device, max_context, max_new_tokens
        self.stats: list[dict] = []

    def __call__(self, request) -> str:
        import torch

        started = monotonic()
        ids = self.tokenizer.apply_chat_template(build_messages(request), add_generation_prompt=True,
                                                 enable_thinking=False, return_tensors="pt").to(self.device)
        prompt_tokens = ids.shape[1]
        if prompt_tokens + self.max_new_tokens > self.max_context:  # rule: input + output <= 16,384
            self.stats.append({"prompt_tokens": prompt_tokens, "new_tokens": 0, "seconds": 0.0, "overflow": True})
            return FALLBACK
        # shortcut: no KV reuse between steps (full re-prefill); add prefix caching if latency > 30 s.
        with torch.inference_mode():
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                      max_new_tokens=self.max_new_tokens,
                                      pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        new = out[0, prompt_tokens:]
        self.stats.append({"prompt_tokens": prompt_tokens, "new_tokens": int(new.shape[0]),
                           "seconds": round(monotonic() - started, 2)})
        return extract_json(self.tokenizer.decode(new, skip_special_tokens=True))


class LocalWhisper:
    """Vietnamese transcription of one wav path."""

    def __init__(self, asr_dir, *, device: str = "cuda", language: str = "vietnamese") -> None:
        from transformers import pipeline

        self.pipe = pipeline("automatic-speech-recognition", model=str(asr_dir), device=device,
                             torch_dtype=_dtype(device), chunk_length_s=30)
        self.parameter_count = self.pipe.model.num_parameters()
        self.language = language

    def __call__(self, path: Path) -> str:
        return self.pipe(str(path), generate_kwargs={"language": self.language, "task": "transcribe"})["text"]
