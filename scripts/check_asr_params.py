"""Count all Qwen3 planner + Whisper ASR parameters from local config JSON, without weights.

Counts tied embeddings once and includes Whisper's frozen encoder position embedding,
matching sum(p.numel() for p in model.parameters()), not only trainable parameters.
Unsupported architectures fail rather than estimating a competition budget.
"""

import argparse
import json
from pathlib import Path


def dimension(config: dict, key: str) -> int:
    value = config[key]
    if type(value) is not int or value <= 0:
        raise ValueError(f"Expected positive integer {key}")
    return value


def whisper_parameters(config: dict) -> int:
    if config.get("model_type") != "whisper":
        raise ValueError("Expected WhisperForConditionalGeneration config")
    d = dimension(config, "d_model")
    ef, df = (dimension(config, key) for key in ("encoder_ffn_dim", "decoder_ffn_dim"))
    enc, dec = (dimension(config, key) for key in ("encoder_layers", "decoder_layers"))
    vocab = dimension(config, "vocab_size")
    # Two biased kernel-3 convolutions; frozen encoder sinusoidal positions.
    frontend = 3 * d * dimension(config, "num_mel_bins") + d + 3 * d * d + d
    frontend += dimension(config, "max_source_positions") * d
    # Attention k has no bias; q/v/out have biases. FFN and layer norms have biases.
    encoder = enc * (4 * d * d + 2 * d * ef + 8 * d + ef) + 2 * d
    decoder = dec * (8 * d * d + 2 * d * df + 13 * d + df) + 2 * d
    decoder += vocab * d + dimension(config, "max_target_positions") * d
    output = 0 if config.get("tie_word_embeddings", True) else vocab * d
    return frontend + encoder + decoder + output


def qwen3_parameters(config: dict) -> int:
    if config.get("model_type") != "qwen3" or config.get("attention_bias", False):
        raise ValueError("Expected Qwen3ForCausalLM with attention_bias=false")
    h, f, layers, vocab = (dimension(config, key) for key in
                           ("hidden_size", "intermediate_size", "num_hidden_layers", "vocab_size"))
    heads, kv_heads = (dimension(config, key) for key in ("num_attention_heads", "num_key_value_heads"))
    hd = dimension(config, "head_dim") if "head_dim" in config else h // heads
    if hd <= 0 or ("head_dim" not in config and h % heads):
        raise ValueError("Invalid attention head dimension")
    q, kv = heads * hd, kv_heads * hd
    # q/k/v/o projections, gated FFN, two RMSNorms, q/k head RMSNorms.
    per_layer = 2 * h * q + 2 * h * kv + 3 * h * f + 2 * h + 2 * hd
    embeddings = vocab * h * (1 if config.get("tie_word_embeddings", False) else 2)
    return embeddings + layers * per_layer + h


def parameter_budget(planner: dict, asr: dict) -> dict:
    p, a = qwen3_parameters(planner), whisper_parameters(asr)
    return {"planner": p, "asr_large": a, "total": p + a,
            "planner_below_5b": p < 5_000_000_000, "total_below_6b": p + a < 6_000_000_000,
            "method": "config dimensions; all parameters including frozen; tied weights counted once"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-config", type=Path, required=True)
    parser.add_argument("--asr-config", type=Path, required=True)
    args = parser.parse_args()
    result = parameter_budget(json.loads(args.planner_config.read_text(encoding="utf-8")),
                              json.loads(args.asr_config.read_text(encoding="utf-8")))
    print(json.dumps(result, indent=2))
    if not result["planner_below_5b"] or not result["total_below_6b"]:
        raise SystemExit("Parameter budget exceeded (limits are strict)")


if __name__ == "__main__":
    main()
