from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from training_code.stream_merge import merge  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def test_stream_merge_matches_peft_merge_and_unload_bit_for_bit(tmp_path: Path):
    from safetensors.torch import load_file

    torch.manual_seed(0)
    config = transformers.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)
    base = transformers.LlamaForCausalLM(config).to(torch.bfloat16)  # the shipped base is stored in bf16
    base.save_pretrained(tmp_path / "base", max_shard_size="20KB")
    lora = peft.get_peft_model(transformers.LlamaForCausalLM(config),
                               peft.LoraConfig(r=4, lora_alpha=8, target_modules=TARGETS, task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, value in lora.named_parameters():
            if "lora_B" in name:
                value.normal_(0, 0.5)  # B starts at zero; make the merge non-trivial
    lora.save_pretrained(tmp_path / "adapter")

    # Reference: exactly what training_code/merge_lora.py does.
    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    reference = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "base", **{dtype_key: torch.float16})
    reference = peft.PeftModel.from_pretrained(reference, tmp_path / "adapter").merge_and_unload()
    expected = reference.state_dict()

    count = merge(tmp_path / "base", tmp_path / "adapter", tmp_path / "out")
    actual = {}
    for shard in (tmp_path / "out").glob("*.safetensors"):
        actual.update(load_file(shard))
    assert count == 2 * len(TARGETS)
    assert len(list((tmp_path / "out").glob("*.safetensors"))) > 1
    for name, value in actual.items():
        assert torch.equal(value, expected[name]), name
