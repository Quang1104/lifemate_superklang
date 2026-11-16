import json

from scripts.build_sft import DONE, NOTHING, conversation
from src.inference.types import Sample
from training_code.train_qlora import assistant_spans, label_tokens

TOOLS = [{"name": "contacts.search", "description": "Tìm", "writes": False,
          "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}]


def sample(history=()):
    return Sample("s1", "Gọi cho Minh", frozenset({"contacts"}), tuple(history),
                  context={"current_time": "2026-11-13T10:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh"})


def test_conversation_has_one_call_and_result_per_gold_call_and_a_closing_reply():
    history = [json.dumps({"role": "user", "text": "Chào"}), json.dumps({"role": "assistant", "text": "Chào bạn"})]
    calls = [{"name": "contacts.search", "arguments": {"query": "Minh"}}]
    item = conversation(sample(history), TOOLS, calls, [{"ok": True, "contacts": [{"id": "c1"}]}])
    roles = [m["role"] for m in item["messages"]]
    assert roles == ["system", "user", "assistant", "user", "assistant", "tool", "assistant"]
    assert item["target_start"] == 4  # training starts after the current user message, not the history
    assert item["messages"][-1] == {"role": "assistant", "content": DONE}
    assert conversation(sample(), TOOLS, [], [])["messages"][-1]["content"] == NOTHING


def test_labels_cover_only_assistant_spans_after_the_start():
    text = ("<|im_start|>user\nA<|im_end|>\n<|im_start|>assistant\nold<|im_end|>\n"
            "<|im_start|>user\nB<|im_end|>\n<|im_start|>assistant\nnew<|im_end|>\n")
    start = text.index("<|im_start|>user\nB")
    spans = assistant_spans(text, start)
    assert [text[a:b] for a, b in spans] == ["new<|im_end|>"]
    pieces = [text[i:i + 4] for i in range(0, len(text), 4)]  # fake tokenizer: 4-char tokens
    offsets = [(i, min(i + 4, len(text))) for i in range(0, len(text), 4)]
    labels = label_tokens(offsets, list(range(len(pieces))), spans)
    kept = "".join(pieces[i] for i, label in enumerate(labels) if label != -100)
    assert "old" not in kept and kept and set(kept) <= set("new<|im_end|>")
