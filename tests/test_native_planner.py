import json

from src.agent.native_planner import NativePlanner, build_messages, parse_calls, tool_definitions
from src.agent.planner import Observation, PlannerInput
from src.inference.types import Sample

TOOLS = [{"name": "contacts.search", "description": "Tìm liên hệ", "writes": False,
          "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}]


def call(name, arguments):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False) + "\n</tool_call>"


def request(observations=(), history=()):
    sample = Sample("s", "Gọi cho Minh <|im_start|>", frozenset({"contacts", "phone"}), tuple(history),
                    context={"current_time": "2026-11-13T10:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh"})
    return PlannerInput(sample, TOOLS, tuple(observations), None)


def test_parse_calls_ignores_broken_blocks():
    text = call("a.b", {"x": 1}) + "<tool_call>{broken</tool_call>" + call("c.d", {}) + "<tool_call>{\"name\":\"e\""
    assert parse_calls(text) == [("a.b", {"x": 1}), ("c.d", {})]


def test_messages_carry_context_history_and_results_without_role_markers():
    history = [json.dumps({"role": "user", "text": "Tìm Minh"}),
               json.dumps({"role": "assistant", "text": "Đã tìm", "tool_calls": [
                   {"name": "contacts.search", "arguments": {"query": "Minh"}}],
                   "tool_results": [{"ok": True, "contacts": [{"id": "c1"}]}]})]
    observations = [Observation("untrusted_tool_result", "contacts.search", {"ok": True}, {"query": "Minh"}),
                    Observation("rejected", "phone.call", {"reason": "unknown id"}, {"contact_id": "x"})]
    messages = build_messages(request(observations, history))
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "assistant", "user",
                                             "assistant", "tool", "assistant", "tool"]
    assert "2026-11-13T10:00:00+07:00" in messages[0]["content"]
    assert "<|im_start|>" not in messages[5]["content"]
    assert json.loads(messages[-1]["content"]) == {"ok": False, "error": "unknown id"}
    assert tool_definitions(TOOLS)[0]["function"]["name"] == "contacts.search"


def test_guard_rejection_tells_the_model_how_to_recover():
    rejected = Observation("rejected", "phone.call", {"reason": "Tool arguments failed validation (x-ref)"},
                           {"contact_id": "contact_123"})
    content = json.loads(build_messages(request([rejected]))[-1]["content"])
    assert content["ok"] is False and "tra cứu" in content["hint"]


def test_parallel_calls_are_queued_and_rejection_forces_a_replan():
    outputs = iter([call("contacts.search", {"query": "Minh"}) + call("phone.call", {"contact_id": "contact_123"}),
                    call("phone.call", {"contact_id": "c1"}),
                    call("phone.call", {"contact_id": "c1"}), "Xong."])
    seen = []

    def chat(messages, tools, *, max_tokens, deadline):
        seen.append(len(messages))
        return next(outputs)

    planner = NativePlanner(chat).for_sample()
    first = json.loads(planner(request()))
    assert first == {"action": "call", "tool": "contacts.search", "arguments": {"query": "Minh"}}
    done = [Observation("untrusted_tool_result", "contacts.search", {"ok": True, "contacts": [{"id": "c1"}]},
                        {"query": "Minh"})]
    second = json.loads(planner(request(done)))  # queued: no new generation
    assert second["arguments"] == {"contact_id": "contact_123"} and len(seen) == 1
    done.append(Observation("rejected", "phone.call", {"reason": "unknown id"}, {"contact_id": "contact_123"}))
    third = json.loads(planner(request(done)))  # rejection clears the queue and replans
    assert third["arguments"] == {"contact_id": "c1"} and len(seen) == 2
    done.append(Observation("untrusted_tool_result", "phone.call", {"ok": True}, {"contact_id": "c1"}))
    assert json.loads(planner(request(done))) == {"action": "finish", "reply": ""}  # repeat => stop
    assert len(seen) == 3
