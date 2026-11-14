"""Planner that speaks the checkpoint's own tool-calling chat format.

Qwen is trained to read tools through its chat template and to answer with
<tool_call>{"name": ..., "arguments": ...}</tool_call>. Using that format, with tool
results fed back as tool messages, replaces the project-specific JSON protocol for
the model while the runner keeps every existing guard.
"""

from collections.abc import Callable
import json

from .planner import PlannerInput

_SYSTEM = """Bạn là LifeMate, trợ lý giọng nói tiếng Việt điều khiển các ứng dụng của người dùng bằng tool.
Câu của người dùng là bản chép lời nói, có thể sai chính tả nhẹ.

Quy tắc:
1. Thực hiện ĐỦ mọi việc người dùng yêu cầu trong câu hiện tại, theo đúng thứ tự, bằng các tool được cung cấp.
2. ID (contact_id, device_id, event_id, bill_id...) chỉ được lấy từ kết quả tool hoặc lịch sử. Chưa có thì gọi tool tra cứu trước, KHÔNG tự đặt ID.
3. Chỉ gọi tool cần thiết. Không tra cứu thừa, không lặp lại lệnh đã chạy, không làm lại việc trong lịch sử.
4. Thời gian viết dạng ISO 8601 có múi giờ, tính từ "Thời gian hiện tại" bên dưới (ví dụ "3 giờ chiều mai"). Số tiền viết số nguyên đồng ("500 nghìn" = 500000).
5. Nếu người dùng đổi ý hoặc bảo đừng làm ("thôi khỏi", "đừng", "à không") thì KHÔNG làm việc đó; với câu sửa lại thì dùng giá trị cuối cùng.
6. Câu có điều kiện ("nếu... thì..."): tra cứu trước, chỉ làm nhánh thỏa điều kiện.
7. Thiếu thông tin bắt buộc mà không tra được bằng tool: gọi sys.ask_missing_info với tool_name và params còn thiếu.
8. Yêu cầu nằm ngoài khả năng của các tool, hoặc chỉ là trò chuyện: KHÔNG gọi tool nào.
9. Khi đã làm xong mọi việc, trả lời một câu ngắn và không gọi tool nữa."""

_RESULT_CHARS = 6_000

# The guard's terse rule names gave the model nothing to act on; smoke40 runs repeated the rejected call and stopped.
_REJECTION_HINTS = {
    "(x-ref)": "ID này chưa có trong kết quả tool hay lịch sử. Gọi tool tra cứu (list/search/get) để lấy ID thật.",
    "(additionalProperties)": "Có tham số không có trong schema của tool. Chỉ dùng đúng tên tham số đã khai báo.",
    "(required)": "Thiếu tham số bắt buộc. Tra cứu để có giá trị, hoặc gọi sys.ask_missing_info.",
    "(enum)": "Giá trị không thuộc danh sách cho phép của tham số.",
}


def tool_definitions(tools: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {"name": tool["name"], "description": tool["description"],
                                              "parameters": tool["parameters"]}} for tool in tools]


def parse_calls(text: str) -> list[tuple[str, dict]]:
    calls = []
    for block in text.split("<tool_call>")[1:]:
        try:
            value = json.loads(block.split("</tool_call>")[0])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("name"), str) and isinstance(value.get("arguments"), dict):
            calls.append((value["name"], value["arguments"]))
    return calls


def _safe(text: str) -> str:
    # Data must never open or close a chat role or a tool-call block.
    return text.replace("<|", "<​|").replace("<tool_call>", "<​tool_call>").replace(
        "</tool_call>", "<​/tool_call>")


def _dump(value) -> str:
    return _safe(json.dumps(value, ensure_ascii=False, separators=(",", ":")))[:_RESULT_CHARS]


def _call_message(name: str, arguments: dict) -> dict:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}]}


def _rejection(reason: str) -> dict:
    hints = [hint for rule, hint in _REJECTION_HINTS.items() if rule in reason]
    return {"ok": False, "error": reason, **({"hint": hints[0]} if hints else {})}


def build_messages(request: PlannerInput) -> list[dict]:
    sample = request.sample
    context = sample.context or {}
    system = (f"{_SYSTEM}\n\nThời gian hiện tại: {context.get('current_time')} ({context.get('timezone')})\n"
              f"Vị trí: {_dump(context.get('location'))}\nHồ sơ người dùng: {_dump(context.get('user_profile'))}")
    messages = [{"role": "system", "content": system}]
    for raw in sample.history:
        try:
            turn = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if turn.get("role") == "user":
            messages.append({"role": "user", "content": _safe(str(turn.get("text", "")))})
            continue
        calls = [call for call in turn.get("tool_calls", ()) if isinstance(call, dict)
                 and isinstance(call.get("name"), str) and isinstance(call.get("arguments"), dict)]
        results = list(turn.get("tool_results", ()))
        for index, call in enumerate(calls):
            messages.append(_call_message(call["name"], call["arguments"]))
            messages.append({"role": "tool", "content": _dump(results[index] if index < len(results) else {})})
        if turn.get("text"):
            messages.append({"role": "assistant", "content": _safe(str(turn["text"]))})
    messages.append({"role": "user", "content": _safe(sample.user_request)})
    for observation in request.observations:
        if observation.tool is None:
            continue
        messages.append(_call_message(observation.tool, observation.arguments or {}))
        content = observation.content if observation.kind != "rejected" else _rejection(
            observation.content.get("reason", "rejected"))
        messages.append({"role": "tool", "content": _dump(content)})
    return messages


def _key(name: str, arguments: dict) -> tuple[str, str]:
    return name, json.dumps(arguments, ensure_ascii=False, sort_keys=True)


class NativePlanner:
    """Per-sample planner; queues parallel tool calls so one generation can drive several steps."""

    def __init__(self, chat: Callable[..., str], *, max_tokens: int = 384) -> None:
        self._chat = chat
        self._max_tokens = max_tokens
        self._queue: list[tuple[str, dict]] = []

    def for_sample(self) -> "NativePlanner":
        return NativePlanner(self._chat, max_tokens=self._max_tokens)

    def __call__(self, request: PlannerInput) -> str:
        observations = request.observations
        if observations and observations[-1].kind == "rejected":
            self._queue.clear()  # Later queued calls may depend on the rejected one; replan with its feedback.
        if not self._queue:
            text = self._chat(build_messages(request), tool_definitions(request.tools),
                              max_tokens=self._max_tokens, deadline=request.deadline)
            attempted = {_key(item.tool, item.arguments or {}) for item in observations if item.tool}
            # A repeated call means the model has nothing new to do; stopping beats looping.
            for name, arguments in parse_calls(text):
                if _key(name, arguments) not in attempted and (name, arguments) not in self._queue:
                    self._queue.append((name, arguments))
            if not self._queue:
                return json.dumps({"action": "finish", "reply": ""})
        name, arguments = self._queue.pop(0)
        return json.dumps({"action": "call", "tool": name, "arguments": arguments}, ensure_ascii=False)
