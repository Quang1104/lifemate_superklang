"""Planner that speaks the checkpoint's own tool-calling chat format.

Qwen is trained to read tools through its chat template and to answer with
<tool_call>{"name": ..., "arguments": ...}</tool_call>. Using that format, with tool
results fed back as tool messages, replaces the project-specific JSON protocol for
the model while the runner keeps every existing guard.
"""

from collections.abc import Callable
from datetime import datetime, timedelta
import json
import os

from .fewshot import RESULTS_BUDGET, native_examples, native_k
from .planner import PlannerInput
from .time_hints_v8 import render_time_hints

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

# Keep the baseline string untouched: the candidate rewrites only four numbered rules.
_PROMPT_V2_RULES = {
    "1.": "1. Chỉ thực hiện các hành động được yêu cầu trong câu hiện tại; hoàn tất TẤT CẢ các hành động đó trước khi trả lời.",
    "7.": "7. Tra cứu trước nếu có tool phù hợp; nếu còn mơ hồ giữa nhiều kết quả hoặc thiếu giá trị bắt buộc, gọi sys.ask_missing_info với tool_name và params còn thiếu.",
    "8.": "8. Câu chỉ trò chuyện hoặc ngoài khả năng tool: trả lời ngắn, KHÔNG gọi tool, kể cả tool tra cứu hay sys.ask_missing_info.",
    "9.": "9. Trước khi kết thúc, kiểm từng việc trong câu hiện tại đã làm xong hoặc đã hỏi thiếu; không dừng ngay sau bước tra cứu.",
}
_SYSTEM_V2 = "\n".join(_PROMPT_V2_RULES.get(line.partition(" ")[0], line)
                       for line in _SYSTEM.split("\n"))

_RESULT_CHARS = 6_000

_WEEKDAYS = ("thứ Hai", "thứ Ba", "thứ Tư", "thứ Năm", "thứ Sáu", "thứ Bảy", "Chủ nhật")


def render_date_table(current_time: str | None) -> str:
    """Render local calendar anchors, independent of the host clock and locale.

    A week starts on Monday. Missing, invalid or offset-free timestamps yield no
    table rather than guessing a timezone. The input's offset defines its date.
    """
    if not isinstance(current_time, str):
        return ""
    try:
        anchor = datetime.fromisoformat(current_time)
        if anchor.utcoffset() is None:
            return ""
        today = anchor.date()
        monday = today - timedelta(days=today.weekday())
        lines = ["Bảng ngày (tuần từ thứ Hai đến Chủ nhật):"]
        for label, offset in (("hôm nay", 0), ("mai", 1), ("ngày kia", 2), ("hôm qua", -1)):
            day = today + timedelta(days=offset)
            lines.append(f"{label}: {_WEEKDAYS[day.weekday()]} {day.isoformat()}")
        for week, suffix in ((0, "này (tuần này)"), (1, "tuần sau")):
            for weekday, name in enumerate(_WEEKDAYS):
                day = monday + timedelta(days=7 * week + weekday)
                lines.append(f"{name} {suffix}: {day.isoformat()}")
        return "\n".join(lines)
    except (ValueError, OverflowError):
        return ""

# The guard's terse rule names gave the model nothing to act on; smoke40 runs repeated the rejected call and stopped.
_REJECTION_HINTS = {
    "(x-ref)": "ID này chưa có trong kết quả tool hay lịch sử. Gọi tool tra cứu (list/search/get) để lấy ID thật.",
    "(additionalProperties)": "Có tham số không có trong schema của tool. Chỉ dùng đúng tên tham số đã khai báo.",
    "(required)": "Thiếu tham số bắt buộc. Tra cứu để có giá trị, hoặc gọi sys.ask_missing_info.",
    "(enum)": "Giá trị không thuộc danh sách cho phép của tham số.",
}


# Vietnamese for the enum values of the TRAIN and public catalogs (LIFEMATE_CONTEXT_V4). Tool results and
# arguments use the English value while users say the Vietnamese word; unknown values get no gloss.
_ENUM_VI = {
    "once": "một lần", "daily": "hằng ngày", "weekdays": "ngày thường", "weekends": "cuối tuần", "weekly": "hằng tuần",
    "washer": "máy giặt", "dishwasher": "máy rửa bát", "dryer": "máy sấy", "rice_cooker": "nồi cơm điện",
    "microwave": "lò vi sóng", "air_purifier": "máy lọc không khí",
    "electricity": "điện", "water": "nước", "internet": "mạng", "tv": "tivi, truyền hình", "tuition": "học phí",
    "phone": "điện thoại", "bank": "ngân hàng", "ewallet": "ví điện tử", "cash": "tiền mặt", "cod": "trả khi nhận hàng",
    "wrong_item": "sai hàng", "damaged": "hư hỏng", "changed_mind": "đổi ý", "other": "khác",
    "archive": "lưu trữ", "delete": "xoá", "walking": "đi bộ", "running": "chạy bộ", "cycling": "đạp xe",
    "swimming": "bơi", "football": "bóng đá", "badminton": "cầu lông", "economy": "phổ thông", "business": "thương gia",
    "diesel": "dầu diesel", "sjc": "vàng miếng SJC", "ring": "vàng nhẫn", "world": "vàng thế giới",
    "on": "bật", "off": "tắt", "today": "hôm nay", "week": "tuần", "month": "tháng", "away": "vắng nhà", "home": "ở nhà",
    "car": "ô tô", "motorbike": "xe máy", "walk": "đi bộ", "transit": "phương tiện công cộng",
    "pause": "tạm dừng", "resume": "phát tiếp", "next": "bài kế", "previous": "bài trước", "volume": "âm lượng",
    "mute": "tắt tiếng", "general": "tổng hợp", "sports": "thể thao", "tech": "công nghệ", "entertainment": "giải trí",
    "bike": "xe máy", "car4": "ô tô 4 chỗ", "car7": "ô tô 7 chỗ", "vacuum": "hút bụi", "mop": "lau nhà", "both": "cả hai",
    "cool": "làm mát", "heat": "sưởi", "dry": "hút ẩm", "fan": "quạt", "light": "đèn", "climate": "điều hoà",
    "curtain": "rèm", "plug": "ổ cắm", "buy": "mua", "sell": "bán", "above": "vượt trên", "below": "xuống dưới",
    "low": "thấp", "normal": "thường", "high": "cao", "open": "chưa xong", "done": "đã xong",
    "train": "tàu hoả", "bus": "xe khách",
}
_ENUM_VI_BY_PARAMETER = {("topic", "business"): "kinh doanh"}


def _context_v4() -> bool:
    return os.environ.get("LIFEMATE_CONTEXT_V4") == "1"


def _glossed(schema, parameter: str = ""):
    """A copy of a parameter schema whose enum descriptions also say what the values mean in Vietnamese."""
    if isinstance(schema, list):
        return [_glossed(item, parameter) for item in schema]
    if not isinstance(schema, dict):
        return schema
    copy = {key: ({name: _glossed(item, name) for name, item in value.items()} if key == "properties"
                  and isinstance(value, dict) else _glossed(value, parameter)) for key, value in schema.items()}
    values = schema.get("enum")
    if isinstance(values, list):
        glosses = [f"{value} = {_ENUM_VI_BY_PARAMETER.get((parameter, value)) or _ENUM_VI[value]}"
                   for value in values if isinstance(value, str) and value in _ENUM_VI]
        if glosses:
            copy["description"] = " ".join(filter(None, [schema.get("description"), "Giá trị: " + "; ".join(glosses)]))
    return copy


def tool_definitions(tools: list[dict]) -> list[dict]:
    gloss = _glossed if _context_v4() else (lambda parameters: parameters)
    return [{"type": "function", "function": {"name": tool["name"], "description": tool["description"],
                                              "parameters": gloss(tool["parameters"])}} for tool in tools]


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


def render_examples(examples: list[dict]) -> str:
    """Retrieved TRAIN requests with the calls that solved them, as plain prompt text.

    Kept out of the chat turns so rule 3 (do not redo history) does not apply to them.
    """
    lines = ["Ví dụ yêu cầu tương tự và các lệnh đã dùng (chỉ tham khảo cách chọn tool và thứ tự; "
             "giá trị và ID phải lấy từ yêu cầu hiện tại và kết quả tool):"]
    for item in examples:
        calls = [call["name"] + _dump(call["arguments"]) for call in item["calls"]]
        if "results" not in item or not calls:
            lines.append(f"- {_dump(item['request'])} → {'; '.join(calls) or 'không gọi tool nào'}")
            continue
        # LIFEMATE_CONTEXT_V4: one call per line, each lookup with what it returned, so the reason for a
        # branch ("ví không đủ thì trừ ngân hàng") is visible. Write calls show no result.
        lines.append(f"- {_dump(item['request'])}")
        lines += [f"  {call} ⇒ {_safe(result)}" if result else f"  {call}" for call, result in zip(calls, item["results"])]
    return "\n".join(lines)


def build_messages(request: PlannerInput) -> list[dict]:
    sample = request.sample
    context = sample.context or {}
    prompt = _SYSTEM_V2 if os.environ.get("LIFEMATE_PROMPT_V2") == "1" else _SYSTEM
    system = (f"{prompt}\n\nThời gian hiện tại: {context.get('current_time')} ({context.get('timezone')})\n"
              f"Vị trí: {_dump(context.get('location'))}\nHồ sơ người dùng: {_dump(context.get('user_profile'))}")
    if os.environ.get("LIFEMATE_DATE_TABLE") == "1":
        date_table = render_date_table(context.get("current_time"))
        if date_table:
            system += "\n\n" + date_table
    if os.environ.get("LIFEMATE_TIME_HINTS_V8") == "1":
        # Dates and times the request states, already computed from the sample clock (same text in training rows).
        time_hints = render_time_hints(sample.user_request, context)
        if time_hints:
            system += "\n\n" + time_hints
    apps = context.get("apps") if _context_v4() else None
    if isinstance(apps, dict) and apps.get("missing"):
        system += "\n\n" + "\n".join((
            "Ứng dụng đã cài: " + ", ".join(apps.get("installed", ())),
            "Ứng dụng KHÔNG cài trên máy này: " + ", ".join(apps["missing"]),
            "Việc nào cần một ứng dụng không cài thì bỏ qua việc đó: không thay bằng ứng dụng khác, không hỏi lại."))
    k = native_k()
    if k > 0:
        # Same selection and text when scripts/build_sft.py --fewshot-k builds the training prompts.
        names = {tool["name"] for tool in request.tools}
        examples = (native_examples().select(sample, names, k, RESULTS_BUDGET, results=True) if _context_v4()
                    else native_examples().select(sample, names, k))
        if examples:
            system += "\n\n" + render_examples(examples)
    messages =[{"role": "system", "content": system}]
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


def _abstain_threshold() -> float | None:
    """None unless LIFEMATE_ABSTAIN_PROBE=1. The default threshold (inf) only records the probability."""
    if os.environ.get("LIFEMATE_ABSTAIN_PROBE") != "1":
        return None
    return float(os.environ.get("LIFEMATE_ABSTAIN_THRESHOLD", "inf"))


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
            messages, tools = build_messages(request), tool_definitions(request.tools)
            threshold = None if observations else _abstain_threshold()  # only the first decision of a sample
            probe = None if threshold is None else {}
            text = self._chat(messages, tools, max_tokens=self._max_tokens, deadline=request.deadline,
                              **({} if probe is None else {"probe": probe}))
            if probe and not parse_calls(text) and probe.get("call_prob", 0.0) >= threshold:
                # The model stopped without acting although a tool call was likely enough: make it commit to one.
                try:
                    text = self._chat(messages, tools, max_tokens=self._max_tokens, deadline=request.deadline,
                                      prefix="<tool_call>\n")
                except TimeoutError:
                    pass  # No time left for a second opinion: keep the first answer.
            attempted = {_key(item.tool, item.arguments or {}) for item in observations if item.tool}
            # A repeated call means the model has nothing new to do; stopping beats looping.
            for name, arguments in parse_calls(text):
                if _key(name, arguments) not in attempted and (name, arguments) not in self._queue:
                    self._queue.append((name, arguments))
            if not self._queue:
                return json.dumps({"action": "finish", "reply": ""})
        name, arguments = self._queue.pop(0)
        return json.dumps({"action": "call", "tool": name, "arguments": arguments}, ensure_ascii=False)
