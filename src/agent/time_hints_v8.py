"""Time expressions of the current request, resolved against the sample clock (LIFEMATE_TIME_HINTS_V8=1).

The planner often gets dates and times wrong although the request states them ("45 phút nữa", "Chủ nhật này",
"hai mươi hai tháng mười"). plan_values already converts such spans deterministically; this module finds the spans
in a transcript and renders them as prompt lines, so the model copies a computed value instead of doing arithmetic.
The same text is built for training rows and at inference (native_planner.build_messages).
"""

import re

from .plan_values import Unresolved, convert_span, normalized

_MAX_WORDS = 9
_WORD = re.compile(r"[\w:]+")
# One-word spans are kept only for these; "nay", "ni" and bare numbers are too often something else.
_SINGLE = frozenset({"mai"})


def time_hints(text: str, context: dict) -> list[tuple[str, str]]:
    """(span, ISO value) for every longest, non-overlapping span that names a date or time, in reading order."""
    words = _WORD.findall(normalized(text))
    hints, start = [], 0
    while start < len(words):
        for size in range(min(_MAX_WORDS, len(words) - start), 0, -1):
            span = " ".join(words[start:start + size])
            if size == 1 and span not in _SINGLE:
                continue
            value = None
            for mode in ("datetime", "date"):
                try:
                    value = convert_span(span, mode, context)
                    break
                except (Unresolved, ValueError, OverflowError):
                    continue
            if value is not None:
                hints.append((span, value))
                start += size
                break
        else:
            start += 1
    return hints


def render_time_hints(text: str, context: dict) -> str:
    """Prompt block for the hints, or "" when the request names no date or time."""
    hints = time_hints(text, context)
    if not hints:
        return ""
    return "\n".join(["Mốc thời gian đã tính sẵn từ câu hiện tại (dùng đúng giá trị này nếu việc cần ngày giờ đó):",
                      *(f'- "{span}" = {value}' for span, value in hints)])
