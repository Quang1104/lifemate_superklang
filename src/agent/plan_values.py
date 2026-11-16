"""Deterministic conversions for explicitly supported Vietnamese spans."""

from datetime import date, datetime, timedelta
from decimal import Decimal
import re
from unicodedata import normalize
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class Unresolved(ValueError):
    """A typed value cannot be resolved with the available evidence."""


_DIGITS = dict(zip("không một hai ba bốn năm sáu bảy tám chín".split(), range(10)))
_DIGITS.update({"tư": 4, "lăm": 5, "mốt": 1})
_MONEY = {"nghìn": 1_000, "ngàn": 1_000, "triệu": 1_000_000, "tỷ": 1_000_000_000,
          "tỉ": 1_000_000_000, "k": 1_000}


def normalized(text: str) -> str:
    return " ".join(normalize("NFC", text).casefold().split())


def _number(text: str) -> Decimal:
    if re.fullmatch(r"-?(?:[0-9]+|[0-9]{1,3}(?:\.[0-9]{3})+)(?:,[0-9]+)?", text):
        return Decimal(text.replace(".", "").replace(",", "."))
    words = text.split()
    if len(words) == 1 and words[0] in _DIGITS:
        return Decimal(_DIGITS[words[0]])
    if text == "chục" or len(words) == 2 and words[1] == "chục" and _DIGITS.get(words[0], 0) >= 1:
        return Decimal(10 * _DIGITS.get(words[0], 1) if len(words) == 2 else 10)  # "hai chục" = 20
    # Colloquial tens without "mươi" ("bốn lăm" = 45); only the unambiguous unit forms.
    if len(words) == 2 and _DIGITS.get(words[0], 0) >= 2 and words[1] in {"lăm", "mốt", "tư"}:
        return Decimal(10 * _DIGITS[words[0]] + _DIGITS[words[1]])
    match = re.fullmatch(r"(?:(\w+) )?(mười|mươi)(?: (\w+))?", text)
    if match:
        tens, form, units = match.groups()
        if form == "mười" and tens is None and (units is None or units in _DIGITS):
            return Decimal(10 + _DIGITS.get(units, 0))
        if form == "mươi" and tens in _DIGITS and (units is None or units in _DIGITS):
            return Decimal(10 * _DIGITS[tens] + _DIGITS.get(units, 0))
    match = re.fullmatch(r"(\w+) trăm(?: (?:lẻ |linh )?(.+))?", text)
    if match and match[1] in _DIGITS:
        return Decimal(100 * _DIGITS[match[1]]) + (_number(match[2]) if match[2] else 0)
    raise Unresolved("Expected an explicit Vietnamese number")


_WEEKDAYS = {"thứ hai": 0, "thứ ba": 1, "thứ tư": 2, "thứ năm": 3, "thứ sáu": 4, "thứ bảy": 5,
             "chủ nhật": 6}
_W = r"(?!tháng)\w+"
_NUM = rf"\d{{1,2}}|mười(?: {_W})?|{_W} mươi(?: {_W})?|{_W} (?:mốt|lăm|tư)|{_W}"


def _strip_prefix(text: str) -> str:
    return re.sub(r"^(?:(?:vào|lúc|đúng|khoảng) )+", "", text).strip()


def _weekday(match: re.Match, today: date) -> date:
    target, when = _WEEKDAYS[match[1]], match[2]
    monday = today - timedelta(days=today.weekday())
    if when in {"này", "tuần này"}:
        result = monday + timedelta(days=target)
        if result < today:
            raise Unresolved("Weekday of this week has already passed")
        return result
    if when in {"tuần sau", "tuần tới"}:
        return monday + timedelta(days=7 + target)
    if when == "tuần sau nữa":
        return monday + timedelta(days=14 + target)
    return today + timedelta(days=(target - today.weekday() - 1) % 7 + 1)  # next occurrence


def _month_day(match: re.Match, today: date) -> date:
    day = _number(match[1])
    month = _number(match[2]) if match[2] else Decimal(today.month)
    try:
        result = date(today.year, int(month), int(day))
        if result < today:  # A past day means the next month (or next year when the month was named).
            result = (date(today.year + 1, int(month), int(day)) if match[2] else
                      date(today.year + today.month // 12, today.month % 12 + 1, int(day)))
    except ValueError as exc:
        raise Unresolved("Calendar day is out of range") from exc
    return result


_DAYS = (
    (r"ngày kia|ngày mốt", lambda m, today: today + timedelta(days=2)),
    (r"ngày mai|mai", lambda m, today: today + timedelta(days=1)),
    (r"hôm qua", lambda m, today: today - timedelta(days=1)),
    (r"hôm nay|bữa nay|hôm ni|bữa ni|nay|ni", lambda m, today: today),
    (r"(thứ (?:hai|ba|tư|năm|sáu|bảy)|chủ nhật)(?: (này|tới|tuần sau nữa|tuần sau|tuần tới|tuần này))?",
     _weekday),
    (rf"(?:ngày|mùng|mồng|hôm) ({_NUM})(?: tháng ({_NUM}))?", _month_day),
    (rf"({_NUM}) tháng ({_NUM})", _month_day),
)


def _day(text: str, today: date) -> tuple[date | None, str]:
    """Return the calendar day named in the span, if any, and the span without it."""
    for pattern, resolve in _DAYS:
        match = re.search(rf"(?<!\w)(?:{pattern})(?!\w)", text)
        if match:
            return resolve(match, today), " ".join((text[:match.start()] + text[match.end():]).split())
    return None, text


def _clock(text: str) -> tuple[int, int, int]:
    """Parse an explicit clock reading into (hour, minute, minutes before the hour)."""
    before = 0
    if match := re.fullmatch(r"(\d{1,2})(?::|h)(\d{2})?", text):
        hours, minutes = _number(match[1]), _number(match[2] or "0")
    elif match := re.fullmatch(r"(.+?) (?:giờ|h)(?: (?:kém (.+?)|(.+?))(?: phút)?)?", text):
        hours = _number(match[1])
        minutes = Decimal(30) if match[3] == "rưỡi" else _number(match[3]) if match[3] else Decimal(0)
        before = _number(match[2]) if match[2] else 0
    elif match := re.fullmatch(r"(.+?) rưỡi", text):
        hours, minutes = _number(match[1]), Decimal(30)
    else:
        raise Unresolved("Expected an explicit clock time")
    if (any(value != int(value) for value in (hours, minutes, before)) or not 0 <= hours < 24
            or not 0 <= minutes < 60 or not 0 <= before < 60 or before and minutes):
        raise Unresolved("Clock time is out of range")
    return int(hours), int(minutes), int(before)


def _time(text: str, context: dict, mode: str) -> str:
    current = context.get("current_time")
    timezone = context.get("timezone")
    if not isinstance(current, str) or not isinstance(timezone, str):
        raise Unresolved("Sample time and timezone are required")
    try:
        now = datetime.fromisoformat(current)
    except ValueError as exc:
        raise Unresolved("Sample time is invalid") from exc
    try:
        zone = ZoneInfo(timezone)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        # shortcut: without a tz database use the sample clock's fixed offset; exact for zones without DST.
        if now.tzinfo is None:
            raise Unresolved("Sample timezone is unavailable") from exc
        zone = now.tzinfo
    now = now.astimezone(zone) if now.tzinfo is not None else now.replace(tzinfo=zone)

    text = _strip_prefix(text)
    relative = re.fullmatch(r"(.+) (phút|giờ|tiếng) nữa", text)
    if text == "nửa tiếng nữa":
        target = now + timedelta(minutes=30)
    elif relative:
        amount = _number(relative[1])
        if not 0 <= amount <= 10_000:
            raise Unresolved("Relative time is out of supported range")
        target = now + timedelta(minutes=float(amount) * (1 if relative[2] == "phút" else 60))
    else:
        base, text = _day(text, now.date())
        period = re.search(r"(?<!\w)(sáng|trưa|chiều|tối|đêm)(?!\w)", text)
        if period:
            text = " ".join((text[:period.start()] + text[period.end():]).split())
        text = _strip_prefix(text)
        if mode == "date" and not text and base is not None:
            return base.isoformat()
        hours, minutes, before = _clock(text)
        name = period[1] if period else None
        if name in {"chiều", "tối"} and 1 <= hours < 12 or name == "trưa" and 1 <= hours <= 4 \
                or name == "đêm" and 7 <= hours < 12:
            hours += 12
        elif name in {"sáng", "đêm"} and hours == 12:
            raise Unresolved("Clock time is ambiguous")
        # Without a named day (or AM/PM), take the nearest reading that is at most an hour in the past.
        days = [base] if base is not None else [now.date() + timedelta(days=d) for d in (-1, 0, 1)]
        hour_options = [hours, hours + 12] if base is None and name is None and 1 <= hours < 12 else [hours]
        target = min((datetime.combine(day, datetime.min.time(), tzinfo=zone).replace(hour=hour, minute=minutes)
                      - timedelta(minutes=before) for day in days for hour in hour_options),
                     key=lambda moment: (moment < now - timedelta(hours=1), abs(moment - now)))
    if mode == "date":
        return target.date().isoformat()
    if mode == "time":
        return target.strftime("%H:%M")
    return target.isoformat(timespec="seconds")


_MAX_SHIFT_MINUTES = 7 * 24 * 60


def _duration(text: str) -> int:
    """Minutes in an explicit span such as "1 tiếng", "nửa tiếng", "1 tiếng rưỡi" or "2 tiếng 15 phút"."""
    text = re.sub(r"^(?:(?:thêm|chừng|khoảng|tầm|cỡ|độ) )+", "", text)
    if text == "nửa tiếng":
        return 30
    if text == "tiếng rưỡi":
        return 90
    if match := re.fullmatch(r"(.+?) (?:tiếng|giờ|h)(?: (?:(rưỡi)|(.+?) phút))?", text):
        amount = _number(match[1]) * 60 + (30 if match[2] else _number(match[3]) if match[3] else 0)
    elif match := re.fullmatch(r"(.+?) phút", text):
        amount = _number(match[1])
    else:
        raise Unresolved("Expected an explicit duration")
    if amount != int(amount) or not 0 < amount <= _MAX_SHIFT_MINUTES:
        raise Unresolved("Duration is out of supported range")
    return int(amount)


def shift_time(value, minutes) -> str:
    """Move an observed ISO datetime or HH:MM time, keeping its form and offset."""
    if type(minutes) not in (int, float) or not 0 < abs(minutes) <= _MAX_SHIFT_MINUTES:
        raise Unresolved("Shift needs an observed duration in minutes")
    if not isinstance(value, str):
        raise Unresolved("Shift needs an observed time")
    delta = timedelta(minutes=minutes)
    if re.fullmatch(r"\d{2}:\d{2}", value):
        return (datetime.strptime(value, "%H:%M") + delta).strftime("%H:%M")
    if not re.fullmatch(r"\d{4}-\d\d-\d\d[T ]\d\d:\d\d.*", value):
        raise Unresolved("Shift needs a time of day")
    try:
        moment = datetime.fromisoformat(value)
    except ValueError as exc:
        raise Unresolved("Shift needs a valid time") from exc
    return (moment + delta).isoformat(timespec="seconds")


def convert_span(text: str, mode: str, context: dict):
    if mode == "text":
        return text
    value = normalized(text)
    if mode in {"datetime", "date", "time"}:
        return _time(value, context, mode)
    if mode == "duration":
        return _duration(value)
    if mode == "money":
        value = re.sub(r"\s*(?:đồng|vnđ|vnd|₫)$", "", value).strip()
        match = re.fullmatch(r"(.+?)\s*(nghìn|ngàn|triệu|tỷ|tỉ|k)", value)
        amount = _number(match[1].strip()) * _MONEY[match[2]] if match else _number(value)
    elif mode in {"number", "integer"}:
        amount = _number(value)
    else:
        raise Unresolved("Unknown span conversion")
    if amount == amount.to_integral_value():
        return int(amount)
    if mode in {"money", "integer"}:
        raise Unresolved("Expected a whole number")
    return float(amount)
