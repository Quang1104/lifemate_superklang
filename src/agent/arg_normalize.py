"""Conservative, schema-driven representation fixes for planner arguments.

The pure helper is deliberately independent of environment flags. The runner
opts in with LIFEMATE_ARG_NORMALIZE=1 before its existing authorization guard.
No values are filled in, and identifiers, names and fuzzy text stay verbatim.
"""

from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import re
from typing import Any
import unicodedata


_DATE = re.compile(r"([0-9]{4})([-/])([0-9]{1,2})\2([0-9]{1,2})")
_DATETIME = re.compile(
    r"([0-9]{4})-([0-9]{1,2})-([0-9]{1,2})[Tt ]"
    r"([0-9]{1,2}):([0-9]{2})(?::([0-9]{2})(\.[0-9]{1,6})?)?"
    r"([Zz]|[+-][0-9]{2}:?[0-9]{2})?"
)
_INTEGER = re.compile(r"[+-]?[0-9]+")
_SCALED_INTEGER = re.compile(
    r"([+-]?[0-9]+(?:[.,][0-9]{1,2})?)\s*(k|nghin|ngan|trieu|ty)"
    r"(?:\s*(?:dong|vnd|d))?"
)
_CURRENCY_INTEGER = re.compile(r"([+-]?[0-9]+)\s*(?:dong|vnd|d)")
_SCALE = {"k": 1_000, "nghin": 1_000, "ngan": 1_000,
          "trieu": 1_000_000, "ty": 1_000_000_000}
_COMPLEX = {"$ref", "$dynamicRef", "allOf", "anyOf", "oneOf", "if", "then", "else"}


def _fold(value: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFD", value.casefold())
                   if not unicodedata.combining(char)).replace("đ", "d")


def _protected(field: str, schema: dict[str, Any]) -> bool:
    # Tokenize camelCase as well as snake_case. Container annotations protect
    # their children too (for example an array described as a list of IDs).
    tokens = re.sub(r"([a-z])([A-Z])", r"\1_\2", field).lower().split("_")
    return (schema.get("x-match") == "fuzzy" or "x-ref" in schema
            or any(token in {"id", "ids", "name", "names"} for token in tokens)
            or bool(re.search(r"\b(?:IDs?|names?|tên)\b", str(schema.get("description", "")), re.IGNORECASE)))


def _integer(value: str) -> int | None:
    if _INTEGER.fullmatch(value):
        try:
            return int(value)
        except ValueError:
            return None
    folded = _fold(value)
    currency = _CURRENCY_INTEGER.fullmatch(folded)
    if currency:
        try:
            return int(currency[1])
        except ValueError:
            return None
    scaled = _SCALED_INTEGER.fullmatch(folded)
    if not scaled:
        return None
    try:
        number = Decimal(scaled[1].replace(",", ".")) * _SCALE[scaled[2]]
        if number == number.to_integral_value():
            return int(number)
    except (InvalidOperation, ValueError, OverflowError):
        pass
    return None


def _context_timezone(context: dict[str, Any] | None):
    current_time = (context or {}).get("current_time")
    if not isinstance(current_time, str):
        return None
    try:
        current = datetime.fromisoformat(current_time)
        return current.tzinfo if current.utcoffset() is not None else None
    except (ValueError, OverflowError):
        return None


def _date(value: str) -> str:
    match = _DATE.fullmatch(value)
    if match:
        try:
            return date(int(match[1]), int(match[3]), int(match[4])).isoformat()
        except ValueError:
            pass
    return value


def _datetime(value: str, context: dict[str, Any] | None) -> str:
    match = _DATETIME.fullmatch(value)
    if not match:
        return value
    year, month, day, hour, minute, second, fraction, offset = match.groups()
    offset = offset or ""
    if offset.upper() == "Z":
        offset = "+00:00"
    elif offset and ":" not in offset:
        offset = offset[:3] + ":" + offset[3:]
    # Do not allow datetime.fromisoformat to silently accept offsets such as
    # +07:99, whose normalization would conceal an invalid supplied value.
    if offset and (int(offset[1:3]) > 23 or int(offset[4:]) > 59):
        return value
    iso = (f"{year}-{int(month):02d}-{int(day):02d}T{int(hour):02d}:"
           f"{minute}:{second or '00'}{fraction or ''}{offset}")
    try:
        parsed = datetime.fromisoformat(iso)
        timezone = _context_timezone(context)
        if parsed.utcoffset() is None:
            if timezone is None:
                return value  # Never consult the host clock or infer an offset.
            parsed = parsed.replace(tzinfo=timezone)
        elif timezone is not None:
            parsed = parsed.astimezone(timezone)  # Preserve the instant.
        return parsed.isoformat()
    except (ValueError, OverflowError):
        return value


def _string_format(schema: dict[str, Any]) -> str | None:
    declared = schema.get("format")
    if declared is not None:
        return declared
    # Several official schemas state the format only in their description.
    description = str(schema.get("description", ""))
    if re.search(r"\bISO[- ]8601\b", description, re.IGNORECASE):
        return "date-time"
    if "YYYY-MM-DD" in description:
        return "date"
    return None


def _normalize(value: Any, schema: Any, field: str,
               context: dict[str, Any] | None) -> Any:
    if (not isinstance(schema, dict) or _protected(field, schema)
            or _COMPLEX.intersection(schema)):
        return deepcopy(value)
    declared = schema.get("type")
    if declared == "object" and isinstance(value, dict):
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            return deepcopy(value)
        return {key: _normalize(item, properties[key], key, context)
                if key in properties else deepcopy(item) for key, item in value.items()}
    if declared == "array" and isinstance(value, list):
        items = schema.get("items")
        return [_normalize(item, items, field, context) for item in value]
    if not isinstance(value, str):
        return deepcopy(value)
    # A type must explicitly permit the target representation. Unknown and
    # union schemas are left to the guard, including whitespace.
    if declared != "string" and declared != "integer":
        return value
    stripped = value.strip()
    if declared == "integer":
        converted = _integer(stripped)
        return converted if converted is not None else value
    allowed = schema.get("enum")
    if isinstance(allowed, list):
        if stripped in allowed:
            return stripped
        matches = [item for item in allowed if isinstance(item, str)
                   and _fold(item.strip()) == _fold(stripped)]
        if len(matches) == 1:
            return matches[0]
        return stripped
    string_format = _string_format(schema)
    if string_format == "date":
        return _date(stripped)
    if string_format == "date-time":
        return _datetime(stripped, context)
    return stripped


def normalize_arguments(arguments: dict[str, Any], parameters: dict[str, Any],
                        context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a fresh argument object; never add keys, infer values or mutate inputs.

    Only direct object/array schemas are traversed. Ambiguous enum matches,
    date arithmetic, non-ISO day-first dates, unit conversions, unknown keys,
    schema references/compositions and protected text are left unchanged.
    Currency scale suffixes are parsed exactly; fractional results never round.
    Missing datetime offsets may use only context.current_time's explicit offset.
    """
    return _normalize(arguments, parameters, "", context)
