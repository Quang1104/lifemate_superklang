"""Bounded strict JSON handling shared by the planner and tool boundary."""

import json
from typing import Any


class InvalidJSON(ValueError):
    """Input is outside the accepted JSON format or size."""


def _object_with_unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidJSON("Duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise InvalidJSON(f"Non-finite JSON number: {value}")


def encode_json(value: Any, *, max_bytes: int) -> str:
    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True,
                             separators=(",", ":"))
    except (TypeError, ValueError, RecursionError) as exc:
        raise InvalidJSON("Value cannot be serialized as strict JSON") from exc
    if len(encoded) > max_bytes:
        raise InvalidJSON("JSON exceeds its byte limit")
    return encoded


def decode_json(text: str, *, max_bytes: int) -> Any:
    if not isinstance(text, str) or len(text) > max_bytes:
        raise InvalidJSON("Expected bounded JSON text")
    try:
        if len(text.encode("utf-8")) > max_bytes:
            raise InvalidJSON("JSON exceeds its byte limit")
        value = json.loads(text, object_pairs_hook=_object_with_unique_keys,
                           parse_constant=_reject_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise InvalidJSON("Malformed JSON") from exc
    # json.loads may parse an overflowing JSON exponent as infinity.
    encode_json(value, max_bytes=max_bytes)
    return value
