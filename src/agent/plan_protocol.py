"""Typed plan sources and exact matching against visible data."""

from dataclasses import dataclass
import re
from typing import Any
from unicodedata import normalize

from ..inference.types import Sample
from ..tooling.json_data import InvalidJSON, decode_json
from .plan_values import Unresolved, convert_span, normalized, shift_time


class InvalidPlan(ValueError):
    """A plan violates the local compiler contract."""


@dataclass(frozen=True)
class Source:
    kind: str
    value: Any
    path: tuple[str, ...] = ()
    where: dict[str, "Source"] | None = None
    field: tuple[str, ...] = ()
    mode: str = "text"

    def dependencies(self) -> set[str]:
        result = {self.value} if self.kind == "result" else set()
        nested = self.where.values() if self.where is not None else ()
        if self.kind == "object":
            nested = self.value.values()
        if self.kind in {"array", "shift"}:
            nested = self.value
        for item in nested:
            result.update(item.dependencies())
        return result

    def as_plan_value(self) -> dict:
        if self.kind == "value":
            return {"value": self.value}
        if self.kind == "text":
            return {"text": self.value, "as": self.mode}
        if self.kind == "context":
            return {"context": self.path}
        if self.kind == "history":
            return {"history": self.value, "path": self.path}
        if self.kind == "history_text":
            return {"history_text": self.value[0], "text": self.value[1], "as": self.mode}
        if self.kind == "object":
            return {"object": {key: value.as_plan_value() for key, value in self.value.items()}}
        if self.kind == "array":
            return {"array": [value.as_plan_value() for value in self.value]}
        if self.kind == "shift":
            return {"shift": self.value[0].as_plan_value(), "by": self.value[1].as_plan_value(), "dir": self.mode}
        result = {"result": self.value, "path": self.path, "field": self.field}
        if self.where is not None:
            result["where"] = {key: value.as_plan_value() for key, value in self.where.items()}
        return result


def _path(value: Any, *, empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not empty) or len(value) > 8 or not all(
        isinstance(key, str) and key for key in value
    ):
        raise InvalidPlan("Expected a bounded property path")
    return tuple(value)


def parse_source(raw: Any, *, depth: int = 0) -> Source:
    if not isinstance(raw, dict) or depth >= 6:
        raise InvalidPlan("Expected a bounded typed argument source")
    keys = set(raw)
    if keys == {"value"}:
        if isinstance(raw["value"], (dict, list)):
            raise InvalidPlan("Use typed object or array sources")
        return Source("value", raw["value"])
    if keys in ({"text"}, {"text", "as"}) and isinstance(raw["text"], str) and raw["text"].strip():
        mode = raw.get("as", "text")
        if mode not in ("text", "integer", "number", "money", "date", "time", "datetime", "duration"):
            raise InvalidPlan("Unknown span conversion")
        return Source("text", raw["text"], mode=mode)
    if keys in ({"history_text", "text"}, {"history_text", "text", "as"}):
        index = raw["history_text"]
        quote = raw["text"]
        mode = raw.get("as", "text")
        if (type(index) is not int or index < 1 or not isinstance(quote, str) or not quote.strip()
                or mode not in ("text", "integer", "number", "money", "duration")):
            raise InvalidPlan("Expected a typed span from an earlier user turn")
        return Source("history_text", (index, quote), mode=mode)
    if keys == {"context"}:
        return Source("context", None, _path(raw["context"]))
    if keys == {"history", "path"} and type(raw["history"]) is int and raw["history"] >= 1:
        return Source("history", raw["history"], _path(raw["path"]))
    if keys == {"object"} and isinstance(raw["object"], dict):
        return Source("object", {key: parse_source(item, depth=depth + 1)
                                 for key, item in raw["object"].items()})
    if keys == {"shift", "by", "dir"} and raw["dir"] in ("later", "earlier"):
        return Source("shift", (parse_source(raw["shift"], depth=depth + 1),
                                parse_source(raw["by"], depth=depth + 1)), mode=raw["dir"])
    if keys == {"array"} and isinstance(raw["array"], list):
        return Source("array", [parse_source(item, depth=depth + 1) for item in raw["array"]])
    if "result" in raw and keys <= {"result", "path", "where", "field"} and isinstance(raw["result"], str):
        where = raw.get("where")
        if where is not None and not isinstance(where, dict):
            raise InvalidPlan("Expected exact selector fields")
        selectors = ({key: parse_source(item, depth=depth + 1) for key, item in where.items()}
                     if where is not None else None)
        return Source("result", raw["result"], _path(raw.get("path", []), empty=True), selectors,
                      _path(raw.get("field", []), empty=True))
    if keys == {"lookup"} and isinstance(raw["lookup"], dict) and raw["lookup"]:
        return Source("lookup", None, where={key: parse_source(item, depth=depth + 1)
                                             for key, item in raw["lookup"].items()})
    raise InvalidPlan("Unrecognized typed argument source")


def value_at(root: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(root, dict) or key not in root:
            raise Unresolved("A referenced field is unavailable")
        root = root[key]
    return root


def matches(value: Any, expected: Any) -> bool:
    if isinstance(value, str) and isinstance(expected, str):
        return normalized(value) == normalized(expected)
    return type(value) is type(expected) and value == expected


def exact_span(text: str, quote: str) -> str:
    words = normalize("NFC", quote).split()
    if not words:
        raise Unresolved("Expected a nonempty transcript span")
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(word) for word in words) + r"(?!\w)"
    found = re.search(pattern, normalize("NFC", text), flags=re.IGNORECASE)
    if found is None:
        raise Unresolved("Expected a verbatim transcript span")
    return found[0]


def select_record(records: Any, selectors: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(records, list):
        raise Unresolved("The selected tool field is not a collection")
    candidates = [
        record for record in records if isinstance(record, dict)
        and all(key in record and matches(record[key], expected) for key, expected in selectors.items())
    ]
    if len(candidates) != 1:
        raise Unresolved("Exact selector did not identify one result")
    return candidates[0]


def evaluate(source: Source, sample: Sample, results: dict[str, dict[str, Any]]) -> Any:
    if source.kind == "value":
        return source.value
    if source.kind == "text":
        return convert_span(exact_span(sample.user_request, source.value), source.mode, sample.context)
    if source.kind == "history_text":
        index, quote = source.value
        if index > len(sample.history):
            raise Unresolved("The referenced historical turn is unavailable")
        try:
            turn = decode_json(sample.history[index - 1], max_bytes=64_000)
        except InvalidJSON as exc:
            raise Unresolved("The referenced historical turn is unavailable") from exc
        if not isinstance(turn, dict) or turn.get("role") != "user" or not isinstance(turn.get("text"), str):
            raise Unresolved("Historical text must be an observed user utterance")
        return convert_span(exact_span(turn["text"], quote), source.mode, sample.context)
    if source.kind == "context":
        return value_at(sample.context, source.path)
    if source.kind == "history":
        if source.value > len(sample.prior_tool_results):
            raise Unresolved("Historical tool result is unavailable")
        prior = sample.prior_tool_results[source.value - 1].payload
        if prior.get("ok") is not True or prior.get("done") is False:
            raise Unresolved("Historical tool result did not succeed")
        return value_at(prior, source.path)
    if source.kind == "object":
        return {key: evaluate(item, sample, results) for key, item in source.value.items()}
    if source.kind == "array":
        return [evaluate(item, sample, results) for item in source.value]
    if source.kind == "shift":
        minutes = evaluate(source.value[1], sample, results)
        if type(minutes) in (int, float) and source.mode == "earlier":
            minutes = -minutes
        return shift_time(evaluate(source.value[0], sample, results), minutes)
    if source.kind == "result":
        if source.value not in results:
            raise Unresolved("A prerequisite result is unavailable")
        value = value_at(results[source.value], source.path)
        if source.where is not None or isinstance(value, list):
            filters = {key: evaluate(item, sample, results) for key, item in (source.where or {}).items()}
            value = select_record(value, filters)
        return value_at(value, source.field)
    raise Unresolved("No unique lookup was compiled for this argument")

