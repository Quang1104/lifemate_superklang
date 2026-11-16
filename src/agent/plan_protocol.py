"""Typed plan sources and exact matching against visible data."""

from dataclasses import dataclass
from datetime import datetime
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
    raw: Any = None  # the model's literal for "ref", "seen" and raw-routed "lookup" sources
    hints: tuple[str, ...] = ()  # grounded strings that may single out one record for an unobserved raw ID

    def dependencies(self) -> set[str]:
        result = ({self.value} if self.kind in {"result", "ref"} else
                  set(self.value) if self.kind == "seen" else set())
        nested = self.where.values() if self.where is not None else ()
        if self.kind == "object":
            nested = self.value.values()
        if self.kind in {"array", "shift"}:
            nested = self.value
        for item in nested:
            result.update(item.dependencies())
        return result

    def as_plan_value(self) -> Any:
        if self.kind in {"raw", "grounded"}:
            return self.value
        if self.kind in {"ref", "seen"}:
            return self.raw
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
    if depth >= 6:
        raise InvalidPlan("Expected a bounded typed argument source")
    if not isinstance(raw, dict):
        return Source("raw", raw)  # a literal: the compiler grounds it against the field schema or rejects it
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
    return Source("raw", raw)  # possibly a literal object: grounded per field by the compiler, or rejected


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
    if source.kind in {"value", "grounded"}:
        return source.value
    if source.kind == "raw":
        raise Unresolved("An argument value is not grounded in the request, context or results")
    if source.kind == "seen":
        if any(same_value(leaf, source.raw) for step in source.value for leaf in leaves(results.get(step, {}))):
            return source.raw
        raise Unresolved("A copied value is not in the referenced results")
    if source.kind == "ref":
        return _observed_id(source, results)
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


def _observed_id(source: Source, results: dict[str, dict[str, Any]]) -> Any:
    """The raw ID if its producer returned it, else the one record the producer's result singles out."""
    if source.value not in results:
        raise Unresolved("A prerequisite result is unavailable")
    collection = value_at(results[source.value], source.path)
    rows = collection if source.field else [collection]
    if not isinstance(rows, list):
        raise Unresolved("The selected tool field is not a collection")
    found = []
    for row in rows:
        try:
            found.append((row, value_at(row, source.field)))
        except Unresolved:
            continue
    if any(same_value(ident, source.raw) for _, ident in found):
        return source.raw
    if len(found) > 1:  # several records: only one may have a field the step's words name (anchor, query)
        found = [(row, ident) for row, ident in found if isinstance(row, dict) and any(
            isinstance(field, str) and field.strip() and (matches(field, hint) or quoted(hint, field))
            for field in row.values() for hint in source.hints)]
    if len(found) == 1:
        return found[0][1]
    raise Unresolved(f"ID {source.raw!r} was not returned by step {source.value!r}; "
                     "copy one real ID from visible_results")


def leaves(root: Any):
    if isinstance(root, dict):
        root = list(root.values())
    if isinstance(root, list):
        for child in root:
            yield from leaves(child)
    else:
        yield root


def scalar_paths(root: Any, path: tuple[str, ...] = ()):
    """(dict-only path, scalar) pairs; value_at cannot index lists."""
    if isinstance(root, dict):
        for key, child in root.items():
            yield from scalar_paths(child, path + (key,))
    elif not isinstance(root, list) and path:
        yield path, root


_DATETIME = r"\d{4}-\d\d-\d\d[T ]\d\d:\d\d.*"


def same_value(found: Any, wanted: Any) -> bool:
    """Exact JSON equality, plus the same instant written with or without seconds or offset."""
    if isinstance(found, bool) or isinstance(wanted, bool):
        return type(found) is type(wanted) and found == wanted
    if type(found) in (int, float) and type(wanted) in (int, float):
        return found == wanted
    if not (isinstance(found, str) and isinstance(wanted, str)):
        return False
    if found == wanted:
        return True
    if not (re.fullmatch(_DATETIME, found) and re.fullmatch(_DATETIME, wanted)):
        return False
    try:
        first, second = datetime.fromisoformat(found), datetime.fromisoformat(wanted)
    except ValueError:
        return False
    if (first.tzinfo is None) != (second.tzinfo is None):  # an offset-less literal means sample-local time
        first, second = first.replace(tzinfo=None), second.replace(tzinfo=None)
    return first == second


def enum_values(schema: dict) -> list:
    values = list(schema.get("enum", []))
    if "const" in schema:
        values.append(schema["const"])
    for branch in schema.get("anyOf", []) + schema.get("oneOf", []):
        values.extend(enum_values(branch))
    return values


_MAX_SPAN_WORDS = 9
_EDGE = ".,;:!?…\"'()“”"
_FORMS = (("datetime", _DATETIME), ("date", r"\d{4}-\d\d-\d\d"), ("time", r"\d\d:\d\d"))


def spans(text: str) -> list[str]:
    """Contiguous word spans (edge punctuation stripped) that exact_span can find again."""
    words = text.split()
    found = []
    for i in range(len(words)):
        for j in range(i + 1, min(len(words), i + _MAX_SPAN_WORDS) + 1):
            span = " ".join(words[i:j]).strip(_EDGE)
            if span and span not in found:
                found.append(span)
    return found


def quoted(text: str, quote: str, *, exact: bool = False) -> bool:
    try:
        found = exact_span(text, quote)
    except Unresolved:
        return False
    return not exact or found == quote


class Grounding:
    """Static evidence for a raw literal: request spans, context, prior results and earlier user turns."""

    def __init__(self, sample: Sample) -> None:
        self.sample = sample
        self._converted: dict[tuple[str, str], list[tuple[str, Any]]] = {}
        self._turns = []
        for index, raw in enumerate(sample.history, 1):
            try:
                turn = decode_json(raw, max_bytes=64_000)
            except InvalidJSON:
                continue
            if isinstance(turn, dict) and turn.get("role") == "user" and isinstance(turn.get("text"), str):
                self._turns.append((index, turn["text"]))

    def _span(self, value: Any, modes: tuple[str, ...], text: str) -> tuple[str, str] | None:
        """The longest span of text (at most 9 words) that converts to value under one of the modes."""
        hits = []
        for mode in modes:
            if (text, mode) not in self._converted:
                converted = []
                for span in spans(text):
                    try:
                        converted.append((span, convert_span(span, mode, self.sample.context)))
                    except (Unresolved, ValueError, OverflowError):
                        continue
                self._converted[(text, mode)] = converted
            hits += [(span, mode) for span, found in self._converted[(text, mode)] if same_value(found, value)]
        return max(hits, key=lambda hit: len(hit[0])) if hits else None

    def source(self, value: Any, schema: dict, *, free_text: bool = True) -> Source | None:
        """An equivalent typed source for a scalar literal, or None when nothing observed grounds it."""
        if isinstance(value, (dict, list)) or value is None:
            return None
        xref = schema.get("x-ref")
        enums = enum_values(schema)
        if type(value) is bool or not xref and any(type(item) is type(value) and item == value for item in enums):
            return Source("value", value)
        text = self.sample.user_request
        modes: tuple[str, ...] = ()
        if isinstance(value, str):
            form = next((mode for mode, pattern in _FORMS if re.fullmatch(pattern, value)), None)
            if form is None and value.strip() and quoted(text, value, exact=True):
                return Source("text", value)
            if form is None and value.strip() and not xref and quoted(text, value):
                return Source("grounded", value)  # same words in another case: keep the model's casing
            modes = (form,) if form else ()
        elif type(value) in (int, float):
            modes = ("integer", "money", "duration", "number") if type(value) is int else ("number",)
        if hit := self._span(value, modes, text):
            return Source("text", hit[0], mode=hit[1])
        for path, found in scalar_paths(self.sample.context):
            if same_value(found, value):
                return Source("context", None, path)
        for index, prior in enumerate(self.sample.prior_tool_results, 1):
            if prior.payload.get("ok") is not True or prior.payload.get("done") is False:
                continue
            for path, found in scalar_paths(prior.payload):
                if same_value(found, value):
                    return Source("history", index, path)
            if any(same_value(found, value) for found in leaves(prior.payload)):
                return Source("grounded", value)  # inside a list of a successful earlier result
        for index, said in self._turns:
            if isinstance(value, str) and not modes and value.strip() and quoted(said, value, exact=True):
                return Source("history_text", (index, value))
            if isinstance(value, str) and not modes and value.strip() and not xref and quoted(said, value):
                return Source("grounded", value)
            if type(value) in (int, float) and (hit := self._span(value, modes, said)):
                return Source("history_text", (index, hit[0]), mode=hit[1])
        if free_text and isinstance(value, str) and value.strip() and not xref and not enums \
                and schema.get("x-match") == "fuzzy":
            return Source("grounded", value)  # free text: scored by token F1, outside the state hash
        return None

