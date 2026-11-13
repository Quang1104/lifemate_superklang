"""Compact planner-facing tool signatures without dropping schema constraints.

Only repetitive JSON Schema structure is abbreviated. Descriptions, enum values,
references and unrecognised constraints remain literal compact JSON. Required
properties have no marker, optional properties end in ? and object schemas ending
in ! forbid extra properties. Unsupported shapes stay JSON in full.
"""

from collections.abc import Iterable, Mapping
import json
import re
from typing import Any


_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")
_PRIMITIVES = frozenset({"string", "integer", "number", "boolean", "null"})
_LEGEND = (
    "Signatures: ?=optional; other args required; !=no extra properties; "
    "JSON suffixes retain constraints/metadata; ->writes=state mutation; "
    "x-one-of=at least one listed group with nonempty values; [schema]=array."
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _name(value: str) -> str:
    return value if _NAME.fullmatch(value) else _json(value)


def _suffix(metadata: Mapping[str, Any]) -> str:
    return _json(dict(metadata)) if metadata else ""


def _object_parts(schema: Mapping[str, Any]) -> tuple[str, str] | None:
    """Abbreviate only ordinary objects whose required fields can be marked."""
    properties = schema.get("properties")
    required = schema.get("required", [])
    if (schema.get("type") != "object" or not isinstance(properties, Mapping)
            or not all(isinstance(name, str) for name in properties)
            or not isinstance(required, list)
            or not all(isinstance(name, str) and name in properties for name in required)):
        return None
    required_names = set(required)
    fields = ",".join(
        f"{_name(name)}{'' if name in required_names else '?'}:{render_schema(child)}"
        for name, child in properties.items()
    )
    metadata = {key: value for key, value in schema.items()
                if key not in {"type", "properties", "required"}}
    closed = metadata.get("additionalProperties") is False
    if closed:
        del metadata["additionalProperties"]
    return fields, ("!" if closed else "") + _suffix(metadata)


def render_schema(schema: Any) -> str:
    """Render one JSON Schema, retaining unsupported constructs verbatim."""
    if not isinstance(schema, Mapping):
        return _json(schema)
    schema_type = schema.get("type")
    if isinstance(schema_type, str) and schema_type in _PRIMITIVES:
        return schema_type + _suffix({key: value for key, value in schema.items() if key != "type"})
    if schema_type == "object":
        parts = _object_parts(schema)
        if parts is not None:
            fields, suffix = parts
            return "{" + fields + "}" + suffix
    if schema_type == "array" and isinstance(schema.get("items"), (Mapping, bool)):
        # Brackets group item annotations unambiguously for nested arrays/objects.
        return "[" + render_schema(schema["items"]) + "]" + _suffix({
            key: value for key, value in schema.items() if key not in {"type", "items"}
        })
    return _json(dict(schema))


def render_tool(tool: Mapping[str, Any]) -> str:
    """Render a registry record or catalog tool, including unknown metadata."""
    name = tool.get("name")
    parameters = tool.get("parameters")
    parts = _object_parts(parameters) if isinstance(parameters, Mapping) else None
    if not isinstance(name, str) or parts is None:
        return _json(dict(tool))
    fields, suffix = parts
    metadata = {key: value for key, value in tool.items() if key not in {"name", "parameters"}}
    mutation = ""
    if type(metadata.get("writes")) is bool:
        mutation = "->writes=" + _json(metadata.pop("writes"))
    elif isinstance(metadata.get("kind"), str) and metadata["kind"] in {"read", "write", "meta"}:
        mutation = "->" + metadata.pop("kind")
    return f"{_name(name)}({fields}){suffix}{mutation}" + (
        " " + _suffix(metadata) if metadata else ""
    )


def render_tools(tools: Iterable[Mapping[str, Any]]) -> str:
    """Render the available tools in their stable registry order.

    No tool or property is sorted, filtered or mutated. Empty input has no legend.
    Both registry records (writes) and official catalog records (kind) are
    accepted. Other record fields are retained as literal JSON metadata.
    """
    signatures = [render_tool(tool) for tool in tools]
    return _LEGEND + "\n" + "\n".join(signatures) if signatures else ""
