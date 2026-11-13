"""Compact JSON Schema for the compiled plan's structural decoding contract.

The compiler remains responsible for grounding, unique step IDs, dependency order,
SOURCE depth and resolved parameter types. Shared references keep this grammar
linear in the installed signatures rather than expanding the recursive SOURCE DSL.
"""


def _object(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": False}


def plan_json_schema(tools: list[dict], tool_limit: int) -> dict:
    """Allow only installed signatures and at most tool_limit planned calls.

    Raw scalars and arrays are deliberate alternatives to typed SOURCES: the
    compiler's grounded coercion must still approve them before any tool call.
    Input is the registry's installed-tool list, never a catalog of all tools.
    """
    if type(tool_limit) is not int or tool_limit < 0:
        raise ValueError("tool_limit must be a nonnegative integer")

    ref = lambda name: {"$ref": f"#/$defs/{name}"}
    scalar = {"type": ["string", "number", "boolean", "null"]}
    string = {"type": "string", "minLength": 1}
    identifier = {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_-]{0,31}$",
                  "maxLength": 32}
    index = {"type": "integer", "minimum": 1}
    mode = {"enum": ["text", "integer", "number", "money", "datetime", "date", "time", "duration"]}
    source_map = {"type": "object", "additionalProperties": ref("source")}
    definitions = {
        "id": identifier,
        "path": {"type": "array", "items": string, "maxItems": 8},
        "nonempty_path": {"type": "array", "items": string, "minItems": 1, "maxItems": 8},
        "json": {"anyOf": [scalar, {"type": "array", "items": ref("json")},
                            {"type": "object", "additionalProperties": ref("json")}]},
        "argument": {"anyOf": [ref("source"), scalar,
                                {"type": "array", "items": ref("json")}]},
        "condition": _object({"left": ref("source"),
                               "op": {"enum": ["lt", "le", "gt", "ge", "eq", "ne"]},
                               "right": ref("source")}, ["left", "op", "right"]),
        "after": {"type": "array", "items": ref("id")},
        "source": {"anyOf": [
            _object({"text": string, "as": mode}, ["text"]),
            _object({"value": scalar}, ["value"]),
            _object({"context": ref("nonempty_path")}, ["context"]),
            _object({"history": index, "path": ref("nonempty_path")}, ["history", "path"]),
            _object({"history_text": index, "text": string,
                     "as": {"enum": ["text", "integer", "number", "money", "duration"]}},
                    ["history_text", "text"]),
            _object({"result": ref("id"), "path": ref("path"),
                     "where": source_map, "field": ref("path")}, ["result"]),
            _object({"lookup": {**source_map, "minProperties": 1}}, ["lookup"]),
            _object({"shift": ref("source"), "by": ref("source"),
                     "dir": {"enum": ["later", "earlier"]}}, ["shift", "by", "dir"]),
            _object({"object": source_map}, ["object"]),
            _object({"array": {"type": "array", "items": ref("source")}}, ["array"]),
        ]},
    }
    variants = []
    seen = set()
    for tool in tools:
        name, parameters = tool["name"], tool["parameters"]
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("Installed tool names must be nonempty and unique")
        seen.add(name)
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])
        if (not isinstance(properties, dict) or not all(isinstance(key, str) for key in properties)
                or not isinstance(required, list)
                or not all(isinstance(key, str) and key in properties for key in required)):
            raise ValueError("Expected declared tool parameter names")
        variants.append(_object({
            "id": ref("id"), "tool": {"const": name}, "anchor": string,
            "args": _object({key: ref("argument") for key in properties}, list(required)),
            "after": ref("after"), "when": ref("condition"),
        }, ["id", "tool", "anchor", "args"]))

    # An empty installed list permits only a no-op, without an invalid anyOf:[].
    steps = {"type": "array", "maxItems": tool_limit if variants else 0,
             "items": {"anyOf": variants} if variants else False}
    return {**_object({"steps": steps}, ["steps"]), "$defs": definitions}
