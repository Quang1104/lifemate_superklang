"""Validate project-local tool definitions and gate actions before execution."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable

from jsonschema import Draft202012Validator, validators
from jsonschema.exceptions import SchemaError, ValidationError
from referencing import Registry

from .evidence import EvidenceLedger, EvidencePath
from .json_data import encode_json


_SINGLE_SCHEMAS = {
    "additionalProperties", "contains", "else", "if", "items", "not", "propertyNames",
    "then", "unevaluatedItems", "unevaluatedProperties", "contentSchema",
}
_ARRAYS_OF_SCHEMAS = {"allOf", "anyOf", "oneOf", "prefixItems"}
_MAPS_OF_SCHEMAS = {"$defs", "properties", "patternProperties", "dependentSchemas"}


def _validate_extensions(schema: dict[str, Any] | bool) -> None:
    if not isinstance(schema, dict):
        return
    for key, value in schema.items():
        if key.startswith("x-") and key not in {"x-one-of", "x-ref", "x-match"}:
            raise ValueError(f"Unsupported project-local schema extension: {key}")
        if key in {"$schema", "$id"}:
            raise ValueError("Tool schemas cannot switch dialect or scope")
        if key in {"$ref", "$dynamicRef"} and not value.startswith("#"):
            raise ValueError("Only local schema references are supported")
        if key == "x-ref" and (not isinstance(value, str) or not value):
            raise ValueError("x-ref must name an evidence kind")
        if key == "x-match" and value != "fuzzy":
            raise ValueError("Unknown x-match annotation")
        if key == "x-one-of":
            if (not isinstance(value, list) or not value or not all(
                isinstance(group, list) and group and all(
                    isinstance(field, str) and field in schema.get("properties", {})
                    for field in group
                ) for group in value
            )):
                raise ValueError("x-one-of must contain nonempty groups of declared fields")
        if key in _SINGLE_SCHEMAS:
            _validate_extensions(value)
        elif key in _ARRAYS_OF_SCHEMAS:
            for branch in value:
                _validate_extensions(branch)
        elif key in _MAPS_OF_SCHEMAS:
            for child_schema in value.values():
                _validate_extensions(child_schema)


def _validator(spec: "ToolSpec", ledger: EvidenceLedger, arguments: dict[str, Any],
               check_formats: bool) -> Any:
    scope_fields = dict(spec.reference_scopes)

    def validate_reference(validator: Any, kind: str, value: Any, schema: Any) -> Any:
        scope_field = scope_fields.get(kind)
        scope = arguments.get(scope_field) if scope_field is not None else None
        if not ledger.contains(kind, value, scope):
            yield ValidationError("Argument reference was not observed in this sample")

    def validate_groups(validator: Any, groups: list[list[str]], value: Any, schema: Any) -> Any:
        if isinstance(value, dict) and not any(
            all(value.get(field) not in (None, "", [], {}) for field in group) for group in groups
        ):
            yield ValidationError("One complete group of fields is required")

    with_evidence = validators.extend(Draft202012Validator, {
        "x-one-of": validate_groups, "x-ref": validate_reference,
    })
    return with_evidence(spec.parameters, registry=Registry(),
                         format_checker=Draft202012Validator.FORMAT_CHECKER if check_formats else None)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    app: str
    description: str
    parameters: dict[str, Any]
    writes: bool
    evidence_paths: tuple[EvidencePath, ...] = ()
    reference_scopes: tuple[tuple[str, str], ...] = ()


class ToolRejected(ValueError):
    """The requested action failed a pre-execution check."""


class ToolRegistry:
    def __init__(self, specs: Iterable[ToolSpec], *, check_formats: bool = True) -> None:
        self._check_formats = check_formats
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            if not all(isinstance(value, str) and value for value in (spec.name, spec.app)):
                raise ValueError("Tool name and app must be nonempty strings")
            if not isinstance(spec.description, str) or type(spec.writes) is not bool:
                raise ValueError("Tool description and write flag must be explicit")
            if spec.name in self._specs:
                raise ValueError(f"Duplicate tool name: {spec.name}")
            if not isinstance(spec.evidence_paths, tuple) or not all(
                isinstance(path, EvidencePath) for path in spec.evidence_paths
            ):
                raise ValueError("Tool evidence paths must be explicit")
            parameters = spec.parameters
            if (not isinstance(parameters, dict) or parameters.get("type") != "object"
                    or parameters.get("additionalProperties") is not False):
                raise ValueError("Tool parameters must be an object schema with additionalProperties false")
            encode_json(parameters, max_bytes=64_000)
            try:
                Draft202012Validator.check_schema(parameters)
                _validate_extensions(parameters)
            except SchemaError as exc:
                raise ValueError(f"Invalid parameter schema for {spec.name}") from exc
            self._specs[spec.name] = deepcopy(spec)

    def available(self, installed_apps: frozenset[str], allowed_tools: frozenset[str] | None = None
                  ) -> list[dict[str, Any]]:
        return [
            {"name": spec.name, "description": spec.description,
             "parameters": deepcopy(spec.parameters), "writes": spec.writes}
            for spec in self._specs.values()
            if spec.app in installed_apps and (allowed_tools is None or spec.name in allowed_tools)
        ]

    def authorize(self, name: str, arguments: dict[str, Any], installed_apps: frozenset[str],
                  ledger: EvidenceLedger, allowed_tools: frozenset[str] | None = None) -> ToolSpec:
        spec = self._specs.get(name)
        if (spec is None or spec.app not in installed_apps
                or (allowed_tools is not None and name not in allowed_tools)):
            raise ToolRejected("Tool is unavailable for the installed apps")
        if not isinstance(arguments, dict):
            raise ToolRejected("Tool arguments must be an object")
        try:
            error = next(_validator(spec, ledger, arguments, self._check_formats)
                         .iter_errors(arguments), None)
        except Exception as exc:
            raise ToolRejected("Tool schema could not be evaluated safely") from exc
        if error is not None:
            # Do not send submitted values or interpolated error strings into the next prompt.
            rule = error.validator
            raise ToolRejected(f"Tool arguments failed validation ({rule})")
        if name == "sys.ask_missing_info":
            target_name = arguments.get("tool_name")
            target = self._specs.get(target_name)
            fields = arguments.get("params")
            if (target is None or target.app == "sys" or target.app not in installed_apps
                    or (allowed_tools is not None and target_name not in allowed_tools)
                    or not isinstance(fields, list) or not fields or not all(
                        field in target.parameters.get("properties", {}) for field in fields
                    )):
                raise ToolRejected("Clarification must name available tool parameters")
        return deepcopy(spec)

    def evidence_paths(self, name: str) -> tuple[EvidencePath, ...]:
        spec = self._specs.get(name)
        return spec.evidence_paths if spec is not None else ()
