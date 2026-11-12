"""Keep references observed at explicitly declared successful result paths."""

from dataclasses import dataclass
from typing import Any, Iterator


ReferenceValue = str | int


def _values_at_path(value: Any, path: tuple[str | int, ...]) -> Iterator[Any]:
    if not path:
        yield value
        return
    part, *rest = path
    remaining = tuple(rest)
    if part == "*" and isinstance(value, list):
        for entry in value:
            yield from _values_at_path(entry, remaining)
    elif isinstance(part, str) and isinstance(value, dict) and part in value:
        yield from _values_at_path(value[part], remaining)
    elif type(part) is int and isinstance(value, list) and part < len(value):
        yield from _values_at_path(value[part], remaining)


@dataclass(frozen=True)
class EvidencePath:
    kind: str
    path: tuple[str | int, ...]
    scope_argument: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("Evidence kind must be a nonempty string")
        if not isinstance(self.path, tuple) or not all(
            isinstance(part, str) or (type(part) is int and part >= 0) for part in self.path
        ):
            raise ValueError("Evidence path must be a tuple of keys or nonnegative indexes")


class EvidenceLedger:
    def __init__(self) -> None:
        self._references: dict[tuple[str, ReferenceValue | None], set[tuple[type, ReferenceValue]]] = {}

    def record(self, payload: Any, paths: tuple[EvidencePath, ...], arguments: dict[str, Any]) -> None:
        for evidence in paths:
            scope = None
            if evidence.scope_argument is not None:
                scope = arguments.get(evidence.scope_argument)
                if type(scope) not in (str, int):
                    continue
            for value in _values_at_path(payload, evidence.path):
                if (isinstance(value, str) and value) or type(value) is int:
                    self._references.setdefault((evidence.kind, scope), set()).add((type(value), value))

    def contains(self, kind: str, value: Any, scope: ReferenceValue | None = None) -> bool:
        if type(value) not in (str, int) or (scope is not None and type(scope) not in (str, int)):
            return False
        return (type(value), value) in self._references.get((kind, scope), ())
