"""Official ToolEnv participant API and released split metadata adapter."""

from collections.abc import Iterator
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Any, Protocol

from ..tooling.lifemate_catalog import build_registry
from ..tooling.registry import ToolRejected
from .paths import safe_path
from .types import Sample


_SPLITS = {"train", "public_test", "private_test"}
_SAMPLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SYSTEM_TOOL = "sys.ask_missing_info"


class KitError(RuntimeError):
    """The released kit or official response did not satisfy its contract."""


class _ParticipantSession(Protocol):
    def validate(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...
    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...
    def get_tool_calls(self) -> list[dict[str, Any]]: ...
    def get_state_hash(self) -> str: ...


@dataclass(frozen=True)
class OfficialRecord:
    sample_id: str
    tool_calls: list[dict[str, Any]]
    state_hash: str
    elapsed_ms: int | None

    def as_dict(self) -> dict[str, Any]:
        row = {"id": self.sample_id, "tool_calls": self.tool_calls, "state_hash": self.state_hash}
        if self.elapsed_ms is not None:
            row["time"] = self.elapsed_ms
        return row


class OfficialEnvironment:
    def __init__(self, session: _ParticipantSession) -> None:
        self._session = session

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        preflight = self._session.validate(name, arguments)
        if not isinstance(preflight, dict) or preflight.get("ok") is not True:
            raise ToolRejected("Official preflight did not approve the call")
        return self._session.call(name, arguments)

    def read_output(self) -> tuple[list[dict[str, Any]], str]:
        calls = self._session.get_tool_calls()
        state_hash = self._session.get_state_hash()
        if (not isinstance(calls, list) or not all(
            isinstance(call, dict) and set(call) == {"name", "arguments"}
            and isinstance(call["name"], str) and isinstance(call["arguments"], dict)
            for call in calls
        ) or not isinstance(state_hash, str) or not _SHA256.fullmatch(state_hash)):
            raise KitError("Official ToolEnv returned a malformed submission record")
        return calls, state_hash


def _load_release(package_file: Path, split: str) -> ModuleType:
    name = f"_lifemate_release_{split}"
    existing = sys.modules.get(name)
    if existing is not None:
        if existing.__file__ != str(package_file):
            raise KitError("This process already loaded a different kit for the split")
        return existing
    spec = importlib.util.spec_from_file_location(name, package_file)
    if spec is None or spec.loader is None:
        raise KitError("Cannot load the split's official zaic_tools package")
    module = importlib.util.module_from_spec(spec)
    original_bytecode_setting = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if not callable(getattr(module, "ToolEnv", None)):
            raise KitError("Official package does not export ToolEnv")
    except Exception as exc:
        for imported in tuple(sys.modules):
            if imported == name or imported.startswith(name + "."):
                del sys.modules[imported]
        raise KitError("Cannot import the split's official zaic_tools package") from exc
    finally:
        sys.dont_write_bytecode = original_bytecode_setting
    return module


class OfficialKit:
    def __init__(self, root: str | Path, split: str) -> None:
        if split not in _SPLITS:
            raise KitError("Unrecognized official split")
        self.root = safe_path(root)
        self.split = split
        self.directory = self.root / split
        self._input = safe_path(self.directory / f"{split}.jsonl")
        catalog_path = safe_path(self.directory / f"{split}_tools_catalog.json")
        with catalog_path.open(encoding="utf-8") as source:
            self.registry = build_registry(json.load(source))
        package = safe_path(self.directory / "zaic_tools" / "__init__.py")
        self._release = _load_release(package, split)

    def rows(self, sample_ids: set[str] | None = None) -> Iterator[dict[str, Any]]:
        seen = set()
        with self._input.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                raw = json.loads(line)
                sample_id = raw.get("id")
                if not isinstance(sample_id, str) or not _SAMPLE_ID.fullmatch(sample_id):
                    raise KitError("Malformed split sample ID")
                if sample_ids is not None and sample_id not in sample_ids:
                    continue
                if sample_id in seen:
                    raise KitError(f"Duplicate split sample ID: {sample_id}")
                if self.split != "train" and ("label" in raw or "audio_text" in raw):
                    raise KitError("Test input includes unexpected target fields")
                seen.add(sample_id)
                fields = ("id", "audio", "context", "history")
                if self.split == "train":
                    fields += ("audio_text",)
                yield {key: raw[key] for key in fields}
        if sample_ids is not None and seen != sample_ids:
            raise KitError("The requested ID set is incomplete in the selected split")
        if not seen:
            raise KitError("No samples selected")

    def audio_path(self, sample_id: str, relative_path: str, *, history: bool) -> Path:
        if not _SAMPLE_ID.fullmatch(sample_id) or not isinstance(relative_path, str):
            raise KitError("Malformed audio path metadata")
        stem = re.escape(sample_id) + (r"_u[1-9]\d*" if history else "")
        expected = rf"audio/{re.escape(sample_id)}/{stem}\.wav"
        if not re.fullmatch(expected, relative_path):
            raise KitError("Audio path does not match the released sample layout")
        return safe_path(self.directory / relative_path)

    def open_environment(self, sample: Sample) -> OfficialEnvironment:
        if not _SAMPLE_ID.fullmatch(sample.sample_id):
            raise KitError("Malformed sample ID")
        return OfficialEnvironment(self._release.ToolEnv(self.split, sample.sample_id, root=self.root))


def installed_permissions(context: dict[str, Any]) -> tuple[frozenset[str], frozenset[str]]:
    installed = context.get("installed_apps")
    if not isinstance(installed, dict):
        raise KitError("Expected a per-sample installed app mapping")
    app_names = {"sys"}
    tool_names = {_SYSTEM_TOOL}
    for app, spec in installed.items():
        if not isinstance(app, str) or not app or not isinstance(spec, dict):
            raise KitError("Malformed installed app mapping")
        tools = spec.get("tools")
        if not isinstance(tools, list) or not all(
            isinstance(tool, str) and tool.partition(".")[0] == app for tool in tools
        ):
            raise KitError("Malformed installed tool allowlist")
        app_names.add(app)
        tool_names.update(tools)
    return frozenset(app_names), frozenset(tool_names)
