"""Data types for project-local inference, independent of official contracts."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PriorToolResult:
    name: str
    arguments: dict[str, Any]
    payload: dict[str, Any]


@dataclass(frozen=True)
class Sample:
    sample_id: str
    user_request: str
    installed_apps: frozenset[str]
    history: tuple[str, ...] = ()
    allowed_tools: frozenset[str] | None = None
    context: dict[str, Any] = field(default_factory=dict)
    prior_tool_results: tuple[PriorToolResult, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a nonempty string")
        if not isinstance(self.user_request, str) or not self.user_request:
            raise ValueError("user_request must be a nonempty string")
        if not isinstance(self.installed_apps, frozenset) or not all(
            isinstance(app, str) and app for app in self.installed_apps
        ):
            raise ValueError("installed_apps must be a frozenset of app names")
        if not isinstance(self.history, tuple) or not all(
            isinstance(item, str) for item in self.history
        ):
            raise ValueError("history must be a tuple of context strings")
        if self.allowed_tools is not None and (
            not isinstance(self.allowed_tools, frozenset) or not all(
                isinstance(tool, str) and tool for tool in self.allowed_tools
            )
        ):
            raise ValueError("allowed_tools must be a frozenset of tool names")
        if not isinstance(self.context, dict):
            raise ValueError("context must be a JSON object")
        if not isinstance(self.prior_tool_results, tuple) or not all(
            isinstance(result, PriorToolResult) for result in self.prior_tool_results
        ):
            raise ValueError("prior_tool_results must contain historical observations")
