"""Run released input records and export only official ToolEnv outputs."""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic
from typing import Any

from ..agent.planner import Planner
from ..agent.runner import CallbackError, CallTimeout, Limits, RunResult, _bounded_call, run_sample
from ..tooling.json_data import encode_json
from .official import KitError, OfficialEnvironment, OfficialKit, OfficialRecord, installed_permissions
from .paths import safe_path
from .types import PriorToolResult, Sample


Transcribe = Callable[[Path], str]


def _transcribe(kit: OfficialKit, sample_id: str, item: dict[str, Any], transcribe: Transcribe,
                deadline: float, *, history: bool) -> str:
    audio_path = kit.audio_path(sample_id, item["audio"], history=history)
    deadline_transcriber = getattr(transcribe, "transcribe", None)
    if callable(deadline_transcriber):
        callback = lambda: deadline_transcriber(audio_path, deadline=deadline - 0.1)
    else:
        callback = lambda: transcribe(audio_path)
    text = _bounded_call(callback, deadline - monotonic())
    if not isinstance(text, str) or not text.strip():
        raise CallbackError("ASR returned no usable transcript")
    return text.strip()


def _sample_from_row(kit: OfficialKit, row: dict[str, Any], transcribe: Transcribe | None,
                     deadline: float, *, control_noop: bool) -> Sample:
    sample_id = row["id"]
    apps, tools = installed_permissions(row["context"])
    context = {key: row["context"][key] for key in ("current_time", "timezone", "location", "user_profile")
               if key in row["context"]}
    if control_noop:
        return Sample(sample_id, "Held-out TRAIN no-op control", apps, allowed_tools=tools)
    if transcribe is None:
        user_text = row["audio_text"]
    else:
        user_text = _transcribe(kit, sample_id, row, transcribe, deadline, history=False)

    history = []
    prior = []
    for item in row["history"]:
        if item.get("role") == "user":
            text = item["audio_text"] if transcribe is None else _transcribe(
                kit, sample_id, item, transcribe, deadline, history=True,
            )
            history.append(encode_json({"role": "user", "text": text}, max_bytes=32_000))
        elif item.get("role") == "assistant":
            context_item = {key: item[key] for key in ("role", "text", "tool_calls", "tool_results")
                            if key in item}
            history.append(encode_json(context_item, max_bytes=64_000))
            for call, payload in zip(item.get("tool_calls", ()), item.get("tool_results", ())):
                if (isinstance(call, dict) and isinstance(call.get("name"), str)
                        and isinstance(call.get("arguments"), dict) and isinstance(payload, dict)):
                    prior.append(PriorToolResult(call["name"], call["arguments"], payload))
        else:
            raise KitError("Unrecognized released history role")
    return Sample(sample_id, user_text, apps, tuple(history), tools, context, tuple(prior))


def predict_records(kit: OfficialKit, planner: Planner, transcribe: Transcribe | None, *,
                    sample_ids: set[str] | None = None, control_noop: bool = False,
                    limits: Limits = Limits(), on_result: Callable[[RunResult, int], None] | None = None,
                    planner_factory: Callable[[], Planner] | None = None,
                    export_reserve_seconds: float = 3.0,
                    on_sample: Callable[[str, str, int, RunResult | None, bool], None] | None = None,
                    continue_train_failures: bool = False,
                    ) -> Iterator[OfficialRecord]:
    if limits.sample_seconds > 30 and kit.split != "train":  # TRAIN may scale limits to measure accuracy only.
        raise KitError("Official inference must fit the 30-second per-sample limit")
    if not 0 <= export_reserve_seconds < limits.sample_seconds:
        raise KitError("Export reserve must fit the sample time budget")
    if continue_train_failures and kit.split != "train":
        raise KitError("Failure-tolerant scoring is only available for TRAIN")
    if kit.split != "train" and (transcribe is None or control_noop or sample_ids is not None):
        raise KitError("A test run requires ASR and the complete test split")
    for row in kit.rows(sample_ids):
        started = monotonic()
        deadline = started + limits.sample_seconds
        execution_deadline = deadline - export_reserve_seconds
        environment: OfficialEnvironment | None = None
        result = None

        def open_environment(sample: Sample) -> OfficialEnvironment:
            nonlocal environment
            if environment is None:
                environment = kit.open_environment(sample)
            return environment

        try:
            try:
                sample = _sample_from_row(kit, row, transcribe, execution_deadline, control_noop=control_noop)
                remaining = execution_deadline - monotonic()
                if remaining <= 0:
                    raise CallTimeout("Preprocessing deadline elapsed")
            except (CallTimeout, CallbackError) as exc:
                status = ("preprocessing_timeout" if isinstance(exc, CallTimeout)
                          or isinstance(exc.__cause__, TimeoutError) else "preprocessing_error")
                # No app call was dispatched. Export the genuine untouched environment.
                apps, tools = installed_permissions(row["context"])
                sample = Sample(row["id"], "Audio unavailable", apps, allowed_tools=tools)
                _bounded_call(lambda: open_environment(sample), deadline - monotonic())
                result = RunResult(status, None, 0, 0, 0, True)
            else:
                active_planner = planner_factory() if planner_factory is not None else planner
                result = run_sample(sample, kit.registry, active_planner, open_environment,
                                    limits=replace(limits, sample_seconds=remaining))
            if environment is None or not result.session_safe_to_export:
                raise KitError(f"Sample {row['id']} failed safely ({result.status})")
            calls, state_hash = _bounded_call(environment.read_output, deadline - monotonic())
            if monotonic() > deadline:
                raise CallTimeout("Sample exceeded its deadline")
        except (CallTimeout, CallbackError, KitError, OSError, ValueError, KeyError) as exc:
            if not continue_train_failures:
                raise KitError(f"Sample {row['id']} could not export official output: {exc}") from exc
            timed_out = isinstance(exc, CallTimeout) or (
                isinstance(exc, CallbackError) and isinstance(exc.__cause__, TimeoutError)
            )
            status = result.status if result is not None else (
                "preprocessing_timeout" if timed_out else "preprocessing_error")
            if timed_out and not status.endswith("_timeout"):
                status = "export_timeout"
            if not timed_out and not status.endswith(("_timeout", "_error")):
                status = "export_error"
            if on_sample is not None:
                on_sample(row["id"], status, int((monotonic() - started) * 1_000), result, False)
            continue
        elapsed_ms = int((monotonic() - started) * 1_000)
        if on_result is not None:
            on_result(result, elapsed_ms)
        if on_sample is not None:
            on_sample(row["id"], result.status, elapsed_ms, result, True)
        yield OfficialRecord(row["id"], calls, state_hash, elapsed_ms if kit.split == "private_test" else None)


def write_official_records(path: str | Path, records: Iterable[OfficialRecord], *,
                           allow_empty: bool = False) -> int:
    output = safe_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    count = 0
    seen: set[str] = set()
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", delete=False,
                                dir=output.parent, prefix=f".{output.name}.") as sink:
            temporary = Path(sink.name)
            for record in records:
                if not isinstance(record, OfficialRecord) or record.sample_id in seen:
                    raise KitError("Submission expects one official record per sample ID")
                seen.add(record.sample_id)
                sink.write(json.dumps(record.as_dict(), ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
        if not count and not allow_empty:
            raise KitError("Submission is empty")
        temporary.replace(output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return count
