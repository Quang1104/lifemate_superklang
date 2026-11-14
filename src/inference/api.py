"""Shared in-process model loading and private-test inference for CLI and notebook."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from ..agent.runner import Limits
from .official import KitError


DEFAULT_SEED = 20261112


def seed_runtime(seed: int = DEFAULT_SEED) -> None:
    """Set offline defaults and Python/NumPy/PyTorch seeds before model loading."""
    from ..agent.local_model import initialize_runtime

    initialize_runtime() if seed == DEFAULT_SEED else initialize_runtime(seed)


@dataclass(frozen=True)
class ModelHandle:
    """Models loaded once; per-sample planner sessions remain owned by the pipeline."""

    planner: Any
    transcribe: Any
    planner_factory: Callable | None
    parameter_counts: dict[str, int] | None
    gpu: Any
    load_ms: int
    model_dir: Path | None
    asr_dir: Path | None
    device: str
    planner_mode: str
    control_noop: bool
    limits: Limits


def load_models(model_dir: str | Path | None, asr_dir: str | Path | None, *,
                device: str = "cpu", planner_mode: str = "native",
                control_noop: bool = False, limits: Limits = Limits()) -> ModelHandle:
    """Seed and load the same local planner/ASR used by the official CLI exactly once."""
    if planner_mode not in {"direct", "compiled", "native"}:
        raise KitError("Unknown planner mode")
    if control_noop and (model_dir or asr_dir or planner_mode != "direct"):
        raise KitError("The no-op control cannot load models or use a planner wrapper")
    if not control_noop and model_dir is None:
        raise KitError("Supply local planner weights")
    model_dir = Path(model_dir) if model_dir is not None else None
    asr_dir = Path(asr_dir) if asr_dir is not None else None
    planner_factory = None
    parameter_counts = None
    gpu = None
    load_started = monotonic()
    if control_noop:
        def planner(_):
            return '{"action":"finish","reply":""}'

        transcribe = None
    else:
        from ..agent.local_model import LocalPlanner, LocalWhisper

        seed_runtime()
        planner = LocalPlanner(model_dir, device=device)
        transcribe = LocalWhisper(asr_dir, device=device) if asr_dir else None
        parameter_counts = {"planner": planner.parameter_count,
                            "asr": transcribe.parameter_count if transcribe else 0}
        if parameter_counts["planner"] >= 5_000_000_000:
            raise KitError("The planner must contain fewer than five billion parameters")
        if sum(parameter_counts.values()) >= 6_000_000_000:
            raise KitError("Combined models must contain fewer than six billion parameters")
        if planner_mode == "compiled":
            from ..agent.compiled_plan import CompiledPlanner

            planner_factory = CompiledPlanner(planner.generate, tool_limit=limits.max_steps - 1,
                                              constrain=True).for_sample
        if planner_mode == "native":
            from ..agent.native_planner import NativePlanner

            planner_factory = NativePlanner(planner.chat).for_sample
        if device.startswith("cuda"):
            import torch

            gpu = torch.cuda
    load_ms = int((monotonic() - load_started) * 1_000)
    return ModelHandle(planner, transcribe, planner_factory, parameter_counts, gpu, load_ms,
                       model_dir, asr_dir, device, planner_mode, control_noop, limits)


def run_private_test(models: ModelHandle, *, data_root: str | Path = "/data",
                     output: str | Path = "/result/jupyter_submission.jsonl") -> dict:
    """Run the complete mounted private split through the shared official CLI loop."""
    if models.control_noop or models.transcribe is None:
        raise KitError("Private-test inference requires loaded planner and ASR models")
    # Imported lazily so the CLI can import model loading without a module cycle.
    from .__main__ import _run_official

    args = argparse.Namespace(
        split="private_test", data_root=data_root, output=output,
        model_dir=models.model_dir, asr_dir=models.asr_dir, device=models.device,
        planner_mode=models.planner_mode, train_manifest=None, train_transcripts=False,
        control_noop=False, time_scale=1.0, train_report=None,
    )
    return _run_official(args, models=models)
