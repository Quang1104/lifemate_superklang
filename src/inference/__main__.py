"""Run the synthetic smoke, TRAIN control, or configured local-model inference."""

import argparse
from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path
from statistics import mean
import sys
from time import monotonic

from ..agent.runner import Limits
from .mock import run_mock
from .official import KitError, OfficialKit
from .paths import safe_path
from .pipeline import predict_records, write_official_records


def _read_manifest(path: str | None) -> set[str] | None:
    if path is None:
        return None
    with safe_path(path).open(encoding="utf-8") as source:
        manifest = json.load(source)
    if manifest.get("source_split") != "train":
        raise KitError("Expected a TRAIN manifest")
    ids = manifest["validation_ids"]
    if not isinstance(ids, list) or not ids or not all(isinstance(item, str) for item in ids):
        raise KitError("Expected a nonempty validation_ids list in the TRAIN manifest")
    if len(set(ids)) != len(ids):
        raise KitError("TRAIN manifest contains repeated IDs")
    return set(ids)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    mock = commands.add_parser("mock", help="Exercise the runner with synthetic data")
    mock.add_argument("--samples", type=int, default=40)
    mock.add_argument("--output", type=Path)
    official = commands.add_parser("official", help="Run one split through its official ToolEnv")
    official.add_argument("--data-root", required=True)
    official.add_argument("--split", choices=("train", "public_test", "private_test"), required=True)
    official.add_argument("--output", required=True)
    official.add_argument("--model-dir", type=Path)
    official.add_argument("--asr-dir", type=Path)
    official.add_argument("--device", default="cpu")
    official.add_argument("--planner-mode", choices=("direct", "compiled"), default="direct")
    official.add_argument("--train-manifest")
    official.add_argument("--train-transcripts", action="store_true")
    official.add_argument("--control-noop", action="store_true")
    official.add_argument("--time-scale", type=float, default=1.0,
                          help="TRAIN only: multiply every time limit to measure accuracy on slower GPUs")
    official.add_argument("--train-report", type=Path,
                          help="Record TRAIN timing and failure status separately from official logs")
    commands.add_parser("oracle", help="Explain why inference never uses TRAIN labels")
    return parser


def _sample_model_timings(planner, transcribe) -> dict:
    planner_snapshot = getattr(planner, "timing_snapshot", None)
    asr_snapshot = getattr(transcribe, "timing_snapshot", None)
    calls = planner_snapshot() if callable(planner_snapshot) else []
    audio = asr_snapshot() if callable(asr_snapshot) else []
    fields = ("prompt_tokens", "generated_tokens", "reused_prefix_tokens",
              "grammar_compile_ms", "generate_ms", "grammar_status")
    return {"asr_ms": sum(row["asr_ms"] for row in audio), "planner_calls": len(calls),
            "asr_timing_complete": all(row.get("timing_complete", True) for row in audio),
            "timing_complete": [row.get("timing_complete", True) for row in calls],
            **{field: [row[field] for row in calls] for field in fields}}


def _model_timing_summary(samples: list[dict]) -> dict:
    generate = sorted(value for sample in samples for value in sample["generate_ms"])
    prompts = [value for sample in samples for value in sample["prompt_tokens"]]
    return {"mean_planner_generate_ms": mean(generate) if generate else 0.0,
            "p95_planner_generate_ms": generate[math.ceil(0.95 * len(generate)) - 1] if generate else 0.0,
            "mean_prompt_tokens": mean(prompts) if prompts else 0.0,
            "prompts_above_12000": sum(value > 12_000 for value in prompts),
            "total_grammar_compile_ms": sum(value for sample in samples
                                            for value in sample["grammar_compile_ms"])}


def _run_official(args: argparse.Namespace) -> dict:
    train = args.split == "train"
    if not math.isfinite(args.time_scale) or args.time_scale < 1:
        raise KitError("The time scale must be a finite number of at least 1")
    if not train and (args.control_noop or args.train_manifest or args.train_transcripts or args.train_report
                      or args.time_scale != 1):
        raise KitError("TRAIN controls and transcripts require the TRAIN split")
    manifest = _read_manifest(args.train_manifest)
    if args.control_noop and (not manifest or args.model_dir or args.asr_dir or args.train_transcripts):
        raise KitError("The no-op control requires a TRAIN manifest and no model options")
    if not args.control_noop and (not args.model_dir or (not args.asr_dir and not args.train_transcripts)):
        raise KitError("Supply local planner and ASR weights, or use TRAIN transcripts for a TRAIN run")
    if args.asr_dir and args.train_transcripts:
        raise KitError("Choose either ASR or released TRAIN transcripts")
    if args.control_noop and args.planner_mode != "direct":
        raise KitError("The no-op control cannot use the compiled planner")
    output = safe_path(args.output)
    report_path = safe_path(args.train_report) if args.train_report else None
    if report_path == output:
        raise KitError("TRAIN report and official output require separate files")
    kit = OfficialKit(args.data_root, args.split)
    defaults = Limits()
    limits = replace(defaults, sample_seconds=defaults.sample_seconds * args.time_scale,
                     planner_seconds=defaults.planner_seconds * args.time_scale,
                     tool_seconds=defaults.tool_seconds * args.time_scale)
    planner_factory = None
    parameter_counts = None
    gpu = None
    load_started = monotonic()
    if args.control_noop:
        def planner(_):
            return '{"action":"finish","reply":""}'

        transcribe = None
    else:
        from ..agent.local_model import LocalPlanner, LocalWhisper, initialize_runtime

        initialize_runtime()
        planner = LocalPlanner(args.model_dir, device=args.device)
        transcribe = LocalWhisper(args.asr_dir, device=args.device) if args.asr_dir else None
        parameter_counts = {"planner": planner.parameter_count,
                            "asr": transcribe.parameter_count if transcribe else 0}
        if parameter_counts["planner"] >= 5_000_000_000:
            raise KitError("The planner must contain fewer than five billion parameters")
        if sum(parameter_counts.values()) >= 6_000_000_000:
            raise KitError("Combined models must contain fewer than six billion parameters")
        if args.planner_mode == "compiled":
            from ..agent.compiled_plan import CompiledPlanner

            planner_factory = CompiledPlanner(planner.generate, tool_limit=limits.max_steps - 1,
                                              constrain=True).for_sample
        if args.device.startswith("cuda"):
            import torch

            gpu = torch.cuda
    load_ms = int((monotonic() - load_started) * 1_000)
    statuses: Counter[str] = Counter()
    timings = []
    details = []

    def track(sample_id, status, elapsed_ms, result, exported):
        statuses[status] += 1
        timings.append(elapsed_ms)
        if report_path is not None:
            details.append({"id": sample_id, "status": status, "time": elapsed_ms, "exported": exported,
                            "steps": result.steps if result else 0,
                            "dispatched_calls": result.dispatched_calls if result else 0,
                            "rejections": result.rejections if result else 0,
                            **_sample_model_timings(planner, transcribe)})

    count = write_official_records(output, predict_records(
        kit, planner, transcribe, sample_ids=manifest, control_noop=args.control_noop,
        limits=limits, planner_factory=planner_factory, on_sample=track,
        continue_train_failures=report_path is not None,
    ), allow_empty=report_path is not None)
    timing = sorted(timings)
    summary = {"mode": "train_noop_control" if args.control_noop else "official_local_model",
               "planner_mode": args.planner_mode, "split": args.split, "records": count,
               "attempted": len(timing), "statuses": dict(statuses), "parameters": parameter_counts,
               "planner_artifact": str(safe_path(args.model_dir)) if args.model_dir else None,
               "asr_artifact": str(safe_path(args.asr_dir)) if args.asr_dir else None,
               "device": args.device, "time_scale": args.time_scale,
               "grammar": dict(planner.grammar_stats) if hasattr(planner, "grammar_stats") else None, "transcript_source": (
                   "control" if args.control_noop else "train_oracle" if args.train_transcripts else "asr"),
               "model_load_ms": load_ms, "mean_sample_time_ms": mean(timing),
               "p95_sample_time_ms": timing[math.ceil(0.95 * len(timing)) - 1],
               "max_sample_time_ms": timing[-1], "competition_score": "NOT MEASURED"}
    if gpu is not None:
        summary["gpu_name"] = gpu.get_device_name(args.device)
        summary["peak_gpu_memory_mib"] = round(gpu.max_memory_allocated(args.device) / 2**20, 1)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({**summary, **_model_timing_summary(details),
                                           "samples": details}, indent=2) + "\n",
                               encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "oracle":
            raise KitError("Inference never consumes TRAIN gold calls. Use the separate local TRAIN oracle analysis.")
        if args.command == "mock":
            report = run_mock(args.samples)
            if args.output:
                output = safe_path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            if report["checks_passed"] != args.samples:
                print(json.dumps(report), file=sys.stderr)
                return 1
        else:
            report = _run_official(args)
    except (KitError, OSError, ValueError, KeyError) as exc:
        print(f"Inference stopped: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
