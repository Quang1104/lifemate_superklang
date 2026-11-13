"""Run the synthetic smoke, TRAIN control, or configured local-model inference."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from .mock import run_mock
from .official import KitError, OfficialKit
from .paths import safe_path
from .pipeline import predict_records, write_official_records


def _read_manifest(path: str | None) -> set[str] | None:
    if path is None:
        return None
    with safe_path(path).open(encoding="utf-8") as source:
        ids = json.load(source)["validation_ids"]
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
    official.add_argument("--train-manifest")
    official.add_argument("--train-transcripts", action="store_true")
    official.add_argument("--control-noop", action="store_true")
    commands.add_parser("oracle", help="Explain why inference never uses TRAIN labels")
    return parser


def _run_official(args: argparse.Namespace) -> dict:
    train = args.split == "train"
    if not train and (args.control_noop or args.train_manifest or args.train_transcripts):
        raise KitError("TRAIN controls and transcripts require the TRAIN split")
    manifest = _read_manifest(args.train_manifest)
    if args.control_noop and (not manifest or args.model_dir or args.asr_dir or args.train_transcripts):
        raise KitError("The no-op control requires a TRAIN manifest and no model options")
    if not args.control_noop and (not args.model_dir or (not args.asr_dir and not args.train_transcripts)):
        raise KitError("Supply local planner and ASR weights, or use TRAIN transcripts for a TRAIN run")
    if args.asr_dir and args.train_transcripts:
        raise KitError("Choose either ASR or released TRAIN transcripts")
    kit = OfficialKit(args.data_root, args.split)
    if args.control_noop:
        def planner(_):
            return '{"action":"finish","reply":""}'

        transcribe = None
    else:
        from ..agent.local_model import LocalPlanner, LocalWhisper, initialize_runtime

        initialize_runtime()
        planner = LocalPlanner(args.model_dir, device=args.device)
        transcribe = LocalWhisper(args.asr_dir, device=args.device) if args.asr_dir else None
        if planner.parameter_count + (transcribe.parameter_count if transcribe else 0) >= 6_000_000_000:
            raise KitError("Combined models must contain fewer than six billion parameters")
    statuses: Counter[str] = Counter()
    timings = []

    def track(result, elapsed_ms):
        statuses[result.status] += 1
        timings.append(elapsed_ms)

    count = write_official_records(args.output, predict_records(
        kit, planner, transcribe, sample_ids=manifest, control_noop=args.control_noop, on_result=track,
    ))
    return {"mode": "train_noop_control" if args.control_noop else "official_local_model",
            "split": args.split, "records": count, "statuses": dict(statuses),
            "max_sample_time_ms": max(timings), "competition_score": "NOT MEASURED"}


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
