"""Prepare the locked TRAIN manifest and compare two measured prediction runs."""

import argparse
from collections import defaultdict
import json

from research.data_audit import extract_tools, read_jsonl
from research.published_scorer import score_sample
from src.inference.paths import safe_path


def read_json(path):
    return json.loads(safe_path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    target = safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train_ids(path):
    manifest = read_json(path)
    ids = manifest["validation_ids"]
    if (manifest.get("source_split") != "train" or not isinstance(ids, list) or not ids
            or not all(isinstance(item, str) and item for item in ids) or len(set(ids)) != len(ids)):
        raise ValueError("Expected unique validation_ids in a nonempty TRAIN manifest")
    return ids


def locked_ids(validation, smoke):
    validation_ids = train_ids(validation)
    smoke_ids = set(train_ids(smoke))
    if not smoke_ids < set(validation_ids):
        raise ValueError("Smoke IDs must be a proper subset of validation IDs")
    return [sample_id for sample_id in validation_ids if sample_id not in smoke_ids]


def index(rows):
    result = {}
    for row in rows:
        if row["id"] in result:
            raise ValueError("Duplicate sample ID")
        result[row["id"]] = row
    return result


def average(values):
    return sum(values) / len(values) if values else None


def evaluate(gold, predictions_path, report_path, tools):
    predictions = index(read_jsonl(safe_path(predictions_path)))
    report = read_json(report_path)
    diagnostics = index(report["samples"])
    if report["split"] != "train" or diagnostics.keys() != gold.keys():
        raise ValueError("TRAIN diagnostics must cover exactly the chosen manifest")
    if predictions.keys() != {key for key, value in diagnostics.items() if value["exported"]}:
        raise ValueError("Exported IDs in the diagnostic report and JSONL disagree")
    scores = {}
    observed = []
    losses = defaultdict(float)
    timeouts = failures = 0
    for sample_id, expected in gold.items():
        diagnostic = diagnostics[sample_id]
        status = diagnostic["status"]
        if status.endswith("_timeout") or diagnostic["time"] > 30_000:
            timeouts += 1
            reason = "timeout"
        elif not diagnostic["exported"] or status.endswith("_error") or status == "plan_invalid":
            failures += 1
            reason = "invalid_plan" if status == "plan_invalid" else "execution_failure"
        else:
            measured = score_sample(expected, predictions[sample_id], tools)
            observed.append(measured)
            scores[sample_id] = measured["score"]
            if measured["score"] < 1:
                if measured["gold_empty"]:
                    reason = "unexpected_action"
                elif not predictions[sample_id]["tool_calls"]:
                    reason = "missing_all_actions"
                elif measured["state_match"] is False:
                    reason = "state_mismatch"
                else:
                    reason = "call_mismatch_with_correct_state"
                losses[reason] += 1 - measured["score"]
            continue
        scores[sample_id] = 0.0  # Measured failures get zero without manufacturing a prediction.
        losses[reason] += 1
    return scores, {
        "planner_mode": report["planner_mode"],
        "local_proxy_0_to_100": 100 * average(list(scores.values())),
        "state_hash_accuracy_nonempty_completed": average(
            [item["state_match"] for item in observed if item["state_match"] is not None]),
        "call_f1_on_correct_state": average(
            [item["call_f1"] for item in observed if item["state_match"] is True]),
        "empty_accuracy_completed": average([item["score"] for item in observed if item["gold_empty"]]),
        "timeout_count": timeouts,
        "other_failure_count": failures,
        "p95_sample_time_ms": report["p95_sample_time_ms"],
        "max_sample_time_ms": report["max_sample_time_ms"],
        "peak_gpu_memory_mib": report.get("peak_gpu_memory_mib"),
        "lost_proxy_points_by_outcome": {
            reason: 100 * points / len(gold) for reason, points in sorted(losses.items())},
        "source": {key: report.get(key) for key in ("mode", "planner_artifact", "asr_artifact", "parameters",
                                                    "device", "gpu_name", "transcript_source")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--validation", default="research/validation_split.json")
    prepare.add_argument("--smoke", default="research/smoke_split.json")
    prepare.add_argument("--output", required=True)
    compare = subparsers.add_parser("compare")
    for name in ("manifest", "gold", "catalog", "direct", "direct-report", "compiled",
                 "compiled-report", "output"):
        compare.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        ids = locked_ids(args.validation, args.smoke)
        write_json(args.output, {"source_split": "train", "validation_ids": ids})
        print(f"TRAIN samples outside smoke: {len(ids)}")
        return

    ids = train_ids(args.manifest)
    gold_rows = index(read_jsonl(safe_path(args.gold)))
    if not set(ids) <= gold_rows.keys():
        raise ValueError("TRAIN gold export does not cover the manifest")
    gold = {sample_id: gold_rows[sample_id] for sample_id in ids}
    tools = extract_tools(read_json(args.catalog))
    direct, direct_summary = evaluate(gold, args.direct, args.direct_report, tools)
    compiled, compiled_summary = evaluate(gold, args.compiled, args.compiled_report, tools)
    if (direct_summary["planner_mode"], compiled_summary["planner_mode"]) != ("direct", "compiled"):
        raise ValueError("Pass the direct and compiled reports in the matching argument slots")
    if (direct_summary["source"] != compiled_summary["source"]
            or direct_summary["source"]["transcript_source"] != "asr"):
        raise ValueError("The two benchmark arms must use the same planner, ASR and device with audio input")
    changes = [compiled[key] - direct[key] for key in ids]
    summary = {
        "dataset": "TRAIN",
        "samples": len(ids),
        "direct": direct_summary,
        "compiled": compiled_summary,
        "delta_proxy_points": 100 * average(changes),
        "improved_samples": sum(delta > 0 for delta in changes),
        "regressed_samples": sum(delta < 0 for delta in changes),
        "competition_score": "NOT MEASURED",
        "reference_evaluator_parity": "UNVERIFIED",
    }
    write_json(args.output, {**summary, "paired_samples": [
        {"id": sample_id, "direct_score": direct[sample_id], "compiled_score": compiled[sample_id]}
        for sample_id in ids]})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
