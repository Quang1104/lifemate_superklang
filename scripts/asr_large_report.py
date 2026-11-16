"""Read official smoke output time (milliseconds), epoch WER, and the local proxy score.

Never edits official logs. Missing official rows or incomplete timing cannot pass the
numeric candidate gates; the user still judges a clear WER improvement and target hardware.
"""

import argparse
import json
import math
from pathlib import Path


def latency_summary(rows: list[dict]) -> dict:
    times = []
    for row in rows:
        value = row.get("time")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Every official record must have a finite nonnegative time in milliseconds")
        times.append(value)
    if not times:
        return {"count": 0, "unit": "ms", "quantiles": "nearest rank",
                "p50": None, "p95": None, "max": None, "over_30000_ms": 0}
    times.sort()
    return {"count": len(times), "unit": "ms", "quantiles": "nearest rank",
            "p50": times[math.ceil(0.50 * len(times)) - 1],
            "p95": times[math.ceil(0.95 * len(times)) - 1], "max": times[-1],
            "over_30000_ms": sum(t > 30_000 for t in times)}


def build_report(rows, train_report, expected_ids, scored, training_log) -> dict:
    latency = latency_summary(rows)
    epochs = []
    for line in training_log.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue  # warnings/progress text can be interleaved with training JSON
        if isinstance(item, dict) and "epoch" in item and "wer" in item:
            value = item["wer"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid epoch validation WER")
            epochs.append(item)
    tuned = [epoch["wer"] for epoch in epochs if epoch["epoch"] > 0]
    if not tuned:
        raise ValueError("No fine-tuned epoch WER found")
    ids = [row["id"] for row in rows]
    complete = len(ids) == len(expected_ids) and len(set(ids)) == len(ids) and set(ids) == set(expected_ids)
    samples = train_report["samples"]
    detail_ids = [row["id"] for row in samples]
    details_complete = (len(detail_ids) == len(expected_ids) and len(set(detail_ids)) == len(detail_ids)
                        and set(detail_ids) == set(expected_ids))
    all_latency = latency_summary(samples)  # includes failed/omitted samples in the TRAIN report
    score = scored["local_proxy_0_to_100"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError("Invalid local proxy score")
    wer = min(tuned)
    gates = {"smoke_has_40_samples": len(expected_ids) == 40, "wer_below_9_9_percent": wer < 0.099, "smoke_at_least_61_29": score >= 61.29,
             "all_smoke_rows_exported": complete, "all_sample_timings_present": details_complete,
             "time_scale_is_1": train_report.get("time_scale") == 1,
             "all_samples_within_30s": (all_latency["max"] is not None and latency["max"] is not None
                                       and all_latency["max"] <= 30_000 and latency["max"] <= 30_000)}
    return {"validation_epochs": epochs, "best_tuned_validation_wer": wer,
            "smoke_local_proxy_0_to_100": score, "official_time": latency, "all_sample_time": all_latency,
            "expected_samples": len(expected_ids), "statuses": train_report.get("statuses"),
            "model_load_ms": train_report.get("model_load_ms"), "numeric_gates": gates,
            "candidate_passes_numeric_gates": all(gates.values()),
            "acceptance": "Keep large only if validation WER is clearly below 9.9% and smoke40 >= 56.29 + 5 = 61.29; "
                          "also require every sample <= 30 s on confirmed scoring hardware. Numeric WER alone does not judge clearly.",
            "competition_score": "NOT MEASURED"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("predictions", "train-report", "scored", "manifest", "training-log", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = build_report(rows, read(args.train_report), read(args.manifest)["validation_ids"],
                          read(args.scored), args.training_log.read_text(encoding="utf-8"))
    text = json.dumps(result, indent=2) + "\n"
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
