"""Score one TRAIN run against replayed gold and dump per-sample details for error analysis."""

import argparse
import json

from research.data_audit import extract_tools, read_jsonl
from scripts.compare_train import evaluate, index, read_json, train_ids, write_json
from src.inference.paths import safe_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "gold", "catalog", "predictions", "report", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--train-jsonl", help="TRAIN rows, to include the released transcript in the dump")
    args = parser.parse_args()
    ids = train_ids(args.manifest)
    gold_rows = index(read_jsonl(safe_path(args.gold)))
    gold = {sample_id: gold_rows[sample_id] for sample_id in ids}
    scores, summary = evaluate(gold, args.predictions, args.report, extract_tools(read_json(args.catalog)))
    predictions = index(read_jsonl(safe_path(args.predictions)))
    report = read_json(args.report)
    status = {row["id"]: row for row in report["samples"]}
    text = ({row["id"]: row.get("audio_text") for row in read_jsonl(safe_path(args.train_jsonl))}
            if args.train_jsonl else {})
    samples = [{"id": sample_id, "score": scores[sample_id], "status": status[sample_id]["status"],
                "time_ms": status[sample_id]["time"], "transcript": text.get(sample_id),
                "gold": gold[sample_id]["tool_calls"],
                "predicted": predictions.get(sample_id, {}).get("tool_calls"),
                "state_match": predictions.get(sample_id, {}).get("state_hash") == gold[sample_id].get("state_hash")}
               for sample_id in ids]
    extra = {key: report.get(key) for key in ("statuses", "grammar", "mean_sample_time_ms", "timing") if key in report}
    write_json(args.output, {**summary, **extra, "samples": samples})
    print(json.dumps({**summary, **extra}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
