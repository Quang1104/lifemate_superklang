"""Build src/agent/data/fewshot_plans.jsonl from verified gold plans (TRAIN only, never the 309 validation IDs).

Input: outputs/gold_plans.jsonl from scripts/gold_to_plan.py. Only rows with verified=true are kept.
"""

import argparse
import json
from pathlib import Path
from time import monotonic

from src.agent.fewshot import POOL
from src.inference.official import OfficialKit
from src.inference.pipeline import _sample_from_row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--plans", type=Path, default=Path("outputs/gold_plans.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("research/validation_split.json"))
    parser.add_argument("--output", type=Path, default=POOL)
    args = parser.parse_args()
    validation = set(json.loads(args.validation.read_text(encoding="utf-8"))["validation_ids"])
    if len(validation) != 309:
        raise ValueError("Expected the 309-ID TRAIN validation manifest")
    with args.plans.open(encoding="utf-8") as lines:
        verified = [row for row in map(json.loads, lines) if row["verified"]]
    if any(row["id"] in validation for row in verified):
        raise ValueError("A validation ID reached the few-shot pool")
    kit = OfficialKit(args.data_root, "train")
    apps = {row["id"]: sorted(_sample_from_row(kit, row, None, monotonic() + 60, control_noop=False).installed_apps)
            for row in kit.rows({row["id"] for row in verified})}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as sink:
        for row in sorted(verified, key=lambda item: item["id"]):
            sink.write(json.dumps({"id": row["id"], "transcript": row["transcript"], "apps": apps[row["id"]],
                                   "plan": row["plan"]}, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"rows": len(verified), "bytes": args.output.stat().st_size, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
