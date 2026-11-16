"""Recompute CPU-only TRAIN failure groups from the reviewed Markdown table.

Only explicit allowlisted files are opened; TRAIN contributes IDs/current_time.
No ToolEnv, runtime states, model weights, or test labels are accessed.
"""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
TRAIN = Path("C:/Users/ADMIN/OneDrive - VNU-HCMUS/Documents/XuanQuang/datalife/train+public_test/train/train.jsonl")
CATEGORIES = (
    "date arithmetic",
    "missing ask_missing_info",
    "action on chit-chat/empty gold",
    "zero calls",
    "early stop in multi-step",
    "wrong argument value",
    "ASR word error",
    "other",
)


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main():
    scored = read_json("outputs/colab/planner_sft/smoke40_sft50_x3/native.scored.json")
    report = read_json("outputs/colab/planner_sft/smoke40_sft50_x3/native.report.json")
    leave = read_json("research/leave_app_out_split.json")
    validation = read_json("research/validation_split.json")
    markdown = (ROOT / "research/sft50_failure_analysis.md").read_text(encoding="utf-8")
    assigned = {}
    for line in markdown.splitlines():
        fields = [field.strip() for field in line.split("|")]
        if len(fields) == 6 and re.fullmatch(r"train_[0-9]+", fields[1]):
            sample_id, displayed_score, category = fields[1:4]
            assert sample_id not in assigned, sample_id
            assert category in CATEGORIES, category
            assigned[sample_id] = (category, float(displayed_score))

    samples = scored["samples"]
    ids = {sample["id"] for sample in samples}
    assert len(ids) == len(samples) == 40
    failures = {sample["id"] for sample in samples if sample["score"] < 1}
    assert set(assigned) == failures
    assert len(failures) == 34
    proxy = 100 * math.fsum(sample["score"] for sample in samples) / len(samples)
    assert math.isclose(proxy, scored["local_proxy_0_to_100"], abs_tol=1e-10)
    counts = Counter()
    deficits = defaultdict(list)
    for sample in samples:
        assert 0 <= sample["score"] <= 1
        if sample["score"] == 1:
            continue
        category, displayed_score = assigned[sample["id"]]
        assert math.isclose(displayed_score, sample["score"], abs_tol=0.00000051)
        counts[category] += 1
        deficits[category].append(1 - sample["score"])
    points = {category: 100 / len(samples) * math.fsum(deficits[category]) for category in CATEGORIES}
    lost = math.fsum(points.values())
    assert math.isclose(lost, 100 - proxy, abs_tol=1e-10)

    train_current_times = {}
    with TRAIN.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            assert record["id"] not in train_current_times
            train_current_times[record["id"]] = record["context"]["current_time"]
    train_ids = set(train_current_times)
    assert leave["source_split"] == validation["source_split"] == "train"
    validation_ids = set(validation["validation_ids"])
    assert validation_ids <= train_ids
    sft_ids = train_ids - validation_ids
    holdout_ids = set(leave["validation_ids"])
    assert len(holdout_ids) == len(leave["validation_ids"]) == 171
    assert holdout_ids <= train_ids
    contaminated = holdout_ids & sft_ids
    details = {sample["id"]: sample for sample in report["samples"]}
    zero_failures = sorted(sample_id for sample_id in failures if assigned[sample_id][0] == "zero calls")
    for sample_id in zero_failures:
        assert details[sample_id]["steps"] == 1
        assert details[sample_id]["dispatched_calls"] == details[sample_id]["rejections"] == 0
    assert failures <= train_ids

    print(f"samples={len(samples)} nonperfect={len(failures)} proxy={proxy:.6f} lost={lost:.6f}; "
          f"TRAIN={len(train_ids)} validation={len(validation_ids)} SFT={len(sft_ids)} "
          f"holdout={len(holdout_ids)} contaminated={len(contaminated)}")
    for category in sorted(CATEGORIES, key=lambda category: (-points[category], CATEGORIES.index(category))):
        print(f"{category}: count={counts[category]} lost={points[category]:.6f}")
    print("zero-call failures: " + ",".join(zero_failures) + "; each steps=1 dispatched=0 rejections=0")
    print("checks=OK")


if __name__ == "__main__":
    main()
