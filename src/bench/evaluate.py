"""Benchmark a planner (+ optional ASR) on held-out TRAIN IDs via the official ToolEnv and local scorer.

Resumable: predictions are appended per sample, finished IDs are skipped on rerun.
"""

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import re
import subprocess
import sys
from time import monotonic

from research.published_scorer import extract_tools, read_jsonl, score_records

from ..agent.runner import Limits, run_sample
from ..inference.official import OfficialKit
from ..inference.pipeline import _sample_from_row

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "research" / "validation_split.json"
SMOKE = ROOT / "research" / "smoke_split.json"
_FAILED = {"exception", "tool_timeout", "environment_error", "environment_timeout"}


def select_ids(manifest=SMOKE, limit=None, offset=0) -> list[str]:
    ids = sorted(json.loads(Path(manifest).read_text(encoding="utf-8"))["validation_ids"])
    return ids[offset:None if limit is None else offset + limit]


def ensure_gold(data_root, output, manifest=VALIDATION) -> Path:
    """Gold calls + real state hashes for all 309 held-out IDs (separate process, ~10 s on CPU)."""
    output = Path(output)
    if not output.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "replay_train_gold.py"), "--data-root",
                        str(data_root), "--manifest", str(manifest), "--output", str(output)], check=True)
    return output


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = re.findall(r"\w+", reference.lower()), re.findall(r"\w+", hypothesis.lower())
    row = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, 1):
        diagonal, row[0] = row[0], i
        for j, guess in enumerate(hyp, 1):
            diagonal, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, diagonal + (word != guess))
    return row[-1] / max(len(ref), 1)


def _run_one(kit, row, planner, transcribe, limits: Limits) -> dict:
    started = monotonic()
    opened, record = [], {"id": row["id"], "tool_calls": [], "state_hash": None}
    stats = getattr(planner, "stats", None)
    first = len(stats) if stats is not None else 0

    def open_environment(sample):
        opened.append(kit.open_environment(sample))  # exactly one ToolEnv per sample
        return opened[0]

    try:
        sample = _sample_from_row(kit, row, transcribe, started + limits.sample_seconds, control_noop=False)
        if transcribe is not None:
            record.update(transcript=sample.user_request, wer=round(wer(row["audio_text"], sample.user_request), 4))
        left = max(limits.sample_seconds - (monotonic() - started), 1e-3)
        result = run_sample(sample, kit.registry, planner, open_environment,
                            limits=replace(limits, sample_seconds=left))
        record.update(run_status=result.status, steps=result.steps, rejections=result.rejections)
        if opened and result.status != "tool_timeout":
            record["tool_calls"], record["state_hash"] = opened[0].read_output()
    except Exception as exc:  # keep benchmarking; this sample scores zero
        record.update(run_status="exception", error=f"{type(exc).__name__}: {exc}"[:500])
    record["time"] = int((monotonic() - started) * 1000)
    if record["run_status"] in _FAILED or record["state_hash"] is None:
        record["status"] = "execution_failed"
    elif record["run_status"].endswith("timeout"):
        record["status"] = "timed_out"
    if stats is not None:
        record["planner_steps"] = stats[first:]
    return record


def run(data_root, ids, gold_path, planner, out_dir, *, transcribe=None,
        limits: Limits = Limits(), verbose: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    predictions = out / "predictions.jsonl"
    done = {r["id"] for r in read_jsonl(predictions)} if predictions.exists() else set()
    todo = set(ids) - done
    if todo:
        kit = OfficialKit(data_root, "train")
        with predictions.open("a", encoding="utf-8") as sink:
            for row in kit.rows(todo):
                record = _run_one(kit, row, planner, transcribe, limits)
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                if verbose:
                    print(record["id"], record["run_status"], f'{record["time"] / 1000:.1f}s',
                          len(record["tool_calls"]), "calls", flush=True)
    return summarize(data_root, gold_path, out, ids)


def classify(detail: dict, gold: dict, prediction: dict) -> str:
    if detail.get("failure_override"):
        return detail["failure_override"]
    if detail["score"] == 1:
        return "correct"
    if detail["gold_empty"]:
        return "acted_on_empty_gold"
    if not prediction["tool_calls"]:
        return "no_calls"
    names = lambda calls: Counter(call["name"] for call in calls)
    if names(gold["tool_calls"]) != names(prediction["tool_calls"]):
        return "tool_selection"
    if detail["matched"] < len(gold["tool_calls"]):
        return "arguments"
    return "state_only"


def _scorable(record: dict, timed: bool) -> dict:
    row = {"id": record["id"], "tool_calls": record["tool_calls"], "state_hash": record["state_hash"] or ""}
    if record.get("status") == "execution_failed" or (timed and record.get("status")):
        row["status"] = record["status"]
    if timed:
        row["time"] = record["time"]
    return row


def summarize(data_root, gold_path, out_dir, ids=None) -> dict:
    out = Path(out_dir)
    predictions = {r["id"]: r for r in read_jsonl(out / "predictions.jsonl")}
    if ids is not None:
        predictions = {i: predictions[i] for i in ids if i in predictions}
    gold = {g["id"]: g for g in read_jsonl(Path(gold_path)) if g["id"] in predictions}
    catalog = json.loads((Path(data_root) / "train" / "train_tools_catalog.json").read_text(encoding="utf-8-sig"))
    tools = extract_tools(catalog)
    official = score_records(list(gold.values()), [_scorable(p, True) for p in predictions.values()], tools)
    untimed = score_records(list(gold.values()), [_scorable(p, False) for p in predictions.values()], tools)
    classes, missing, extra = Counter(), Counter(), Counter()
    for detail in untimed["samples"]:
        g, p = gold[detail["id"]], predictions[detail["id"]]
        classes[classify(detail, g, p)] += 1
        gold_names, pred_names = Counter(c["name"] for c in g["tool_calls"]), Counter(c["name"] for c in p["tool_calls"])
        missing.update(gold_names - pred_names)
        extra.update(pred_names - gold_names)
    times = sorted(p["time"] for p in predictions.values())
    wers = [p["wer"] for p in predictions.values() if "wer" in p]
    prompts = [s["prompt_tokens"] for p in predictions.values() for s in p.get("planner_steps", ())]
    summary = {
        "samples": len(predictions),
        "score_official": round(official["normalized_score"], 4),
        "score_untimed": round(untimed["normalized_score"], 4),
        "state_hash_accuracy_nonempty": untimed["state_hash_accuracy_nonempty"],
        "macro_call_f1": round(untimed["macro_call_f1"], 4),
        "error_classes": dict(classes.most_common()),
        "run_status": dict(Counter(p.get("run_status") for p in predictions.values())),
        "missing_tools_top": dict(missing.most_common(10)),
        "extra_tools_top": dict(extra.most_common(10)),
        "latency_s": {"mean": round(sum(times) / len(times) / 1000, 2), "p95": times[int(0.95 * (len(times) - 1))] / 1000,
                      "max": times[-1] / 1000, "over_30s": sum(t > 30_000 for t in times)},
        "mean_rejections": round(sum(p.get("rejections", 0) for p in predictions.values()) / len(predictions), 2),
        "max_prompt_tokens": max(prompts, default=None),
        "mean_wer": round(sum(wers) / len(wers), 4) if wers else None,
        "scorer_parity": "local published-metric reimplementation, UNVERIFIED vs organizer",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "scores.json").write_text(json.dumps({"official": official, "untimed": untimed}, ensure_ascii=False),
                                     encoding="utf-8")
    return summary


def compare(results_root) -> list[dict]:
    """One row per run directory that has a summary.json (for checkpoint comparison)."""
    rows = []
    for path in sorted(Path(results_root).glob("*/summary.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"run": path.parent.name, "n": s["samples"], "official": s["score_official"],
                     "untimed": s["score_untimed"], "call_f1": s["macro_call_f1"],
                     "p95_s": s["latency_s"]["p95"], "over_30s": s["latency_s"]["over_30s"],
                     "wer": s["mean_wer"], **{f"err_{k}": v for k, v in s["error_classes"].items()}})
    return rows
