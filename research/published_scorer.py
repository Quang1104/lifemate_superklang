"""Published LifeMate metric, with explicit unresolved tokenizer convention.

This is a local reimplementation, not the organizer's reference evaluator.
Source: https://api-challenge.zalo.ai/fe/api/config-detail?name=evaluationMetric&title=1
Snapshot: docs/sources/evaluationMetric.json. Tokenization is whitespace because
the publication specifies lowercasing/NFC/token-F1 but does not define tokens.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import unicodedata

from research.data_audit import call_name, extract_tools, read_jsonl, safe_path


def token_f1(left: str, right: str) -> float:
    def tokens(value):
        return unicodedata.normalize("NFC", value.lower()).split()
    left_tokens, right_tokens = tokens(left), tokens(right)
    if not left_tokens or not right_tokens:
        return float(left_tokens == right_tokens)
    common = sum((Counter(left_tokens) & Counter(right_tokens)).values())
    return 2 * common / (len(left_tokens) + len(right_tokens))


def strict_equal(left, right) -> bool:
    # JSON numeric values 1 and 1.0 compare equal; booleans must not equal numbers.
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(strict_equal(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(strict_equal(a, b) for a, b in zip(left, right))
    return left == right


def arguments_match(gold, predicted, schema: dict) -> bool:
    if schema.get("x-match") == "fuzzy":
        return isinstance(gold, str) and isinstance(predicted, str) and token_f1(gold, predicted) > 0.6
    if isinstance(gold, dict) and isinstance(predicted, dict):
        if gold.keys() != predicted.keys():
            return False
        properties = schema.get("properties", {})
        return all(arguments_match(gold[k], predicted[k], properties.get(k, {})) for k in gold)
    if isinstance(gold, list) and isinstance(predicted, list):
        return len(gold) == len(predicted) and all(arguments_match(a, b, schema.get("items", {}))
                                                 for a, b in zip(gold, predicted))
    return strict_equal(gold, predicted)


def call_matches(gold: dict, predicted: dict, tools: dict[str, dict]) -> bool:
    name = call_name(gold)
    if name not in tools:
        raise ValueError(f"Gold tool missing from catalog: {name}")
    try:
        if name != call_name(predicted):
            return False
        gold_args, predicted_args = gold.get("arguments"), predicted.get("arguments")
        if not isinstance(gold_args, dict) or not isinstance(predicted_args, dict):
            return False
        return arguments_match(gold_args, predicted_args, tools[name].get("parameters", {}))
    except (TypeError, KeyError, ValueError, AttributeError):
        return False


def maximum_matches(gold: list[dict], predicted: list[dict], tools: dict[str, dict]) -> int:
    edges = [[g for g, expected in enumerate(gold) if call_matches(expected, call, tools)]
             for call in predicted]
    owners: dict[int, int] = {}

    def augment(pred_index: int, visited: set[int]) -> bool:
        for gold_index in edges[pred_index]:
            if gold_index in visited:
                continue
            visited.add(gold_index)
            if gold_index not in owners or augment(owners[gold_index], visited):
                owners[gold_index] = pred_index
                return True
        return False

    return sum(augment(index, set()) for index in range(len(predicted)))


def score_sample(gold: dict, prediction: dict, tools: dict[str, dict]) -> dict:
    gold_calls, predicted_calls = gold["tool_calls"], prediction["tool_calls"]
    if not isinstance(gold_calls, list) or not isinstance(predicted_calls, list):
        raise ValueError("tool_calls must be lists")
    elapsed_ms = prediction.get("time")
    status = prediction.get("evaluation_status", prediction.get("status"))
    if status in ("timed_out", "execution_failed") or (
        isinstance(elapsed_ms, (int, float)) and not isinstance(elapsed_ms, bool) and elapsed_ms > 30000
    ):
        return {"score": 0.0, "gold_empty": not gold_calls, "matched": 0,
                "unmatched": len(predicted_calls), "state_match": None, "call_f1": 0.0,
                "failure_override": "timeout" if status == "timed_out" or
                (isinstance(elapsed_ms, (int, float)) and elapsed_ms > 30000) else "execution_failed"}
    if not gold_calls:
        return {"score": float(not predicted_calls), "gold_empty": True,
                "matched": 0, "unmatched": len(predicted_calls), "state_match": None,
                "call_f1": float(not predicted_calls)}
    for record in (gold, prediction):
        if not isinstance(record.get("state_hash"), str) or not record["state_hash"]:
            raise ValueError("Nonempty-call scoring requires real ToolEnv state_hash values")
    matches = maximum_matches(gold_calls, predicted_calls, tools)
    unmatched = len(predicted_calls) - matches
    precision = matches / len(predicted_calls) if predicted_calls else 0.0
    recall = matches / len(gold_calls)
    f1 = 2 * precision * recall / (precision + recall) if matches else 0.0
    state_match = gold["state_hash"] == prediction["state_hash"]
    return {"score": f1 if state_match else 0.0, "gold_empty": False,
            "matched": matches, "unmatched": unmatched, "state_match": state_match, "call_f1": f1}


def score_records(gold: list[dict], predictions: list[dict], tools: dict[str, dict]) -> dict:
    def index(records):
        keyed = {str(r["id"]): r for r in records}
        if len(keyed) != len(records):
            raise ValueError("Duplicate sample IDs")
        return keyed
    expected, actual = index(gold), index(predictions)
    if expected.keys() != actual.keys():
        raise ValueError("Prediction/gold ID coverage mismatch")
    if not expected:
        raise ValueError("Cannot score empty dataset")
    details = [{"id": sample_id, **score_sample(expected[sample_id], actual[sample_id], tools)}
               for sample_id in sorted(expected)]
    mean = sum(r["score"] for r in details) / len(details)
    known_state = [r for r in details if r["state_match"] is not None]
    return {"evaluator": "local_published_metric_v1", "reference_evaluator_parity": "UNVERIFIED",
            "sample_count": len(details), "normalized_score": mean, "published_scale_score": 100 * mean,
            "state_hash_accuracy_nonempty": sum(r["state_match"] for r in known_state) / len(known_state) if known_state else None,
            "macro_call_f1": sum(r["call_f1"] for r in details) / len(details),
            "fuzzy_tokenization": "NFC + lower + whitespace split; strict >0.6; empty/empty similarity=1 (convention not specified)",
            "limitations": "Tokenizer, empty fuzzy strings and number type equivalence need organizer-reference confirmation. This is not a leaderboard result.",
            "samples": details}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="Separate TRAIN replay JSONL with real state hashes")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    tools = extract_tools(json.loads(safe_path(args.catalog).read_text(encoding="utf-8-sig")))
    report = score_records(read_jsonl(args.gold), read_jsonl(args.predictions), tools)
    report.update(dataset=args.dataset, configuration=args.config, checkpoint=args.checkpoint,
                  predictions=str(args.predictions), gold=str(args.gold))
    target = safe_path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("sample_count", "normalized_score", "published_scale_score", "reference_evaluator_parity")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
