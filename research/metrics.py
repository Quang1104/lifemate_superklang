"""Diagnostic metrics only: NOT the official LifeMate scorer."""
from __future__ import annotations
from collections import Counter
import json
import unicodedata

from research.data_audit import call_name


def edit_distance(reference, hypothesis) -> int:
    previous = list(range(len(hypothesis) + 1))
    for index, token in enumerate(reference, 1):
        current = [index]
        for position, candidate in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[position] + 1,
                               previous[position - 1] + (token != candidate)))
        previous = current
    return previous[-1]


def asr_metrics(references: list[str], hypotheses: list[str]) -> dict:
    if len(references) != len(hypotheses):
        raise ValueError("Reference/hypothesis sample counts differ")
    word_edits = words = char_edits = chars = 0
    for ref, hyp in zip(references, hypotheses):
        ref = " ".join(unicodedata.normalize("NFC", ref).casefold().split())
        hyp = " ".join(unicodedata.normalize("NFC", hyp).casefold().split())
        word_edits += edit_distance(ref.split(), hyp.split())
        words += len(ref.split())
        char_edits += edit_distance(ref, hyp)
        chars += len(ref)
    return {"sample_count": len(references), "normalization": "NFC, casefold, collapse whitespace; preserve punctuation; CER includes spaces",
            "word_edits": word_edits, "reference_words": words,
            "character_edits": char_edits, "reference_characters": chars,
            "wer": word_edits / words if words else None,
            "cer": char_edits / chars if chars else None}


def canonical_call(call: dict) -> str:
    args = call.get("arguments", call.get("args"))
    if args is None and isinstance(call.get("function"), dict):
        args = call["function"].get("arguments")
    if isinstance(args, str):
        args = json.loads(args)
    if not isinstance(args, dict):
        raise ValueError("Tool arguments must be an object")
    return json.dumps([call_name(call), args], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def exact_call_metrics(gold: list[dict], predicted: list[dict]) -> dict:
    expected = Counter(canonical_call(call) for call in gold)
    actual = Counter(canonical_call(call) for call in predicted)
    matched = sum((expected & actual).values())
    precision = matched / len(predicted) if predicted else float(not gold)
    recall = matched / len(gold) if gold else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"metric": "diagnostic_exact_call_multiset_f1", "official_competition_score": "NOT MEASURED",
            "gold_calls": len(gold), "predicted_calls": len(predicted), "matched_calls": matched,
            "precision": precision, "recall": recall, "f1": f1,
            "limitations": "No fuzzy arguments, state hashes, order, invalid-call or timeout policy. Do not call this competition score."}
