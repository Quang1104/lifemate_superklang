"""Published metric tests on synthetic calls/hashes, never competition scores."""
import pytest
from research.published_scorer import arguments_match, maximum_matches, score_sample, score_records, token_f1

TOOLS = {"notes.create": {"parameters": {"type": "object", "properties": {
    "title": {"type": "string", "x-match": "fuzzy"}, "count": {"type": "integer"}}}}}


def call(title, count=1):
    return {"name": "notes.create", "arguments": {"title": title, "count": count}}


def test_fuzzy_threshold_strict_and_unicode():
    assert token_f1("a b c", "a b c d e f g") == 0.6
    assert not arguments_match("a b c", "a b c d e f g", {"x-match": "fuzzy"})
    assert token_f1("CAFE\u0301", "caf\u00e9") == 1


def test_maximum_matching_not_greedy():
    # First prediction can match both golds, second can match only first.
    gold = [call("a b"), call("a b c d")]
    predicted = [call("a b c"), call("a")]
    assert maximum_matches(gold, predicted, TOOLS) == 2
    assert maximum_matches(gold, list(reversed(predicted)), TOOLS) == 2


def test_hash_gate_empty_exception_and_duplicate_penalty():
    gold = {"tool_calls": [call("a")], "state_hash": "synthetic-hash"}
    pred = {"tool_calls": [call("a"), call("a")], "state_hash": "synthetic-hash"}
    assert score_sample(gold, pred, TOOLS)["score"] == pytest.approx(2 / 3)
    pred["state_hash"] = "different-synthetic-hash"
    assert score_sample(gold, pred, TOOLS)["score"] == 0
    assert score_sample({"tool_calls": []}, {"tool_calls": []}, TOOLS)["score"] == 1
    with pytest.raises(ValueError, match="state_hash"):
        score_sample(gold, {"tool_calls": []}, TOOLS)


def test_exact_args_invalid_calls_and_coverage():
    gold = {"id": "s", "tool_calls": [call("a")], "state_hash": "synthetic"}
    pred = {"id": "s", "tool_calls": [call("a", True)], "state_hash": "synthetic"}
    assert score_records([gold], [pred], TOOLS)["normalized_score"] == 0
    with pytest.raises(ValueError, match="coverage"):
        score_records([gold], [], TOOLS)


def test_timeout_boundary_execution_failure_and_malformed_call():
    empty = {"tool_calls": []}
    assert score_sample(empty, {"tool_calls": [], "time": 30000}, TOOLS)["score"] == 1
    assert score_sample(empty, {"tool_calls": [], "time": 30001}, TOOLS)["score"] == 0
    assert score_sample(empty, {"tool_calls": [], "status": "execution_failed"}, TOOLS)["score"] == 0
    assert score_sample(empty, {"tool_calls": [], "evaluation_status": "timed_out"}, TOOLS)["score"] == 0
    gold = {"tool_calls": [call("a")], "state_hash": "synthetic"}
    pred = {"tool_calls": [None, {}, {"name": "unknown", "arguments": {}}], "state_hash": "synthetic"}
    assert score_sample(gold, pred, TOOLS)["unmatched"] == 3
