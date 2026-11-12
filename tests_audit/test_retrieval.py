from research.retrieval_benchmark import candidate_sets, dependency_closure, rank_tools

TOOLS = {
    "a.read": {"name": "a.read", "kind": "read", "description": "Find things", "returns": "things[]", "parameters": {}},
    "a.write": {"name": "a.write", "kind": "write", "description": "Change thing", "parameters": {"properties": {"id": {"x-ref": "a.things"}}}},
    "b.hidden": {"name": "b.hidden", "kind": "read", "description": "Change thing", "parameters": {}},
    "sys.ask_missing_info": {"name": "sys.ask_missing_info", "kind": "meta", "parameters": {}},
}
INSTALLED = {"a": {"tools": ["a.read", "a.write", "b.hidden"]}}


def test_installed_filter_and_xref_dependency():
    ranking = rank_tools("Change thing", INSTALLED, TOOLS)
    assert "b.hidden" not in ranking
    assert "sys.ask_missing_info" in ranking
    closure = dependency_closure(["a.write"], ranking, TOOLS)
    assert set(closure) == {"a.read", "a.write"}


def test_no_label_input_or_label_effect():
    # Ranking API takes transcript/apps/schema only; poisoned labels cannot enter.
    base = candidate_sets("Change thing", INSTALLED, TOOLS)
    sample = {"audio_text": "Change thing", "context": {"installed_apps": INSTALLED}, "label": {"tool_calls": [{"name": "b.hidden"}]}}
    assert base == candidate_sets(sample["audio_text"], sample["context"]["installed_apps"], TOOLS)
    for names in base.values():
        assert "b.hidden" not in names
