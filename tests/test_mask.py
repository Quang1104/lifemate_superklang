import json

from scripts.build_sft import mask_tool_names


def item(sample_id):
    tool = lambda name: {"type": "function", "function": {"name": name, "description": "d", "parameters": {}}}
    return {"id": sample_id, "target_start": 1,
            "tools": [tool("alarm.create"), tool("alarm.create_recurring"), tool("sys.ask_missing_info")],
            "messages": [{"role": "system", "content": 'Ví dụ: "dậy" → alarm.create{"time":"6:00"}'},
                         {"role": "assistant", "tool_calls": [{"type": "function", "function": {
                             "name": "alarm.create_recurring", "arguments": {}}}]},
                         {"role": "assistant", "tool_calls": [{"type": "function", "function": {
                             "name": "sys.ask_missing_info", "arguments": {"tool_name": "alarm.create"}}}]}]}


def test_fraction_zero_and_one():
    assert mask_tool_names(item("t1"), 0.0) == item("t1")
    masked = mask_tool_names(item("t1"), 1.0)
    text = json.dumps(masked, ensure_ascii=False)
    assert "alarm." not in text and "sys.ask_missing_info" in text  # sys tools keep their names
    names = [tool["function"]["name"] for tool in masked["tools"]]
    assert len(set(names)) == 3 and all(n.startswith("fn_") for n in names[:2])
    # the same tool gets the same opaque name everywhere in the sample
    assert masked["messages"][1]["tool_calls"][0]["function"]["name"] == names[1]
    assert masked["messages"][2]["tool_calls"][0]["function"]["arguments"]["tool_name"] == names[0]
    assert names[0] in masked["messages"][0]["content"]


def test_selection_is_stable_and_close_to_fraction():
    picked = [mask_tool_names(item(f"train_{i:04d}"), 0.3) != item(f"train_{i:04d}") for i in range(1000)]
    assert picked == [mask_tool_names(item(f"train_{i:04d}"), 0.3) != item(f"train_{i:04d}") for i in range(1000)]
    assert 250 <= sum(picked) <= 350
