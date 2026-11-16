import json

from research.leave_app_out_split import apps_used, make_split
from scripts.gold_to_plan import Converter, find_span, spans, where_for
from src.agent.compiled_plan import _compile
from src.agent.planner import PlannerInput
from src.inference.types import PriorToolResult, Sample


CONTEXT = {"current_time": "2026-11-13T10:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh",
           "user_profile": {"city": "TP.HCM"}}
STRING = {"type": "string"}
TOOLS = {
    "smart_home.get_status": {"name": "smart_home.get_status", "description": "Xem trạng thái thiết bị",
                              "writes": False, "parameters": {"type": "object", "properties": {},
                                                              "additionalProperties": False},
                              "evidence_paths": [{"namespace": "smart_home.devices",
                                                  "path": ["devices", "*", "id"], "scope_argument": None}]},
    "smart_home.light_set": {"name": "smart_home.light_set", "description": "Bật tắt đèn", "writes": True,
                             "parameters": {"type": "object", "additionalProperties": False,
                                            "required": ["device_id"],
                                            "properties": {"device_id": {"type": "string",
                                                                         "x-ref": "smart_home.devices"},
                                                           "power": {"type": "string", "enum": ["on", "off"]}}},
                             "evidence_paths": []},
    "reminders.create": {"name": "reminders.create", "description": "Tạo nhắc nhở", "writes": True,
                         "parameters": {"type": "object", "additionalProperties": False,
                                        "required": ["content", "time"],
                                        "properties": {"content": STRING, "time": STRING}},
                         "evidence_paths": []},
}
DEVICES = {"ok": True, "devices": [
    {"id": "light_living", "name": "đèn phòng khách", "type": "light", "floor": 1},
    {"id": "light_bed", "name": "đèn phòng ngủ", "type": "light", "floor": 2},
]}


def sample(text, **extra):
    return Sample("train_x", text, frozenset({"smart_home", "reminders", "sys"}), context=CONTEXT, **extra)


def test_spans_strip_edge_punctuation():
    assert "mai" in spans("Nhắc tôi mai.")
    assert "mai." not in spans("Nhắc tôi mai.")


def test_find_span_prefers_longest_exact_conversion():
    text = "Nhắc tôi gọi mẹ lúc 3 giờ chiều mai nhé"
    assert find_span("2026-11-14T15:00:00+07:00", "datetime", text, CONTEXT) == "lúc 3 giờ chiều mai"
    assert find_span(500000, "money", "trả 500 nghìn", CONTEXT) == "500 nghìn"
    assert find_span("2026-11-14T16:00:00+07:00", "datetime", text, CONTEXT) is None


def test_where_for_grounds_text_then_integer_fields():
    rows = DEVICES["devices"]
    assert where_for(rows, rows[1], "tắt đèn phòng ngủ", CONTEXT) == {"name": {"text": "đèn phòng ngủ"}}
    assert where_for(rows, rows[1], "tắt đèn tầng 2", CONTEXT) == {"floor": {"text": "2", "as": "integer"}}
    assert where_for(rows, rows[1], "tắt đèn", CONTEXT) is None  # type=light is not unique


def test_converter_builds_selector_enum_and_span_plan_that_compiles():
    request = "Tắt đèn phòng ngủ rồi nhắc tôi gọi mẹ lúc 3 giờ chiều mai"
    converter = Converter(sample(request), TOOLS)
    calls = [{"name": "smart_home.get_status", "arguments": {}},
             {"name": "smart_home.light_set", "arguments": {"device_id": "light_bed", "power": "off"}},
             {"name": "reminders.create", "arguments": {"content": "gọi mẹ", "time": "2026-11-14T15:00:00+07:00"}}]
    plan, reasons = converter.convert(calls, [DEVICES, {"ok": True}, {"ok": True}])
    assert reasons == []
    light, remind = plan["steps"][1]["args"], plan["steps"][2]["args"]
    assert light == {"device_id": {"result": "s1", "path": ["devices"], "where": {"name": {"text": "đèn phòng ngủ"}},
                                   "field": ["id"]}, "power": {"value": "off"}}
    assert remind == {"content": {"text": "gọi mẹ"}, "time": {"text": "lúc 3 giờ chiều mai", "as": "datetime"}}
    assert all(step["anchor"] for step in plan["steps"])
    tools = [{**tool, "name": name} for name, tool in TOOLS.items()]
    steps = _compile(json.dumps(plan, ensure_ascii=False), PlannerInput(sample(request), tools, ()),
                     TOOLS, [], 7)
    assert [step.tool for step in steps] == [call["name"] for call in calls]


def test_converter_groups_unexpressible_reasons():
    converter = Converter(sample("nhắc tôi Gọi Mẹ"), TOOLS)
    _, reasons = converter.convert([{"name": "reminders.create",
                                     "arguments": {"content": "gọi mẹ", "time": "2026-11-14T15:00:00+07:00"}}],
                                   [{"ok": True}])
    assert reasons == ["text:case_differs", "datetime:no_span_or_field"]
    history = sample("tắt cái đó", prior_tool_results=(PriorToolResult("smart_home.get_status", {}, DEVICES),))
    _, reasons = Converter(history, TOOLS).convert(
        [{"name": "smart_home.light_set", "arguments": {"device_id": "light_bed", "power": "off"}}], [{"ok": True}])
    assert reasons == ["xref:in_history_list"]


def test_shift_from_observed_time():
    converter = Converter(sample("dời muộn 1 tiếng"), TOOLS)
    converter.results.append(("s1", "calendar.list_events", {"ok": True, "item": {"start": "2026-11-14T08:00:00+07:00"}}))
    assert converter.source("2026-11-14T09:00:00+07:00", STRING, "calendar.update_event") == {
        "shift": {"result": "s1", "path": ["item", "start"]}, "by": {"text": "1 tiếng", "as": "duration"},
        "dir": "later"}


def row(sample_id, *names, history=()):
    return {"id": sample_id, "label": {"tool_calls": [{"name": name, "arguments": {}} for name in names]},
            "history": list(history)}


def test_leave_app_out_keeps_held_apps_out_of_the_pool():
    rows = [row(f"t{i}", "news.search") for i in range(6)] + [row(f"a{i}", "alarm.create") for i in range(6)]
    rows.append(row("mix", "alarm.list", history=[{"role": "assistant", "tool_calls": [{"name": "news.search"}]}]))
    rows.append({"id": "ask", "history": [], "label": {"tool_calls": [
        {"name": "sys.ask_missing_info", "arguments": {"tool_name": "news.search", "params": ["query"]}}]}})
    assert apps_used(rows[-1]) == {"news"} and apps_used(rows[-2]) == {"alarm", "news"}
    split = make_split(rows, {"a0"}, {"news": 2, "alarm": 4}, fraction=0.6, seed=1)
    held = set(split["holdout_apps"])
    assert held and "a0" not in split["train_ids"] + split["validation_ids"]
    pool = {item["id"]: item for item in rows}
    assert not any(apps_used(pool[sample_id]) & held for sample_id in split["train_ids"])
    assert all(apps_used(pool[sample_id]) & held for sample_id in split["validation_ids"])
