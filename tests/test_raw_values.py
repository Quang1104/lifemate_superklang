import json

from src.agent.compiled_plan import CompiledPlanner
from src.agent.fewshot import Examples
from src.agent.runner import run_sample
from src.inference.types import PriorToolResult, Sample
from src.tooling.evidence import EvidencePath
from src.tooling.registry import ToolRegistry, ToolSpec


CONTEXT = {"current_time": "2026-11-22T21:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh",
           "user_profile": {"city": "TP.HCM"}}
STRING = {"type": "string"}
TEXT = "Đặt báo thức 6 giờ sáng mai, rồi gọi cho Minh giúp tôi."
ALARM = {"tool": "alarm.create", "anchor": "Đặt báo thức 6 giờ sáng mai",
         "args": {"time": "2026-11-23T06:00:00+07:00", "label": "Dậy sớm", "repeat": "once"}}


def schema(fields, required=()):
    return {"type": "object", "properties": fields, "required": list(required), "additionalProperties": False}


REGISTRY = ToolRegistry([
    ToolSpec("alarm.create", "alarm", "Create an alarm",
             schema({"time": STRING, "label": {"type": "string", "x-match": "fuzzy"},
                     "repeat": {"type": "string", "enum": ["once", "daily"]}}, ("time",)), True),
    ToolSpec("contacts.search", "contacts", "Search contacts", schema({"query": STRING}, ("query",)), False,
             (EvidencePath("contacts.contacts", ("contacts", "*", "id")),)),
    ToolSpec("phone.call", "phone", "Call a contact",
             schema({"contact_id": {"type": "string", "x-ref": "contacts.contacts"}}, ("contact_id",)), True),
    ToolSpec("sms.send", "sms", "Send a message",
             schema({"contact_id": {"type": "string", "x-ref": "contacts.contacts"},
                     "content": {"type": "string", "x-match": "fuzzy"}}, ("contact_id", "content")), True),
    ToolSpec("sys.ask_missing_info", "sys", "Ask for missing information",
             schema({"tool_name": STRING, "params": {"type": "array", "items": STRING}}, ("tool_name", "params")),
             False),
])


class Env:
    def __init__(self, contacts):
        self.calls = []
        self.contacts = contacts

    def call(self, name, args):
        self.calls.append((name, args))
        if name == "contacts.search":
            return {"ok": True, "contacts": self.contacts}
        return {"ok": True}


def run(text, plans, contacts=({"id": "c_07", "name": "Minh"},), **sample_fields):
    prompts = []
    plans = iter(plans)

    def generate(system, user, *, deadline=None):
        prompts.append(json.loads(user))
        plan = next(plans)
        return plan if isinstance(plan, str) else json.dumps({"steps": plan}, ensure_ascii=False)

    env = Env(list(contacts))
    sample = Sample("raw", text, frozenset({"alarm", "contacts", "phone", "sms", "sys"}), context=CONTEXT,
                    **sample_fields)
    result = run_sample(sample, REGISTRY, CompiledPlanner(generate, fewshot=0).for_sample(), lambda _: env)
    return result, env.calls, prompts


def logged_plan(contact_id="contact_123"):
    """The pattern logged on smoke40: raw literals and an invented ID instead of typed sources."""
    return [{"id": "a", **ALARM},
            {"id": "b", "tool": "contacts.search", "anchor": "gọi cho Minh", "args": {"query": "Minh"}},
            {"id": "c", "tool": "phone.call", "anchor": "gọi cho Minh", "args": {"contact_id": contact_id}}]


def test_logged_raw_plan_compiles_and_routes_invented_id_to_the_unique_result():
    result, calls, prompts = run(TEXT, [logged_plan()])
    assert calls == [("alarm.create", {"time": "2026-11-23T06:00:00+07:00", "label": "Dậy sớm", "repeat": "once"}),
                     ("contacts.search", {"query": "Minh"}), ("phone.call", {"contact_id": "c_07"})]
    assert len(prompts) == 1 and result.reply == ""


def test_offset_less_datetime_literal_is_sent_in_canonical_form():
    plan = [{"id": "a", **ALARM, "args": {"time": "2026-11-23T06:00"}}]
    _, calls, _ = run(TEXT, [plan])
    assert calls == [("alarm.create", {"time": "2026-11-23T06:00:00+07:00"})]


def test_ambiguous_invented_id_waits_for_the_read_then_repairs_with_visible_results():
    contacts = ({"id": "c_07", "name": "Minh Anh"}, {"id": "c_08", "name": "Minh Tuấn"})
    repair = [{"id": "c", "tool": "phone.call", "anchor": "gọi cho Minh", "args": {"contact_id": "c_08"}}]
    result, calls, prompts = run(TEXT, [logged_plan(), repair], contacts)
    assert [name for name, _ in calls] == ["alarm.create", "contacts.search", "phone.call"]
    assert calls[-1] == ("phone.call", {"contact_id": "c_08"})
    assert "copy one real ID" in prompts[1]["repair"]["reason"]
    assert prompts[1]["repair"]["visible_results"]["b"]["contacts"][1]["id"] == "c_08"


def test_invented_id_is_never_dispatched_when_the_repair_invents_again():
    contacts = ({"id": "c_07", "name": "Minh Anh"}, {"id": "c_08", "name": "Minh Tuấn"})
    repair = [{"id": "c", "tool": "phone.call", "anchor": "gọi cho Minh", "args": {"contact_id": "c_99"}}]
    result, calls, _ = run(TEXT, [logged_plan(), repair], contacts)
    assert [name for name, _ in calls] == ["alarm.create", "contacts.search"]
    assert result.status == "plan_incomplete"


def test_grounded_selection_uses_the_query_among_several_records():
    contacts = ({"id": "c_07", "name": "Minh"}, {"id": "c_08", "name": "Minh Tuấn"})
    _, calls, _ = run(TEXT, [logged_plan()], contacts)
    assert calls[-1] == ("phone.call", {"contact_id": "c_07"})


def test_ungrounded_literals_stay_rejected_before_any_dispatch():
    wrong_time = [{"id": "a", **ALARM, "args": {"time": "2026-11-23T07:00:00+07:00"}}]
    no_producer = [{"id": "c", "tool": "phone.call", "anchor": "gọi cho Minh", "args": {"contact_id": "c_07"}}]
    invented_enum = [{"id": "a", **ALARM, "args": {"time": "2026-11-23T06:00:00+07:00", "repeat": "weekly"}}]
    for plan in (wrong_time, no_producer, invented_enum):
        result, calls, prompts = run(TEXT, [plan, plan])
        assert not calls and len(prompts) == 2 and result.status == "plan_invalid"


def test_free_text_may_be_literal_but_ids_and_plain_fields_may_not():
    text = "Nhắn cho Minh là tối nay anh về muộn."
    plan = [{"id": "b", "tool": "contacts.search", "anchor": "Nhắn cho Minh", "args": {"query": "minh"}},
            {"id": "s", "tool": "sms.send", "anchor": "Nhắn cho Minh",
             "args": {"contact_id": "contact_1", "content": "Anh sẽ về muộn tối nay"}}]
    _, calls, _ = run(text, [plan])
    # "minh" keeps the model's casing; the paraphrased body is free text (x-match fuzzy).
    assert calls == [("contacts.search", {"query": "minh"}),
                     ("sms.send", {"contact_id": "c_07", "content": "Anh sẽ về muộn tối nay"})]


def test_literals_from_context_and_successful_prior_results():
    prior = PriorToolResult("contacts.search", {"query": "Minh"},
                            {"ok": True, "contacts": [{"id": "c_42", "name": "Minh"}]})
    plan = [{"id": "c", "tool": "phone.call", "anchor": "gọi cho Minh", "args": {"contact_id": "c_42"}},
            {"id": "a", **ALARM, "args": {"time": "2026-11-23T06:00:00+07:00", "label": "TP.HCM"}}]
    _, calls, _ = run(TEXT, [plan], prior_tool_results=(prior,))
    assert calls == [("phone.call", {"contact_id": "c_42"}),
                     ("alarm.create", {"time": "2026-11-23T06:00:00+07:00", "label": "TP.HCM"})]


def test_negation_guard_still_blocks_raw_literals():
    text = "Đừng đặt báo thức 6 giờ sáng mai, gọi cho Minh thôi."
    blocked = [{"id": "a", "tool": "alarm.create", "anchor": "đặt báo thức",
                "args": {"time": "2026-11-23T06:00:00+07:00"}}]
    calls_only = logged_plan()[1:]
    result, calls, prompts = run(text, [blocked, calls_only])
    assert [name for name, _ in calls] == ["contacts.search", "phone.call"]
    assert len(prompts) == 2


def test_harmless_shapes_are_accepted():
    bare_list = json.dumps([{"tool": "contacts.search", "args": {"query": "Minh"}}], ensure_ascii=False)
    fenced = "```json\n" + json.dumps({"steps": [{"id": "a", "reason": "user asked", **ALARM}]},
                                      ensure_ascii=False) + "\n```"
    for raw, expected in ((bare_list, "contacts.search"), (fenced, "alarm.create")):
        _, calls, prompts = run(TEXT, [raw])
        assert [name for name, _ in calls] == [expected] and len(prompts) == 1


def test_near_verbatim_anchor_maps_to_the_request_window():
    plan = [{"id": "a", **ALARM, "anchor": "đặt một báo thức 6 giờ sáng mai"}]
    _, calls, prompts = run(TEXT, [plan])
    assert [name for name, _ in calls] == ["alarm.create"] and len(prompts) == 1


def test_clarification_accepts_literal_arguments():
    plan = [{"id": "q", "tool": "sys.ask_missing_info", "anchor": "Đặt báo thức",
             "args": {"tool_name": "alarm.create", "params": ["time"]}}]
    result, calls, _ = run("Đặt báo thức giúp tôi.", [plan])
    assert calls == [("sys.ask_missing_info", {"tool_name": "alarm.create", "params": ["time"]})]


ROWS = [
    {"id": "train_1", "transcript": "Đặt báo thức 6 rưỡi sáng mai rồi gọi cho Lan", "apps": ["alarm", "phone"],
     "plan": {"steps": [{"id": "s1", "tool": "alarm.create", "anchor": "Đặt báo thức", "args": {}}]}},
    {"id": "train_2", "transcript": "Gọi cho Lan", "apps": ["phone"],
     "plan": {"steps": [{"id": "s1", "tool": "phone.call", "anchor": "Gọi", "args": {}}]}},
    {"id": "train_3", "transcript": "Đặt báo thức 6 giờ sáng mai rồi gọi cho Minh", "apps": ["alarm"],
     "plan": {"steps": [{"id": "s1", "tool": "flights.book", "anchor": "Đặt", "args": {}}]}},
    {"id": "raw", "transcript": TEXT, "apps": ["alarm"], "plan": {"steps": []}},
]


def test_fewshot_selection_is_deterministic_installed_only_and_bounded():
    examples = Examples(ROWS)
    sample = Sample("raw", TEXT, frozenset({"alarm", "phone"}))
    tools = {"alarm.create", "phone.call"}
    chosen = examples.select(sample, tools, k=2)
    # the sample itself and examples using uninstalled tools are never shown
    assert [item["request"] for item in chosen] == [ROWS[0]["transcript"], ROWS[1]["transcript"]]
    assert examples.select(sample, tools, k=0) == []
    assert examples.select(sample, tools, k=2, budget=150) == chosen[:1]


def test_examples_come_before_the_request_in_the_prompt():
    prompts = []

    def generate(system, user, *, deadline=None):
        prompts.append(json.loads(user))
        return '{"steps":[]}'

    sample = Sample("raw", TEXT, frozenset({"alarm", "contacts", "phone", "sms", "sys"}), context=CONTEXT)
    run_sample(sample, REGISTRY, CompiledPlanner(generate, fewshot=2, examples=Examples(ROWS)).for_sample(),
               lambda _: Env([]))
    assert list(prompts[0])[:2] == ["examples", "request"] and len(prompts[0]["examples"]) == 2
