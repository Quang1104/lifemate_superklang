import json

import pytest

from src.agent.compiled_plan import CompiledPlanner
from src.agent.plan_protocol import exact_span
from src.agent.plan_values import Unresolved, convert_span
from src.agent.runner import run_sample
from src.inference.types import Sample
from src.tooling.evidence import EvidencePath
from src.tooling.registry import ToolRegistry, ToolSpec


CONTEXT = {"current_time": "2026-11-13T10:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh"}
TEXT = "Nếu hóa đơn điện dưới 500 nghìn thì thanh toán. Dù thế nào, nhắc tôi gọi mẹ lúc 3 giờ chiều mai."
BILL = {"type": "object", "properties": {"bill_id": {"type": "string", "x-ref": "bill.bills"}},
        "required": ["bill_id"], "additionalProperties": False}


def schema(fields, required=()):
    return {"type": "object", "properties": fields, "required": list(required),
            "additionalProperties": False}


def step(step_id, tool, anchor, args=None, **extra):
    return {"id": step_id, "tool": tool, "anchor": anchor, "args": args or {}, **extra}


def span(text, mode="text"):
    return {"text": text, "as": mode}


def selected(field):
    return {"result": "lookup", "path": ["bills"], "where": {"type": {"value": "electric"}},
            "field": [field]}


def example():
    return [
        step("lookup", "bill.list", "hóa đơn điện"),
        step("pay", "bill.pay", "thanh toán", {"bill_id": selected("id")},
             when={"left": selected("amount"), "op": "lt", "right": span("500 nghìn", "money")}),
        step("remind", "reminders.create", "nhắc tôi gọi mẹ lúc 3 giờ chiều mai",
             {"text": span("gọi mẹ"), "time": span("3 giờ chiều mai", "datetime")}),
    ]


@pytest.fixture
def registry():
    string = {"type": "string"}
    tools = [
        ToolSpec("bill.list", "bill", "Read pending bills", schema({}), False,
                 (EvidencePath("bill.bills", ("bills", "*", "id")),)),
        ToolSpec("bill.pay", "bill", "Pay a bill", BILL, True),
        ToolSpec("bill.cancel", "bill", "Cancel a bill", BILL, True),
        ToolSpec("reminders.create", "reminders", "Create a reminder",
                 schema({"text": string, "time": string}, ("text", "time")), True),
        ToolSpec("sys.ask_missing_info", "sys", "Ask for missing information",
                 schema({"tool_name": string, "params": {"type": "array", "items": string}},
                        ("tool_name", "params")), False),
    ]
    return ToolRegistry(tools)


class FakeEnv:
    def __init__(self, bills, done=True):
        self.calls = []
        self.bills = bills
        self.done = done

    def call(self, name, args):
        self.calls.append((name, args))
        if name == "bill.list":
            return {"ok": True, "bills": self.bills}
        if name == "bill.pay":
            return {"ok": True, "done": self.done}
        if name == "sys.ask_missing_info":
            return {"ok": True, "waiting_for_user": True}
        return {"ok": True}


def run(registry, text, plans, *, amounts=(450_000,), done=True):
    prompts = []
    plans = iter(plans)

    def generate(system, user, *, deadline=None):
        assert deadline is not None
        prompts.append(json.loads(user))
        return json.dumps({"steps": next(plans)})

    planner = CompiledPlanner(generate).for_sample()
    environment = FakeEnv([
        {"id": str(index), "type": "electric", "amount": amount}
        for index, amount in enumerate(amounts)
    ], done)
    opened = []

    def open_environment(sample):
        opened.append(sample.sample_id)
        return environment

    sample = Sample("sample", text, frozenset({"bill", "reminders", "sys"}), context=CONTEXT)
    result = run_sample(sample, registry, planner, open_environment)
    assert opened == ["sample"]
    return result, environment.calls, prompts


@pytest.mark.parametrize("amount, expected", [(450_000, ["bill.list", "bill.pay", "reminders.create"]),
                                              (500_000, ["bill.list", "reminders.create"]),
                                              (600_000, ["bill.list", "reminders.create"])])
def test_condition_and_independent_goal(registry, amount, expected):
    result, calls, prompts = run(registry, TEXT, [example()], amounts=(amount,))
    assert [name for name, _ in calls] == expected
    assert calls[-1][1] == {"text": "gọi mẹ", "time": "2026-11-14T15:00:00+07:00"}
    assert len(prompts) == 1
    assert result.reply == ""


def test_failed_write_blocks_dependents_without_repeating_or_blocking_independent_goal(registry):
    steps = example()
    steps.insert(2, step("cancel", "bill.cancel", "thanh toán", {"bill_id": selected("id")}, after=["pay"]))
    result, calls, prompts = run(registry, TEXT, [steps], done=False)
    assert [name for name, _ in calls] == ["bill.list", "bill.pay", "reminders.create"]
    assert len(prompts) == 1
    assert result.status == "plan_incomplete"


def test_invalid_later_step_rejects_plan_before_any_dispatch(registry):
    steps = example()
    steps[-1]["args"]["time"] = span("4 giờ chiều mai", "datetime")
    result, calls, prompts = run(registry, TEXT, [steps, steps])
    assert not calls
    assert len(prompts) == 2
    assert result.status == "plan_invalid"


def test_ambiguous_entity_repair_updates_only_focus(registry):
    steps = example()
    updated = example()[1]
    updated["args"]["bill_id"]["where"]["amount"] = {"value": 450_000}
    updated["when"]["left"]["where"]["amount"] = {"value": 450_000}
    result, calls, prompts = run(registry, TEXT, [steps, [updated]], amounts=(450_000, 460_000))
    # The proposed repair must preserve the user condition and cannot drop the reminder.
    assert [name for name, _ in calls] == ["bill.list", "reminders.create"]
    assert prompts[1]["repair"]["focus"]["id"] == "pay"
    assert result.status == "plan_incomplete"


def test_ambiguous_entity_resolved_with_one_repair(registry):
    text = "Thanh toán hóa đơn điện 450 nghìn rồi nhắc tôi gọi mẹ lúc 3 giờ chiều mai."
    steps = [example()[0], step("pay", "bill.pay", "Thanh toán", {"bill_id": selected("id")}), example()[2]]
    updated = step("pay", "bill.pay", "Thanh toán", {"bill_id": selected("id")})
    updated["args"]["bill_id"]["where"]["amount"] = span("450 nghìn", "money")
    result, calls, prompts = run(registry, text, [steps, [updated]], amounts=(450_000, 460_000))
    assert len(prompts) == 2
    assert calls[1] == ("bill.pay", {"bill_id": "0"})
    assert calls[-1][0] == "reminders.create"
    assert result.reply == ""


def test_repair_cannot_drop_unrelated_work(registry):
    result, calls, prompts = run(registry, TEXT, [example(), []], amounts=(450_000, 460_000))
    assert len(prompts) == 2
    assert [name for name, _ in calls] == ["bill.list", "reminders.create"]
    assert result.status == "plan_incomplete"


def test_automatic_lookup_uses_unique_installed_producer(registry):
    request = "Thanh toán hóa đơn điện."
    plan = [step("pay", "bill.pay", "Thanh toán",
                 {"bill_id": {"lookup": {"type": {"value": "electric"}}}})]
    result, calls, prompts = run(registry, request, [plan])
    assert calls == [("bill.list", {}), ("bill.pay", {"bill_id": "0"})]
    assert len(prompts) == 1
    assert result.reply == ""


def test_negation_keeps_positive_followup(registry):
    request = "Đừng thanh toán rồi nhắc tôi gọi mẹ lúc 3 giờ chiều mai."
    result, calls, prompts = run(
        registry, request, [[step("pay", "bill.pay", "thanh toán", {"bill_id": {"value": "fake"}})],
                            [example()[2]]]
    )
    assert [name for name, _ in calls] == ["reminders.create"]
    assert len(prompts) == 2
    assert result.reply == ""


def test_clarification_and_unsupported_are_distinct(registry):
    clarify = step("ask", "sys.ask_missing_info", "nhắc tôi gọi mẹ",
                   {"tool_name": {"value": "reminders.create"}, "params": {"array": [{"value": "time"}]}})
    result, calls, _ = run(registry, "nhắc tôi gọi mẹ", [[clarify]])
    assert calls == [("sys.ask_missing_info", {"tool_name": "reminders.create", "params": ["time"]})]
    assert result.status == "waiting_for_user"
    result, calls, prompts = run(registry, "dịch bài thơ", [[]])
    assert not calls
    assert len(prompts) == 1
    assert result.reply == ""


@pytest.mark.parametrize("request_text, anchor, blocked", [
    ("Đừng quên nhắc tôi gọi mẹ.", "nhắc tôi gọi mẹ", False),
    ("Quạt thì khỏi bật, bật điều hoà 26 độ.", "bật điều hoà 26 độ", False),
    ("Quạt thì khỏi bật, bật điều hoà 26 độ.", "khỏi bật", True),
    ("Đèn bật 40 phần trăm, quạt thì đừng bật.", "bật", False),  # one occurrence is positive
    ("Quạt thì đừng bật nhé.", "bật", True),
    ("Thôi khỏi chuyển ngân hàng nữa, coi giùm ví.", "chuyển ngân hàng", True),
    ("Bỏ nón lá khỏi danh sách đi chợ giùm tôi.", "danh sách đi chợ", False),
    ("Ví đủ thì xài ví, không thì trả ngân hàng.", "trả ngân hàng", False),
    ("Đừng thanh toán rồi nhắc tôi gọi mẹ.", "thanh toán", True),
])
def test_negation_respects_clause_and_occurrence(request_text, anchor, blocked):
    from src.agent.compiled_plan import Step, _blocked_write

    sample = Sample("x", request_text, frozenset(), context=CONTEXT)
    assert _blocked_write(Step("s", "t", anchor, {}, ()), sample) is blocked


@pytest.mark.parametrize("text, mode, expected", [
    ("8 giờ kém 15 tối mai", "datetime", "2026-11-14T19:45:00+07:00"),
    ("6 rưỡi sáng mai", "datetime", "2026-11-14T06:30:00+07:00"),
    ("sáu giờ rưỡi sáng mai", "datetime", "2026-11-14T06:30:00+07:00"),
    ("mười một giờ đêm", "time", "23:00"),
    ("11 giờ trưa", "time", "11:00"),
    ("1 giờ trưa", "time", "13:00"),
    ("lúc 15h30", "time", "15:30"),
    ("năm giờ bốn lăm", "datetime", "2026-11-13T17:45:00+07:00"),  # nearest future reading
    ("9 giờ", "datetime", "2026-11-13T09:00:00+07:00"),  # an hour ago still counts
    ("ba giờ rưỡi sáng", "datetime", "2026-11-14T03:30:00+07:00"),
    ("chiều nay", "date", "2026-11-13"),
    ("bữa ni", "date", "2026-11-13"),
    ("thứ hai tuần sau", "date", "2026-11-16"),  # sample day is Friday
    ("chủ nhật này", "date", "2026-11-15"),
    ("thứ ba tới", "date", "2026-11-17"),
    ("ba giờ chiều thứ ba tuần sau", "datetime", "2026-11-17T15:00:00+07:00"),
    ("ngày 19", "date", "2026-11-19"),
    ("ngày 5", "date", "2026-12-05"),
    ("tám giờ sáng hai mốt tháng mười một", "datetime", "2026-11-21T08:00:00+07:00"),
    ("mùng sáu tháng mười", "date", "2027-10-06"),
])
def test_vietnamese_clock_and_calendar_forms(text, mode, expected):
    assert convert_span(text, mode, CONTEXT) == expected


@pytest.mark.parametrize("text", ["thứ tư này", "12 giờ đêm", "ngày 31 tháng hai", "8 giờ kém 15 30"])
def test_unsupported_or_impossible_times_stay_unresolved(text):
    with pytest.raises(Unresolved):
        convert_span(text, "datetime", CONTEXT)


@pytest.mark.parametrize("text, minutes", [("1 tiếng", 60), ("nửa tiếng", 30), ("hai tiếng", 120),
                                           ("bốn mươi lăm phút", 45), ("một tiếng rưỡi", 90),
                                           ("2 tiếng 15 phút", 135), ("60 phút", 60),
                                           ("thêm 45 phút", 45), ("hai chục phút", 20)])
def test_spoken_durations(text, minutes):
    assert convert_span(text, "duration", CONTEXT) == minutes


@pytest.mark.parametrize("value, minutes, expected", [
    ("2026-11-29T08:00:00+07:00", 60, "2026-11-29T09:00:00+07:00"),
    ("2026-11-29T00:15:00+07:00", -45, "2026-11-28T23:30:00+07:00"),
    ("21:30", 45, "22:15"),
])
def test_shift_keeps_form_and_offset(value, minutes, expected):
    from src.agent.plan_values import shift_time

    assert shift_time(value, minutes) == expected


@pytest.mark.parametrize("value, minutes", [("2026-11-29", 60), ("tomorrow", 60), ("21:30", 0),
                                            ("21:30", True), ("21:30", 99_999)])
def test_shift_rejects_dates_and_unobserved_amounts(value, minutes):
    from src.agent.plan_values import shift_time

    with pytest.raises(Unresolved):
        shift_time(value, minutes)


def _calendar_run(plan_steps, text="Dời lịch chạy bộ mai muộn 1 tiếng nhé."):
    string = {"type": "string"}
    registry = ToolRegistry([
        ToolSpec("calendar.list_events", "calendar", "List events", schema({}), False,
                 (EvidencePath("calendar.events", ("events", "*", "id")),)),
        ToolSpec("calendar.update_event", "calendar", "Move an event",
                 schema({"event_id": {"type": "string", "x-ref": "calendar.events"}, "start": string, "end": string},
                        ("event_id",)), True),
    ])

    class Calendar:
        calls = []

        def call(self, name, args):
            self.calls.append((name, args))
            if name == "calendar.list_events":
                return {"ok": True, "events": [{"id": "evt_01", "title": "Chạy bộ",
                                                "start": "2026-11-14T08:00:00+07:00",
                                                "end": "2026-11-14T09:00:00+07:00"}]}
            return {"ok": True}

    plans = iter([plan_steps, plan_steps])  # the repair turn repeats the plan
    planner = CompiledPlanner(lambda *args, **kwargs: json.dumps({"steps": next(plans)})).for_sample()
    environment = Calendar()
    sample = Sample("s", text, frozenset({"calendar"}), context=CONTEXT)
    return run_sample(sample, registry, planner, lambda _: environment), environment.calls


def test_event_moved_later_by_spoken_duration():
    event = {"result": "list", "path": ["events"], "where": {"title": span("chạy bộ")}}
    later = {"text": "1 tiếng", "as": "duration"}
    plan = [step("list", "calendar.list_events", "lịch chạy bộ"),
            step("move", "calendar.update_event", "Dời lịch chạy bộ mai muộn 1 tiếng", {
                "event_id": {**event, "field": ["id"]},
                "start": {"shift": {**event, "field": ["start"]}, "by": later, "dir": "later"},
                "end": {"shift": {**event, "field": ["end"]}, "by": later, "dir": "later"}})]
    result, calls = _calendar_run(plan)
    assert calls[1] == ("calendar.update_event", {"event_id": "evt_01", "start": "2026-11-14T09:00:00+07:00",
                                                  "end": "2026-11-14T10:00:00+07:00"})
    assert result.reply == ""


def test_shift_amount_must_be_grounded():
    event = {"result": "list", "path": ["events"], "where": {"title": span("chạy bộ")}}
    plan = [step("list", "calendar.list_events", "lịch chạy bộ"),
            step("move", "calendar.update_event", "Dời lịch chạy bộ mai muộn 1 tiếng", {
                "event_id": {**event, "field": ["id"]},
                "start": {"shift": {**event, "field": ["start"]}, "by": {"value": 60}, "dir": "later"}})]
    result, calls = _calendar_run(plan)
    assert not calls
    assert result.status == "plan_invalid"


def test_fixed_offset_used_without_tz_database():
    context = {"current_time": "2026-11-13T10:00:00+07:00", "timezone": "No/Such_Zone"}
    assert convert_span("3 giờ chiều mai", "datetime", context) == "2026-11-14T15:00:00+07:00"


def test_current_and_relative_time_are_distinct():
    assert convert_span("3 giờ chiều mai", "datetime", CONTEXT) == "2026-11-14T15:00:00+07:00"
    assert convert_span("3 giờ nữa", "datetime", CONTEXT) == "2026-11-13T13:00:00+07:00"
    assert convert_span("năm trăm nghìn đồng", "money", CONTEXT) == 500_000
    with pytest.raises(Unresolved):
        exact_span("trả 15 nghìn", "5 nghìn")
    assert exact_span("Gọi Mẹ.", "gọi mẹ") == "Gọi Mẹ"
