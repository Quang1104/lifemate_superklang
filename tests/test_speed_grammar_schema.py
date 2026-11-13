"""Compiled-plan grammar shape checks; fixtures mirror verified compiler examples."""

from copy import deepcopy
import json
import unittest

try:
    from jsonschema import Draft202012Validator
except ImportError:
    Draft202012Validator = None

from src.agent.plan_grammar import plan_json_schema


def _tool(name, fields, required=()):
    return {"name": name, "parameters": {
        "type": "object", "properties": fields, "required": list(required),
        "additionalProperties": False,
    }}


def _step(identifier, tool, args=None, **extra):
    return {"id": identifier, "tool": tool, "anchor": "thanh toán", "args": args or {}, **extra}


def _selected(field):
    return {"result": "lookup", "path": ["bills"],
            "where": {"type": {"value": "electric"}}, "field": [field]}


TOOLS = [
    _tool("bill.list", {}),
    _tool("bill.pay", {"bill_id": {"type": "string", "x-ref": "bill.bills"}}, ("bill_id",)),
    _tool("reminders.create", {"text": {"type": "string"}, "time": {"type": "string"}},
          ("text", "time")),
    _tool("calendar.update_event", {
        "event_id": {"type": "string", "x-ref": "calendar.events"},
        "start": {"type": "string"}, "end": {"type": "string"},
    }, ("event_id",)),
    _tool("sys.ask_missing_info", {
        "tool_name": {"type": "string"}, "params": {"type": "array", "items": {"type": "string"}},
    }, ("tool_name", "params")),
    _tool("demo.echo", {"payload": {}}, ("payload",)),
]


@unittest.skipUnless(Draft202012Validator, "jsonschema unavailable; grammar acceptance NOT MEASURED")
class PlanGrammarSchemaTests(unittest.TestCase):
    def validator(self, tools=None, limit=8):
        result = plan_json_schema(TOOLS if tools is None else tools, limit)
        Draft202012Validator.check_schema(result)
        return Draft202012Validator(result)

    def assert_valid(self, plan, *, tools=None, limit=8):
        self.validator(tools, limit).validate(plan)

    def assert_invalid(self, plan, *, tools=None, limit=8):
        self.assertFalse(self.validator(tools, limit).is_valid(plan), repr(plan))

    def test_verified_conditional_bill_and_independent_reminder_shape(self):
        # Same selectors, comparison and datetime SOURCE as test_compiled_plan.example.
        plan = {"steps": [
            _step("lookup", "bill.list"),
            _step("pay", "bill.pay", {"bill_id": _selected("id")},
                  when={"left": _selected("amount"), "op": "lt",
                        "right": {"text": "500 nghìn", "as": "money"}}),
            _step("remind", "reminders.create", {
                "text": {"text": "gọi mẹ"},
                "time": {"text": "3 giờ chiều mai", "as": "datetime"},
            }, after=["lookup"]),
        ]}
        self.assert_valid(plan, limit=3)

    def test_verified_shift_lookup_and_clarification_shapes(self):
        event = {"result": "lookup", "path": ["events"],
                 "where": {"title": {"text": "chạy bộ"}}}
        duration = {"text": "1 tiếng", "as": "duration"}
        for plan in [
            {"steps": [_step("move", "calendar.update_event", {
                "event_id": {**event, "field": ["id"]},
                "start": {"shift": {**event, "field": ["start"]},
                          "by": duration, "dir": "later"},
                "end": {"shift": {**event, "field": ["end"]},
                        "by": duration, "dir": "later"},
            })]},
            {"steps": [_step("pay", "bill.pay", {
                "bill_id": {"lookup": {"type": {"value": "electric"}}},
            })]},
            {"steps": [_step("ask", "sys.ask_missing_info", {
                "tool_name": {"value": "reminders.create"},
                "params": {"array": [{"value": "time"}]},
            })]},
        ]:
            with self.subTest(plan=plan):
                self.assert_valid(plan)

    def test_verified_converter_selector_enum_and_span_shape(self):
        tools = [
            _tool("smart_home.get_status", {}),
            _tool("smart_home.light_set", {
                "device_id": {"type": "string", "x-ref": "smart_home.devices"},
                "power": {"type": "string", "enum": ["on", "off"]},
            }, ("device_id",)),
            _tool("reminders.create", {"content": {"type": "string"}, "time": {"type": "string"}},
                  ("content", "time")),
        ]
        # Shape asserted by test_quality_gold_to_plan's verified converter example.
        self.assert_valid({"steps": [
            _step("s1", "smart_home.get_status"),
            _step("s2", "smart_home.light_set", {
                "device_id": {"result": "s1", "path": ["devices"],
                              "where": {"name": {"text": "đèn phòng ngủ"}}, "field": ["id"]},
                "power": {"value": "off"},
            }),
            _step("s3", "reminders.create", {
                "content": {"text": "gọi mẹ"},
                "time": {"text": "lúc 3 giờ chiều mai", "as": "datetime"},
            }),
        ]}, tools=tools)

    def test_source_forms_and_nested_source_values(self):
        sources = [
            {"text": "mẹ"}, {"text": "2", "as": "integer"},
            {"value": True}, {"value": "on"}, {"value": None},
            {"context": ["user_profile", "home_address"]},
            {"history": 1, "path": ["item", "id"]},
            {"history_text": 1, "text": "2", "as": "number"},
            {"history_text": 1, "text": "1 tiếng", "as": "duration"},
            {"result": "lookup"},
            {"result": "lookup", "path": [], "field": []},
            {"result": "lookup", "path": ["item", "id"]},
            {"result": "lookup", "path": ["items"],
             "where": {"name": {"text": "mẹ"}}, "field": ["id"]},
            {"lookup": {"name": {"text": "mẹ"}}},
            {"shift": {"context": ["start"]},
             "by": {"text": "1 tiếng", "as": "duration"}, "dir": "earlier"},
            {"object": {"name": {"text": "mẹ"}, "enabled": {"value": True}}},
            {"array": [{"text": "mẹ"}, {"object": {"n": {"text": "2", "as": "integer"}}}]},
        ]
        for source in sources:
            with self.subTest(source=source):
                self.assert_valid({"steps": [_step("s1", "demo.echo", {"payload": source})]})

    def test_raw_json_scalars_and_arrays_are_accepted_for_grounded_coercion(self):
        for raw in [None, False, True, 0, -2, 1.5, "mẹ", [],
                    ["mẹ", 2, False, None, [1, 2], {"note": "JSON array fixture"}]]:
            with self.subTest(raw=raw):
                self.assert_valid({"steps": [_step("s1", "demo.echo", {"payload": raw})]})

    def test_unknown_tools_and_cross_tool_argument_names_are_rejected(self):
        for step in [
            _step("s1", "uninstalled.action"),
            _step("s1", "bill.list", {"bill_id": {"text": "mẹ"}}),
            _step("s1", "bill.pay", {"bill_id": {"text": "mẹ"}, "time": {"text": "mai"}}),
            _step("s1", "reminders.create", {
                "text": {"text": "mẹ"}, "time": {"text": "mai"}, "invented_arg": True,
            }),
        ]:
            with self.subTest(step=step):
                self.assert_invalid({"steps": [step]})

    def test_required_arguments_and_bounded_steps_are_enforced(self):
        self.assert_invalid({"steps": [_step("s1", "bill.pay")]})
        self.assert_valid({"steps": []}, limit=0)
        self.assert_invalid({"steps": [_step("s1", "bill.list")]}, limit=0)
        steps = [_step("s1", "bill.list"), _step("s2", "bill.list")]
        self.assert_valid({"steps": steps}, limit=2)
        self.assert_invalid({"steps": steps}, limit=1)
        self.assert_valid({"steps": []}, tools=[], limit=8)
        self.assert_invalid({"steps": [_step("s1", "bill.list")]}, tools=[], limit=8)

    def test_plan_and_step_structure_rejects_extra_fields_and_invalid_ids(self):
        valid_step = _step("s1", "bill.list")
        for plan in [
            {}, {"steps": {}}, {"steps": [], "explanation": "extra"},
            {"steps": [{**valid_step, "extra": True}]},
            {"steps": [{key: value for key, value in valid_step.items() if key != "anchor"}]},
            {"steps": [{**valid_step, "args": []}]},
            {"steps": [{**valid_step, "anchor": 1}]},
            {"steps": [{**valid_step, "after": [1]}]},
        ]:
            with self.subTest(plan=plan):
                self.assert_invalid(plan)
        for identifier in ["", "1step", "s x", "s" * 33, "s/1"]:
            with self.subTest(identifier=identifier):
                self.assert_invalid({"steps": [{**valid_step, "id": identifier}]})
        self.assert_valid({"steps": [{**valid_step, "id": "s_1-a"}]})

    def test_invalid_typed_source_shapes_are_rejected(self):
        for source in [
            {}, {"not_a_source": "mẹ"}, {"text": "mẹ", "extra": 1},
            {"text": "mẹ", "as": "invented_conversion"},
            {"value": {}}, {"value": []},
            {"context": []}, {"context": [""]}, {"context": ["a"] * 9},
            {"history": 0, "path": ["id"]}, {"history": True, "path": ["id"]},
            {"history_text": 0, "text": "mẹ"},
            {"history_text": 1, "text": "mai", "as": "datetime"},
            {"result": 1}, {"result": "s1", "path": [0]},
            {"lookup": {}}, {"lookup": []},
            {"shift": {"context": ["start"]}, "by": {"text": "1 tiếng"}, "dir": "sideways"},
            {"object": []}, {"array": {}},
        ]:
            with self.subTest(source=source):
                self.assert_invalid({"steps": [_step("s1", "demo.echo", {"payload": source})]})

    def test_condition_uses_supported_operator_and_typed_sources(self):
        for operator in ["lt", "le", "gt", "ge", "eq", "ne"]:
            self.assert_valid({"steps": [_step("s1", "bill.list", when={
                "left": {"context": ["amount"]}, "op": operator, "right": {"text": "2"},
            })]})
        for when in [
            {"left": {"text": "1"}, "op": "equals", "right": {"text": "2"}},
            {"left": 1, "op": "lt", "right": {"text": "2"}},
            {"left": {"text": "1"}, "op": "lt"},
            {"left": {"text": "1"}, "op": "lt", "right": {"text": "2"}, "extra": 1},
        ]:
            with self.subTest(when=when):
                self.assert_invalid({"steps": [_step("s1", "bill.list", when=when)]})

    def test_schema_uses_shared_sources_and_stays_compact_without_mutating_tools(self):
        tools = [_tool("demo.tool_" + str(index), {"a": {}, "b": {}, "c": {}}, ("a",))
                 for index in range(25)]
        for tool in tools:
            tool["description"] = "Irrelevant tool description. " * 500
        before = deepcopy(tools)
        result = plan_json_schema(tools, 8)
        encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        self.assertEqual(tools, before)
        self.assertIn("$ref", encoded)
        self.assertLess(len(encoded), 100_000)
        self.assertNotIn("Irrelevant tool description", encoded)
        Draft202012Validator.check_schema(result)


if __name__ == "__main__":
    unittest.main()


def test_constrained_planner_passes_schema_and_falls_back_without_xgrammar(monkeypatch):
    import json
    import sys
    from collections import Counter
    from src.agent.compiled_plan import CompiledPlanner
    from src.agent.local_model import LocalPlanner

    seen = {}

    def generate(system, user, *, deadline=None, json_schema=None):
        seen["schema"] = json_schema
        return json.dumps({"steps": []})

    from src.agent.planner import PlannerInput
    from src.inference.types import Sample
    tools = [{"name": "a.b", "parameters": {"type": "object", "properties": {}, "required": []}}]
    planner = CompiledPlanner(generate, constrain=True).for_sample()
    planner(PlannerInput(Sample("s", "x", frozenset({"a"})), tools, (), None))
    assert seen["schema"]["properties"]["steps"]["maxItems"] == 7

    local = LocalPlanner.__new__(LocalPlanner)
    local._grammars, local.grammar_stats = {}, Counter()
    monkeypatch.setitem(sys.modules, "xgrammar", None)
    assert local._grammar_processor(None) is None
    assert local._grammar_processor(seen["schema"]) is None
    assert local.grammar_stats == Counter({"unavailable": 1})


def test_grammar_logits_masks_with_int_tokens_and_falls_back(monkeypatch):
    import sys
    import types
    from collections import Counter

    torch = __import__("pytest").importorskip("torch")
    from src.agent.local_model import _GrammarLogits

    accepted, masks = [], []

    class Matcher:
        def __init__(self, compiled):
            self.done = False

        def is_terminated(self):
            return self.done

        def accept_token(self, token):
            assert type(token) is int  # the upstream adapter bug passed a tensor here
            accepted.append(token)
            return token != 99

        def fill_next_token_bitmask(self, bitmask):
            masks.append(True)

    def apply(scores, bitmask, *, vocab_size):
        scores[0, 0] = -1e9

    fake = types.SimpleNamespace(GrammarMatcher=Matcher, allocate_token_bitmask=lambda b, v: torch.zeros(b, 1),
                                 apply_token_bitmask_inplace=apply)
    monkeypatch.setitem(sys.modules, "xgrammar", fake)
    stats = Counter()
    processor = _GrammarLogits("compiled", 4, stats)
    first = processor(torch.tensor([[1, 2]]), torch.zeros(1, 4))
    assert first[0, 0] == -1e9 and accepted == []
    processor(torch.tensor([[1, 2, 3]]), torch.zeros(1, 4))
    assert accepted == [3]
    untouched = processor(torch.tensor([[1, 2, 3, 99]]), torch.zeros(1, 4))  # rejected token
    assert untouched[0, 0] == 0 and stats == Counter({"runtime_fallback": 1})
    assert processor(torch.tensor([[1, 2, 3, 99, 5]]), torch.zeros(1, 4))[0, 0] == 0
