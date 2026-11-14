"""CPU-only representation rules; integration runs when jsonschema is installed."""

from copy import deepcopy
import importlib.util
import json
import os
import unittest
from unittest.mock import patch

from src.agent.arg_normalize import normalize_arguments


CONTEXT = {"current_time": "2026-11-15T01:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh"}


def schema(fields):
    return {"type": "object", "properties": fields, "additionalProperties": False}


def normalize(value, field_schema, *, field="value", context=CONTEXT):
    return normalize_arguments({field: value}, schema({field: field_schema}), context)[field]


class ArgumentNormalizationTests(unittest.TestCase):
    def test_trims_only_explicit_string_fields(self):
        self.assertEqual(normalize("  north \n", {"type": "string"}), "north")
        self.assertEqual(normalize("  north ", {}), "  north ")
        self.assertEqual(normalize("  north ", {"type": ["string", "null"]}), "  north ")
        self.assertEqual(normalize_arguments({"unknown": " text "}, schema({})), {"unknown": " text "})

    def test_identifiers_and_names_remain_verbatim(self):
        for field in ("id", "contact_id", "ids", "deviceIds", "name", "recipient_name", "tool_name"):
            with self.subTest(field=field):
                self.assertEqual(normalize("  Má \n", {"type": "string", "enum": ["ma"]}, field=field),
                                 "  Má \n")
        self.assertEqual(normalize(" 001 ", {"type": "integer", "x-ref": "things"}), " 001 ")
        self.assertEqual(normalize(" Sếp Dũng ", {"type": "string", "description": "Tên hoặc số điện thoại"},
                                   field="query"), " Sếp Dũng ")
        self.assertEqual(normalize([" c1 "], {"type": "array", "items": {"type": "string"},
                                             "description": "Danh sách ID liên hệ"}), [" c1 "])

    def test_fuzzy_annotations_protect_entire_subtree(self):
        self.assertEqual(normalize("  Hẹn Sếp Dũng \n", {"type": "string", "x-match": "fuzzy"}),
                         "  Hẹn Sếp Dũng \n")
        value = {"text": "  nguyên văn "}
        self.assertEqual(normalize(value, {"type": "object", "x-match": "fuzzy",
                                          "properties": {"text": {"type": "string"}}}), value)

    def test_integer_strings_and_explicit_currency_scales(self):
        for value, expected in (("500 nghìn", 500_000), ("500k", 500_000), (" 500 K ", 500_000),
                                ("2 triệu đồng", 2_000_000), ("1,5 triệu", 1_500_000),
                                ("1.25 tỷ", 1_250_000_000), ("0500", 500), ("-12", -12),
                                ("500000 VNĐ", 500_000), ("2 ngàn", 2_000)):
            with self.subTest(value=value):
                actual = normalize(value, {"type": "integer"})
                self.assertEqual(actual, expected)
                self.assertIs(type(actual), int)

    def test_integer_rule_never_rounds_guesses_groups_or_converts_other_units(self):
        for value in ("500.000", "500,000", "1.001k", "một triệu", "500 usd", "30 phút", "1e3", "--5"):
            with self.subTest(value=value):
                self.assertEqual(normalize(value, {"type": "integer"}), value)
        self.assertEqual(normalize("500k", {"type": "string"}), "500k")
        self.assertEqual(normalize("500k", {"type": "number"}), "500k")
        self.assertIs(normalize(True, {"type": "integer"}), True)
        self.assertEqual(normalize(1.5, {"type": "integer"}), 1.5)

    def test_enum_matches_case_and_vietnamese_diacritics_uniquely(self):
        enum = {"type": "string", "enum": ["ĐIỆN", "nước", "gas"]}
        self.assertEqual(normalize(" dien ", enum), "ĐIỆN")
        self.assertEqual(normalize("NƯỚC", enum), "nước")
        self.assertEqual(normalize("unknown", enum), "unknown")
        self.assertEqual(normalize(" ma ", {"type": "string", "enum": ["má", "mà"]}), "ma")
        self.assertEqual(normalize("má", {"type": "string", "enum": ["má", "mà"]}), "má")
        self.assertEqual(normalize("true", {"type": "string", "enum": [True]}), "true")

    def test_date_formatting_uses_only_an_unambiguous_year_first_date(self):
        for value in ("2026-1-2", "2026/1/2", " 2026-01-02 "):
            self.assertEqual(normalize(value, {"type": "string", "format": "date"}), "2026-01-02")
        for value in ("02/01/2026", "mai", "2026-02-30", "2026-01-02T12:00:00+07:00"):
            self.assertEqual(normalize(value, {"type": "string", "format": "date"}), value)

    def test_datetime_adds_only_context_offset_and_canonical_seconds(self):
        date_time = {"type": "string", "format": "date-time"}
        self.assertEqual(normalize("2026-1-2 3:04", date_time), "2026-01-02T03:04:00+07:00")
        self.assertEqual(normalize("2026-01-02t03:04:05.12+0700", date_time),
                         "2026-01-02T03:04:05.120000+07:00")
        self.assertEqual(normalize("2026-01-01T20:04:00Z", date_time), "2026-01-02T03:04:00+07:00")
        offset_context = {"current_time": "2026-07-01T12:00:00-03:30"}
        self.assertEqual(normalize("2026-01-02T03:04", date_time, context=offset_context),
                         "2026-01-02T03:04:00-03:30")

    def test_datetime_never_infers_a_date_or_uses_the_host_timezone(self):
        date_time = {"type": "string", "format": "date-time"}
        for value in ("mai 03:04", "03:04", "2026-01-02", "2026-02-30T03:04", "2026-01-02T03:04+07:99",
                      "2026-01-02T25:04", "2026-01-02T03:04:05.1234567Z"):
            with self.subTest(value=value):
                self.assertEqual(normalize(value, date_time), value)
        for context in (None, {}, {"current_time": "invalid"}, {"current_time": "2026-01-02T12:00:00"}):
            self.assertEqual(normalize("2026-01-02 03:04", date_time, context=context), "2026-01-02 03:04")
        self.assertEqual(normalize("2026-01-02T03:04Z", date_time, context=None),
                         "2026-01-02T03:04:00+00:00")

    def test_schema_description_format_markers_and_explicit_precedence(self):
        self.assertEqual(normalize("2026-1-2 3:04", {"type": "string", "description": "ISO-8601 đầy đủ giây"}),
                         "2026-01-02T03:04:00+07:00")
        self.assertEqual(normalize("2026/1/2", {"type": "string", "description": "YYYY-MM-DD, ví dụ ..."}),
                         "2026-01-02")
        self.assertEqual(normalize("2026-1-2 3:04", {"type": "string", "format": "time",
                                                   "description": "ISO-8601"}), "2026-1-2 3:04")
        self.assertEqual(normalize("2026-1-2 3:04", {"type": "string"}), "2026-1-2 3:04")

    def test_recursive_fields_are_pure_and_never_add_or_remove_keys(self):
        parameters = schema({"entries": {"type": "array", "items": schema({
            "quantity": {"type": "integer"}, "item_id": {"type": "string"},
            "text": {"type": "string", "x-match": "fuzzy"}})}})
        arguments = {"entries": [{"quantity": "2", "item_id": " c1 ", "text": " text "}], "extra": " X "}
        before = deepcopy((arguments, parameters, CONTEXT))
        output = normalize_arguments(arguments, parameters, CONTEXT)
        self.assertEqual(output, {"entries": [{"quantity": 2, "item_id": " c1 ", "text": " text "}],
                                  "extra": " X "})
        self.assertEqual((arguments, parameters, CONTEXT), before)
        output["entries"][0]["text"] = "changed"
        self.assertEqual(arguments, before[0])
        self.assertEqual(normalize_arguments({}, schema({"required": {"type": "string"}})), {})

    def test_schema_composition_and_references_are_left_for_the_guard(self):
        for field_schema in ({"type": "string", "$ref": "#/$defs/fuzzy"},
                             {"type": "integer", "anyOf": [{"type": "integer"}, {"type": "string"}]}):
            self.assertEqual(normalize(" 500 ", field_schema), " 500 ")


@unittest.skipUnless(importlib.util.find_spec("jsonschema"), "jsonschema is unavailable; runner integration NOT MEASURED")
class ArgumentNormalizationRunnerTests(unittest.TestCase):
    def run_calls(self, actions, *, flag=None, fields=None, installed=True):
        from src.agent.runner import run_sample
        from src.inference.types import Sample
        from src.tooling.registry import ToolRegistry, ToolSpec
        fields = fields or {"amount": {"type": "integer", "minimum": 1}}
        parameters = schema(fields)
        parameters["required"] = list(fields)
        registry = ToolRegistry([ToolSpec("bank.pay", "bank", "Pay", parameters, True)])
        sample = Sample("fake", "Pay", frozenset({"bank"} if installed else {}), context=CONTEXT)
        outputs = iter([json.dumps({"action": "call", "tool": "bank.pay", "arguments": args}) for args in actions]
                       + [json.dumps({"action": "finish", "reply": ""})])
        calls = []

        class Environment:
            def call(self, name, arguments):
                calls.append((name, arguments))
                return {"ok": True}

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LIFEMATE_ARG_NORMALIZE", None)
            if flag is not None:
                os.environ["LIFEMATE_ARG_NORMALIZE"] = flag
            result = run_sample(sample, registry, lambda request: next(outputs), lambda sample: Environment())
        return result, calls

    def test_default_off_and_only_exact_one_enables(self):
        for flag in (None, "0", "true", "yes"):
            result, calls = self.run_calls([{"amount": "500k"}], flag=flag)
            self.assertEqual(calls, [])
            self.assertEqual(result.rejections, 1)
        result, calls = self.run_calls([{"amount": "500k"}], flag="1")
        self.assertEqual(calls, [("bank.pay", {"amount": 500_000})])
        self.assertEqual(result.rejections, 0)

    def test_guard_still_rejects_out_of_range_unknown_and_unavailable_tools(self):
        for arguments in ({"amount": "-1"}, {"amount": "2", "extra": True}):
            result, calls = self.run_calls([arguments], flag="1")
            self.assertEqual(calls, [])
            self.assertEqual(result.rejections, 1)
        result, calls = self.run_calls([{"amount": "500k"}], flag="1", installed=False)
        self.assertEqual(calls, [])
        self.assertEqual(result.rejections, 1)

    def test_normalized_duplicate_write_cannot_dispatch_twice(self):
        result, calls = self.run_calls([{"amount": "500k"}, {"amount": 500_000}], flag="1")
        self.assertEqual(calls, [("bank.pay", {"amount": 500_000})])
        self.assertEqual(result.rejections, 1)

    def test_unobserved_reference_remains_rejected_and_unmodified(self):
        result, calls = self.run_calls([{"account_id": " guessed "}], flag="1",
                                       fields={"account_id": {"type": "string", "x-ref": "bank.accounts"}})
        self.assertEqual(calls, [])
        self.assertEqual(result.rejections, 1)
        self.assertIn("(x-ref)", result.guard_rejections[0]["reason"])


if __name__ == "__main__":
    unittest.main()
