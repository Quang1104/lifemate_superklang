"""Preserve planner-relevant tool constraints while removing JSON boilerplate."""

from copy import deepcopy
import json
import unittest

from src.agent.schema_text import render_schema, render_tool, render_tools


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class SchemaTextTests(unittest.TestCase):
    def test_required_optional_enums_refs_and_write_effect_are_explicit(self):
        tool = {
            "name": "contacts.update", "description": "Sửa liên hệ", "writes": True,
            "parameters": {
                "type": "object", "properties": {
                    "contact_id": {"type": "string", "x-ref": "contacts.contacts"},
                    "phone": {"type": "string", "description": "Giữ nguyên chuỗi 09123456785"},
                    "mode": {"type": "string", "enum": ["home", "work"]},
                }, "required": ["contact_id"], "additionalProperties": False,
                "x-one-of": [["phone"], ["mode"]],
            }, "returns": "contact",
        }
        self.assertEqual(
            render_tool(tool),
            'contacts.update(contact_id:string{"x-ref":"contacts.contacts"},'
            'phone?:string{"description":"Giữ nguyên chuỗi 09123456785"},'
            'mode?:string{"enum":["home","work"]})!'
            '{"x-one-of":[["phone"],["mode"]]}->writes=true '
            '{"description":"Sửa liên hệ","returns":"contact"}',
        )
        self.assertIn("x-one-of=at least one listed group with nonempty values", render_tools([tool]))

    def test_nested_objects_arrays_and_numeric_limits_are_not_dropped(self):
        schema = {
            "type": "array", "items": {
                "type": "object", "properties": {
                    "item_id": {"type": "string", "x-ref": "food_delivery.menu"},
                    "quantity": {"type": "integer", "minimum": 1, "maximum": 100},
                    "note": {"type": "string", "x-match": "fuzzy"},
                }, "required": ["item_id", "quantity"], "additionalProperties": False,
            }, "minItems": 1, "uniqueItems": True,
        }
        self.assertEqual(
            render_schema(schema),
            '[{item_id:string{"x-ref":"food_delivery.menu"},'
            'quantity:integer{"minimum":1,"maximum":100},'
            'note?:string{"x-match":"fuzzy"}}!]{"minItems":1,"uniqueItems":true}',
        )
        self.assertEqual(render_schema({"type": "array", "items": False}), "[false]")

    def test_all_unknown_constraints_and_metadata_remain_literal_json(self):
        annotations = {
            "format": "date-time", "description": "ISO-8601, đủ giây và offset",
            "pattern": "^[0-9]+$", "minLength": 2, "maxLength": 99,
            "default": "09", "examples": ["09123456785"], "x-future": {"allow": [None, False, 0]},
        }
        self.assertEqual(render_schema({"type": "string", **annotations}), "string" + compact(annotations))
        paths = [{"namespace": "contacts.contacts", "path": ["contacts", "*", "id"],
                  "scope_argument": None}]
        metadata = {"evidence_paths": paths, "reference_scopes": [["food_delivery.menu", "restaurant_id"]],
                    "future": {"enabled": True}}
        tool = {"name": "demo.lookup", "parameters": {"type": "object", "properties": {}},
                "writes": False, **metadata}
        self.assertEqual(render_tool(tool), "demo.lookup()->writes=false " + compact(metadata))

    def test_unsupported_and_odd_schemas_use_json_fallback_without_semantic_loss(self):
        for schema in [
            True, False, {}, {"type": ["string", "null"], "enum": [None, "yes"]},
            {"oneOf": [{"type": "string"}, {"type": "integer"}]},
            {"type": "array", "prefixItems": [{"const": 1}], "unevaluatedItems": False},
            {"type": "object", "properties": {}, "required": ["undeclared"]},
            {"type": "object", "properties": {}, "required": "wrong-shape"},
        ]:
            with self.subTest(schema=schema):
                self.assertEqual(render_schema(schema), compact(schema))
        record = {"name": "future.tool", "parameters": {"oneOf": [{"type": "object"}]},
                  "kind": {"future": "kind"}, "description": "keep"}
        self.assertEqual(render_tool(record), compact(record))
        self.assertEqual(render_tool({"name": "demo.tool"}), '{"name":"demo.tool"}')

    def test_object_level_unknown_constraints_survive_compaction(self):
        constraints = {
            "additionalProperties": {"type": "string"}, "dependentRequired": {"a": ["b"]},
            "$defs": {"ref": {"type": "integer", "exclusiveMinimum": 0}},
            "allOf": [{"not": {"required": ["b"]}}],
        }
        schema = {"type": "object", "properties": {"a": {"type": "string"}}, **constraints}
        self.assertEqual(render_schema(schema), "{a?:string}" + compact(constraints))

    def test_catalog_kinds_and_conflicting_unknown_metadata_are_preserved(self):
        for kind in ["read", "write", "meta"]:
            record = {"name": "sys.tool", "kind": kind,
                      "parameters": {"type": "object", "properties": {}, "required": []}}
            self.assertEqual(render_tool(record), "sys.tool()->" + kind)
        record["writes"] = False
        self.assertEqual(render_tool(record), 'sys.tool()->writes=false {"kind":"meta"}')
        record.pop("writes")
        record["kind"] = {"future": ["kind"]}
        self.assertEqual(render_tool(record), 'sys.tool() {"kind":{"future":["kind"]}}')

    def test_text_escaping_empty_input_and_stable_order(self):
        first = {"name": "demo.one", "parameters": {"type": "object", "properties": {
            "odd? name": {"type": "string", "description": 'Line one\n"quotes" and */ <end>'},
        }, "required": ["odd? name"]}}
        second = {"name": "demo.two", "parameters": {"type": "object", "properties": {}}}
        before = deepcopy([first, second])
        rendered = render_tools(iter([first, second]))
        self.assertIn('demo.one("odd? name":string', rendered)
        self.assertIn(compact(first["parameters"]["properties"]["odd? name"]["description"]), rendered)
        self.assertLess(rendered.index("demo.one"), rendered.index("demo.two"))
        self.assertEqual([first, second], before)
        self.assertEqual(rendered, render_tools([first, second]))
        self.assertEqual(render_tools([]), "")

    def test_repetitive_schema_structure_is_materially_smaller(self):
        tools = [{"name": "demo." + str(index), "description": "Lấy danh sách", "writes": False,
                  "returns": "items[]", "evidence_paths": [], "parameters": {
                      "type": "object", "properties": {
                          "query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1},
                      }, "required": ["query"], "additionalProperties": False,
                  }} for index in range(8)]
        self.assertLess(len(render_tools(tools)), len(compact(tools)))

    def test_non_json_numeric_data_is_rejected(self):
        with self.assertRaises(ValueError):
            render_schema({"type": "number", "minimum": float("nan")})


if __name__ == "__main__":
    unittest.main()
