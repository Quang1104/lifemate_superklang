"""Pure fixture tests for the explicit TRAIN exposure audit (no kit access)."""
import copy
import hashlib
from pathlib import Path
import unittest

from research.unseen_clean import build_manifest, checked_path, manifest_ids


class CleanSplitAuditTests(unittest.TestCase):
    def setUp(self):
        self.tool = "fixture_holdout.lookup"
        self.rows = [
            {"id": "fixture_keep", "context": {"installed_apps": {"fixture_holdout": {"tools": [self.tool]}}},
             "history": [{"role": "assistant", "text": "Used fixture_holdout.lookup before.", "tool_calls": []}],
             "label": {"tool_calls": []}},
            {"id": "fixture_validation", "context": {"installed_apps": {}}, "history": [],
             "label": {"tool_calls": []}},
            {"id": "fixture_holdout", "context": {"installed_apps": {}}, "history": [],
             "label": {"tool_calls": [{"name": self.tool, "arguments": {}}]}},
        ]
        self.catalog = {"apps": [{"id": "fixture_holdout", "tools": [{"name": self.tool}]}]}
        self.validation = {"source_split": "train", "validation_ids": ["fixture_validation"]}
        self.leave_app = {"source_split": "train", "validation_ids": ["fixture_holdout"],
                          "holdout_apps": ["fixture_holdout"]}

    def build(self):
        return build_manifest(self.rows, self.catalog, self.validation, self.leave_app, hashlib.sha256(b"fixture").hexdigest())

    def test_union_exclusion_and_schema_history_exposure(self):
        result = self.build()
        self.assertEqual(result["train_ids"], ["fixture_keep"])
        self.assertEqual(result["validation_ids"], ["fixture_holdout", "fixture_validation"])
        self.assertEqual(result["train_count"], 1)
        self.assertEqual(result["validation_count"], 2)
        audit = result["exposure_audit"]
        self.assertEqual(audit["exposed_tools"], [self.tool])
        self.assertEqual(audit["exposed_conversation_count"], 1)
        self.assertEqual(audit["per_tool"][self.tool], {
            "installed_tool_lists_samples": 1, "history_anywhere_samples": 1,
            "history_calls_samples": 0, "current_gold_calls_samples": 0})

    def test_current_and_history_calls_are_audited_separately(self):
        self.rows[0]["history"].append({"tool_calls": [{"name": self.tool, "arguments": {}}]})
        self.rows[0]["label"]["tool_calls"] = [{"name": self.tool, "arguments": {}}]
        audit = self.build()["exposure_audit"]
        self.assertEqual(audit["history_call_exposed_tool_count"], 1)
        self.assertEqual(audit["current_gold_call_exposed_tool_count"], 1)
        self.assertEqual(audit["per_tool"][self.tool]["history_anywhere_samples"], 1)

    def test_mentions_require_a_complete_tool_name(self):
        self.rows[0]["context"]["installed_apps"] = {}
        self.rows[0]["history"] = [{"text": "fixture_holdout.lookup_extra"}]
        self.assertEqual(self.build()["exposure_audit"]["exposed_tool_count"], 0)

    def test_inputs_are_not_mutated_and_result_is_deterministic(self):
        before = copy.deepcopy(self.rows)
        self.assertEqual(self.build(), self.build())
        self.assertEqual(self.rows, before)

    def test_invalid_ids_or_nontrain_manifests_are_rejected(self):
        for manifest in ({"source_split": "public_test", "validation_ids": []},
                         {"source_split": "train", "validation_ids": ["x", "x"]}):
            with self.assertRaises(ValueError):
                manifest_ids(manifest)
        self.leave_app["validation_ids"] = ["fixture_validation"]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.build()
        self.leave_app["validation_ids"] = ["fixture_absent"]
        with self.assertRaisesRegex(ValueError, "absent"):
            self.build()

    def test_forbidden_paths_are_rejected_without_opening(self):
        for value in (Path("fixture_environment") / "train.jsonl", Path("state.json"),
                      Path("fixture_environment") / ".." / "train.jsonl"):
            with self.assertRaisesRegex(ValueError, "Forbidden"):
                checked_path(value)


if __name__ == "__main__":
    unittest.main()
