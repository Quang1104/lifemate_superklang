"""Snapshot and scope checks for the opt-in native prompt candidate."""

import hashlib
import os
import unittest
from unittest.mock import patch

from src.agent.native_planner import _SYSTEM, _SYSTEM_V2, build_messages, render_date_table
from src.agent.planner import Observation, PlannerInput
from src.inference.types import Sample


class PromptV2Tests(unittest.TestCase):
    def setUp(self):
        self.sample = Sample(
            "fixture", "Gọi cho Minh rồi đặt báo thức.", frozenset({"contacts", "alarm"}),
            context={"current_time": "2026-11-15T01:30:00+07:00", "timezone": "Asia/Ho_Chi_Minh"})
        self.request = PlannerInput(self.sample, [], (), None)

    def messages(self, flag=None, date_table=None):
        with patch.dict(os.environ, {}, clear=True):
            if flag is not None:
                os.environ["LIFEMATE_PROMPT_V2"] = flag
            if date_table is not None:
                os.environ["LIFEMATE_DATE_TABLE"] = date_table
            return build_messages(self.request)

    def test_baseline_prompt_snapshot_is_byte_identical(self):
        # Digest measured from the original _SYSTEM before this change.
        self.assertEqual(hashlib.sha256(_SYSTEM.encode("utf-8")).hexdigest(),
                         "8fed4b1839aff7cfedf2c787f20559626e43bf304a9be79c37239835d9b2be95")
        expected = (_SYSTEM + "\n\nThời gian hiện tại: 2026-11-15T01:30:00+07:00 (Asia/Ho_Chi_Minh)\n"
                    "Vị trí: null\nHồ sơ người dùng: null")
        for value in (None, "0", "true", "2", ""):
            with self.subTest(flag=value):
                self.assertEqual(self.messages(value)[0]["content"].encode(), expected.encode())

    def test_candidate_changes_only_four_rules(self):
        before, after = _SYSTEM.splitlines(), _SYSTEM_V2.splitlines()
        self.assertEqual(len(before), len(after))
        changed = [old.partition(" ")[0] for old, new in zip(before, after) if old != new]
        self.assertEqual(changed, ["1.", "7.", "8.", "9."])
        candidate = self.messages("1")
        self.assertTrue(candidate[0]["content"].startswith(_SYSTEM_V2 + "\n\n"))
        self.assertEqual(candidate[1:], self.messages()[1:])
        for text in ("Tra cứu trước", "nhiều kết quả", "thiếu giá trị bắt buộc",
                     "sys.ask_missing_info với tool_name và params", "kể cả tool tra cứu",
                     "TẤT CẢ", "không dừng ngay sau bước tra cứu"):
            self.assertIn(text, _SYSTEM_V2)

    def test_date_table_flag_is_independent(self):
        table = render_date_table(self.sample.context["current_time"])
        self.assertNotIn(table, self.messages("1")[0]["content"])
        self.assertEqual(self.messages("1", "1")[0]["content"],
                         self.messages("1")[0]["content"] + "\n\n" + table)
        self.assertEqual(self.messages(None, "1")[0]["content"],
                         self.messages()[0]["content"] + "\n\n" + table)

    def test_history_and_observations_unchanged(self):
        self.request = PlannerInput(self.sample, [], (
            Observation("untrusted_tool_result", "contacts.search", {"ok": True}, {"query": "Minh"}),), None)
        self.assertEqual(self.messages("1")[1:], self.messages()[1:])


if __name__ == "__main__":
    unittest.main()
