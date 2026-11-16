"""Calendar anchors and the opt-in native prompt contract (stdlib tests)."""

import hashlib
import os
import unittest
from unittest.mock import patch

from src.agent.native_planner import build_messages, render_date_table
from src.agent.planner import PlannerInput
from src.inference.types import Sample


def _request(current_time="2026-11-01T00:30:00+07:00"):
    sample = Sample(
        "s", "Nhắc tôi sáng mai", frozenset(), (),
        context={"current_time": current_time, "timezone": "Asia/Ho_Chi_Minh",
                 "location": {"city": "Huế"}, "user_profile": {"name": "An"}},
    )
    return PlannerInput(sample, [], ())


class DateTableTests(unittest.TestCase):
    def test_sunday_month_boundary_has_all_current_and_next_week_rows(self):
        self.assertEqual(render_date_table("2026-11-01T00:30:00+07:00").splitlines(), [
            "Bảng ngày (tuần từ thứ Hai đến Chủ nhật):",
            "hôm nay: Chủ nhật 2026-11-01",
            "mai: thứ Hai 2026-11-02",
            "ngày kia: thứ Ba 2026-11-03",
            "hôm qua: thứ Bảy 2026-10-31",
            "thứ Hai này (tuần này): 2026-10-26",
            "thứ Ba này (tuần này): 2026-10-27",
            "thứ Tư này (tuần này): 2026-10-28",
            "thứ Năm này (tuần này): 2026-10-29",
            "thứ Sáu này (tuần này): 2026-10-30",
            "thứ Bảy này (tuần này): 2026-10-31",
            "Chủ nhật này (tuần này): 2026-11-01",
            "thứ Hai tuần sau: 2026-11-02",
            "thứ Ba tuần sau: 2026-11-03",
            "thứ Tư tuần sau: 2026-11-04",
            "thứ Năm tuần sau: 2026-11-05",
            "thứ Sáu tuần sau: 2026-11-06",
            "thứ Bảy tuần sau: 2026-11-07",
            "Chủ nhật tuần sau: 2026-11-08",
        ])

    def test_calendar_rollover_uses_input_local_date(self):
        cases = [
            ("2026-11-02T00:01:00+07:00", [
                "hôm qua: Chủ nhật 2026-11-01",
                "thứ Hai này (tuần này): 2026-11-02", "Chủ nhật tuần sau: 2026-11-15"]),
            ("2026-12-31T23:59:00-10:00", [
                "hôm nay: thứ Năm 2026-12-31", "mai: thứ Sáu 2027-01-01",
                "ngày kia: thứ Bảy 2027-01-02", "thứ Hai tuần sau: 2027-01-04"]),
            ("2024-02-28T23:59:00+14:00", [
                "hôm nay: thứ Tư 2024-02-28", "mai: thứ Năm 2024-02-29",
                "ngày kia: thứ Sáu 2024-03-01"]),
            ("2026-11-01T00:30:00Z", ["hôm nay: Chủ nhật 2026-11-01"]),
        ]
        for anchor, expected in cases:
            with self.subTest(anchor=anchor):
                rows = render_date_table(anchor).splitlines()
                for row in expected:
                    self.assertIn(row, rows)

    def test_no_table_when_anchor_cannot_be_used_without_guessing(self):
        for anchor in (None, "", "bad", "2026-11-01", "2026-11-01T00:30:00",
                       "2026-02-30T00:30:00+07:00", "9999-12-31T00:30:00+07:00", 123):
            with self.subTest(anchor=anchor):
                self.assertEqual(render_date_table(anchor), "")

    def test_flag_off_preserves_pre_change_system_prompt_bytes(self):
        for flag in (None, "0", "", "true", "01", " 1"):
            with self.subTest(flag=flag), patch.dict(os.environ):
                if flag is None:
                    os.environ.pop("LIFEMATE_DATE_TABLE", None)
                else:
                    os.environ["LIFEMATE_DATE_TABLE"] = flag
                encoded = build_messages(_request())[0]["content"].encode("utf-8")
                # Captured with this fixed request before the edit on 2026-11-14.
                self.assertEqual(len(encoded), 1690)
                self.assertEqual(hashlib.sha256(encoded).hexdigest(),
                                 "7383af6ad28e090717f6d37e8e71574f26503114527071e6703ea9b13d14b70e")

    def test_flag_on_only_appends_to_system_prompt(self):
        with patch.dict(os.environ):
            os.environ.pop("LIFEMATE_DATE_TABLE", None)
            original = build_messages(_request())
            os.environ["LIFEMATE_DATE_TABLE"] = "1"
            enabled = build_messages(_request())
        self.assertEqual(enabled[0]["content"],
                         original[0]["content"] + "\n\n"
                         + render_date_table("2026-11-01T00:30:00+07:00"))
        self.assertEqual(enabled[1:], original[1:])

    def test_invalid_anchor_does_not_change_prompt_when_enabled(self):
        with patch.dict(os.environ):
            os.environ.pop("LIFEMATE_DATE_TABLE", None)
            original = build_messages(_request(None))
            os.environ["LIFEMATE_DATE_TABLE"] = "1"
            self.assertEqual(build_messages(_request(None)), original)


if __name__ == "__main__":
    unittest.main()
