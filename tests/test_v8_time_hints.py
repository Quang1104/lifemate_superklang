"""LIFEMATE_TIME_HINTS_V8: dates and times stated in the request, computed from the sample clock, in the prompt."""

import pytest

from scripts.build_sft import conversation
from src.agent.native_planner import build_messages
from src.agent.planner import PlannerInput
from src.agent.time_hints_v8 import render_time_hints, time_hints
from src.inference.types import Sample

ZONE = "Asia/Ho_Chi_Minh"
TOOLS = [{"name": "reminders.create", "description": "Nhắc", "writes": True,
          "parameters": {"type": "object", "properties": {"time": {"type": "string"}}}}]


def context(now):
    return {"current_time": now, "timezone": ZONE}


# The four smoke40 requests whose date or time the planner got wrong and whose value the request states.
@pytest.mark.parametrize("now,text,expected", [
    ("2026-11-05T13:10:00+07:00", "45 phút nữa nhắc chị gọi lại cho khách nhé.",
     [("45 phút nữa", "2026-11-05T13:55:00+07:00")]),
    ("2026-10-01T09:00:00+07:00", "Chủ nhật này em có buổi chạy địa hình trên Ba Vì đấy, anh xem trời có mưa không.",
     [("chủ nhật này", "2026-10-04")]),
    ("2026-10-01T09:00:00+07:00", "Mở phòng họp online tên là họp điều phối kho, ba giờ chiều hôm hai mươi hai tháng mười nhé",
     [("ba giờ chiều hôm hai mươi hai tháng mười", "2026-10-22T15:00:00+07:00")]),
    ("2026-09-25T09:00:00+07:00", "Ghi buổi yoga bốn mươi lăm phút em tập từ sáu rưỡi sáng nay nha, rồi bảy giờ tối mai nhắc em uống magie.",
     [("sáu rưỡi sáng nay", "2026-09-25T06:30:00+07:00"), ("bảy giờ tối mai", "2026-09-26T19:00:00+07:00")]),
    ("2026-11-22T14:40:00+07:00", "Mai nếu mưa thì nhắc tôi mang ô", [("mai", "2026-11-23")]),
])
def test_stated_dates_and_times_are_resolved_against_the_sample_clock(now, text, expected):
    assert time_hints(text, context(now)) == expected


@pytest.mark.parametrize("text", [
    "Ghi lịch cô ra tiệm vàng trước họp tổ hai tiếng nghe con",  # relative to another event: not in the request
    "Chuyển ba trăm rưỡi cho Lan nhé", "Bật điều hoà hai mươi sáu độ", "Tắt đèn tầng hai đi",
    "Cái ni để đó nghe", "Gọi cho Minh giúp tôi", "",
])
def test_requests_without_a_resolvable_time_get_no_hint(text):
    assert time_hints(text, context("2026-10-18T13:40:00+07:00")) == []
    assert render_time_hints(text, context("2026-10-18T13:40:00+07:00")) == ""


def test_no_hint_without_a_usable_clock():
    assert time_hints("45 phút nữa nhắc chị", {}) == []
    assert time_hints("45 phút nữa nhắc chị", {"current_time": "hôm nay", "timezone": ZONE}) == []


def request(text):
    sample = Sample("train_0900", text, frozenset({"reminders"}), context=context("2026-11-05T13:10:00+07:00"))
    return PlannerInput(sample, TOOLS, (), None)


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in ("LIFEMATE_TIME_HINTS_V8", "LIFEMATE_DATE_TABLE", "LIFEMATE_CONTEXT_V4", "LIFEMATE_NATIVE_FEWSHOT_K"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("flag", [None, "0", "true"])
def test_flag_off_leaves_the_prompt_unchanged(monkeypatch, flag):
    plain = build_messages(request("45 phút nữa nhắc chị gọi lại cho khách nhé."))
    if flag is not None:
        monkeypatch.setenv("LIFEMATE_TIME_HINTS_V8", flag)
    assert build_messages(request("45 phút nữa nhắc chị gọi lại cho khách nhé.")) == plain
    assert "Mốc thời gian" not in plain[0]["content"]


def test_flag_on_appends_the_hints_to_the_system_prompt_only_when_there_are_any(monkeypatch):
    plain = build_messages(request("45 phút nữa nhắc chị gọi lại cho khách nhé."))
    monkeypatch.setenv("LIFEMATE_TIME_HINTS_V8", "1")
    hinted = build_messages(request("45 phút nữa nhắc chị gọi lại cho khách nhé."))
    assert hinted[1:] == plain[1:]
    assert hinted[0]["content"] == plain[0]["content"] + (
        "\n\nMốc thời gian đã tính sẵn từ câu hiện tại (dùng đúng giá trị này nếu việc cần ngày giờ đó):\n"
        '- "45 phút nữa" = 2026-11-05T13:55:00+07:00')
    assert build_messages(request("Gọi cho Minh giúp tôi")) == [
        {**plain[0]}, {"role": "user", "content": "Gọi cho Minh giúp tôi"}]


def test_training_rows_and_inference_build_the_same_hinted_prompt(monkeypatch):
    monkeypatch.setenv("LIFEMATE_TIME_HINTS_V8", "1")
    planned = request("45 phút nữa nhắc chị gọi lại cho khách nhé.")
    row = conversation(planned.sample, planned.tools, [], [])
    assert row["messages"][:2] == build_messages(planned) and "13:55" in row["messages"][0]["content"]
