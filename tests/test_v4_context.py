"""LIFEMATE_CONTEXT_V4: uninstalled apps, examples with lookup results, Vietnamese enum glosses."""

from copy import deepcopy
import json
from pathlib import Path
from time import monotonic

import pytest

from scripts.build_sft import conversation
from src.agent import fewshot
from src.agent.native_planner import _SYSTEM, build_messages, tool_definitions
from src.agent.planner import PlannerInput
from src.inference.pipeline import _sample_from_row

BILL_TYPE = {"type": "string", "enum": ["electricity", "water", "x_unknown"]}
TOOLS = [
    {"name": "bill_payment.list_bills", "description": "Hoá đơn", "writes": False,
     "parameters": {"type": "object", "properties": {"type": BILL_TYPE}, "additionalProperties": False}},
    {"name": "bill_payment.pay", "description": "Trả", "writes": True,
     "parameters": {"type": "object", "additionalProperties": False, "properties": {
         "bill_id": {"type": "string", "description": "ID hoá đơn", "x-ref": "bill_payment.bills"},
         "source": {"type": "string", "enum": ["bank", "ewallet"]},
         "tags": {"type": "array", "items": {"type": "string", "enum": ["business"]}},
         "topic": {"type": "string", "enum": ["business"], "description": "Chủ đề."},
         "code": {"type": "string", "enum": ["zz1", "zz2"], "description": "Mã"}}}},
]
POOL = [
    {"id": "train_0001", "transcript": "Trả tiền điện bằng ví, ví không đủ thì trừ ngân hàng", "apps": ["bill_payment", "sys"],
     "calls": [{"name": "bill_payment.list_bills", "arguments": {}},
               {"name": "bill_payment.pay", "arguments": {"bill_id": "bill_02", "source": "bank"}}],
     "results": ['{"ok":true,"bills":[{"id":"bill_02","type":"electricity","amount":845000}]}', None]},
    {"id": "train_0002", "transcript": "Chúc cuối tuần vui nha", "apps": ["sys"], "calls": [], "results": []},
]
ROW = {"id": "train_0900", "audio": "audio/train_0900/train_0900.wav", "audio_text": "Trả tiền điện bằng ví giúp tôi",
       "context": {"current_time": "2026-11-13T10:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh",
                   "installed_apps": {"bill_payment": {"tools": ["bill_payment.list_bills", "bill_payment.pay"]}}},
       "history": []}
OLD_EXAMPLES = ("Ví dụ yêu cầu tương tự và các lệnh đã dùng (chỉ tham khảo cách chọn tool và thứ tự; "
                "giá trị và ID phải lấy từ yêu cầu hiện tại và kết quả tool):\n"
                '- "Trả tiền điện bằng ví, ví không đủ thì trừ ngân hàng" → bill_payment.list_bills{}; '
                'bill_payment.pay{"bill_id":"bill_02","source":"bank"}\n'
                '- "Chúc cuối tuần vui nha" → không gọi tool nào')


class Kit:
    directory = Path("kit") / "train"
    app_names = {"bill_payment": "Hoá đơn", "email": "Email", "library": "Thư viện", "sys": "Hệ thống"}

    def audio_path(self, sample_id, relative_path, *, history):
        return self.directory / relative_path


@pytest.fixture
def pool(monkeypatch, tmp_path):
    path = tmp_path / "pool.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in POOL), encoding="utf-8")
    monkeypatch.setattr(fewshot, "NATIVE_POOL", path)
    monkeypatch.setenv("LIFEMATE_NATIVE_FEWSHOT_K", "2")
    monkeypatch.delenv("LIFEMATE_FEWSHOT_SPLIT", raising=False)
    monkeypatch.delenv("LIFEMATE_CONTEXT_V4", raising=False)
    monkeypatch.delenv("LIFEMATE_DATE_TABLE", raising=False)
    monkeypatch.delenv("LIFEMATE_PROMPT_V2", raising=False)


def request():
    sample = _sample_from_row(Kit(), ROW, None, monotonic() + 60, control_noop=False)
    return PlannerInput(sample, deepcopy(TOOLS), (), None)


@pytest.mark.parametrize("flag", [None, "0", "true"])
def test_flag_off_keeps_the_old_prompt_and_tool_definitions(pool, monkeypatch, flag):
    if flag is not None:
        monkeypatch.setenv("LIFEMATE_CONTEXT_V4", flag)
    planned = request()
    assert "apps" not in planned.sample.context
    system = build_messages(planned)[0]["content"]
    assert system == (f"{_SYSTEM}\n\nThời gian hiện tại: 2026-11-13T10:00:00+07:00 (Asia/Ho_Chi_Minh)\n"
                      f"Vị trí: null\nHồ sơ người dùng: null\n\n{OLD_EXAMPLES}")
    definitions = tool_definitions(planned.tools)
    assert [item["function"]["parameters"] for item in definitions] == [tool["parameters"] for tool in TOOLS]
    assert definitions[0]["function"]["parameters"] is planned.tools[0]["parameters"]


def test_flag_on_names_the_missing_apps_and_shows_lookup_results_in_examples(pool, monkeypatch):
    plain = build_messages(request())[0]["content"]
    monkeypatch.setenv("LIFEMATE_CONTEXT_V4", "1")
    planned = request()
    assert planned.sample.context["apps"] == {"installed": ["Hoá đơn (bill_payment)"],
                                              "missing": ["Email (email)", "Thư viện (library)"]}
    system = build_messages(planned)[0]["content"]
    head = plain[:plain.index("\n\nVí dụ yêu cầu")]
    assert system.startswith(head + "\n\nỨng dụng đã cài: Hoá đơn (bill_payment)\n"
                             "Ứng dụng KHÔNG cài trên máy này: Email (email), Thư viện (library)\n"
                             "Việc nào cần một ứng dụng không cài thì bỏ qua việc đó: không thay bằng ứng dụng khác, "
                             "không hỏi lại.\n\nVí dụ yêu cầu")
    assert system.endswith(
        '- "Trả tiền điện bằng ví, ví không đủ thì trừ ngân hàng"\n'
        '  bill_payment.list_bills{} ⇒ {"ok":true,"bills":[{"id":"bill_02","type":"electricity","amount":845000}]}\n'
        '  bill_payment.pay{"bill_id":"bill_02","source":"bank"}\n'
        '- "Chúc cuối tuần vui nha" → không gọi tool nào')


def test_flag_on_glosses_known_enum_values_without_touching_the_schema_it_was_given(pool, monkeypatch):
    monkeypatch.setenv("LIFEMATE_CONTEXT_V4", "1")
    tools = deepcopy(TOOLS)
    lookup, pay = (item["function"]["parameters"]["properties"] for item in tool_definitions(tools))
    assert tools == TOOLS  # copies, never an edit of the registry's schema
    assert lookup["type"] == {**BILL_TYPE, "description": "Giá trị: electricity = điện; water = nước"}
    assert pay["source"]["description"] == "Giá trị: bank = ngân hàng; ewallet = ví điện tử"
    assert pay["topic"]["description"] == "Chủ đề. Giá trị: business = kinh doanh"
    assert pay["tags"]["items"]["description"] == "Giá trị: business = thương gia"
    assert pay["code"] == TOOLS[1]["parameters"]["properties"]["code"]  # unknown values: nothing added
    assert pay["bill_id"] == TOOLS[1]["parameters"]["properties"]["bill_id"]


@pytest.mark.parametrize("flag", [None, "1"])
def test_training_data_and_inference_build_the_same_prompt_and_tools(pool, monkeypatch, flag):
    if flag:
        monkeypatch.setenv("LIFEMATE_CONTEXT_V4", flag)
    planned = request()
    calls = [{"name": "bill_payment.list_bills", "arguments": {}}]
    item = conversation(planned.sample, planned.tools, calls, [{"ok": True, "bills": []}])
    assert item["messages"][:2] == build_messages(planned) and item["tools"] == tool_definitions(planned.tools)
    assert ("KHÔNG cài" in item["messages"][0]["content"]) is bool(flag)
    assert ("Giá trị:" in json.dumps(item["tools"], ensure_ascii=False)) is bool(flag)


def test_results_count_against_the_larger_budget_only_when_asked_for(pool):
    examples = fewshot.native_examples()
    sample = request().sample
    names = {tool["name"] for tool in TOOLS}
    assert all("results" not in item for item in examples.select(sample, names, 2))
    assert "results" in examples.select(sample, names, 2, fewshot.RESULTS_BUDGET, results=True)[0]
    small = len(json.dumps({"request": POOL[0]["transcript"], "calls": POOL[0]["calls"]}, ensure_ascii=False,
                           separators=(",", ":")))
    assert examples.select(sample, names, 1, small)[0]["request"] == POOL[0]["transcript"]
    assert examples.select(sample, names, 1, small, results=True)[0]["request"] == POOL[1]["transcript"]


def test_shipped_pool_stores_one_capped_result_slot_per_call():
    with fewshot.NATIVE_POOL.open(encoding="utf-8") as lines:
        rows = [json.loads(line) for line in lines if line.strip()]
    assert rows and all(len(row["results"]) == len(row["calls"]) for row in rows)
    stored = [text for row in rows for text in row["results"] if text is not None]
    assert stored and max(map(len, stored)) <= fewshot.RESULT_CAP + 1
    assert all(text is None for row in rows for call, text in zip(row["calls"], row["results"])
               if call["name"].startswith("sys."))
