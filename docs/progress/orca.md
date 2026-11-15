# Tiến độ Orca: planner mất điểm ở đâu khi transcript hoàn hảo

Brief: `lifemate-orca-ctx-brief.md` (root, 15/11). Chỉ CPU, chỉ viết docs. Kết quả cuối: [CONTEXT_PLAN.md](../CONTEXT_PLAN.md).

## Bước 1 (15/11 12:10): đọc dữ liệu

- Đã làm: đọc `outputs/colab/smoke_ab_drive/smoke_ab/{base,oracle_transcripts,w_fewshot2,v3}/native.{scored,report}.json`
  và kết quả tool gold của 24 mẫu chưa trọn điểm trong `train.jsonl` (`label.tool_results`, không đọc `final_state`).
- Số: base 24.83; oracle_transcripts 49.92 (16 mẫu trọn điểm, 6 mẫu điểm lẻ, 18 mẫu 0); w_fewshot2 30.42; v3 34.58.
  Ba cờ cũ trên W với ASR: `ARG_NORMALIZE` 24.83, `DATE_TABLE` 25.06, `PROMPT_V2` 24.39.
- Ghi nhận: chưa có lượt nào chạy v3 với transcript gold, nên phần planner-only của v3 là NOT MEASURED.
- Tiếp: bảng phân loại lỗi của lượt oracle và so cặp v3 với base theo từng ID.

## Bước 2 (15/11 12:25): bảng phân loại lượt oracle và so cặp v3 với base

- Đã làm: gán mỗi mẫu chưa trọn điểm của `oracle_transcripts` vào đúng một nhóm theo lỗi đầu tiên chặn điểm
  (phân loại thủ công từ gold, predicted và kết quả tool gold); tổng các nhóm 50.083 khớp 100 − 49.917.
- Số (điểm mất / số mẫu): suy luận điều kiện trên kết quả tool 13.125 / 6; thiếu `ask_missing_info` 12.5 / 5;
  bỏ cuộc hoặc không bỏ cuộc sai 7.5 / 3; sai giá trị đối số 6.75 / 4; thừa/thiếu đối số tuỳ chọn hoặc tra cứu 5.208 / 4;
  tính ngày giờ 5.0 / 2.
- v3 so với base (cùng ASR): 9 mẫu tăng (+20.42 điểm), 5 mẫu giảm (−10.67), ròng +9.75. v3 trọn điểm 12 mẫu, base 7,
  oracle 16. 9 mẫu oracle trọn điểm mà v3 chưa trọn (0177, 0280, 0286, 0578, 0781, 0937, 1012, 1015, 1161).
- Tiếp: số liệu trên 1236 ID train (không dùng 40 ID smoke) làm căn cứ cho cách sửa ba nhóm đầu.

## Bước 3 (15/11 12:55): căn cứ trên 1236 ID train cho cách sửa ba nhóm đầu

- Đã làm: đo trên gold train ngoài 309 ID validation (script tạm, chỉ đọc `train.jsonl` và catalog).
- Thiếu `ask_missing_info`: trong 104 lần gold hỏi một ID, 53 lần kết quả tra cứu ngay trước có các bản ghi trùng tên,
  38 lần không, 13 lần hỏi ngay lời gọi đầu. Trong 954 lời gọi gold dùng một ID, chỉ 6 lần ID đó thuộc nhóm trùng tên.
  Riêng `contacts.search`: 1 kết quả thì 107/107 dùng thẳng; ≥ 2 kết quả trùng tên thì 49/51 hỏi.
  Gold hỏi một giá trị (không phải ID): 76 lần, nhiều nhất `ewallet.transfer.amount` 18, `reminders.create.time` 7.
- Bỏ cuộc: đọc tay 89 mẫu train gold rỗng: 33 do cần app/khả năng không có trên máy, 26 đổi ý ("à thôi khỏi"),
  11 điều kiện đã sai theo lịch sử, 10 trò chuyện, 5 yêu cầu không hợp lệ, 4 khác. Catalog train có 46 app,
  mỗi mẫu cài 3–18 app (trung vị 9). v3 không còn bỏ cuộc sai (`missing_all_actions` 0) nhưng làm thừa trên gold rỗng.
- Điều kiện: 474 câu train có từ điều kiện; 168 câu có bản sao Jaccard ≥ 0.5, trong đó 127 cùng chuỗi tool
  (8 câu khác một đối số enum) và 41 khác chuỗi tool. Catalog không có chú giải tiếng Việt cho enum (`water`, `internet`).
- Nhóm nhỏ: `calendar.update_event` gold luôn có `end` (27/27); `create_event` có `end` 23/34; `read_chat` không bao giờ
  có `limit` (0/11); email đọc đánh vần ("a còng") 7/1236 câu; "N phút/tiếng nữa" 3/329 câu có đối số ngày giờ.
- Tiếp: viết `docs/CONTEXT_PLAN.md`.

## Bước 4 (15/11 13:15): viết xong `docs/CONTEXT_PLAN.md`

- Đã làm: bảng mất điểm lượt oracle, so cặp v3 với base, cách sửa cho ba nhóm đầu (chữ prompt, chỗ đặt trong code,
  có cần train lại không, trần trên), kế hoạch xếp hạng tới 12:00 ngày 16/11.
- Số chính: ba nhóm đầu mất 13.13 + 12.50 + 7.50 = 33.13 điểm trên lượt oracle. Guard trùng tên không cần train (trần 7.5);
  dòng app không cài và ví dụ kèm bằng chứng cần một lượt QLoRA (trần 5.0 và 13.13).
- Chưa đo: điểm của mọi đề xuất, v3 với transcript gold, token và độ trễ thêm.
- Tiếp (nếu root duyệt): code guard trùng tên và các thay đổi prompt sau cờ; lượt GPU đầu tiên nên là v3 với `--train-transcripts`.

# Việc v4 (brief `lifemate-orca-v4-brief.md`, root duyệt 15/11 14:55): ba cờ, mặc định tắt

## V4-1 (15/11 15:00): `LIFEMATE_DUP_GUARD=1` xong

- Đã làm: trong `src/agent/runner.py`, ngay trước guard `registry.authorize`: nếu lời gọi dùng một ID (đối số có `x-ref`)
  thuộc bản ghi trùng `name`/`title` với bản ghi khác trong cùng danh sách kết quả của lượt hiện tại, thay bằng
  `sys.ask_missing_info{"tool_name": T, "params": [p]}`. Chỉ khi mẫu có tool `sys.ask_missing_info`; không đụng lời gọi `sys.*`.
- File: `src/agent/runner.py`, `tests/test_orca_v4_dup_guard.py`, `scripts/orca_v4_measure.py`.
- Đo trên gold của 1236 ID train (`python -B -m scripts.orca_v4_measure dup --data-root <kit>`, schema lấy từ registry):
  979 lời gọi gold dùng một ID, guard đổi 6; 112 lần gold hỏi một ID, guard tái tạo đúng 53 từ một lời gọi thẳng;
  68 lần gold hỏi một giá trị (guard không áp dụng). Điểm: NOT MEASURED.
- Test: `python -B -m pytest -q -p no:cacheprovider` → 316 passed, 1 skipped; `python -B -m src.inference mock --samples 40` → 40/40.
- Tiếp: `LIFEMATE_CONTEXT_V4` (app không cài, ví dụ kèm kết quả tra cứu, chú giải enum).
