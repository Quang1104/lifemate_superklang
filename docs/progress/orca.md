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
