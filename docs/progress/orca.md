# Tiến độ Orca: planner mất điểm ở đâu khi transcript hoàn hảo

Brief: `lifemate-orca-ctx-brief.md` (root, 15/11). Chỉ CPU, chỉ viết docs. Kết quả cuối: [CONTEXT_PLAN.md](../CONTEXT_PLAN.md).

## Bước 1 (15/11 12:10): đọc dữ liệu

- Đã làm: đọc `outputs/colab/smoke_ab_drive/smoke_ab/{base,oracle_transcripts,w_fewshot2,v3}/native.{scored,report}.json`
  và kết quả tool gold của 24 mẫu chưa trọn điểm trong `train.jsonl` (`label.tool_results`, không đọc `final_state`).
- Số: base 24.83; oracle_transcripts 49.92 (16 mẫu trọn điểm, 6 mẫu điểm lẻ, 18 mẫu 0); w_fewshot2 30.42; v3 34.58.
  Ba cờ cũ trên W với ASR: `ARG_NORMALIZE` 24.83, `DATE_TABLE` 25.06, `PROMPT_V2` 24.39.
- Ghi nhận: chưa có lượt nào chạy v3 với transcript gold, nên phần planner-only của v3 là NOT MEASURED.
- Tiếp: bảng phân loại lỗi của lượt oracle và so cặp v3 với base theo từng ID.
