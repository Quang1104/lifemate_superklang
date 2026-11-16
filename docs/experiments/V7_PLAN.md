# v7: thêm mẫu SFT từ lượt lịch sử (BRAINSTORM_70 mục 1)

Viết 16/11 00:30, chỉ CPU. Điểm v7: NOT MEASURED.

## Thay đổi

`scripts/build_sft.py --history-turns F` (mặc định 0; tắt thì file SFT trùng từng byte với trước):

- Với một phần F các lượt trợ lý trong lịch sử (chọn bằng hash ổn định của `"<id>#history<k>"`), dựng thêm một dòng:
  lịch sử = các lượt trước lượt user đứng ngay trước nó; câu nói = lượt user đó (ASR qua `--transcripts`, qua cùng lớp
  `asr_fix.wrapped` như lúc chạy); đích = lời gọi đã ghi kèm kết quả đã ghi (không cần ToolEnv); lượt không gọi tool có
  đích là câu "không thực hiện thao tác nào". ID dòng là `<id>_h<k>`; mẫu vẫn giữ ID gốc nên ví dụ truy hồi loại chính mẫu.
- Bỏ lượt có lời gọi tới tool không cài, sai schema, hoặc thiếu kết quả; bỏ lượt thiếu transcript.
- Masking, đổi tên và `LIFEMATE_FEWSHOT_GATE` áp cho từng dòng như dòng gốc (cùng hàm, cùng `build_messages`).

## Số đo CPU (transcript PhoWhisper-medium cũ `outputs/asr_train.jsonl`, CONTEXT_V4 + SMALL_FIXES + GATE, đổi tên 0.3)

| Build | Dòng | Dòng lịch sử | Lịch sử không gọi tool | Đổi tên | Có ví dụ | Ký tự p50 / max |
|---|---:|---:|---:|---:|---:|---|
| `--limit 50`, history 0.5 | 65 | 15 | 2 | 22 | 60 | 24,060 / 34,079 |
| đủ TRAIN, history 0 (= v6) | 1236 | 0 | 0 | 398 | 994 | 20,520 / 38,003 |
| đủ TRAIN, history 0.5 | 2621 | 1385 | 204 | 830 | 1789 | 20,884 / 38,003 |

- Dòng lịch sử: ký tự p50 21,181, p95 28,760, max 35,435 (dòng cuối: 20,521 / 31,233 / 38,004). Token: NOT MEASURED (không có
  tokenizer Qwen trong máy).
- Kiểm 1385 dòng lịch sử: câu nói đúng là ASR của lượt user đó 1385/1385; đích khớp lời gọi đã ghi (616/616 dòng không đổi tên);
  mọi lời gọi đích nằm trong danh sách tool; một kết quả mỗi lời gọi; câu kết khớp có/không gọi tool; 0 dòng lộ lượt sau hay
  chính mẫu nguồn trong ví dụ; 1236 dòng cuối trùng bản v6. 0 lượt bị bỏ vì lời gọi lỗi.
- `--history-turns 0` trùng byte bản trước ở 4 cấu hình (mặc định, K=2, mask 0.3, cấu hình v6).

## Rủi ro

- Tỷ lệ mẫu có hỏi thiếu giảm: 180/1236 (14.6%) → 186/2621 (7.1%); chỉ 6/1385 dòng lịch sử có hỏi. Public vốn hỏi thiếu ít
  (PUBLIC_GAP). Đo tỷ lệ hỏi của v7 trên smoke/VAL so với v4.
- Thời gian train khoảng 2.1 lần v4 (ước lượng theo số dòng: 154 bước ≈ 43 phút → khoảng 90 phút trên A100); số token có thể
  vượt `--max-tokens` của `colab_v7.sh`: xem dòng `rows` của `train_qlora.py`.
- Chất lượng lượt lịch sử như nhãn gold là giả định; mới kiểm được schema và sự nhất quán.
