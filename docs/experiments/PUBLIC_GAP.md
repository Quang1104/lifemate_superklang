# Vì sao public thấp hơn smoke40, và nên đổi gì cho v5/v6

Viết 15/11 20:20, chỉ CPU. Chỉ dùng đầu vào public (context, lịch sử, app đã cài), catalog và **dự đoán của chính mình**
(`outputs/colab/public_drive/public/{v3,v3_asrft}/submission.jsonl`); không có và không suy ra nhãn public. Tiến độ:
[progress/orca.md](progress/orca.md), mục G-1 → G-3.

## Khoảng cách

- Public / smoke40: v3 + ASR cũ 27.98 / 34.58 (−6.6); v3 + ASR fine-tune 42.25 / 51.73 (−9.5).
- smoke40 có sai số chuẩn khoảng 6.6 điểm, nên một phần khoảng cách có thể là nhiễu. Sai số chuẩn trên 634 mẫu public:
  khoảng 1.7 điểm (ước lượng, giả sử độ lệch chuẩn từng mẫu như smoke40).
- smoke40 không có tool chưa thấy; trên public 410/634 mẫu có cài app chỉ có ở public và 285 mẫu (45.0%) có gọi tool loại đó.
  Phần khoảng cách do tool chưa thấy: NOT MEASURED (cần nhãn).

## Tín hiệu (v3 + ASR fine-tune, so với TRAIN gold)

| Tín hiệu | TRAIN gold | Public, app của train | Public, app chỉ có ở public |
|---|---|---|---|
| Có `ask_missing_info` cho app mà mẫu gọi hoặc hỏi tới | 180/2012 (8.9%) | 52/709 (7.3%) | 8/317 (2.5%) |
| Lời gọi cuối là lệnh đọc (theo mẫu) | 105/1236 (8.5%) | 38/634 | 47/634 |
| Lời gọi app có kèm lệnh ghi của app đó | 1216/1731 (70%) | 420/630 (67%) | 216/302 (72%) |

- Dừng sau lệnh đọc, theo nhóm: có app public-only mà lịch sử chưa dùng 23.5% (226 mẫu); lịch sử đã dùng 9.8% (184);
  không có app public-only 6.2% (224). Lịch sử có lời gọi tool public-only là ví dụ duy nhất model thấy cho tool đó.
- Hỏi thiếu theo nhóm: không có app public-only 17.0%; có app public-only 5.4%; lịch sử đã dùng app public-only 2.2%.
- Thay bằng app train cùng loại: 67 trong 472 cặp (mẫu, app public-only đã cài nhưng không được gọi) có lời gọi tới một app
  train "anh em" (grocery → shopping_list, train_bus_tickets → flights, ...). Đúng hay sai: NOT MEASURED.
- Tham số không mô tả: `stocks.*.symbol` nhận tên công ty thay cho mã ở 5/40 giá trị. Nhưng catalog train cũng có 79/246
  tham số không mô tả (public 68/219), nên đây không phải điểm khác riêng của public.
- Không lệch: 0 lời gọi tool không cài, 0 lời gọi lặp; tỷ lệ kèm lệnh ghi như TRAIN (bảng trên).

"Trần" bên dưới = số mẫu public mang tín hiệu × 100/634 (0.158 điểm mỗi mẫu), tức mức tối đa nếu mọi mẫu đó đang 0 điểm và
thành trọn điểm. Không phải dự báo; không có nhãn để biết các mẫu đó sai thật.

## Thay đổi cho lượt train kế tiếp, xếp hạng

1. **Giữ masking 30% của v5 (đã sẵn, `scripts/build_sft.py --mask-fraction`).** Nhắm vào hai tín hiệu trên: model gần như
   không hỏi thiếu cho tool lạ (8 lần thay vì khoảng 23 nếu theo tỷ lệ 7.3%) và dừng sau lệnh đọc khi không có ví dụ.
   Trong bản build masking đủ TRAIN (398 mẫu bị mask), 61 mẫu có lượt đích hỏi `ask_missing_info` về một tool đã mask.
   Trần: 15 mẫu thiếu hỏi + khoảng 31 mẫu dừng sau đọc vượt mức TRAIN (85 − 0.085 × 634) ≈ 7.3 điểm. Không cần đổi file.
   Đo: nộp public v5; kèm ba tín hiệu trên chạy lại trên `submission.jsonl` mới (tỷ lệ hỏi cho app public-only, tỷ lệ dừng sau
   đọc ở nhóm 226, số lần thay bằng app anh em). Chấp nhận khi public tăng và ít nhất hai tín hiệu tiến về mức TRAIN.
2. **v6: đổi tên có nghĩa thay cho mã băm, trên một phần mẫu.** Tên tool public là tên có nghĩa (`grocery.search`), còn masking
   dùng `fn_<hash>`. Đổi tên app/tool của TRAIN sang tên khác có nghĩa (vd. `shopping_list` → `market_list`), bảng viết tay chỉ
   từ app train, áp cho cả lời gọi, lịch sử, ví dụ và `ask_missing_info.tool_name` như `mask_tool_names`. File:
   `scripts/build_sft.py` (một hàm cạnh `mask_tool_names`), test mới. Cần train lại. Trần: 67 mẫu thay bằng app anh em ≈ 10.6
   điểm (cùng tập mẫu với mục 1 một phần). Đo: như mục 1.
3. **Ví dụ truy hồi khi không có ví dụ nào gần.** Pool chỉ có tool train, nên với yêu cầu cần tool lạ, hai ví dụ luôn là app
   train và có thể kéo model sang app anh em (67 trường hợp ở trên). Đề xuất: bỏ ví dụ khi Jaccard của ví dụ đầu dưới ngưỡng τ,
   chọn τ trên 171 ID leave-app-out, cùng luật lúc build SFT và lúc suy luận (`fewshot.select`, `native_planner.build_messages`).
   Cần train lại. Phân bố độ giống trên public: NOT MEASURED (không lưu transcript ASR public).
4. **Đo tool chưa thấy ngoài portal.** Một adapter train trên `research/unseen_clean_split.json` (1065 ID) và chấm 171 ID của 14
   tool giữ ra là phép đo offline duy nhất cho mục 1–3. Một lượt QLoRA (v3: 154 bước × 71 s trên L4; G4: NOT MEASURED) + chấm.

## Không làm

- Viết tay mô tả cho tham số của tool public: là chỉnh theo catalog public, và train cũng thiếu mô tả với tỷ lệ tương tự.
- Đổi masking từ 30% lên 50% trước khi có kết quả public của v5 hoặc phép đo mục 4: không có căn cứ chọn số.
- Dựa vào smoke40 để chọn giữa v4/v5/v6 cho phần tool lạ: smoke40 không có tool lạ.

## Tín hiệu của v4 trên 261 mẫu đầu (16/11, `research/public_signals.py`)

`python -B -m research.public_signals <submission.jsonl> [--compare other.jsonl] [--first N]` tính lại các tín hiệu trên cho
một file nộp bất kỳ; chạy trên `v3_asrft` cho đúng các số ở bảng trên (8/317, 52/709, 53/226, 18/184, 14/224, 67/472).
`mixed2_v4_261.jsonl` có 261 dòng đầu là v4 (134 dòng khác v3), so với v3 + ASR fine-tune trên cùng 261 ID:

| Tín hiệu | v4 | v3 + ASR fine-tune |
|---|---|---|
| Hỏi thiếu, app chỉ có ở public | 7/67 (10.4%) | 5/71 (7.0%) |
| Hỏi thiếu, app của train | 41/400 (10.2%) | 38/410 (9.3%) |
| Cuối là lệnh đọc: có app public-only, lịch sử chưa dùng | 7/56 (12.5%) | 5/56 (8.9%) |
| Cuối là lệnh đọc: không có app public-only | 6/179 (3.4%) | 9/179 (5.0%) |
| App public-only đã cài không được gọi, gọi app train anh em | 18/71 (25.4%) | 22/67 (32.8%) |
| Mẫu gọi tool public-only | 62/261 (23.8%) | 63/261 (24.1%) |
| Lời gọi mỗi mẫu | 2.76 | 2.79 |

- 261 ID đầu nghiêng về app của train: chỉ 82/261 mẫu có app public-only (cả tập: 410/634), nên file trộn 42.07 chủ yếu thử v4
  trên mẫu giống train, chưa thử trên phần tool lạ.
- Trên 261 ID đó các tín hiệu của v4 và v3 gần như nhau; không tín hiệu nào cho thấy v4 xử lý tool lạ khác v3.
