# Brainstorm: mất điểm ở đâu và 3 cải tiến lớn nhất

Viết 15/11 04:00 GMT+7, làn phân tích sâu. Chỉ đọc file local, không chạy GPU, không đụng Colab.

Nguồn số liệu:

- `outputs/colab/planner_sft/smoke40_sft154_x3/native.{scored,report}.json` (bản W, QLoRA bước 154, T4, ASR thật, TIME_SCALE=3)
- `outputs/colab/planner_sft/smoke40_sft50_x3/native.{scored,report}.json`
- `<kit>/train/train.jsonl` (gold TRAIN, chỉ dùng cho phân tích oracle trên CPU), `research/validation_split.json`, `research/smoke_split.json`
- Mục "Trạng thái đo được" của [HANDOFF.md](HANDOFF.md)

Mọi con số "trần trên" bên dưới là số điểm đang mất ở nhóm mẫu đó, không phải dự báo mức tăng. Mức tăng thật của cả ba đề xuất: NOT MEASURED.

## Kết luận ngắn

1. Bản W đạt 24.83 trên smoke40, sai số chuẩn 6.6 điểm. Chênh lệch bước 50 → 154 là +3.02 ± 4.66 (ghép cặp), tức chưa phân biệt được với nhiễu.
2. Ba việc đáng làm nhất, theo thứ tự: (1) đo rồi khép khoảng cách ASR giữa lúc train và lúc suy luận, (2) đưa mẫu TRAIN giống nhất vào prompt và train lại với đúng prompt đó, (3) hiệu chỉnh quyết định "không làm gì" ở bước đầu.
3. Không có số đo nào cho thấy 80–90 đạt được trước hạn 16/11 23:59. Việc số 0 vẫn là xuất được Docker của W (HANDOFF ghi export đang kẹt vì ổ C còn 11.6 GB so với image 19.7 GB).

## Độ tin cậy của smoke40

| Đại lượng | Giá trị |
|---|---|
| Bước 50 | 21.81, sai số chuẩn 6.24 |
| Bước 154 | 24.83, sai số chuẩn 6.60 |
| Chênh ghép cặp 154 − 50 | +3.02, sai số chuẩn 4.66; 6 mẫu đổi điểm (4 tăng, 2 giảm) |
| Một mẫu lật 0 ↔ 1 | 2.5 điểm |

`scripts/colab_overnight.sh` chọn cờ thắng bằng `S > BASE` trên 40 mẫu: một mẫu lật là đủ đổi kết quả. Đề xuất: quyết định A/B trên ít nhất 120 ID validation ngoài smoke40 (ước lượng 120 × 11.7 s ≈ 23 phút mỗi biến thể trên T4, từ mean 11.68 s/mẫu đã đo), hoặc tối thiểu đòi chênh ≥ 5 điểm trên smoke40.

smoke40 cũng khó hơn tập validation: gold rỗng 12.5% (VAL309 7.1%), mẫu ≥ 3 lệnh 55% (VAL309 40.8%), có `ask_missing_info` 17.5% (VAL309 12.0%). Điểm trên 309 ID và locked269: NOT MEASURED.

## Xếp hạng nguyên nhân mất điểm (bước 154, mất 75.17 điểm)

Mỗi mẫu chưa trọn điểm gán đúng một nhóm theo lỗi đầu tiên chặn điểm; tổng các nhóm khớp 75.168.

| Hạng | Nhóm | Mẫu | Điểm mất | ID |
|---|---|---:|---:|---|
| 1 | B. Sai giá trị ngày/giờ/số | 7 | 17.50 | 0611, 0583, 1012, 0373 (chỉ sai đúng một giá trị); 0131, 1645 (sai ngày kéo theo sai nhánh); 0413 (sai giờ + thiếu hỏi) |
| 2 | A. Bỏ cuộc sai ở bước 1 (0 lệnh) | 6 | 15.00 | 0229, 0246, 0781, 0937, 1041, 1296 |
| 2 | C. Suy luận điều kiện/nhiều bước trên kết quả tool | 6 | 15.00 | 0045, 0087, 0088, 0169, 1095, 1015 |
| 4 | F. Chữ trong đối số bị ASR làm hỏng | 4 | 8.33 | 0086 ("Minh"→"cha"), 0578 ("uống magie"→"y bốn mai dê"), 0707 ("họp"→"học"), 1202 |
| 5 | D. Sai quyết định `ask_missing_info` | 3 | 7.50 | 0350, 1598 (đáng hỏi mà làm thẳng); 0188 (hỏi thừa) |
| 6 | E. Hành động khi gold rỗng | 2 | 5.00 | 1582, 1655 |
| 6 | G. Kế hoạch lạc đề | 2 | 5.00 | 0280 (banking thay vì đèn + SMS), 0675 |
| 8 | H. Lệch nhỏ theo khớp chính xác | 3 | 1.83 | 0286, 0695 (tra cứu thừa), 0665 ("cô Hạnh" thay vì "Cô Hạnh") |

Ba phát hiện cắt ngang các nhóm:

**1. Train bằng transcript sạch, suy luận bằng transcript ASR.** `scripts/build_sft.py:63` gọi `_sample_from_row(kit, row, None, …)`, và `src/inference/pipeline.py:44-45` lấy `audio_text` khi `transcribe is None`. Model chưa từng thấy lỗi ASR khi train. Lỗi ASR nhìn thấy được trong đối số chiếm 4 mẫu (8.33 điểm). Phần ASR nằm sau các lỗi giờ/số (0373: 30 → 70; 0413: 06:30 → 05:30; 0583: 16:00 → 17:00; 1012: 03:30 → 05:45; 0707: 15:00 → 09:00) và sau 0280: NOT MEASURED, vì report không lưu transcript ASR. Chưa có lượt chạy nào của bản SFT với `--train-transcripts`.

**2. Model thua cả trên mẫu có "bản sao" trong tập train.** Với mỗi ID validation, lấy mẫu gần nhất trong 1236 ID train theo Jaccard tập từ của transcript gold (tách từ `\w+`, chữ thường):

| Tập | Mẫu gần nhất có cùng chuỗi tên tool với gold | Trong top-3 | Khi Jaccard ≥ 0.5 |
|---|---:|---:|---:|
| VAL309 | 115/309 (37.2%) | 140/309 (45.3%) | 72/84 (85.7%) |
| smoke40 | 18/40 (45.0%) | 21/40 | 9/11 |

Trên 18 mẫu smoke40 có bản sao, W chỉ đạt trung bình 0.338 (4 mẫu trọn điểm), mất 29.81 điểm; 22 mẫu còn lại đạt 0.175. Ví dụ: 0246 có bản sao giống 0.92 (train_0157, cùng 4 lệnh) nhưng W trả 0 lệnh; 0373 có bản sao giống 0.86 (train_0275, cũng "mờ 30 phần trăm") nhưng W điền 70. Bản sao theo nhóm lỗi: C 4/6, B 3/7, D 2/3, H 2/3, A 1/6, F 1/4, G 1/2, E 0/2. Hai cách giải thích (ASR làm lệch đầu vào, hoặc 1 epoch chưa đủ để thuộc mẫu): chưa tách được, NOT MEASURED.

**3. Bỏ cuộc sai là một quyết định một token.** Ở bước 154 có 9/40 mẫu sinh đúng 15 token ở bước 1 rồi dừng, chính là câu `NOTHING` của `scripts/build_sft.py:22`. Chỉ 3 mẫu có gold rỗng (0432, 1161, 1360): độ chính xác 33%. Bước 50: 6 lần bỏ cuộc, 2 đúng. Tỷ lệ gold rỗng trong train là 7.2% (89/1236), model bỏ cuộc 22.5%. 5/6 mẫu bỏ cuộc sai chứa từ điều kiện (nếu, thì, trừ, mà); trong train, câu có các từ này có 434 mẫu hành động và 25 mẫu rỗng, và 18/25 mẫu rỗng đó có lịch sử (điều kiện đã được giải trong lịch sử). Giả thuyết của tôi: model bám vào dấu hiệu bề mặt. Văn bản thô của các lượt này: NOT MEASURED (run trước commit `385057a`).

## Ba cải tiến đề xuất

### 1. Đo rồi khép khoảng cách ASR

Căn cứ: phát hiện 1 và 2; nhóm F 8.33 điểm; 5 mẫu sai giờ/số chưa rõ nguồn (12.5 điểm nếu tính 0373, 0413, 0583, 1012, 0707).

| Bước | Việc | Chi phí ước lượng |
|---|---|---|
| a | smoke40 trên W với `--train-transcripts` (cờ đã có trong `src/inference/__main__.py`). Hiệu số so với 24.83 là phần ASR lấy mất | ~9 phút T4 (40 × 11.7 s + nạp model 78 s) |
| b | Chép transcript ASR của 1236 ID train (câu hiện tại và lượt user trong lịch sử) ra jsonl, tính tỷ lệ lỗi từ so với `audio_text` | ~54 phút T4 (1236 × 2.6 s, từ ASR mean 2.62 s đã đo); script ~30 dòng |
| c | `build_sft.py` thêm `--transcripts <jsonl>` để câu user là bản ASR, gold giữ nguyên; train v2 trên dữ liệu này | code ~10 dòng; train 1 epoch ≈ 5.4 h T4 (127 s/bước × 154 đã đo); A100: NOT MEASURED |
| d | Chỉ khi (a) cho thấy mất ≥ 10 điểm: fine-tune PhoWhisper-medium trên 1236 cặp audio/transcript train, hoặc đổi sang PhoWhisper-large | NOT MEASURED. Tham số: planner 4,022,468,096 + large ~1.55 tỷ (theo kiến trúc Whisper large, chưa đếm) vẫn dưới 6 tỷ; độ trễ large NOT MEASURED |

Quy tắc dừng: nếu (a) cho hiệu số < 5 điểm thì bỏ b–d và dồn GPU cho đề xuất 2.

Làm được trước hạn: a, b, c. Bước d chỉ nên thử nếu còn thừa một phiên GPU sau khi đã có bản dự phòng.

### 2. Đưa mẫu train giống nhất vào prompt, và train lại với đúng prompt đó

Căn cứ: phát hiện 2. Trần trên smoke40 là 29.81 điểm (điểm đang mất trên 18 mẫu có bản sao). Bản sao phủ 4/6 mẫu nhóm C, là nhóm chưa có cách sửa nào khác trong hàng đợi.

Cách làm, dùng lại code sẵn có:

- Chọn ví dụ bằng `src/agent/fewshot.py` (Jaccard transcript + 0.2 × Jaccard app, tất định, đã loại ID validation và chính mẫu đó), nhưng nguồn là chuỗi lệnh gold của cả 1236 mẫu trong `outputs/sft_native.jsonl` thay vì 530 plan DSL.
- Render 1–2 ví dụ thành dòng chữ trong system prompt (`"<câu nói>" → tool_a{…}; tool_b{…}`), không dựng thành lượt hội thoại, để không đụng quy tắc 3 "không làm lại việc trong lịch sử".
- `build_sft.py` render cùng ví dụ đó cho từng mẫu train, để train và suy luận thấy cùng dạng prompt.

Chi phí: 2–3 giờ code + test trên CPU. A/B chỉ-suy-luận trên W: ~9 phút T4, có thể giảm điểm vì W chưa từng thấy ví dụ. Train lại: gộp chung một lượt với bước 1c (≈ 5.4 h T4 cộng phần token tăng thêm; số token tăng NOT MEASURED, few-shot DSL k=2 trước đây tăng 4.04% ký tự).

Giới hạn cần ghi rõ: tỷ lệ bản sao đo trên transcript gold, với transcript ASR sẽ thấp hơn (NOT MEASURED). Public test có 34 tool / 11 app chỉ xuất hiện ở public, nên tỷ lệ bản sao ở đó chưa biết (NOT MEASURED, và không được dùng nhãn public để đo).

Làm được trước hạn nếu code xong trong sáng 15/11 và chỉ train một lượt.

### 3. Hiệu chỉnh quyết định "không làm gì" ở bước 1

Căn cứ: phát hiện 3. Nhóm A mất 15.0 điểm, là nhóm lớn nhất sửa được mà không cần train; rủi ro là 3 mẫu rỗng đang đúng (7.5 điểm).

Cách làm: ở bước 1, khi model không sinh `<tool_call>`, lấy chênh logit giữa token `<tool_call>` và token đầu của câu `NOTHING`. Nếu chênh nhỏ hơn ngưỡng τ thì sinh lại một lần với tiền tố `<tool_call>` bị ép. Chọn τ trên các ID validation ngoài smoke40, rồi mới chấm smoke40.

Chi phí: cần `LocalPlanner.generate` trả logit bước đầu (file `src/agent/local_model.py`, làn Codex) và ~15 dòng trong `native_planner.py`; ~2 giờ code. GPU: ghi chênh logit bước 1 cho 269 ID ≈ 30 phút T4 (ước lượng 269 × (ASR 2.6 s + sinh bước 1 ~4.4 s)), rồi smoke ~9 phút. Thêm tối đa một lượt sinh mỗi mẫu (~1.3 s với prefix cache, theo số đo các bước sau).

Trần trên smoke40 là +15.0, nhưng lệnh bị ép trên các câu điều kiện khó sẽ không đúng hết: mức tăng thật NOT MEASURED. Nếu đề xuất 2 được train lại thì phải đo lại nhóm này trước khi bật.

Làm được trước hạn; đây là việc rẻ nhất trong ba việc.

## Đã có trong hàng đợi, không đề xuất lại

- Cờ `LIFEMATE_DATE_TABLE`, `LIFEMATE_ARG_NORMALIZE`, `LIFEMATE_PROMPT_V2` (việc #11): điểm NOT MEASURED. Bảng ngày chỉ liệt kê ngày, nên trần trên smoke40 là 0611 (2.5 điểm) cộng 0131 và 1645 nếu phần còn lại của chuỗi đúng (5.0 điểm). Bốn lỗi giờ trong ngày (0413, 0583, 1012, 0707) không được cờ nào xử lý; kết quả bước 1a sẽ cho biết chúng do ASR hay do planner.
- Retrain v2 = SFT + pilot300 (bước 4 của `colab_overnight.sh`): vẫn train trên transcript sạch và không có ví dụ truy hồi. Nếu lượt này chưa bắt đầu, nên thay dữ liệu của nó bằng bản 1c + 2.

## Lịch đề xuất tới hạn 16/11 23:59

| Khi | Việc | GPU |
|---|---|---|
| Ngay | Xuất được Docker của W (24.83) làm bản dự phòng; giải quyết chỗ trống ổ đĩa | không |
| 15/11 sáng | Code đề xuất 2, 1b/1c, 3 trên CPU, kèm test | không |
| 15/11 trưa | Một phiên đo: 1a (9 phút), 1b (54 phút), ghi chênh logit cho 3 (30 phút) | ~1.5 h T4 |
| 15/11 chiều → đêm | Một lượt train v2: transcript ASR + ví dụ truy hồi, 1 epoch | ≥ 5.4 h T4 |
| 16/11 sáng | Chấm v2 và W trên smoke40 + ≥ 120 ID validation, chọn bản, áp ngưỡng τ | ~1 h T4 |
| 16/11 12:00 | Đóng băng ứng viên; sau đó chỉ build, chạy offline, export, MD5 | không |

Không kịp trước hạn: train ≥ 2 epoch trên T4 hoặc nhiều seed, SFT có lời giải thích từng bước, dữ liệu ngoài (xLAM-VI), fine-tune ASR có kiểm chứng đầy đủ, và mục tiêu 80–90.

## Chưa đo (NOT MEASURED)

- Điểm của W với transcript gold; tỷ lệ lỗi từ của PhoWhisper-medium trên audio TRAIN.
- Điểm của W trên VAL309, locked269, public test; điểm ở TIME_SCALE=1.
- Văn bản thô của 9 lượt bỏ cuộc và 5 mẫu sai giờ/số.
- Tác dụng của ba cờ việc #11 và của cả ba đề xuất trên.
- Tốc độ train và suy luận trên A100, CU/giờ thực tế.
- Tỷ lệ bản sao khi dùng transcript ASR và trên public test.

## Cách tái lập số liệu

Hai script tạm (không nằm trong repo) đọc đúng các file ở mục Nguồn số liệu: thống kê thành phần và láng giềng gần nhất trên `train.jsonl` với pool là 1236 ID ngoài `validation_ids`; sai số chuẩn là độ lệch chuẩn mẫu của điểm từng mẫu chia √40; bảng nhóm lỗi là phân loại thủ công từ `gold` và `predicted` trong `native.scored.json`, đã kiểm tổng bằng 100 − 24.832.
