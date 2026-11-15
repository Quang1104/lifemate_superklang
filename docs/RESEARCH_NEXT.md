# Sau v4 còn lấy điểm ở đâu

Viết 15/11 19:00, chỉ CPU, không sửa code. Nguồn: `outputs/colab/smoke_ab_drive/smoke_ab/{v3_asrft,w_asrft,oracle_transcripts}`,
gold TRAIN, catalog và đầu vào public (không nhãn, không transcript). Tiến độ: [progress/orca.md](progress/orca.md), mục R-1 → R-5.

Cách đọc: smoke40 có sai số chuẩn khoảng 6.6 điểm, một mẫu là 2.5 điểm. "Trần" là điểm đang mất trên các mẫu đó ở lượt
`v3_asrft` (51.73), không phải dự báo. Kết quả v4 và `LIFEMATE_DUP_GUARD`: NOT MEASURED lúc viết. Thời gian trên RTX 5090 là
ước lượng từ số đo trên G4 (RTX PRO 6000 Blackwell), coi hai máy ngang nhau.

## 1. v3_asrft mất 48.27 điểm ở đâu

| Nhóm | Mẫu | Mất | ID |
|---|---:|---:|---|
| Điều kiện / nhiều bước trên kết quả tool | 7 | 13.54 | 0087, 0229, 1360, 1645, 1095 (0 điểm); 0169, 0286 (trả bằng ví trước) |
| Sai giá trị đối số | 5 | 11.25 | 0131 (`"Anh Tuấn"`), 0665 (`"Hạnh"`), 0280 (nhầm đèn), 0675 (email), 0695 |
| Ngày giờ | 4 | 10.00 | 0188 (45 phút nữa), 0611 (Chủ nhật này), 0707 (22/10 → 20/10), 1202 |
| Thiếu `ask_missing_info` | 2 | 5.00 | 1296, 1598 |
| Hành động khi gold rỗng | 2 | 5.00 | 1161, 1582 (đều thiếu app email) |
| Thừa/thiếu đối số tuỳ chọn | 3 | 3.48 | 0246 (`floor`), 0578 (thiếu `time`), 1015 |

Tách nguồn bằng ba lượt (v3 + ASR mới, W + ASR mới, W + transcript gold):

- 12 mẫu sai ở cả ba lượt, 26.9 điểm: lỗi planner không phụ thuộc ASR.
- 7 mẫu W với cùng ASR làm tốt hơn v3 (0707, 0280, 0131, 0695, 1161, 0286, 0246), 13.0 điểm: lỗi riêng của v3. v3 train trên
  transcript PhoWhisper cũ nhưng chạy với ASR đã fine-tune; v4 train trên transcript ASR mới nên phải đo lại nhóm này.
- 4 mẫu chỉ đúng khi có transcript gold (0578, 1202, 1095, 1015), 8.4 điểm: phần còn do ASR.
- Ba mẫu trùng tên (0086, 0350, 0413) v3_asrft đã trọn điểm: `LIFEMATE_DUP_GUARD` không còn điểm nào để lấy trên smoke40.

## 2. Tool chỉ có ở public (34 tool, 11 app)

- Prompt: cả 34 tool có mô tả; 9/54 tham số không có mô tả (`symbol`, `side`, `direction`, `address`, `payment`, `type`...);
  bảng chú giải enum của `CONTEXT_V4` đã gồm enum của catalog public. Guard: mọi `x-ref` của catalog public đều có tool sinh
  ra trong `lifemate_catalog.py` (0 loại ID bị bỏ sót), nên không chặn nhầm ID của tool lạ.
- Đầu vào public: 410/634 mẫu có cài app lạ; 184 mẫu có lịch sử trợ lý đã gọi tool lạ (lịch sử đó vào prompt như lượt hội
  thoại); số tool cài mỗi mẫu trung vị 30 (train 31).
- Truy hồi: mẫu public nào cũng có ≥ 97 dòng pool dùng được, nhưng pool chỉ có tool của train nên ví dụ không bao giờ minh hoạ
  tool lạ. Ví dụ cũng có thể rò nội dung: ở 0578 model ghi "Uống canxi" (chữ của ví dụ train gần nhất) thay cho "Uống magie".
- Chưa có phép đo sạch nào về tool chưa thấy (v3, v4 train trên cả 171 ID leave-app-out). Tỷ lệ yêu cầu public nhắm vào
  tool lạ: NOT MEASURED.

## 3. Ngân sách thời gian: đang dùng 2.2 / 30 s

Đo trên G4 (v3_asrft): trung bình 2.23 s/mẫu, p95 3.70 s; ASR 0.51 s; sinh 1.68 s cho 3.6 lượt gọi planner; 57.6 token/s;
prompt bước 1 trung vị 5,367 token, lớn nhất 8,430.

| Phương pháp | Cơ chế | Thêm (ước lượng) | Căn cứ / rủi ro |
|---|---|---|---|
| Beam search cho ASR | Bớt lỗi thay từ | +0.5–1.0 s | Phần còn do ASR là 8.4 điểm. WER với beam: NOT MEASURED; đo trên 309 ID validation trước |
| N-best + chấm lại theo tên trong context | Sửa tên riêng | +0.5–1.0 s | Không mẫu nào trong 4 mẫu còn do ASR là lỗi tên: không làm |
| Hỏi lại khi guard từ chối | Model sửa lệnh sai | +0.5 s | Chỉ 4 mẫu có guard từ chối (5 lần); đã có gợi ý phục hồi: không làm |
| Ép tên tool / enum khi giải mã | Chặn tên sai | ~0 s | 2 lần gọi tool của app không cài, 2 lần hỏi sai tham số: quá ít |
| K = 3–4 ví dụ | Thêm bản sao | +0.1–0.3 s | Model train với K = 2, đổi K là lệch prompt; chỉ thử nếu train lại |
| Tìm lại danh bạ (mục 4, việc 3) | Sửa truy vấn sai dạng | +0.3 s | Có số đo trên TRAIN |

## 4. Việc nên làm, xếp theo trần

Chấp nhận một việc chỉ khi smoke40 tăng ≥ 5 điểm so với bản đang giữ (hoặc ≥ 2 mẫu lật trong nhóm nó nhắm tới).

1. **Chấm v4 theo đúng bảng mục 1, và chạy v4 với `--train-transcripts`.** Không sửa file, không train. GPU ~3 phút mỗi lượt
   trên G4 (ước lượng từ 2.2 s/mẫu + nạp model). Trần của các nhóm v4 nhắm tới: điều kiện 13.54, gold rỗng 5.0, cộng 13.0 điểm
   lỗi riêng của v3. Quyết định: nhóm nào v4 không lật được ≥ 2 mẫu thì chuyển sang việc bên dưới.
2. **Ngày giờ: train bản kế tiếp với `LIFEMATE_DATE_TABLE=1`** (cờ đã có trong `native_planner.py`), bật cho cả `build_sft.py` và
   suy luận. Căn cứ: 329/1236 câu train có đối số ngày giờ, 33 câu có tên thứ; bảng phủ 447/476 giá trị ngày giờ gold của train
   (số đo cũ trong `research/date_table_coverage.md`); cờ này mới chỉ được thử ở dạng chỉ-suy-luận trên W (25.06 so với 24.83).
   Cần train lại; gộp vào lượt QLoRA kế tiếp (v3 mất 154 bước × 71 s trên L4; G4: NOT MEASURED). Trần 10.0 (4 mẫu); bảng không
   giải được 1202 và "45 phút nữa" (chỉ 3/329 câu train có dạng này).
3. **Tìm lại danh bạ khi truy vấn sai dạng**, không cần train. Căn cứ: 161/161 truy vấn gold của `contacts.search` bằng đúng tên
   một liên hệ được trả về; 38 tên bắt đầu bằng từ xưng hô viết hoa ("Cô Hạnh", "Sếp Dũng"); 11 câu nói "anh X" nhưng tên lưu
   chỉ là "X". Luật: tìm không ra và truy vấn có từ xưng hô thì tìm lại bằng phần tên. Chỗ sửa: `src/agent/runner.py`, sau cờ.
   Trần 3.75 (0131, 0665). Biến thể "gọi lại `contacts.search` bằng đúng tên đã lưu" chỉ để khớp cách chấm (F1 0.5 → 0.8 ở
   0665) là tối ưu theo thước đo, root quyết định có dùng hay không.
4. **PhoWhisper-large fine-tune trên cùng 4,062 câu.** Tham số 4.02 tỷ + ~1.55 tỷ < 6 tỷ. Ước lượng từ medium (3 epoch 7.5 phút
   trên G4, 0.51 s/mẫu): train 15–25 phút, ASR 1.0–1.5 s/mẫu. File: `training_code/train_asr.py`, `scripts/colab_*.sh`, Docker
   (model lớn hơn). Trần 8.4 (4 mẫu còn do ASR). WER của large và hành vi trên từ vựng app lạ: NOT MEASURED. Chấp nhận nếu WER
   validation thấp hơn 9.9% rõ rệt và smoke tăng ≥ 5.
5. **Đo tool chưa thấy trước khi tin điểm smoke.** Một trong hai: điểm public trên portal của bản đang giữ, hoặc một adapter
   train trên `research/unseen_clean_split.json` (1065 ID) rồi chấm 171 ID giữ ra (một lượt QLoRA + ~10 phút chấm). Không có
   trần trên smoke40 vì smoke40 không có tool lạ; 410/634 mẫu public có cài app lạ.
6. **Hai lỗi thiếu hỏi còn lại (1296, 1598), trần 5.0:** chưa có luật tất định; gold hỏi một giá trị 68 lần trong train. Chỉ
   làm nếu còn lượt train: không đổi code, xem v4 có lật mẫu nào không.

## 5. Xử lý dữ liệu: đã kiểm, không thấy lỗi lớn

- Không có transcript trùng nhau trong 1236 ID train; không kết quả tool nào (2954 trong lịch sử, 2840 gold) dài quá 6000 ký tự
  nên `_RESULT_CHARS` không cắt gì; lịch sử dài hơn 20 lượt ở 41 mẫu.
- Nhãn có quy ước khó đoán: truy vấn danh bạ phải bằng tên đã lưu (việc 3); `create_event` có `end` 23/34 lần; nội dung tin
  nhắn gold chỉ nguyên văn trong câu nói ở 70/235 lần (`content`), 46/118 (`title`).
- Lệch train/suy luận còn lại: v3 train trên transcript ASR cũ (v4 đã sửa); số mẫu bị `--max-tokens 12000` loại khi train v4:
  NOT MEASURED, xem dòng `rows` của `train_qlora.py`.

## Không làm

- Chọn giữa W và v3 theo từng mẫu: trần 64.41 nếu chọn đúng tuyệt đối, nhưng không có luật chọn nào được kiểm, và hai model đã
  merge là 8 tỷ tham số, vượt giới hạn.
- Chỉnh thêm `LIFEMATE_DUP_GUARD`, n-best theo tên, ép giải mã tên tool, hỏi lại khi guard từ chối: trần trên smoke40 gần 0.
- Dòng nháp suy luận trước lệnh ghi: không có dữ liệu train có lời giải.
- Đổi K hoặc bỏ ví dụ lúc suy luận mà không train lại: lệch prompt với model QLoRA.

## Đã làm (15/11, CPU; điểm: NOT MEASURED)

Việc 3, chỉ bản trung thực: `LIFEMATE_CONTACT_RETRY=1` (mặc định tắt, không cần train). Trong `src/agent/runner.py`, khi
`contacts.search` trả danh sách rỗng và truy vấn mở đầu bằng từ xưng hô (anh, chị, thầy, ông, sếp, bác, cô, thằng, bà, vợ, mẹ,
chú, bé: các từ có trong tên liên hệ hoặc câu nói của TRAIN), runner gọi lại `contacts.search` với phần tên còn lại. Lần gọi
lại là một lời gọi ToolEnv thật, có trong `tool_calls`. Không làm bản "tìm lại bằng tên đã lưu để khớp cách chấm".

- Đo trên 1236 ID train với `contacts.search` thật: `python -B -m scripts.orca_v4_measure contact --data-root <kit>`.
  Luật kích hoạt trên 0/161 truy vấn gold. Với 11 câu nói "<xưng hô> tên" mà tên lưu không có từ xưng hô: truy vấn kèm từ
  xưng hô tìm không ra ở 11/11, lần tìm lại trả đúng liên hệ gold ở 11/11.
- Trên smoke40, mẫu duy nhất rơi vào luật này là 0131 (2.5 điểm đang mất); 0665 không được sửa vì truy vấn "Hạnh" vẫn tìm ra.
  Khi luật chạy, lời gọi đầu vẫn nằm trong `tool_calls` nên mẫu đó không thể trọn điểm (thừa một lời gọi).
- Test: `python -B -m pytest -q -p no:cacheprovider` → 360 passed, 1 skipped; `python -B -m src.inference mock --samples 40` → 40/40.
