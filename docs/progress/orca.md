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

## V4-2 (15/11 15:05): `LIFEMATE_CONTEXT_V4=1` xong

- Đã làm, tất cả trong code dùng chung cho train và suy luận:
  - App không cài: `OfficialKit.app_names` (tên `name_vi` theo catalog của split); `pipeline._sample_from_row` ghi
    `context["apps"]`; `native_planner.build_messages` in ba dòng (đã cài / KHÔNG cài / bỏ qua việc cần app không cài).
  - Ví dụ kèm kết quả tra cứu: `build_sft.py --fewshot-k` luôn ghi thêm `results` vào pool (kết quả gold của lệnh đọc,
    cắt 500 ký tự, lệnh ghi và `sys.*` để trống); chỉ khi bật cờ thì `fewshot.select` trả `results` với ngân sách
    6000 ký tự và `render_examples` in mỗi lệnh một dòng, lệnh đọc kèm `⇒ kết quả`.
  - Chú giải enum: bảng viết tay 85 giá trị của catalog train + public trong `native_planner`; `tool_definitions` thêm
    `Giá trị: electricity = điện; ...` vào mô tả tham số (bản sao, không sửa schema gốc); giá trị lạ để nguyên.
- File: `src/inference/official.py`, `src/inference/pipeline.py`, `src/agent/fewshot.py`, `src/agent/native_planner.py`,
  `scripts/build_sft.py`, `src/agent/data/fewshot_native.jsonl` (586,509 → 1,085,562 byte), `tests/test_orca_v4_context.py`.
- Kiểm tắt cờ: build SFT mặc định và build `--fewshot-k 2` với pool mới trùng byte với bản trước khi sửa
  (sha256 đầu `340f7fad…` và `dfba1337…`).
- Đo bật cờ (`python -B -m scripts.orca_v4_measure prompt --data-root <kit>`, smoke40, K=2, transcript gold):
  system prompt thêm median 2034 ký tự (p95 3041, max 3110); định nghĩa tool thêm median 650 (p95 1096);
  tổng thêm median 2505 (p95 3778, max 3885). Bản ghi SFT 1236 mẫu: ký tự p50 18,214 → 20,667, max 33,193 → 37,872.
  Token Qwen và điểm: NOT MEASURED. Lưu ý: v3 có max 9,714 token ở 33,193 ký tự; `train_qlora.py --max-tokens 12000`
  nên cần xem dòng `rows` khi train v4 để biết có mẫu nào bị loại.
- Test: `python -B -m pytest -q -p no:cacheprovider` → 325 passed, 1 skipped; `python -B -m src.inference mock --samples 40` → 40/40.
- Tiếp: `LIFEMATE_SMALL_FIXES` (bù `end` cho `calendar.update_event`, email đọc đánh vần).

## V4-3 (15/11 15:10): `LIFEMATE_SMALL_FIXES=1` xong

- Đã làm:
  - `calendar.update_event`: khi model đổi `start` mà không có `end` và sự kiện cũ (có `start`, `end`) nằm trong kết quả
    đã thấy (lượt hiện tại hoặc lịch sử), thêm `end` = `start` mới + thời lượng cũ. Hàm `moved_event_end` trong `runner.py`.
  - Email đọc đánh vần: `asr_fix.spoken_email` ghép "mai chấm anh năm mốt a còng example chấm test" thành
    `mai.anh51@example.test`; chạy trong lớp bọc transcriber (cùng chỗ nối với `LIFEMATE_ASR_FIX`, hai cờ độc lập).
- File: `src/agent/runner.py`, `src/agent/asr_fix.py`, `tests/test_orca_v4_small_fixes.py`.
- Căn cứ trên 1236 ID train (`python -B -m scripts.orca_v4_measure small --data-root <kit>`): 27 lời gọi gold dời `start`,
  27 có `end`, 27 có sự kiện cũ trong kết quả đã thấy, 27 có `end` = `start` mới + thời lượng cũ. 7 câu gold đọc đánh vần
  địa chỉ email; luật ghép ra đúng địa chỉ gold ở 7/7 câu với transcript gold và 0/7 với transcript PhoWhisper-medium
  (`outputs/asr_train.jsonl`); không đổi câu nào khác. Với ASR đã fine-tune: NOT MEASURED.
- Test: `python -B -m pytest -q -p no:cacheprovider` → 344 passed, 1 skipped; `python -B -m src.inference mock --samples 40` → 40/40.
- Tiếp: mục "Đã làm" trong `docs/CONTEXT_PLAN.md`.

## V4-4 (15/11 15:15): ghi mục "Đã làm" trong `docs/CONTEXT_PLAN.md`

- Đã làm: bảng ba cờ, số đo CPU và lệnh chính xác; kiểm trên kit thật: với `LIFEMATE_CONTEXT_V4=1`, prompt và định nghĩa
  tool lúc suy luận trùng bản ghi SFT ở 50/50 ID train lấy mẫu; mỗi mẫu liệt kê 29–40 app không cài.
- Trạng thái: cả ba cờ đã commit, test 344 passed / 1 skipped, mock 40/40. Chưa push.
- Việc của root: thêm cờ vào `scripts/colab_v4.sh` (`LIFEMATE_CONTEXT_V4=1` cho cả `build_sft.py` và suy luận;
  `LIFEMATE_DUP_GUARD=1`, `LIFEMATE_SMALL_FIXES=1` lúc suy luận). Mọi điểm số: NOT MEASURED.

# Research sau v4 (brief `lifemate-orca-research2-brief.md`, 15/11 18:20): chỉ CPU, kết quả ở `docs/RESEARCH_NEXT.md`

## R-1 (15/11 18:30): v3_asrft mất 48.27 điểm ở đâu

- Nguồn: `outputs/colab/smoke_ab_drive/smoke_ab/{v3_asrft,w_asrft,oracle_transcripts}/native.{scored,report}.json`.
  v3_asrft 51.73 (17 mẫu trọn điểm), w_asrft 44.77, oracle (W + transcript gold) 49.92. Phân loại thủ công theo lỗi đầu tiên.
- Điểm mất theo nhóm: điều kiện/nhiều bước 13.54 (0087, 0169, 0229, 0286, 1095, 1360, 1645); sai giá trị đối số 11.25
  (0131, 0280, 0665, 0675, 0695); ngày giờ 10.0 (0188, 0611, 0707, 1202); thiếu hỏi 5.0 (1296, 1598);
  hành động trên gold rỗng 5.0 (1161, 1582); thừa/thiếu đối số tuỳ chọn 3.48 (0246, 0578, 1015). Tổng 48.27.
- Tách nguồn bằng ba lượt: 12 mẫu sai ở cả ba lượt (planner, 26.9 điểm); 7 mẫu W + cùng ASR đúng hơn v3 (0707, 0280, 0131,
  0695, 1161, 0286, 0246: lỗi riêng của planner v3, 13.0 điểm); 4 mẫu chỉ đúng khi có transcript gold (0578, 1202, 1095,
  1015: còn do ASR, 8.4 điểm).
- Ba mẫu trùng tên (0086, 0350, 0413) v3_asrft đã trọn điểm, nên `LIFEMATE_DUP_GUARD` không còn điểm nào để lấy trên smoke40.
- Tiếp: câu 2, tổng quát hoá sang 34 tool chỉ có ở public.

## R-2 (15/11 18:40): tổng quát hoá sang tool chỉ có ở public

- Catalog public: 122 tool, 34 tool / 11 app không có trong train (17 đọc, 17 ghi). Cả 34 có mô tả; 9/54 tham số không có mô tả
  (`symbol`, `side`, `direction`, `address`, `payment`, `type`...); 14 tool có tham số ID. Bảng evidence trong
  `src/tooling/lifemate_catalog.py` đã có đủ nguồn cho mọi `x-ref` của catalog public (0 loại ID không có tool sinh ra),
  nên guard không chặn nhầm ID của tool lạ. Bảng chú giải enum của `CONTEXT_V4` đã gồm enum của catalog public.
- Đầu vào public (634 mẫu, không nhãn, không transcript): 410 mẫu có cài ít nhất một app lạ; 120 mẫu có ≥ 30% tool cài là
  tool lạ; 184 mẫu có lịch sử trợ lý đã gọi tool lạ; 419 mẫu có lịch sử (1299 lượt user). Số tool cài mỗi mẫu: trung vị 30
  (train 31).
- Truy hồi: mẫu public nào cũng có ≥ 97 dòng pool dùng được (trung vị 186), nhưng pool chỉ có tool của train, nên ví dụ
  không bao giờ minh hoạ được tool lạ. Tỷ lệ yêu cầu public nhắm vào tool lạ và độ giống transcript: NOT MEASURED (không có
  transcript). Chưa có phép đo sạch nào về tool chưa thấy: v3/v4 đều train trên cả 171 ID của split leave-app-out.
- Tiếp: câu 3–5 (ngân sách thời gian, ASR lớn hơn, xử lý dữ liệu).

## R-3 (15/11 18:50): ngân sách thời gian

- Đo trên G4 (v3_asrft): 2.23 s/mẫu trung bình, p95 3.70 s; ASR 0.51 s; sinh 1.68 s cho 3.6 lượt planner; 57.6 token/s;
  prompt bước 1 trung vị 5,367 token, lớn nhất 8,430. Chỉ 4 mẫu có guard từ chối (5 lần: 2 tool của app không cài, 2 hỏi sai
  tham số, 1 x-ref).
- Kết luận: n-best theo tên, ép giải mã tên tool và hỏi lại khi guard từ chối có trần gần 0 trên smoke40. Beam search cho ASR
  cần đo WER trước (NOT MEASURED). Đổi K lúc suy luận là lệch prompt với model đã train.
- Tiếp: câu 4.

## R-4 (15/11 18:55): PhoWhisper-large

- Tham số: 4,022,468,096 + ~1.55 tỷ < 6 tỷ (số của large theo kiến trúc, chưa đếm). Ước lượng từ medium (3 epoch 7.5 phút
  trên G4; 0.51 s/mẫu): train 15–25 phút, ASR 1.0–1.5 s/mẫu. WER của large: NOT MEASURED.
- Trần trên smoke40: 8.4 điểm (4 mẫu chỉ đúng khi có transcript gold: 0578, 1202, 1095, 1015). Xếp sau các việc về planner.
- Tiếp: câu 5.

## R-5 (15/11 19:00): xử lý dữ liệu, và viết xong `docs/RESEARCH_NEXT.md`

- Đếm trên 1236 ID train: 0 transcript trùng nhau; 0 kết quả tool dài quá 6000 ký tự (2954 trong lịch sử, 2840 gold);
  41 mẫu có lịch sử dài hơn 20 lượt. Không thấy lỗi xử lý lớn.
- Quy ước nhãn: 161/161 truy vấn gold của `contacts.search` bằng đúng tên một liên hệ được trả về; 38 tên bắt đầu bằng từ xưng
  hô viết hoa; 11 câu nói "anh X" nhưng tên lưu là "X"; truy vấn nguyên văn trong câu nói 129/161 (bỏ hoa-thường 157/161).
  Nội dung chữ gold nguyên văn trong câu nói: `content` 70/235, `title` 46/118.
- Chọn đúng tuyệt đối giữa W và v3 theo từng mẫu cho 64.41 (v3 51.73), nhưng không có luật chọn và hai model merge vượt
  giới hạn tham số: không đề xuất.
- `docs/RESEARCH_NEXT.md`: 6 việc xếp theo trần, kèm danh sách không làm. Mọi mức tăng: NOT MEASURED.

# Việc tìm lại danh bạ (brief `lifemate-orca-contact-brief.md`, root quyết định 15/11 19:05)

## C-1 (15/11): `LIFEMATE_CONTACT_RETRY=1` xong, mặc định tắt

- Đã làm: trong `src/agent/runner.py`, khi `contacts.search` trả `ok` với danh sách rỗng và truy vấn mở đầu bằng một từ xưng hô
  rồi tới tên, runner tự xếp một lệnh `contacts.search` với phần tên còn lại. Lệnh này đi qua mọi guard và được gọi vào ToolEnv
  như lệnh của planner (có trong `tool_calls`); planner thấy cả hai lần tìm ở lượt kế. Chỉ làm bản trung thực; không có bản
  "tìm lại bằng tên đã lưu để khớp cách chấm".
- Danh sách từ xưng hô lấy từ TRAIN: tên lưu kèm từ xưng hô (sếp 33, bác 27, cô 23, anh 15, thằng 4, thầy 3, bà 3, vợ 3, chị 2,
  ông 2, mẹ 1, chú 1, bé 1) và câu nói có từ xưng hô mà tên lưu không có (anh 6, chị 3, thầy 1, ông 1).
- File: `src/agent/runner.py`, `tests/test_orca_contact_retry.py`, `scripts/orca_v4_measure.py` (thêm lệnh `contact`).
- Đo trên 1236 ID train bằng `contacts.search` thật (lệnh đọc, một ToolEnv mỗi mẫu):
  `python -B -m scripts.orca_v4_measure contact --data-root <kit>` → luật kích hoạt trên 0/161 truy vấn gold; với 11 câu nói
  "<xưng hô> tên" mà tên lưu không có từ xưng hô, truy vấn kèm từ xưng hô tìm không ra ở 11/11 và lần tìm lại trả đúng liên hệ
  gold ở 11/11 (cả dạng viết hoa lẫn viết thường). Điểm: NOT MEASURED.
- Test: `python -B -m pytest -q -p no:cacheprovider` → 360 passed, 1 skipped; `python -B -m src.inference mock --samples 40` → 40/40.
- Tiếp: ghi mục "Đã làm" trong `docs/RESEARCH_NEXT.md`.

## C-2 (15/11): ghi mục "Đã làm" trong `docs/RESEARCH_NEXT.md`

- Đã làm: mô tả cờ, lệnh đo, số đo (0/161 và 11/11), và giới hạn: trên smoke40 chỉ 0131 rơi vào luật (2.5 điểm đang mất),
  0665 không được sửa, và mẫu được cứu vẫn thừa một lời gọi nên không trọn điểm.
- Trạng thái: đã commit, chưa push. Root cần thêm `LIFEMATE_CONTACT_RETRY=1` vào lệnh suy luận nếu muốn bật.

# Việc phát hành (brief `lifemate-orca-release-brief.md`)

## P-1 (15/11): review masking v5 (`3dbe746`), sửa một lỗi

- Build đủ TRAIN trên CPU với `--mask-fraction 0.3 --fewshot-k 0` (và thêm một lượt với `LIFEMATE_CONTEXT_V4=1
  LIFEMATE_DATE_TABLE=1 --fewshot-k 2`): 398/1236 mẫu bị mask (32.2%); 0 lời gọi có tên không nằm trong tool của mẫu;
  0 `ask_missing_info.tool_name` lệch; 0 tên gốc còn sót trong mẫu bị mask; 0 tên mask trùng nhau; `totals["masked"]` = 398 khớp.
  Không cặp tên tool nào trong catalog chứa nhau. Ví dụ và kết quả của `CONTEXT_V4` được mask nhất quán.
- Lỗi thật: thay chuỗi không xét ranh giới làm hỏng hai loại `x-ref` trong định nghĩa tool: `music.playlists` (192 chỗ) và
  `flights.bookings` (52 chỗ) thành `fn_<hash>lists` / `fn_<hash>ings`, ở 120 mẫu bị mask. Không ảnh hưởng lời gọi.
- Sửa: `mask_tool_names` chỉ thay tên trọn vẹn (regex có ranh giới, đi từng chuỗi thay vì trên JSON đã dump).
  File: `scripts/build_sft.py`, `tests/test_orca_mask_whole_names.py`. Build lại: 398 mẫu mask, 0 tên mask dính chữ,
  120 mẫu khác bản cũ và chỉ khác ở hai loại `x-ref` đó.
- Test sau khi sửa: `python -B -m pytest -q -p no:cacheprovider` → 364 passed, 1 skipped.
- Tiếp: phần 2, cờ của bản được chọn trong Docker.

## P-2 (15/11): cờ của bản được chọn là một đầu vào của release

- Đã làm: `docker/release_flags.py` (allowlist 9 cờ bật/tắt có trong `src/`, chỉ nhận `TÊN=1`); `docker/release.ps1 -Flags`
  kiểm cờ trước mọi việc khác, truyền mỗi cờ thành `--build-arg`, và ghi `<archive>.manifest.json` (cờ, SHA256 adapter,
  SHA256 ASR, MD5 archive); `docker/Dockerfile` khai báo `ARG`/`ENV` cho đúng 9 cờ, đặt sau bước tải/merge model nên đổi cờ
  không phải build lại model. `predict.sh` và notebook không đặt hay xoá cờ nào (không cần sửa).
- File: `docker/release_flags.py`, `docker/release.ps1`, `docker/Dockerfile`, `tests/test_orca_release_flags.py`,
  `docs/SUBMISSION_CHECKLIST.md` (lệnh release cho v3 + ASR fine-tune, v4, v4 có guard, v5).
- Kiểm: `release.ps1` phân tích cú pháp 0 lỗi; các dòng xử lý cờ chạy trực tiếp trong PowerShell cho đúng `--build-arg` với
  đầu vào hợp lệ, 0 cờ với đầu vào rỗng, mã lỗi 1 với tên lạ. Máy này chặn chạy file `.ps1` (execution policy), nên chưa chạy
  cả script và chưa build image: NOT MEASURED.
- Test: `python -B -m pytest -q -p no:cacheprovider` → 379 passed, 1 skipped; `python -B -m src.inference mock --samples 40` → 40/40.
