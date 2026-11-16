# Planner mất điểm ở đâu khi transcript hoàn hảo, và nên đưa context gì

Viết 15/11 13:10, chỉ CPU, không sửa code. Nguồn: `outputs/colab/smoke_ab_drive/smoke_ab/{base,oracle_transcripts,w_fewshot2,v3}`,
gold TRAIN (`label.tool_calls`, `label.tool_results`). Tiến độ từng bước: [progress/orca.md](progress/orca.md).

Cách đọc: smoke40 có sai số chuẩn khoảng 6.6 điểm, một mẫu là 2.5 điểm. Bảng nhóm là phân loại thủ công theo lỗi đầu tiên
chặn điểm. "Trần trên" là số điểm đang mất trên các mẫu đó, không phải dự báo. Mức tăng thật của mọi đề xuất: NOT MEASURED.

## 1. Lượt `oracle_transcripts` (W + transcript gold): 49.92, mất 50.08

16 mẫu trọn điểm, 6 mẫu điểm lẻ, 18 mẫu 0. Cột cuối là điểm v3 (chạy với ASR) trên chính các ID đó.

| Hạng | Nhóm | Mẫu | Mất | ID và lỗi | v3 đạt |
|---|---|---:|---:|---|---:|
| 1 | Suy luận điều kiện trên kết quả tool | 6 | 13.13 | 0087, 0229 (chọn sai hoá đơn theo loại, so sai số dư); 0169 (ví 713.606 < 845.000 vẫn trả bằng ví); 0246 (tắt nhầm đèn phòng có quạt chạy); 1645 (`get_current` rồi dừng); 1360 (3000 điểm < 6000 vẫn đổi) | 2.5 / 15 |
| 2 | Thiếu `ask_missing_info` | 5 | 12.50 | 0086, 0350, 0413 (hai liên hệ trùng tên, model chọn cái đầu); 1296 ("tắt bớt một cái"); 1598 (tự đặt `term_months` 12) | 4.17 / 12.5 |
| 3 | Bỏ cuộc sai, cả hai chiều | 3 | 7.50 | 1582, 1655 (gold rỗng vì không có app email / thư viện, model làm việc khác rồi hỏi); 1041 (0 lệnh) | 2.5 / 7.5 |
| 4 | Sai giá trị đối số | 4 | 6.75 | 0373 (mờ 30% → 70); 0675 (email đọc đánh vần); 0131 (nội dung SMS viết lại quá xa); 0665 ("cô Hạnh") | 5.0 / 10 |
| 5 | Thừa/thiếu đối số tuỳ chọn hoặc tra cứu | 4 | 5.21 | 0707 (`update_event` thiếu `end`); 1202 (thừa `end`); 1095 (thừa `limit`); 0695 (tra cứu thừa) | 0 / 10 |
| 6 | Tính ngày giờ | 2 | 5.00 | 0188 (13:10 + 45 phút → 14:05); 0611 ("Chủ nhật này" → 02/10) | 0 / 5 |

Chi tiết nhóm 1: 0087 lấy 202.000 (hoá đơn điện thoại) cho "tiền nước" và không cộng 209.000 + 848.000 để so với số dư
904.681; 0229 lấy 206.000 (nước) cho "tiền mạng" và rút tiết kiệm dù tài khoản thanh toán đủ.

## 2. v3 so với base (cùng ASR, ghép cặp theo ID)

- Tăng 9 mẫu, +20.42 điểm: 0045, 0088, 0246, 0350, 0373, 0583, 1655 (0 → 1); 0665 (0.5 → 1); 0781 (0 → 0.67).
- Giảm 5 mẫu, −10.67 điểm: 0286 (0.91 → 0, trả nhầm hoá đơn nước); 0695 (0.86 → 0); 1161, 1360 (1 → 0, hành động trên gold rỗng);
  0177 (1 → 0.5). Ròng +9.75 = 34.58 − 24.83.
- v3 không còn bỏ cuộc sai (`missing_all_actions` 15.0 → 0) nhưng gold rỗng chỉ đúng 2/5: lỗi đã đổi chiều.
- 9 mẫu W trọn điểm với transcript gold mà v3 chưa trọn: 0177, 0280, 0286, 0578, 0781, 0937, 1012, 1015, 1161.
- Chưa có lượt v3 với transcript gold, nên không tách được phần ASR và phần planner của v3: NOT MEASURED.

## 3. Cách sửa ba nhóm đầu

### 3.1 Thiếu hỏi khi kết quả trùng tên: hậu xử lý tất định, không cần train

Căn cứ trên 1236 ID train: trong 104 lần gold hỏi một ID, 53 lần kết quả tra cứu ngay trước có bản ghi trùng tên; trong 954
lời gọi gold dùng một ID, chỉ 6 lần ID đó thuộc nhóm trùng tên. Riêng `contacts.search`: một kết quả thì 107/107 dùng thẳng,
từ hai kết quả trùng tên thì 49/51 hỏi.

```
khi planner gọi tool T với đối số p (có x-ref) = v:
  nếu v là id của một bản ghi trong danh sách kết quả của lượt hiện tại,
  và danh sách đó có bản ghi khác cùng name/title (bỏ hoa-thường)
  → thay lời gọi bằng sys.ask_missing_info{"tool_name": T, "params": [p]}
```

Chỗ đặt: `src/agent/runner.py`, cạnh guard x-ref trước khi dispatch, sau cờ mặc định tắt. Không đổi prompt nên dùng được
ngay cho W và v3. Trần trên lượt oracle: 7.5 (0086, 0350, 0413). Còn lại 5.0 (1296, 1598) chưa có luật tất định: gold hỏi
một giá trị 76 lần trong train (`ewallet.transfer.amount` 18, `reminders.create.time` 7), phải dựa vào model.

### 3.2 Hành động khi gold rỗng: nêu rõ app không cài, cần train lại

Căn cứ: đọc tay 89 mẫu train gold rỗng, 33 mẫu là do cần app hoặc khả năng không có trên máy (26 đổi ý, 11 điều kiện đã sai
theo lịch sử, 10 trò chuyện, 5 không hợp lệ, 4 khác). Catalog train có 46 app, mỗi mẫu cài trung vị 9. Trên smoke: 1582 và
1655 ở lượt oracle; 1161 và 1582 ở v3 (đều thiếu app email hoặc thư viện).

Thêm vào cuối system prompt, tên lấy từ `name_vi` của catalog của chính split đang chạy:

```
Ứng dụng đã cài: Lịch (calendar), Ghi chú (notes), ...
Ứng dụng KHÔNG cài trên máy này: Email (email), Thư viện (library), ...
Việc nào cần một ứng dụng không cài thì bỏ qua việc đó: không thay bằng ứng dụng khác, không hỏi lại.
```

Chỗ đặt: `src/inference/pipeline.py::_sample_from_row` ghi danh sách app không cài vào `context`; `native_planner.build_messages`
in ba dòng trên; `build_sft.py` tự có vì dùng chung `build_messages`. Cần train lại. Trần trên lượt oracle: 5.0 (1582, 1655);
1041 (2.5) là bỏ cuộc sai chiều ngược, v3 không còn lỗi này. Số token thêm: NOT MEASURED.

### 3.3 Suy luận điều kiện: ví dụ kèm bằng chứng và chú giải enum, cần train lại

Căn cứ: 474 câu train có từ điều kiện; 168 câu có bản sao Jaccard ≥ 0.5, trong đó 41 câu khác chuỗi tool và 8 câu khác một
đối số enum. Tức khoảng 29% bản sao đi nhánh khác, mà ví dụ hiện nay (`"câu" → lệnh; lệnh`) không cho thấy vì sao.
Hai trong sáu mẫu nhóm 1 chọn sai hoá đơn theo loại; catalog chỉ có enum tiếng Anh (`water`, `internet`), không có chú giải.

(a) Ví dụ truy hồi kèm kết quả của lệnh tra cứu, cắt mỗi kết quả ở một số ký tự cố định, bỏ kết quả của lệnh ghi:

```
- "Trả tiền mạng bằng ví điện tử giúp chị, ví không đủ thì trừ tài khoản ngân hàng nhé."
  bill_payment.list_bills{} ⇒ {"bills":[{"id":"bill_01","type":"internet","amount":219000,...}]}
  ewallet.get_balance{} ⇒ {"balance":713606}
  bill_payment.pay{"bill_id":"bill_01","source":"ewallet"}
```

Chỗ đặt: `build_sft.py` ghi thêm `results` vào `src/agent/data/fewshot_native.jsonl`; `native_planner.render_examples` in dạng
trên; nâng ngân sách ký tự của `fewshot.select`.

(b) Chú giải tiếng Việt cho giá trị enum trong mô tả tham số, ví dụ `type: electricity = điện; water = nước; internet = mạng;
tv = truyền hình; tuition = học phí; phone = điện thoại`. Chỗ đặt: một bảng viết tay trong `native_planner.tool_definitions`,
phủ enum của catalog train và public; enum lạ thì để nguyên.

Cả hai đổi prompt nên cần train lại. Trần trên lượt oracle: 13.13 (cả nhóm 1). Chưa có cách tất định cho phép so sánh số
(904.681 với 209.000 + 848.000); một dòng nháp trước lệnh ghi cần dữ liệu train có lời giải, không kịp trước hạn.

## 4. Nhóm còn lại, việc nhỏ

- `calendar.update_event`: gold luôn có `end` khi đổi `start` (27/27). Có thể bù `end` = `start` + thời lượng cũ. Trần 2.5 (0707).
- Email đọc đánh vần ("a còng", "chấm"): 7/1236 câu train. Chuẩn hoá trong `asr_fix`. Trần 2.5 (0675).
- `read_chat` gold không bao giờ có `limit` (0/11); `create_event` có `end` 23/34 lần, không rõ quy ước.
- "N phút/tiếng nữa" chỉ có ở 3/329 câu train có đối số ngày giờ; `LIFEMATE_DATE_TABLE` trên W đo được 25.06 so với 24.83.

## 5. Kế hoạch tới 12:00 ngày 16/11

| Thứ tự | Việc | GPU | Train lại | Trần trên lượt oracle |
|---|---|---|---|---:|
| 1 | Guard trùng tên (3.1): code + test trên CPU, rồi smoke40 trên v3 | ~9 phút | không | 7.5 |
| 2 | Chạy v3 với `--train-transcripts` để biết phần planner của v3 | ~9 phút | không | phép đo |
| 3 | v4: một lượt QLoRA gộp 3.2 + 3.3a + 3.3b, transcript từ ASR tốt nhất đang có | 2–3 h | có | 18.13 |
| 4 | Bù `end` cho `update_event`, chuẩn hoá email đánh vần | ~9 phút | không | 5.0 |

- Chiều 15/11: code việc 1, 3, 4 sau cờ mặc định tắt, build dữ liệu v4 trên CPU. Tối 15/11: train v4 (v3 mất 71 s/bước trên L4,
  154 bước; thời gian v4: NOT MEASURED vì prompt dài hơn). Sáng 16/11: smoke40 của v4 và v3 có guard, chọn bản, build Docker.
- Việc 3 gộp ba thay đổi trong một lượt nên không tách được tác dụng từng cái. Nếu v4 không hơn v3 rõ rệt thì nộp v3 + việc 1.
- Việc 2 quyết định thứ tự ưu tiên: nếu v3 với transcript gold đã cao hơn W nhiều thì phần còn lại chủ yếu là ASR.

## Chưa đo (NOT MEASURED)

Điểm của mọi đề xuất; v3 với transcript gold; số token và độ trễ thêm của 3.2 và 3.3; tỷ lệ bản sao và app không cài trên
public test; hành vi của guard 3.1 khi model chọn ID từ lịch sử thay vì từ lượt hiện tại.

## Đã làm (15/11 15:15, CPU; mọi điểm số: NOT MEASURED)

Ba cờ, mặc định tắt. Tắt cờ thì prompt và lời gọi không đổi: build SFT mặc định và `--fewshot-k 2` trùng byte với bản trước.
`<kit>` = thư mục `train+public_test`.

| Cờ | Làm gì | Ở đâu | Cần train |
|---|---|---|---|
| `LIFEMATE_DUP_GUARD=1` | Mục 3.1: ID thuộc nhóm bản ghi trùng tên trong kết quả của lượt hiện tại → `sys.ask_missing_info{"tool_name": T, "params": [p]}` | `runner.py`, trước `registry.authorize` | không |
| `LIFEMATE_CONTEXT_V4=1` | Mục 3.2 + 3.3: ba dòng app đã cài / không cài; ví dụ in từng lệnh, lệnh đọc kèm `⇒ kết quả` (cắt 500 ký tự, ngân sách 6000); chú giải 85 giá trị enum | `pipeline._sample_from_row`, `native_planner.build_messages` / `tool_definitions`, `fewshot.select`, pool do `build_sft.py --fewshot-k` ghi | có; đặt cờ cho cả `build_sft.py` và suy luận |
| `LIFEMATE_SMALL_FIXES=1` | Mục 4: bù `end` cho `calendar.update_event`; ghép email đọc đánh vần | `runner.py`; lớp bọc transcriber trong `asr_fix.py` | không (phần email đổi transcript, nên đặt cả khi build SFT) |

Số đo:

- Guard trùng tên trên gold của 1236 ID train: đổi 6/979 lời gọi dùng ID; tái tạo đúng 53/112 lần gold hỏi một ID.
  `python -B -m scripts.v4_measure dup --data-root <kit>`
- `CONTEXT_V4` trên smoke40 (K=2, transcript gold): system prompt thêm median 2034 ký tự (p95 3041, max 3110); định nghĩa
  tool thêm median 650 (p95 1096); tổng thêm median 2505 (p95 3778, max 3885). Bản ghi SFT: ký tự p50 18,214 → 20,667,
  max 33,193 → 37,872. Trên kit thật, prompt và định nghĩa tool lúc suy luận trùng bản ghi SFT ở 50/50 ID train lấy mẫu.
  `python -B -m scripts.v4_measure prompt --data-root <kit>`
- `SMALL_FIXES` trên 1236 ID train: 27/27 lời gọi gold dời `start` có `end` = `start` mới + thời lượng cũ; luật email ra đúng
  địa chỉ gold ở 7/7 câu đọc đánh vần với transcript gold, 0/7 với transcript PhoWhisper-medium, không đổi câu nào khác.
  `python -B -m scripts.v4_measure small --data-root <kit>`
- Test: `python -B -m pytest -q -p no:cacheprovider` → 344 passed, 1 skipped; `python -B -m src.inference mock --samples 40` → 40/40.

Lưu ý cho lượt GPU: pool `src/agent/data/fewshot_native.jsonl` đã đổi (thêm `results`, 1,085,562 byte), sau `build_sft.py
--fewshot-k 2` thì `git status` phải sạch. Số token của prompt v4 chưa đo; v3 có max 9,714 token ở 33,193 ký tự, nên với
`--max-tokens 12000` cần xem dòng `rows` của `train_qlora.py` để biết có mẫu nào bị loại.
