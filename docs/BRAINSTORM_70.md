# Từ 42 lên 70: phải đổi gì trong cách dựng dữ liệu

Viết 15/11 23:57 (Orca), chỉ CPU, không sửa code. Không ai trực nên tôi tự trả lời các câu hỏi; mỗi giả định ghi ở mục cuối.
Nguồn: NEXT_PLAN, PUBLIC_GAP, V6_PLAN, RESEARCH_NEXT, CONTEXT_PLAN, COMPETITION_RULES, `scripts/build_sft.py`,
`src/agent/native_planner.py`, `src/agent/fewshot.py`, gold TRAIN. `docs/ERROR_BUDGET.md` của Codex lúc viết mới có
2 dòng nháp, chưa dùng được.

Cách đọc: "trần" là điểm đang mất trên các mẫu mà thay đổi nhắm tới, không phải dự báo. Public không có nhãn nên trần public
thường không tách được: ghi NOT MEASURED. smoke40 có sai số chuẩn khoảng 6.6 điểm và đã không dự báo được public
(v4 +4.6 trên smoke, public trộn 122 v4 + 512 v3 = 42.07 so với 42.25).

## 1. Hiện trạng: điểm mất nằm ở đâu

- Public 42.25 (v3 + ASR fine-tune). smoke40: v3 51.73, v4 56.29, v5 50.19.
- Với transcript gold, planner W chỉ đạt 49.92 trên smoke40; phần còn do ASR trên v3 là 8.4 điểm (4 mẫu). Tức là kể cả
  ASR hoàn hảo, planner hiện tại còn cách 70 rất xa: **giới hạn chính là planner, không phải ASR**.
- smoke40 của v3 + ASR mới mất 48.27 điểm: điều kiện/nhiều bước 13.54, sai giá trị đối số 11.25, ngày giờ 10.0, thiếu hỏi
  5.0, làm thừa trên gold rỗng 5.0, đối số tuỳ chọn 3.48 (RESEARCH_NEXT mục 1).
- Public thấp hơn smoke 6.6–9.5 điểm; 410/634 mẫu public có app không có trong TRAIN, model hỏi thiếu cho app lạ 2.5% so với
  7.3% cho app quen, và 67 lần gọi sang app train "anh em" (PUBLIC_GAP).
- Dữ liệu train rất ít: **1236 mẫu, 1 epoch** (154 bước). Mọi cải tiến từ v3 tới v6 đều là đổi prompt hoặc biến đổi chính
  1236 mẫu đó.

## 2. Ba hướng

| Hướng | Làm gì | Được gì | Mất gì |
|---|---|---|---|
| A. Thêm dữ liệu thật có sẵn | Biến mỗi lượt trợ lý trong lịch sử TRAIN thành một mẫu SFT | Dữ liệu ×2–3, cùng phân bố, cùng ASR, không cần sinh | GPU train lâu hơn; phân bố lượt lịch sử dễ hơn (ít hỏi thiếu) |
| B. Tổng quát hoá cho tool lạ | Đổi tên/che tên (v5, v6), dữ liệu tổng hợp từ app train | Nhắm thẳng vào khoảng cách public/smoke | v5 làm smoke giảm; chỉ public đo được; số lần nộp có hạn |
| C. Ngoài dữ liệu | Luật hậu xử lý, tự kiểm, model planner khác | Có thể nhanh cho từng lỗi | Các luật trước (`DUP_GUARD`, `CONTACT_RETRY`) có trần smoke gần 0; đổi model là làm lại từ đầu |

**Tôi chọn A làm trục chính, giữ B ở mức v6 đang train, bỏ C trừ ASR-large đã xếp hàng.** Lý do: A là hướng duy nhất tăng
lượng tín hiệu học mà không cần nhãn mới hay dữ liệu tổng hợp, và nó đánh vào cả hai giới hạn (planner trên smoke, tổng quát
hoá trên public). B đã có v6 đang chạy; thêm một biến thể B nữa trước khi biết public của v6 là đoán mò.

## 3. Thay đổi đề xuất, xếp hạng

**1. Mẫu SFT từ lượt lịch sử (hướng A).** Orca.
- Cái gì: `build_sft.py --history-turns F`: với mỗi lượt trợ lý thứ k trong lịch sử, dựng một mẫu có lịch sử = các lượt trước
  k, câu nói = lượt user k (ASR qua `--transcripts`, đã có cho cả lượt lịch sử), đích = lời gọi đã ghi kèm kết quả đã ghi
  (không cần ToolEnv), lượt không gọi tool thì đích là câu "không thực hiện thao tác nào". Chọn F phần lượt bằng hash ổn định.
  Ví dụ truy hồi dùng đúng `build_messages` hiện tại; pool loại chính ID gốc nên không lộ lượt sau.
- Vì sao: 1236 ID train có **2826 lượt trợ lý trong lịch sử** (825 mẫu có lịch sử; 2420 lượt có lời gọi, 2954 lời gọi, nhiều
  hơn 2840 lời gọi gold); 2821 lượt là duy nhất; chỉ 1 lời gọi không qua schema; 0 kết quả lỗi. ASR cho các lượt user này đã
  nằm trong `asr_train_ft.jsonl` (4062 dòng = 1236 + 2826). Lượt lịch sử cũng có `current_time` như lượt cuối (vd. train_0086:
  "9 giờ sáng mai" → 2026-11-23T09:00 với current_time 22/11), nên ngữ cảnh khớp.
- Trần: toàn bộ điểm planner đang mất trên smoke40 của v3 + ASR mới là 26.9 (12 mẫu sai ở cả ba lượt) + 13.0 (lỗi riêng của v3);
  public: không tách được, NOT MEASURED.
- Chi phí: CPU khoảng 2 h (code + test + build). GPU: số dòng 1236 + 0.5 × 2826 ≈ 2649 → khoảng 2.1 lần v4, ước lượng 90 phút
  trên A100 (v4: 154 bước ≈ 43 phút; prompt lượt lịch sử không dài hơn: trung vị 2 lượt trước đó).
- Rủi ro: lượt lịch sử chỉ có 17/2826 lần hỏi thiếu (0.6%) so với 14.6% mẫu gold, nên tỷ lệ hỏi trong dữ liệu giảm từ
  180/1236 (14.6%) xuống khoảng 189/2649 (7.1%) nếu lấy 50%; public vốn đã hỏi thiếu ít. Đo tỷ lệ hỏi của v7 trên smoke và VAL.
- File: `scripts/build_sft.py`, `tests/test_orca_v7_history.py`.

**2. Một phép đo thay cho smoke40 (bắt buộc cho mọi quyết định còn lại).** Codex.
- Cái gì: chạy 309 ID validation (đã có `research/validation_split.json`, chứa smoke40) với ASR fine-tune, TIME_SCALE=1, chấm
  bằng `score_run`; gold replay riêng cho 309 ID.
- Vì sao: smoke40 nói v4 hơn v3 4.6 điểm, public nói không. 309 mẫu cho sai số chuẩn khoảng 6.6 × √(40/309) ≈ 2.4 điểm
  (ước lượng cùng độ lệch từng mẫu), đỡ tốn lượt nộp public.
- Chi phí: khoảng 50 phút A100 mỗi model (309 mẫu với tốc độ public v4 ~6 mẫu/phút). Không tốn lượt nộp.
- Rủi ro: 309 ID vẫn không có tool lạ; chỉ dùng để chọn giữa các planner, không thay public cho phần tool lạ.
- File: `scripts/colab_val.sh` (mới), dùng lại `colab_lib.sh` với biến `SMOKE`/`GOLD` riêng.

**3. Gộp v6 vào v7 chỉ khi public v6 ≥ 42.25.** Lead.
- Cái gì: v7 = (v6 nếu public v6 không thấp hơn v3, nếu không thì v4) + mục 1. Đổi tên/cổng truy hồi của v6 áp lên cả mẫu lượt
  lịch sử (cùng `rename_tool_names`, cùng `LIFEMATE_FEWSHOT_GATE`).
- Vì sao: PUBLIC_GAP: 410/634 mẫu có app lạ; v5 (che `fn_`) làm smoke giảm 6.1, nên chỉ giữ phần B nào public xác nhận.
- Trần public của phần tool lạ: 67 mẫu thay app anh em ≈ 10.6 điểm (cách tính trong PUBLIC_GAP). Chi phí: 0 thêm.
- File: `scripts/colab_v7.sh` (mới, Codex viết theo mẫu `colab_v6.sh`).

**4. ASR-large (đang xếp hàng).** Codex, đã có `colab_asr_large.sh`.
- Trần smoke: 8.4 điểm (4 mẫu còn do ASR). Chỉ nhận nếu WER validation thấp hơn 9.9% rõ rệt. Nếu nhận, phải dựng lại
  transcript TRAIN bằng ASR-large trước khi train v7, nếu không train/chạy lệch nhau (v3 train trên ASR cũ, chạy ASR mới: 7 mẫu
  W làm tốt hơn v3 với cùng ASR, 13.0 điểm smoke; lệch ASR là giả thuyết của tôi, chưa tách được).

**5. Thêm 269 ID validation ngoài smoke vào bản cuối, nếu còn một lượt train.** Lead.
- `build_sft.py --exclude research/smoke_split.json` đã làm được (manifest có `validation_ids` = 40), không sửa code. Thêm 269 mẫu
  cuối + 588 lượt lịch sử. Mất phép đo của mục 2 cho chính bản đó, nên chỉ làm sau khi đã chọn công thức bằng mục 2.

Không đưa vào danh sách (đã xét): dữ liệu tổng hợp từ catalog public (brief v6 cấm chép tên public; rủi ro luật), học theo
giáo trình hay đổi định dạng đích (không có số đo nào chỉ ra), nhân đôi mẫu bằng cả chữ gold và chữ ASR (2 lần mỗi mẫu trong
1 epoch; v3/v4 đều 1 epoch để tránh học thuộc), tăng epoch (loss train đã xuống 0.06 ở bước 80 của bước 154 cũ: thiếu dữ liệu
chứ không thiếu lượt học).

## 4. Lịch tới 17:00 (A100, theo hàng đợi trong HANDOFF 23:50)

| Giờ | GPU | CPU |
|---|---|---|
| 00:00–00:45 | v6 train (đang chạy) → smoke v6 | Orca viết mục 1 (code + test); Codex viết `colab_val.sh`, `colab_v7.sh` |
| 00:45–02:30 | public v6 nếu qua cổng | Codex review mục 1; build SFT v7 thử trên CPU (`--limit 50`) |
| 02:30–03:30 | ASR-large fine-tune + smoke | nộp public v6; ghi điểm |
| 03:30–04:20 | VAL309: v4 (và v6) — mục 2 | chốt công thức v7 theo public v6 + VAL |
| 04:20–06:00 | train v7 (ước lượng 90 phút) | |
| 06:00–07:00 | smoke40 + VAL309 v7 | |
| 07:00–08:45 | public v7 nếu VAL v7 > VAL của bản đang giữ | |
| 09:00–13:00 | (tuỳ chọn) mục 5: train lại trên 1505 ID + VAL không dùng được, chỉ public | |
| 13:00–17:00 | dự phòng; chốt bản cuối trước 17:00 | chuẩn bị lệnh release đúng cờ; Docker sau 17:00 (NEXT_PLAN) |

## 5. Điều sẽ không đạt và vì sao

70 trên public trước 17:00 là không thực tế. Với transcript gold, planner W đạt 49.92 trên smoke40; bản tốt nhất có smoke
56.29 nhưng public 42.07–42.25. Để lên 70 cần thêm 27.75 điểm public, tức khoảng 176 trong 634 mẫu từ 0 lên trọn điểm.
Các thay đổi trên chỉ có trần cộng dồn cỡ vài chục điểm smoke (không cộng được với nhau, trùng mẫu), còn trên public phần lớn
là NOT MEASURED. Phần còn thiếu nằm ở suy luận nhiều bước trên kết quả tool (13.54 điểm smoke ở 7 mẫu) và tool lạ (410/634
mẫu public), cả hai cần dữ liệu có lời giải từng bước hoặc nhiều dữ liệu tool khác loại, không làm kịp trong một đêm.
Mục tiêu thực tế trước 17:00: một bản có public cao hơn 42.25, đo bằng VAL309 trước khi tốn lượt nộp.

## Giả định (tự trả lời vì không ai trực)

- Lượt trợ lý trong lịch sử TRAIN có chất lượng như nhãn gold (cùng bộ sinh dữ liệu; chỉ kiểm được schema: 1 lỗi / 2954).
- Lấy 50% lượt lịch sử để vừa lịch GPU; con số không có căn cứ nào khác ngoài thời gian.
- Tốc độ train tỉ lệ với số dòng; số token của mẫu lượt lịch sử: NOT MEASURED.
- Không dùng tên hay schema của catalog public để sinh dữ liệu train.
- Tốc độ chạy 309 ID bằng tốc độ public v4 trên A100 (~6 mẫu/phút).
