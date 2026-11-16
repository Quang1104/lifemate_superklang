# Sửa transcript ASR theo ngữ cảnh (`LIFEMATE_ASR_FIX=1`)

Viết 15/11, chỉ CPU. Code: `src/agent/asr_fix.py`, nối ở một dòng trong `src/inference/pipeline.py::_sample_from_row`
(bọc transcriber, nên áp dụng cho câu hiện tại, lượt user trong lịch sử, và cả `build_sft.py --transcripts`).
Không đặt cờ thì trả lại đúng transcriber cũ. Điểm thi: **NOT MEASURED** cho tới khi có smoke trên GPU.

## Kết luận
- Sửa chữ chỉ gỡ được một phần nhỏ: WER 28.68% → 28.58% (tune), 29.12% → 29.07% (held-out).
  88.4% số từ gold bị sai là lỗi nghe (phương ngữ, từ gần âm), 8.4% là từ viết hoa, 3.2% là từ có chữ số.
- Phần có ích cho planner là chữ hoa của tên riêng: giá trị đối số gold dạng tên xuất hiện nguyên văn trong
  transcript tăng từ 0/507 lên 55/507 (tune) và 0/120 lên 15/120 (held-out); trần với transcript gold là 181 và 45.
- Giả định "gold viết số bằng chữ số" chỉ đúng với thiểu số: 381/4062 câu gold có chữ số (29.2% ở ID < 400,
  3.3% ở ID ≥ 400). Đổi mọi số sang chữ số làm WER tăng, nên module không làm việc đó.

## Luật (chọn trên TRAIN ngoài 309 ID validation)
1. **Mã có số**: `ron chín lăm` → `RON95`, tương tự E, VTV, HTV, SCTV, VOV, RV. Chỉ ghép khi phần sau là đúng
   một số; đọc từng chữ số (`rv một không bảy`) để nguyên.
2. **Tên trong ngữ cảnh mẫu** (hồ sơ người dùng, `text`/`tool_calls`/`tool_results` của lượt trợ lý): cụm trùng
   tên khi bỏ hoa-thường thì lấy cách viết của tên. Tên từ hai từ trở lên còn khớp khi chỉ lệch dấu thanh,
   trừ khi mọi từ nghe được đều là từ thường trong gold (`anh khoá` không thành `Anh Khoa`).
   Từ đầu cụm mà gold thường viết thường thì giữ thường (`anh Hùng`).
3. **Từ gold luôn viết hoa**: từ được viết hoa ≥ 5 lần và ≥ 10 lần số lần viết thường trong 1236 câu TRAIN
   (`Lan`, `Zalo`, `SMS`). Bảng đếm lấy từ `src/agent/data/fewshot_native.jsonl` đã có sẵn.
   Từ đơn mà gold thường viết thường không bao giờ bị đổi (`sáng mai` không thành `sáng Mai`).

Đã thử và bỏ: đổi số trước đơn vị. Số câu gold viết chữ số / viết chữ: giờ 84/447, nghìn 22/84, rưỡi 16/87,
phút 28/49, tiếng 3/52, độ 44/32, triệu 36/35, phần trăm 16/14. Trên tune chỉ `độ` (−18 lỗi) và `triệu` (−59)
giảm lỗi, `giờ` tăng 226; trên held-out `độ` +2, `triệu` 0, nên cả hai bị bỏ. Đây là lần duy nhất held-out
được dùng để quyết định.

## Số đo
`tune` = ID có số không chia hết cho 5 (3259 audio); `held-out` = phần còn lại (803 audio, bảng chữ hoa dựng lại không có chúng). WER như HANDOFF: chữ thường, `\w+`.

| Đại lượng | tune trước → sau | held-out trước → sau |
|---|---|---|
| WER | 28.68% → 28.58% (13413 → 13365 lỗi / 46767 từ) | 29.12% → 29.07% (3363 → 3357 / 11549) |
| WER phân biệt hoa-thường | 36.61% → 35.02% | 36.93% → 35.47% |
| Từ gold có chữ số chưa đúng | 396/400 → 378/400 | 81/82 → 78/82 |
| Từ gold viết hoa (không đầu câu) chưa đúng | 2458/2458 → 1754/2458 | 576/576 → 414/576 |
| Từ gold khác chưa đúng | 10500/43909 → 10513/43909 | 2751/10891 → 2757/10891 |
| Đối số có chữ hoa, nguyên văn trong transcript | 0/507 → 55/507 (gold: 181) | 0/120 → 15/120 (gold: 45) |
| Đối số có chữ số, nguyên văn | 0/220 → 1/220 (gold: 6) | 0/57 → 0/57 (gold: 1) |
| Đối số chữ thường, nguyên văn | 39/112 → 39/112 (gold: 65) | 9/29 → 9/29 (gold: 15) |

Đối số = giá trị chuỗi của lời gọi gold, bỏ ID (`x-ref`), enum, thời gian và `sys.*`. Theo từng luật trên tune (đo trước khi siết
luật lệch dấu): tên ngữ cảnh đổi 336 câu (326 tốt hơn, 7 xấu hơn); bảng chữ hoa 432 (338, 3); mã có số 20 (19, 0).

## Lệnh
```
python -B -m src.agent.asr_fix --data-root "<kit>" --transcripts outputs/asr_train.jsonl   # in ra bảng trên
python -B -m pytest -q -p no:cacheprovider      # 305 passed, 1 skipped, 1 warning, 146 subtests passed
python -B -m src.inference mock --samples 40    # checks_passed 40
smoke "$M/sft154" asrfix LIFEMATE_ASR_FIX=1     # GPU, hàm của scripts/colab_lib.sh; chưa chạy
```

## Còn lại
- Điểm smoke40 của W và v3 khi bật cờ: NOT MEASURED. Muốn train khớp thì đặt cờ khi chạy `build_sft.py --transcripts`.
- 309 ID validation: NOT MEASURED, vì `outputs/asr_train.jsonl` không có transcript của chúng.
- Với ASR đã fine-tune hoặc `LIFEMATE_ASR_PROMPT`: văn bản đã viết hoa đi qua nguyên vẹn (có test); tác dụng cộng thêm NOT MEASURED.
- Lỗi nghe (`mình`/`Minh`, `stc`/`SJC`, `chỉ`/`chị`, phương ngữ) và kiểu đặt dấu `hòa`/`hoà`: chưa xử lý, cần ASR tốt hơn.
