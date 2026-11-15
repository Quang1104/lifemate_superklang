# Kế hoạch tới hạn nộp (viết 15/11 22:50, root)

Hạn: Docker nộp 16/11 23:59 GMT+7. Đóng băng bản cuối: **16/11 17:00**, sau đó chỉ build/kiểm/xuất Docker (cần ~2 h cả build, chạy thử offline, xuất, MD5, upload).

## Số đo hiện có (smoke40 = 40 ID TRAIN giữ riêng, TIME_SCALE=1; public = portal, hạng mục SLM)

| Bản | smoke40 | public |
|---|---:|---:|
| v3 (ASR gốc, K=2) | 34.58 | 27.98 |
| v3 + ASR fine-tune (PhoWhisper-medium, WER 28.1% → 9.9%) | 51.73 | **42.25** |
| v4 = v3 + `CONTEXT_V4` + `SMALL_FIXES`, train trên transcript ASR mới | **56.29** | NOT MEASURED (public v4 đang chờ chạy sau v5 trên A100) |
| v5 = v4 + `DATE_TABLE` + che tên tool 30% | NOT MEASURED (đang train trên A100) | NOT MEASURED |

Public thấp hơn smoke ~6.6–9.5 điểm; [PUBLIC_GAP.md](PUBLIC_GAP.md): 410/634 mẫu public có app không có trong TRAIN; với app lạ model ít hỏi lại (2.5% vs 7.3%), hay dừng sau lệnh đọc, 67 lần gọi sang app "anh em" của TRAIN. A100 80GB: QLoRA 154 bước ≈ 43 phút, public 634 mẫu ≈ 25 phút (ước), ASR fine-tune 3 epoch < 10 phút.

## Việc cần làm, đã lọc (trần = điểm đang mất, không phải dự báo)

1. **[GPU, đang chạy] v5 → smoke; public v4 (+ v5 nếu smoke v5 > v4).** Nộp bản public cao hơn 42.25.
2. **[Orca] v6 dữ liệu cho tool lạ** (PUBLIC_GAP #2, #3): đổi tên app/tool TRAIN sang tên khác *có nghĩa* trên một phần mẫu (thay cho/bổ sung `fn_<hash>`), áp nhất quán cho tools, lời gọi, lịch sử, ví dụ, `ask_missing_info.tool_name`; bỏ ví dụ truy hồi khi độ giống của ví dụ đầu dưới ngưỡng τ (cùng luật lúc build và lúc chạy). Cờ mặc định tắt, test, đo trên CPU. Trần ≈ 10.6 điểm public (67 mẫu thay app anh em) — đo bằng public.
3. **[Codex] PhoWhisper-large fine-tune** (ASR còn mất 8.4 điểm trên smoke): `train_asr.py` đã tổng quát; cần `colab_asr_large.sh` (tải pin large trước 26/10/2026, fine-tune, smoke W/v4 với ASR large), kiểm tổng tham số < 6B (planner 4,022,468,096 + large ~1.55B), độ trễ/mẫu < 30 s, Docker chứa được (dung lượng). Chấp nhận nếu WER validation < 9.9% rõ và smoke ≥ +5.
4. **[Codex review] mọi thay đổi của Orca** trước khi train v6 (lệch prompt train/chạy là lỗi đắt nhất).
5. **[GPU sáng 16/11] v6 = bản tốt nhất + mục 2 (+ ASR large nếu thắng)** → smoke → public. Mỗi bản chỉ giữ khi public tăng.
6. **[Root/Codex] Docker bản cuối** trước 17:00 build: `docker/release.ps1 -AdapterDir <adapter cuối, có thư mục asr>` + `-Flags` đúng bản; chạy thử offline trên vài mẫu, xuất `.tar.gz` + MD5; ổ C còn ~54 GB.

Không làm trước hạn: SFT kèm lời giải từng bước (cần sinh ~474 lời giải, rủi ro thời gian), tối ưu theo smoke40 cho tool lạ (smoke40 không có tool lạ), đổi K không train lại.

## Quy tắc
AGENTS.md; không bịa số (chưa đo = NOT MEASURED); không dòng ghi công AI trong commit; mọi brief có mục Progress (HANDOFF quy tắc 6); push `colab-smoke`, không push `main`.
