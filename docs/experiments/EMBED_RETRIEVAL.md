# Chọn ví dụ bằng embedding (`LIFEMATE_EMBED_RETRIEVAL=1`)

Viết 16/11, chỉ CPU. Điểm VAL309 / public khi bật cờ: NOT MEASURED (chạy `scripts/colab_val_embed.sh`).

## Thay đổi

- Cờ tắt (mặc định): prompt và file SFT trùng từng byte với trước (sha256 bản mặc định `340f7fad…`, cấu hình v7 `185d4531…`).
- Cờ bật: `fewshot.Examples.select` giữ nguyên bộ lọc (không lấy chính mẫu, tool của ví dụ ⊆ tool đã cài), cổng
  `LIFEMATE_FEWSHOT_GATE` (vẫn tính trên độ trùng từ như lúc train), ngân sách ký tự, khử trùng lặp và K; chỉ thứ tự ứng viên
  đổi sang cosine của embedding câu nói. Định dạng prompt không đổi nên thử được trên model v7 mà không train lại.
- `src/agent/embed_retrieval.py`: nạp model từ thư mục cục bộ (`LIFEMATE_EMBEDDER_DIR`, mặc định `/models/embedder`,
  `local_files_only`), mean pooling, chuẩn hoá L2, tiền tố `query: `. `api.load_models` nạp model và nhúng pool ngay lúc tải
  model (không tính vào 30 giây của mẫu đầu) và cộng tham số của nó vào kiểm tra < 6 tỷ.
- Docker: `release.ps1 -Flags "...,LIFEMATE_EMBED_RETRIEVAL=1"` → Dockerfile tải embedder ở một lớp riêng
  (`fetch_models.py --embedder /models/embedder`) chỉ khi có cờ. Colab: `python -B -m scripts.fetch_models --roles embedder`.

## Model (số từ Hugging Face API, 16/11)

`intfloat/multilingual-e5-small`: giấy phép MIT, tạo 2023-06-30, có nhãn ngôn ngữ `vi`, 117,654,272 tham số
(`model.parameters()` đếm 117,653,760). Pin `614241f622f53c4eeff9890bdc4f31cfecc418b3` (commit 2026-04-02, trước 26/10/2026)
trong `configs/colab_models.json`, vai `embedder`. Tổng tham số: 4,022,468,096 (planner) + 763,857,920 (PhoWhisper-medium) +
117,654,272 = 4,903,980,288 < 6 tỷ; với PhoWhisper-large: 5,565,773,056 + 117,654,272 = 5,683,427,328 < 6 tỷ.

## Đo trên CPU: VAL309, chữ gold, K=2

Thước đo có dùng nhãn TRAIN: ví dụ được chọn có đúng chuỗi tên tool của gold hay không.

| Cách xếp hạng | Ví dụ đầu cùng chuỗi tool | Một trong hai cùng chuỗi tool |
|---|---:|---:|
| Trùng từ + 0.2 × app (hiện tại) | 120/309 | 142/309 |
| Cosine (bản đã cài đặt) | 121/309 | 148/309 |
| Cosine + 0.05 × app | 126/309 | 150/309 |
| Cosine + 0.2 × app | 106/309 | 128/309 |
| Trùng từ + 4 × cosine | 130/309 | 150/309 |

- Embedding chọn ví dụ đầu khác trùng từ ở 202/309 mẫu, bộ hai ví dụ khác ở 259/309 mẫu, nhưng thước đo trên gần như không
  đổi (121 so với 120; 148 so với 142). Không biến thể nào hơn rõ; chọn biến thể tốt nhất trên chính 309 mẫu này là chỉnh theo
  tập đo, nên tôi giữ cosine thuần.
- Cosine của model này nằm trong dải hẹp (p5 0.828, p50 0.860, p95 0.898), nên trọng số app 0.2 quyết định gần hết thứ tự:
  đã bỏ số hạng app.
- Thời gian trên CPU: nhúng pool 1236 dòng 30.8–35.8 giây (một lần lúc tải model), 24 ms mỗi lần chọn (trùng từ: 1–2 ms).
  Trên GPU: NOT MEASURED.
- Chưa đo với transcript ASR (không có transcript ASR của ID validation trong máy) và chưa đo trên public.

## Ví dụ (ví dụ đầu: trùng từ → embedding)

- "Bạn ơi, tắt đèn phòng khách giúp tôi nhé." — "đặt phòng khách sạn ở Đà Lạt" (không gọi tool) → "Bật giúp em cái đèn trần
  phòng khách" (đúng chuỗi tool).
- "Nhắn qua Zalo cho Minh là đến rồi thì gọi tôi nhé." — "Đặt báo thức 6 rưỡi sáng mai, rồi gọi cho Minh" → "Nhắn Zalo cho Minh
  giúp chị là mai gửi mình file" (đúng chuỗi tool).
- "Nè, hôm nay mày làm tốt lắm đó, khen một tiếng nha!" — "thêm một buổi review…" → "Thôi mình đi làm đây, chúc bạn một ngày
  thật dễ chịu" (cùng là không gọi tool).
- "Alô, cho em Linh hỏi giá vàng SJC hôm nay bao nhiêu" — "Vàng SJC hôm nay mà mua vào dưới 120 triệu thì nhắc em…" (đúng tool) →
  "lúc nãy giá vàng SJC mua vào bao nhiêu nhỉ, mình quên mất" (ví dụ không gọi tool: tệ hơn).
- "Đặt giùm tui báo thức 7 rưỡi sáng mai nha." — "Đặt báo thức 5 giờ nhé, à không, 5 rưỡi" (đúng tool) → "Xoá giúp tôi cái báo
  thức sáu rưỡi sáng mai" (sai hành động: tệ hơn).

Mười cặp đầy đủ: chạy lại phép đo (script tạm, không nằm trong repo) hoặc xem báo cáo.
