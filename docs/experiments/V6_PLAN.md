# v6: dữ liệu cho tool chưa thấy (NEXT_PLAN mục 2)

Viết 15/11 23:20, chỉ CPU. Lý do: [PUBLIC_GAP.md](PUBLIC_GAP.md) mục 2 và 3. Điểm v6: NOT MEASURED.

## Thay đổi (tắt mặc định; tắt thì file SFT và prompt trùng byte với trước)

- `scripts/build_sft.py --rename-fraction F`: trên một phần mẫu chọn bằng hash ổn định, đổi mọi app/tool không phải
  `sys.` sang tên khác **có nghĩa** (vd. `calendar.list_events` → `agenda.show_entries`), nhất quán ở định nghĩa tool,
  lời gọi, kết quả, lịch sử, ví dụ trong prompt, "(app)" trong danh sách app, `x-ref` và `ask_missing_info.tool_name`.
  Chỉ thay tên trọn vẹn (`music.playlists` không đổi). Bảng tên viết tay từ tên TRAIN (45 app × 2 lựa chọn, 53 từ hành động
  × 2); không dùng tên nào của catalog public (test kiểm). Masking lấy dải hash [0, mask), đổi tên lấy [mask, mask + F):
  một mẫu không bao giờ bị cả hai.
- `LIFEMATE_FEWSHOT_GATE=1` (`src/agent/fewshot.py`): không đưa ví dụ khi ví dụ tốt nhất có điểm < τ = 0.23. Cùng đường
  code lúc build và lúc chạy (`build_messages` → `native_examples().select`). Đã thêm vào `docker/release_flags.py`.
- `scripts/colab_v6.sh`: `BASE=v4` mặc định (root 23:20: smoke v5 50.19 < v4 56.29); `BASE=v5` thêm bảng ngày và mask 30%.
  Build, smoke và public đều có `LIFEMATE_FEWSHOT_GATE=1`; public chạy khi smoke v6 ≥ smoke tốt nhất trước đó.

## Chọn τ (không dùng nhãn)

Điểm = Jaccard từ của câu nói + 0.2 × Jaccard app đã cài, của ví dụ tốt nhất mà mọi tool đều đã cài.

| Tập | n | p25 | p50 | dưới τ = 0.23 |
|---|---:|---:|---:|---:|
| TRAIN, bỏ chính mẫu, chữ gold | 1236 | 0.279 | 0.377 | 108 (8.7%) |
| TRAIN, bỏ chính mẫu, ASR cũ (WER 28%) | 1236 | 0.243 | 0.323 | 242 (19.6%) |
| Validation 309, chữ gold | 309 | 0.275 | 0.390 | 27 (8.7%) |
| Leave-app-out 171 (pool = `train_ids` của split), chữ gold | 171 | 0.227 | 0.266 | 44 (25.7%) |
| Leave-app-out 171, ASR cũ | 171 | 0.202 | 0.241 | 71 (41.5%) |

Không có transcript ASR public trong máy, nên tỷ lệ public dưới τ: NOT MEASURED. Chọn 0.23 vì với hỗn hợp "khoảng 45% mẫu
public gọi tool lạ" giữa hai dòng TRAIN và leave-app-out, phần dưới τ nằm khoảng 15–30% tuỳ chất lượng ASR (ước lượng).

## Số đo CPU (build đủ 1236 ID TRAIN, chữ gold)

- v4 (CONTEXT_V4 + SMALL_FIXES, K=2): 1236 dòng, 1236 prompt có ví dụ, ký tự p50 20,667, max 37,872 (trùng byte bản v4 cũ).
- v6 trên BASE=v4 (+ đổi tên 0.3 + cổng): 398 mẫu đổi tên, 1128/1236 prompt có ví dụ (108 bị cổng), p50 20,636, max 38,004.
- Kiểm nhất quán (bản mask 0.3 + đổi tên 0.3): 377 mẫu đổi tên; 0 tên không hợp lệ, 0 trùng tên TRAIN hay public, 0 lời gọi
  ngoài danh sách tool, 0 `ask_missing_info.tool_name` lệch, 0 tên gốc còn sót; hoàn nguyên được 377/377 về đúng bản v5;
  859 mẫu còn lại trùng byte bản v5.

## Rủi ro

- Ví dụ truy hồi có thể nhắc tới tool không cài trong mẫu (vd. `ask_missing_info{"tool_name":"phone.call"}`); tên đó không
  bị đổi vì không thuộc mẫu. Đã có từ trước v6.
- Mask 30% (v5) làm smoke giảm; đổi tên cũng là nhiễu với smoke40. Chỉ public đo được tác dụng trên tool lạ.
- Số token sau khi đổi tên: NOT MEASURED (tên mới dài hơn một chút; max ký tự 37,872 → 38,004).
