# Bàn giao Claude ⇄ Codex

File này là nguồn sự thật chung. **Đọc trước khi làm, cập nhật trước khi dừng** (kể cả khi sắp hết token). Luật dự án: [AGENTS.md](../AGENTS.md), [quy định BTC](COMPETITION_RULES.md).

## Mục tiêu

- 80–90/100 trên public test. Không có nhãn public, nên ước lượng bằng proxy trên TRAIN holdout (309 ID) và một split **leave-app-out** mô phỏng 34 tool chưa thấy của public.
- Hạn nộp Docker: **2026-11-16 23:59 GMT+7**. Ngày 15/11 phải chốt cấu hình và build Docker.

## Trạng thái đo được

| Ngày | Cấu hình | Kết quả |
|---|---|---|
| 13/11 | smoke40, Qwen3.5-4B + PhoWhisper-medium, T4, x1 | direct 10.0; compiled 0.0 (35/40 planner_timeout); sàn no-op 12.5 |
| 13/11 | đo tốc độ Qwen3.5-4B trên T4 | ~14 tok/s (thiếu kernel linear-attention); prefill 7.9k token ≈ 6 s |
| 13/11 | smoke40, Qwen3-4B-Instruct-2507, T4, TIME_SCALE=3 | **đang chạy** → `outputs/colab/fallback_planner/smoke40_x3/` trên Colab |

Prompt planner hiện tại: median ~15k ký tự (~8k token) mỗi mẫu, phần lớn là schema tool.

## Phân làn (không sửa file của làn khác khi chưa bàn giao)

| Làn | Chủ | Sở hữu | Nhánh |
|---|---|---|---|
| Tích hợp, docs, notebook, Colab | Claude (root) | `docs/`, `configs/`, `lifemate_colab.ipynb`, merge | `colab-smoke` |
| Tốc độ và runtime | Codex | `src/agent/local_model.py`, `src/inference/`, `docker/`, `pyproject.toml` | `codex-speed` |
| Chất lượng planner | Claude subagent | `src/agent/compiled_plan.py`, `plan_*.py`, `planner.py`, `runner.py`, `research/`, `scripts/` | `claude-quality` |

Tests: mỗi làn thêm test vào file `tests/test_<làn>_*.py` riêng để tránh xung đột.

## Bảng việc

| # | Việc | Làn | Trạng thái | Ghi chú |
|---|---|---|---|---|
| 1 | Prefix KV-cache cho phần system+tools dùng lại giữa các lượt gọi trong một mẫu | Codex | todo | direct gọi model nhiều lần với cùng prefix |
| 2 | Render schema tool gọn (chữ ký kiểu `app.tool(arg: type, …)`) và đo số token trước/sau | Codex | todo | dùng chung cho cả hai nhánh; không đổi ngữ nghĩa |
| 3 | Bộ chuyển gold tool_calls TRAIN → kế hoạch DSL compiled, đo tỷ lệ chuyển được | Claude | todo | nền cho few-shot và SFT |
| 4 | Few-shot động: lấy k mẫu TRAIN giống nhất (chỉ 1236 ID train, **không** 309 validation) đưa vào prompt | Claude | todo | cần #3 cho nhánh compiled |
| 5 | Split leave-app-out để đo tổng quát hóa sang app chưa thấy | Claude | todo | `research/` |
| 6 | Phân tích lỗi smoke40_x3, xếp hạng nhóm mất điểm | root | chờ Colab | quyết định ưu tiên tiếp theo |
| 7 | Constrained decoding JSON (chỉ tên tool đã cài, enum) | Codex | sau #1–2 | |
| 8 | ASR: gợi ý tên riêng từ context vào prompt Whisper | Codex | sau #6 | chỉ làm nếu #6 cho thấy ASR mất nhiều điểm |
| 9 | QLoRA trên 1236 ID với nhãn DSL từ #3 | Claude | quyết sau #6 | chỉ khi planner là nguồn mất điểm chính và còn thời gian |
| 10 | Đo đúng 30 s (TIME_SCALE=1), chạy locked269 một lần, build và chạy thử Docker offline | root | 15/11 | |

## Quy tắc bàn giao

1. Làm trên nhánh của làn mình, commit nhỏ, message rõ. Không push lên `main`. Không force-push.
2. Trước khi dừng: cập nhật dòng của việc mình trong bảng (trạng thái, commit, lệnh test đã chạy, bước tiếp theo) và mục **Nhật ký** bên dưới.
3. Kết quả chưa đo ghi `NOT MEASURED`. Không bịa số.
4. Dữ liệu: dùng đường dẫn tuyệt đối tới kit, ví dụ `C:/Users/ADMIN/OneDrive - VNU-HCMUS/Documents/XuanQuang/datalife/train+public_test`. Không đọc thư mục `*_environment` hay `state.json`.
5. Test trước khi bàn giao: `python -B -m pytest -q -p no:cacheprovider` và `python -B -m src.inference mock --samples 40`.

## Nhật ký

- 13/11 Claude: tạo file này; smoke40_x3 đang chạy trên Colab (tài khoản Google thứ hai, T4).
