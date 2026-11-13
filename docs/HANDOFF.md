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
| 13/11 | smoke40, Qwen3-4B-Instruct-2507, T4, TIME_SCALE=3 | direct 5.8 (state_mismatch 52.5, missing 22.5, unexpected 7.5; 0 timeout); compiled 7.5 (**invalid_plan 67.5**, 27/40); compiled + transcript gốc 2.5 (invalid_plan 92.5) |
| 13/11 | chấm lại smoke40_x3 bằng `compare_train` đã sửa (record đã xuất được chấm như BTC) | direct 5.83; compiled 12.5; compiled + transcript gốc 12.5 = đúng mức sàn no-op (chỉ ăn điểm 5 mẫu gold rỗng). Direct dưới sàn vì đoán sai → **bỏ qua khi không chắc** tốt hơn đoán |
| 13/11 | chẩn đoán 8 mẫu compiled (transcript gốc) | lỗi chính `InvalidPlan: Expected a bounded typed argument source`: model ghi giá trị thô (`"time":"2026-11-23T06:00:00+07:00"`, `"query":"Minh"`) và bịa ID (`"contact_id":"contact_123"`) thay vì SOURCE DSL |

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
| 1 | Prefix KV-cache cho phần system+tools dùng lại giữa các lượt gọi trong một mẫu | Codex | done (Claude áp dụng patch) | Patch tạm: LocalPlanner(reuse_prefix_cache=True), LCP/crop, full-context cap, deadline/lock, shape/device/dtype guards; cache recurrent/sliding fallback. Fake tests passed; real CPU Qwen NOT MEASURED (thiếu torch/transformers). Chưa áp dụng/commit: worktree từ chối ghi. |
| 2 | Render schema tool gọn (chữ ký kiểu `app.tool(arg: type, …)`) và đo số token trước/sau | Codex | done (Claude áp dụng patch) | API src.agent.schema_text.render_tools(tools); DIRECT dùng tools/context/history/user_request trước observations. Smoke40 payload: 660,976 → 524,779 ký tự (−20.605%); median 16,489.5 → 13,259.5. Qwen tokens NOT MEASURED (không tokenizer offline). CLI đo kèm patch; Claude có thể dùng renderer cho compiled sau khi áp dụng. |
| 3 | Bộ chuyển gold tool_calls TRAIN → kế hoạch DSL compiled, đo tỷ lệ chuyển được | Claude | done `410dcc6` | `python -B -m scripts.gold_to_plan --data-root <kit>` → `outputs/gold_plans.jsonl` (+ `.report.json`), 38 s. 1236 ID (ngoài 309): chuyển trọn 530 (42.88%), xác minh khớp calls+hash 530/530 chuyển được (42.88% tổng; 441 mẫu có call). Lý do theo số đối số: text không nguyên văn 405, x-ref có trong kết quả nhưng không có where đơn trị bám câu nói 273, x-ref chỉ nằm trong list của history 226, datetime không span/field 102, text khác hoa-thường 98, int 61. Hạn chế: kế hoạch chỉ chứa nhánh gold đã chạy (không sinh `when`). Tiếp: #4 dùng 530 plan làm few-shot; cân nhắc mở DSL cho `history`+where (226) và where mờ/lồng (273) |
| 4 | Few-shot động: lấy k mẫu TRAIN giống nhất (chỉ 1236 ID train, **không** 309 validation) đưa vào prompt | Claude | done `cad161c` | `src/agent/fewshot.py` + `src/agent/data/fewshot_plans.jsonl` (530 plan verified, 310,211 byte, ship trong Docker vì `COPY src`), dựng bằng `python -B -m scripts.build_fewshot --data-root <kit>`. Chọn tất định: Jaccard từ transcript + 0.2·Jaccard app cài; chỉ ví dụ mà mọi tool đều đang cài; loại chính ID mẫu; ngân sách 3,000 ký tự; `examples` đứng trước `request`. `LIFEMATE_FEWSHOT_K` (mặc định 2, 0 = tắt). **Đo leave-app-out phải đặt `LIFEMATE_FEWSHOT_SPLIT=research/leave_app_out_split.json`** (52 ID holdout nằm trong pool; split còn 478). Prompt smoke40 (system+payload, ký tự): k=0 tổng 858,601, median 21,756, max 37,080; k=2 +4.04% (thêm median 681, max 2,623); k=3 +5.91% (median 1,064, max 2,771) — `python -B -m scripts.raw_plan_check --data-root <kit> --prompt-size`. Token Qwen và điểm model: NOT MEASURED. Tiếp: chạy smoke40 Colab k=0 vs k=2 |
| 4b | Chấp nhận giá trị thô có căn cứ (nguyên văn, span chuyển được, context, enum, kết quả đã thấy); ID bịa → repair với kết quả thật | Claude | done `cad161c` | `plan_protocol.Grounding` + `compiled_plan._Resolver`: literal → text span (đúng hoa-thường; khác hoa-thường giữ chữ của model), span ≤9 từ chuyển bằng convert_span (datetime chấp nhận thiếu giây/offset), context, enum/bool, prior_results thành công, history_text, field `x-match: fuzzy` (text tự do); khi repair: giá trị có trong visible_results. x-ref thô không quan sát → chờ read sinh namespace (hoặc reader tự động duy nhất), dùng nguyên ID nếu read trả về, else bản ghi duy nhất, else đúng 1 bản ghi có field khớp query/anchor, else Unresolved → repair 1 lần có visible_results. `{"value":X}` ngoài enum ở args cũng được grounding. Không grounding được → vẫn InvalidPlan trước mọi dispatch. Shape: list steps trần, code fence/prose, khóa bình luận, thiếu id/args, `arguments`, read thiếu anchor; anchor gần đúng → cửa sổ chung ≥50% từ. Oracle (`python -B -m scripts.raw_plan_check --data-root <kit> --sets smoke40 leave_app_out_holdout verified530 train1236`, 26 s, ToolEnv thật): literal gold thô exact calls+hash smoke40 31/40 (77.5%), holdout 116/171 (67.84%), **verified530 491/530 (92.64%)**, train1236 891/1236 (72.09%; DSL #3 chỉ 42.88%); ID bịa (+repair mô phỏng chép ID thật) smoke40 12/22 (5 không cần repair), holdout 26/57, verified530 99/149, train1236 247/501 với 2 ca gửi sai ID. Còn lỗi: giá trị suy từ kết quả chưa thấy (giờ end, số tiền hóa đơn), enum dịch (nhanh→quick, đô→USD). Điểm model: NOT MEASURED. Tiếp: Codex #7 grammar nên cho phép literal vô hướng trong args |
| 5 | Split leave-app-out để đo tổng quát hóa sang app chưa thấy | Claude | done `410dcc6` | `python -B -m research.leave_app_out_split --input <kit>/train/train.jsonl --catalog <kit>/train/train_tools_catalog.json` → `research/leave_app_out_split.json`: giữ ra 6 app (calculator, exchange_rate, household_chores, library, news, package_tracking; 14 tool), 171 ID holdout, 1065 ID pool; mẫu có call (gold, history hoặc ask_missing_info) tới app giữ ra đều vào holdout. Tiếp: few-shot #4 chỉ lấy từ `train_ids` của file này khi đo trên holdout |
| 6 | Phân tích lỗi smoke40_x3, xếp hạng nhóm mất điểm | root | done | xem Trạng thái đo được; ưu tiên #4b, #4, #7 |
| 7 | Constrained decoding theo JSON schema của DSL (tên tool đã cài, SOURCE hợp lệ, enum) | Codex → Claude | done (chưa đo trên GPU) | Codex viết `src/agent/plan_grammar.py` + test rồi hết token (reset 22:00); Claude nối vào `LocalPlanner.generate(json_schema=)` qua xgrammar (tùy chọn, fallback khi thiếu/compile lỗi), `CompiledPlanner(constrain=True)` trong CLI; report có `grammar` (constrained/unavailable/compile_failed). xgrammar 0.1.34 xác minh trên Colab (API from_huggingface/compile_json_schema/contrib.hf.LogitsProcessor có thật); compile schema đủ mọi tool mất 7.19 s, cache theo bộ app cài → rủi ro độ trễ ở TIME_SCALE=1 |
| 8 | ASR: gợi ý tên riêng từ context vào prompt Whisper | Codex | sau #6 | chỉ làm nếu #6 cho thấy ASR mất nhiều điểm |
| 9 | QLoRA trên 1236 ID với nhãn DSL từ #3 | Claude | quyết sau #6 | chỉ khi planner là nguồn mất điểm chính và còn thời gian |
| 10 | Đo đúng 30 s (TIME_SCALE=1), chạy locked269 một lần, build và chạy thử Docker offline | root | 15/11 | |

## Khi một bên hết token: bên kia tiếp quản thế nào

1. `git status` và `git log --oneline -5` trên `colab-smoke` (repo chính). Code chưa commit là việc dở của bên trước: đọc diff, chạy test, rồi commit hoặc hoàn tác.
2. Kiểm tra hai worktree làn: `git -C ../lifemate_quality log --oneline colab-smoke..claude-quality` và `git -C ../lifemate_codex log --oneline colab-smoke..codex-speed`. Commit nào chưa có trên `colab-smoke` thì xem, chạy test, rồi `git cherry-pick` vào `colab-smoke`. Patch Codex để ở `C:/Users/ADMIN/AppData/Local/Temp/lifemate-*-staging/` cũng là việc dở.
3. Chạy `python -B -m pytest -q -p no:cacheprovider` và `python -B -m src.inference mock --samples 40` trước khi push `colab-smoke`.
4. Làm tiếp việc có trạng thái `đang làm` hoặc `todo` ưu tiên cao nhất trong bảng việc.
5. Đo trên GPU: chủ repo mở notebook Colab bằng tài khoản Google thứ hai (`?authuser=1` hoặc tài khoản đang đăng nhập có GPU), chạy cell 1 → 2 → 3 → smoke → tách lỗi ASR. Kết quả in ngay trong cell; chép số vào bảng **Trạng thái đo được**.

## Quy tắc bàn giao

1. Làm trên nhánh của làn mình, commit nhỏ, message rõ. Không push lên `main`. Không force-push. **Không thêm dòng `Co-Authored-By` hay ghi công AI** vào commit hoặc PR; tác giả duy nhất là chủ repo.
2. Trước khi dừng: cập nhật dòng của việc mình trong bảng (trạng thái, commit, lệnh test đã chạy, bước tiếp theo) và mục **Nhật ký** bên dưới.
3. Kết quả chưa đo ghi `NOT MEASURED`. Không bịa số.
4. Dữ liệu: dùng đường dẫn tuyệt đối tới kit, ví dụ `C:/Users/ADMIN/OneDrive - VNU-HCMUS/Documents/XuanQuang/datalife/train+public_test`. Không đọc thư mục `*_environment` hay `state.json`.
5. Test trước khi bàn giao: `python -B -m pytest -q -p no:cacheprovider` và `python -B -m src.inference mock --samples 40`.

## Nhật ký

- 13/11 Claude: tạo file này; smoke40_x3 đang chạy trên Colab (tài khoản Google thứ hai, T4).

- 13/11 Codex speed (`codex-speed`): chuẩn bị prefix KV-cache bật mặc định và renderer schema lossless trong `C:/Users/ADMIN/AppData/Local/Temp/lifemate-speed-staging/`, chưa áp dụng code vào checkout vì src/tests từ chối ghi (`EPERM`/`PermissionError`); chỉ cập nhật HANDOFF.md; `exec_command` cũng thiếu `codex-windows-sandbox-setup.exe`. Smoke40 TRAIN oracle prompt analysis (không ToolEnv/label): payload sum 660,976 → 524,779 (−20.605%), median 16,489.5 → 13,259.5, min 7,085 → 5,727, max 26,229 → 21,666; system+payload sum 694,416 → 558,219 (−19.613%), median 17,325.5 → 14,095.5; Qwen tokens, real-model equivalence, T4 latency và score NOT MEASURED. Lệnh đã chạy: `python -B -m unittest discover -s tests -p test_speed_*.py -v` trong staging: 22 passed, 2 skipped; `python -B C:/Users/ADMIN/AppData/Local/Temp/lifemate-speed-staging/prompt_size_analysis.py` đo ký tự bằng catalog/sample transforms gốc, bỏ schema validation riêng cho phân tích độ dài vì thiếu jsonschema. Hai lệnh bắt buộc đã thử tại checkout: `python -B -m pytest -q -p no:cacheprovider` thất bại do thiếu pytest; `python -B -m src.inference mock --samples 40` thất bại do thiếu jsonschema; cài dependency bị mạng chặn. Commit: NOT CREATED (sandbox chặn ghi). Bước tiếp: sửa quyền ghi/dependency, áp dụng patch, chạy hai lệnh bắt buộc và CLI `python -B -m src.inference.prompt_size --data-root "C:/Users/ADMIN/OneDrive - VNU-HCMUS/Documents/XuanQuang/datalife/train+public_test" --manifest research/smoke_split.json`, chạy real CPU Qwen test nếu có torch/transformers, commit với trailer yêu cầu; Claude dùng render_tools cho compiled, rồi đo TIME_SCALE=1 trên T4.
- 13/11 Claude root: áp dụng patch Codex vào `codex-speed`, cho test Qwen3 thật tự skip khi transformers local < 5 (chạy trên Colab); `python -B -m pytest -q -p no:cacheprovider` 120 passed, 1 skipped; mock 40/40. Đo token Qwen thật và độ trễ T4: NOT MEASURED.
- 13/11 Claude quality: #3 và #5 xong (`410dcc6`, nhánh `claude-quality`). Gold→plan: 42.88% chuyển trọn, 100% số chuyển được xác minh khớp calls+hash. Test: `python -B -m pytest -q -p no:cacheprovider` 104 passed; `python -B -m src.inference mock --samples 40` checks_passed 40.
- 13/11 Claude root: Codex hết hạn mức giữa #7; Claude tiếp quản từ staging, nối grammar vào LocalPlanner/CompiledPlanner. `python -B -m pytest -q -p no:cacheprovider` 139 passed, 1 skipped; mock 40/40. Tác dụng trên điểm: NOT MEASURED (cần Colab).
- 13/11 Claude quality: #4b + #4 xong trên `claude-quality`. Literal có căn cứ được chuyển thành SOURCE, ID bịa chờ read rồi chọn bản ghi/repair; few-shot k=2 từ 530 plan verified. Oracle literal thô: verified530 92.64%, smoke40 77.5%, holdout 67.84%; prompt smoke40 +4.04% ký tự với k=2. Test: `python -B -m pytest -q -p no:cacheprovider` 141 passed, 1 skipped; `python -B -m src.inference mock --samples 40` checks_passed 40; `python -B -m scripts.gold_to_plan --data-root <kit> --output outputs/gold_plans_recheck.jsonl` vẫn 530/530 verified. Điểm model thật: NOT MEASURED (Colab).
- 13/11 Claude quality: #4b + #4 xong (`cad161c`, `claude-quality`). Literal có căn cứ được chuyển thành SOURCE, ID bịa chờ read rồi chọn bản ghi/repair; few-shot k=2 từ 530 plan verified. Oracle literal thô: verified530 92.64%, smoke40 77.5%, holdout 67.84%; prompt smoke40 +4.04% ký tự với k=2. Test: `python -B -m pytest -q -p no:cacheprovider` 141 passed, 1 skipped; `python -B -m src.inference mock --samples 40` checks_passed 40; `python -B -m scripts.gold_to_plan --data-root <kit> --output outputs/gold_plans_recheck.jsonl` vẫn 530/530 verified. Điểm model thật: NOT MEASURED (Colab).
