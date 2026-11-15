# Submission checklist — 2026-11-14

Đối chiếu toàn bộ bullet trong [Docker contract](COMPETITION_RULES.md#docker-contract), mở rộng từng yêu cầu thành một dòng. `PASS` nghĩa là hiện diện trong mã theo audit tĩnh; `FIXED` là yêu cầu đã được sửa trong audit hoặc phiên này, không khẳng định đã chạy GPU/container. Các test mới nằm trong `tests/test_codex_notebook_api.py`, `tests/test_codex_training_repro.py` và `tests/test_codex_release.py`; runtime thiếu dependency và mọi phép đo Docker/GPU chưa có đều ghi `NOT MEASURED`.

| # | Yêu cầu | Trạng thái | Bằng chứng / việc còn lại |
|---|---|---|---|
| 1 | Entrypoint /code/predict.sh | PASS | [docker/Dockerfile](../docker/Dockerfile): copy executable và CMD `/code/predict.sh`. |
| 2 | Input /data/private_test/ | PASS | [docker/predict.sh:8](../docker/predict.sh#L8), [src/inference/official.py:106](../src/inference/official.py#L106): data-root /data + split private_test. |
| 3 | Output /result/submission.jsonl; tạo thư mục | PASS | [docker/predict.sh:11](../docker/predict.sh#L11), [src/inference/pipeline.py:152](../src/inference/pipeline.py#L152). Test tạo thư mục đã có nhưng chưa chạy được vì thiếu jsonschema. |
| 4 | zaic_tools từ mounted split, không fallback bản TRAIN | PASS | [src/inference/official.py:111](../src/inference/official.py#L111), [src/inference/official.py:79](../src/inference/official.py#L79): load package theo đường dẫn split. Test cô lập import chưa chạy được vì thiếu jsonschema. |
| 5 | Các field id, tool_calls, state_hash giữ giá trị API | PASS | [src/inference/official.py:43](../src/inference/official.py#L43), [src/inference/official.py:60](../src/inference/official.py#L60): log và hash lấy từ API. Không sửa schema. |
| 6 | Field time là số nguyên milliseconds cho private_test | PASS | [src/inference/pipeline.py:141](../src/inference/pipeline.py#L141), [src/inference/pipeline.py:146](../src/inference/pipeline.py#L146): monotonic ×1000, bắt đầu trước audio/ASR. Runtime fixture test chưa chạy vì thiếu jsonschema. |
| 7 | Có /code/predict_notebook.ipynb | PASS | [docker/Dockerfile:20](../docker/Dockerfile#L20). |
| 8 | Notebook cell cấu hình fixed seed và offline | FIXED | [Notebook](../docker/predict_notebook.ipynb) gọi `seed_runtime` ở cell đầu; [API](../src/inference/api.py) tái dùng `initialize_runtime`, seed 20261112. Fixture cell qua trong phiên này. |
| 9 | Notebook cell load model riêng trước cell inference | FIXED | [API `load_models`](../src/inference/api.py) nạp planner + ASR một lần, trả handle và `load_ms`; [notebook](../docker/predict_notebook.ipynb) có cell load riêng trước input/inference, in milliseconds. CLI dùng cùng loader và vòng lặp; compile/thứ tự/thực thi cell bằng fake API đã qua. Test parity pipeline skip do thiếu jsonschema. |
| 10 | Notebook cell input và inference rõ ràng | FIXED | [Notebook](../docker/predict_notebook.ipynb): seed → model load → mounted input → `run_private_test` → compare. Test fixture kiểm tra thứ tự và handle được dùng lại đã qua. |
| 11 | Notebook output /result/jupyter_submission.jsonl có time | PASS | [`run_private_test`](../src/inference/api.py) dùng vòng lặp CLI và writer hiện có; [notebook](../docker/predict_notebook.ipynb) kiểm tra time là integer 0..30000. Không đổi output schema. |
| 12 | Notebook/script có predictions trùng trên model thật | NOT MEASURED | [Test mới](../tests/test_codex_notebook_api.py) xác nhận notebook so sánh calls/hash và cho phép time khác qua fake API; test API/CLI qua pipeline skip vì thiếu jsonschema. Cần chạy cả hai trên model thật cùng cấu hình GPU/container. |
| 13 | Image có JupyterLab | PASS | [docker/Dockerfile:16](../docker/Dockerfile#L16). Khả năng cài/build thực tế chưa đo. |
| 14 | /code/start_jupyter.sh, port 9777 | PASS | [docker/Dockerfile:19](../docker/Dockerfile#L19), [docker/start_jupyter.sh:5](../docker/start_jupyter.sh#L5). |
| 15 | README tự đủ trong image | FIXED | [docker/README.md](../docker/README.md) được copy thành `/code/README.md`: input/output, model, notebook API, training và cách chạy release script trên host. Dependency/image thực tế chưa build. |
| 16 | Có training_code/ trong image | PASS | [docker/Dockerfile:21](../docker/Dockerfile#L21). |
| 17 | training_code/ tái lập đúng training hiện tại | FIXED | [README tiếng Việt](../training_code/README.md): TRAIN 1236 ID loại 309 validation, builder → QLoRA một epoch → merge, seed/hyper-parameter, T4 fp16, resume không khôi phục optimizer, torchao, NOT MEASURED. [Dockerfile](../docker/Dockerfile) thêm builder, manifest và khai báo PEFT/bitsandbytes; 4 [test tĩnh](../tests/test_codex_training_repro.py) qua, gồm đối chiếu mọi file nguồn README với COPY. Chưa chạy training/build image. |
| 18 | Không chứa base training weights thừa | NOT MEASURED | [docker/Dockerfile:17](../docker/Dockerfile#L17) và [docker/Dockerfile:18](../docker/Dockerfile#L18) copy named contexts; chưa kiểm tra artifact/image cuối. Chỉ đóng gói planner merged cuối và ASR. |
| 19 | Inference cấu hình offline và local files | PASS | [docker/Dockerfile:5](../docker/Dockerfile#L5), [src/agent/local_model.py:36](../src/agent/local_model.py#L36), [src/agent/local_model.py:327](../src/agent/local_model.py#L327), [src/agent/local_model.py:518](../src/agent/local_model.py#L518). |
| 20 | Fixed Python/NumPy/PyTorch seeds và deterministic decoding | FIXED | PYTHONHASHSEED=20261112 trong [predict.sh](../docker/predict.sh), [Dockerfile](../docker/Dockerfile), notebook; [API](../src/inference/api.py) gọi seed trước load, tái dùng `initialize_runtime` và greedy decoding hiện có. Repeatability model thật chưa đo. |
| 21 | Chạy offline clean GPU container | NOT MEASURED | Không build/chạy Docker hoặc GPU theo phạm vi Windows CPU; cần chạy trên GPU BTC xác nhận. [docker/README.md:15](../docker/README.md#L15). |
| 22 | Export image archive | NOT MEASURED | Chưa có image build/export trong lượt này. [docker/README.md:15](../docker/README.md#L15). |
| 23 | MD5 checksum archive thật | NOT MEASURED | Chưa có archive để tính checksum. [docker/README.md:15](../docker/README.md#L15). |
| 24 | Download link truy cập được | NOT MEASURED | Chưa export/upload, chưa có link để kiểm chứng. [docker/README.md:15](../docker/README.md#L15). |
Tổng 24 dòng: PASS 12; FIXED 6; FAIL/TODO 0; NOT MEASURED 6.

Release ngày 2026-11-15: [docker/release.ps1](../docker/release.ps1) chỉ nhận thư mục adapter; Docker tải planner gốc + ASR đúng revision và merge bằng `training_code/stream_merge.py` (bằng từng bit với `merge_lora.py`, test `tests/test_stream_merge.py`) trong lúc build, xuất `.tar.gz` + MD5 ra ngoài OneDrive; [docker/cleanup.ps1](../docker/cleanup.ps1) trả lại dung lượng ([ghi chú](../docker/LOCAL_DISK_NOTES.md)). Chỉ test tĩnh, **chưa chạy**; các dòng 18, 21–24 vẫn NOT MEASURED.

Kiểm chứng phiên này với đúng interpreter: unittest thu thập 12 test, 11 qua, 1 skip (jsonschema), exit 0, nguyên văn `Ran 12 tests in 0.592s` / `OK (skipped=1)`. Hai lệnh bắt buộc pytest và mock40 đều exit 1 vì thiếu pytest/jsonschema; full suite, mock40 và parity API/CLI qua pipeline là **NOT MEASURED**. Lệnh, stdout/stderr nguyên văn và Get-Date sau mỗi task ở [docker/VERIFICATION_2026-11-14.md](../docker/VERIFICATION_2026-11-14.md). Không dùng số pass/score từ phiên cũ cho kết quả hiện tại.

Hai fixture notebook cũ trong [tests/test_codex_submission.py](../tests/test_codex_submission.py) vẫn bắt buộc subprocess và input trước load. Chúng nằm ngoài phạm vi sửa (chỉ được thêm test mới), nên giữ nguyên và bàn giao để cập nhật khi được phép; lỗi của chúng chưa được chạy trong phiên thiếu pytest này.

Các blocker ngoài bảng: nguồn BTC còn mâu thuẫn A30/RTX5090 và CUDA ≤13.2/CUDA13.2 bắt buộc; [COMPETITION_RULES.md](COMPETITION_RULES.md#unresolved-official-source-conflicts--release-blockers) giữ nguyên cả hai nguồn. [Dockerfile](../docker/Dockerfile) chọn 13.2, chưa xác nhận máy đích. Artifact train cuối, score, hash, notebook/script trên model thật, training trong image và clean GPU run đều **NOT MEASURED**.

Không thay đổi output schema/scoring hoặc official zaic_tools. Bảng neo ngày vẫn mặc định tắt (LIFEMATE_DATE_TABLE=1 mới bật); không đo lại coverage hay chất lượng model trong phiên này.

## Lệnh release theo từng ứng viên (15/11)

Cờ của model là một đầu vào của release: `release.ps1 -Flags "TÊN=1,..."` → kiểm bằng allowlist `docker/release_flags.py`
(tên lạ, giá trị khác `1` hoặc lặp tên thì dừng trước khi build) → mỗi cờ thành một `--build-arg` → dòng `ENV` trong image →
ghi vào `<archive>.manifest.json` cùng SHA256 của adapter, SHA256 của ASR và MD5 của archive. `predict.sh` và
`predict_notebook.ipynb` không đặt hay xoá cờ nào (test `tests/test_orca_release_flags.py`). Mọi ứng viên chạy với
`LIFEMATE_NATIVE_FEWSHOT_K=2` (mặc định của Dockerfile). Masking của v5 chỉ dùng lúc train nên không có cờ suy luận.

| Ứng viên | Lệnh (ASR đã fine-tune đặt trong `<AdapterDir>\asr`) |
|---|---|
| v3 + ASR fine-tune | `powershell.exe -NoProfile -File .\docker\release.ps1 -AdapterDir <adapter v3>` |
| v4 | `powershell.exe -NoProfile -File .\docker\release.ps1 -AdapterDir <adapter v4> -Flags "LIFEMATE_CONTEXT_V4=1,LIFEMATE_SMALL_FIXES=1"` |
| v5 | `powershell.exe -NoProfile -File .\docker\release.ps1 -AdapterDir <adapter v5> -Flags "LIFEMATE_CONTEXT_V4=1,LIFEMATE_SMALL_FIXES=1,LIFEMATE_DATE_TABLE=1"` |

Sau khi build, đối chiếu cờ trong image với manifest: `docker image inspect <tag> --format '{{json .Config.Env}}'`.
Chưa build image nào với `-Flags`: kích thước, thời gian và kết quả chạy trong container là NOT MEASURED. Trên máy này
PowerShell chặn chạy file script (execution policy), nên `release.ps1` mới được phân tích cú pháp (0 lỗi) và các dòng xử lý
cờ được chạy trực tiếp trong PowerShell: đầu vào hợp lệ cho đúng các `--build-arg`, rỗng cho 0 cờ, tên lạ trả mã lỗi 1.
