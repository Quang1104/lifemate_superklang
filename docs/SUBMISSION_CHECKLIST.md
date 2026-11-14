# Submission checklist — 2026-11-14

Đối chiếu toàn bộ bullet trong [Docker contract](COMPETITION_RULES.md#docker-contract), mở rộng từng yêu cầu thành một dòng. `PASS` nghĩa là hiện diện trong mã theo audit tĩnh; không khẳng định đã chạy GPU/container. `FIXED` là thay đổi nhỏ trong lượt này và có test trong tests/test_codex_submission.py. Test runtime chưa chạy được được ghi rõ; mọi phép đo GPU/Docker còn thiếu đều là `NOT MEASURED`.

| # | Yêu cầu | Trạng thái | Bằng chứng / việc còn lại |
|---|---|---|---|
| 1 | Entrypoint /code/predict.sh | PASS | [docker/Dockerfile:19](../docker/Dockerfile#L19), [docker/Dockerfile:26](../docker/Dockerfile#L26): copy executable và CMD. |
| 2 | Input /data/private_test/ | PASS | [docker/predict.sh:8](../docker/predict.sh#L8), [src/inference/official.py:106](../src/inference/official.py#L106): data-root /data + split private_test. |
| 3 | Output /result/submission.jsonl; tạo thư mục | PASS | [docker/predict.sh:11](../docker/predict.sh#L11), [src/inference/pipeline.py:152](../src/inference/pipeline.py#L152). Test tạo thư mục đã có nhưng chưa chạy được vì thiếu jsonschema. |
| 4 | zaic_tools từ mounted split, không fallback bản TRAIN | PASS | [src/inference/official.py:111](../src/inference/official.py#L111), [src/inference/official.py:79](../src/inference/official.py#L79): load package theo đường dẫn split. Test cô lập import chưa chạy được vì thiếu jsonschema. |
| 5 | Các field id, tool_calls, state_hash giữ giá trị API | PASS | [src/inference/official.py:43](../src/inference/official.py#L43), [src/inference/official.py:60](../src/inference/official.py#L60): log và hash lấy từ API. Không sửa schema. |
| 6 | Field time là số nguyên milliseconds cho private_test | PASS | [src/inference/pipeline.py:141](../src/inference/pipeline.py#L141), [src/inference/pipeline.py:146](../src/inference/pipeline.py#L146): monotonic ×1000, bắt đầu trước audio/ASR. Runtime fixture test chưa chạy vì thiếu jsonschema. |
| 7 | Có /code/predict_notebook.ipynb | PASS | [docker/Dockerfile:20](../docker/Dockerfile#L20). |
| 8 | Notebook cell cấu hình fixed seed và offline | FIXED | Thêm cell [docker/predict_notebook.ipynb:24](../docker/predict_notebook.ipynb#L24), seed môi trường child; CLI seed các RNG trước model tại [src/inference/__main__.py:121](../src/inference/__main__.py#L121). Test fixture chạy cell thành công. |
| 9 | Notebook cell load model riêng trước cell inference | FAIL/TODO | [docker/predict_notebook.ipynb:72](../docker/predict_notebook.ipynb#L72) chỉ kiểm tra local model dirs; load model và inference cùng trong subprocess [docker/predict_notebook.ipynb:101](../docker/predict_notebook.ipynb#L101). CLI load một lần tại [src/inference/__main__.py:122](../src/inference/__main__.py#L122). Chưa tách load model riêng; cần refactor có kiểm chứng sau khi môi trường chạy đủ dependency. |
| 10 | Notebook cell input và inference rõ ràng | FIXED | Thêm input cell [docker/predict_notebook.ipynb:51](../docker/predict_notebook.ipynb#L51), giữ entrypoint inference chung tại [docker/predict_notebook.ipynb:95](../docker/predict_notebook.ipynb#L95). Test compile/execute bằng fixture qua. |
| 11 | Notebook output /result/jupyter_submission.jsonl có time | PASS | [docker/predict_notebook.ipynb:101](../docker/predict_notebook.ipynb#L101), [docker/predict_notebook.ipynb:117](../docker/predict_notebook.ipynb#L117): script chung, kiểm tra time. |
| 12 | Notebook/script có predictions trùng trên model thật | NOT MEASURED | [docker/predict_notebook.ipynb:121](../docker/predict_notebook.ipynb#L121) so sánh id/calls/hash nếu có submission.jsonl; fixture chấp nhận time khác, từ chối calls khác. Cần chạy cả hai trên GPU/container với cùng cấu hình; time đo riêng từng run. |
| 13 | Image có JupyterLab | PASS | [docker/Dockerfile:16](../docker/Dockerfile#L16). Khả năng cài/build thực tế chưa đo. |
| 14 | /code/start_jupyter.sh, port 9777 | PASS | [docker/Dockerfile:19](../docker/Dockerfile#L19), [docker/start_jupyter.sh:5](../docker/start_jupyter.sh#L5). |
| 15 | README tự đủ trong image | FIXED | Thêm [docker/README.md:3](../docker/README.md#L3), copy thành /code/README.md tại [docker/Dockerfile:14](../docker/Dockerfile#L14). Có mount/output, model, Jupyter, seed, training và release TODO; test kiểm tra qua. |
| 16 | Có training_code/ trong image | PASS | [docker/Dockerfile:21](../docker/Dockerfile#L21). |
| 17 | training_code/ tái lập đúng training hiện tại | FAIL/TODO | [training_code/README.md:3](../training_code/README.md#L3) vẫn nói chưa fine-tune; [training_code/train_qlora.py:1](../training_code/train_qlora.py#L1) phụ thuộc SFT từ scripts/build_sft.py. [docker/Dockerfile:21](../docker/Dockerfile#L21) chỉ copy training_code, chưa copy builder và research/validation_split.json. Chủ repo cần cập nhật README, dependency training, lệnh/resume và artifact cuối; các file nguồn training bị cấm sửa trong lượt này. |
| 18 | Không chứa base training weights thừa | NOT MEASURED | [docker/Dockerfile:17](../docker/Dockerfile#L17) và [docker/Dockerfile:18](../docker/Dockerfile#L18) copy named contexts; chưa kiểm tra artifact/image cuối. Chỉ đóng gói planner merged cuối và ASR. |
| 19 | Inference cấu hình offline và local files | PASS | [docker/Dockerfile:5](../docker/Dockerfile#L5), [src/agent/local_model.py:36](../src/agent/local_model.py#L36), [src/agent/local_model.py:327](../src/agent/local_model.py#L327), [src/agent/local_model.py:518](../src/agent/local_model.py#L518). |
| 20 | Fixed Python/NumPy/PyTorch seeds và deterministic decoding | FIXED | Thêm PYTHONHASHSEED tại [docker/predict.sh:4](../docker/predict.sh#L4), [docker/Dockerfile:6](../docker/Dockerfile#L6), notebook seed cell. RNG seeds đã có tại [src/agent/local_model.py:45](../src/agent/local_model.py#L45), cuDNN tại :50, greedy tại :122. Test cấu hình qua; repeatability model thật chưa đo. |
| 21 | Chạy offline clean GPU container | NOT MEASURED | Không build/chạy Docker hoặc GPU theo phạm vi Windows CPU; cần chạy trên GPU BTC xác nhận. [docker/README.md:15](../docker/README.md#L15). |
| 22 | Export image archive | NOT MEASURED | Chưa có image build/export trong lượt này. [docker/README.md:15](../docker/README.md#L15). |
| 23 | MD5 checksum archive thật | NOT MEASURED | Chưa có archive để tính checksum. [docker/README.md:15](../docker/README.md#L15). |
| 24 | Download link truy cập được | NOT MEASURED | Chưa export/upload, chưa có link để kiểm chứng. [docker/README.md:15](../docker/README.md#L15). |

Tổng 24 dòng: PASS 12; FIXED 4; FAIL/TODO 2; NOT MEASURED 6.

Các blocker ngoài bảng: nguồn BTC còn mâu thuẫn A30/RTX5090 và CUDA ≤13.2/CUDA13.2 bắt buộc; [COMPETITION_RULES.md](COMPETITION_RULES.md#unresolved-official-source-conflicts--release-blockers) giữ nguyên cả hai nguồn. [docker/Dockerfile:2](../docker/Dockerfile#L2) chọn 13.2, chưa xác nhận máy đích.

Hai lệnh bắt buộc đã chạy lúc 18:25 GMT+7, exit 1 trước kiểm thử: pytest thiếu pytest, mock thiếu jsonschema. Kết quả full suite và mock40: **NOT MEASURED**; không dùng số pass từ phiên trước. Không cài dependency vì bị cấm. Kiểm tra bổ sung stdlib: `Ran 12 tests in 0.038s`, `OK (skipped=3)` — 9 qua, 3 runtime test skip do thiếu jsonschema. Lệnh và toàn bộ stdout/stderr nằm trong [research/submission_verification.md](../research/submission_verification.md).

Bảng neo ngày mặc định tắt (`LIFEMATE_DATE_TABLE=1` mới bật); không đổi schema/scoring. Coverage TRAIN và exact command/output: [research/date_table_coverage.md](../research/date_table_coverage.md). Score, độ trễ và độ lặp lại với prompt mới: **NOT MEASURED**.

Kiểm lại bởi root 18:45 GMT+7 (Python có đủ dependency): `python -B -m pytest -q -p no:cacheprovider` → 184 passed, 1 skipped; `python -B -m src.inference mock --samples 40` → checks_passed 40.
