# Kiểm chứng phiên Windows CPU — 2026-11-14

Nhánh được đọc từ .git/HEAD: `colab-smoke`. Interpreter chỉ định thiếu cả pytest và jsonschema; không cài/gỡ package. Không chạy release.ps1, Docker build/save, model thật hay dữ liệu inference chính thức.

## Lệnh

```powershell
& "C:\Program Files\Python312\python.exe" -B -m unittest -v tests.test_notebook_api tests.test_training_repro tests.test_release
```

Exit code: `0`.

stdout (rỗng):

```text
```

stderr:

```text
test_code_compiles_and_api_calls_have_distinct_cells_in_contract_order (tests.test_notebook_api.NotebookCellTests.test_code_compiles_and_api_calls_have_distinct_cells_in_contract_order) ... ok
test_notebook_accepts_absent_script_output (tests.test_notebook_api.NotebookCellTests.test_notebook_accepts_absent_script_output) ... ok
test_notebook_executes_with_matching_predictions_and_different_times (tests.test_notebook_api.NotebookCellTests.test_notebook_executes_with_matching_predictions_and_different_times) ... ok
test_notebook_rejects_different_predictions (tests.test_notebook_api.NotebookCellTests.test_notebook_rejects_different_predictions) ... ok
test_api_and_cli_write_identical_predictions_and_load_once_per_run (tests.test_notebook_api.SharedRuntimeTests.test_api_and_cli_write_identical_predictions_and_load_once_per_run) ... skipped 'jsonschema is unavailable in this interpreter'
test_builder_repo_imports_are_copied_without_other_scripts (tests.test_training_repro.TrainingReproducibilityTests.test_builder_repo_imports_are_copied_without_other_scripts) ... ok
test_documented_command_uses_current_hyperparameters (tests.test_training_repro.TrainingReproducibilityTests.test_documented_command_uses_current_hyperparameters) ... ok
test_documented_train_partition_matches_manifest (tests.test_training_repro.TrainingReproducibilityTests.test_documented_train_partition_matches_manifest) ... ok
test_every_documented_code_file_is_in_copy_image_file_list (tests.test_training_repro.TrainingReproducibilityTests.test_every_documented_code_file_is_in_copy_image_file_list) ... ok
test_build_context_names_match_dockerfile_exactly (tests.test_release.ReleaseScriptTests.test_build_context_names_match_dockerfile_exactly) ... ok
test_powershell_51_syntax_and_build_context_ast (tests.test_release.ReleaseScriptTests.test_powershell_51_syntax_and_build_context_ast) ... ok
test_validates_both_model_directories_before_build (tests.test_release.ReleaseScriptTests.test_validates_both_model_directories_before_build) ... ok

----------------------------------------------------------------------
Ran 12 tests in 0.592s

OK (skipped=1)
```

## Lệnh

```powershell
& "C:\Program Files\Python312\python.exe" -B -m pytest -q -p no:cacheprovider
```

Exit code: `1`.

stdout (rỗng):

```text
```

stderr:

```text
C:\Program Files\Python312\python.exe: No module named pytest
```

## Lệnh

```powershell
& "C:\Program Files\Python312\python.exe" -B -m src.inference mock --samples 40
```

Exit code: `1`.

stdout (rỗng):

```text
```

stderr:

```text
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\ADMIN\OneDrive - VNU-HCMUS\Documents\XuanQuang\lifemate_superklang\src\inference\__main__.py", line 12, in <module>
    from ..agent.runner import Limits
  File "C:\Users\ADMIN\OneDrive - VNU-HCMUS\Documents\XuanQuang\lifemate_superklang\src\agent\runner.py", line 14, in <module>
    from ..tooling.registry import ToolRegistry, ToolRejected
  File "C:\Users\ADMIN\OneDrive - VNU-HCMUS\Documents\XuanQuang\lifemate_superklang\src\tooling\registry.py", line 7, in <module>
    from jsonschema import Draft202012Validator, validators
ModuleNotFoundError: No module named 'jsonschema'
```

## Get-Date sau mỗi task

Lệnh: `Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'`; tất cả exit 0, stderr rỗng.

- Task 1: `2026-11-14 21:58:30 +07:00`.
- Task 2: `2026-11-14 21:57:59 +07:00`.
- Task 3: `2026-11-14 21:57:24 +07:00`.
- Sau kiểm chứng tích hợp: `2026-11-14 21:59:13 +07:00`.

12 test được thu thập: 11 qua, 1 skip rõ ràng do thiếu jsonschema. Chỉ fixture thực thi các cell notebook và kiểm tra tĩnh được xác nhận; parity API/CLI qua pipeline thật, full pytest và mock40 là **NOT MEASURED**. Model/GPU, build sạch, training trong image, image size, archive, MD5 và download link đều **NOT MEASURED**.

Hai fixture cũ trong `tests/test_submission.py` còn bắt buộc notebook gọi subprocess và chạy input trước load; chúng không còn phù hợp contract mới. Không sửa vì phạm vi chỉ cho thêm file test mới; cần cập nhật các fixture này khi được phép, trước khi chạy full suite trong môi trường đủ dependency. Đây là kết luận từ đọc mã, không phải lỗi test đã được thực thi trong phiên.

Test parser release và parser COPY từng lỗi ở lần đầu rồi được sửa, qua lần thứ hai. Không còn lỗi trong các test mới được chạy cuối. Không tạo commit/push/chuyển nhánh/stash/reset.

## Tệp đổi trong phiên

- `docker/Dockerfile`
- `docker/README.md`
- `docker/predict_notebook.ipynb`
- `docker/release.ps1`
- `docker/VERIFICATION_2026-11-14.md`
- `src/inference/api.py`
- `src/inference/__main__.py`
- `training_code/README.md`
- `tests/test_notebook_api.py`
- `tests/test_training_repro.py`
- `tests/test_release.py`
- `docs/SUBMISSION_CHECKLIST.md`
- `docs/HANDOFF.md`

HANDOFF chỉ thêm đúng một dòng Nhật ký; git diff --check trên các đường dẫn cho phép exit 0 (stdout rỗng, chỉ cảnh báo LF → CRLF). Tổng checklist được đếm từ 24 dòng: PASS 12, FIXED 6, FAIL/TODO 0, NOT MEASURED 6. Get-Date kiểm tra cuối: 2026-11-14 22:01:10 +07:00.
