# Kiểm chứng CPU và các blocker

Đồng hồ sau kiểm chứng: `2026-11-14 18:25:32 +07:00`. Tất cả lệnh Python dùng đúng interpreter được yêu cầu, không cài package, không build Docker hay tải model/dataset. Hai lệnh bắt buộc không khởi chạy được phần kiểm thử do dependency thiếu sẵn trong môi trường; kết quả toàn bộ suite và mock40: **NOT MEASURED**. Các kiểm tra stdlib chỉ dùng fixture tổng hợp, không phải dữ liệu hay điểm thi.

```powershell
& "C:\Program Files\Python312\python.exe" -B -m pytest -q -p no:cacheprovider
```

Exit code: `1`.

stdout:

```text
(empty)
```

stderr:

```text
C:\Program Files\Python312\python.exe: No module named pytest
```

```powershell
& "C:\Program Files\Python312\python.exe" -B -m src.inference mock --samples 40
```

Exit code: `1`.

stdout:

```text
(empty)
```

stderr:

```text
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\ADMIN\OneDrive - VNU-HCMUS\Documents\XuanQuang\lifemate_superklang\src\inference\__main__.py", line 13, in <module>
    from ..agent.runner import Limits
  File "C:\Users\ADMIN\OneDrive - VNU-HCMUS\Documents\XuanQuang\lifemate_superklang\src\agent\runner.py", line 14, in <module>
    from ..tooling.registry import ToolRegistry, ToolRejected
  File "C:\Users\ADMIN\OneDrive - VNU-HCMUS\Documents\XuanQuang\lifemate_superklang\src\tooling\registry.py", line 7, in <module>
    from jsonschema import Draft202012Validator, validators
ModuleNotFoundError: No module named 'jsonschema'
```

```powershell
& "C:\Program Files\Python312\python.exe" -B -m unittest tests.test_codex_date_table tests.test_codex_submission -v
```

Exit code: `0`.

stdout:

```text
Planner: C:\Users\ADMIN\AppData\Local\Temp\lifemate-submission-test-e7tdhsfd\models\planner; ASR: C:\Users\ADMIN\AppData\Local\Temp\lifemate-submission-test-e7tdhsfd\models\asr; mode: native
Planner: C:\Users\ADMIN\AppData\Local\Temp\lifemate-submission-test-2f9jpvu0\models\planner; ASR: C:\Users\ADMIN\AppData\Local\Temp\lifemate-submission-test-2f9jpvu0\models\asr; mode: native
1 official records
```

stderr:

```text
test_calendar_rollover_uses_input_local_date (tests.test_codex_date_table.DateTableTests.test_calendar_rollover_uses_input_local_date) ... ok
test_flag_off_preserves_pre_change_system_prompt_bytes (tests.test_codex_date_table.DateTableTests.test_flag_off_preserves_pre_change_system_prompt_bytes) ... ok
test_flag_on_only_appends_to_system_prompt (tests.test_codex_date_table.DateTableTests.test_flag_on_only_appends_to_system_prompt) ... ok
test_invalid_anchor_does_not_change_prompt_when_enabled (tests.test_codex_date_table.DateTableTests.test_invalid_anchor_does_not_change_prompt_when_enabled) ... ok
test_no_table_when_anchor_cannot_be_used_without_guessing (tests.test_codex_date_table.DateTableTests.test_no_table_when_anchor_cannot_be_used_without_guessing) ... ok
test_sunday_month_boundary_has_all_current_and_next_week_rows (tests.test_codex_date_table.DateTableTests.test_sunday_month_boundary_has_all_current_and_next_week_rows) ... ok
test_image_has_self_contained_readme_and_fixed_hash_seed (tests.test_codex_submission.SubmissionContractTests.test_image_has_self_contained_readme_and_fixed_hash_seed) ... ok
test_mounted_split_package_wins_over_an_importable_training_copy (tests.test_codex_submission.SubmissionContractTests.test_mounted_split_package_wins_over_an_importable_training_copy) ... skipped 'jsonschema is unavailable in this interpreter'
test_notebook_rejects_different_predictions (tests.test_codex_submission.SubmissionContractTests.test_notebook_rejects_different_predictions) ... ok
test_notebook_setup_and_matching_predictions (tests.test_codex_submission.SubmissionContractTests.test_notebook_setup_and_matching_predictions) ... ok
test_private_record_keeps_public_api_values_and_millisecond_time (tests.test_codex_submission.SubmissionContractTests.test_private_record_keeps_public_api_values_and_millisecond_time) ... skipped 'jsonschema is unavailable in this interpreter'
test_writer_creates_missing_result_directory (tests.test_codex_submission.SubmissionContractTests.test_writer_creates_missing_result_directory) ... skipped 'jsonschema is unavailable in this interpreter'

----------------------------------------------------------------------
Ran 12 tests in 0.038s

OK (skipped=3)
```

Không đổi mã để bỏ qua dependency hoặc thay đổi schema/scoring. Không chạy lại các lệnh thất bại; chủ repo cần cung cấp môi trường dependency đã chuẩn bị rồi chạy lại hai lệnh bắt buộc.
