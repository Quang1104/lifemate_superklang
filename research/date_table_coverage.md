# Độ phủ bảng ngày trên TRAIN

Đo lúc `2026-11-14T18:22:02.6819238+07:00` bằng helper `render_date_table` thực tế trong `src/agent/native_planner.py`. Đây là oracle độ phủ đối số, không phải điểm model; tác động tới điểm và độ trễ model: **NOT MEASURED**.

Chỉ đọc JSONL TRAIN được allowlist và `research/validation_split.json`; loại 309 ID validation trước khi phân tích context/gold. Tập ID còn lại được kiểm tra khớp hoàn toàn `train_ids` trong manifest. Không khởi tạo ToolEnv, không đọc environment hay trạng thái.

Mẫu số là số lần xuất hiện giá trị scalar string ISO calendar date hoặc datetime hợp lệ trong gold `tool_calls[].arguments`, đệ quy qua object/list. Đối số lặp được đếm từng lần; chỉ xét phần ngày theo offset ghi trong chuỗi, không đổi múi giờ. Ngày được xem là phủ nếu có trong bảng dựng từ `context.current_time` của chính mẫu.

Lệnh PowerShell chính xác:

```powershell
& "C:\Program Files\Python312\python.exe" -B -m research.date_table_coverage --train "C:/Users/ADMIN/OneDrive - VNU-HCMUS/Documents/XuanQuang/datalife/train+public_test/train/train.jsonl"
```

Output chính xác:

```text
TRAIN date-table oracle coverage (argument occurrences; no model score)
rows_read: 1545
validation_samples_excluded: 309
train_samples_selected: 1236
invalid_or_missing_anchors: 0
gold_calls: 2840
samples_with_iso_arguments: 329
covered_iso_date_arguments: 115/121 (95.04%)
covered_iso_datetime_arguments: 332/355 (93.52%)
covered_all_iso_arguments: 447/476 (93.91%)
samples_with_all_iso_dates_covered: 311/329 (94.53%)
```

Kiểm tra đồng hồ sau milestone:

```powershell
Get-Date -Format o
```

```text
2026-11-14T18:22:02.6819238+07:00
```
