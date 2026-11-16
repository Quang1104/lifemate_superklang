# Ước lượng PUBLIC không dùng nhãn PUBLIC

Kết quả ngày 16/11/2026: **không thể xếp hạng đáng tin cậy v6 và v8a** bằng phương pháp này.
Chênh lệch ước lượng cùng mốc chỉ 2.924 điểm; sai số kiểm tra ngược lớn nhất 4.782 điểm.
Các số ước lượng dưới đây là ngoại suy offline, **không phải điểm portal đã đo**.

## Cách chạy

Từ thư mục gốc repo, chỉ đọc các đường dẫn cụ thể, không tạo cache hoặc ghi kết quả:
```powershell
python -B -m research.public_estimate
python -B -m unittest discover -s tests -p test_codex_public_estimate.py -v
```
Có thể đổi `--repo-root`, `--scored-dir`, `--mixed`, `--data-root`; mặc định kit là `../datalife/train+public_test`.
Script tự đọc các tên `pub_v8b.jsonl`, `smoke_v8b.json`, `val_v8b.json` khi chúng có mặt; thiếu dữ liệu ghi NOT MEASURED.
Không dùng ToolEnv, không đọc nhãn PUBLIC, môi trường, state.json, hoặc tính lại hash.

## Phương pháp và giới hạn

Với mốc R: `ước_lượng(C) = điểm_portal(R) + n_diff/634 × d`.
`n_diff` đếm ID có tool_calls hoặc state_hash khác; giữ thứ tự calls, bỏ qua thứ tự khóa JSON.
`d = 100 × trung_bình(score_C − score_R)` trên các cặp TRAIN có đầu ra khác.
Ưu tiên VAL khi cả hai có VAL, sau đó smoke; nếu chỉ một bên có VAL thì lấy đủ các ID smoke chung.
VAL309 của v7 chứa đủ smoke40 nên kiểm tra trực tiếp v7 với v3_asrft được, không cần nối hai ước lượng.
Kiểm tra trùng ID, đủ ID PUBLIC, đủ cặp TRAIN, score [0,1], và cùng gold TRAIN trước khi tính.
**Dump TRAIN thiếu state_hash dự đoán**: chỉ nhận diện chắc chắn khi calls hoặc state_match khác nhau.
Calls bằng nhau và cả hai state_match=false vẫn có thể khác hash: không coi là chắc chắn bằng nhau.
Điểm trung tâm dùng các cặp nhận diện khác; báo số cặp mơ hồ và miền d khi thêm các delta=0 đó.
Nếu có hash TRAIN trong dump thì so trực tiếp; không dựng hash hoặc suy ra trạng thái ẩn.
Khoảng xấp xỉ 95% dùng mean ± 1.96×SE, lấy bao ngoài qua mọi mẫu số mơ hồ rồi nhân n_diff/634.
Đây là khoảng bất định lấy mẫu TRAIN có điều kiện; không bao gồm lệch miền, sai khác scorer, chọn model hay làm tròn portal.
Mẫu nhỏ; không diễn giải khoảng này thành khoảng dự báo PUBLIC đã hiệu chuẩn. Không đủ hai delta: khoảng NOT MEASURED.

## Kiểm tra ngược, không khớp tham số theo điểm thật

| C | R | n_diff | d | Ước lượng | Khoảng SE | Điểm thật | Sai số ước lượng − thật |
|---|---|---:|---:|---:|---|---:|---:|
| mixed | v3_asrft | 66 | 10.143 | 43.306 | [40.309, 46.303] | 42.07 | +1.236 |
| v7 | v3_asrft | 373 | -6.853 | 38.218 | [24.758, 51.679] | 43 | -4.782 |

mixed dùng smoke_v4 − smoke_v3asrft (18 cặp khác, 6 mơ hồ); v7 dùng val_v7 ∩ smoke_v3asrft (22 khác, 6 mơ hồ).
Điểm v7=43 do người dùng báo, chưa biết độ làm tròn; sai số tính theo đúng số đã báo.
122 hàng được thay của mixed chỉ tạo 66 đầu ra khác v3_asrft; không dùng 122 làm n_diff.

## Ứng viên

| C | R | n_diff | d | Ước lượng | Khoảng SE | Nguồn TRAIN (40 cặp) |
|---|---|---:|---:|---:|---|---|
| v6 | v3_asrft | 346 | 3.593 | 44.211 | [29.094, 59.328] | smoke_v6 / smoke_v3asrft; 20 khác, 6 mơ hồ |
| v8a | v3_asrft | 364 | 8.507 | 47.134 | [33.593, 60.676] | smoke_v8a / smoke_v3asrft; 23 khác, 4 mơ hồ |
| v6 | v7 | 299 | 13.914 | 49.562 | [38.499, 60.625] | smoke_v6 / val_v7; 16 khác, 8 mơ hồ |
| v8a | v7 | 311 | 16.497 | 51.092 | [36.851, 65.333] | smoke_v8a / val_v7; 21 khác, 5 mơ hồ |
| v8b | mọi mốc | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | chưa có file PUBLIC |

v3=27.98 thiếu TRAIN tương ứng; mixed=42.07 thiếu TRAIN cho chính sách trộn: ước lượng từ hai mốc này NOT MEASURED.
Không dùng TRAIN v4 như thể đó là toàn bộ chính sách mixed; chỉ dùng cho kiểm tra phần thay thế được yêu cầu.
Miền d do thiếu hash: v6/v3_asrft [2.764,3.593], v8a/v3_asrft [7.247,8.507].
Thay mốc sang v7 đổi mạnh ước lượng; không lấy trung bình các mốc để che bất nhất này.

## Phân tầng và quyết định

PUBLIC có 224 mẫu chỉ cài app TRAIN, 410 mẫu cài ít nhất một app chỉ có ở PUBLIC; không suy ra app đích.
Số ID khác theo hai nhóm TRAIN-app / public-only-app: mixed/v3_asrft 66/0; v7/v3_asrft 120/253.
v6/v3_asrft 111/235; v8a/v3_asrft 120/244; v6/v7 98/201; v8a/v7 111/200.
TRAIN không có các app PUBLIC-only nên hiệu ứng điểm riêng nhóm này **NOT MEASURED**; d gộp chỉ là giả định ngoại suy.
Kiểm tra mixed chỉ phủ nhóm TRAIN-app; không xác nhận chất lượng trên 410 mẫu còn lại.
v8a cao hơn v6 theo điểm trung tâm, nhưng sai số kiểm tra ngược lớn hơn chênh lệch và các khoảng chồng lấn rộng.
**Không dùng bảng này để tuyên bố v8a tốt hơn v6 hoặc v7; lựa chọn ứng viên tốt nhất chưa được đo đáng tin cậy.**
