# Phân tích smoke40 QLoRA bước 50
Nguồn: `outputs/colab/planner_sft/smoke40_sft50_x3/native.scored.json` và `native.report.json`; TRAIN chỉ lấy ID và `context.current_time`.
40 mẫu; proxy 21.809524/100; 6 mẫu trọn điểm, 34 chưa trọn. Đây là proxy TRAIN, không phải điểm thi chính thức.
Mỗi mẫu gán một nhóm theo sai khác quan sát nổi bật; nhóm không chứng minh nguyên nhân gốc. Nhóm ASR chỉ ghi khi văn bản đối số khác rõ transcript; lỗi do ASR hay planner: NOT MEASURED vì không có transcript ASR/raw generation.

| ID | Score [0,1] | Category | Bằng chứng gold → predicted |
|---|---:|---|---|
| train_0045 | 0 | early stop in multi-step | Gold 4 lệnh gồm get_balance + transfer; predicted chỉ list_bills + list_beneficiaries (2 rejection, lý do NOT MEASURED). |
| train_0086 | 0.666667 | ASR word error | contacts.search.query: “Minh” → “cha”, khác transcript “gọi cho Minh”; alarm và ask_missing_info khớp. |
| train_0087 | 0 | other | Gold 2 banking.transfer; predicted banking.open_savings(209000,12 tháng) và list_transactions. |
| train_0088 | 0 | wrong argument value | bill_payment.pay: bill_02 + bill_03 → chỉ bill_01; không đủ dữ liệu để quy lỗi tính hạn ngày. |
| train_0131 | 0 | date arithmetic | “mai”, current_time 2026-11-09: forecast 2026-11-10 → 2026-11-09; sau đó chọn nhánh ride thay vì hủy/SMS. |
| train_0169 | 0 | wrong argument value | pay.source: bank → ewallet; predicted cũng thiếu ewallet.get_balance. |
| train_0188 | 0 | ASR word error | reminder.content “gọi lại cho khách” → “đổi lại cho khách”, khác transcript; time 13:55 → 13:15 (current_time 13:10, +45 phút). |
| train_0229 | 0 | zero calls | Gold 4 lệnh thanh toán/chuyển tiền → []; report steps=1, dispatched=0, rejections=0. |
| train_0246 | 0 | zero calls | Gold get_status + 2 light_set + climate_set → []; report steps=1, dispatched=0, rejections=0. |
| train_0280 | 0 | other | Gold tắt đèn + contacts.search + ask_missing_info SMS → banking.get_balance/list_beneficiaries/transfer(500000); gold không rỗng. |
| train_0286 | 0.8 | wrong argument value | contacts.search.query “Sếp Dũng” → “sếp dũng”; cả 5 lệnh có mặt, state khớp, SMS cũng đổi cách diễn đạt. |
| train_0350 | 0 | missing ask_missing_info | Sau search Lan, gold hỏi contact_id → predicted phone.call(c_05). |
| train_0373 | 0 | wrong argument value | light_set.brightness 30 → 70, trong khi transcript nói “30 phần trăm”; 2 lệnh còn lại khớp. |
| train_0413 | 0 | missing ask_missing_info | Sau search Lan, gold hỏi contact_id → phone.call(c_08); alarm cũng lệch 06:30 → 05:30. |
| train_0578 | 0 | early stop in multi-step | Gold log_workout(time 06:30) + reminders.create → chỉ log_workout không time. |
| train_0583 | 0 | other | Gold calendar.update_event(evt_03,16–17h) → calendar.create_event(15–16h). |
| train_0611 | 0 | date arithmetic | “Chủ nhật này”, current_time 2026-10-01: forecast 2026-10-04 → 2026-10-02; state khớp nhưng lệnh sai. |
| train_0665 | 0 | other | Gold search “Cô Hạnh” + ewallet.transfer(720000) → chỉ ask_missing_info(contact_id), hỏi trước khi tra cứu. |
| train_0675 | 0 | wrong argument value | email.to team.data@example.test → n@data.com, subject/body đổi ý; còn thiếu subscriptions.cancel. |
| train_0695 | 0.857143 | other | Gold 3 lệnh → predicted 4, dư calendar.list_events(date_from/date_to=2026-10-21); state khớp. |
| train_0707 | 0 | other | Gold video_meeting.create + calendar.list_events/update_event → video_meeting.list_meetings/join. |
| train_0781 | 0.4 | wrong argument value | create_rule thiếu title “Tắt bình nóng lạnh lúc 22:30”; còn thiếu home_energy.get_usage(period=week), state khớp. |
| train_0937 | 0 | zero calls | Gold alarm.create + subscriptions.list/cancel → []; report steps=1, dispatched=0, rejections=0. |
| train_1012 | 0 | wrong argument value | alarm.time 2026-10-24T03:30 → 05:45; flights.my_bookings/check_in khớp. |
| train_1015 | 0 | early stop in multi-step | Gold tasks.list + contacts.search “Mạ” + phone.call → chỉ tasks.list (1 rejection, lý do NOT MEASURED). |
| train_1041 | 0 | zero calls | Gold 5 lệnh đọc Zalo/nhạc và thêm playlist → []; report steps=1, dispatched=0, rejections=0. |
| train_1095 | 0 | other | Sau list_chats, gold read_chat(chat_352) + delete_rule → send_message(chat_6932), sai hành động/đối tượng. |
| train_1202 | 0 | ASR word error | Transcript “trước họp tổ hai tiếng” → tasks.title “rước học tổ hai tiếng”; gold calendar.create_event 17:30 → tasks.add 19:30. |
| train_1360 | 0 | action on chit-chat/empty gold | Gold [] (nhánh không đổi thưởng) → loyalty.redeem; không suy diễn nguyên nhân từ trạng thái ẩn. |
| train_1493 | 0 | missing ask_missing_info | Gold hỏi tracking_id → reschedule_delivery với một tracking_id và time cụ thể. |
| train_1582 | 0 | action on chit-chat/empty gold | Gold [] → banking.list_beneficiaries + contacts.search “Trần Văn Tuấn”; state vẫn khớp. |
| train_1598 | 0 | missing ask_missing_info | Gold hỏi term_months → list_beneficiaries + open_savings(amount=1000000,term_months=12). |
| train_1645 | 0 | date arithmetic | “sáng ni”, current_time 2026-12-19: forecast 2026-12-19 → 2026-12-20; thiếu gửi vị trí cho Nhi. |
| train_1655 | 0 | action on chit-chat/empty gold | Gold [] → tasks.list + tasks.complete(task_05), dù transcript nhờ giữ sách thư viện. |

Điểm mất = `100/40 * Σ(1-score)`; tổng 78.190476 điểm. Không cộng lại các lỗi phụ trong cùng mẫu.
| Category | Số mẫu | Điểm proxy mất /100 |
|---|---:|---:|
| other | 7 | 15.357143 |
| wrong argument value | 7 | 14.500000 |
| missing ask_missing_info | 4 | 10.000000 |
| zero calls | 4 | 10.000000 |
| date arithmetic | 3 | 7.500000 |
| action on chit-chat/empty gold | 3 | 7.500000 |
| early stop in multi-step | 3 | 7.500000 |
| ASR word error | 3 | 5.833333 |

Lệnh tính/kiểm tra (chỉ đọc các file allowlist; bảng nhóm ở trên là đầu vào phân loại thủ công):
`& "C:\Program Files\Python312\python.exe" -B research/sft50_failure_analysis.py`
```text
samples=40 nonperfect=34 proxy=21.809524 lost=78.190476; TRAIN=1545 validation=309 SFT=1236 holdout=171 contaminated=171
other: count=7 lost=15.357143
wrong argument value: count=7 lost=14.500000
missing ask_missing_info: count=4 lost=10.000000
zero calls: count=4 lost=10.000000
date arithmetic: count=3 lost=7.500000
action on chit-chat/empty gold: count=3 lost=7.500000
early stop in multi-step: count=3 lost=7.500000
ASR word error: count=3 lost=5.833333
zero-call failures: train_0229,train_0246,train_0937,train_1041; each steps=1 dispatched=0 rejections=0
checks=OK
```

Ba nhóm đầu theo điểm mất (missing ask_missing_info và zero calls đồng hạng; chọn nhóm hỏi thiếu theo thứ tự category của yêu cầu):
| Nhóm | Ý tưởng sửa tối thiểu, chưa triển khai | Cần GPU để xác nhận? | Rủi ro với mẫu đang đúng |
|---|---|---|---|
| other (15.357143) | Thêm một quy tắc prompt: chọn tool theo động từ/đối tượng của yêu cầu hiện tại, không lấy hành động từ lịch sử; ví dụ “dời”→update_event, “đọc”→read_chat. | Có, để đo lại model + proxy; fixture đối chiếu tool chạy CPU được. | Có thể làm yếu xử lý yêu cầu tham chiếu lịch sử (vd. 1369); không cấm dùng lịch sử để lấy đối tượng. |
| wrong argument value (14.500000) | Thêm một câu trước khi gọi: giữ nguyên tên riêng, số và giờ trong transcript khi điền đối số, đối chiếu từng trường với yêu cầu/kết quả đã thấy (vd. query “Sếp Dũng”, brightness=30, alarm=03:30). | Có cho mức tuân thủ/điểm thực; kiểm tra ví dụ số/chuỗi dùng CPU. | Sao chép quá sát có thể hại chuyển enum/giờ tương đối và fuzzy text đang đúng; đo cả 6 mẫu trọn điểm. |
| missing ask_missing_info (10.000000) | Sau tra cứu không chọn duy nhất được ID, hoặc còn thiếu trường chưa có căn cứ (term_months), gọi ask_missing_info đúng tool/params thay vì tự chọn ID/kỳ hạn. | Có cho quyết định model; fixture ambiguity chạy CPU. | Hỏi thừa có thể làm hỏng mẫu giải được từ lịch sử, như lỗi 0665; phải thử 1296 và 0177. |
Các ý tưởng chỉ là giả thuyết; mức tăng điểm và tác động mẫu đúng: NOT MEASURED. Với zero calls, cả 4 lỗi đều bước 1/0 rejection: không có bằng chứng guard→lặp ở run này; raw model text/lý do dừng: NOT MEASURED. Hai mẫu [] đúng 0432/1161 không tính vào nhóm lỗi.

Leave-app-out: 171/171 ID holdout nằm trong SFT training = 1545 TRAIN − 309 validation = 1236 ID (0 holdout ngoài SFT); đối chiếu manifest leave_app_out_split.json và validation_split.json.
Tập SFT chứa toàn bộ holdout nên split này không đo sạch tổng quát hóa tool/app chưa thấy; số holdout thực sự đã được optimizer đọc tới bước 50: NOT MEASURED. Cần huấn luyện adapter khác với các ID holdout bị loại rồi đánh giá một lần; chỉ đổi pool few-shot không loại phơi nhiễm từ SFT. Kết quả unseen-tool sạch: NOT MEASURED.
