# Ngân sách lỗi LifeMate — 15/11/2026

**Kết luận:** run lưu cục bộ tốt nhất `v3_asrft`: **51.726190/100**, mất **48.273810 điểm**; 17/40 trọn điểm, 23/40 chưa trọn. Đây là proxy TRAIN, không phải điểm portal.
CPU, chỉ phân tích tệp; không chạy model/ToolEnv, không đọc môi trường hay nhãn public. Không có đo GPU mới.
## Nguồn và cách tính
- Smoke: `outputs/colab/smoke_ab_drive/smoke_ab/<run>/native.{jsonl,report.json,scored.json}`; cả ba tệp có đủ ở chín run dưới.
- `research/smoke_split.json`: 40 validation ID; `research/validation_split.json`: 309 validation ID. TRAIN: `../datalife/train+public_test/train/train.jsonl`.
- Đọc `AGENTS.md`, `docs/{COMPETITION_RULES,PUBLIC_GAP,NEXT_PLAN}.md`, `scripts/score_run.py` và scorer phụ thuộc.
- Điểm mẫu trong tệp là 0–1; bảng đổi thành 0–100. Mất điểm = Σ(100 − điểm mẫu)/40. Nhãn chính được gán tay theo khác biệt lời gọi, không phải phân rã nhân quả.
- Mỗi mẫu lỗi có một nhãn chính để tổng không trùng. Nhãn phụ có thể chồng lấp; trần = sửa hoàn toàn mọi lỗi của các mẫu thuộc nhóm, không phải hiệu quả riêng của một sửa đổi.
- Fuzzy theo `research/published_scorer.py`: NFC/lower/whitespace, F1 > 0.6; hash thật từ export đã lưu. Đối chiếu scorer chính thức: chưa xác minh.

Run (proxy): `base` 24.832251; `LIFEMATE_ARG_NORMALIZE` 24.832251; `LIFEMATE_DATE_TABLE` 25.059524; `LIFEMATE_PROMPT_V2` 24.392857; `oracle_transcripts` 49.916667; `v3` 34.583333; `v3_asrft` 51.726190; `w_asrft` 44.767857; `w_fewshot2` 30.416667

Không có v4/v5 trong các thư mục đã kiểm; số v4 trong NEXT_PLAN không có tệp chạy kèm tại đây. Hai run phụ `outputs/colab/planner_sft/smoke40_sft{154,50}_x3` có 40 mẫu, điểm 24.832251 / 21.809524; TIME_SCALE=3, không chứng minh đạt 30 giây.
**Validation309: NOT MEASURED.** Chỉ có export gold309; không có run dự đoán309 trong cây output đã kiểm. Hai control ở `outputs/` đều 40 mẫu. Không chấm gold như dự đoán.
## Xếp hạng nhãn chính và 5 hướng sửa
| Hạng | Nhãn chính | Mẫu | Điểm mất / trần smoke | Sửa dữ liệu/prompt |
|---|---|---:|---:|---|
| 1 | Ngày/giờ | 5 | 12.500000 | Dữ liệu TRAIN tương đối/tuần/ngày; prompt giữ mốc và duration, điền time đã nêu. |
| 2 | Gọi thừa | 6 | 8.898810 | Ví dụ điều kiện sai → không gọi; kiểm điều kiện trước ghi, bỏ đọc không phục vụ yêu cầu. |
| 3 | ID thực thể | 3 | 7.500000 | Ví dụ chọn ID theo kết quả đọc và điều kiện số dư; không lấy ID chỉ theo vị trí. |
| 4 | Sai/thiếu tool | 2 | 5.000000 | Ví dụ TRAIN đọc→rẽ nhánh→ghi đúng app; phân biệt đọc Zalo và gửi tin. |
| 5 | Đáng lẽ hỏi thiếu | 2 | 5.000000 | Ví dụ tương phản thiếu/đủ appliance_id, term_months; hỏi đúng trường, không tự điền. |
| — | Tên/liên hệ | 2 | 3.750000 | Ngoài top 5 |
| — | Hỏi không cần | 1 | 2.500000 | Ngoài top 5 |
| — | Dừng sau đọc | 1 | 2.500000 | Ngoài top 5 |
| — | Số/tham số số | 1 | 0.625000 | Ngoài top 5 |

Hai nhóm 5 điểm đồng hạng. Trần các nhóm chính cộng đúng 48.273810; nhiều mẫu có lỗi phụ nên sửa riêng nguyên nhân chính có thể chưa tăng điểm. Trần tăng public / khả năng lên 70: **NOT MEASURED**.
## Từng mẫu smoke (điểm /100)
Nhãn trong bảng là nhãn chính; chi tiết nêu lỗi phụ. O = điểm run `oracle_transcripts`; **ASR-caused của từng mẫu = NOT MEASURED** vì oracle khác planner.
| ID | Điểm | Nhãn chính / bằng chứng | O |
|---|---:|---|---:|
| train_0045 | 100.0000 | Đạt | 100.0000 |
| train_0086 | 100.0000 | Đạt | 0.0000 |
| train_0087 | 0.0000 | ID thực thể: Sai tài khoản nguồn; thêm open_savings | 0.0000 |
| train_0088 | 100.0000 | Đạt | 100.0000 |
| train_0131 | 0.0000 | Hỏi không cần: Anh Tuấn thay Tuấn; hỏi thay gửi SMS | 80.0000 |
| train_0169 | 75.0000 | Gọi thừa: Thử ví thiếu tiền và đọc thêm số dư ngân hàng | 75.0000 |
| train_0177 | 100.0000 | Đạt | 100.0000 |
| train_0188 | 0.0000 | Ngày/giờ: 13:19 thay 13:55 | 0.0000 |
| train_0229 | 0.0000 | ID thực thể: acc_02 thay acc_01 | 0.0000 |
| train_0246 | 75.0000 | Số/tham số số: Thêm floor=2 vào get_status | 0.0000 |
| train_0280 | 0.0000 | ID thực thể: light_bedroom2 thay light_bedroom | 100.0000 |
| train_0286 | 83.3333 | Gọi thừa: Thử ví trước và đọc thêm số dư | 100.0000 |
| train_0350 | 100.0000 | Đạt | 0.0000 |
| train_0373 | 100.0000 | Đạt | 0.0000 |
| train_0413 | 100.0000 | Đạt | 0.0000 |
| train_0432 | 100.0000 | Đạt | 100.0000 |
| train_0450 | 100.0000 | Đạt | 100.0000 |
| train_0578 | 0.0000 | Ngày/giờ: Thiếu time buổi tập; magie thành canxi | 100.0000 |
| train_0583 | 100.0000 | Đạt | 100.0000 |
| train_0611 | 0.0000 | Ngày/giờ: 03/10 thay 04/10; thêm list_events | 0.0000 |
| train_0665 | 50.0000 | Tên/liên hệ: Hạnh thay Cô Hạnh (query khớp chính xác) | 50.0000 |
| train_0675 | 0.0000 | Tên/liên hệ: Sai người nhận email và nội dung | 0.0000 |
| train_0695 | 0.0000 | Sai/thiếu tool: contacts.search thay list_chats; hỏi thay gửi Zalo | 75.0000 |
| train_0707 | 0.0000 | Ngày/giờ: 20/10 thay 22/10; thiếu update_event | 0.0000 |
| train_0781 | 100.0000 | Đạt | 100.0000 |
| train_0937 | 100.0000 | Đạt | 100.0000 |
| train_1012 | 100.0000 | Đạt | 100.0000 |
| train_1015 | 85.7143 | Gọi thừa: Thêm recent_calls | 100.0000 |
| train_1041 | 100.0000 | Đạt | 0.0000 |
| train_1095 | 0.0000 | Sai/thiếu tool: Gửi ba tin thay đọc chat rồi xoá rule | 66.6667 |
| train_1161 | 0.0000 | Gọi thừa: Gold rỗng; thêm contacts.search | 100.0000 |
| train_1202 | 0.0000 | Ngày/giờ: 19:00 thay 17:30; thêm end (title vẫn qua fuzzy) | 50.0000 |
| train_1296 | 0.0000 | Đáng lẽ hỏi thiếu: Tự chọn máy thay hỏi appliance_id | 0.0000 |
| train_1360 | 0.0000 | Gọi thừa: Gold rỗng; vẫn redeem | 0.0000 |
| train_1369 | 100.0000 | Đạt | 100.0000 |
| train_1493 | 100.0000 | Đạt | 100.0000 |
| train_1582 | 0.0000 | Gọi thừa: Gold rỗng; ba lời gọi ngân hàng | 0.0000 |
| train_1598 | 0.0000 | Đáng lẽ hỏi thiếu: Tự chọn kỳ hạn thay hỏi term_months | 0.0000 |
| train_1645 | 0.0000 | Dừng sau đọc: Chỉ get_current; thiếu forecast và gửi vị trí | 0.0000 |
| train_1655 | 100.0000 | Đạt | 0.0000 |

## Nhãn phụ chồng lấp: không cộng các dòng
| Loại quan sát | Điểm mất trên tập mẫu | ID (bỏ tiền tố train_) |
|---|---:|---|
| wrong_args:entity_ids | 7.500000 | 0087, 0229, 0280 |
| extra_call | 16.398810 | 0087, 0169, 0286, 0611, 1015, 1095, 1161, 1360, 1582 |
| asked_when_should_not | 5.000000 | 0131, 0695 |
| wrong_args:names_contacts | 8.750000 | 0131, 0665, 0675, 0695 |
| wrong_missing_tool | 17.500000 | 0131, 0695, 0707, 1095, 1296, 1598, 1645 |
| wrong_args:enums | 1.041667 | 0169, 0286 |
| wrong_args:dates_times | 12.500000 | 0188, 0578, 0611, 0707, 1202 |
| wrong_args:numbers | 0.625000 | 0246 |
| wrong_args:free_text | 5.000000 | 0578, 0675 |
| should_have_asked | 5.000000 | 1296, 1598 |
| stopped_after_read | 2.500000 | 1645 |

Sai args gồm ngày/giờ, tên/liên hệ, số, enum, văn bản và ID. Enum ở đây là chọn sai nguồn thanh toán, không phải giá trị ngoài schema; số là thêm bộ lọc floor, không phải nghe sai lượng. Văn bản title ở 1202 qua fuzzy, không tính là lỗi văn bản.
**ASR:** oracle dùng sft154/train_oracle (49.916667), v3_asrft dùng sft_v3/asr_ft (51.726190). Chênh lệch +15.065476 ở 9 mẫu, −16.875000 ở 8 mẫu, ròng −1.809524 là đối chiếu có nhiễu planner, không phải ngân sách ASR.
`w_asrft` là ASR fine-tune, không phải gold; cùng tên artifact sft154 với oracle nhưng chưa chứng minh cùng prompt/cờ. Chênh +5.148810 của oracle chỉ là mô tả. Transcript trong scored là TRAIN gold, không phải bản ASR; không dùng nó để kết luận ASR sai.
## Public: chỉ thống kê dự đoán, không quy lỗi
Nguồn: `outputs/colab/public_drive/public/{sft154,v3,v3_asrft}/submission.jsonl`. TRAIN gold: toàn bộ1545 và tập fit1236 (= bỏ validation309), chỉ đọc `label.tool_calls`.
| Tập | N | Calls / mẫu | Rỗng | Có hỏi | Cuối là read | Chỉ read | Mẫu trùng call |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN_all | 1545 | 2.2919 | 111 | 217 | 138 | 124 | 2 |
| TRAIN_fit1236 | 1236 | 2.2977 | 89 | 180 | 105 | 96 | 2 |
| sft154 | 634 | 1.8612 | 140 | 43 | 96 | 87 | 0 |
| v3 | 634 | 2.4164 | 29 | 62 | 92 | 83 | 0 |
| v3_asrft | 634 | 2.4543 | 29 | 60 | 85 | 71 | 0 |

Có hỏi = ít nhất một sys.ask_missing_info/mẫu; cuối read = catalog kind=read ở call cuối; chỉ read = mọi call đều read, không rỗng. Đây là tín hiệu quan sát, không chứng minh dừng sai hay hỏi sai.
| Tập | Ngày/giờ: sai hình dạng/tổng | Số: sai kiểu/tổng | Enum: ngoài tập/tổng |
|---|---:|---:|---:|
| TRAIN_all | 0/580 | 0/320 | 0/481 |
| TRAIN_fit1236 | 0/456 | 0/262 | 0/382 |
| sft154 | 0/231 | 0/237 | 0/179 |
| v3 | 0/256 | 0/293 | 0/199 |
| v3_asrft | 0/261 | 0/310 | 0/187 |

Ngày/giờ kiểm hình dạng YYYY-MM-DD, ISO datetime hoặc HH:MM[:SS] trên khóa ngày/giờ; không kiểm nghĩa/ngày lịch. Số và enum theo schema catalog. Không phát hiện tool ngoài catalog ở các tập trên; chưa đo app được cài vì không đọc đầu vào public.
Sai/thiếu tool, gọi thừa về nghĩa, sai ngày/tên/số/văn bản/ID, đáng lẽ hỏi, hỏi không cần, dừng sai và ASR-caused trên public: **NOT MEASURED**. Chỉ đếm calls, lặp, hỏi, đọc và hình dạng; không suy nhãn từ chênh phân bố.
## Kiểm chứng và tái lập
`node research/error_budget.mjs --check` kiểm 360 điểm bằng proxy hiện có; kiểm ID40, lời gọi/hash với export thật, trung bình, tổng ngân sách, gold TRAIN và nhãn chính; không replay.
`node research/error_budget.mjs --write` viết tài liệu và thử viết PROGRESS.md/REPORT.md vào staging; lỗi quyền ghi được báo rõ, không đổi log chính thức.
Bản này không sửa src/config/log, không commit, không train/download. Hardware/CUDA còn mâu thuẫn như COMPETITION_RULES; không chọn ngầm một cấu hình.
