# Benchmark LifeMate trên Colab T4

Mục tiêu là 80/100 trên phép chấm cục bộ theo quy tắc công bố. Kết quả GPU và điểm đầu cuối hiện **NOT MEASURED**. Notebook dùng kit TRAIN chính thức đã có trên Colab. Đặt repo và kit ở hai đường dẫn riêng.

1. Tải `train+public_test` lên Google Drive một lần. Mở notebook từ GitHub: `https://colab.research.google.com/github/Quang1104/lifemate_superklang/blob/colab-smoke/lifemate_colab.ipynb`, chọn GPU T4. Cell đầu clone nhánh `colab-smoke`, mount Drive và chép kit sang `/content`. Cell cuối của phần smoke lưu file zip kết quả lên `MyDrive/lifemate_results` và tải về máy. Mức sàn: chạy no-op được 12,5 điểm proxy trên 40 mẫu smoke, vì đó là các mẫu có gold rỗng.
2. Chạy các cell chuẩn bị và smoke. Hai nhánh dùng cùng checkpoint, ASR, 40 ID và seed. Nhánh `direct` gọi model mỗi bước. Nhánh `compiled` lập kế hoạch một lần và có tối đa một lượt sửa.
3. Sau khi chốt prompt và model bằng smoke, chạy cell **269 mẫu ngoài smoke**. Notebook giữ riêng dự đoán, chẩn đoán timeout và điểm proxy. Cấu hình hết VRAM có thể chuyển sang `fallback_planner` trước khi bắt đầu lại cả hai nhánh. So sánh theo điểm proxy, tỷ lệ đúng hash, F1 trên mẫu đúng hash, số mẫu cải thiện/hồi quy, p95 và max.

Benchmark đọc toàn bộ audio của mỗi mẫu ở cả hai nhánh. Các model tải ở cell chuẩn bị. Trong lượt suy luận, CLI chỉ nạp weights cục bộ và dùng API chính thức. Báo cáo Colab chứa kết quả TRAIN và nên giữ riêng.

Khi cần phân tích lỗi, chạy lại nhánh bị lỗi trên TRAIN với `--train-transcripts` thay cho `--asr-dir` và ghi ra một tệp riêng. So sánh các ID có lỗi tên, tiền, giờ, tool hoặc thiếu bước với kết quả từ audio. Báo cáo A/B chính yêu cầu cả hai nhánh dùng ASR.

### Kiểm tra cục bộ

- `python -B -m pytest -q -p no:cacheprovider` chạy test compiler, backend mô phỏng, runner, pipeline và scorer.
- `python -B -m src.inference mock --samples 40` kiểm tra vòng lặp với tool giả lập.
- `python -B -m src.inference official --split train --data-root '../datalife/train+public_test' --train-manifest research/smoke_split.json --control-noop --output outputs/implementation_train_control.jsonl --train-report outputs/implementation_train_control.report.json` đã xuất 40/40 bản ghi từ ToolEnv chính thức. Lượt kiểm tra chỉ xác nhận đường xuất JSONL.
- `python -B -m scripts.compare_train prepare --output outputs/implementation_locked269.json` đã tạo manifest 269 ID ngoài vòng smoke.
- `python -B -m scripts.time_span_reach` đo trên TRAIN số tham số ngày/giờ gold mà một đoạn transcript chuyển được. Kết quả: 176/592 (29,7%) trước khi bổ sung `giờ kém`, `N rưỡi`, `đêm`, thứ trong tuần, `ngày/mùng N tháng M` và cách chọn giờ gần nhất. Sau khi bổ sung: 270/592 (45,6%). Phần còn thiếu chủ yếu là giờ tính từ kết quả tra cứu, ví dụ “muộn 1 tiếng”. Đây là chẩn đoán oracle, không phải điểm.
- Nhánh `compiled` có nguồn `shift` để dời một thời điểm đã quan sát theo thời lượng trong câu nói, ví dụ “dời muộn 1 tiếng” hoặc “đón trước giờ hẹn 45 phút”. Python tính giờ mới, model chỉ chọn mốc và đoạn thời lượng. Trên TRAIN, 122/122 cụm thời lượng sau “muộn/sớm/trước/sau/lùi/thêm” có một đoạn con chuyển được. 144/322 tham số giờ gold chưa với tới nằm trong các câu dạng này. Đây là cận trên. Mức tăng thực tế phụ thuộc model và vẫn **NOT MEASURED**.
- `tzdata` có trong dependency vì image CUDA Ubuntu không có cơ sở dữ liệu múi giờ. Khi thiếu `tzdata`, bộ chuyển giờ dùng offset của `current_time`.

## Gói Docker để kiểm tra trên máy đích

Chạy từ repo. Truyền hai thư mục weights cục bộ làm build context riêng. Mặc định dùng `PLANNER_MODE=compiled`. Đổi sang `direct` khi đó là bản thắng theo benchmark.

```bash
docker build -f docker/Dockerfile \
  --build-context planner_weights=/path/to/planner \
  --build-context asr_weights=/path/to/asr \
  --build-arg PLANNER_MODE=compiled -t lifemate:validated .

docker run --rm --gpus all --network none \
  -v /path/to/kit-root:/data:ro -v /path/to/result:/result \
  lifemate:validated
```

Image đọc split private do BTC mount tại `/data/private_test`. Script chính ghi `/result/submission.jsonl`. Notebook đo thời gian gọi cùng entry point và ghi `/result/jupyter_submission.jsonl`. So sánh ID, calls và hash giữa hai lần chạy. Xác nhận cách BTC tính context 16K, GPU đích và phiên bản CUDA trước khi nộp. Base image đang dùng CUDA 13.2 theo phần hướng dẫn bắt buộc trong PDF. [Quy định và nguồn có khác biệt](COMPETITION_RULES.md#unresolved-official-source-conflicts--release-blockers)
