# Tái lập planner QLoRA của LifeMate

Planner được chọn là Qwen/Qwen3-4B-Instruct-2507 fine-tune bằng QLoRA, sau đó merge adapter thành checkpoint fp16 để chạy inference với `--planner-mode native`. ASR mặc định là PhoWhisper-medium; có thể thay bằng checkpoint medium/large fine-tune bằng `training_code/train_asr.py` (xem mục ASR bên dưới). Theo thông tin phiên Colab đang chạy do chủ repo cung cấp, cấu hình đích là một epoch; lần kiểm tra Windows CPU này không xác nhận checkpoint cuối đã hoàn tất.

Chạy ba lệnh dưới đây từ thư mục làm việc `/code` của image, hoặc từ gốc repo trên Colab. Image chứa các tệp nguồn sau; các import nội bộ của builder đã nằm trong phần `src` được đóng gói cho inference:

| Tệp trong image | Vai trò |
|---|---|
| `/code/scripts/build_sft.py` | Dựng hội thoại SFT native từ TRAIN. |
| `/code/src/agent/native_planner.py` | `build_messages` và `tool_definitions` giống inference. |
| `/code/src/agent/planner.py` | `Observation`, `PlannerInput`. |
| `/code/src/inference/official.py` | `OfficialKit`, ToolEnv API chính thức. |
| `/code/src/inference/pipeline.py` | `_sample_from_row`. |
| `/code/src/tooling/registry.py` | `ToolRejected`. |
| `/code/research/validation_split.json` | Manifest loại validation, `source_split=train`. |
| `/code/training_code/train_qlora.py` | Huấn luyện adapter. |
| `/code/training_code/merge_lora.py` | Merge adapter vào base fp16. |
| `/code/configs/colab_models.json` | Revision model đã ghim. |
| `/code/pyproject.toml` | Dependency runtime/model. |

Đầu vào dữ liệu là bản kit đầy đủ `../datalife/train+public_test`, chỉ dùng split TRAIN. Khi tái lập, đặt biến môi trường `TRAIN_KIT` trỏ đến gốc kit đã mount; builder đọc `train/train.jsonl` (context, history, `audio_text` của lượt hiện tại/lịch sử và `label.tool_calls` TRAIN), `train/train_tools_catalog.json` và package `train/zaic_tools/__init__.py`. Đây là các đầu vào ngoài image, không được COPY dữ liệu hay package chính thức vào image. Không truy cập nhãn PUBLIC/PRIVATE; không đọc trực tiếp trạng thái môi trường. Builder tạo đúng một ToolEnv cho mỗi mẫu, chỉ gọi các gold call của yêu cầu hiện tại qua API để lấy kết quả tool; history chỉ là ngữ cảnh và không được replay. Bước dựng SFT này là phân tích oracle TRAIN cục bộ, không phải đường inference.

Manifest có **1236 ID train**, loại **309 ID validation** bằng `--exclude`; hai tập rời nhau, tổng 1545 ID. Các số này được kiểm lại trực tiếp từ manifest trong phiên CPU. Không truyền `--limit`. Nếu kit khác bản gốc hoặc validation manifest khác, phải dừng thay vì nhận tập ID mới. Builder có thể bỏ mẫu bị `ToolRejected`; số dòng SFT thực tế và số dòng còn lại sau tokenize phải lấy từ stdout lần chạy, hiện **NOT MEASURED** trong phiên này.

Đặt `BASE_MODEL` trỏ đến bản local đã chuẩn bị sẵn của Qwen/Qwen3-4B-Instruct-2507, revision `cdbee75f17c01a7cc42f958dc650907174af0554` (khóa `fallback_planner` trong cấu hình). Không dùng khóa `planner` Qwen3.5-4B. Đặt `RUN_DIR` là thư mục ghi kết quả bên ngoài image. Các đường dẫn dưới `RUN_DIR` là sản phẩm được tạo, không phải file được COPY. Base để train/merge phải được mount riêng khi tái lập: image phát hành chỉ chứa planner đã merge tại `/models/planner/` và ASR tại `/models/asr/`, không chứa thêm base hoặc adapter trung gian.

```sh
python -B -m scripts.build_sft --data-root "$TRAIN_KIT" --exclude research/validation_split.json --output "$RUN_DIR/sft_native.jsonl"
python -B training_code/train_qlora.py --model-dir "$BASE_MODEL" --data "$RUN_DIR/sft_native.jsonl" --output-dir "$RUN_DIR/qlora_native" --epochs 1 --save-every 25 --max-tokens 12000 --seed 20261113 --lr 1e-4 --rank 16 --alpha 32 --grad-accum 8
python -B training_code/merge_lora.py --model-dir "$BASE_MODEL" --adapter-dir "$RUN_DIR/qlora_native" --output-dir "$RUN_DIR/planner_merged"
```

Không dùng `--max-steps` cho lượt một epoch này. Hàm encode áp chat template với `enable_thinking=False`, không thêm generation prompt, bỏ hội thoại vượt 12000 token hoặc template không prefix-stable. Loss chỉ phủ token assistant của yêu cầu hiện tại, bao gồm tool call/câu kết và token kết thúc; system/history/user/tool result có nhãn -100. SFT dùng lời chép TRAIN đã phát hành; inference thật dùng ASR, nên độ lệch do ASR vẫn cần đo.

Các hyper-parameter lấy từ mã nguồn: seed train **20261113** cho Python random và PyTorch; manifest/inference seed **20261112** là mục đích khác. Micro-batch một mẫu; gradient accumulation 8; LoRA rank 16, alpha 32, dropout 0.05, task CAUSAL_LM, target modules `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`. Base nạp 4-bit NF4, double quantization, compute dtype theo GPU; SDPA, gradient checkpointing, `use_cache=False`. Optimizer AdamW, learning rate 1e-4, weight decay 0, gradient clipping norm 1.0; cosine schedule, warmup `max(1, total_steps // 20)`. `total_steps = int(max(1, used_examples // 8) * epochs)`; `--max-steps` nếu được truyền sẽ ghi đè epochs. Không suy ra số step thực tế khi chưa có stdout tokenizer.

Phần cứng thực sự của lượt train được chủ repo xác nhận là **Colab T4**. Mã dùng `torch.cuda.is_bf16_supported(including_emulation=False)`; T4 chỉ hỗ trợ bf16 giả lập nên chọn **fp16**, autocast CUDA và GradScaler. Không gọi bf16 giả lập trên T4: SDPA có thể rơi về math kernel bộ nhớ bậc hai. Merge nạp base fp16, `merge_and_unload`, kiểm tra số tham số không đổi rồi lưu safetensors/tokenizer và chép generation config/license/README nếu có.

Image khai báo dependency `peft` và `bitsandbytes` cùng nhóm model; Colab hiện dùng hai package này không ghim version. Trước train/merge trên Colab, ô thiết lập gốc có lệnh `pip uninstall -y -q torchao`: Colab cài sẵn torchao 0.10 khiến PEFT từ chối khi nạp adapter để merge. Đây là bước sửa môi trường Colab, không cần gỡ nếu môi trường không có torchao. Phiên CPU này không cài/gỡ package; version PEFT/bitsandbytes và khả năng build sạch của image là **NOT MEASURED**, phải ghi môi trường thực tế khi phát hành.

Resume bằng cách chạy lại **đúng lệnh train**, cùng base, SFT, seed, hyper-parameter và `RUN_DIR/qlora_native`. Checkpoint định kỳ mỗi 25 optimizer step và khi kết thúc. Script đọc khóa `step` từ `trainer_state.json` trong thư mục adapter, nạp adapter trainable, tái tạo thứ tự shuffle từ seed rồi bỏ số micro-batch đã xong; scheduler được tiến lại đến step đó. **Optimizer moments, GradScaler và RNG checkpoint không được lưu/khôi phục**; AdamW bắt đầu lại nên resume không tương đương bit-for-bit với một lượt liên tục. Không sửa/xóa metadata để giả bắt đầu hoặc giả hoàn thành; file này chỉ do script chính thức quản lý.

Trước release, con người phải xác nhận log kết thúc train, nguồn checkpoint được merge và lưu SHA256 thật của artifact cuối. Trạng thái hoàn tất Colab, số mẫu train dùng thực tế, loss/score cuối, thời gian train/merge, peak GPU memory, hash adapter/merged, lặp lại training, build/offline GPU container và parity notebook/script trên model thật đều **NOT MEASURED** trong phiên Windows CPU này. Không có hash hay điểm số mô phỏng. Chỉ đóng gói checkpoint `planner_merged` đã xác minh và ASR đã xác minh: mặc định PhoWhisper-medium revision `55a7e3eb6c906de891f8f06a107754427dd3be79`, hoặc checkpoint medium/large fine-tune đã qua kiểm WER/score/latency; dữ liệu, cache và checkpoint trung gian không được commit.

## ASR large (ứng viên, chưa đo GPU)

Trên Colab từ gốc repo, sau ô 1-2, chạy `bash scripts/colab_asr_large.sh` (chi tiết tại `docs/ASR_LARGE.md` trong repo). Script ghim revision large qua HF API trước 26/10/2026, dùng batch 4, bf16 autocast và gradient checkpointing hiện có; mặc định batch 16 của `train_asr.py` không đổi. Lệnh train độc lập với các file trong image: `python -B -m training_code.train_asr --data-root "$TRAIN_KIT" --model-dir "$ASR_BASE" --output-dir "$RUN_DIR/asr_large_ft" --epochs 3 --batch 4`. Cần cấp checkpoint base đã ghim riêng. Chỉ thay ASR khi WER validation < 9.9% rõ và smoke40 >= 61.29, tổng tham số < 6B, mỗi mẫu <= 30 s trên máy chấm đã xác nhận. Đặt toàn bộ checkpoint thắng vào `<AdapterDir>/asr` để đóng gói; Docker không đổi tensor. WER, smoke40, latency và dung lượng artifact large thực tế: NOT MEASURED.
