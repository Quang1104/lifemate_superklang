# Build Docker trên máy Windows: chiếm gì, ở đâu, dọn thế nào

Mọi dung lượng dưới đây là **ước tính** cho tới khi build thật lần đầu (khi đó ghi số đo vào HANDOFF). Hai số đo duy nhất: planner gốc 8,06 GB và ASR 3,06 GB (tải trên Colab).

## Quy trình

1. Kiểm tra chỗ trống: `powershell -NoProfile -File .\docker\release.ps1 -CheckOnly` (cần khoảng 40 GB trống trên ổ C).
2. Tải adapter QLoRA đã chọn (~130 MB, thư mục `qlora_native` trên Drive) vào `C:\Users\ADMIN\lifemate_release\qlora_native`.
3. Mở Docker Desktop, rồi chạy: `powershell -NoProfile -File .\docker\release.ps1 -AdapterDir C:\Users\ADMIN\lifemate_release\qlora_native`.
   Trong lúc build, Docker tự tải planner gốc + ASR đúng revision đã ghim, merge adapter (`training_code/stream_merge.py`, cùng kết quả từng bit với `merge_lora.py`, ít RAM), rồi xoá planner gốc ngay trong cùng bước.
4. Upload `C:\Users\ADMIN\lifemate_release\lifemate-2026-11-15.tar.gz` và file `.md5` lên Drive 5 TB, lấy link.
5. Dọn: chạy PowerShell **Run as Administrator**: `powershell -NoProfile -File .\docker\cleanup.ps1 -DeleteArchive -DeleteAdapter -CompactDisk`.

## Những gì được tạo trên máy

| Thứ | Nằm ở đâu | Ước tính | Ai xoá |
|---|---|---|---|
| Base `nvidia/cuda:13.2.0-runtime` (Docker tự pull) | ổ ảo Docker `%LOCALAPPDATA%\Docker\wsl\disk\docker_data.vhdx` | ~3–4 GB | `cleanup.ps1` (chỉ xoá image không có trước khi build) |
| Image `lifemate:2026-11-15` | như trên | ~21 GB | `cleanup.ps1` |
| Build cache | như trên | vài GB | `cleanup.ps1` (`docker builder prune --all`, xoá **toàn bộ** cache build, kể cả của project khác) |
| Planner gốc tải trong build | trong container build, xoá ngay sau merge | 8,06 GB tạm thời | tự xoá |
| Adapter | `C:\Users\ADMIN\lifemate_release\qlora_native` | ~0,13 GB | `cleanup.ps1 -DeleteAdapter` |
| File nộp + MD5 | `C:\Users\ADMIN\lifemate_release\` (ngoài OneDrive để không bị đồng bộ) | ~15–20 GB | `cleanup.ps1 -DeleteArchive`, **chỉ sau khi đã upload** |
| Trạng thái build (danh sách image có sẵn) | `C:\Users\ADMIN\lifemate_release\release-state.json` | nhỏ | `cleanup.ps1` khi xoá cả archive lẫn adapter |

Docker Desktop **không tự trả dung lượng cho Windows** khi xoá image: file `docker_data.vhdx` vẫn giữ kích thước cũ. `-CompactDisk` tắt Docker Desktop + WSL rồi nén file này bằng `diskpart` (cần quyền Administrator). Trước build, file này đã nặng 12,8 GB do dữ liệu Docker cũ; cleanup không xoá image có từ trước.

## Số đo build thử 14/11 (adapter bước 50)

- Mọi bước chạy được: base CUDA runtime, torch cu132, dependencies, tải base + ASR (~10 phút), `merged 252 LoRA modules` (bước model mất ~30 phút tổng). Thất bại duy nhất: hết đĩa khi Docker "unpacking" image.
- Ổ C: 55,1 GB trống → 3,6 GB khi huỷ (dùng ~51,5 GB). `docker_data.vhdx`: 12,8 GB → 62,3 GB. Docker báo image `lifemate:trial-step50` 14,87 GB, build cache 14,89 GB.
- Nguyên nhân (suy luận, chưa kiểm chứng): Docker Desktop đang dùng **containerd image store**, giữ cả blob nén lẫn bản giải nén của mỗi layer, cộng build cache. Ước tính 40 GB trước đó là sai.
- Đề xuất trước lần build thật: Docker Desktop → Settings → General → tắt "Use containerd for pulling and storing images" (lưu kiểu cũ, không nhân đôi layer). Dung lượng cần khi đó: NOT MEASURED.
- Đã dọn bên trong Docker (image thử + build cache, ~29,8 GB) nhưng Windows chỉ lấy lại chỗ sau khi nén `docker_data.vhdx`: PowerShell **Run as Administrator** → `powershell -NoProfile -ExecutionPolicy Bypass -File "<repo>\docker\cleanup.ps1" -CompactDisk`.

## Thay đổi cấu hình đã làm

- `C:\Users\ADMIN\.wslconfig`: `memory=5GB` → `memory=10GB` (theo yêu cầu, máy đã có 23,8 GB RAM). Bản cũ: `C:\Users\ADMIN\.wslconfig.bak-2026-11-14`. Muốn trả lại: chép đè bản `.bak`, rồi `wsl --shutdown`.
