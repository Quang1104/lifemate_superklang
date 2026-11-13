# Run on Google Colab (5 steps)

1. **Zip the kit once (Windows PowerShell)** — opaque archive, nothing inside `*_environment` is opened:
   ```powershell
   tar.exe -a -c -f "$HOME\train+public_test.zip" -C "C:\Users\ADMIN\OneDrive - VNU-HCMUS\Documents\XuanQuang\datalife" "train+public_test"
   ```
   Upload it to Google Drive as `MyDrive/lifemate/train+public_test.zip` (~1.1 GB). Never commit it.
2. **Private repo only:** Colab left bar → 🔑 Secrets → add `GITHUB_TOKEN` (read-only fine-grained token), enable notebook access.
3. Open `notebooks/01_environment_setup.ipynb` from GitHub in Colab (File → Open notebook → GitHub, branch `claude/zalo-lifemate-handoff-5c2f7d`). Runtime → Change runtime type → **GPU**. Run all.
4. Run `02_baseline_inference.ipynb`: 5-sample smoke → 40 gold-transcript → 40 ASR. Results land in `MyDrive/lifemate/results/<run>/summary.json`.
5. Fine-tune only after step 4 (`03_finetuning.ipynb`, set `RUN_TRAINING = True`), compare in `04_evaluation.ipynb`.

Paths/models: `configs/colab.json`. Pinned libs: `requirements-colab.txt`. Models download to runtime disk each session (not Drive, to save quota); `PINNED.json` records the resolved commit.
Each notebook is a thin entrypoint over `src/bench/*` and `src/agent/local_model.py`; put logic there, not in cells.
