# LifeMate — Zalo AI Challenge 2026 (team Locked In)

Offline Vietnamese voice agent: audio → fine-tuned PhoWhisper-medium (ASR) → Qwen3-4B-Instruct-2507 + QLoRA planner
(native tool calling, 2 retrieved TRAIN examples) → official ToolEnv. Total parameters < 6B, ≤ 30 s per sample.

## Results (public test, portal)

| Version | Planner data / prompt | Public |
|---|---|---:|
| v3 + ASR fine-tune | 1236 TRAIN samples, K=2 examples | 42.25 |
| v7 | + TRAIN history turns as extra SFT rows | 43 |
| **v8b (final)** | v7 + date/time hints + 269 validation samples | **46** |

## Layout

- `src/` — inference pipeline (ASR, planner, ToolEnv runner); `python -B -m src.inference --help`.
- `training_code/` — QLoRA training, merge, ASR fine-tune; see [training_code/README.md](training_code/README.md).
- `scripts/` — SFT builder (`build_sft.py`), scoring, Colab run scripts (`colab_v*.sh`).
- `docker/` — image, `predict.sh`, notebook, `release.ps1` / `cleanup.ps1`.
- `docs/` — [competition rules](docs/COMPETITION_RULES.md), [submission checklist](docs/SUBMISSION_CHECKLIST.md),
  [Colab guide](docs/LIFEMATE_COLAB.md); experiment notes in `docs/experiments/`.
- `research/` — splits (smoke40, VAL309) and label-free analysis tools.
- [Colab notebook](lifemate_colab.ipynb) for training and evaluation runs.

## Final release

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\docker\release.ps1 -AdapterDir C:\lifemate_release\qlora_v8b `
  -Flags "LIFEMATE_CONTEXT_V4=1,LIFEMATE_SMALL_FIXES=1,LIFEMATE_TIME_HINTS_V8=1" `
  -ImageTag "zac2026-lockedin:v1" -OutputArchive "C:\lifemate_release\zac2026_LockedIn.tar.gz"
```

`<AdapterDir>` holds the v8b QLoRA adapter and `asr\` the fine-tuned PhoWhisper-medium.

## Tests

```bash
python -B -m pytest -q -p no:cacheprovider
python -B -m src.inference mock --samples 40
```
