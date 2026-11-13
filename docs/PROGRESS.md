# PROGRESS (read first; keep short)

## State — 2026-11-12
- Astra scaffold committed unchanged (`f725f72`): runner, ToolEnv adapter, scorer, audits, docs. Gold TRAIN replay works (`scripts/replay_train_gold.py`).
- **Competition score: NOT MEASURED.** Only references: no-op control 0.125 on 40 smoke (gold-empty samples); oracle replay = 1.0 (bench sanity).
- **Policy: no GPU work on the Windows box.** GPU inference/benchmark/fine-tuning run on Google Colab only; Windows = code, git, CPU tests. Final = offline Docker.

## Local qwen3:4b (Ollama, GTX1650) — stopped, partial, NOT a score
3 smoke samples before stop: 1 finished, 2 `step_limit` with 8/8 guard rejections; ~4.3–4.5k prompt tokens (all installed schemas); 140–200 s/sample on that GPU.
→ Rejection loop is the first planner issue to fix (see docs/HANDOFF_CODEX.md #1).

## Colab pipeline (created, CPU-tested, **NOT yet run on Colab**)
- Guide: docs/COLAB.md. Notebooks 01–04 are thin entrypoints over `src/bench/*` + `src/agent/local_model.py`.
- Models (`configs/colab.json`): planner `Qwen/Qwen3-4B` @1cfa9a72 (Apache-2.0, 4.02B; HF counterpart of Ollama `qwen3:4b`, digest not cross-checked),
  alt `Qwen/Qwen3-4B-Instruct-2507` (revision/licence recorded at download via PINNED.json — verify before release), ASR `openai/whisper-small` @973afd24. Total 4.26B < 6B.
- Tests: `LIFEMATE_DATA=<kit> python -m pytest tests_audit tests_bench -q` → 29 passed (includes oracle planner through real runner+ToolEnv = 1.0, SFT build on 3 train IDs).

## Next (in order)
1. User: zip kit to Drive, run notebook 01 then 02 (5 → 40 gold transcript → 40 ASR). Record summary.json numbers here.
2. Fix top error class from 02 (expected: rejection loop / tool selection). Only then 03 fine-tuning (train_ids only).
