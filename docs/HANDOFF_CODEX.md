# Handoff to Codex (planner / ToolEnv / inference core / Docker owner) — 2026-11-12

Claude added (Claude-owned from now on): `src/bench/*`, `tests_bench/`, `notebooks/`, `configs/colab.json`, `requirements-colab.txt`.
Claude also created the missing `src/agent/local_model.py` that `src/inference/__main__.py` already imports
(`LocalPlanner`, `LocalWhisper`, `initialize_runtime`; HF Transformers, lazy imports). Ollama backend removed — final Docker must not depend on it.
Ownership of `local_model.py` passes to Codex; keep `build_messages` shared, because SFT data (`src/bench/sft_data.py`) uses the same prompt.

Requested changes in Codex-owned code (none made by Claude):
1. **Rejection loop (highest impact seen):** local qwen3:4b smoke had 2/3 samples end `step_limit` with 8/8 guard rejections.
   Rejected `Observation`s carry only `{"reason"}`, not the attempted action, so the model cannot correct itself. Add the attempted
   tool+arguments to the rejected observation and stop after ~3 consecutive rejections.
2. `pipeline.predict_records` raises on any unsafe/failed sample and aborts the whole split. For submission, one bad sample should
   still export that ToolEnv's calls/hash (or empty) and continue.
3. Planner timeouts leave the generate thread running on GPU; next sample then contends for the GPU. Prefer a token budget or
   `max_time` in `generate` over thread abandonment.
4. Add `src.bench*` to `pyproject.toml` packages if the package is installed anywhere.
5. Docker: same `local_model.py` backends, offline weights in image, `transformers==4.56.2` (Qwen3 needs ≥4.51; local 4.46.3 cannot load it).
