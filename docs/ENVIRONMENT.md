# Environment inventory — 2026-11-12

- Windows, PowerShell, Python3.12.6. NVIDIA GTX1650 4096MiB (driver610.91; nvidia-smi CUDA driver capability13.3, not a Python toolkit).
- RAM23.84GiB via psutil; disk free20.14GiB at inspection (varies). CIM RAM query denied; psutil succeeded.
- torch2.5.1+cpu, torch.cuda.is_available=False, transformers4.46.3; pytest, jsonschema, NumPy, psutil, huggingface_hub installed. faster_whisper/soundfile absent initially.
- No planner or ASR weights in default HF model cache. Existing cache folders were vision/embedding models, not used here. No broad filesystem/model search performed.
- Docker CLI installed; Linux Docker engine pipe missing. No image build or GPU inference tested.
- Canonical dataset root: `../datalife/train+public_test`, supplied/extracted by user. Earlier workspace train/public_test trees contained AppleDouble metadata. `data_kit` holds only exact permitted ZIP members recovered before user supplied full kit.
- No environment directory was traversed or file inspected/copied. Official ToolEnv accesses its own files internally. External kit is read-only; Python bytecode writes disabled in gold replay.

## Capabilities and delegation

- CLI/Python, patching, official web search/fetch, native subagent orchestration available. No installed specialized ML/PyTorch/evaluation skill found in the exposed catalog. Irrelevant media/site skills not used.
- PDF skill used read-only for Docker instructions; pypdf extraction + Poppler rendering and visual verification of conflicting pages.
- `gpt-6.1-sol` invoked as engineering subagent; a second bounded audit subagent. Shared checkout with disjoint file ownership, no recursive delegation. Worktree creation tool not exposed; no overlapping writes assigned.
- Claude CLI2.1.299 authenticated. Initial sandbox probe failed DNS; approved network probe succeeded with response metadata `canonicalModel=claude-opus-5-5`. Actual independent review saved in docs/research/OPUS_REVIEW.md.
- Sending the review summary was initially blocked by automatic approval review; user explicitly approved exact task document, then review succeeded. No dataset examples/audio/state sent to external model. No external APIs used by inference.
- No unrelated connected mail/calendar/Drive data accessed.
