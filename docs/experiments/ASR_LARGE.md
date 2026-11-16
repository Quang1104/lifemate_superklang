# PhoWhisper-large candidate (no GPU measurements yet)

This patch is uncommitted. Apply it to the Colab checkout after cell 1 has fetched/reset the
branch (or after the existing cells 1-2); otherwise that reset discards local patched code.
Then run from the Colab repo after notebook cells 1-2 on the A100 80GB:

~~~bash
bash scripts/colab_asr_large.sh
~~~

The script downloads large to /content/lifemate_models/asr_large, fine-tunes to asr_large_ft,
and copies the winning epoch to /content/drive/MyDrive/lifemate_results/asr_large_ft.
It merges /content/drive/MyDrive/lifemate_results/qlora_v4 into /content/lifemate_models/sft_v4
only when the latter is missing, and uses v4 with K=2, CONTEXT_V4=1, SMALL_FIXES=1, TIME_SCALE=1.
It refuses existing checkpoints, run output, logs and transcript output; move those artifacts
aside before a fresh run. It does not resume training or reuse older smoke results.

## Eligibility and pin

The cached official rule in docs/sources/rule.json says: "teams may use public data and
open-source models released before 26 October 2026, provided their licences allow it."
This agrees with docs/COMPETITION_RULES.md and the date in NEXT_PLAN item 3.

The HF API was unreachable in the preparation session, so configs/colab_models.json explicitly
contains REVISION_NOT_RESOLVED for asr_large. No API SHA/date has been fabricated. The online
setup uses scripts.fetch_models --resolve-asr-large to select the latest eligible commit from
HfApi.list_repo_commits, saves its real full SHA to the config, and verifies its UTC date is
strictly before 2026-10-26 before downloading. An already resolved pin is never moved by this
option. The chosen SHA/date is printed in asr_large_fetch.log and the pin file is copied into
the fine-tuned checkpoint as base_model_pins.json. Retain these artifacts with the final release.
Unresolved, shortened or mutable pins fail closed in the fetchers; Docker does not auto-resolve.

Notebook cell 2 has optional FETCH_ASR_LARGE = True to fetch large alongside medium during setup.
The large run script also fetches/resolves large itself, so the default notebook need not change.
Standalone online setup:

~~~bash
python -B -m scripts.fetch_models --models-dir /content/lifemate_models --roles asr_large --resolve-asr-large
~~~

## Memory, parameter budget and measurements

train_asr.py already uses fp32 master parameters/AdamW, bf16 autocast, SDPA and non-reentrant
gradient checkpointing. Large uses the existing --batch 4 argument instead of the medium default
16. No training defaults were changed. Full fp32 parameters, gradients and Adam moments alone
are approximately 24.7 GB before activations, workspaces and temporary buffers (estimate).
Feasibility at batch 4 on A100 80GB, peak GPU memory, training duration and WER are NOT MEASURED.
Reducing batch changes optimizer updates per epoch; no learning-rate or accumulation recipe was
silently substituted. Epoch 0 and each of three epochs print validation WER; only a checkpoint
that improves on the initial model is saved by train_asr.py.

The dimension-only calculation is planner 4,022,468,096 + large 1,543,304,960 = 5,565,773,056,
434,226,944 below the strict 6B limit; planner is also strictly below 5B. It counts frozen encoder
positions and tied embeddings once. Dimension sources: [pinned Qwen3 config](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/config.json)
and [published large config](https://huggingface.co/vinai/PhoWhisper-large/blob/main/config.json).
This is a config-derived count, not a loaded-weight measurement; the unresolved large API pin
must be checked against the actual downloaded config. The run script checks the actual v4 and
large configs before training, without loading weights for the count:

~~~bash
python -B -m scripts.check_asr_params --planner-config /content/lifemate_models/sft_v4/config.json --asr-config /content/lifemate_models/asr_large/config.json
python -B -m unittest discover -s tests -p test_asr_large.py -v
~~~

When torch/transformers are installed, the same test includes CPU/meta-model parity for the
count formulas, with no pretrained weights or GPU allocation. The real inference report also
records loaded parameter counts; confirm these agree with the preflight count.

## Smoke, acceptance and transcripts

asr_large_eval.json reports nearest-rank p50/p95/max from the official output time field in
milliseconds. It also reports all TRAIN sample times, including samples omitted from the
official output, and model load separately. Missing rows, missing/invalid timing, non-unit time
scale or any sample above 30,000 ms cannot pass the numeric gates. The original logs are only read.

Keep large only if validation WER is clearly below 9.9% and smoke40 >= 56.29 + 5 = 61.29.
A numeric WER comparison cannot decide what constitutes a clear improvement. Require every
sample <= 30 s on confirmed scoring hardware as well. Large WER, smoke40, latency and public
score are NOT MEASURED. The A30 versus RTX5090 and CUDA <=13.2 versus mandatory 13.2 conflicts
in docs/COMPETITION_RULES.md remain unresolved; an A100 run does not settle target latency.

TRAIN transcripts are unnecessary for smoke and are skipped by default. Once large wins:

~~~bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -B -m scripts.asr_transcripts --data-root /content/train+public_test --asr-dir /content/lifemate_models/asr_large_ft --device cuda --output /content/drive/MyDrive/lifemate_results/asr_train_large_ft.jsonl
~~~

Alternatively ASR_LARGE_TRANSCRIPTS=1 bash scripts/colab_asr_large.sh generates them after smoke
only when the numeric gates pass. Judge the clear WER margin before opting in. Existing transcript
output is refused for a fresh training run so it cannot mix checkpoints.

## Docker

Place the full asr_large_ft contents in <AdapterDir>/asr. release.ps1 -AdapterDir copies that
folder through docker/fetch_models.py unchanged into /models/asr, including tokenizer/config and
safetensors shards. Fine-tuned ASR takes precedence over either pinned fallback. There is no
medium dimension assumption in the loader or Docker copy path. For a base large fallback only,
first resolve the config pin, then use release.ps1 -AsrRole asr_large (Docker ARG ASR_ROLE).

For the fine-tuned v4 candidate on Windows:

~~~powershell
powershell.exe -NoProfile -File .\docker\release.ps1 -AdapterDir C:\lifemate_release\qlora_v4_large -Flags "LIFEMATE_CONTEXT_V4=1,LIFEMATE_SMALL_FIXES=1"
~~~

Only copy large into the adapter after it wins. Estimated fp16 tensor payload is ~3.09 GB
(~2.87 GiB), excluding small config/tokenizer/serialization overhead; actual checkpoint/image
size and offline GPU Docker parity are NOT MEASURED. release.ps1 currently hashes a single
model.safetensors for its manifest; the large payload is estimated below 5GB, but its actual saved
file layout is NOT MEASURED. A sharded checkpoint would need a shard-aware manifest improvement
(copy/loading already work).
The unchanged generic inference settings are processor-defined chunks (normally 30 seconds),
long-audio timestamps, greedy beam=1, Vietnamese transcription and a 256-token output budget.
These need latency measurement, not a model-size-specific replacement.
