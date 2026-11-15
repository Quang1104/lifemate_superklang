#!/usr/bin/env bash
# Fine-tune the ASR on TRAIN audio (run from a notebook cell after notebook cells 1-2; bf16 GPU), then smoke40
# at TIME_SCALE=1 with the fine-tuned ASR for W and, when its adapter exists, v3 (K=2 examples).
#   -> lifemate_results/asr_ft (fp16 weights, best validation WER), lifemate_results/smoke_ab/{w_asrft,v3_asrft}
set -euo pipefail
source "$(dirname "$0")/colab_lib.sh"
FT=$M/asr_ft

# Refuse stale artifacts: a no-improvement run must not evaluate an older checkpoint.
[ ! -e "$FT" ] && [ ! -e "$R/asr_ft" ] || { echo "ASR_FT output already exists; move it before a fresh run"; exit 1; }
env $OFF python -B -m training_code.train_asr --data-root "$K" --model-dir "$M/asr" --output-dir "$FT" --epochs 3 || exit 1
[ -f "$FT/config.json" ] || { echo "ASR_FT no epoch beat the base WER"; exit 1; }
mkdir -p "$R/asr_ft" && cp -r "$FT/." "$R/asr_ft/"

merge "$R/qlora154" "$M/sft154" || exit 1
score=$(SOURCE="--asr-dir $FT" smoke "$M/sft154" w_asrft | tail -1) || exit 1
echo "SMOKE w_asrft $score"
if [ -f "$R/qlora_v3/adapter_model.safetensors" ]; then
  merge "$R/qlora_v3" "$M/sft_v3" || exit 1
  score=$(SOURCE="--asr-dir $FT" smoke "$M/sft_v3" v3_asrft LIFEMATE_NATIVE_FEWSHOT_K=2 | tail -1) || exit 1
  echo "SMOKE v3_asrft $score"
fi
date; echo ASR_DONE
