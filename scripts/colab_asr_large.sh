#!/usr/bin/env bash
# A100 80GB preparation: run after notebook cells 1-2 from the repo root.
# No TRAIN transcripts are needed for smoke. Opt in after large wins with ASR_LARGE_TRANSCRIPTS=1.
set -euo pipefail
source "$(dirname "$0")/colab_lib.sh"
FT_L=$M/asr_large_ft
TRAIN_LOG=$R/asr_large_train.log
EVAL=$R/asr_large_eval.json
OUT=outputs/colab/smoke_ab/v4_asrlarge

case "${ASR_LARGE_TRANSCRIPTS:-0}" in 0|1) ;; *) echo 'ASR_LARGE_TRANSCRIPTS must be 0 or 1'; exit 1 ;; esac
# A failed/no-improvement run must never reuse an older checkpoint, score, or transcript.
for artifact in "$FT_L" "$R/asr_large_ft" "$TRAIN_LOG" "$EVAL" "$OUT" "$R/smoke_ab/v4_asrlarge" \
                "$R/asr_large_fetch.log" "$R/asr_large_parameters.json" "$R/asr_train_large_ft.jsonl"; do
  [ ! -e "$artifact" ] || { echo "ASR_LARGE output already exists: $artifact; move it before a fresh run"; exit 1; }
done
mkdir -p "$R"
# Online setup only. Resolve the placeholder once via the HF API and persist an eligible SHA/date.
python -B -m scripts.fetch_models --models-dir "$M" --roles asr_large --resolve-asr-large \
  2>&1 | tee "$R/asr_large_fetch.log"
merge "$R/qlora_v4" "$M/sft_v4"
env $OFF python -B -m scripts.check_asr_params --planner-config "$M/sft_v4/config.json" \
  --asr-config "$M/asr_large/config.json" | tee "$R/asr_large_parameters.json"

# train_asr already uses fp32 master weights, bf16 autocast, SDPA and non-reentrant gradient
# checkpointing. Batch 4 reduces activation memory for large; medium's default stays 16.
# A100 80GB feasibility is an estimate until measured; no optimizer/precision recipe change.
env $OFF python -B -m training_code.train_asr --data-root "$K" --model-dir "$M/asr_large" \
  --output-dir "$FT_L" --epochs 3 --batch 4 2>&1 | tee "$TRAIN_LOG"
[ -f "$FT_L/config.json" ] || { echo 'ASR_LARGE no epoch beat the base WER'; exit 1; }
cp configs/colab_models.json "$FT_L/base_model_pins.json"
mkdir -p "$R/asr_large_ft" && cp -r "$FT_L/." "$R/asr_large_ft/"

score=$(SOURCE="--asr-dir $FT_L" smoke "$M/sft_v4" v4_asrlarge \
  LIFEMATE_NATIVE_FEWSHOT_K=2 LIFEMATE_CONTEXT_V4=1 LIFEMATE_SMALL_FIXES=1 | tail -1)
echo "SMOKE v4_asrlarge $score"
env $OFF python -B -m scripts.asr_large_report --predictions "$OUT/native.jsonl" \
  --train-report "$OUT/native.report.json" --scored "$OUT/native.scored.json" --manifest "$SMOKE" \
  --training-log "$TRAIN_LOG" --output "$EVAL"
# The report includes official time p50/p95/max in ms, failed/omitted samples, and model load separately.
if [ "${ASR_LARGE_TRANSCRIPTS:-0}" = 1 ]; then
  if python -B -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["candidate_passes_numeric_gates"] else 1)' "$EVAL"; then
    env $OFF python -B -m scripts.asr_transcripts --data-root "$K" --asr-dir "$FT_L" --device cuda \
      --output "$R/asr_train_large_ft.jsonl"
  else
    echo 'TRAIN transcripts skipped: large did not pass the numeric candidate gates.'
  fi
else
  echo 'TRAIN transcripts skipped (unneeded for smoke). If large wins, run:'
  echo "env $OFF python -B -m scripts.asr_transcripts --data-root $K --asr-dir $FT_L --device cuda --output $R/asr_train_large_ft.jsonl"
fi
date
echo 'Acceptance: keep large only if validation WER is clearly below 9.9% and smoke40 >= 56.29 + 5 = 61.29.'
echo 'Also require total parameters < 6B and every sample <= 30 s on confirmed scoring hardware; A100 alone does not settle this.'
echo ASR_LARGE_DONE
