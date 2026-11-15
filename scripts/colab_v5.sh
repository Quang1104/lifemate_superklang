#!/usr/bin/env bash
# Planner v5 (after notebook cells 1-2): v4 recipe + LIFEMATE_DATE_TABLE + function masking (Hammer) on 30% of
# training samples. Anti-overfitting: 1 epoch, same lr/rank as v3/v4, masking replaces samples (no duplicates),
# and v5 is only kept when smoke40 beats the best earlier smoke by >= 5 points; public then confirms it.
#   1. TRAIN transcripts with the fine-tuned ASR (reused from v4 when present, resumable)
#   2. SFT v5 + 1 epoch QLoRA from the base          -> lifemate_results/qlora_v5
#   3. smoke40 v5; public v5 only when it wins
source "$(dirname "$0")/colab_lib.sh"
FS=LIFEMATE_NATIVE_FEWSHOT_K=2
V5FLAGS=(LIFEMATE_CONTEXT_V4=1 LIFEMATE_SMALL_FIXES=1 LIFEMATE_DATE_TABLE=1)  # same flags at build and inference
FT=$M/asr_ft
[ -f "$FT/config.json" ] || { mkdir -p "$FT" && cp -r "$R/asr_ft/." "$FT/"; } || exit 1
env $OFF python -B -m scripts.asr_transcripts --data-root "$K" --asr-dir "$FT" --device cuda \
    --output "$R/asr_train_ft.jsonl" || exit 1
env $OFF "${V5FLAGS[@]}" python -B -m scripts.build_sft --data-root "$K" --transcripts "$R/asr_train_ft.jsonl" \
    --fewshot-k 2 --mask-fraction 0.3 --output outputs/colab/sft_v5.jsonl || exit 1
git status --short src/agent/data/fewshot_native.jsonl  # must print nothing
python -B training_code/train_qlora.py --model-dir "$M/fallback_planner" --data outputs/colab/sft_v5.jsonl \
    --output-dir "$R/qlora_v5" --epochs 1 --save-every 25 --max-tokens 14000 2>&1 | grep --line-buffered -E 'step|resumed|rows|Error' || exit 1
merge "$R/qlora_v5" "$M/sft_v5" || exit 1

score() { python -c "import json;print(json.load(open('$R/smoke_ab/$1/native.scored.json'))['local_proxy_0_to_100'])" 2>/dev/null || echo 0; }
V5=$(SOURCE="--asr-dir $FT" smoke "$M/sft_v5" v5_asrft "$FS" "${V5FLAGS[@]}" | tail -1) || exit 1
BEST=$(python -c "print(max($(score v3_asrft), $(score v3_asrft_fix), $(score v4_asrft)))") || exit 1
echo "SMOKE v5_asrft $V5 best_before $BEST"
if python -c "import sys;sys.exit(0 if $V5 >= $BEST + 5 else 1)"; then
  ASR_DIR="$FT" public "$M/sft_v5" v5_asrft "$FS" "${V5FLAGS[@]}"
fi
date; echo V5_DONE
