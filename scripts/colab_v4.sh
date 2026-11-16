#!/usr/bin/env bash
# Evening run (after notebook cells 1-2): planner v4 = v3 recipe + docs/experiments/CONTEXT_PLAN.md flags, on fine-tuned-ASR transcripts.
#   0. public v3 + fine-tuned ASR (resumable), and smoke40 of it with the no-retrain flags (dup guard, small fixes)
#   1. TRAIN transcripts with the fine-tuned ASR                  -> lifemate_results/asr_train_ft.jsonl
#   2. SFT v4 (K=2, LIFEMATE_CONTEXT_V4, small fixes) + 1 epoch    -> lifemate_results/qlora_v4
#   3. smoke40 v4 with all flags; public v4 only when it beats the best v3 smoke by >= 5 points
source "$(dirname "$0")/colab_lib.sh"
FS=LIFEMATE_NATIVE_FEWSHOT_K=2
FIX=(LIFEMATE_DUP_GUARD=1 LIFEMATE_SMALL_FIXES=1)
V4FLAGS=(LIFEMATE_CONTEXT_V4=1 LIFEMATE_SMALL_FIXES=1)  # must match between build_sft and inference
FT=$M/asr_ft
[ -f "$FT/config.json" ] || { mkdir -p "$FT" && cp -r "$R/asr_ft/." "$FT/"; } || exit 1
merge "$R/qlora_v3" "$M/sft_v3" || exit 1
ASR_DIR="$FT" public "$M/sft_v3" v3_asrft "$FS" || exit 1
echo "SMOKE v3_asrft_fix $(SOURCE="--asr-dir $FT" smoke "$M/sft_v3" v3_asrft_fix "$FS" "${FIX[@]}" | tail -1)"

env $OFF python -B -m scripts.asr_transcripts --data-root "$K" --asr-dir "$FT" --device cuda \
    --output "$R/asr_train_ft.jsonl" || exit 1
env $OFF "${V4FLAGS[@]}" python -B -m scripts.build_sft --data-root "$K" --transcripts "$R/asr_train_ft.jsonl" \
    --fewshot-k 2 --output outputs/colab/sft_v4.jsonl || exit 1
git status --short src/agent/data/fewshot_native.jsonl  # must print nothing: train and inference share this pool
# v4 prompts are ~14% longer than v3 (max 37,872 chars); 14000 keeps every row that fit v3 at 12000.
python -B training_code/train_qlora.py --model-dir "$M/fallback_planner" --data outputs/colab/sft_v4.jsonl \
    --output-dir "$R/qlora_v4" --epochs 1 --save-every 25 --max-tokens 14000 2>&1 | grep --line-buffered -E 'step|resumed|rows|Error' || exit 1
merge "$R/qlora_v4" "$M/sft_v4" || exit 1

score() { python -c "import json;print(json.load(open('$R/smoke_ab/$1/native.scored.json'))['local_proxy_0_to_100'])"; }
V4=$(SOURCE="--asr-dir $FT" smoke "$M/sft_v4" v4_asrft "$FS" "${V4FLAGS[@]}" LIFEMATE_DUP_GUARD=1 | tail -1) || exit 1
BEST=$(python -c "print(max($(score v3_asrft), $(score v3_asrft_fix)))") || exit 1
echo "SMOKE v4_asrft $V4 best_v3 $BEST"
if python -c "import sys;sys.exit(0 if $V4 >= $BEST + 5 else 1)"; then
  ASR_DIR="$FT" public "$M/sft_v4" v4_asrft "$FS" "${V4FLAGS[@]}" LIFEMATE_DUP_GUARD=1
fi
date; echo V4_DONE
