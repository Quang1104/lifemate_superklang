#!/usr/bin/env bash
# Planner v6 (after notebook cells 1-2): the best earlier recipe (BASE) + meaningful tool renames on 30% of training
# samples + no retrieved examples when none is similar (LIFEMATE_FEWSHOT_GATE). Both target tools absent from TRAIN
# (docs/experiments/PUBLIC_GAP.md, docs/experiments/V6_PLAN.md), which smoke40 cannot measure: public runs whenever smoke v6 is not worse
# than the best earlier smoke, and public decides.
#   1. TRAIN transcripts with the fine-tuned ASR (reused when present, resumable)
#   2. SFT v6 + 1 epoch QLoRA from the base          -> lifemate_results/qlora_v6
#   3. smoke40 v6; public v6 when smoke v6 >= best earlier smoke - 3 (smoke40 has no unseen tools; public decides)
# Usage: bash scripts/colab_v6.sh   (BASE=v4 by default, root 15/11 23:20: v5 smoke 50.19 < v4 56.29; BASE=v5 adds
# the date table and 30% fn_ masking)
source "$(dirname "$0")/colab_lib.sh"
set -o pipefail
BASE=${BASE:-v4}
FS=LIFEMATE_NATIVE_FEWSHOT_K=2
case "$BASE" in
  v4) V6FLAGS=(LIFEMATE_CONTEXT_V4=1 LIFEMATE_SMALL_FIXES=1); MASK=0.0 ;;
  v5) V6FLAGS=(LIFEMATE_CONTEXT_V4=1 LIFEMATE_SMALL_FIXES=1 LIFEMATE_DATE_TABLE=1); MASK=0.3 ;;
  *) echo "BASE must be v4 or v5"; exit 1 ;;
esac
V6FLAGS+=(LIFEMATE_FEWSHOT_GATE=1)  # same flags at build, smoke and public: the prompt must match training
echo "V6 BASE=$BASE flags: ${V6FLAGS[*]} mask=$MASK rename=0.3"
# v6 output names are shared by both BASE recipes; never mix checkpoints or resume logs.
for artifact in outputs/colab/sft_v6.jsonl "$R/qlora_v6" "$M/sft_v6" \
                outputs/colab/smoke_ab/v6_asrft "$R/smoke_ab/v6_asrft" \
                /content/public_runs/v6_asrft "$R/public/v6_asrft" "$R/public/v6_asrft.progress.jsonl"; do
  [ ! -e "$artifact" ] || { echo "V6 output already exists: $artifact; move it before a fresh run" >&2; exit 1; }
done
FT=$M/asr_ft
[ -f "$FT/config.json" ] || { mkdir -p "$FT" && cp -r "$R/asr_ft/." "$FT/"; } || exit 1
env $OFF python -B -m scripts.asr_transcripts --data-root "$K" --asr-dir "$FT" --device cuda \
    --output "$R/asr_train_ft.jsonl" || exit 1
env $OFF "${V6FLAGS[@]}" python -B -m scripts.build_sft --data-root "$K" --transcripts "$R/asr_train_ft.jsonl" \
    --fewshot-k 2 --mask-fraction "$MASK" --rename-fraction 0.3 --output outputs/colab/sft_v6.jsonl || exit 1
git status --short src/agent/data/fewshot_native.jsonl  # must print nothing
python -B training_code/train_qlora.py --model-dir "$M/fallback_planner" --data outputs/colab/sft_v6.jsonl \
    --output-dir "$R/qlora_v6" --epochs 1 --save-every 25 --max-tokens 14000 2>&1 | grep --line-buffered -E 'step|resumed|rows|Error' || exit 1
merge "$R/qlora_v6" "$M/sft_v6" || exit 1

score() { python -c "import json;print(json.load(open('$R/smoke_ab/$1/native.scored.json'))['local_proxy_0_to_100'])" 2>/dev/null || echo 0; }
V6=$(SOURCE="--asr-dir $FT" smoke "$M/sft_v6" v6_asrft "$FS" "${V6FLAGS[@]}" | tail -1) || exit 1
BEST=$(python -c "print(max($(score v3_asrft), $(score v3_asrft_fix), $(score v4_asrft), $(score v5_asrft)))") || exit 1
echo "SMOKE v6_asrft $V6 best_before $BEST"
if python -c "import sys;sys.exit(0 if $V6 >= $BEST - 3 else 1)"; then
  ASR_DIR="$FT" public "$M/sft_v6" v6_asrft "$FS" "${V6FLAGS[@]}" || exit 1
fi
date; echo V6_DONE
