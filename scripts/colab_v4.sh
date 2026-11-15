#!/usr/bin/env bash
# Evening run (after notebook cells 1-2): planner v4 = v3 recipe on transcripts from the fine-tuned ASR.
#   0. public v3 + fine-tuned ASR (resumable; skipped samples already in the Drive progress log)
#   1. TRAIN transcripts with the fine-tuned ASR        -> lifemate_results/asr_train_ft.jsonl
#   2. SFT v4 (K=2 retrieved examples) + 1 epoch QLoRA  -> lifemate_results/qlora_v4
#   3. smoke40 v4 with the fine-tuned ASR; public v4 only when it beats v3_asrft by >= 5 points
source "$(dirname "$0")/colab_lib.sh"
FS=LIFEMATE_NATIVE_FEWSHOT_K=2
FT=$M/asr_ft
[ -f "$FT/config.json" ] || { mkdir -p "$FT" && cp -r "$R/asr_ft/." "$FT/"; } || exit 1
merge "$R/qlora_v3" "$M/sft_v3" || exit 1
ASR_DIR="$FT" public "$M/sft_v3" v3_asrft "$FS" || exit 1

env $OFF python -B -m scripts.asr_transcripts --data-root "$K" --asr-dir "$FT" --device cuda \
    --output "$R/asr_train_ft.jsonl" || exit 1
env $OFF python -B -m scripts.build_sft --data-root "$K" --transcripts "$R/asr_train_ft.jsonl" --fewshot-k 2 \
    --output outputs/colab/sft_v4.jsonl || exit 1
python -B training_code/train_qlora.py --model-dir "$M/fallback_planner" --data outputs/colab/sft_v4.jsonl \
    --output-dir "$R/qlora_v4" --epochs 1 --save-every 25 --max-tokens 12000 2>&1 | grep --line-buffered -E 'step|resumed|rows|Error' || exit 1
merge "$R/qlora_v4" "$M/sft_v4" || exit 1

score() { python -c "import json;print(json.load(open('$R/smoke_ab/$1/native.scored.json'))['local_proxy_0_to_100'])"; }
V4=$(SOURCE="--asr-dir $FT" smoke "$M/sft_v4" v4_asrft "$FS" | tail -1) || exit 1
V3=$(score v3_asrft) || exit 1
echo "SMOKE v4_asrft $V4 v3_asrft $V3"
if python -c "import sys;sys.exit(0 if $V4 >= $V3 + 5 else 1)"; then
  ASR_DIR="$FT" public "$M/sft_v4" v4_asrft "$FS"
fi
date; echo V4_DONE
