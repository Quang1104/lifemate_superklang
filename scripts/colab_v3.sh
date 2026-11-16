#!/usr/bin/env bash
# SFT v3 on Colab (run from a notebook cell after notebook cells 1-2): train on ASR transcripts with two retrieved
# TRAIN examples in the system prompt (LIFEMATE_NATIVE_FEWSHOT_K=2 at train and inference time), then smoke40.
#   1. ASR transcripts of the 1236 TRAIN IDs (resumable)        -> lifemate_results/asr_train.jsonl
#   2. build SFT v3 + 1 epoch QLoRA from the base (resumable)   -> lifemate_results/qlora_v3
#   3. smoke40 at TIME_SCALE=1: W with K=2 (examples untrained), v3 with K=2; step-1 probe recorded only
#   4. public test of v3 only when it beats W's baseline smoke by >= 5 points (smoke40 noise ~±6.6)
source "$(dirname "$0")/colab_lib.sh"
FS=LIFEMATE_NATIVE_FEWSHOT_K=2
PROBE=LIFEMATE_ABSTAIN_PROBE=1  # records abstain_call_prob in the TRAIN report; output unchanged

env $OFF python -B -m scripts.asr_transcripts --data-root "$K" --asr-dir "$M/asr" --device cuda \
    --output "$R/asr_train.jsonl" || exit 1
env $OFF python -B -m scripts.build_sft --data-root "$K" --transcripts "$R/asr_train.jsonl" --fewshot-k 2 \
    --output outputs/colab/sft_v3.jsonl || exit 1
git status --short src/agent/data/fewshot_native.jsonl  # must print nothing: train and inference share this pool
python -B training_code/train_qlora.py --model-dir "$M/fallback_planner" --data outputs/colab/sft_v3.jsonl \
    --output-dir "$R/qlora_v3" --epochs 1 --save-every 25 --max-tokens 12000 2>&1 | grep --line-buffered -E 'step|resumed|rows|Error' || exit 1
merge "$R/qlora_v3" "$M/sft_v3" || exit 1

merge "$R/qlora154" "$M/sft154" || exit 1
echo "SMOKE w_fewshot2 $(smoke "$M/sft154" w_fewshot2 "$FS" "$PROBE" | tail -1)"
V3=$(smoke "$M/sft_v3" v3 "$FS" "$PROBE" | tail -1); echo "SMOKE v3 $V3"
BASE=$(python -c "import json;print(json.load(open('$R/smoke_ab/base/native.scored.json'))['local_proxy_0_to_100'])" 2>/dev/null)
echo "SMOKE base(W, overnight) ${BASE:-missing}"
if [ -n "$BASE" ] && python -c "import sys;sys.exit(0 if float('$V3')>=float('$BASE')+5 else 1)"; then
  public "$M/sft_v3" v3 "$FS"
fi
date; echo V3_DONE
