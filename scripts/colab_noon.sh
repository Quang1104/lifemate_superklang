#!/usr/bin/env bash
# 15/11 afternoon, unattended (run from a notebook cell after notebook cells 1-2):
#   1. public test with v3 (K=2 examples), resumable via colab_lib public()
#   2. ASR fine-tune + smoke40 of W and v3 with it (scripts/colab_asr.sh)
#   3. public v3 with the fine-tuned ASR, only when its smoke beats v3 (old ASR) by >= 5 points
source "$(dirname "$0")/colab_lib.sh"
merge "$R/qlora_v3" "$M/sft_v3" || exit 1
public "$M/sft_v3" v3 LIFEMATE_NATIVE_FEWSHOT_K=2 || exit 1
bash scripts/colab_asr.sh || exit 1
score() { python -c "import json;print(json.load(open('$R/smoke_ab/$1/native.scored.json'))['local_proxy_0_to_100'])"; }
V3=$(score v3) && FT=$(score v3_asrft) || exit 1
echo "SMOKE v3 $V3 v3_asrft $FT"
if python -c "import sys;sys.exit(0 if $FT >= $V3 + 5 else 1)"; then
  mv "$M/asr" "$M/asr_base" && ln -s "$M/asr_ft" "$M/asr" || exit 1  # public() always reads $M/asr
  public "$M/sft_v3" v3_asrft LIFEMATE_NATIVE_FEWSHOT_K=2
fi
date; echo NOON_DONE
