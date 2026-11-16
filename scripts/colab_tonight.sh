#!/usr/bin/env bash
# 15/11 night (after notebook cells 1-2): public v4 first (best smoke so far, 56.29), then v5 (scripts/colab_v5.sh).
source "$(dirname "$0")/colab_lib.sh"
FT=$M/asr_ft
[ -f "$FT/config.json" ] || { mkdir -p "$FT" && cp -r "$R/asr_ft/." "$FT/"; } || exit 1
merge "$R/qlora_v4" "$M/sft_v4" || exit 1
ASR_DIR="$FT" public "$M/sft_v4" v4_asrft LIFEMATE_NATIVE_FEWSHOT_K=2 LIFEMATE_CONTEXT_V4=1 LIFEMATE_SMALL_FIXES=1 || exit 1
bash scripts/colab_v5.sh
date; echo TONIGHT_DONE
