#!/usr/bin/env bash
# Public test of v3 (K=2 examples) with the fine-tuned ASR from Drive (lifemate_results/asr_ft); resumable.
# Run from a notebook cell after notebook cells 1-2.
source "$(dirname "$0")/colab_lib.sh"
[ -f "$M/asr_ft/config.json" ] || { mkdir -p "$M/asr_ft" && cp -r "$R/asr_ft/." "$M/asr_ft/"; } || exit 1
merge "$R/qlora_v3" "$M/sft_v3" || exit 1
ASR_DIR="$M/asr_ft" public "$M/sft_v3" v3_asrft LIFEMATE_NATIVE_FEWSHOT_K=2 || exit 1
date; echo PUBLIC_ASRFT_DONE
