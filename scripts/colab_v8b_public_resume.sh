#!/usr/bin/env bash
# Continue public v8b if the runtime was lost (16/11). Run after notebook cells 1-2 (A100 or L4).
# The adapter (398/398) and the per-sample progress log live on Drive; public() resumes from that log.
set -o pipefail
source "$(dirname "$0")/colab_lib.sh"
grep -q '"step": 398' "$R/qlora_v8b/trainer_state.json" || { echo "qlora_v8b is not at step 398" >&2; exit 1; }
[ -f "$M/asr_ft/config.json" ] || { mkdir -p "$M/asr_ft" && cp -r "$R/asr_ft/." "$M/asr_ft/"; } || exit 1
merge "$R/qlora_v8b" "$M/sft_v8b" || exit 1
wc -l "$R/public/v8b_asrft.progress.jsonl" 2>/dev/null
ASR_DIR="$M/asr_ft" public "$M/sft_v8b" v8b_asrft LIFEMATE_NATIVE_FEWSHOT_K=2 \
    LIFEMATE_CONTEXT_V4=1 LIFEMATE_SMALL_FIXES=1 LIFEMATE_FEWSHOT_GATE=0 LIFEMATE_TIME_HINTS_V8=1 || exit 1
date; echo V8B_PUBLIC_DONE
