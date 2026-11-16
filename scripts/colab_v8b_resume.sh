#!/usr/bin/env bash
# Finish v8b after the A100 runtime was lost at step 325/398 (16/11 07:41). Run after notebook cells 1-2.
# v8b = colab_v8.sh with HIST=0.5 V6=0 TIME_HINTS=1 FULL=1 and the ASR transcripts that include the 269 validation IDs.
# The SFT file is rebuilt with the same commit, flags and transcripts (deterministic), then train_qlora resumes the
# adapter on Drive (LR schedule restored, optimizer moments restart), merge, smoke40 (sanity only) and public.
set -o pipefail
source "$(dirname "$0")/colab_lib.sh"
FS=LIFEMATE_NATIVE_FEWSHOT_K=2
FLAGS=(LIFEMATE_CONTEXT_V4=1 LIFEMATE_SMALL_FIXES=1 LIFEMATE_FEWSHOT_GATE=0 LIFEMATE_TIME_HINTS_V8=1)
FT=$M/asr_ft
TRANSCRIPTS=$R/asr_train_ft_full.jsonl
[ -f "$R/qlora_v8b/trainer_state.json" ] || { echo "Missing $R/qlora_v8b checkpoint" >&2; exit 1; }
[ -s "$TRANSCRIPTS" ] || { echo "Missing $TRANSCRIPTS" >&2; exit 1; }
[ -f "$FT/config.json" ] || { mkdir -p "$FT" && cp -r "$R/asr_ft/." "$FT/"; } || exit 1
cat "$R/qlora_v8b/trainer_state.json"; echo
env $OFF "$FS" "${FLAGS[@]}" python -B -m scripts.build_sft --data-root "$K" --transcripts "$TRANSCRIPTS" \
    --exclude research/smoke_split.json --fewshot-k 2 --history-turns 0.5 --output outputs/colab/sft_v8b.jsonl || exit 1
python -B training_code/train_qlora.py --model-dir "$M/fallback_planner" --data outputs/colab/sft_v8b.jsonl \
    --output-dir "$R/qlora_v8b" --epochs 1 --save-every 25 --max-tokens 14000 2>&1 | grep --line-buffered -E 'step|resumed|rows|Error' || exit 1
grep -q '"step": 398' "$R/qlora_v8b/trainer_state.json" || { echo "v8b did not reach step 398" >&2; exit 1; }
merge "$R/qlora_v8b" "$M/sft_v8b" || exit 1
echo "SMOKE v8b_asrft $(SOURCE="--asr-dir $FT" smoke "$M/sft_v8b" v8b_asrft "$FS" "${FLAGS[@]}" | tail -1)" || exit 1
ASR_DIR="$FT" public "$M/sft_v8b" v8b_asrft "$FS" "${FLAGS[@]}" || exit 1
echo 'Release: powershell.exe -NoProfile -File .\docker\release.ps1 -AdapterDir <qlora_v8b + asr\> -Flags "LIFEMATE_CONTEXT_V4=1,LIFEMATE_SMALL_FIXES=1,LIFEMATE_TIME_HINTS_V8=1"'
date; echo V8B_DONE
