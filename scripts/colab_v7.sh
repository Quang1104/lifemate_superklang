#!/usr/bin/env bash
# After notebook cells 1-2: v4 (default), or V6=1 after the lead accepts v6, + history SFT.
# HIST=0.5 by default. For large ASR set BOTH ASR_DIR and ASR_TRANSCRIPTS to matching artifacts.
# PUBLIC=1 explicitly opts into public inference; smoke40 and VAL309 always run first.
set -o pipefail
source "$(dirname "$0")/colab_lib.sh"
case "${V6:-0}" in 0|1) ;; *) echo 'V6 must be 0 or 1' >&2; exit 1 ;; esac
case "${PUBLIC:-0}" in 0|1) ;; *) echo 'PUBLIC must be 0 or 1' >&2; exit 1 ;; esac
HIST=${HIST:-0.5}
python -B -c 'import math,sys; f=float(sys.argv[1]); sys.exit(0 if math.isfinite(f) and 0 <= f <= 1 else 1)' "$HIST" \
    || { echo 'HIST must be a finite fraction between 0 and 1' >&2; exit 1; }
FS=LIFEMATE_NATIVE_FEWSHOT_K=2
V7FLAGS=(LIFEMATE_CONTEXT_V4=1 LIFEMATE_SMALL_FIXES=1 LIFEMATE_FEWSHOT_GATE=0)
RENAME=()
BASE=v4
if [ "${V6:-0}" = 1 ]; then
  BASE=v6
  V7FLAGS=(LIFEMATE_CONTEXT_V4=1 LIFEMATE_SMALL_FIXES=1 LIFEMATE_FEWSHOT_GATE=1)
  RENAME=(--rename-fraction 0.3)
fi
echo "V7 BASE=$BASE flags: ${V7FLAGS[*]} history=$HIST"
# Both recipes share names: never mix SFT rows, adapters, merged weights or evaluation outputs.
for artifact in outputs/colab/sft_v7.jsonl "$R/qlora_v7" "$M/sft_v7" \
                outputs/colab/smoke_ab/v7_asrft "$R/smoke_ab/v7_asrft" \
                outputs/colab/val309/v7_asrft "$R/val309/v7_asrft" \
                /content/public_runs/v7_asrft "$R/public/v7_asrft" "$R/public/v7_asrft.progress.jsonl"; do
  [ ! -e "$artifact" ] || { echo "V7 output already exists: $artifact; move it before a fresh run" >&2; exit 1; }
done
FT=${ASR_DIR:-$M/asr_ft}
TRANSCRIPTS=${ASR_TRANSCRIPTS:-$R/asr_train_ft.jsonl}
# Restore the standard fine-tuned ASR as in v6. Custom ASR must already be staged locally.
if [ "$FT" = "$M/asr_ft" ] && [ ! -f "$FT/config.json" ]; then
  mkdir -p "$FT" && cp -r "$R/asr_ft/." "$FT/" || exit 1
fi
[ -f "$FT/config.json" ] || { echo "Missing ASR: $FT/config.json" >&2; exit 1; }
[ -s "$TRANSCRIPTS" ] || { echo "Missing TRAIN ASR transcripts: $TRANSCRIPTS; generate them with the selected ASR first" >&2; exit 1; }
env $OFF "$FS" "${V7FLAGS[@]}" python -B -m scripts.build_sft --data-root "$K" --transcripts "$TRANSCRIPTS" \
    --exclude research/validation_split.json --fewshot-k 2 "${RENAME[@]}" --history-turns "$HIST" \
    --output outputs/colab/sft_v7.jsonl || exit 1
git status --short src/agent/data/fewshot_native.jsonl  # build and inference use the same pool
python -B training_code/train_qlora.py --model-dir "$M/fallback_planner" --data outputs/colab/sft_v7.jsonl \
    --output-dir "$R/qlora_v7" --epochs 1 --save-every 25 --max-tokens 14000 2>&1 | grep --line-buffered -E 'step|resumed|rows|Error' || exit 1
merge "$R/qlora_v7" "$M/sft_v7" || exit 1

SMOKE_SCORE=$(SOURCE="--asr-dir $FT" smoke "$M/sft_v7" v7_asrft "$FS" "${V7FLAGS[@]}" | tail -1) || exit 1
echo "SMOKE v7_asrft $SMOKE_SCORE"
VAL_SCORE=$(ASR_DIR="$FT" bash scripts/colab_val.sh "$M/sft_v7" v7_asrft "$FS" "${V7FLAGS[@]}" | tail -1) || exit 1
echo "VAL v7_asrft $VAL_SCORE"
if [ "${PUBLIC:-0}" = 1 ]; then
  ASR_DIR="$FT" public "$M/sft_v7" v7_asrft "$FS" "${V7FLAGS[@]}" || exit 1
fi
date
echo V7_DONE
