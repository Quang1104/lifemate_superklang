#!/usr/bin/env bash
# Run from the repo root after notebook cells 1-2. FULL=1 reserves only smoke40.
# Set BOTH ASR_DIR and ASR_TRANSCRIPTS for custom ASR; PUBLIC defaults to 1.
set -o pipefail
# Match colab_lib.sh paths so validation, recipe logging and guards precede its setup.
R=/content/drive/MyDrive/lifemate_results
M=/content/lifemate_models
HIST=${HIST-0.5}
V6=${V6-0}
TIME_HINTS=${TIME_HINTS-1}
FULL=${FULL-1}
PUBLIC=${PUBLIC-1}
case "$V6" in 0|1) ;; *) echo 'V6 must be 0 or 1' >&2; exit 1 ;; esac
case "$TIME_HINTS" in 0|1) ;; *) echo 'TIME_HINTS must be 0 or 1' >&2; exit 1 ;; esac
case "$FULL" in 0|1) ;; *) echo 'FULL must be 0 or 1' >&2; exit 1 ;; esac
case "$PUBLIC" in 0|1) ;; *) echo 'PUBLIC must be 0 or 1' >&2; exit 1 ;; esac
python -B -c 'import math,sys; f=float(sys.argv[1]); sys.exit(0 if math.isfinite(f) and 0 <= f <= 1 else 1)' "$HIST" \
    || { echo 'HIST must be a finite fraction between 0 and 1' >&2; exit 1; }
if [ "${ASR_DIR+x}" != "${ASR_TRANSCRIPTS+x}" ]; then
  echo 'Set ASR_DIR and ASR_TRANSCRIPTS together' >&2; exit 1
fi
FT=${ASR_DIR-$M/asr_ft}
TRANSCRIPTS=${ASR_TRANSCRIPTS-$R/asr_train_ft.jsonl}
[ -n "$FT" ] && [ -n "$TRANSCRIPTS" ] || { echo 'ASR paths must not be empty' >&2; exit 1; }
# colab_lib.sh smoke splits SOURCE into arguments; a directory with whitespace cannot work.
[[ "$FT" != *[[:space:]]* ]] || { echo 'ASR_DIR must not contain whitespace' >&2; exit 1; }
ASR_NAME=${FT%/}
ASR_NAME=${ASR_NAME##*/}
TRANSCRIPT_NAME=${TRANSCRIPTS##*/}
if { [[ "$ASR_NAME" == *large* ]] && [[ "$TRANSCRIPT_NAME" != *large* ]]; } || \
   { [[ "$ASR_NAME" != *large* ]] && [[ "$TRANSCRIPT_NAME" == *large* ]]; }; then
  echo 'ASR mismatch: both artifact names must contain large, or neither may contain large' >&2; exit 1
fi
FS=LIFEMATE_NATIVE_FEWSHOT_K=2
# Explicit zeroes prevent inherited flags from silently changing a disabled recipe.
V8FLAGS=(LIFEMATE_CONTEXT_V4=1 LIFEMATE_SMALL_FIXES=1 LIFEMATE_FEWSHOT_GATE=$V6 LIFEMATE_TIME_HINTS_V8=$TIME_HINTS)
RENAME=()
BASE=v4
if [ "$V6" = 1 ]; then
  BASE=v6
  RENAME=(--rename-fraction 0.3)
fi
EXCLUDE=research/validation_split.json
if [ "$FULL" = 1 ]; then
  EXCLUDE=research/smoke_split.json
fi
echo "V8 BASE=$BASE HIST=$HIST V6=$V6 TIME_HINTS=$TIME_HINTS FULL=$FULL PUBLIC=$PUBLIC ASR_DIR=$FT ASR_TRANSCRIPTS=$TRANSCRIPTS flags: $FS ${V8FLAGS[*]} exclude=$EXCLUDE rename=${RENAME[*]} epochs=1 adapter=$R/qlora_v8 merged=$M/sft_v8 smoke=v8_asrft"
# Every recipe shares these names: refuse stale artifacts even for disabled stages.
for artifact in outputs/colab/sft_v8.jsonl "$R/qlora_v8" "$M/sft_v8" \
                outputs/colab/smoke_ab/v8_asrft "$R/smoke_ab/v8_asrft" \
                outputs/colab/val309/v8_asrft "$R/val309/v8_asrft" \
                /content/public_runs/v8_asrft "$R/public/v8_asrft" "$R/public/v8_asrft.progress.jsonl"; do
  { [ ! -e "$artifact" ] && [ ! -L "$artifact" ]; } || { echo "V8 output already exists: $artifact; move it before a fresh run" >&2; exit 1; }
done
source "$(dirname "$0")/colab_lib.sh" || exit 1
# Restore standard fine-tuned ASR as in v7; custom ASR must already be staged locally.
if [ "$FT" = "$M/asr_ft" ] && [ ! -f "$FT/config.json" ]; then
  mkdir -p "$FT" && cp -r "$R/asr_ft/." "$FT/" || exit 1
fi
[ -f "$FT/config.json" ] || { echo "Missing ASR: $FT/config.json" >&2; exit 1; }
[ -s "$TRANSCRIPTS" ] || { echo "Missing TRAIN ASR transcripts: $TRANSCRIPTS; generate them with the selected ASR first" >&2; exit 1; }
env $OFF "$FS" "${V8FLAGS[@]}" python -B -m scripts.build_sft --data-root "$K" --transcripts "$TRANSCRIPTS" \
    --exclude "$EXCLUDE" --fewshot-k 2 "${RENAME[@]}" --history-turns "$HIST" \
    --output outputs/colab/sft_v8.jsonl || exit 1
git status --short src/agent/data/fewshot_native.jsonl  # build and inference use the same pool
python -B training_code/train_qlora.py --model-dir "$M/fallback_planner" --data outputs/colab/sft_v8.jsonl \
    --output-dir "$R/qlora_v8" --epochs 1 --save-every 25 --max-tokens 14000 2>&1 | grep --line-buffered -E 'step|resumed|rows|Error' || exit 1
merge "$R/qlora_v8" "$M/sft_v8" || exit 1

SMOKE_SCORE=$(SOURCE="--asr-dir $FT" smoke "$M/sft_v8" v8_asrft "$FS" "${V8FLAGS[@]}" | tail -1) || exit 1
echo "SMOKE v8_asrft $SMOKE_SCORE"
if [ "$FULL" = 0 ]; then
  VAL_SCORE=$(ASR_DIR="$FT" bash scripts/colab_val.sh "$M/sft_v8" v8_asrft "$FS" "${V8FLAGS[@]}" | tail -1) || exit 1
  echo "VAL v8_asrft $VAL_SCORE"
else
  echo 'VAL309 skipped (trained on it)'
fi
if [ "$PUBLIC" = 1 ]; then
  ASR_DIR="$FT" public "$M/sft_v8" v8_asrft "$FS" "${V8FLAGS[@]}" || exit 1
fi
date
# release.ps1 accepts only enabled feature flags; Docker already defaults fewshot K to 2.
# TIME_HINTS=1 requires matching Dockerfile and release_flags.py support before release.
RELEASE_FLAGS=()
for setting in "${V8FLAGS[@]}"; do
  [[ "$setting" != *=1 ]] || RELEASE_FLAGS+=("$setting")
done
RELEASE_CSV=$(IFS=,; echo "${RELEASE_FLAGS[*]}")
printf 'docker/release.ps1 -Flags "%s"\n' "$RELEASE_CSV"
echo V8_DONE
