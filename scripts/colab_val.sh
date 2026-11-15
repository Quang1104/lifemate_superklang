#!/usr/bin/env bash
# Usage (after notebook cells 1-2): bash scripts/colab_val.sh <model_dir> <name> [ENV=1 ...]
# Independent of colab_lib.sh: sourcing it would create/overwrite smoke gold outputs.
set -euo pipefail
set -x
R=/content/drive/MyDrive/lifemate_results
M=/content/lifemate_models
K=/content/train+public_test
REPO=/content/lifemate_superklang
OFF="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
VAL=research/validation_split.json

[ "$#" -ge 2 ] || { echo 'Usage: colab_val.sh <model_dir> <name> [ENV=1 ...]' >&2; exit 1; }
MODEL=$1
NAME=$2
shift 2
[[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo 'Use a simple run name (letters, digits, _ or -)' >&2; exit 1; }
for setting in "$@"; do
  [[ "$setting" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || { echo "Expected ENV=value: $setting" >&2; exit 1; }
done
cd "$REPO"
FT=${ASR_DIR:-$M/asr_ft}
OUT=outputs/colab/val309/$NAME
DEST=$R/val309/$NAME
GOLD=$OUT/gold.jsonl
# --resume-log accepts TRAIN, but predict_records skips on_sample for restored IDs.
# TRAIN records also omit time, so score_run cannot recover their diagnostics honestly.
# Until the CLI persists TRAIN diagnostics, scored VAL309 requires a fresh run.
echo 'VAL309 resume unavailable for scoring: TRAIN --resume-log loses restored diagnostics/timing; use a fresh name.' >&2
for artifact in "$OUT" "$DEST"; do
  [ ! -e "$artifact" ] || { echo "VAL309 output already exists: $artifact; move it before a fresh run" >&2; exit 1; }
done
[ -f "$MODEL/config.json" ] || { echo "Missing planner: $MODEL/config.json" >&2; exit 1; }
[ -f "$FT/config.json" ] || { echo "Missing ASR: $FT/config.json" >&2; exit 1; }
python -B - "$VAL" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
ids = manifest["validation_ids"]
if manifest.get("source_split") != "train" or len(ids) != 309 or len(set(ids)) != 309:
    raise SystemExit("VAL309 requires exactly 309 distinct TRAIN validation IDs")
PY
mkdir -p "$OUT"
# Separate local TRAIN oracle pass; inference below never receives gold or transcripts.
env $OFF python -B -m scripts.replay_train_gold --data-root "$K" --manifest "$VAL" --output "$GOLD"
env $OFF "$@" python -B -m src.inference official --split train --data-root "$K" --train-manifest "$VAL" \
    --model-dir "$MODEL" --asr-dir "$FT" --device cuda --planner-mode native --time-scale 1 \
    --output "$OUT/native.jsonl" --train-report "$OUT/native.report.json"
env $OFF python -B -m scripts.score_run --manifest "$VAL" --gold "$GOLD" --catalog "$K/train/train_tools_catalog.json" \
    --predictions "$OUT/native.jsonl" --report "$OUT/native.report.json" --train-jsonl "$K/train/train.jsonl" \
    --output "$OUT/native.scored.json" > /dev/null
mkdir -p "$DEST"
cp -r "$OUT/." "$DEST/"
# Keep the measured local proxy as the last stdout line for callers using tail -1.
python -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["local_proxy_0_to_100"])' "$OUT/native.scored.json"
