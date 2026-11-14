#!/usr/bin/env bash
set -euo pipefail

cd /code
exec python -B -m src.inference official \
    --split private_test --data-root /data \
    --model-dir "${LIFEMATE_PLANNER_MODEL_DIR:-/models/planner}" --asr-dir /models/asr \
    --device cuda --planner-mode "${LIFEMATE_PLANNER_MODE:-native}" \
    --output "${1:-/result/submission.jsonl}"
