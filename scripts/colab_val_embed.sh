#!/usr/bin/env bash
# VAL309 of the existing planner v7 with embedding retrieval (LIFEMATE_EMBED_RETRIEVAL=1), no retraining:
# only the choice of the K=2 examples changes, the prompt format is the one v7 was trained with.
# Usage (after notebook cells 1-2): bash scripts/colab_val_embed.sh
#   V7_FLAGS overrides the v7 flags (default: BASE=v4 recipe of colab_v7.sh, gate off).
set -euo pipefail
M=/content/lifemate_models
REPO=/content/lifemate_superklang
V7_BASELINE=56.73  # VAL309 of v7 with word-overlap retrieval (docs/HANDOFF.md)
read -r -a FLAGS <<< "${V7_FLAGS:-LIFEMATE_CONTEXT_V4=1 LIFEMATE_SMALL_FIXES=1 LIFEMATE_FEWSHOT_GATE=0}"
cd "$REPO"
[ -f "$M/sft_v7/config.json" ] || { echo "Missing planner: $M/sft_v7" >&2; exit 1; }
# The pinned embedder (configs/colab_models.json, role embedder); the fetch checks its commit date online.
[ -f "$M/embedder/model.safetensors" ] || python -B -m scripts.fetch_models --models-dir "$M" --roles embedder
SCORE=$(bash scripts/colab_val.sh "$M/sft_v7" v7_embed LIFEMATE_NATIVE_FEWSHOT_K=2 "${FLAGS[@]}" \
    LIFEMATE_EMBED_RETRIEVAL=1 "LIFEMATE_EMBEDDER_DIR=$M/embedder" | tail -1)
echo "VAL309 v7_embed $SCORE (v7 with word-overlap retrieval: $V7_BASELINE)"
