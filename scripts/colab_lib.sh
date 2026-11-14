#!/usr/bin/env bash
# Shared paths and helpers (merge, public, smoke) for the Colab run scripts; source it from the repo root.
set -x
R=/content/drive/MyDrive/lifemate_results
M=/content/lifemate_models
K=/content/train+public_test
REPO=/content/lifemate_superklang
SMOKE=research/smoke_split.json
OFF="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
cd "$REPO" || exit 1
pip install -q peft bitsandbytes && pip uninstall -y -q torchao
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date

merge() {  # merge <adapter_dir> <out_dir>
  [ -f "$2/config.json" ] || python -B training_code/stream_merge.py --base "$M/fallback_planner" --adapter "$1" --output "$2"
}
public() {  # public <model_dir> <name> [ENV=1 ...]
  local out=/content/public_runs/$2; mkdir -p "$out"
  env $OFF "${@:3}" python -B -m src.inference official --split public_test --data-root "$K" --model-dir "$1" \
      --asr-dir "$M/asr" --device cuda --planner-mode native --output "$out/submission.jsonl" || return 1
  mkdir -p "$R/public" && cp -r "$out" "$R/public/" && echo "PUBLIC_DONE $2 $(wc -l < "$out/submission.jsonl")"
}
smoke() {  # smoke <model_dir> <name> [ENV=1 ...]; prints the proxy score. SOURCE overrides the ASR input.
  local out=outputs/colab/smoke_ab/$2; mkdir -p "$out"
  env $OFF "${@:3}" python -B -m src.inference official --split train --data-root "$K" --train-manifest "$SMOKE" \
      --model-dir "$1" ${SOURCE:---asr-dir $M/asr} --device cuda --planner-mode native --time-scale 1 \
      --output "$out/native.jsonl" --train-report "$out/native.report.json" || return 1
  env $OFF python -B -m scripts.score_run --manifest "$SMOKE" --gold "$GOLD" --catalog "$K/train/train_tools_catalog.json" \
      --predictions "$out/native.jsonl" --report "$out/native.report.json" --train-jsonl "$K/train/train.jsonl" \
      --output "$out/native.scored.json" > /dev/null || return 1
  mkdir -p "$R/smoke_ab" && cp -r "$out" "$R/smoke_ab/"
  python -c "import json;print(json.load(open('$out/native.scored.json'))['local_proxy_0_to_100'])"
}

GOLD=outputs/colab/smoke_ab/gold.jsonl
mkdir -p outputs/colab/smoke_ab
[ -f "$GOLD" ] || env $OFF python -B -m scripts.replay_train_gold --data-root "$K" --manifest "$SMOKE" --output "$GOLD" || exit 1
