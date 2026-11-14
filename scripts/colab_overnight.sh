#!/usr/bin/env bash
# Overnight Colab Pro pipeline (run from a notebook cell so the kernel stays busy):
#   1. public test with the step-154 adapter (W)                -> lifemate_results/public/sft154
#   2. smoke40 at the real 30 s limit: baseline + one flag each   -> lifemate_results/smoke_ab
#   3. public test with W + the flags that beat the baseline     -> lifemate_results/public/sft154_<flags>
#   4. retrain v2 from the base on TRAIN SFT + pilot300 (1 epoch) -> lifemate_results/qlora_v2, smoke, public
# Inputs on Drive: lifemate_results/qlora154/{adapter_config.json,adapter_model.safetensors},
#                  lifemate_results/sft_focus_pilot300.jsonl, train+public_test.zip (unzipped by notebook cell 1).
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
smoke() {  # smoke <model_dir> <name> [ENV=1 ...]; prints the proxy score
  local out=outputs/colab/smoke_ab/$2; mkdir -p "$out"
  env $OFF "${@:3}" python -B -m src.inference official --split train --data-root "$K" --train-manifest "$SMOKE" \
      --model-dir "$1" --asr-dir "$M/asr" --device cuda --planner-mode native --time-scale 1 \
      --output "$out/native.jsonl" --train-report "$out/native.report.json" || return 1
  env $OFF python -B -m scripts.score_run --manifest "$SMOKE" --gold "$GOLD" --catalog "$K/train/train_tools_catalog.json" \
      --predictions "$out/native.jsonl" --report "$out/native.report.json" --train-jsonl "$K/train/train.jsonl" \
      --output "$out/native.scored.json" > /dev/null || return 1
  mkdir -p "$R/smoke_ab" && cp -r "$out" "$R/smoke_ab/"
  python -c "import json;print(json.load(open('$out/native.scored.json'))['local_proxy_0_to_100'])"
}

# 1. Public test with W.
merge "$R/qlora154" "$M/sft154" || exit 1
public "$M/sft154" sft154

# 2. Smoke A/B at TIME_SCALE=1, one change at a time.
GOLD=outputs/colab/smoke_ab/gold.jsonl
mkdir -p outputs/colab/smoke_ab
[ -f "$GOLD" ] || env $OFF python -B -m scripts.replay_train_gold --data-root "$K" --manifest "$SMOKE" --output "$GOLD" || exit 1
BASE=$(smoke "$M/sft154" base | tail -1); echo "SMOKE base $BASE"
WIN=()
for flag in LIFEMATE_ARG_NORMALIZE LIFEMATE_DATE_TABLE LIFEMATE_PROMPT_V2; do
  S=$(smoke "$M/sft154" "$flag" "$flag=1" | tail -1); echo "SMOKE $flag $S"
  python -c "import sys;sys.exit(0 if float('$S')>float('$BASE') else 1)" 2>/dev/null && WIN+=("$flag=1")
done
echo "WINNERS ${WIN[*]:-none}"

# 3. Public test with the winning flags combined (skipped when none beat the baseline).
if [ ${#WIN[@]} -gt 0 ]; then
  NAME=sft154_$(printf '%s' "${WIN[*]}" | tr ' =' '__' | tr 'A-Z' 'a-z')
  [ ${#WIN[@]} -gt 1 ] && echo "SMOKE combined $(smoke "$M/sft154" combined "${WIN[@]}" | tail -1)"
  public "$M/sft154" "$NAME" "${WIN[@]}"
fi

# 4. Retrain v2 from the base on TRAIN SFT + pilot300 (pilot rows are drawn from TRAIN, so this up-weights them).
SFT=outputs/colab/sft_native.jsonl
[ -f "$SFT" ] || env $OFF python -B -m scripts.build_sft --data-root "$K" --output "$SFT" || exit 1
cat "$SFT" "$R/sft_focus_pilot300.jsonl" > outputs/colab/sft_v2.jsonl && wc -l outputs/colab/sft_v2.jsonl
python -B training_code/train_qlora.py --model-dir "$M/fallback_planner" --data outputs/colab/sft_v2.jsonl \
    --output-dir "$R/qlora_v2" --epochs 1 --save-every 25 --max-tokens 12000 2>&1 | grep --line-buffered -E 'step|resumed|rows|Error' || exit 1
merge "$R/qlora_v2" "$M/sft_v2" || exit 1
echo "SMOKE v2 $(smoke "$M/sft_v2" v2 | tail -1)"
public "$M/sft_v2" v2
date; echo OVERNIGHT_DONE
