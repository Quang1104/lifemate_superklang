#!/usr/bin/env bash
# Overnight Colab Pro pipeline (run from a notebook cell so the kernel stays busy):
#   0. smoke40 of W on released transcripts (ASR cost)          -> lifemate_results/smoke_ab/oracle_transcripts
#   1. public test with the step-154 adapter (W)                -> lifemate_results/public/sft154
#   2. smoke40 at the real 30 s limit: baseline + one flag each   -> lifemate_results/smoke_ab (win = +5 points)
#   3. public test with W + the flags that beat the baseline     -> lifemate_results/public/sft154_<flags>
#   4. retrain v2 from the base on TRAIN SFT + pilot300 (1 epoch) -> lifemate_results/qlora_v2, smoke, public
# Inputs on Drive: lifemate_results/qlora154/{adapter_config.json,adapter_model.safetensors},
#                  lifemate_results/sft_focus_pilot300.jsonl, train+public_test.zip (unzipped by notebook cell 1).
source "$(dirname "$0")/colab_lib.sh"

merge "$R/qlora154" "$M/sft154" || exit 1

# 0. W with the released TRAIN transcripts instead of ASR: the gap to the ASR baseline is the ASR cost.
echo "SMOKE oracle_transcripts $(SOURCE=--train-transcripts smoke "$M/sft154" oracle_transcripts | tail -1)"

# 1. Public test with W.
public "$M/sft154" sft154

# 2. Smoke A/B at TIME_SCALE=1, one change at a time. smoke40 noise is ~±6.6 points (one sample = 2.5),
#    so a flag only wins with >= 5 points over the baseline.
BASE=$(smoke "$M/sft154" base | tail -1); echo "SMOKE base $BASE"
WIN=()
for flag in LIFEMATE_ARG_NORMALIZE LIFEMATE_DATE_TABLE LIFEMATE_PROMPT_V2; do
  S=$(smoke "$M/sft154" "$flag" "$flag=1" | tail -1); echo "SMOKE $flag $S"
  python -c "import sys;sys.exit(0 if float('$S')>=float('$BASE')+5 else 1)" 2>/dev/null && WIN+=("$flag=1")
done
echo "WINNERS ${WIN[*]:-none}"

# 3. Public test with the winning flags combined (skipped when none beat the baseline).
if [ ${#WIN[@]} -gt 0 ]; then
  NAME=sft154_$(printf '%s' "${WIN[*]}" | tr ' =' '__' | tr 'A-Z' 'a-z')
  [ ${#WIN[@]} -gt 1 ] && echo "SMOKE combined $(smoke "$M/sft154" combined "${WIN[@]}" | tail -1)"
  public "$M/sft154" "$NAME" "${WIN[@]}"
fi

# 4. Retrain v2 from the base on TRAIN SFT + pilot300 (pilot rows are drawn from TRAIN, so this up-weights them).
#    v2 is a comparison candidate only; it never replaces W automatically.
SFT=outputs/colab/sft_native.jsonl
[ -f "$SFT" ] || env $OFF python -B -m scripts.build_sft --data-root "$K" --output "$SFT" || exit 1
cat "$SFT" "$R/sft_focus_pilot300.jsonl" > outputs/colab/sft_v2.jsonl && wc -l outputs/colab/sft_v2.jsonl
python -B training_code/train_qlora.py --model-dir "$M/fallback_planner" --data outputs/colab/sft_v2.jsonl \
    --output-dir "$R/qlora_v2" --epochs 1 --save-every 25 --max-tokens 12000 2>&1 | grep --line-buffered -E 'step|resumed|rows|Error' || exit 1
merge "$R/qlora_v2" "$M/sft_v2" || exit 1
echo "SMOKE v2 $(smoke "$M/sft_v2" v2 | tail -1)"
public "$M/sft_v2" v2
date; echo OVERNIGHT_DONE
