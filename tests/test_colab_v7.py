"""Static Colab recipe contracts; no models, datasets, ToolEnv or GPU required."""

import json
from pathlib import Path
import re
import shlex

import pytest


ROOT = Path(__file__).resolve().parents[1]
V7 = (ROOT / "scripts/colab_v7.sh").read_text(encoding="utf-8")
VAL = (ROOT / "scripts/colab_val.sh").read_text(encoding="utf-8")


def commands(script):
    """Join Bash continuations while keeping quoted argument boundaries intact."""
    return [line.strip() for line in script.replace("\\\n", " ").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def command(script, marker):
    matches = [line for line in commands(script) if marker in line]
    assert len(matches) == 1, (marker, matches)
    return matches[0]


def tokens(line):
    return shlex.split(line, comments=True)


def option(line, flag):
    parts = tokens(line)
    return parts[parts.index(flag) + 1]


def guarded_paths(script):
    loop = re.search(r"for artifact in (.*?)\; do\n(.*?)\ndone", script, re.S)
    assert loop
    assert '[ ! -e "$artifact" ]' in loop[2]
    assert "exit 1" in loop[2]
    return set(shlex.split(loop[1].replace("\\\n", " ")))


def test_recipe_selection_and_no_masking():
    recipes = re.findall(r"V7FLAGS=\(([^)]+)\)", V7)
    assert [set(shlex.split(recipe)) for recipe in recipes] == [
        {"LIFEMATE_CONTEXT_V4=1", "LIFEMATE_SMALL_FIXES=1", "LIFEMATE_FEWSHOT_GATE=0"},
        {"LIFEMATE_CONTEXT_V4=1", "LIFEMATE_SMALL_FIXES=1", "LIFEMATE_FEWSHOT_GATE=1"},
    ]
    assert "BASE=v4" in V7
    v6 = re.search(r'if \[ "\$\{V6:-0\}" = 1 \]; then\n(.*?)\nfi', V7, re.S)
    assert v6 and "BASE=v6" in v6[1]
    assert "RENAME=(--rename-fraction 0.3)" in v6[1]
    assert "RENAME=()" in V7[:v6.start()]
    assert "--mask-fraction" not in V7
    assert "LIFEMATE_DATE_TABLE" not in V7


@pytest.mark.parametrize("marker", [
    "-m scripts.build_sft", 'smoke "$M/sft_v7"',
    'bash scripts/colab_val.sh "$M/sft_v7"', 'public "$M/sft_v7"',
])
def test_all_stages_share_recipe_flags(marker):
    line = command(V7, marker)
    assert '"$FS" "${V7FLAGS[@]}"' in line
    assert "LIFEMATE_" not in line  # no stage-specific overrides
    assert "FS=LIFEMATE_NATIVE_FEWSHOT_K=2" in V7


def test_history_transcripts_and_validation_exclusion():
    build = command(V7, "-m scripts.build_sft")
    assert "HIST=${HIST:-0.5}" in V7
    assert option(build, "--history-turns") == "$HIST"
    assert "math.isfinite(f) and 0 <= f <= 1" in V7
    assert option(build, "--fewshot-k") == "2"
    assert option(build, "--exclude") == "research/validation_split.json"
    assert option(build, "--transcripts") == "$TRANSCRIPTS"
    assert '${RENAME[@]}' in tokens(build)
    assert "TRANSCRIPTS=${ASR_TRANSCRIPTS:-$R/asr_train_ft.jsonl}" in V7
    assert '[ -s "$TRANSCRIPTS" ]' in V7


def test_asr_parity_across_evaluations():
    assert "FT=${ASR_DIR:-$M/asr_ft}" in V7
    assert "FT=${ASR_DIR:-$M/asr_ft}" in VAL
    assert 'SOURCE="--asr-dir $FT"' in command(V7, 'smoke "$M/sft_v7"')
    for marker in ['bash scripts/colab_val.sh "$M/sft_v7"', 'public "$M/sft_v7"']:
        assert 'ASR_DIR="$FT"' in command(V7, marker)
    infer = command(VAL, "-m src.inference official")
    assert option(infer, "--asr-dir") == "$FT"
    assert 'env $OFF "$@" python' in infer
    assert "--train-transcripts" not in infer


def test_fresh_output_guards_cover_all_v7_artifacts():
    assert guarded_paths(V7) == {
        "outputs/colab/sft_v7.jsonl", "$R/qlora_v7", "$M/sft_v7",
        "outputs/colab/smoke_ab/v7_asrft", "$R/smoke_ab/v7_asrft",
        "outputs/colab/val309/v7_asrft", "$R/val309/v7_asrft",
        "/content/public_runs/v7_asrft", "$R/public/v7_asrft",
        "$R/public/v7_asrft.progress.jsonl",
    }
    assert V7.index("for artifact in") < V7.index("-m scripts.build_sft")
    assert guarded_paths(VAL) == {"$OUT", "$DEST"}
    assert VAL.index("for artifact in") < VAL.index("-m scripts.replay_train_gold")


def test_one_epoch_merge_and_failure_propagation():
    train = command(V7, "training_code/train_qlora.py")
    assert option(train, "--epochs") == "1"
    assert option(train, "--output-dir") == "$R/qlora_v7"
    assert option(train, "--data") == "outputs/colab/sft_v7.jsonl"
    assert 'merge "$R/qlora_v7" "$M/sft_v7" || exit 1' in V7
    assert "set -o pipefail" in V7
    assert "set -euo pipefail" in VAL
    for marker in ["-m scripts.build_sft", "training_code/train_qlora.py",
                   'smoke "$M/sft_v7"', 'bash scripts/colab_val.sh "$M/sft_v7"',
                   'public "$M/sft_v7"']:
        assert command(V7, marker).endswith("|| exit 1")


def test_public_is_explicit_opt_in_after_smoke_and_val():
    gate = re.search(r'if \[ "\$\{PUBLIC:-0\}" = 1 \]; then\n(.*?)\nfi', V7, re.S)
    assert gate and 'public "$M/sft_v7"' in gate[1]
    assert V7.index('echo "SMOKE v7_asrft $SMOKE_SCORE"') < V7.index('echo "VAL v7_asrft $VAL_SCORE"') < gate.start()
    assert sum('public "$M/sft_v7"' in line for line in commands(V7)) == 1


def test_val_is_train_only_with_separate_gold_and_outputs():
    assert "VAL=research/validation_split.json" in VAL
    assert "OUT=outputs/colab/val309/$NAME" in VAL
    assert "DEST=$R/val309/$NAME" in VAL
    assert "GOLD=$OUT/gold.jsonl" in VAL
    assert not any(line.startswith("source ") for line in commands(VAL))
    assert "smoke_ab" not in VAL
    assert "SMOKE=" not in VAL
    infer = command(VAL, "-m src.inference official")
    assert option(infer, "--split") == "train"
    assert option(infer, "--time-scale") == "1"
    assert option(infer, "--train-manifest") == "$VAL"
    assert option(infer, "--train-report") == "$OUT/native.report.json"
    assert option(infer, "--output") == "$OUT/native.jsonl"
    replay = command(VAL, "-m scripts.replay_train_gold")
    score = command(VAL, "-m scripts.score_run")
    assert option(replay, "--manifest") == option(score, "--manifest") == "$VAL"
    assert option(replay, "--output") == option(score, "--gold") == "$GOLD"
    assert option(score, "--predictions") == "$OUT/native.jsonl"
    assert option(score, "--report") == "$OUT/native.report.json"
    assert 'cp -r "$OUT/." "$DEST/"' in VAL
    assert "local_proxy_0_to_100" in commands(VAL)[-1]


def test_val_manifest_is_309_including_smoke40():
    validation = json.loads((ROOT / "research/validation_split.json").read_text())
    smoke = json.loads((ROOT / "research/smoke_split.json").read_text())
    ids = validation["validation_ids"]
    assert validation["source_split"] == "train"
    assert len(ids) == len(set(ids)) == 309
    assert len(smoke["validation_ids"]) == 40
    assert set(smoke["validation_ids"]) <= set(ids)
    assert 'len(ids) != 309 or len(set(ids)) != 309' in VAL


def test_val_explains_unscoreable_train_resume_instead_of_faking_diagnostics():
    # The existing CLI accepts TRAIN resume but cannot restore its report. Revisit
    # this contract when durable TRAIN diagnostics are added to the inference CLI.
    assert "resume unavailable for scoring" in VAL
    assert "--resume-log" not in command(VAL, "-m src.inference official")
    assert "predict_records skips on_sample" in VAL
    assert "TRAIN records also omit time" in VAL
    assert "write_text" not in VAL


def test_val_rejects_path_traversal_and_non_environment_arguments():
    assert '[[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]' in VAL
    assert '[[ "$setting" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]' in VAL
    assert '[ "$#" -ge 2 ]' in VAL
