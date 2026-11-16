"""V8 recipe contracts and side-effect-free Bash preflight; no data or GPU access."""

import itertools
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
V8 = (ROOT / "scripts/colab_v8.sh").read_text(encoding="utf-8")
BASH = shutil.which("bash")
# Stop before guards/setup: never source colab_lib, install packages or run inference.
PREFLIGHT = V8.split("# Every recipe shares these names:", 1)[0]
RELEASE = V8[V8.index("RELEASE_FLAGS=()") :]
KNOBS = ("HIST", "V6", "TIME_HINTS", "FULL", "PUBLIC", "ASR_DIR", "ASR_TRANSCRIPTS")


def commands(script):
    return [line.strip() for line in script.replace("\\\n", " ").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def command(marker):
    matches = [line for line in commands(V8) if marker in line]
    assert len(matches) == 1, (marker, matches)
    return matches[0]


def option(line, flag):
    parts = shlex.split(line, comments=True)
    return parts[parts.index(flag) + 1]


def preflight(settings=None, release=False):
    if BASH is None:
        pytest.skip("Bash unavailable; static contracts still run")
    environment = {key: value for key, value in os.environ.items()
                   if key not in (*KNOBS, "BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS")}
    environment.update(settings or {})
    return subprocess.run(
        [BASH, "--noprofile", "--norc", "-c", PREFLIGHT + (RELEASE if release else "")],
        env=environment, cwd=ROOT, capture_output=True, text=True, timeout=15,
    )


@pytest.mark.parametrize("name,default", [
    ("HIST", "0.5"), ("V6", "0"), ("TIME_HINTS", "1"), ("FULL", "1"), ("PUBLIC", "1"),
])
def test_defaults_and_explicit_empty_values_are_validated(name, default):
    assert f"{name}=${{{name}-{default}}}" in V8
    if name != "HIST":
        assert f'case "${name}" in 0|1)' in V8
    assert "math.isfinite(f) and 0 <= f <= 1" in V8


@pytest.mark.parametrize("name", ["V6", "TIME_HINTS", "FULL", "PUBLIC"])
@pytest.mark.parametrize("value", ["", "2", "true"])
def test_reject_invalid_boolean_knobs(name, value):
    result = preflight({name: value})
    assert result.returncode != 0
    assert f"{name} must be 0 or 1" in result.stderr


@pytest.mark.parametrize("value", ["", "nan", "inf", "-0.1", "1.01", "bad"])
def test_reject_invalid_history(value):
    result = preflight({"HIST": value})
    assert result.returncode != 0
    assert "HIST must be a finite fraction between 0 and 1" in result.stderr


@pytest.mark.parametrize("value", ["0", "0.5", "1"])
def test_history_boundaries(value):
    result = preflight({"HIST": value})
    assert result.returncode == 0, result.stderr
    assert f"HIST={value}" in result.stdout


@pytest.mark.parametrize("settings,message", [
    ({"ASR_DIR": "/models/asr_large_ft"}, "together"),
    ({"ASR_TRANSCRIPTS": "/results/asr_train_large_ft.jsonl"}, "together"),
    ({"ASR_DIR": "", "ASR_TRANSCRIPTS": ""}, "must not be empty"),
    ({"ASR_DIR": "/models/asr large", "ASR_TRANSCRIPTS": "/results/asr.jsonl"}, "whitespace"),
    ({"ASR_DIR": "/models/asr_large_ft", "ASR_TRANSCRIPTS": "/results/asr_train_ft.jsonl"}, "mismatch"),
    ({"ASR_DIR": "/models/asr_ft", "ASR_TRANSCRIPTS": "/results/asr_train_large_ft.jsonl"}, "mismatch"),
    # The artifact name matters, not a parent directory whose name happens to contain large.
    ({"ASR_DIR": "/large/asr_ft", "ASR_TRANSCRIPTS": "/results/asr_train_large_ft.jsonl"}, "mismatch"),
])
def test_reject_unpaired_empty_or_mismatched_asr(settings, message):
    result = preflight(settings)
    assert result.returncode != 0
    assert message in result.stderr


@pytest.mark.parametrize("settings", [
    {},
    {"ASR_DIR": "/models/asr_ft", "ASR_TRANSCRIPTS": "/results/asr_train_ft.jsonl"},
    {"ASR_DIR": "/models/asr_large_ft/", "ASR_TRANSCRIPTS": "/results/asr_train_large_ft.jsonl"},
])
def test_matching_asr_pairs(settings):
    result = preflight(settings)
    assert result.returncode == 0, result.stderr
    assert "ASR_DIR=" in result.stdout and "ASR_TRANSCRIPTS=" in result.stdout


@pytest.mark.parametrize("v6,time_hints,full", list(itertools.product("01", repeat=3)))
def test_recipe_selection_and_release_flags(v6, time_hints, full):
    result = preflight({"V6": v6, "TIME_HINTS": time_hints, "FULL": full}, release=True)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    recipe = lines[0]
    assert recipe.startswith(f"V8 BASE=v{'6' if v6 == '1' else '4'} ")
    assert "PUBLIC=1" in recipe
    assert "LIFEMATE_NATIVE_FEWSHOT_K=2" in recipe
    assert f"LIFEMATE_FEWSHOT_GATE={v6}" in recipe
    assert f"LIFEMATE_TIME_HINTS_V8={time_hints}" in recipe
    assert f"exclude=research/{'smoke' if full == '1' else 'validation'}_split.json" in recipe
    assert ("rename=--rename-fraction 0.3" in recipe) == (v6 == "1")
    expected = ["LIFEMATE_CONTEXT_V4=1", "LIFEMATE_SMALL_FIXES=1"]
    if v6 == "1":
        expected.append("LIFEMATE_FEWSHOT_GATE=1")
    if time_hints == "1":
        expected.append("LIFEMATE_TIME_HINTS_V8=1")
    assert lines[-2] == f'docker/release.ps1 -Flags "{",".join(expected)}"'
    assert lines[-1] == "V8_DONE"


@pytest.mark.parametrize("marker", [
    "-m scripts.build_sft", 'smoke "$M/sft_v8"',
    'bash scripts/colab_val.sh "$M/sft_v8"', 'public "$M/sft_v8"',
])
def test_all_stages_share_flags_and_propagate_failures(marker):
    line = command(marker)
    assert '"$FS" "${V8FLAGS[@]}"' in line
    assert "LIFEMATE_" not in line
    assert line.endswith("|| exit 1")


def test_build_and_asr_contract():
    build = command("-m scripts.build_sft")
    assert option(build, "--exclude") == "$EXCLUDE"
    assert option(build, "--history-turns") == "$HIST"
    assert option(build, "--fewshot-k") == "2"
    assert option(build, "--transcripts") == "$TRANSCRIPTS"
    assert "${RENAME[@]}" in shlex.split(build)
    assert "--mask-fraction" not in V8
    assert "FT=${ASR_DIR-$M/asr_ft}" in V8
    assert "TRANSCRIPTS=${ASR_TRANSCRIPTS-$R/asr_train_ft.jsonl}" in V8
    assert '[ -f "$FT/config.json" ]' in V8
    assert '[ -s "$TRANSCRIPTS" ]' in V8
    assert 'SOURCE="--asr-dir $FT"' in command('smoke "$M/sft_v8"')
    for marker in ['bash scripts/colab_val.sh "$M/sft_v8"', 'public "$M/sft_v8"']:
        assert 'ASR_DIR="$FT"' in command(marker)


def test_full_mode_never_runs_contaminated_validation():
    gate = re.search(r'if \[ "\$FULL" = 0 \]; then\n(.*?)\nelse\n(.*?)\nfi', V8, re.S)
    assert gate
    assert 'bash scripts/colab_val.sh "$M/sft_v8"' in gate[1]
    assert gate[2].strip() == "echo 'VAL309 skipped (trained on it)'"
    assert sum('bash scripts/colab_val.sh' in line for line in commands(V8)) == 1


def test_public_runs_once_after_smoke_and_validation_gate():
    gate = re.search(r'if \[ "\$PUBLIC" = 1 \]; then\n(.*?)\nfi', V8, re.S)
    assert gate and 'public "$M/sft_v8"' in gate[1]
    assert V8.index('echo "SMOKE v8_asrft') < V8.index('if [ "$FULL" = 0 ]') < gate.start()
    assert sum('public "$M/sft_v8"' in line for line in commands(V8)) == 1
    result = preflight({"PUBLIC": "0"})
    assert result.returncode == 0 and "PUBLIC=0" in result.stdout


def test_guards_cover_every_v8_artifact_before_setup_or_training():
    loop = re.search(r"for artifact in (.*?)\; do\n(.*?)\ndone", V8, re.S)
    assert loop
    assert '[ ! -e "$artifact" ]' in loop[2]
    assert '[ ! -L "$artifact" ]' in loop[2]
    assert "exit 1" in loop[2]
    assert set(shlex.split(loop[1].replace("\\\n", " "))) == {
        "outputs/colab/sft_v8.jsonl", "$R/qlora_v8", "$M/sft_v8",
        "outputs/colab/smoke_ab/v8_asrft", "$R/smoke_ab/v8_asrft",
        "outputs/colab/val309/v8_asrft", "$R/val309/v8_asrft",
        "/content/public_runs/v8_asrft", "$R/public/v8_asrft",
        "$R/public/v8_asrft.progress.jsonl",
    }
    assert V8.index('echo "V8 BASE=') < loop.start() < V8.index('source "$(') < V8.index("-m scripts.build_sft")


def test_one_epoch_merge_and_pipefail():
    train = command("training_code/train_qlora.py")
    assert option(train, "--epochs") == "1"
    assert option(train, "--output-dir") == "$R/qlora_v8"
    assert option(train, "--data") == "outputs/colab/sft_v8.jsonl"
    assert train.endswith("|| exit 1")
    assert 'merge "$R/qlora_v8" "$M/sft_v8" || exit 1' in V8
    assert "set -o pipefail" in V8
    assert commands(V8)[-1] == "echo V8_DONE"
