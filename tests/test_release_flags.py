"""Release flags: one validated input that reaches the image as ENV lines (static checks, Docker never runs)."""

import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCKER = ROOT / "docker"
_spec = importlib.util.spec_from_file_location("release_flags", DOCKER / "release_flags.py")
release_flags = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release_flags)


def test_flags_are_parsed_sorted_and_empty_means_none():
    assert release_flags.parse("LIFEMATE_SMALL_FIXES=1, LIFEMATE_CONTEXT_V4=1") == [
        "LIFEMATE_CONTEXT_V4", "LIFEMATE_SMALL_FIXES"]
    assert release_flags.parse("") == [] and release_flags.parse(" , ") == []


@pytest.mark.parametrize("text", [
    "LIFEMATE_CONTEXT_V5=1",  # no such flag
    "LIFEMATE_NATIVE_FEWSHOT_K=3",  # a real variable, but not an on/off release flag
    "LIFEMATE_CONTEXT_V4=0", "LIFEMATE_CONTEXT_V4=true", "LIFEMATE_CONTEXT_V4", "LIFEMATE_CONTEXT_V4=",
    "LIFEMATE_CONTEXT_V4=1,LIFEMATE_CONTEXT_V4=1", "lifemate_context_v4=1", "LIFEMATE_CONTEXT_V4=1;LIFEMATE_DUP_GUARD=1",
])
def test_unknown_names_other_values_and_repeats_are_rejected(text):
    with pytest.raises(ValueError):
        release_flags.parse(text)


def test_command_line_prints_build_arguments_or_fails():
    script = str(DOCKER / "release_flags.py")
    ok = subprocess.run([sys.executable, "-B", script, "flags:LIFEMATE_DATE_TABLE=1,LIFEMATE_CONTEXT_V4=1"],
                        capture_output=True, text=True)
    assert ok.returncode == 0 and ok.stdout.split() == ["LIFEMATE_CONTEXT_V4=1", "LIFEMATE_DATE_TABLE=1"]
    none = subprocess.run([sys.executable, "-B", script, "flags:"], capture_output=True, text=True)
    assert none.returncode == 0 and none.stdout == ""
    bad = subprocess.run([sys.executable, "-B", script, "flags:LIFEMATE_NOPE=1"], capture_output=True, text=True)
    assert bad.returncode != 0 and "LIFEMATE_NOPE" in bad.stderr and bad.stdout == ""


def test_every_allowed_flag_is_an_on_off_switch_read_in_src():
    source = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "src").rglob("*.py"))
    for name in release_flags.ALLOWED:
        assert re.search(r'os\.environ\.get\("%s"\) (==|!=) "1"' % name, source), name
    assert tuple(sorted(release_flags.ALLOWED)) == release_flags.ALLOWED


def test_dockerfile_declares_exactly_the_allowed_flags_as_build_arguments_and_env():
    dockerfile = (DOCKER / "Dockerfile").read_text(encoding="utf-8")
    for name in release_flags.ALLOWED:
        assert re.search(rf"^ARG {name}=$", dockerfile, re.M), name
        assert f"{name}=${{{name}}}" in dockerfile, name
    declared = set(re.findall(r"^ARG (LIFEMATE_\w+)=", dockerfile, re.M))
    assert declared == set(release_flags.ALLOWED)
    assert dockerfile.index("ARG LIFEMATE_") > dockerfile.index("fetch_models.py /tmp/adapter")  # no model rebuild per flag


def test_release_script_validates_passes_and_records_the_flags():
    script = (DOCKER / "release.ps1").read_text(encoding="utf-8")
    assert "[string]$Flags = ''" in script and "release_flags.py" in script
    assert script.index("release_flags.py') \"flags:$Flags\"") < script.index("if ($CheckOnly)")  # validated first
    assert "$buildArguments += @('--build-arg', $flag)" in script
    assert script.index("'--build-arg', $flag") < script.index("& docker @buildArguments")
    for field in ("flags = $flagLines", "adapter_sha256", "asr_sha256", ".manifest.json"):
        assert field in script, field


def test_predict_script_and_notebook_never_set_or_clear_a_release_flag():
    script = (DOCKER / "predict.sh").read_text(encoding="utf-8")
    notebook = json.loads((DOCKER / "predict_notebook.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    for name in (*release_flags.ALLOWED, "LIFEMATE_NATIVE_FEWSHOT_K"):
        assert not re.search(rf"\b{name}=", script) and f"unset {name}" not in script, name
        assert name not in code, name
    assert "environ.clear" not in code and "env -i" not in script
