from fnmatch import fnmatch
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def _included(path: str, rules: list[str]) -> bool:
    # Docker semantics: the last matching rule wins; "!" re-includes.
    keep = True
    for rule in rules:
        negate = rule.startswith("!")
        pattern = rule[1:] if negate else rule
        if fnmatch(path, pattern) or fnmatch(path, pattern.rstrip("/") + "/**"):
            keep = negate
    return keep


def test_every_dockerfile_copy_source_survives_dockerignore():
    rules = [line.strip() for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    sources = [src for line in dockerfile.splitlines() if line.startswith("COPY ") and "--from=" not in line
               for src in re.sub(r"--\S+\s+", "", line[5:]).split()[:-1]]
    files = [str(path.relative_to(ROOT)).replace("\\", "/") for src in sources
             for path in ([ROOT / src] if (ROOT / src).is_file() else (ROOT / src).rglob("*"))
             if path.is_file() and "__pycache__" not in path.parts]
    assert files and [f for f in files if not _included(f, rules)] == []
