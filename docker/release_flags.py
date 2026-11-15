"""Allowlist and parser for the feature flags baked into a release image (docker/release.ps1 -Flags).

A planner must run with the prompt flags it was trained with, so the flags are an explicit build input:
`release.ps1 -Flags "LIFEMATE_CONTEXT_V4=1,LIFEMATE_SMALL_FIXES=1"` validates them here, passes each as a
build argument, the Dockerfile turns it into an ENV line, and the release manifest records them.

    python -B docker/release_flags.py "LIFEMATE_CONTEXT_V4=1,LIFEMATE_SMALL_FIXES=1"   # prints one NAME=1 per line
"""

import sys

# On/off switches read in src/ with `os.environ.get(NAME) == "1"`; the only accepted value is 1.
ALLOWED = (
    "LIFEMATE_ARG_NORMALIZE", "LIFEMATE_ASR_FIX", "LIFEMATE_ASR_PROMPT", "LIFEMATE_CONTACT_RETRY",
    "LIFEMATE_CONTEXT_V4", "LIFEMATE_DATE_TABLE", "LIFEMATE_PROMPT_V2", "LIFEMATE_SMALL_FIXES",
)


def parse(text: str) -> list[str]:
    """Sorted flag names from "NAME=1,NAME=1"; unknown names, other values and repeats are errors."""
    names = []
    for part in filter(None, (item.strip() for item in text.split(","))):
        name, separator, value = part.partition("=")
        if name not in ALLOWED:
            raise ValueError(f"Unknown release flag {name!r}; allowed: {', '.join(ALLOWED)}")
        if separator != "=" or value != "1":
            raise ValueError(f"{name} must be written as {name}=1 (leave a flag out to keep it off)")
        if name in names:
            raise ValueError(f"{name} is listed twice")
        names.append(name)
    return sorted(names)


if __name__ == "__main__":
    try:
        text = " ".join(sys.argv[1:])
        for name in parse(text[6:] if text.startswith("flags:") else text):
            print(f"{name}=1")
    except ValueError as error:
        sys.exit(f"release flags rejected: {error}")
