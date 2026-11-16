"""Shared immutable model pins. No model code/weights are imported here."""

from datetime import datetime, timezone
import re


CUTOFF = datetime(2026, 10, 26, tzinfo=timezone.utc)  # strictly before; docs/COMPETITION_RULES.md
UNRESOLVED = "REVISION_NOT_RESOLVED"


def checked_pin(pins: dict, role: str) -> dict[str, str]:
    pin = pins[role]
    if not isinstance(pin.get("repo_id"), str) or not pin["repo_id"]:
        raise ValueError(f"Missing repo_id for {role}")
    if not re.fullmatch(r"[0-9a-f]{40}", pin.get("revision", "")):
        raise ValueError(f"{role}: resolve an eligible full commit SHA before downloading (got {pin.get('revision')!r})")
    return {key: pin[key] for key in ("repo_id", "revision")}


def commit_date(commit) -> datetime:
    date = commit.created_at
    if isinstance(date, str):
        date = datetime.fromisoformat(date.replace("Z", "+00:00"))
    if date.tzinfo is None:
        raise ValueError("Hugging Face commit date must include a timezone")
    return date.astimezone(timezone.utc)


def eligible_revision(commits) -> str:
    """Choose the latest eligible commit from the public HF API, never infer a hash."""
    eligible = [commit for commit in commits if commit_date(commit) < CUTOFF]
    if not eligible:
        raise ValueError("No model revision released before 2026-10-26")
    commit = max(eligible, key=commit_date)
    return checked_pin({"candidate": {"repo_id": "vinai/PhoWhisper-large", "revision": commit.commit_id}},
                       "candidate")["revision"]


def verify_revision(pin: dict, commit) -> str:
    if commit.commit_id != pin["revision"]:
        raise ValueError("HF API revision does not match the configured full SHA")
    date = commit_date(commit)
    if date >= CUTOFF:
        raise ValueError(f"Ineligible pretrained revision date: {date.isoformat()}")
    return date.isoformat()
