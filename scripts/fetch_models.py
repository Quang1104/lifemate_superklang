"""Fetch selected immutable Colab models, checking HF API commit dates before weights.

Default downloads are unchanged. --resolve-asr-large fills only the unresolved large pin;
subsequent runs keep that SHA. It is intended for an online Colab setup, never inference.
"""

import argparse
import json
from pathlib import Path

from training_code.model_pins import UNRESOLVED, checked_pin, eligible_revision, verify_revision


PATTERNS = ["*.json", "*.jinja", "*.safetensors", "*.bin", "merges.txt", "LICENSE", "README.md"]
PIN_FILE = Path(__file__).resolve().parents[1] / "configs" / "colab_models.json"


def fetch_models(models_dir: Path, roles, *, resolve_asr_large: bool = False,
                 pin_file: Path = PIN_FILE) -> None:
    from huggingface_hub import HfApi, snapshot_download

    pins = json.loads(pin_file.read_text(encoding="utf-8"))
    api = HfApi()
    if resolve_asr_large and pins["asr_large"]["revision"] == UNRESOLVED:
        repo_id = pins["asr_large"]["repo_id"]
        pins["asr_large"]["revision"] = eligible_revision(api.list_repo_commits(repo_id=repo_id))
        # Persist the real resolved SHA before any weights are fetched.
        temporary = pin_file.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(pins, indent=2) + "\n", encoding="utf-8")
        temporary.replace(pin_file)
    for role in roles:
        pin = checked_pin(pins, role)
        commits = api.list_repo_commits(repo_id=pin["repo_id"], revision=pin["revision"])
        if not commits:
            raise ValueError(f"No commit metadata for {role}")
        date = verify_revision(pin, commits[0])
        print(json.dumps({"role": role, **pin, "commit_date_utc": date}), flush=True)
        snapshot_download(**pin, local_dir=models_dir / role, allow_patterns=PATTERNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("/content/lifemate_models"))
    parser.add_argument("--roles", nargs="+", choices=("planner", "fallback_planner", "asr", "asr_large"),
                        default=["fallback_planner", "asr"])
    parser.add_argument("--resolve-asr-large", action="store_true")
    args = parser.parse_args()
    fetch_models(args.models_dir, args.roles, resolve_asr_large=args.resolve_asr_large)


if __name__ == "__main__":
    main()
