"""Measure first-call DIRECT prompts using released TRAIN input fields only.

This is LOCAL ORACLE PROMPT ANALYSIS: TRAIN transcripts replace ASR. No ToolEnv
is opened, and neither top-level labels nor gold calls are accessed. Only the
explicit manifest, train/train.jsonl, and train/train_tools_catalog.json are read.
"""

import argparse
from collections.abc import Callable
import json
from pathlib import Path
from statistics import median
from typing import Any

from .paths import safe_path


_INPUT_FIELDS = ("id", "audio", "context", "history", "audio_text")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _legacy_payload(request) -> str:
    """The uncompressed DIRECT payload preceding the speed-lane change."""
    history = []
    for turn in request.sample.history:
        try:
            history.append(json.loads(turn))
        except json.JSONDecodeError:
            history.append(turn)
    return _json({
        "user_request": request.sample.user_request,
        "context": request.sample.context,
        "history": history,
        "tools": request.tools,
        "observations": [],
    })


def _summary(values: list[int]) -> dict[str, int | float]:
    if not values:
        raise ValueError("Prompt measurement requires at least one sample")
    return {"sum": sum(values), "median": median(values), "min": min(values), "max": max(values)}


def _comparison(before: list[int], after: list[int]) -> dict[str, Any]:
    return {
        "before": _summary(before),
        "after": _summary(after),
        "sum_reduction_percent": round(100 * (sum(before) - sum(after)) / sum(before), 3),
    }


def _train_requests(data_root: str | Path, manifest: str | Path) -> tuple[list, dict[str, str]]:
    # Construct the exact allowlisted files; do not inspect or enumerate the kit.
    directory = safe_path(data_root)
    input_path = safe_path(directory / "train" / "train.jsonl")
    catalog_path = safe_path(directory / "train" / "train_tools_catalog.json")
    manifest_path = safe_path(manifest)
    with manifest_path.open(encoding="utf-8") as source:
        split = json.load(source)
    ids = split.get("validation_ids")
    if (split.get("source_split") != "train" or not isinstance(ids, list) or not ids
            or not all(isinstance(item, str) and item for item in ids) or len(set(ids)) != len(ids)):
        raise ValueError("Expected a TRAIN manifest with unique, nonempty validation_ids")
    selected = set(ids)

    # Use production normalization, catalog evidence annotations and app filtering.
    # These imports never load a model or instantiate the official ToolEnv.
    from ..agent.planner import PlannerInput
    from ..tooling.lifemate_catalog import build_registry
    from .pipeline import _sample_from_row

    with catalog_path.open(encoding="utf-8") as source:
        registry = build_registry(json.load(source))
    requests = {}
    with input_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("id") not in selected:
                continue
            # Match OfficialKit.rows: discard fields outside released inputs.
            # In particular, never access the TRAIN top-level label.
            row = {key: raw[key] for key in _INPUT_FIELDS}
            if row["id"] in requests:
                raise ValueError("Repeated requested TRAIN sample ID")
            # The no-ASR branch only uses input metadata. No kit or ToolEnv exists.
            sample = _sample_from_row(None, row, None, float("inf"), control_noop=False)
            tools = registry.available(sample.installed_apps, sample.allowed_tools)
            requests[sample.sample_id] = PlannerInput(sample, tools, ())
    if set(requests) != selected:
        raise ValueError("The requested TRAIN IDs are incomplete in train.jsonl")
    return [requests[item] for item in ids], {
        "manifest": str(manifest_path), "train_input": str(input_path), "catalog": str(catalog_path),
    }


def _offline_qwen_token_counter(directory: str | Path) -> Callable[[str, str | None], int]:
    """Load only a user-supplied, already-downloaded Qwen3 tokenizer."""
    directory = safe_path(directory)
    with safe_path(directory / "config.json").open(encoding="utf-8") as source:
        model_type = json.load(source).get("model_type")
    if model_type not in {"qwen3", "qwen3_5", "qwen3_5_text"}:
        raise ValueError("Token measurement requires an explicit local Qwen3-family tokenizer directory")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)

    def count(user: str, system: str | None) -> int:
        # Match LocalPlanner's escaping before applying its chat template.
        for token in tokenizer.all_special_tokens:
            if token.startswith("<"):
                user = user.replace(token, r"\u003c" + token[1:])
        if system is None:
            return len(tokenizer.encode(user, add_special_tokens=False))
        return len(tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True, tokenize=True, enable_thinking=False,
        ))

    return count


def measure_prompts(data_root: str | Path, manifest: str | Path, *,
                    tokenizer_dir: str | Path | None = None) -> dict[str, Any]:
    """Compare exact legacy and current DIRECT first-call payloads."""
    from ..agent.local_model import _DIRECT_INSTRUCTIONS, _direct_payload

    requests, source = _train_requests(data_root, manifest)
    before = [_legacy_payload(request) for request in requests]
    after = [_direct_payload(request) for request in requests]
    system_chars = len(_DIRECT_INSTRUCTIONS)
    report = {
        "purpose": "LOCAL ORACLE PROMPT ANALYSIS: released TRAIN transcripts, no ASR",
        "samples": len(requests),
        "source": source,
        "observations": "empty: first DIRECT planner call; no ToolEnv execution",
        "characters": {
            "unit": "Python Unicode characters; system_plus_user excludes chat-template framing",
            "system": system_chars,
            "user_payload": _comparison([len(item) for item in before], [len(item) for item in after]),
            "system_plus_user": _comparison([system_chars + len(item) for item in before],
                                            [system_chars + len(item) for item in after]),
        },
        "qwen3_tokens": "NOT MEASURED",
        "latency": "NOT MEASURED",
        "competition_score": "NOT MEASURED",
    }
    if tokenizer_dir is not None:
        count = _offline_qwen_token_counter(tokenizer_dir)
        report["qwen3_tokens"] = {
            "tokenizer_directory": str(safe_path(tokenizer_dir)),
            "user_payload": _comparison([count(item, None) for item in before],
                                        [count(item, None) for item in after]),
            "chat_prompt": _comparison([count(item, _DIRECT_INSTRUCTIONS) for item in before],
                                       [count(item, _DIRECT_INSTRUCTIONS) for item in after]),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path,
                        help="Explicit kit root; only train/train.jsonl and train/train_tools_catalog.json are read")
    parser.add_argument("--manifest", type=Path, default=Path("research/smoke_split.json"))
    parser.add_argument("--tokenizer-dir", type=Path,
                        help="Already-downloaded Qwen3 tokenizer; no cache search or network access")
    args = parser.parse_args(argv)
    try:
        report = measure_prompts(args.data_root, args.manifest, tokenizer_dir=args.tokenizer_dir)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.exit(2, f"Prompt measurement stopped: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
