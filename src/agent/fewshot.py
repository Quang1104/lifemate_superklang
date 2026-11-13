"""Deterministic retrieval of verified TRAIN example plans for the compiled planner prompt (no model calls).

The shipped pool (data/fewshot_plans.jsonl) holds only plans from scripts/gold_to_plan.py that compiled and
reproduced the gold calls and state hash, for TRAIN IDs outside the 309 validation IDs. Built by
scripts/build_fewshot.py. LIFEMATE_FEWSHOT_K sets k (default 2, 0 disables); LIFEMATE_FEWSHOT_SPLIT may name
a split JSON whose train_ids restrict the pool (for leave-app-out evaluation).
"""

from functools import lru_cache
import json
import os
from pathlib import Path
import re

from ..inference.types import Sample
from .plan_values import normalized

POOL = Path(__file__).with_name("data") / "fewshot_plans.jsonl"
DEFAULT_K = 2
DEFAULT_BUDGET = 3_000  # characters of compact JSON for all examples together


def _words(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"\w+", normalized(text)))


def _jaccard(first: frozenset, second: frozenset) -> float:
    return len(first & second) / len(first | second) if first or second else 0.0


def _compact(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class Examples:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = [(row, _words(row["transcript"]), frozenset(row["apps"]),
                       frozenset(step["tool"] for step in row["plan"]["steps"])) for row in rows]

    @classmethod
    def load(cls, path: Path = POOL, allowed_ids: frozenset[str] | None = None) -> "Examples":
        rows = []
        if path.is_file():
            with path.open(encoding="utf-8") as lines:
                rows = [row for row in map(json.loads, filter(str.strip, lines))
                        if allowed_ids is None or row["id"] in allowed_ids]
        return cls(rows)

    def select(self, sample: Sample, tool_names: set[str], k: int = DEFAULT_K,
               budget: int = DEFAULT_BUDGET) -> list[dict]:
        """The k most similar examples whose tools are all installed here, within the character budget."""
        if k <= 0:
            return []
        words, apps = _words(sample.user_request), frozenset(sample.installed_apps)
        ranked = sorted(
            (-(_jaccard(words, row_words) + 0.2 * _jaccard(apps, row_apps)), row["id"], row)
            for row, row_words, row_apps, row_tools in self._rows
            if row["id"] != sample.sample_id and row_tools <= tool_names
        )
        chosen, used, transcripts = [], 0, set()
        for _, _, row in ranked:
            item = {"request": row["transcript"], "plan": row["plan"]}
            size = len(_compact(item))
            if used + size > budget or row["transcript"] in transcripts:
                continue
            chosen.append(item)
            used += size
            transcripts.add(row["transcript"])
            if len(chosen) == k:
                break
        return chosen


@lru_cache(maxsize=4)
def configured(split: str | None = None) -> Examples:
    allowed = None
    if split:
        allowed = frozenset(json.loads(Path(split).read_text(encoding="utf-8"))["train_ids"])
    return Examples.load(POOL, allowed)


def default_k() -> int:
    return int(os.environ.get("LIFEMATE_FEWSHOT_K", DEFAULT_K))


def default_examples() -> Examples:
    return configured(os.environ.get("LIFEMATE_FEWSHOT_SPLIT") or None)
