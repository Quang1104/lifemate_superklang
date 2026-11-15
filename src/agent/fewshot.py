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
# Gold calls of the same TRAIN IDs for the native planner prompt; written by scripts/build_sft.py --fewshot-k.
NATIVE_POOL = POOL.with_name("fewshot_native.jsonl")
DEFAULT_K = 2
DEFAULT_BUDGET = 3_000  # characters of compact JSON for all examples together
RESULT_CAP = 500  # characters kept of each read-call result stored with a native example (TRAIN p75 is 493)
RESULTS_BUDGET = 6_000  # budget when examples carry those results (LIFEMATE_CONTEXT_V4)


def _words(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"\w+", normalized(text)))


def _jaccard(first: frozenset, second: frozenset) -> float:
    return len(first & second) / len(first | second) if first or second else 0.0


def _compact(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class Examples:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = [(row, _words(row["transcript"]), frozenset(row["apps"]),
                       frozenset(step["tool"] for step in row["plan"]["steps"]) if "plan" in row
                       else frozenset(call["name"] for call in row["calls"])) for row in rows]

    @classmethod
    def load(cls, path: Path = POOL, allowed_ids: frozenset[str] | None = None) -> "Examples":
        rows = []
        if path.is_file():
            with path.open(encoding="utf-8") as lines:
                rows = [row for row in map(json.loads, filter(str.strip, lines))
                        if allowed_ids is None or row["id"] in allowed_ids]
        return cls(rows)

    def select(self, sample: Sample, tool_names: set[str], k: int = DEFAULT_K,
               budget: int = DEFAULT_BUDGET, results: bool = False) -> list[dict]:
        """The k most similar examples whose tools are all installed here, within the character budget.

        `results` also returns (and budgets) the stored read-call results of native examples.
        """
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
            item = {"request": row["transcript"], **{key: row[key] for key in ("plan", "calls") if key in row}}
            if results and "results" in row:
                item["results"] = row["results"]
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
def configured(split: str | None = None, path: Path = POOL) -> Examples:
    allowed = None
    if split:
        allowed = frozenset(json.loads(Path(split).read_text(encoding="utf-8"))["train_ids"])
    return Examples.load(path, allowed)


def default_k() -> int:
    return int(os.environ.get("LIFEMATE_FEWSHOT_K", DEFAULT_K))


def default_examples() -> Examples:
    return configured(os.environ.get("LIFEMATE_FEWSHOT_SPLIT") or None)


def native_k() -> int:
    """Examples in the native planner prompt; off unless LIFEMATE_NATIVE_FEWSHOT_K is a positive number."""
    return int(os.environ.get("LIFEMATE_NATIVE_FEWSHOT_K") or 0)


def native_examples() -> Examples:
    return configured(os.environ.get("LIFEMATE_FEWSHOT_SPLIT") or None, NATIVE_POOL)
