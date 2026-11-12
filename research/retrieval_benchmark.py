"""TRAIN-oracle schema retrieval diagnostic. Never ranks using labels.

Fixed lexical scoring and x-ref producer closure; no model weights or tuning.
Gold labels are accessed only after candidate sets have been frozen per sample.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import time
import unicodedata

from research.data_audit import call_name, calls_for, extensions, extract_tools, read_jsonl, safe_path


def tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFC", text).casefold(), flags=re.UNICODE)


def allowed_names(installed_apps: dict, tools: dict[str, dict]) -> list[str]:
    if not isinstance(installed_apps, dict):
        raise ValueError("Official installed_apps must be an object mapping apps to tools")
    allowed = set()
    for app, metadata in installed_apps.items():
        if not isinstance(metadata, dict) or not isinstance(metadata.get("tools"), list):
            raise ValueError("Unsupported installed app schema")
        for name in metadata["tools"]:
            if isinstance(name, str) and name.startswith(app + ".") and name in tools:
                allowed.add(name)
    if "sys.ask_missing_info" in tools:
        allowed.add("sys.ask_missing_info")
    return sorted(allowed)


def rank_tools(query: str, installed_apps: dict, tools: dict[str, dict]) -> list[str]:
    """Only user transcript + installed metadata + public schema, never labels."""
    names = allowed_names(installed_apps, tools)
    documents = {name: Counter(tokenize(json.dumps(tools[name], ensure_ascii=False, sort_keys=True))) for name in names}
    df = Counter(token for document in documents.values() for token in document)
    query_terms = Counter(tokenize(query))
    scores = {}
    for name, document in documents.items():
        score = sum(min(frequency, document[token]) * (math.log((len(names) + 1) / (df[token] + 1)) + 1)
                    for token, frequency in query_terms.items())
        scores[name] = score / math.sqrt(max(1, sum(document.values())))
    return sorted(names, key=lambda name: (-scores[name], name))


def dependency_closure(selected: list[str], allowed: list[str], tools: dict[str, dict]) -> list[str]:
    """Follow declared x-ref namespaces to installed read-tool return fields.

    Producer metadata can be incomplete. Fallback is all installed read tools
    from that referenced app, explicitly a conservative analysis heuristic.
    """
    selected_set = set(selected)
    frontier = list(selected)
    while frontier:
        name = frontier.pop()
        refs = [item["value"] for item in extensions(tools[name]) if item["extension"] == "x-ref"]
        for ref in refs:
            if not isinstance(ref, str) or "." not in ref:
                continue
            app, collection = ref.split(".", 1)
            readers = [candidate for candidate in allowed if candidate.split(".")[0] == app and tools[candidate].get("kind") == "read"]
            producers = [candidate for candidate in readers if collection in tokenize(str(tools[candidate].get("returns", "")))]
            for producer in producers or readers:
                if producer not in selected_set:
                    selected_set.add(producer)
                    frontier.append(producer)
    return [name for name in allowed if name in selected_set]


def candidate_sets(query: str, installed_apps: dict, tools: dict[str, dict]) -> dict[str, list[str]]:
    ranking = rank_tools(query, installed_apps, tools)
    allowed = sorted(ranking)
    result = {"all_installed": allowed}
    for k in (8, 16):
        selected = ranking[:k]
        # Clarification mechanism is always available, possibly making k+1.
        if "sys.ask_missing_info" in allowed and "sys.ask_missing_info" not in selected:
            selected = [*selected, "sys.ask_missing_info"]
        result[f"lexical_top{k}"] = selected
        result[f"lexical_top{k}_xref_closure"] = dependency_closure(selected, allowed, tools)
    return result


def benchmark(rows: list[dict], tools: dict[str, dict]) -> dict:
    aggregated = {}
    per_sample = []
    total_seconds = 0.0
    for row in rows:
        begin = time.perf_counter()
        candidates = candidate_sets(row["audio_text"], row["context"]["installed_apps"], tools)
        elapsed = time.perf_counter() - begin
        total_seconds += elapsed
        # Evaluation labels are first accessed AFTER ranking has completed.
        gold_names = {call_name(call) for call in calls_for(row)}
        serialized_sizes = {name: len(json.dumps(tools[name], ensure_ascii=False, sort_keys=True)) for name in candidates["all_installed"]}
        full_size = sum(serialized_sizes.values())
        details = {"id": str(row["id"]), "gold_unique_tool_count": len(gold_names), "variants": {}}
        for variant, selected in candidates.items():
            matched = len(gold_names & set(selected))
            size = sum(serialized_sizes[name] for name in selected)
            acc = aggregated.setdefault(variant, Counter())
            acc.update(samples=1, gold_tools=len(gold_names), recalled_tools=matched,
                       all_gold_recalled=int(gold_names <= set(selected)),
                       candidate_count=len(selected), schema_characters=size,
                       full_schema_characters=full_size, nonempty_samples=int(bool(gold_names)),
                       nonempty_all_recalled=int(bool(gold_names) and gold_names <= set(selected)))
            details["variants"][variant] = {"recalled": matched, "gold": len(gold_names),
                                            "candidates": len(selected), "schema_characters": size}
        per_sample.append(details)
    reports = {}
    for variant, counts in aggregated.items():
        reports[variant] = {
            "gold_tool_recall_micro": counts["recalled_tools"] / counts["gold_tools"] if counts["gold_tools"] else None,
            "all_gold_tools_recalled_nonempty_fraction": counts["nonempty_all_recalled"] / counts["nonempty_samples"] if counts["nonempty_samples"] else None,
            "mean_candidates": counts["candidate_count"] / counts["samples"],
            "mean_schema_characters": counts["schema_characters"] / counts["samples"],
            "schema_character_reduction": 1 - counts["schema_characters"] / counts["full_schema_characters"] if counts["full_schema_characters"] else None,
            "recalled_tools": counts["recalled_tools"], "gold_tools": counts["gold_tools"],
        }
    return {"diagnostic": "TRAIN oracle schema retrieval, not competition accuracy",
            "configuration": "fixed Unicode lexical TF-IDF-like overlap; k=8,16; sys clarification always included; optional x-ref closure",
            "checkpoint": "none (deterministic lexical retrieval)", "sample_count": len(rows),
            "mean_retrieval_ms_all_variants": 1000 * total_seconds / len(rows) if rows else None,
            "competition_score": "NOT MEASURED", "variants": reports,
            "decision": "Do not adopt pruning unless recall and end-to-end validation justify lost tools. No held-out labels used in ranking or tuning.",
            "samples": per_sample}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("research/validation_split.json"))
    parser.add_argument("--output", type=Path, default=Path("research/retrieval_report.json"))
    args = parser.parse_args()
    manifest = json.loads(safe_path(args.manifest).read_text(encoding="utf-8"))
    ids = set(manifest["validation_ids"])
    rows = [row for row in read_jsonl(args.train, train_only=True) if str(row["id"]) in ids]
    if {str(row["id"]) for row in rows} != ids:
        raise ValueError("Manifest IDs not fully present in TRAIN")
    tools = extract_tools(json.loads(safe_path(args.catalog).read_text(encoding="utf-8")))
    report = benchmark(rows, tools)
    report.update(dataset=str(args.train), manifest=str(args.manifest), seed=manifest["seed"])
    safe_path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
