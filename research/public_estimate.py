"""Label-free PUBLIC estimates anchored to user-supplied portal scores.

Run: python -B -m research.public_estimate
Reads only explicitly named prediction files, TRAIN score dumps and released
catalogs/PUBLIC inputs. No ToolEnv, state access, writes, downloads or dependencies.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev


NOT_MEASURED = "NOT MEASURED"
KNOWN = {"v3": 27.98, "v3_asrft": 42.25, "mixed": 42.07, "v7": 43.0}
PUBLIC_SIZE = 634


def safe_path(path):
    """Reject forbidden components before even resolving/reading the path."""
    path = Path(path)
    if any(p.lower().endswith("_environment") or p.lower() == "state.json"
           for p in path.parts):
        raise ValueError("Forbidden data path")
    resolved = path.resolve()
    if any(p.lower().endswith("_environment") or p.lower() == "state.json"
           for p in resolved.parts):
        raise ValueError("Forbidden resolved data path")
    return resolved


def read_json(path):
    with safe_path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def read_jsonl(path):
    with safe_path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def index_rows(rows):
    result = {}
    for row in rows:
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in result:
            raise ValueError("Missing, invalid or duplicate sample ID")
        result[sample_id] = row
    return result


def canonical(value):
    # Object-key order is irrelevant; call order and list order are preserved.
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def read_public(path, expected_ids):
    rows = index_rows(read_jsonl(path))
    if set(rows) != set(expected_ids):
        raise ValueError(f"PUBLIC ID coverage mismatch: {path}")
    for row in rows.values():
        if not isinstance(row.get("tool_calls"), list):
            raise ValueError("PUBLIC tool_calls must be a list")
        if not isinstance(row.get("state_hash"), str) or not row["state_hash"]:
            raise ValueError("PUBLIC state_hash must be present and nonempty")
        canonical(row["tool_calls"])
    return rows


def read_train(path):
    rows = index_rows(read_json(path)["samples"])
    for row in rows.values():
        score = row.get("score")
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("TRAIN score must be finite and within [0, 1]")
        if not isinstance(row.get("predicted"), list):
            raise ValueError("TRAIN predictions missing; exact output comparison unavailable")
        if not isinstance(row.get("state_match"), bool):
            raise ValueError("TRAIN state_match must be boolean")
        if "gold" not in row:
            raise ValueError("TRAIN gold required to verify matched scoring targets")
        if "state_hash" in row and (not isinstance(row["state_hash"], str)
                                     or not row["state_hash"]):
            raise ValueError("Invalid optional predicted TRAIN state_hash")
    return rows


def public_groups(data_root):
    root = safe_path(data_root)
    train = read_json(root / "train/train_tools_catalog.json")
    public = read_json(root / "public_test/public_test_tools_catalog.json")
    new_apps = {a["id"] for a in public["apps"]} - {a["id"] for a in train["apps"]}
    rows = index_rows(read_jsonl(root / "public_test/public_test.jsonl"))
    if len(rows) != PUBLIC_SIZE:
        raise ValueError(f"Expected {PUBLIC_SIZE} released PUBLIC inputs")
    return {i: "public-only-app" if set(r["context"]["installed_apps"]) & new_apps
            else "TRAIN-app" for i, r in rows.items()}


def changed_public(candidate, reference):
    if not candidate:
        raise ValueError("PUBLIC comparison must not be empty")
    if candidate.keys() != reference.keys():
        raise ValueError("Compared PUBLIC files must have identical IDs")
    return {i for i in candidate if (
        canonical(candidate[i]["tool_calls"]) != canonical(reference[i]["tool_calls"])
        or candidate[i]["state_hash"] != reference[i]["state_hash"])}


def select_train(candidate, reference):
    """Prefer full paired VAL, then smoke, then a complete smoke subset of VAL."""
    for csplit, rsplit in (("val", "val"), ("smoke", "smoke"),
                           ("val", "smoke"), ("smoke", "val")):
        if csplit not in candidate or rsplit not in reference:
            continue
        cpath, crows = candidate[csplit]
        rpath, rrows = reference[rsplit]
        ids = crows.keys() & rrows.keys()
        if csplit == rsplit and (crows.keys() != rrows.keys()):
            raise ValueError("Same-split TRAIN runs have different ID coverage")
        if csplit != rsplit:
            smoke = crows if csplit == "smoke" else rrows
            if ids != smoke.keys():
                raise ValueError("VAL run does not cover the complete paired smoke set")
        if ids:
            source = f"{'VAL' if csplit == rsplit == 'val' else 'smoke'}{len(ids)}: {Path(cpath).name} / {Path(rpath).name}"
            return crows, rrows, sorted(ids), source
    return None


def train_deltas(pair):
    """Identify changed outputs without treating equal false state_match as equal hashes.

    Ambiguous pairs have identical calls, both hashes unequal to gold, and zero
    score delta. Include any number of these zeros for denominator sensitivity.
    """
    candidate, reference, ids, source = pair
    deltas, ambiguous = [], 0
    for sample_id in ids:
        c, r = candidate[sample_id], reference[sample_id]
        if canonical(c["gold"]) != canonical(r["gold"]):
            raise ValueError("Paired TRAIN runs have different gold targets")
        delta = 100.0 * (c["score"] - r["score"])
        calls_differ = canonical(c["predicted"]) != canonical(r["predicted"])
        have_hashes = "state_hash" in c and "state_hash" in r
        if have_hashes:
            hash_differs = c["state_hash"] != r["state_hash"]
            if not hash_differs and c["state_match"] != r["state_match"]:
                raise ValueError("Inconsistent TRAIN hash evidence")
            different = calls_differ or hash_differs
            uncertain = False
        else:
            different = calls_differ or c["state_match"] != r["state_match"]
            uncertain = not different and not c["state_match"]
        if different:
            deltas.append(delta)
        elif abs(delta) > 1e-9:
            raise ValueError("Same/ambiguous TRAIN outputs have inconsistent scores")
        elif uncertain:
            ambiguous += 1
    return {"deltas": deltas, "ambiguous": ambiguous, "paired": len(ids), "source": source}


def summarize_delta(deltas, ambiguous=0):
    if not deltas:
        return {"d": None, "se": None, "interval": None, "d_sensitivity": None}
    d = mean(deltas)
    se = stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) >= 2 else None
    intervals = []
    # Envelope of normal-SE intervals for every possible missing-hash denominator.
    # This covers denominator sensitivity, not PUBLIC domain shift or portal rounding.
    for extra in range(ambiguous + 1):
        values = deltas + [0.0] * extra
        if len(values) >= 2:
            error = 1.96 * stdev(values) / math.sqrt(len(values))
            intervals.append((mean(values) - error, mean(values) + error))
    interval = ((min(x[0] for x in intervals), max(x[1] for x in intervals))
                if se is not None else None)
    endpoints = (d, sum(deltas) / (len(deltas) + ambiguous))
    return {"d": d, "se": se, "interval": interval,
            "d_sensitivity": (min(endpoints), max(endpoints))}


def estimate(candidate, reference, reference_score, train_pair=None, groups=None):
    changed = changed_public(candidate, reference)
    fraction = len(changed) / len(candidate)
    result = {"n_diff": len(changed), "groups": {}, "d": None, "estimate": None,
              "interval": None, "source": NOT_MEASURED}
    if groups is not None:
        if groups.keys() != candidate.keys():
            raise ValueError("PUBLIC group IDs do not match submissions")
        result["groups"] = {g: sum(groups[i] == g for i in changed)
                            for g in sorted(set(groups.values()))}
    if not changed:
        result.update(estimate=reference_score, interval=(reference_score, reference_score),
                      source="identical PUBLIC outputs; no TRAIN delta needed")
        return result
    if train_pair is None:
        return result
    evidence = train_deltas(train_pair)
    summary = summarize_delta(evidence["deltas"], evidence["ambiguous"])
    result.update({k: v for k, v in evidence.items() if k != "deltas"})
    result.update(summary, train_changed=len(evidence["deltas"]))
    if summary["d"] is not None:
        result["estimate"] = reference_score + fraction * summary["d"]
        result["interval"] = (tuple(reference_score + fraction * x for x in summary["interval"])
                              if summary["interval"] else None)
    return result


def number(value):
    return NOT_MEASURED if value is None else f"{value:.3f}"


def interval_text(value):
    return NOT_MEASURED if value is None else f"[{value[0]:.3f}, {value[1]:.3f}]"


def table(rows):
    lines = ["| candidate | reference | n_diff | d (points) | estimate | ~95% SE envelope | data source |",
             "|---|---|---:|---:|---:|---|---|"]
    for candidate, reference, result in rows:
        source = result["source"]
        if "train_changed" in result:
            source += f"; changed={result['train_changed']}, ambiguous={result['ambiguous']}"
        lines.append(f"| {candidate} | {reference} | {result['n_diff']} | {number(result['d'])} | "
                     f"{number(result['estimate'])} | {interval_text(result['interval'])} | {source} |")
    return "\n".join(lines)


def run(args):
    repo, scored = safe_path(args.repo_root), safe_path(args.scored_dir)
    groups = public_groups(args.data_root)
    paths = {"v3": repo / "outputs/colab/public_drive/public/v3/submission.jsonl",
             "v3_asrft": repo / "outputs/colab/public_drive/public/v3_asrft/submission.jsonl",
             "mixed": safe_path(args.mixed)}
    paths.update({v: scored / f"pub_{v}.jsonl" for v in ("v6", "v7", "v8a", "v8b")})
    paths = {v: safe_path(p) for v, p in paths.items()}
    public = {v: read_public(p, groups) for v, p in paths.items() if p.is_file()}
    train = {}
    # Exact allowlist; no directory listing or recursive dataset search.
    for v, stem in (("v3_asrft", "v3asrft"), ("v4", "v4"), ("v6", "v6"),
                    ("v7", "v7"), ("v8a", "v8a"), ("v8b", "v8b")):
        train[v] = {}
        for split in ("val", "smoke"):
            path = safe_path(scored / f"{split}_{stem}.json")
            if path.is_file():
                train[v][split] = (path, read_train(path))

    def compare(c, r, train_candidate=None):
        if c not in public or r not in public:
            return {"n_diff": NOT_MEASURED, "d": None, "estimate": None, "interval": None,
                    "source": "NOT MEASURED: PUBLIC file missing", "groups": {}}
        pair = select_train(train.get(train_candidate or c, {}), train.get(r, {}))
        return estimate(public[c], public[r], KNOWN[r], pair, groups)

    backtests = [("mixed", "v3_asrft", compare("mixed", "v3_asrft", "v4")),
                 ("v7", "v3_asrft", compare("v7", "v3_asrft"))]
    candidates = [(c, r, compare(c, r)) for c in ("v6", "v8a", "v8b") for r in KNOWN]
    print("All estimates below are offline extrapolations, NOT measured portal scores.")
    print("Known portal anchors: v3=27.98; v3_asrft=42.25; mixed=42.07; v7=43 (user-reported; rounding unknown).")
    print("Formula: anchor + n_diff/634 * mean paired TRAIN score delta in percentage points.")
    print("PUBLIC equality requires identical ordered calls AND state_hash (JSON object-key order ignored).")
    print("TRAIN dumps omit hashes: point d uses identified changes; ambiguous equal-call/false-match pairs add zero deltas.")
    print("Intervals: normal 1.96*SE envelopes across all ambiguous denominators; small-sample approximation only.")
    print("They omit PUBLIC domain shift, local-scorer discrepancy, model selection and anchor rounding; not calibrated prediction intervals.")
    print("\nBack-tests (fixed method; no fitting to known outcomes):")
    print(table(backtests))
    errors = []
    for c, _, result in backtests:
        error = None if result["estimate"] is None else result["estimate"] - KNOWN[c]
        print(f"{c}: observed={KNOWN[c]}; signed estimate-minus-observed error={number(error)} points.")
        if error is not None:
            errors.append(abs(error))
    print("mixed uses v4-vs-v3_asrft TRAIN deltas only as the requested replacement-policy back-test.")
    print("\nCandidates:")
    print(table(candidates))
    print("v3 has no paired TRAIN dump; mixed has no TRAIN run of its mixed policy.")
    print("\nPUBLIC changed-ID strata (installed apps, not inferred target app); TRAIN d sensitivity:")
    print(f"All PUBLIC inputs: { {g: list(groups.values()).count(g) for g in sorted(set(groups.values()))} }")
    for c, r, result in backtests + candidates:
        print(f"{c}/{r}: {result['groups'] or NOT_MEASURED}; d sensitivity="
              f"{interval_text(result.get('d_sensitivity'))}; identified-change SE={number(result.get('se'))}")
    print("Stratified score effect for public-only apps: NOT MEASURED. TRAIN has no such apps; pooled d is extrapolated to them.")
    primary = [result["estimate"] for _, r, result in candidates
               if r == "v3_asrft" and result["estimate"] is not None]
    spread = max(primary) - min(primary) if len(primary) >= 2 else None
    max_error = max(errors) if errors else None
    print(f"\nCommon-anchor candidate spread={number(spread)}; max back-test absolute error={number(max_error)} points.")
    if spread is not None and max_error is not None and max_error > spread:
        print("CANNOT RANK candidates: back-test error exceeds the candidate spread.")
    else:
        print("Ranking reliability: NOT MEASURED; compare intervals and domain-shift limitations before selection.")
    print("v7 back-test error is relative to reported 43; no rounding tolerance was invented.")
    print("\nExact input paths (missing files are NOT MEASURED):")
    for name, path in paths.items():
        print(f"PUBLIC {name}: {path} ({'read' if name in public else NOT_MEASURED})")
    for name, splits in train.items():
        for split, (path, _) in splits.items():
            print(f"TRAIN {name}/{split}: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--scored-dir", type=Path, default=Path("C:/Users/ADMIN/Downloads/scored"))
    parser.add_argument("--mixed", type=Path, default=Path("C:/Users/ADMIN/Downloads/mixed_v4_v3asrft.jsonl"))
    parser.add_argument("--data-root", type=Path, default=Path("../datalife/train+public_test"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
