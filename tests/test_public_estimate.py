"""Synthetic files stay in memory: tests create no extra filesystem artifacts."""

import io
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from research import public_estimate as pe


def public_row(sample_id="synthetic-1", calls=None, state="synthetic-state-a"):
    return {"id": sample_id, "tool_calls": calls or [], "state_hash": state}


def train_row(sample_id, score, calls=None, match=True, **extra):
    return {"id": sample_id, "score": score, "predicted": calls or [],
            "state_match": match, "gold": [], **extra}


class PublicEstimateTests(unittest.TestCase):
    def load_public(self, rows, expected=("synthetic-1",)):
        contents = "\n".join(json.dumps(row) for row in rows)
        with patch.object(Path, "open", return_value=io.StringIO(contents)):
            return pe.read_public(Path("synthetic.jsonl"), expected)

    def test_synthetic_public_jsonl(self):
        self.assertEqual(len(self.load_public([public_row()])), 1)

    def test_duplicate_missing_and_unexpected_public_ids(self):
        for rows in ([public_row(), public_row()], [], [public_row("synthetic-2")]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.load_public(rows)

    def test_missing_hash_rejected(self):
        row = public_row()
        del row["state_hash"]
        with self.assertRaises(ValueError):
            self.load_public([row])

    def test_empty_public_comparison_rejected(self):
        with self.assertRaises(ValueError):
            pe.changed_public({}, {})

    def test_calls_and_hash_both_required_for_equality(self):
        base = {"synthetic-1": public_row(calls=[{"name": "x", "arguments": {"a": 1, "b": 2}}])}
        reordered = {"synthetic-1": public_row(calls=[{"arguments": {"b": 2, "a": 1}, "name": "x"}])}
        self.assertEqual(pe.changed_public(base, reordered), set())
        reordered["synthetic-1"]["state_hash"] = "synthetic-state-b"
        self.assertEqual(pe.changed_public(base, reordered), {"synthetic-1"})
        reversed_calls = {"synthetic-1": public_row(calls=[{"name": "b"}, {"name": "a"}])}
        ordered_calls = {"synthetic-1": public_row(calls=[{"name": "a"}, {"name": "b"}])}
        self.assertTrue(pe.changed_public(ordered_calls, reversed_calls))

    def test_forbidden_paths_rejected_before_io(self):
        with patch.object(Path, "resolve", side_effect=AssertionError("must not resolve")):
            for path in ("bad_environment/input.json", "State.json", "BAD_ENVIRONMENT/x"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    pe.safe_path(path)

    def test_synthetic_train_json_and_score_validation(self):
        for score in (0.5, -0.1, 1.1, float("nan"), True):
            data = json.dumps({"samples": [train_row("synthetic-1", score)]})
            with patch.object(Path, "open", return_value=io.StringIO(data)):
                if score == 0.5:
                    self.assertEqual(pe.read_train("synthetic.json")["synthetic-1"]["score"], 0.5)
                else:
                    with self.assertRaises(ValueError):
                        pe.read_train("synthetic.json")

    def test_missing_hash_ambiguity_not_false_equality(self):
        c = {"a": train_row("a", 1, match=True), "b": train_row("b", 0, match=False),
             "c": train_row("c", 1, match=True)}
        r = {"a": train_row("a", 0, match=False), "b": train_row("b", 0, match=False),
             "c": train_row("c", 1, match=True)}
        result = pe.train_deltas((c, r, list(c), "synthetic"))
        self.assertEqual(result["deltas"], [100])
        self.assertEqual(result["ambiguous"], 1)
        self.assertEqual(pe.summarize_delta(result["deltas"], 1)["d_sensitivity"], (50, 100))

    def test_real_hashes_resolve_ambiguous_pair(self):
        c = {"a": train_row("a", 0, match=False, state_hash="synthetic-a")}
        r = {"a": train_row("a", 0, match=False, state_hash="synthetic-b")}
        result = pe.train_deltas((c, r, ["a"], "synthetic"))
        self.assertEqual(result["deltas"], [0])
        self.assertEqual(result["ambiguous"], 0)

    def test_gold_and_same_output_score_inconsistency_rejected(self):
        c = {"a": train_row("a", 1)}
        r = {"a": train_row("a", 0)}
        with self.assertRaises(ValueError):
            pe.train_deltas((c, r, ["a"], "synthetic"))
        r["a"]["gold"] = [{"name": "different-target"}]
        with self.assertRaises(ValueError):
            pe.train_deltas((c, r, ["a"], "synthetic"))

    def test_val_preferred_then_complete_smoke_overlap(self):
        val = {i: {} for i in ("a", "b", "c")}
        smoke = {"a": {}}
        c = {"val": ("val_c.json", val), "smoke": ("smoke_c.json", smoke)}
        r = {"val": ("val_r.json", val), "smoke": ("smoke_r.json", smoke)}
        self.assertEqual(pe.select_train(c, r)[2], ["a", "b", "c"])
        self.assertEqual(pe.select_train({"val": c["val"]}, {"smoke": r["smoke"]})[2], ["a"])
        with self.assertRaises(ValueError):
            pe.select_train({"val": c["val"]}, {"smoke": ("bad.json", {"z": {}})})
        self.assertIsNone(pe.select_train(c, {}))

    def test_formula_percentage_scale_and_strata(self):
        cpub = {i: public_row(i) for i in ("a", "b", "c", "d")}
        rpub = {i: public_row(i) for i in cpub}
        cpub["a"]["state_hash"] = "synthetic-changed"
        ctrain = {"x": train_row("x", 0.8, [{"name": "c"}]),
                  "y": train_row("y", 0.6, [{"name": "c"}])}
        rtrain = {"x": train_row("x", 0.4), "y": train_row("y", 0.4)}
        groups = {i: "public-only-app" if i == "a" else "TRAIN-app" for i in cpub}
        result = pe.estimate(cpub, rpub, 42.25, (ctrain, rtrain, ["x", "y"], "synthetic"), groups)
        self.assertAlmostEqual(result["d"], 30)
        self.assertAlmostEqual(result["estimate"], 49.75)
        self.assertAlmostEqual(result["se"], 10)
        self.assertEqual(result["groups"], {"TRAIN-app": 0, "public-only-app": 1})
        self.assertAlmostEqual(result["interval"][0], 42.25 + 0.25 * (30 - 1.96 * 10))

    def test_identical_public_needs_no_train_but_missing_pair_is_not_zero(self):
        rows = {"a": public_row("a")}
        result = pe.estimate(rows, rows, 43)
        self.assertEqual(result["estimate"], 43)
        self.assertEqual(result["interval"], (43, 43))
        changed = {"a": public_row("a", state="synthetic-other")}
        self.assertIsNone(pe.estimate(changed, rows, 43)["estimate"])

    def test_empty_and_singleton_train_uncertainty(self):
        self.assertIsNone(pe.summarize_delta([])["d"])
        self.assertIsNone(pe.summarize_delta([50])["interval"])
        self.assertIsNone(pe.summarize_delta([50], 3)["interval"])

    def test_sensitivity_envelope_includes_all_denominators(self):
        result = pe.summarize_delta([100, -20, 50], 4)
        for extra in range(5):
            values = [100, -20, 50] + [0] * extra
            center = sum(values) / len(values)
            se = math.sqrt(sum((x - center) ** 2 for x in values) / (len(values) - 1) / len(values))
            self.assertLessEqual(result["interval"][0], center - 1.96 * se + 1e-12)
            self.assertGreaterEqual(result["interval"][1], center + 1.96 * se - 1e-12)


if __name__ == "__main__":
    unittest.main()
