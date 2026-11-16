"""Notebook/API contract checks with synthetic fixtures and no real checkpoints."""

import ast
import builtins
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
HAS_SCHEMA = importlib.util.find_spec("jsonschema") is not None


def notebook_cells():
    notebook = json.loads((ROOT / "docker/predict_notebook.ipynb").read_text(encoding="utf-8"))
    return [(cell["id"], "".join(cell["source"])) for cell in notebook["cells"]
            if cell["cell_type"] == "code"]


class NotebookCellTests(unittest.TestCase):
    def test_code_compiles_and_api_calls_have_distinct_cells_in_contract_order(self):
        cells = notebook_cells()
        self.assertEqual([cell_id for cell_id, _ in cells],
                         ["fixed-seed", "model-paths", "input-paths", "run-predict", "check-parity"])
        calls = {}
        for cell_id, source in cells:
            compile(source, "predict_notebook.ipynb:" + cell_id, "exec")
            tree = ast.parse(source)
            calls[cell_id] = {node.func.id for node in ast.walk(tree)
                              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
            self.assertNotIn("subprocess", source)
        self.assertIn("seed_runtime", calls["fixed-seed"])
        self.assertIn("load_models", calls["model-paths"])
        self.assertNotIn("run_private_test", calls["model-paths"])
        self.assertIn("run_private_test", calls["run-predict"])
        self.assertNotIn("load_models", calls["run-predict"])

    def _execute_notebook(self, *, mismatch=False, script_present=True):
        with TemporaryDirectory(prefix="lifemate-notebook-api-") as directory:
            fixture_root = Path(directory)
            for relative in ("models/planner", "models/asr", "data/private_test/zaic_tools", "result"):
                (fixture_root / relative).mkdir(parents=True, exist_ok=True)
            for relative in ("private_test.jsonl", "private_test_tools_catalog.json", "zaic_tools/__init__.py"):
                (fixture_root / "data/private_test" / relative).write_text("", encoding="utf-8")
            # Opaque synthetic adapter value, never a generated competition hash.
            record = {"id": "synthetic-notebook-fixture", "tool_calls": [],
                      "state_hash": "test-only-participant-api-marker", "time": 17}
            if script_present:
                script_record = {**record, "time": 29}
                if mismatch:
                    script_record["tool_calls"] = [{"name": "synthetic.fixture", "arguments": {}}]
                (fixture_root / "result/submission.jsonl").write_text(
                    json.dumps(script_record) + "\n", encoding="utf-8")
            events = []
            namespace = {}
            loaded = SimpleNamespace(load_ms=13)

            def seed_runtime():
                import os
                self.assertEqual(os.environ["PYTHONHASHSEED"], "20261112")
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
                self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
                events.append("seed")

            def load_models(planner_dir, asr_dir, *, device, planner_mode):
                self.assertEqual(events, ["seed"])
                self.assertNotIn("input_split", namespace)
                self.assertEqual(planner_dir, fixture_root / "models/planner")
                self.assertEqual(asr_dir, fixture_root / "models/asr")
                self.assertEqual((device, planner_mode), ("cuda", "native"))
                events.append("load")
                return loaded

            def run_private_test(models, *, data_root, output):
                self.assertEqual(events, ["seed", "load"])
                self.assertEqual(namespace["input_split"], fixture_root / "data/private_test")
                self.assertIs(models, loaded)
                self.assertEqual(data_root, "/data")
                self.assertEqual(output, "/result/jupyter_submission.jsonl")
                (fixture_root / output.lstrip("/")).write_text(json.dumps(record) + "\n", encoding="utf-8")
                events.append("inference")
                return {"records": 1, "model_load_ms": loaded.load_ms}

            real_import = builtins.__import__

            def import_module(name, *args, **kwargs):
                if name == "pathlib":
                    return SimpleNamespace(Path=lambda value: fixture_root / str(value).lstrip("/"))
                if name == "src.inference.api":
                    return SimpleNamespace(seed_runtime=seed_runtime, load_models=load_models,
                                           run_private_test=run_private_test)
                return real_import(name, *args, **kwargs)

            output = io.StringIO()
            with patch.dict("os.environ", {}, clear=True), \
                    patch.object(builtins, "__import__", import_module), redirect_stdout(output):
                for cell_id, source in notebook_cells():
                    if cell_id == "check-parity" and mismatch:
                        with self.assertRaises(AssertionError):
                            exec(source, namespace)
                    else:
                        exec(source, namespace)
            self.assertEqual(events, ["seed", "load", "inference"])
            self.assertIn("Model load: 13 ms", output.getvalue())
            self.assertEqual(namespace["notebook_records"], [(record["id"], [], record["state_hash"])])

    def test_notebook_executes_with_matching_predictions_and_different_times(self):
        self._execute_notebook()

    def test_notebook_rejects_different_predictions(self):
        self._execute_notebook(mismatch=True)

    def test_notebook_accepts_absent_script_output(self):
        self._execute_notebook(script_present=False)


@unittest.skipUnless(HAS_SCHEMA, "jsonschema is unavailable in this interpreter")
class SharedRuntimeTests(unittest.TestCase):
    def test_api_and_cli_write_identical_predictions_and_load_once_per_run(self):
        from src.agent import local_model
        from src.inference import api
        from src.inference import __main__ as cli
        from src.inference.mock import MockNotes, _TOOLS, _mock_planner
        from src.tooling.registry import ToolRegistry

        loaded = []
        seeded = []
        kits = []

        class FakePlanner:
            parameter_count = 42

            def __init__(self, path, *, device):
                loaded.append(("planner", Path(path), device))

            def __call__(self, request):
                return _mock_planner(request)

            def chat(self, messages, tools, *, max_tokens, deadline):
                results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
                if not results:
                    call = {"name": "mock_notes.list", "arguments": {}}
                elif len(results) == 1:
                    call = {"name": "mock_notes.append", "arguments": {
                        "note_id": results[0]["notes"][0]["id"], "text": " extra"}}
                else:
                    return "Done."
                return "<tool_call>\n" + json.dumps(call) + "\n</tool_call>"

        class FakeWhisper:
            parameter_count = 7

            def __init__(self, path, *, device):
                loaded.append(("asr", Path(path), device))

            def __call__(self, path):
                return "Duplicate attempt" if str(path).endswith("2") else "Append"

        class SyntheticKit:
            registry = ToolRegistry(_TOOLS)

            def __init__(self, root, split):
                self.split = split
                self.opened = []
                kits.append(self)

            def rows(self, sample_ids):
                self_outer.assertIsNone(sample_ids)
                for sample_id in ("synthetic-1", "synthetic-2"):
                    yield {"id": sample_id, "audio": sample_id, "history": [], "context": {
                        "installed_apps": {"mock_notes": {"tools": ["mock_notes.list", "mock_notes.append"]}}}}

            def audio_path(self, sample_id, relative_path, *, history):
                self_outer.assertFalse(history)
                return relative_path

            def open_environment(self, sample):
                self_outer.assertNotIn(sample.sample_id, self.opened)
                self.opened.append(sample.sample_id)
                notes = MockNotes("synthetic-note-" + sample.sample_id)
                calls = []

                def call(name, arguments):
                    calls.append({"name": name, "arguments": arguments})
                    return notes.call(name, arguments)

                return SimpleNamespace(call=call, read_output=lambda: (
                    calls, "test-only-participant-api-marker"))

        self_outer = self
        with TemporaryDirectory(prefix="lifemate-api-cli-") as directory, \
                patch.object(cli, "OfficialKit", SyntheticKit), \
                patch.object(local_model, "LocalPlanner", FakePlanner), \
                patch.object(local_model, "LocalWhisper", FakeWhisper), \
                patch.object(local_model, "initialize_runtime", lambda seed=api.DEFAULT_SEED: seeded.append(seed)):
            for mode in ("direct", "native"):
                with self.subTest(planner_mode=mode):
                    loaded.clear()
                    seeded.clear()
                    kits.clear()
                    model_dir = Path(directory) / "planner"
                    asr_dir = Path(directory) / "asr"
                    api_output = Path(directory) / mode / "jupyter_submission.jsonl"
                    cli_output = Path(directory) / mode / "submission.jsonl"
                    models = api.load_models(model_dir, asr_dir, device="cpu", planner_mode=mode)
                    self.assertEqual(len(loaded), 2)
                    summary = api.run_private_test(models, data_root=directory, output=api_output)
                    self.assertEqual(len(loaded), 2, "Inference must reuse the existing models")
                    with redirect_stdout(io.StringIO()):
                        code = cli.main(["official", "--split", "private_test", "--data-root", directory,
                                         "--model-dir", str(model_dir), "--asr-dir", str(asr_dir),
                                         "--device", "cpu", "--planner-mode", mode, "--output", str(cli_output)])
                    self.assertEqual(code, 0)
                    self.assertEqual(summary["records"], 2)
                    self.assertEqual(len(loaded), 4)
                    self.assertEqual(seeded, [api.DEFAULT_SEED, api.DEFAULT_SEED])
                    self.assertEqual([kit.opened for kit in kits], [["synthetic-1", "synthetic-2"]] * 2)
                    # Compare serialized bytes, replacing only the separately measured time values.
                    normalized = lambda path: re.sub(rb'"time": [0-9]+', b'"time": <measured>', path.read_bytes())
                    self.assertEqual(normalized(api_output), normalized(cli_output))
                    records = [json.loads(line) for line in api_output.read_text(encoding="utf-8").splitlines()]
                    self.assertTrue(all(len(row["tool_calls"]) == 2 for row in records))
                    self.assertTrue(all(set(row) == {"id", "tool_calls", "state_hash", "time"} for row in records))


if __name__ == "__main__":
    unittest.main()
