"""CPU contract checks using synthetic fixtures; runtime checks need jsonschema."""

import builtins
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
DOCKER = ROOT / "docker"
HAS_SCHEMA = importlib.util.find_spec("jsonschema") is not None


def notebook_cells():
    notebook = json.loads((DOCKER / "predict_notebook.ipynb").read_text(encoding="utf-8"))
    return {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code"}


class SubmissionContractTests(unittest.TestCase):
    def test_image_has_self_contained_readme_and_fixed_hash_seed(self):
        dockerfile = (DOCKER / "Dockerfile").read_text(encoding="utf-8")
        script = (DOCKER / "predict.sh").read_text(encoding="utf-8")
        readme = (DOCKER / "README.md").read_text(encoding="utf-8")
        self.assertIn("COPY docker/README.md ./README.md", dockerfile)
        self.assertIn("PYTHONHASHSEED=20261112", dockerfile)
        self.assertRegex(script, re.compile(r"^export PYTHONHASHSEED=20261112$", re.M))
        self.assertIn("--split private_test --data-root /data", script)
        self.assertIn("/result/submission.jsonl", script)
        for required in ("/data/private_test/", "/result/submission.jsonl",
                         "/result/jupyter_submission.jsonl", "9777", "training_code/",
                         "NOT MEASURED", "CUDA"):
            self.assertIn(required, readme)
        jupyter = (DOCKER / "start_jupyter.sh").read_text(encoding="utf-8")
        self.assertIn("--port 9777", jupyter)

    def _run_notebook(self, mismatch):
        cells = notebook_cells()
        for cell in cells.values():
            compile(cell, "predict_notebook.ipynb", "exec")
        with TemporaryDirectory(prefix="lifemate-submission-test-") as directory:
            root = Path(directory)
            for relative in ("data/private_test/zaic_tools", "models/planner", "models/asr", "result"):
                (root / relative).mkdir(parents=True, exist_ok=True)
            for relative in ("private_test.jsonl", "private_test_tools_catalog.json", "zaic_tools/__init__.py"):
                (root / "data/private_test" / relative).write_text("", encoding="utf-8")
            # Opaque synthetic adapter marker: not a generated competition hash.
            record = {"id": "synthetic-notebook-fixture", "tool_calls": [],
                      "state_hash": "test-only-participant-api-marker", "time": 17}
            script_record = {**record, "time": 29}
            if mismatch:
                script_record["tool_calls"] = [{"name": "synthetic.fixture", "arguments": {}}]
            (root / "result/submission.jsonl").write_text(json.dumps(script_record) + "\n", encoding="utf-8")
            real_import = builtins.__import__

            def mapped_path(value):
                return root / str(value).lstrip("/")

            def import_module(name, *args, **kwargs):
                if name == "pathlib":
                    return SimpleNamespace(Path=mapped_path)
                return real_import(name, *args, **kwargs)

            dispatched = []

            def run(command, *, check, env):
                self.assertEqual(command, ["bash", "/code/predict.sh", "/result/jupyter_submission.jsonl"])
                self.assertIs(check, True)
                self.assertEqual(env["PYTHONHASHSEED"], "20261112")
                self.assertEqual(env["HF_HUB_OFFLINE"], "1")
                self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
                self.assertEqual(env["LIFEMATE_PLANNER_MODE"], "native")
                dispatched.append(command)
                (root / "result/jupyter_submission.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

            with patch.dict("os.environ", {}, clear=True), patch.object(builtins, "__import__", import_module), \
                    patch.object(subprocess, "run", run):
                namespace = {}
                for cell_id in ("fixed-seed", "input-paths", "model-paths", "run-predict"):
                    exec(cells[cell_id], namespace)
                self.assertEqual(len(dispatched), 1)
                if mismatch:
                    with self.assertRaises(AssertionError):
                        exec(cells["check-parity"], namespace)
                else:
                    exec(cells["check-parity"], namespace)
                    self.assertEqual(namespace["notebook_records"], [(record["id"], [], record["state_hash"])])

    def test_notebook_setup_and_matching_predictions(self):
        self._run_notebook(mismatch=False)

    def test_notebook_rejects_different_predictions(self):
        self._run_notebook(mismatch=True)

    @unittest.skipUnless(HAS_SCHEMA, "jsonschema is unavailable in this interpreter")
    def test_mounted_split_package_wins_over_an_importable_training_copy(self):
        from src.inference import official
        from src.tooling.registry import ToolRegistry

        with TemporaryDirectory(prefix="lifemate-mount-test-") as directory:
            root = Path(directory)
            split = root / "private_test"
            package = split / "zaic_tools"
            package.mkdir(parents=True)
            (split / "private_test_tools_catalog.json").write_text("{}", encoding="utf-8")
            (package / "__init__.py").write_text("from .participant import ToolEnv\n", encoding="utf-8")
            (package / "participant.py").write_text("class ToolEnv:\n    pass\n", encoding="utf-8")
            name = "_lifemate_release_private_test"
            previous = {key: sys.modules.pop(key, None) for key in (name, name + ".participant")}
            try:
                with patch.object(official, "build_registry", lambda catalog: ToolRegistry(())), \
                        patch.dict(sys.modules, {"zaic_tools": SimpleNamespace(ToolEnv=object)}):
                    kit = official.OfficialKit(root, "private_test")
                    self.assertEqual(Path(kit._release.__file__), package / "__init__.py")
                    self.assertEqual(kit._release.ToolEnv.__module__, name + ".participant")
                    self.assertIs(sys.modules["zaic_tools"].ToolEnv, object)
            finally:
                for key, value in previous.items():
                    sys.modules.pop(key, None)
                    if value is not None:
                        sys.modules[key] = value

    @unittest.skipUnless(HAS_SCHEMA, "jsonschema is unavailable in this interpreter")
    def test_private_record_keeps_public_api_values_and_millisecond_time(self):
        from src.agent.runner import Limits
        from src.inference.pipeline import predict_records
        from src.tooling.registry import ToolRegistry

        calls = []
        state_marker = object()
        opened = []

        class SyntheticKit:
            split = "private_test"
            registry = ToolRegistry(())

            def rows(self, sample_ids):
                self_outer.assertIsNone(sample_ids)
                yield {"id": "synthetic-private-fixture", "audio": "synthetic-audio",
                       "history": [], "context": {"installed_apps": {}}}

            def audio_path(self, sample_id, relative_path, *, history):
                self_outer.assertFalse(history)
                return relative_path

            def open_environment(self, sample):
                opened.append(sample.sample_id)
                return SimpleNamespace(read_output=lambda: (calls, state_marker))

        self_outer = self
        records = list(predict_records(SyntheticKit(),
                                      lambda request: '{"action":"finish","reply":""}',
                                      lambda path: "synthetic request", limits=Limits()))
        self.assertEqual(opened, ["synthetic-private-fixture"])
        self.assertEqual(len(records), 1)
        row = records[0].as_dict()
        self.assertEqual(set(row), {"id", "tool_calls", "state_hash", "time"})
        self.assertIs(row["tool_calls"], calls)
        self.assertIs(row["state_hash"], state_marker)
        self.assertIs(type(row["time"]), int)
        self.assertLessEqual(0, row["time"])
        self.assertLessEqual(row["time"], 30_000)

    @unittest.skipUnless(HAS_SCHEMA, "jsonschema is unavailable in this interpreter")
    def test_writer_creates_missing_result_directory(self):
        from src.inference.pipeline import write_official_records

        with TemporaryDirectory(prefix="lifemate-output-test-") as directory:
            output = Path(directory) / "result" / "nested" / "submission.jsonl"
            self.assertEqual(write_official_records(output, (), allow_empty=True), 0)
            self.assertTrue(output.is_file())
            self.assertEqual(output.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
