"""TRAIN diagnostics with synthetic planners and no dataset/state access.

Without jsonschema, imports use a scoped registry test double: these tests verify
runner/report plumbing, not real schema validation. No dependency is installed.
"""

import importlib
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


HAS_SCHEMA = importlib.util.find_spec("jsonschema") is not None


def _registry_stub():
    registry = ModuleType("src.tooling.registry")

    class ToolRejected(ValueError):
        pass

    registry.ToolRejected = ToolRejected
    registry.ToolRegistry = type("ToolRegistry", (), {})
    registry.ToolSpec = lambda *args, **kwargs: SimpleNamespace()
    return registry


REASON = "Tool arguments failed validation (x-ref)"
RAW = ('<tool_call>{"name":"contacts.get","arguments":'
       '{"contact_id":"unobserved-test-id"}}</tool_call>' + "x" * 500)


class Registry:
    """Synthetic guard fixture returning an existing, value-free reason."""

    def available(self, *args):
        return []

    def authorize(self, *args):
        raise runner.ToolRejected(REASON)

    def evidence_paths(self, name):
        return ()


class Kit:
    registry = Registry()

    def __init__(self, split="train"):
        self.split = split
        self.opened = []

    def rows(self, sample_ids):
        yield {"id": "synthetic-private-fixture", "audio": "synthetic-audio",
               "audio_text": "synthetic request", "history": [],
               "context": {"installed_apps": {"contacts": {"tools": ["contacts.get"]}}}}

    def audio_path(self, *args, **kwargs):
        return Path("synthetic-audio")

    def open_environment(self, sample):
        self.opened.append(sample.sample_id)

        def dispatch(*args):
            raise AssertionError("A rejected call must never dispatch")

        return SimpleNamespace(call=dispatch,
                               read_output=lambda: ([], "test-only-official-value"))


def _args(directory, *, split="train", report=True):
    argv = ["official", "--split", split, "--data-root", str(directory),
            "--output", str(Path(directory) / "submission.jsonl"),
            "--model-dir", str(Path(directory) / "model"),
            "--asr-dir", str(Path(directory) / "asr"), "--planner-mode", "native"]
    if report:
        argv += ["--train-report", str(Path(directory) / "train-report.json")]
    return inference_main._parser().parse_args(argv)


def _models(args, factory):
    return api.ModelHandle(lambda request: None, lambda path: "synthetic request", factory,
                           None, None, 0, args.model_dir, args.asr_dir, args.device,
                           args.planner_mode, False, runner.Limits())


class DiagnosticReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not HAS_SCHEMA:
            # Keep all imports and runtime assertions inside this test-only scope.
            imports = patch.dict(sys.modules, {"src.tooling.registry": _registry_stub()})
            imports.start()
            cls.addClassCleanup(imports.stop)
            print("jsonschema unavailable: scoped synthetic registry; real schema validation NOT MEASURED.",
                  flush=True)
        global runner, inference_main, pipeline, api, native, inference_types
        runner, inference_main, pipeline, api, native, inference_types = (
            importlib.import_module(name) for name in (
                "src.agent.runner", "src.inference.__main__", "src.inference.pipeline",
                "src.inference.api", "src.agent.native_planner", "src.inference.types"))

    def test_native_xref_then_repeat_reports_ordered_reason_and_raw_text(self):
        requests = []
        sessions = []

        def chat(*args, **kwargs):
            requests.append((args, kwargs))
            return RAW

        def factory():
            planner = native.NativePlanner(chat, max_tokens=123)
            sessions.append(planner)
            return planner

        with TemporaryDirectory() as directory:
            args = _args(directory)
            kit = Kit()
            with patch.object(inference_main, "OfficialKit", return_value=kit):
                inference_main._run_official(args, models=_models(args, factory))
            report = json.loads(args.train_report.read_text(encoding="utf-8"))["samples"][0]
            official = json.loads(Path(args.output).read_text(encoding="utf-8"))
        self.assertEqual(report["guard_rejections"], [{"tool": "contacts.get", "reason": REASON}])
        self.assertEqual(report["last_planner_output"], RAW[:400])
        self.assertEqual(len(report["last_planner_output"]), 400)
        self.assertEqual((report["steps"], report["dispatched_calls"], report["rejections"]), (2, 0, 1))
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(kwargs["max_tokens"] == 123 for _, kwargs in requests))
        self.assertEqual(len(sessions), 1)
        self.assertEqual(kit.opened, ["synthetic-private-fixture"])
        self.assertEqual(set(official), {"id", "tool_calls", "state_hash"})

    def test_direct_output_is_bounded_and_guard_order_preserved(self):
        texts = [json.dumps({"action": "call", "tool": tool, "arguments": {"value": "private-value"}})
                 for tool in ("contacts.get", "contacts.search")]
        texts.append(json.dumps({"action": "finish", "reply": "k" * 600}))
        iterator = iter(texts)
        sample = inference_types.Sample("synthetic-direct-fixture", "test request", frozenset({"contacts"}))
        result = runner.run_sample(sample, Registry(), lambda request: next(iterator),
                                   lambda sample: SimpleNamespace())
        self.assertEqual(result.guard_rejections, (
            {"tool": "contacts.get", "reason": REASON},
            {"tool": "contacts.search", "reason": REASON},
        ))
        self.assertNotIn("private-value", json.dumps(result.guard_rejections))
        self.assertEqual(result.last_planner_output, texts[-1][:400])
        self.assertEqual((result.status, result.steps, result.dispatched_calls, result.rejections),
                         ("finished", 3, 0, 2))

    def test_dispatched_call_omits_last_text_from_train_report(self):
        class ReadRegistry(Registry):
            def authorize(self, *args):
                return SimpleNamespace(writes=False, evidence_paths=())

        class ReadKit(Kit):
            registry = ReadRegistry()

            def open_environment(self, sample):
                return SimpleNamespace(call=lambda *args: {"ok": True},
                                       read_output=lambda: ([{"name": "contacts.get", "arguments": {}}],
                                                            "test-only-official-value"))

        actions = iter(['{"action":"call","tool":"contacts.get","arguments":{}}',
                        '{"action":"finish","reply":"done"}'])
        with TemporaryDirectory() as directory:
            args = _args(directory)
            with patch.object(inference_main, "OfficialKit", return_value=ReadKit()):
                inference_main._run_official(args, models=_models(args, lambda: lambda request: next(actions)))
            row = json.loads(args.train_report.read_text(encoding="utf-8"))["samples"][0]
        self.assertEqual(row["dispatched_calls"], 1)
        self.assertEqual(row["guard_rejections"], [])
        self.assertNotIn("last_planner_output", row)

    def test_private_output_matches_pre_change_bytes(self):
        sessions = []

        def factory():
            planner = native.NativePlanner(lambda *args, **kwargs: RAW)
            sessions.append(planner)
            return planner

        with TemporaryDirectory() as directory:
            args = _args(directory, split="private_test", report=False)
            with patch.object(inference_main, "OfficialKit", return_value=Kit("private_test")), \
                    patch.object(pipeline, "monotonic", return_value=100):
                inference_main._run_official(args, models=_models(args, factory))
            output = Path(args.output).read_bytes()
            self.assertFalse((Path(directory) / "train-report.json").exists())
        self.assertEqual(output,
                         b'{"id": "synthetic-private-fixture", "tool_calls": [], '
                         b'"state_hash": "test-only-official-value", "time": 0}\n')
        self.assertFalse(hasattr(sessions[0], "last_planner_output"))

    def test_private_rejects_train_report_before_opening_kit(self):
        with TemporaryDirectory() as directory:
            args = _args(directory, split="private_test", report=True)
            with patch.object(inference_main, "OfficialKit") as kit:
                with self.assertRaisesRegex(inference_main.KitError, "TRAIN controls"):
                    inference_main._run_official(args)
                kit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
