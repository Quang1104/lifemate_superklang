"""Synthetic checks for offline prompt-size analysis and its input boundary."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.inference import prompt_size


class SpeedPromptSizeTests(unittest.TestCase):
    def test_reserved_dataset_path_is_rejected_before_open(self):
        with patch.object(Path, "open", side_effect=AssertionError("must not read")):
            with self.assertRaisesRegex(ValueError, "official ToolEnv"):
                prompt_size._train_requests(Path("blocked_environment"), Path("manifest.json"))

    def test_manifest_must_identify_unique_train_samples(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            manifest = base / "smoke.json"
            for data in (
                {"source_split": "public_test", "validation_ids": ["synthetic"]},
                {"source_split": "train", "validation_ids": ["synthetic", "synthetic"]},
                {"source_split": "train", "validation_ids": []},
            ):
                manifest.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "TRAIN manifest"):
                    prompt_size._train_requests(base, manifest)

    def test_tokenizer_is_never_searched_without_explicit_directory(self):
        request = SimpleNamespace(
            sample=SimpleNamespace(
                user_request="Thêm ghi chú", context={"timezone": "Asia/Ho_Chi_Minh"},
                history=('{"role":"user","text":"xin chào"}',),
            ),
            tools=[], observations=(),
        )
        with patch.object(prompt_size, "_train_requests", return_value=([request], {})):
            with patch.object(prompt_size, "_offline_qwen_token_counter",
                              side_effect=AssertionError("must not search tokenizers")):
                report = prompt_size.measure_prompts("unused", "unused")
        self.assertEqual(report["samples"], 1)
        self.assertEqual(report["qwen3_tokens"], "NOT MEASURED")
        self.assertEqual(report["latency"], "NOT MEASURED")
        before = json.loads(prompt_size._legacy_payload(request))
        self.assertEqual(list(before), ["user_request", "context", "history", "tools", "observations"])
        self.assertEqual(before["history"], [{"role": "user", "text": "xin chào"}])

    @unittest.skipUnless(importlib.util.find_spec("jsonschema"), "project dependency jsonschema unavailable")
    def test_synthetic_train_inputs_match_runtime_filtering_without_labels(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            train = base / "train"
            train.mkdir()
            manifest = base / "smoke.json"
            manifest.write_text(json.dumps({
                "source_split": "train", "validation_ids": ["synthetic-selected"],
            }), encoding="utf-8")
            tool = {
                "name": "notes.create", "app": "notes", "kind": "write",
                "description": "Tạo ghi chú",
                "parameters": {
                    "type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"],
                },
                "returns": "ghi chú",
            }
            catalog = {"apps": [{"id": "notes", "tools": [tool, {
                **tool, "name": "notes.delete",
            }]}]}
            (train / "train_tools_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
            row = {
                "id": "synthetic-selected", "audio": "unused.wav", "audio_text": "Thêm ghi chú",
                "context": {
                    "installed_apps": {"notes": {"tools": ["notes.create"]}},
                    "timezone": "Asia/Ho_Chi_Minh", "not_planner_context": "discard",
                },
                "history": [{"role": "user", "audio_text": "Nhắc việc này", "audio": "unused-history.wav"}],
                "label": {"synthetic_only": "must not influence prompt size"},
            }
            data_path = train / "train.jsonl"
            data_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            requests, sources = prompt_size._train_requests(base, manifest)
            self.assertEqual(len(requests), 1)
            request = requests[0]
            self.assertEqual(request.sample.context, {"timezone": "Asia/Ho_Chi_Minh"})
            self.assertEqual(request.sample.user_request, "Thêm ghi chú")
            self.assertEqual([item["name"] for item in request.tools], ["notes.create"])
            self.assertEqual(json.loads(request.sample.history[0]),
                             {"role": "user", "text": "Nhắc việc này"})
            self.assertFalse(request.tools[0]["parameters"]["additionalProperties"])
            self.assertEqual(request.observations, ())
            before = prompt_size.measure_prompts(base, manifest)
            row["label"] = {"synthetic_only": "changed" * 1000}
            data_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            after = prompt_size.measure_prompts(base, manifest)
            self.assertEqual(before, after)
            self.assertEqual(Path(sources["train_input"]), data_path)


if __name__ == "__main__":
    unittest.main()
