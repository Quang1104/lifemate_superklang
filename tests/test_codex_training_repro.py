"""Static reproducibility checks; no datasets, models, packages or Docker are loaded."""

import ast
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import unittest


ROOT = Path(__file__).resolve().parents[1]


def copy_rules():
    """Read Docker COPY mappings without enumerating directory contents."""
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    workdir = PurePosixPath("/")
    rules = []
    for line in dockerfile.splitlines():
        if not re.match(r"^(?:WORKDIR|COPY)\s", line, re.I):
            continue
        words = shlex.split(line)
        if not words:
            continue
        if words[0].upper() == "WORKDIR":
            location = PurePosixPath(words[1])
            workdir = location if location.is_absolute() else workdir / location
        if words[0].upper() != "COPY":
            continue
        flags = [word for word in words[1:] if word.startswith("--")]
        operands = [word for word in words[1:] if not word.startswith("--")]
        destination = PurePosixPath(operands[-1])
        if not destination.is_absolute():
            destination = workdir / destination
        context = next((flag.split("=", 1)[1] for flag in flags if flag.startswith("--from=")), None)
        for source in operands[:-1]:
            # Docker treats a directory source as its contents. These paths are
            # matched against explicit candidate files; no recursive glob occurs.
            rules.append((context, PurePosixPath(source), destination, operands[-1].endswith("/")))
    return rules


def image_file_list(source_files):
    """Project explicit source files through COPY; inspect only allowlisted files."""
    result = {}
    for candidate in source_files:
        source_file = PurePosixPath(candidate)
        for context, source, destination, destination_is_dir in copy_rules():
            if context is not None:
                continue
            local_source = ROOT.joinpath(*source.parts)
            if local_source.is_file() and source_file == source:
                image_path = destination / source.name if destination_is_dir else destination
            elif local_source.is_dir() and source_file.is_relative_to(source):
                image_path = destination / source_file.relative_to(source)
            else:
                continue
            if ROOT.joinpath(*source_file.parts).is_file():
                result[image_path.as_posix()] = source_file.as_posix()
    return result


class TrainingReproducibilityTests(unittest.TestCase):
    def test_every_documented_code_file_is_in_copy_image_file_list(self):
        readme = (ROOT / "training_code" / "README.md").read_text(encoding="utf-8")
        named_files = set(re.findall(r"`(/code/[^\s`]+)`", readme))
        self.assertTrue(named_files)
        sources = {name.removeprefix("/code/") for name in named_files}
        image_files = image_file_list(sources)
        self.assertEqual(named_files - image_files.keys(), set())
        self.assertIn("/code/scripts/build_sft.py", image_files)
        self.assertIn("/code/research/validation_split.json", image_files)
        # Code/image files above are separate from the explicitly documented
        # mounted TRAIN/base inputs and generated RUN_DIR products.
        for externally_supplied in ("../datalife/train+public_test", "train/train.jsonl",
                                    "train/train_tools_catalog.json", "train/zaic_tools/__init__.py"):
            self.assertIn(externally_supplied, readme)
        self.assertIn("đầu vào ngoài image", readme)
        self.assertIn("sản phẩm được tạo, không phải file được COPY", readme)
        model_destinations = {destination.as_posix() for context, _, destination, _ in copy_rules()
                              if context is not None}
        self.assertEqual(model_destinations, {"/models/planner", "/models/asr"})

    def test_builder_repo_imports_are_copied_without_other_scripts(self):
        tree = ast.parse((ROOT / "scripts" / "build_sft.py").read_text(encoding="utf-8"))
        modules = {node.module for node in ast.walk(tree)
                   if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src.")}
        self.assertTrue(modules)
        sources = {module.replace(".", "/") + ".py" for module in modules}
        files = image_file_list(sources)
        self.assertEqual(set(files.values()), sources)
        copied_script_sources = {source.as_posix() for context, source, _, _ in copy_rules()
                                 if context is None and source.parts[0] == "scripts"}
        self.assertEqual(copied_script_sources, {"scripts/build_sft.py"})

    def test_documented_train_partition_matches_manifest(self):
        manifest = json.loads((ROOT / "research" / "validation_split.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_split"], "train")
        train, validation = set(manifest["train_ids"]), set(manifest["validation_ids"])
        self.assertEqual(len(train), 1236)
        self.assertEqual(len(validation), 309)
        self.assertFalse(train & validation)
        self.assertEqual(len(train | validation), manifest["sample_count"])

    def test_documented_command_uses_current_hyperparameters(self):
        readme = (ROOT / "training_code" / "README.md").read_text(encoding="utf-8")
        train_command = next(line for line in readme.splitlines()
                             if line.startswith("python -B training_code/train_qlora.py "))
        args = shlex.split(train_command)
        supplied = dict(zip(args[3::2], args[4::2]))
        expected = {"--epochs": "1", "--save-every": "25", "--max-tokens": "12000",
                    "--seed": "20261113", "--lr": "1e-4", "--rank": "16",
                    "--alpha": "32", "--grad-accum": "8"}
        for option, value in expected.items():
            self.assertEqual(supplied[option], value)
        tree = ast.parse((ROOT / "training_code" / "train_qlora.py").read_text(encoding="utf-8"))
        defaults = {}
        for call in ast.walk(tree):
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "add_argument" and call.args):
                default = next((kw.value for kw in call.keywords if kw.arg == "default"), None)
                if isinstance(default, ast.Constant):
                    defaults[ast.literal_eval(call.args[0])] = default.value
        for option in ("--max-tokens", "--seed", "--lr", "--rank", "--alpha", "--grad-accum"):
            self.assertEqual(float(supplied[option]), defaults[option])
        self.assertNotIn("--max-steps", supplied)
        self.assertIn("peft bitsandbytes", (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
