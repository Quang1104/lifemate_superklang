"""Separate TRAIN evaluation pass: replay permitted labels through official ToolEnv.

Never call this from inference. Never inspect environment paths or state objects.
Only labeled TRAIN calls are replayed; history is intentionally untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.resolve()
    if any(p.endswith("_environment") for p in root.parts):
        raise ValueError("Forbidden root")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("source_split") != "train":
        raise ValueError("Gold replay is TRAIN-only")
    ids = manifest["validation_ids"]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate selected IDs")
    rows = {}
    with (root / "train" / "train.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item["id"] in ids:
                if item["id"] in rows:
                    raise ValueError("Duplicate data ID")
                rows[item["id"]] = item
    if set(rows) != set(ids):
        raise ValueError("Manifest IDs missing from TRAIN")
    # Fresh process, explicitly load the matching mounted split implementation.
    if "zaic_tools" in sys.modules:
        raise RuntimeError("Run gold replay as a separate fresh process")
    sys.dont_write_bytecode = True  # external kit is read-only
    sys.path.insert(0, str(root / "train"))
    sdk = importlib.import_module("zaic_tools")
    if Path(sdk.__file__).resolve() != root / "train" / "zaic_tools" / "__init__.py":
        raise RuntimeError("Wrong official SDK loaded")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures, durations = [], []
    started = time.perf_counter()
    with args.output.open("w", encoding="utf-8") as handle:
        for sample_id in ids:
            tick = time.perf_counter()
            env = sdk.ToolEnv("train", sample_id, root=root)
            expected = rows[sample_id]["label"]["tool_calls"]
            for call in expected:
                result = env.call(call["name"], call["arguments"])
                if not result.get("ok", False) or result.get("done") is False:
                    failures.append({"id": sample_id, "tool": call["name"], "ok": result.get("ok"), "done": result.get("done")})
            calls, final_hash = env.get_tool_calls(), env.get_state_hash()
            if calls != expected:
                raise RuntimeError(f"Official replay log differs from labels for {sample_id}")
            handle.write(json.dumps({"id": sample_id, "tool_calls": calls, "state_hash": final_hash}, ensure_ascii=False) + "\n")
            durations.append(time.perf_counter() - tick)
    report = {"purpose": "offline TRAIN gold-hash reconstruction, separate from inference",
              "dataset": "TRAIN v3.2.0", "samples": len(ids), "total_seconds": time.perf_counter()-started,
              "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "failed_or_not_done_calls": failures, "history_replayed": False,
              "direct_state_access": False, "output": str(args.output)}
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "failed_or_not_done_calls"}, indent=2))
    print("failed_or_not_done_calls", len(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
