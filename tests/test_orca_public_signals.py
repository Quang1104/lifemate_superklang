"""research/public_signals.py: label-free counts over our own public predictions."""

import json
import subprocess
import sys

from research.public_signals import load, read, signals


def catalog(apps):
    return {"apps": [{"id": app, "tools": [{"name": f"{app}.{tool}", "kind": kind} for tool, kind in tools]}
                     for app, tools in apps.items()]}


TRAIN = {"shopping_list": [("get", "read"), ("add", "write")], "sys": [("ask_missing_info", "meta")]}
PUBLIC = {**TRAIN, "grocery": [("search", "read"), ("add_to_cart", "write")]}


def sample(sample_id, apps, history=()):
    return {"id": sample_id, "audio": "a.wav", "history": list(history),
            "context": {"installed_apps": {app: {"tools": []} for app in apps}}}


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


def kit(tmp_path):
    for split, apps in (("train", TRAIN), ("public_test", PUBLIC)):
        (tmp_path / split).mkdir()
        (tmp_path / split / f"{split}_tools_catalog.json").write_text(json.dumps(catalog(apps)), encoding="utf-8")
    used = [{"role": "user", "audio": "u.wav"}, {"role": "assistant", "tool_calls": [call("grocery.search", query="sữa")]}]
    rows = [sample("public_0001", ["grocery", "shopping_list"]), sample("public_0002", ["grocery"], used),
            sample("public_0003", ["shopping_list"]), sample("public_0004", ["grocery", "shopping_list"])]
    (tmp_path / "public_test" / "public_test.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    predictions = {
        "public_0001": [call("shopping_list.add", item="sữa")],  # sibling of the installed grocery app
        "public_0002": [call("grocery.search", query="sữa"), call("grocery.add_to_cart", product_id="p1", quantity=1)],
        "public_0003": [call("shopping_list.get")],  # ends on a read
        "public_0004": [call("grocery.search", query="trứng"),
                        call("sys.ask_missing_info", tool_name="grocery.add_to_cart", params=["quantity"])],
    }
    path = tmp_path / "submission.jsonl"
    path.write_text("".join(json.dumps({"id": key, "tool_calls": value, "state_hash": "x"}) + "\n"
                            for key, value in predictions.items()), encoding="utf-8")
    return path


def test_signals_count_only_what_inputs_and_predictions_show(tmp_path):
    path = kit(tmp_path)
    inputs, kind, unseen = load(tmp_path)
    assert unseen == {"grocery.search", "grocery.add_to_cart"}
    runs = read(path)
    result = signals(inputs, kind, unseen, runs, list(runs))
    assert result["samples"] == "4" and result["empty outputs"] == "0/4 (0.0%)" and result["calls per sample"] == "1.50"
    assert result["samples calling a public-only tool"] == "2/4 (50.0%)"
    assert result["calls to public-only tools"] == "3/6 (50.0%)"
    assert result["ask_missing_info, public-only apps acted on or asked about"] == "1/2 (50.0%)"
    assert result["ask_missing_info, TRAIN apps acted on or asked about"] == "0/2 (0.0%)"
    assert result["last call is a read: public-only app, history did not use it"] == "0/2 (0.0%)"
    assert result["last call is a read: history used a public-only tool"] == "0/1 (0.0%)"
    assert result["last call is a read: no public-only app"] == "1/1 (100.0%)"
    assert result["installed public-only app not called, a TRAIN sibling app called instead"] == "1/1 (100.0%)"
    assert signals(inputs, kind, unseen, runs, ["public_0003"])["samples"] == "1"  # a subset of IDs


def test_command_line_prints_one_column_per_file_on_the_same_ids(tmp_path):
    path = kit(tmp_path)
    done = subprocess.run([sys.executable, "-B", "-m", "research.public_signals", str(path), "--compare", str(path),
                           "--first", "2", "--data-root", str(tmp_path)], capture_output=True, text=True, encoding="utf-8")
    assert done.returncode == 0, done.stderr
    lines = done.stdout.strip().splitlines()
    assert lines[0].count("submission.jsonl") == 2 and lines[2] == "| samples | 2 | 2 |"
