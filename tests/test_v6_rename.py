"""--rename-fraction: TRAIN tools get other meaningful names, consistently, in a share disjoint from masking."""

import json
from pathlib import Path
import re

import pytest

from scripts.build_sft import _ACTION_WORDS, _APP_NAMES, _share, mask_tool_names, rename_tool_names

KIT = Path(__file__).resolve().parents[2] / "datalife" / "train+public_test"


def tool(name, properties=None, description=None):
    return {"type": "function", "function": {"name": name, "description": description or f"Mô tả {name}.",
                                             "parameters": {"type": "object", "properties": properties or {}}}}


def item(sample_id="train_0001"):
    return {"id": sample_id, "target_start": 2, "tools": [
        tool("music.play", {"playlist_id": {"type": "string", "x-ref": "music.playlists",
                                            "description": "ID, lấy từ music.list_tracks"}}),
        tool("music.list_tracks"), tool("contacts.search"), tool("sys.ask_missing_info")],
        "messages": [
            {"role": "system", "content": "Ứng dụng đã cài: Nhạc (music), Danh bạ (contacts)\n"
                                          "Ứng dụng KHÔNG cài trên máy này: Thời tiết (weather)\n"
                                          '- "mở nhạc"\n  music.list_tracks{} ⇒ {"ok":true}\n  music.play{"playlist_id":"p1"}'},
            {"role": "user", "content": "mở music.playlists, đừng gọi smart_music.play"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "contacts.search", "arguments": {"query": "Lan"}}}]},
            {"role": "tool", "content": '{"ok":true,"via":"music.play","kind":"music.playlists"}'},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
                "name": "sys.ask_missing_info", "arguments": {"tool_name": "music.play", "params": ["playlist_id"]}}}]}]}


def ids_in(low, high, count=3):
    found = [f"train_{n:04d}" for n in range(5000) if low <= _share(f"train_{n:04d}") < high]
    assert len(found) >= count
    return found[:count]


def test_off_or_outside_the_band_returns_the_same_object():
    sample = item()
    assert rename_tool_names(sample, 0.0) is sample
    for sample_id in ids_in(0.0, 0.3):  # the masked share is never renamed
        sample = item(sample_id)
        assert rename_tool_names(sample, 0.3, skip=0.3) is sample
    for sample_id in ids_in(0.6, 1.0):
        sample = item(sample_id)
        assert rename_tool_names(sample, 0.3, skip=0.3) is sample


def test_every_reference_is_renamed_and_nothing_that_only_contains_a_name():
    sample_id = ids_in(0.3, 0.6, 1)[0]
    original = item(sample_id)
    renamed = rename_tool_names(original, 0.3, skip=0.3)
    play, tracks, search, ask = (entry["function"]["name"] for entry in renamed["tools"])
    app = play.split(".")[0]
    assert ask == "sys.ask_missing_info" and play.split(".")[0] == tracks.split(".")[0] != "music"
    assert app in _APP_NAMES["music"] and search.split(".")[0] in _APP_NAMES["contacts"]
    assert len({play, tracks, search}) == 3
    properties = renamed["tools"][0]["function"]["parameters"]["properties"]["playlist_id"]
    assert properties == {"type": "string", "x-ref": f"{app}.playlists", "description": f"ID, lấy từ {tracks}"}
    system = renamed["messages"][0]["content"]
    assert f"Nhạc ({app})" in system and f"Danh bạ ({search.split('.')[0]})" in system and "Thời tiết (weather)" in system
    assert f"  {tracks}{{}} ⇒" in system and f'  {play}{{"playlist_id":"p1"}}' in system
    assert renamed["messages"][1]["content"] == "mở music.playlists, đừng gọi smart_music.play"
    assert renamed["messages"][2]["tool_calls"][0]["function"]["name"] == search
    assert renamed["messages"][3]["content"] == f'{{"ok":true,"via":"{play}","kind":"music.playlists"}}'
    assert renamed["messages"][4]["tool_calls"][0]["function"]["arguments"] == {"tool_name": play,
                                                                               "params": ["playlist_id"]}
    assert original == item(sample_id)  # the input is not edited
    assert rename_tool_names(item(sample_id), 0.3, skip=0.3) == renamed  # stable


@pytest.mark.skipif(not (KIT / "public_test").is_dir(), reason="needs the released kit next to the repo")
def test_names_stay_valid_unique_and_never_reuse_a_train_or_public_name():
    catalog = json.loads((KIT / "train" / "train_tools_catalog.json").read_text(encoding="utf-8"))
    train_tools = [entry["name"] for app in catalog["apps"] for entry in app["tools"]]
    train_apps = {app["id"] for app in catalog["apps"]}
    public = json.loads((KIT / "public_test" / "public_test_tools_catalog.json").read_text(encoding="utf-8"))
    public_names = {app["id"] for app in public["apps"]} | {entry["name"] for app in public["apps"] for entry in app["tools"]}
    choices = {name for options in _APP_NAMES.values() for name in options}
    assert set(_APP_NAMES) == train_apps - {"sys"} and not choices & train_apps and not choices & public_names
    by_app = {}
    for name in train_tools:
        by_app.setdefault(name.split(".")[0], []).append(name)
    for sample_id in ids_in(0.0, 1.0, 40):
        sample = {"id": sample_id, "messages": [],
                  "tools": [tool(name) for names in by_app.values() for name in names]}
        renamed = [entry["function"]["name"] for entry in rename_tool_names(sample, 1.0)["tools"]]
        assert len(set(renamed)) == len(renamed)
        for old, new in zip(train_tools, renamed):
            assert (new == old) is old.startswith("sys.")
            if new == old:
                continue
            assert re.fullmatch(r"[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*", new) and new not in train_tools
            assert new not in public_names and new.split(".")[0] not in public_names
    words = {word for options in _ACTION_WORDS.values() for word in options}
    assert all(re.fullmatch(r"[a-z][a-z_]*", word) for word in words | choices)


def test_masking_and_renaming_take_disjoint_shares():
    seen = set()
    for sample_id in ids_in(0.0, 1.0, 60):
        original = item(sample_id)
        masked = mask_tool_names(original, 0.3)
        renamed = rename_tool_names(masked, 0.3, skip=0.3)
        assert not (masked is not original and renamed is not masked)  # never both
        seen.add(("masked" if masked is not original else "") + ("renamed" if renamed is not masked else ""))
    assert seen == {"masked", "renamed", ""}
