"""Function masking replaces whole tool names only, wherever they are written, and nothing that merely contains one."""

import json

from scripts.build_sft import mask_tool_names


def tool(name, properties):
    return {"type": "function", "function": {"name": name, "description": f"Xem thêm {name}.",
                                             "parameters": {"type": "object", "properties": properties}}}


ITEM = {"id": "train_0001", "target_start": 2, "tools": [
            tool("music.play", {"playlist_id": {"type": "string", "x-ref": "music.playlists",
                                                "description": "ID, lấy từ music.list_tracks"}}),
            tool("music.list_tracks", {}),
            tool("flights.book", {"booking_id": {"type": "string", "x-ref": "flights.bookings"}})],
        "messages": [
            {"role": "system", "content": 'Ví dụ:\n- "mở nhạc"\n  music.list_tracks{} ⇒ {"ok":true}\n  music.play{"playlist_id":"pl_1"}'},
            {"role": "user", "content": "mở music.playlists hay smart_music.play gì đó"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "music.play", "arguments": {"playlist_id": "pl_1"}}}]},
            {"role": "tool", "content": '{"ok":true,"via":"music.play","kind":"music.playlists"}'}]}


def test_whole_names_are_masked_everywhere_and_longer_strings_are_left_alone():
    masked = mask_tool_names(ITEM, 1.0)
    play, tracks, book = (entry["function"]["name"] for entry in masked["tools"])
    assert len({play, tracks, book}) == 3 and all(name.startswith("fn_") for name in (play, tracks, book))
    properties = masked["tools"][0]["function"]["parameters"]["properties"]["playlist_id"]
    assert properties["x-ref"] == "music.playlists" and properties["description"] == f"ID, lấy từ {tracks}"
    assert masked["tools"][2]["function"]["parameters"]["properties"]["booking_id"]["x-ref"] == "flights.bookings"
    assert masked["tools"][0]["function"]["description"] == f"Xem thêm {play}."
    assert masked["messages"][0]["content"] == (
        f'Ví dụ:\n- "mở nhạc"\n  {tracks}{{}} ⇒ {{"ok":true}}\n  {play}{{"playlist_id":"pl_1"}}')
    assert masked["messages"][1]["content"] == "mở music.playlists hay smart_music.play gì đó"
    assert masked["messages"][2]["tool_calls"][0]["function"]["name"] == play
    assert masked["messages"][3]["content"] == f'{{"ok":true,"via":"{play}","kind":"music.playlists"}}'
    assert ITEM["tools"][0]["function"]["name"] == "music.play"  # the input is not edited


def test_unmasked_samples_and_samples_with_only_sys_tools_come_back_unchanged():
    assert mask_tool_names(ITEM, 0.0) is ITEM
    only_sys = {"id": "train_0002", "tools": [tool("sys.ask_missing_info", {})], "messages": []}
    assert mask_tool_names(only_sys, 1.0) == only_sys
    assert json.dumps(mask_tool_names(ITEM, 1.0), ensure_ascii=False) == json.dumps(
        mask_tool_names(ITEM, 1.0), ensure_ascii=False)
