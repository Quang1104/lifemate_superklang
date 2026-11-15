"""Build QLoRA training conversations from TRAIN gold calls in the native planner's chat format.

Each sample is rendered with the same build_messages/tool_definitions used at inference: the
gold calls are replayed through one official ToolEnv, each becomes an assistant tool call
followed by its real tool result, and a short closing reply ends the turn. The 309 validation
IDs (and any other --exclude manifest) never enter the training file.

--transcripts replaces the released TRAIN transcripts of the user turns with recorded ASR text
(scripts/asr_transcripts.py), so training sees what inference sees. --fewshot-k adds retrieved
TRAIN examples to every prompt through the same code and pool file inference uses. --mask-fraction and
--rename-fraction rename the tools of disjoint shares of samples (opaque ids / other meaningful names).
--history-turns adds a share of the assistant turns recorded in each sample's history as extra rows: the earlier
turns are the history, that turn's user audio (ASR via --transcripts) is the request, and its recorded calls and
results are the target, so no ToolEnv is needed.
"""

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
import re
from pathlib import Path
from time import monotonic

from jsonschema import Draft202012Validator

from src.agent.asr_fix import wrapped
from src.agent.fewshot import NATIVE_POOL, RESULT_CAP, configured
from src.agent.native_planner import build_messages, tool_definitions
from src.agent.planner import Observation, PlannerInput
from src.agent.runner import CallbackError
from src.inference.official import OfficialKit, installed_permissions
from src.inference.pipeline import _sample_from_row
from src.tooling.registry import ToolRejected

DONE = "Đã xong."
NOTHING = "Mình không thực hiện thao tác nào cho yêu cầu này."


def conversation(sample, tools: list[dict], calls: list[dict], payloads: list[dict]) -> dict:
    observations = tuple(Observation("untrusted_tool_result", call["name"], payload, call["arguments"])
                         for call, payload in zip(calls, payloads))
    messages = build_messages(PlannerInput(sample, tools, observations, None))
    target_start = max(i for i, message in enumerate(build_messages(PlannerInput(sample, tools, (), None)))
                       if message["role"] == "user") + 1
    messages.append({"role": "assistant", "content": DONE if calls else NOTHING})
    return {"id": sample.sample_id, "messages": messages, "tools": tool_definitions(tools),
            "target_start": target_start}


def _share(sample_id: str) -> float:
    """Stable position of a sample in [0, 1]; masking takes [0, mask), renaming [mask, mask + rename)."""
    return int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


# Meaningful replacement names, written from TRAIN app and tool names only (never from a test catalog). None of them
# is a TRAIN app id, so a renamed tool cannot collide with a real one (tests/test_orca_v6_rename.py checks this).
_APP_NAMES = {
    "smart_home": ("home_devices", "house_control"), "home_security": ("security_center", "house_guard"),
    "door_lock": ("smart_lock", "entry_lock"), "robot_vacuum": ("cleaning_robot", "vacuum_bot"),
    "appliances": ("home_machines", "kitchen_devices"), "home_energy": ("power_meter", "energy_usage"),
    "alarm": ("wake_clock", "alarm_clock"), "reminders": ("reminder_center", "nudges"),
    "calendar": ("agenda", "schedule_book"), "notes": ("memo", "notebook"), "tasks": ("todo", "task_board"),
    "contacts": ("address_book", "people"), "phone": ("dialer", "voice_call"),
    "sms": ("text_messages", "short_message"), "chat_ott": ("messenger", "chat_app"), "email": ("mail", "mailbox"),
    "video_meeting": ("video_call", "online_meeting"), "banking": ("bank_account", "bank_app"),
    "ewallet": ("wallet", "mobile_wallet"), "bill_payment": ("invoice_pay", "utility_bills"),
    "loyalty": ("rewards", "member_points"), "weather": ("forecast_service", "weather_service"),
    "gold_price": ("gold_market", "gold_rates"), "fuel_price": ("petrol_price", "gas_station"),
    "lottery": ("lotto", "lucky_draw"), "exchange_rate": ("currency", "fx_rates"),
    "news": ("newsfeed", "headlines_app"), "calculator": ("math_tool", "calc"), "maps": ("navigation", "map_service"),
    "food_delivery": ("meal_order", "food_order"), "ride_hailing": ("taxi", "car_booking"),
    "flights": ("air_travel", "airline"), "shopping_list": ("buy_list", "market_list"),
    "package_tracking": ("parcels", "shipment_tracker"), "music": ("audio", "songs"), "tv": ("television", "tv_box"),
    "podcast_radio": ("audio_shows", "radio_app"), "medication": ("medicine", "pill_box"),
    "fitness": ("workout", "exercise"), "learning": ("courses_app", "study"), "bike_rental": ("bike_share", "cycle_rent"),
    "home_automation": ("automation", "routines"), "household_chores": ("home_jobs", "chore_board"),
    "library": ("book_loans", "library_app"), "subscriptions": ("plans", "memberships"),
}
_ACTION_WORDS = {
    "get": ("fetch", "lookup"), "list": ("show", "browse"), "create": ("new", "make"), "add": ("insert", "put"),
    "update": ("edit", "change"), "delete": ("erase", "drop"), "cancel": ("abort", "revoke"), "search": ("find", "query"),
    "send": ("post", "deliver"), "set": ("adjust", "configure"), "start": ("begin", "launch"), "stop": ("halt", "end"),
    "book": ("secure", "arrange"), "pay": ("settle", "clear"), "transfer": ("remit", "wire"), "call": ("dial", "ring"),
    "play": ("stream", "listen"), "read": ("open", "view"), "save": ("keep", "store"), "check": ("verify", "inspect"),
    "track": ("follow", "trace"), "schedule": ("plan", "arrange"), "log": ("note", "register"),
    "status": ("state", "overview"), "balance": ("funds", "amount"), "events": ("entries", "appointments"),
    "event": ("entry", "appointment"), "bills": ("invoices", "statements"), "message": ("text", "msg"),
    "reply": ("answer", "respond"), "archive": ("stash", "file"), "join": ("enter", "attend"),
    "redeem": ("claim", "exchange"), "convert": ("transform", "change"), "complete": ("finish", "close"),
    "enroll": ("register", "signup"), "reserve": ("hold", "secure"), "assign": ("give", "delegate"),
    "find": ("locate", "discover"), "remove": ("discard", "drop"), "subscribe": ("follow", "signup"),
    "queue": ("enqueue", "line_up"), "tune": ("select", "switch_to"), "record": ("capture", "tape"),
    "control": ("operate", "command"), "arm": ("activate", "enable"), "acknowledge": ("confirm", "dismiss"),
    "grant": ("issue", "give"), "open": ("start", "new"), "topup": ("recharge", "reload"),
    "history": ("past", "records"), "quotes": ("prices", "estimates"), "route": ("path", "directions"),
}
_TOOL_NAME = re.compile(r"[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*")


def rename_tool_names(item: dict, fraction: float, skip: float = 0.0) -> dict:
    """In a stable `fraction` of samples (after the masked share `skip`), give every non-sys tool another meaningful name.

    Training only. Public tools have meaningful names the model never saw; renaming TRAIN tools to plausible new
    names teaches it to follow the installed catalog. The names change everywhere in the sample: tool definitions,
    calls, results, history, examples, "(app)" in the app lists, x-ref kinds and ask_missing_info.tool_name.
    """
    if fraction <= 0 or not skip <= _share(item["id"]) < skip + fraction:
        return item
    names = sorted(tool["function"]["name"] for tool in item["tools"]
                   if not tool["function"]["name"].startswith("sys."))
    if not names:
        return item

    def seed(key: str) -> int:
        return int(hashlib.sha256((item["id"] + key).encode("utf-8")).hexdigest()[:8], 16)

    apps = {}
    for app in sorted({name.split(".")[0] for name in names}):
        options = _APP_NAMES.get(app, (f"{app}_app",))
        apps[app] = options[seed(app) % len(options)]
    taken, mapping = {tool["function"]["name"] for tool in item["tools"]}, {}
    for name in names:
        app, action = name.split(".", 1)
        for shift in range(4):  # another synonym when two tools of one app would get the same name
            words = [_ACTION_WORDS.get(word, (word,)) for word in action.split("_")]
            new = apps[app] + "." + "_".join(options[(seed(name + str(index)) + shift) % len(options)]
                                             for index, options in enumerate(words))
            if new not in taken:
                break
        while new in taken:
            new += "_2"
        assert _TOOL_NAME.fullmatch(new), new
        taken.add(new)
        mapping[name] = new
    whole = re.compile(r"(?<![\w.])(?:" + "|".join(map(re.escape, sorted(mapping, key=len, reverse=True))) + r")(?!\w)")
    listed = re.compile(r"\((" + "|".join(map(re.escape, apps)) + r")\)")

    def renamed(value, key=None):
        if isinstance(value, str):
            app, dot, rest = value.partition(".")
            if key == "x-ref" and dot and app in apps:
                return f"{apps[app]}.{rest}"
            value = whole.sub(lambda match: mapping[match.group()], value)
            return listed.sub(lambda match: f"({apps[match.group(1)]})", value)
        if isinstance(value, list):
            return [renamed(entry) for entry in value]
        if isinstance(value, dict):
            return {name: renamed(entry, name) for name, entry in value.items()}
        return value

    return renamed(item)


def mask_tool_names(item: dict, fraction: float) -> dict:
    """Function masking (Hammer): in a stable `fraction` of samples, rename every non-sys tool to an opaque id.

    Training only. The model must read descriptions instead of memorising names, which is what unseen public
    tools need. Only whole names are replaced: "music.play" inside the x-ref kind "music.playlists" stays.
    """
    digest = lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
    if fraction <= 0 or int(digest(item["id"])[:8], 16) / 0xFFFFFFFF >= fraction:
        return item
    names = sorted((tool["function"]["name"] for tool in item["tools"]
                    if not tool["function"]["name"].startswith("sys.")), key=len, reverse=True)
    if not names:
        return item
    # Longest first, so a name never clobbers a longer one that contains it.
    whole = re.compile(r"(?<![\w.])(?:" + "|".join(map(re.escape, names)) + r")(?!\w)")

    def renamed(value):
        if isinstance(value, str):
            return whole.sub(lambda match: f"fn_{digest(item['id'] + match.group())[:8]}", value)
        if isinstance(value, list):
            return [renamed(entry) for entry in value]
        if isinstance(value, dict):
            return {key: renamed(entry) for key, entry in value.items()}
        return value

    return renamed(item)


def history_turn_rows(kit, row: dict, transcribe, tools: list[dict], fraction: float, totals: Counter) -> list[dict]:
    """SFT rows for a stable `fraction` of the assistant turns recorded in `row`'s history (--history-turns).

    Turn k keeps the history before its user turn, the user turn is the request (ASR when `transcribe` is set, through
    the same asr_fix wrapper inference uses), and the recorded calls with their recorded results are the target. The
    sample keeps the source ID, so retrieved examples never include the sample itself; the row ID gets "_h<k>".
    """
    if fraction <= 0:
        return []
    schemas = {tool["name"]: Draft202012Validator(tool["parameters"]) for tool in tools}
    rows, user = [], None
    for index, turn in enumerate(row["history"]):
        if turn.get("role") == "user":
            user = index
            continue
        if user is None or _share(f"{row['id']}#history{index}") >= fraction:
            continue
        calls = [call for call in turn.get("tool_calls") or () if isinstance(call, dict)]
        results = list(turn.get("tool_results") or ())
        if len(results) < len(calls) or any(
                call.get("name") not in schemas or not isinstance(call.get("arguments"), dict)
                or any(schemas[call["name"]].iter_errors(call["arguments"])) for call in calls):
            totals["history_skipped_invalid_call"] += 1
            continue
        shortened = {**row, "history": row["history"][:user]}
        try:
            sample = _sample_from_row(kit, shortened, transcribe, monotonic() + 3_600, control_noop=False)
            text = row["history"][user]["audio_text"]
            if transcribe is not None:
                audio = kit.audio_path(row["id"], row["history"][user]["audio"], history=True)
                text = wrapped(transcribe, shortened)(audio)
        except (CallbackError, KeyError):  # no usable text for one of the user turns in --transcripts
            totals["history_skipped_no_transcript"] += 1
            continue
        if not isinstance(text, str) or not text.strip():
            totals["history_skipped_no_transcript"] += 1
            continue
        item = conversation(replace(sample, user_request=text.strip()), tools, calls, results[:len(calls)])
        rows.append({**item, "id": f"{row['id']}_h{index}"})
        totals["history_rows"] += 1
        totals["history_rows_without_calls"] += not calls
    return rows


def excluded_ids(manifests: list[Path]) -> set[str]:
    excluded = set()
    for manifest in manifests:
        split = json.loads(manifest.read_text(encoding="utf-8"))
        if split.get("source_split") != "train":
            raise ValueError(f"Expected a TRAIN manifest: {manifest}")
        excluded |= set(split["validation_ids"])
    return excluded


def transcript_lookup(path: Path):
    """Stand-in for ASR: the recorded transcript of each audio file (scripts/asr_transcripts.py output)."""
    with path.open(encoding="utf-8") as lines:
        known = {row["audio"]: row["transcript"] for row in map(json.loads, filter(str.strip, lines))}
    return lambda audio: known[Path(*audio.parts[-3:]).as_posix()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/sft_native.jsonl"))
    parser.add_argument("--exclude", type=Path, nargs="+", default=[Path("research/validation_split.json")])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--transcripts", type=Path,
                        help="JSONL from scripts/asr_transcripts.py; user turns use this ASR text")
    parser.add_argument("--mask-fraction", type=float, default=0.0,
                        help="share of samples whose tool names are masked (training-only augmentation)")
    parser.add_argument("--rename-fraction", type=float, default=0.0,
                        help="share of samples (outside the masked share) whose tools get other meaningful names")
    parser.add_argument("--history-turns", type=float, default=0.0,
                        help="share of history assistant turns that become extra rows (no ToolEnv; recorded results)")
    parser.add_argument("--fewshot-k", type=int, default=0,
                        help="retrieved TRAIN examples per prompt; rewrites the native example pool")
    args = parser.parse_args()
    if any(part.endswith("_environment") for part in args.data_root.resolve().parts):
        raise ValueError("Forbidden root")
    if min(args.mask_fraction, args.rename_fraction) < 0 or args.mask_fraction + args.rename_fraction > 1:
        raise ValueError("--mask-fraction and --rename-fraction must be >= 0 and sum to at most 1")
    if not 0 <= args.history_turns <= 1:
        raise ValueError("--history-turns must be between 0 and 1")
    excluded = excluded_ids(args.exclude)
    kit = OfficialKit(args.data_root, "train")
    reads = {tool["name"] for tool in kit.registry.available(frozenset(kit.app_names))
             if not tool["writes"] and not tool["name"].startswith("sys.")}
    gold, pool = {}, []
    with (kit.directory / "train.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] not in excluded:
                gold[row["id"]] = row["label"]["tool_calls"]
                # Gold results of the read calls, cut to a fixed length; prompts use them only under
                # LIFEMATE_CONTEXT_V4, so the pool stays valid for the plain K-example prompt.
                kept = [json.dumps(result, ensure_ascii=False, separators=(",", ":")) if call["name"] in reads else None
                        for call, result in zip(gold[row["id"]], row["label"]["tool_results"])]
                pool.append({"id": row["id"], "transcript": row["audio_text"],
                             "apps": sorted(installed_permissions(row["context"])[0]), "calls": gold[row["id"]],
                             "results": [text if text is None or len(text) <= RESULT_CAP
                                         else text[:RESULT_CAP] + "…" for text in kept]})
    if args.fewshot_k > 0:
        # Inference reads this same file behind LIFEMATE_NATIVE_FEWSHOT_K; it holds no excluded ID.
        with NATIVE_POOL.open("w", encoding="utf-8", newline='\n') as sink:
            sink.writelines(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + '\n' for row in pool)
        configured.cache_clear()
    os.environ["LIFEMATE_NATIVE_FEWSHOT_K"] = str(args.fewshot_k)  # the CLI decides, not the caller's shell
    transcribe = transcript_lookup(args.transcripts) if args.transcripts else None
    selected = set(sorted(gold)[:args.limit] if args.limit else gold)
    totals, lengths = Counter(), []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as sink:
        for row in kit.rows(selected):
            try:
                sample = _sample_from_row(kit, row, transcribe, monotonic() + 3_600, control_noop=False)
            except CallbackError:  # an audio file with no usable text in --transcripts
                totals["skipped_no_transcript"] += 1
                continue
            tools = kit.registry.available(sample.installed_apps, sample.allowed_tools)
            environment = kit.open_environment(sample)  # the one gold-replay ToolEnv for this sample
            calls, payloads = gold[row["id"]], []
            try:
                for call in calls:
                    payloads.append(environment.call(call["name"], call["arguments"]))
            except ToolRejected:
                totals["skipped_preflight_rejected"] += 1
                continue
            extra = history_turn_rows(kit, row, transcribe, tools, args.history_turns, totals)
            for item in [conversation(sample, tools, calls, payloads), *extra]:
                plain = item
                item = mask_tool_names(item, args.mask_fraction)
                totals["masked"] += item is not plain
                renamed = rename_tool_names(item, args.rename_fraction, args.mask_fraction)
                totals["renamed"] += renamed is not item
                item = renamed
                totals["prompts_with_examples"] += "Ví dụ yêu cầu tương tự" in item["messages"][0]["content"]
                sink.write(json.dumps(item, ensure_ascii=False) + "\n")
                totals["written"] += 1
                lengths.append(len(json.dumps(item, ensure_ascii=False)))
            totals["assistant_calls"] += len(calls)
            totals["empty_gold"] += not calls
    lengths.sort()
    pick = lambda q: lengths[min(len(lengths) - 1, int(q * len(lengths)))] if lengths else None
    print(json.dumps({**totals, "excluded_ids": len(excluded), "chars_p50": pick(0.5), "chars_p95": pick(0.95),
                      "chars_max": lengths[-1] if lengths else None, "tokens": "NOT MEASURED (needs tokenizer)",
                      "transcripts": str(args.transcripts) if args.transcripts else "train", "fewshot_k": args.fewshot_k,
                      "mask_fraction": args.mask_fraction, "rename_fraction": args.rename_fraction,
                      "history_turns": args.history_turns,
                      "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
