"""Opt-in text repair of ASR transcripts from the sample's own context (LIFEMATE_ASR_FIX=1).

PhoWhisper writes lower-case text and spells every number. `fix` changes three things, each chosen on TRAIN
gold transcripts outside the 309 validation IDs (docs/experiments/ASR_FIX.md has the counts):

* number words become digits only inside codes such as RON95, E5, VTV6, which gold always writes that way.
  Gold spells most other numbers ("sáu giờ", "hai trăm nghìn"), so converting them raises WER; they stay.
* a phrase equal to a name the sample context exposes (user profile, assistant turns of the history),
  ignoring case, takes that name's spelling. Names of two or more words also match when only the tone
  marks differ, unless every word heard is an ordinary lower-case word in TRAIN gold.
* a word TRAIN gold capitalises at least 5 times and 10 times as often as not ("Lan", "Zalo") is capitalised.
  A single word gold usually writes lower-case is never touched, so "sáng mai" does not become "sáng Mai".

`python -B -m src.agent.asr_fix --data-root <kit>` is the offline TRAIN oracle measurement; inference never
runs it.
"""

from collections import Counter
from functools import lru_cache
import json
import os
import re
from unicodedata import normalize

from .fewshot import NATIVE_POOL

_UNITS = {"không": 0, "một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5, "sáu": 6, "bảy": 7, "bẩy": 7,
          "tám": 8, "chín": 9}
_AFTER_TENS = {**_UNITS, "mốt": 1, "tư": 4, "lăm": 5, "nhăm": 5}
_NUMBER_WORDS = frozenset(_AFTER_TENS) | {"mười", "mươi", "trăm", "linh", "lẻ"}
_CODES = {"ron": "RON", "e": "E", "vtv": "VTV", "htv": "HTV", "sctv": "SCTV", "vov": "VOV", "rv": "RV"}
_WORD = re.compile(r"\w+")
# Switches for the offline measurement; inference uses all three.
RULES = {"near": True, "lexicon": True, "first_word": True}


def _below_100(words: tuple[str, ...]) -> int | None:
    if len(words) == 1:
        return 10 if words[0] == "mười" else _UNITS.get(words[0])
    tens, last = words[0], _AFTER_TENS.get(words[-1])
    if tens == "mười":
        return 10 + last if len(words) == 2 and last and words[1] != "mốt" else None
    if _UNITS.get(tens, 0) < 2:
        return None
    if words[1:] == ("mươi",):
        return _UNITS[tens] * 10
    if last and (len(words) == 2 or (len(words) == 3 and words[1] == "mươi")):
        return _UNITS[tens] * 10 + last  # "hai mươi sáu", and the spoken short form "hai sáu"
    return None


def number(words: tuple[str, ...]) -> int | None:
    """Value of a complete Vietnamese number phrase below 1000, or None when it is not exactly one number."""
    if not words or not _NUMBER_WORDS.issuperset(words):
        return None
    if "trăm" not in words:
        return _below_100(words)
    hundreds, rest = _UNITS.get(words[0]), words[2:]
    if len(words) < 2 or words[1] != "trăm" or not hundreds or "trăm" in rest:
        return None
    if not rest:
        return hundreds * 100
    if rest[0] in ("linh", "lẻ"):
        tail = _UNITS.get(rest[1]) if len(rest) == 2 else None
        return hundreds * 100 + tail if tail else None
    tail = _below_100(rest)
    return hundreds * 100 + tail if tail and tail >= 10 else None


def _numbers(words: list[str]) -> list[str]:
    """Join a code prefix and the number spoken after it ("ron chín lăm" -> "RON95")."""
    out, i = [], 0
    while i < len(words):
        code = _CODES.get(words[i])
        start = end = i + bool(code)
        while end < len(words) and words[end] in _NUMBER_WORDS:
            end += 1
        value = number(tuple(words[start:end])) if code else None
        if not value:  # not one whole number, or a lone "không", which is usually "not"
            out += words[i:max(end, i + 1)]
        else:
            out.append(f"{code}{value}")
        i = max(end, i + 1)
    return out


def _fold(word: str) -> str:
    """Letters without tone or vowel marks, so "dùng" and "Dũng" compare equal."""
    return "".join(char for char in normalize("NFD", word.lower().replace("đ", "d")) if char.isalpha())


@lru_cache(maxsize=2)
def _casing(skip: frozenset = frozenset()) -> dict[str, tuple[int, int, str]]:
    """Per word inside TRAIN gold requests: times capitalised, times lower-case, usual capitalised spelling.

    Sentence starts are not counted. `skip` leaves sample IDs out, for a held-out measurement.
    """
    counts: dict[str, list] = {}
    if NATIVE_POOL.is_file():
        with NATIVE_POOL.open(encoding="utf-8") as lines:
            for row in map(json.loads, filter(str.strip, lines)):
                if row["id"] in skip:
                    continue
                for match in re.finditer(r"(?<=[^\W\d_] )[^\W\d_]+", normalize("NFC", row["transcript"])):
                    word = match.group()
                    entry = counts.setdefault(word.lower(), [0, 0, Counter()])
                    entry[not word[0].isupper()] += 1
                    if word[0].isupper():
                        entry[2][word] += 1
    return {word: (upper, lower, spellings.most_common(1)[0][0] if spellings else word)
            for word, (upper, lower, spellings) in counts.items()}


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def names(row: dict) -> dict[tuple[str, ...], str]:
    """Lower-cased word tuple -> spelling, for capitalised names in the profile and the assistant's history."""
    sources = list(_strings(row.get("context", {}).get("user_profile")))
    for turn in row.get("history", ()):
        if turn.get("role") == "assistant":  # user turns are ASR output themselves
            sources += _strings({key: turn.get(key) for key in ("text", "tool_calls", "tool_results")})
    found: dict[tuple[str, ...], set[str]] = {}
    for source in sources:
        for sentence in re.split(r"[.!?:;\n]+\s", normalize("NFC", source)):
            words = _WORD.findall(sentence)
            only_names = all(word[0].isupper() for word in words)
            start = None
            for index, word in enumerate(words + [""]):
                if word[:1].isupper() and word.isalpha():
                    start = index if start is None else start
                    continue
                if start is not None:
                    # A sentence opens with a capital whatever the word is; a field holding only a name does not.
                    run = words[start + (start == 0 and not only_names):index]
                    if run:
                        found.setdefault(tuple(item.lower() for item in run), set()).add(" ".join(run))
                start = None
    return {key: next(iter(spellings)) for key, spellings in found.items() if len(spellings) == 1}


def _snap(words: list[str], known: dict[tuple[str, ...], str], casing: dict) -> list[str]:
    """Give names their spelling: context names first (longest match), then names TRAIN gold capitalises."""
    longest = max(map(len, known), default=0)
    folded: dict[tuple[str, ...], list] = {}
    for key in known:
        if len(key) > 1:
            folded.setdefault(tuple(map(_fold, key)), []).append(key)
    out, i = [], 0
    while i < len(words):
        upper, lower, usual = casing.get(words[i], (0, 0, words[i]))
        for size in range(min(longest, len(words) - i), 0, -1):
            phrase = tuple(words[i:i + size])
            spelling = known.get(phrase)
            if (spelling is None and size > 1 and RULES["near"]
                    and not all(casing.get(word, (0, 0))[1] for word in phrase)):
                # Same letters, wrong tones. A phrase of ordinary words ("anh khoá" for "Anh Khoa") is left alone.
                near = folded.get(tuple(map(_fold, phrase)), ())
                spelling = known[near[0]] if len(near) == 1 else None
            if spelling is None or (size == 1 and lower >= upper and lower):
                continue  # no such name, or usually an ordinary word ("mai", "ba", "anh")
            named = spelling.split(" ")
            if size > 1 and RULES["first_word"] and lower > upper:
                named[0] = named[0].lower()  # gold writes "anh Hùng", "tiếng Nhật" for a contact "Anh Hùng"
            out += named
            i += size
            break
        else:
            out.append(usual if RULES["lexicon"] and upper >= 5 and upper >= 10 * lower else words[i])
            i += 1
    return out


_EMAIL = re.compile(r"((?:\w+ chấm )*\w+(?: (?:%s))*),? a còng (\w+(?: chấm \w+)+)" % "|".join(sorted(_NUMBER_WORDS)),
                    re.IGNORECASE)


def _ascii(word: str) -> str:
    return "".join(char for char in normalize("NFD", word.lower().replace("đ", "d")) if char.isalnum())


def spoken_email(text: str) -> str:
    """Write a dictated address as one: "mai chấm anh năm mốt a còng example chấm test" -> mai.anh51@example.test.

    LIFEMATE_SMALL_FIXES=1. Seven TRAIN requests dictate an address this way and gold sends the joined form.
    """
    def joined(match: re.Match) -> str:
        parts = []
        for part in re.split(r" chấm ", match.group(1), flags=re.IGNORECASE):
            head, *spoken = part.split()
            digits = number(tuple(word.lower() for word in spoken)) if spoken else ""
            if digits is None:
                return match.group(0)  # the words after the name are not one number: leave the text alone
            parts.append(_ascii(head) + str(digits))
        return ".".join(parts) + "@" + ".".join(map(_ascii, re.split(r" chấm ", match.group(2), flags=re.IGNORECASE)))

    return _EMAIL.sub(joined, normalize("NFC", text))


def _repair(text: str, known: dict, casing: dict) -> str:
    pieces = normalize("NFC", text).split()
    if not pieces:
        return text
    last = pieces[-1].rstrip(".,?!")
    tail, pieces[-1:] = pieces[-1][len(last):], [last] if last else []
    return " ".join(_numbers(_snap(pieces, known, casing))) + tail


def fix(text: str, context: dict) -> str:
    """Repair one ASR transcript; `context` is the sample row (its `context` and `history` are read)."""
    return _repair(text, names(context), _casing())


class _Repaired:
    """A transcriber whose text is repaired with one sample's names, through every entry point it offers."""

    def __init__(self, inner, known: dict | None, casing: dict, emails: bool = False) -> None:
        self._inner, self._known, self._casing, self._emails = inner, known, casing, emails

    def _text(self, text: str) -> str:
        if self._emails:
            text = spoken_email(text)
        return text if self._known is None else _repair(text, self._known, self._casing)

    def __call__(self, path):
        return self._text(self._inner(path))

    def __getattr__(self, name):  # transcribe(path, deadline=) and transcribe_with_context(...), when present
        method = getattr(self._inner, name)
        if name not in ("transcribe", "transcribe_with_context"):
            return method
        return lambda *args, **options: self._text(method(*args, **options))


def wrapped(transcribe, row: dict):
    """The same transcriber, or one that repairs its text for this sample.

    LIFEMATE_ASR_FIX=1 turns on names, casing and codes; LIFEMATE_SMALL_FIXES=1 dictated e-mail addresses.
    """
    names_on = os.environ.get("LIFEMATE_ASR_FIX") == "1"
    emails = os.environ.get("LIFEMATE_SMALL_FIXES") == "1"
    if transcribe is None or not (names_on or emails):
        return transcribe
    return _Repaired(transcribe, names(row) if names_on else None, _casing() if names_on else {}, emails)


def _measure(data_root: str, transcripts: str) -> None:
    """TRAIN oracle analysis: word errors and copyable gold arguments before/after, tuning and held-out IDs."""
    from difflib import SequenceMatcher
    from pathlib import Path

    train = Path(data_root) / "train"
    rows, gold, schema = {}, {}, {}
    with (train / "train.jsonl").open(encoding="utf-8") as lines:
        for row in map(json.loads, lines):
            rows[row["id"]] = row
            gold.update({turn["audio"]: turn["audio_text"] for turn in [row, *row["history"]] if "audio" in turn})
    for app in json.loads((train / "train_tools_catalog.json").read_text(encoding="utf-8"))["apps"]:
        schema.update({tool["name"]: tool["parameters"].get("properties", {}) for tool in app["tools"]})
    with open(transcripts, encoding="utf-8") as lines:
        heard = [json.loads(line) for line in lines if line.strip()]
    held = frozenset(row for row in rows if int(row.rsplit("_", 1)[1]) % 5 == 0)  # rules were picked on the rest

    def tokens(text):
        return _WORD.findall(normalize("NFC", text))

    def distance(ref, hyp):
        previous = list(range(len(hyp) + 1))
        for i, word in enumerate(ref, 1):
            current = [i]
            for j, other in enumerate(hyp, 1):
                current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (word != other)))
            previous = current
        return previous[-1]

    def tally(part, repair):
        total = Counter()
        for item in heard:
            if (item["id"] in held) != (part == "held-out"):
                continue
            text = repair(item)
            ref, hyp = tokens(gold[item["audio"]]), tokens(text)
            lower_ref, lower_hyp = [word.lower() for word in ref], [word.lower() for word in hyp]
            total["words"] += len(ref)
            total["edits"] += distance(lower_ref, lower_hyp)  # WER as in HANDOFF: lower-case, \w+ words
            total["cased_edits"] += distance(ref, hyp)
            right = [False] * len(ref)
            for block in SequenceMatcher(None, lower_ref, lower_hyp, autojunk=False).get_matching_blocks():
                for index in range(block.a, block.a + block.size):
                    # Gold capitalises every sentence start; only later words are held to gold's casing.
                    right[index] = index == 0 or ref[index] == hyp[index - block.a + block.b]
            for index, word in enumerate(ref):
                kind = "number" if any(map(str.isdigit, word)) else "name" if index and word[0].isupper() else "other"
                total[kind] += 1
                total[kind + "_missed"] += not right[index]
            if item["audio"] != rows[item["id"]]["audio"]:
                continue  # history turns carry no gold calls of their own
            for call in rows[item["id"]]["label"]["tool_calls"]:
                for name, value in call["arguments"].items():
                    spec = schema.get(call["name"], {}).get(name, {})
                    if (not isinstance(value, str) or "x-ref" in spec or "enum" in spec
                            or call["name"].startswith("sys.") or re.match(r"\d{4}-\d\d-\d\d|\d\d:\d\d$", value)):
                        continue  # IDs, enums and timestamps are never copied from the transcript
                    kind = ("arg_number" if re.search(r"\d", value)
                            else "arg_name" if any(map(str.isupper, value)) else "arg_other")
                    total[kind] += 1
                    total[kind + "_in_asr"] += normalize("NFC", value) in normalize("NFC", text)
                    total[kind + "_in_gold"] += value in gold[item["audio"]]
        return total

    for part in ("tuning", "held-out"):
        casing = _casing(held if part == "held-out" else frozenset())  # held-out: lexicon built without those IDs
        before = tally(part, lambda item: item["transcript"])
        after = tally(part, lambda item: _repair(item["transcript"], names(rows[item["id"]]), casing))
        words = before["words"]
        print(f"{part}: {words} gold words")
        print(f"  WER: {before['edits'] / words:.2%} -> {after['edits'] / words:.2%} "
              f"({before['edits']} -> {after['edits']} edits)")
        print(f"  case-sensitive WER: {before['cased_edits'] / words:.2%} -> {after['cased_edits'] / words:.2%} "
              f"({before['cased_edits']} -> {after['cased_edits']} edits)")
        for kind in ("number", "name", "other"):
            print(f"  gold {kind} words not reproduced exactly: {before[kind + '_missed']}/{before[kind]} -> "
                  f"{after[kind + '_missed']}/{after[kind]}")
        for kind in ("arg_number", "arg_name", "arg_other"):
            print(f"  gold {kind} values verbatim in transcript: {before[kind + '_in_asr']}/{before[kind]} -> "
                  f"{after[kind + '_in_asr']}/{after[kind]} (in gold transcript: {before[kind + '_in_gold']})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=_measure.__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--transcripts", default="outputs/asr_train.jsonl")
    arguments = parser.parse_args()
    _measure(arguments.data_root, arguments.transcripts)
