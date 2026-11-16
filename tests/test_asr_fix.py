"""Opt-in ASR text repair: off by default, context names, TRAIN-gold casing, spoken codes."""

import json
from pathlib import Path
from time import monotonic

import pytest

from src.agent import asr_fix
from src.agent.asr_fix import _repair, fix, names, number, wrapped
from src.inference.pipeline import _sample_from_row

ROW = {"id": "train_0001", "audio": "audio/train_0001/train_0001.wav",
       "context": {"installed_apps": {"contacts": {"tools": ["contacts.search"]}},
                   "user_profile": {"name": "Anh Khoa", "city": "Đà Nẵng", "home_address": "12 Nguyễn Huệ, Quận 1"}},
       "history": [
           {"role": "user", "audio": "audio/train_0001/train_0001_u1.wav", "audio_text": "Gọi Thu Hà"},
           {"role": "assistant", "text": "Mai có 2 báo thức. Đã gọi Sếp Dũng.",
            "tool_calls": [{"name": "contacts.search", "arguments": {"query": "Tuấn"}}],
            "tool_results": [{"contacts": [{"name": "Trần Văn Tuấn"}, {"name": "Anh Hùng"}, {"name": "Mai"},
                                           {"name": "Hải"}, {"note": "Tiền điện tháng này"}]}]}]}
# (capitalised, lower-case, spelling) as counted in TRAIN gold; the real table comes from the shipped pool.
CASING = {"mai": (40, 431, "Mai"), "anh": (38, 735, "Anh"), "lan": (77, 0, "Lan"), "zalo": (77, 0, "Zalo"),
          "sếp": (27, 19, "Sếp"), "minh": (80, 3, "Minh"), "sms": (64, 0, "SMS"), "tiền": (0, 200, "tiền"),
          "khoá": (0, 20, "khoá"), "nói": (0, 90, "nói"), "hà": (57, 0, "Hà")}


@pytest.mark.parametrize("words,value", [
    ("sáu", 6), ("mười", 10), ("mười lăm", 15), ("hai mươi", 20), ("hai mươi sáu", 26), ("hai sáu", 26),
    ("chín lăm", 95), ("ba mốt", 31), ("một trăm", 100), ("một trăm linh bảy", 107), ("ba trăm hai mươi", 320),
    ("năm trăm bốn mươi tư", 544),
    ("mười mốt", None), ("một không bảy", None), ("mười hai mười ba", None), ("trăm", None), ("hai trăm trăm", None),
    ("không chín tám", None), ("sáu giờ", None), ("", None),
])
def test_number_reads_one_whole_number_or_nothing(words, value):
    assert number(tuple(words.split())) == value


@pytest.mark.parametrize("heard,written", [
    ("xăng ron chín lăm giờ bao nhiêu.", "xăng RON95 giờ bao nhiêu."),
    ("mở kênh vtv sáu", "mở kênh VTV6"),
    ("chuyến rv một trăm linh bảy", "chuyến RV107"),
    ("coi e năm dùm nha", "coi E5 dùm nha"),
    ("chuyến rv một không bảy", "chuyến rv một không bảy"),  # digit by digit: not one number
    ("e không biết nữa", "e không biết nữa"),
])
def test_spoken_codes_are_joined_like_gold_writes_them(heard, written):
    assert _repair(heard, {}, {}) == written


@pytest.mark.parametrize("text", [
    "đặt báo thức sáu giờ sáng mai", "bật điều hòa hai mươi sáu độ", "chuyển năm trăm nghìn", "để mờ ba mươi phần trăm",
    "số không chín tám ba hai một", "bốn mươi lăm phút nữa nhắc chị", "dưới một trăm hai mươi ba triệu thì mua",
])
def test_numbers_gold_usually_spells_are_left_alone(text):
    assert fix(text, ROW) == text


def test_names_come_from_the_profile_and_assistant_turns_but_not_sentence_starts():
    known = names(ROW)
    assert known[("trần", "văn", "tuấn")] == "Trần Văn Tuấn" and known[("sếp", "dũng")] == "Sếp Dũng"
    assert known[("đà", "nẵng")] == "Đà Nẵng" and known[("mai",)] == "Mai"  # the contact, a field of its own
    assert ("đã",) not in known and ("tiền",) not in known  # capitalised only because they open a sentence
    assert ("thu", "hà") not in known  # user turns are ASR text, never a source of spelling
    assert names({}) == {}


def test_context_names_take_their_spelling_including_wrong_tones_on_longer_names():
    heard = "gọi cho tran van tuan với sếp dùng ở đà nẵng rồi nhắn anh hùng."
    assert _repair(heard, names(ROW), CASING) == "gọi cho Trần Văn Tuấn với Sếp Dũng ở Đà Nẵng rồi nhắn anh Hùng."


def test_ordinary_words_are_never_snapped_to_a_name():
    known = names(ROW)
    for text in ("sáng mai nhắc anh nhé", "ngày mai anh đi đà lạt", "mua hai ký hai sản", "anh khoá cửa chưa"):
        assert _repair(text, known, CASING) == text
    assert _repair("anh hải ơi", known, CASING) == "anh Hải ơi"  # a context name TRAIN never lower-cases
    # With the shipped TRAIN table: frequent Vietnamese words that are also given names stay lower-case.
    assert fix("sáng mai ba với em có về nhà hôm thứ năm không", ROW) == "sáng mai ba với em có về nhà hôm thứ năm không"


def test_words_gold_capitalises_are_capitalised_without_context():
    assert _repair("nhắn zalo cho lan rồi gửi sms cho minh", {}, CASING) == "nhắn Zalo cho Lan rồi gửi SMS cho Minh"
    assert _repair("sếp ơi mai họp", {}, CASING) == "sếp ơi mai họp"  # not capitalised often enough


@pytest.mark.parametrize("text", ["Gọi cho Lan, rồi mở VTV6 nhé.", "Nhắn anh Hùng ở Đà Nẵng là 26 độ.", "", "   "])
def test_text_that_is_already_cased_and_punctuated_is_unchanged(text):
    assert _repair(text, names(ROW), CASING) == text


class Kit:
    directory = Path("kit") / "train"

    def audio_path(self, sample_id, relative_path, *, history):
        return self.directory / relative_path


class Whisper:
    def __call__(self, path):
        return "gọi cho tran van tuan ở đà nẵng."

    def transcribe(self, path, *, deadline=None):
        return self(path)

    def transcribe_with_context(self, path, *, deadline=None, context=None):
        return self(path)

    def timing_snapshot(self):
        return ["kept"]


def test_flag_off_leaves_the_transcriber_and_the_sample_untouched(monkeypatch):
    monkeypatch.delenv("LIFEMATE_ASR_FIX", raising=False)
    whisper = Whisper()
    assert wrapped(whisper, ROW) is whisper and wrapped(None, ROW) is None
    sample = _sample_from_row(Kit(), ROW, whisper, monotonic() + 60, control_noop=False)
    assert sample.user_request == "gọi cho tran van tuan ở đà nẵng."
    monkeypatch.setenv("LIFEMATE_ASR_FIX", "0")
    assert wrapped(whisper, ROW) is whisper


def test_flag_on_repairs_the_request_and_history_user_turns_through_every_entry_point(monkeypatch):
    monkeypatch.setenv("LIFEMATE_ASR_FIX", "1")
    monkeypatch.setattr(asr_fix, "_casing", lambda: CASING)
    fixed = "gọi cho Trần Văn Tuấn ở Đà Nẵng."
    sample = _sample_from_row(Kit(), ROW, Whisper(), monotonic() + 60, control_noop=False)
    assert sample.user_request == fixed and json.loads(sample.history[0])["text"] == fixed
    released = {**ROW, "audio_text": "gọi cho tran van tuan", "history": []}  # TRAIN transcripts are never repaired
    assert _sample_from_row(Kit(), released, None, monotonic() + 60, control_noop=False).user_request == \
        "gọi cho tran van tuan"
    repaired = wrapped(Whisper(), ROW)
    assert repaired(Path("a.wav")) == repaired.transcribe(Path("a.wav"), deadline=1.0) == fixed
    assert repaired.transcribe_with_context(Path("a.wav"), deadline=1.0, context={}) == fixed
    assert repaired.timing_snapshot() == ["kept"]
    plain = wrapped(lambda path: "nhắn sếp dùng", ROW)
    assert plain(Path("a.wav")) == "nhắn Sếp Dũng" and getattr(plain, "transcribe", None) is None
