from pathlib import Path

from training_code.train_asr import utterances, wer


def test_wer_ignores_case_and_punctuation():
    assert wer([("Gọi Minh, nhé.", "gọi minh nhé")]) == 0
    assert wer([("một hai ba bốn", "một ba bốn")]) == 0.25  # one deletion over four reference words
    assert wer([("a b", "a b c d")]) == 1.0  # insertions count


class FakeKit:
    def rows(self):
        return [
            {"id": "t1", "audio": "a/t1.wav", "audio_text": "câu một",
             "history": [{"role": "user", "audio": "a/t1_u1.wav", "audio_text": "lượt cũ"},
                         {"role": "assistant", "content": "ok"}]},
            {"id": "v1", "audio": "a/v1.wav", "audio_text": "câu kiểm", "history": []},
        ]

    def audio_path(self, sample_id, audio, *, history):
        return Path(("H:" if history else "C:") + audio)


def test_utterances_split_and_history():
    train = utterances(FakeKit(), lambda i: i != "v1")
    assert train == [(Path("C:a/t1.wav"), "câu một"), (Path("H:a/t1_u1.wav"), "lượt cũ")]
    assert utterances(FakeKit(), lambda i: i == "v1", history=False) == [(Path("C:a/v1.wav"), "câu kiểm")]
