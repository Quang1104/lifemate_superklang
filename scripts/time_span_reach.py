"""TRAIN oracle diagnostic: how many gold date/time arguments can some transcript span be converted to.

An upper bound for the span converter only; it is not task accuracy and never runs during inference.
"""

import argparse
import json
from pathlib import Path
import re

from src.agent.plan_values import Unresolved, convert_span

_FORMS = {"datetime": r"\d{4}-\d\d-\d\dT\d\d:\d\d(?::\d\d)?(?:[+-]\d\d:\d\d)?",
          "date": r"\d{4}-\d\d-\d\d", "time": r"\d\d:\d\d"}


def reachable(row: dict) -> tuple[int, int]:
    words = re.sub(r"([.,])", r" \1", row["audio_text"]).split()
    spans = {" ".join(words[i:j]) for i in range(len(words)) for j in range(i + 1, min(len(words), i + 9) + 1)}
    golds = {(mode, value) for call in row["label"]["tool_calls"] for value in call["arguments"].values()
             if isinstance(value, str) for mode, form in _FORMS.items() if re.fullmatch(form, value)}
    hits = 0
    for mode, gold in golds:
        for span in spans:
            try:
                value = convert_span(span, mode, row["context"])
            except Unresolved:
                continue
            if value == gold or mode == "datetime" and value[:16] == gold[:16]:
                hits += 1
                break
    return hits, len(golds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="../datalife/train+public_test")
    args = parser.parse_args()
    path = Path(args.data_root) / "train" / "train.jsonl"
    hits = total = 0
    with path.open(encoding="utf-8") as rows:
        for line in rows:
            found, count = reachable(json.loads(line))
            hits, total = hits + found, total + count
    print(json.dumps({"reachable": hits, "gold_time_arguments": total, "rate": round(hits / total, 4)}))


if __name__ == "__main__":
    main()
