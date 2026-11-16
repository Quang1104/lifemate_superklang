"""Transcribe TRAIN audio with the local ASR model, so SFT data can use the text inference will see.

Writes one JSON line per audio file, covering the current request and every user turn of the history:
{"id": sample ID, "audio": "audio/<id>/<file>.wav", "transcript": text}. Feed the file to
scripts/build_sft.py --transcripts. Rerunning appends only the files that are still missing, so an
interrupted GPU session can resume. The 309 validation IDs are skipped by default.
"""

import argparse
import json
from pathlib import Path

from scripts.build_sft import excluded_ids
from src.inference.official import OfficialKit


def pending(kit: OfficialKit, excluded: set[str], done: set[str], limit: int | None = None):
    """(sample ID, relative audio path, absolute path) of each audio file that still needs a transcript."""
    samples = 0
    for row in kit.rows():
        if row["id"] in excluded:
            continue
        if limit is not None and samples >= limit:
            return
        samples += 1
        turns = [(row, False)] + [(turn, True) for turn in row["history"] if turn.get("role") == "user"]
        for turn, history in turns:
            if turn["audio"] not in done:
                yield row["id"], turn["audio"], kit.audio_path(row["id"], turn["audio"], history=history)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--asr-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("outputs/asr_train.jsonl"))
    parser.add_argument("--exclude", type=Path, nargs="+", default=[Path("research/validation_split.json")])
    parser.add_argument("--limit", type=int, default=None, help="samples, for a quick check")
    args = parser.parse_args()
    if any(part.endswith("_environment") for part in args.data_root.resolve().parts):
        raise ValueError("Forbidden root")
    done = set()
    if args.output.is_file():
        with args.output.open(encoding="utf-8") as lines:
            done = {json.loads(line)["audio"] for line in lines if line.strip()}
    from src.agent.local_model import LocalWhisper, initialize_runtime  # heavy imports only when running

    initialize_runtime()
    transcribe = LocalWhisper(args.asr_dir, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("a", encoding="utf-8") as sink:
        for sample_id, audio, path in pending(OfficialKit(args.data_root, "train"), excluded_ids(args.exclude),
                                              done, args.limit):
            row = {"id": sample_id, "audio": audio, "transcript": transcribe(path)}
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            written += 1
    print(json.dumps({"already_done": len(done), "written": written, "output": str(args.output)}))


if __name__ == "__main__":
    main()
