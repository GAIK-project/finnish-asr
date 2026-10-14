"""Sentence times for timestamp training and a long-form dialect test, from SKN's annotation files.

SKN times every sentence. A published segment (prepare_skn.py) joins sentences, so its sentences
give Whisper-style timestamps inside the segment: training targets of the form
"<|0.00|> text<|1.84|><|1.84|> text<|5.60|>" (finetune_whisper.py --timestamps). Two outputs:

results/labels/skn_parts.jsonl
    one line per published segment (train, validation, test): id, split, seconds, and parts
    [[start, end, words], ...] relative to the segment start (words: tokens with a letter or
    digit). Sentences that overlap in time (two
    speakers at once) are merged into one part, so the parts never overlap.
results/labels/skn_longform.jsonl
    the long-form test: from each of the 10 test municipalities, about 4 minutes of one interview
    (recording "a", starting at the first sentence after 120 s), with the normalised reference text
    of every sentence inside the window and its parts relative to the window start. The window edges
    sit in pauses that no sentence crosses. The audio is cut from SKN's WAV on LUMI
    (eval_timestamps.py).

The segment index follows prepare_skn.segments(), so ids match the published dataset; a segment whose
recomputed text differs from the published one stops the script.

    uv run python scripts/prepare_timestamps.py <dir with SKN *_full.eaf files>

The *_full.eaf annotation files are in SKN's download packages (Kielipankki, CC BY 4.0); the published
segment texts it checks against are results/labels/skn_{train,validation}.jsonl (prepare_skn.py output,
regenerable and gitignored) and the SKN test references in results/benchmarks/skn_fi/.
Run metadata: results/labels/_run_metadata_timestamps.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_skn import TEST, segments, sentences, split_of  # noqa: E402
from provenance import build_run_metadata, write_run_metadata  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
LABELS = PROJECT / "results" / "labels"
WINDOW_FROM_S, WINDOW_S = 120.0, 240.0


def words(text: str) -> int:
    """Words with a letter or digit: SKN's "-" marks are not words, and the punctuated targets drop them."""
    return sum(1 for w in text.split() if any(c.isalnum() for c in w))


def merged_parts(sents: list[tuple[float, float, str]], origin: float, length: float) -> list[list]:
    """Sentences as non-overlapping [start, end, words] relative to origin, clipped to [0, length]."""
    parts: list[list] = []
    for start, end, text in sorted(sents):
        s, e, n = max(0.0, start - origin), min(length, end - origin), words(text)
        if parts and s < parts[-1][1]:
            parts[-1][1] = max(parts[-1][1], e)
            parts[-1][2] += n
        else:
            parts.append([round(s, 3), round(e, 3), n])
    for p in parts:
        p[1] = round(p[1], 3)
    return parts


def published() -> dict[str, tuple[str, str]]:
    """id (no .wav) -> (split, normalised text) of every published segment."""
    out = {}
    for split, path in (
        ("train", LABELS / "skn_train.jsonl"),
        ("validation", LABELS / "skn_validation.jsonl"),
        ("test", PROJECT / "results" / "benchmarks" / "skn_fi" / "hf_whisper-large-v3.jsonl"),
    ):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            out[row["id"].removesuffix(".wav")] = (split, row["ref"])
    return out


def window(sents: list[dict]) -> tuple[float, float] | None:
    """A window of about WINDOW_S seconds whose edges no sentence crosses."""

    def crossed(t: float) -> bool:
        return any(s["start"] < t < s["end"] for s in sents)

    starts = [s["start"] for s in sents if s["start"] >= WINDOW_FROM_S and not crossed(s["start"])]
    if not starts:
        return None
    ws = starts[0]
    ends = [s["end"] for s in sents if ws < s["end"] <= ws + WINDOW_S and not crossed(s["end"])]
    return (ws, max(ends)) if ends else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eaf_dir", type=Path, help="Directory with SKN's *_full.eaf annotation files.")
    eaf_dir = parser.parse_args().eaf_dir
    pub = published()
    parts_rows, long_rows, seen = [], [], set()
    for f in sorted(eaf_dir.glob("*_full.eaf")):
        rec = f.name.removesuffix("_full.eaf")
        sents = sentences(f.read_bytes())
        for i, g in enumerate(segments(sents)):
            sid = f"{rec}_{i:04d}"
            if sid not in pub:
                continue
            split, ref = pub[sid]
            if g["ref"] != ref:
                raise SystemExit(f"{sid}: recomputed text differs from the published one")
            length = g["end"] - g["start"]
            parts = merged_parts(g["parts"], g["start"], length)
            assert sum(p[2] for p in parts) == words(ref), sid
            parts_rows.append({"id": sid, "split": split, "seconds": round(length, 3), "parts": parts})
            seen.add(sid)
        number = int(rec[3:5])
        if number in TEST and rec[5] == "a" and split_of(number) == "test":
            w = window(sents)
            if w:
                inside = [s for s in sents if s["start"] >= w[0] and s["end"] <= w[1]]
                long_rows.append(
                    {
                        "recording": rec,
                        "municipality": rec.split("_", 1)[1],
                        "start": w[0],
                        "end": w[1],
                        "seconds": round(w[1] - w[0], 3),
                        "ref": " ".join(s["ref"] for s in inside),
                        "parts": merged_parts([(s["start"], s["end"], s["ref"]) for s in inside], w[0], w[1] - w[0]),
                    }
                )
    missing = sorted(set(pub) - seen)
    if missing:
        raise SystemExit(f"{len(missing)} published segments not found, e.g. {missing[:3]}")
    LABELS.mkdir(parents=True, exist_ok=True)
    (LABELS / "skn_parts.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in parts_rows), encoding="utf-8", newline="\n"
    )
    (LABELS / "skn_longform.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in long_rows), encoding="utf-8", newline="\n"
    )
    multi = sum(1 for r in parts_rows if len(r["parts"]) > 1)
    print(f"skn_parts.jsonl: {len(parts_rows)} segments, {multi} with more than one part")
    for r in long_rows:
        print(f"longform {r['recording']}: {r['seconds']:.0f} s, {len(r['parts'])} parts, {words(r['ref'])} words")
    write_run_metadata(
        LABELS,
        build_run_metadata(
            app_under_test="Finnish dialect ASR: SKN sentence times for timestamp training and the long-form test",
            dataset_path=None,
            runner_path=Path(__file__),
            cli_args=["<SKN eaf dir>"],
            extra={
                "segments": len(parts_rows),
                "segments_with_several_parts": multi,
                "longform_windows": len(long_rows),
            },
        ),
        name="_run_metadata_timestamps.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
