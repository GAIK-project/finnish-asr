"""Cut the Border Karelia corpus (Raja-Karjalan korpus) into dialect training segments (CPU job on LUMI).

Corpus: Raja-Karjalan korpus (Kotus, via Kielipankki, CC BY 4.0, http://urn.fi/urn:nbn:fi:lb-2017121322).
171 interviews, 118 h, recorded in the 1960s-70s for Kotus's tape archive (SKNA) with evacuees from six
Border Karelian parishes: Ilomantsi, Korpiselkä, Impilahti, Suistamo, Salmi and Suojärvi. Eastern
Finnish dialects in long contact with Karelian. Every recording has a Praat TextGrid with one tier per
speaker, utterance by utterance, in a coarse dialect transcription (the corpus's Litterointimerkit.pdf),
and an "original" tier that repeats them. The corpus readme says it contains personal data. None of the
six parishes is among SKN's 50 municipalities or LA-murre's 10 test localities, so both dialect tests
stay unseen.

Segments: each speaker tier's utterances joined while the gap is under 1.5 s and the segment stays
under 25 s; kept only when no other speaker's utterance overlaps it, it has at least 3 words and
2 s, and no unclear-speech mark: "(- -)" unclear passage, "@" unclear sound, or any other parenthesis.
Clean-up for the text: "´" (palatalisation) and "*" (short geminate) removed, "_" (legato) becomes a
space, a word cut short ("kes-") keeps the part that was said, a word break inside a word ("-") and
quotation marks are removed. Special letters (ŋ š z ž č) stay: the LLM that writes the standard-Finnish
target (prepare_targets.py normalise) reads them. Audio: 44.1 kHz resampled to 16 kHz mono.

Validation: every recording of two informants (VALIDATION, drawn with seed 0 from Ilomantsi and Suistamo),
at most VALIDATION_CAP segments each, sampled with seed 0. Their other recordings are not trained on.

    python prepare_finka.py [--workers 12]     # reads $FIASR_DATA/finka_raw/Raja-Karjalan_korpus_dl.zip

Output: $FIASR_DATA/finka_fi/{train,validation}/<id>.wav and <split>.jsonl (id, split, parish, recording,
seconds, ref = cleaned dialect text, orig = the transcript as written), and stats.json. CC BY 4.0
allows sharing these texts with attribution; this repository still commits only their statistics, as
the corpus holds personal data.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SR = 16000
MAX_S, MIN_S, MAX_GAP_S, MIN_WORDS = 25.0, 2.0, 1.5, 3
DATA = Path(os.environ.get("FIASR_DATA", Path(__file__).resolve().parents[1] / "data"))
ZIP = "Raja-Karjalan_korpus_dl.zip"
VALIDATION = {"Ilomantsi_13", "Suistamo_06"}  # parish_informant: every recording of these two
VALIDATION_CAP = 100


def informants(stem: str) -> set[str]:
    """'Korpiselka_07nB_08mB_SKNA_3700_1b' -> {'Korpiselka_07', 'Korpiselka_08'}."""
    parish, *rest = stem.split("_")
    ids = rest[: rest.index("SKNA")] if "SKNA" in rest else rest[:1]
    return {f"{parish}_{m.group(1)}" for p in ids if (m := re.match(r"(\d+)[mn]", p))}


def clean(text: str) -> str | None:
    """The transcript as words for the LLM; None if it marks unclear speech."""
    if re.search(r"[()@]", text):
        return None
    text = text.replace("´", "").replace("*", "").replace("_", " ")
    text = re.sub(r"[”“\"]", "", text)
    words = []
    for w in text.split():
        w = re.sub(r"(?<=\w)-(?=\w)", "", w)  # a break inside a word
        w = w.rstrip("-")  # a word cut short keeps what was said
        if w.strip(".,?!"):
            words.append(w)
    return " ".join(words) or None


# A Praat text file as a token stream: quoted strings, numbers and <exists>; the long format's keys
# ("xmin =") and indices ("item [1]:") are skipped, so the long and the short format read alike.
TOKEN = re.compile(r'"((?:[^"]|"")*)"|\[\d*\]|(<exists>)|(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)|[^\s"\[<]+|\S')


def praat_tokens(text: str) -> list[str | float]:
    out: list[str | float] = []
    for m in TOKEN.finditer(text):
        if m.group(1) is not None:
            out.append(m.group(1).replace('""', '"'))
        elif m.group(2):
            out.append("<exists>")
        elif m.group(3):
            out.append(float(m.group(3)))
    return out


def tiers(textgrid: str) -> dict[str, list[tuple[float, float, str]]]:
    """Speaker tier -> its non-empty intervals (start, end, text); the 'original' tier is left out.

    Reads both of Praat's text formats: 52 of the corpus's TextGrids are in the long one, 119 in the short one."""
    t = praat_tokens(textgrid)
    if t[:2] != ["ooTextFile", "TextGrid"] or t[4] != "<exists>":
        raise ValueError("not a Praat TextGrid with tiers")
    out, i = {}, 6
    for _ in range(int(t[5])):
        kind, name, n = t[i], t[i + 1], int(t[i + 4])
        i += 5
        width = 3 if kind == "IntervalTier" else 2  # a TextTier has (time, mark) points
        rows = [t[i + k * width : i + (k + 1) * width] for k in range(n)]
        i += n * width
        if kind == "IntervalTier" and name != "original":
            out[name] = [(float(a), float(b), str(x).strip()) for a, b, x in rows if str(x).strip()]
    return out


def segments(textgrid: str) -> list[dict]:
    spans = tiers(textgrid)
    segs = []
    for name, ivs in spans.items():
        cur = None
        for s, e, t in sorted(ivs):
            if cur and s - cur["end"] < MAX_GAP_S and max(cur["end"], e) - cur["start"] <= MAX_S:
                cur["end"], cur["orig"] = max(cur["end"], e), cur["orig"] + " " + t
            else:
                if cur:
                    segs.append(cur)
                cur = {"speaker": name, "start": s, "end": e, "orig": t}
        if cur:
            segs.append(cur)
    out = []
    for g in sorted(segs, key=lambda g: g["start"]):
        others = (iv for n, ivs in spans.items() if n != g["speaker"] for iv in ivs)
        if any(s < g["end"] and e > g["start"] for s, e, _ in others):
            continue
        ref = clean(g["orig"])
        if ref and MIN_S <= g["end"] - g["start"] <= MAX_S and len(ref.split()) >= MIN_WORDS:
            out.append({**g, "ref": ref})
    return out


def cut_recording(job: tuple[str, str, str, str]) -> tuple[list[dict], Counter]:
    """(zip path, wav member, TextGrid member, recording id) -> its segments on disk, and why others were dropped."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.functional as AF

    zpath, wav, tg, rec = job
    split = "validation" if informants(rec) & VALIDATION else "train"
    with zipfile.ZipFile(zpath) as z:
        grid = z.read(tg).decode("utf-8")
        audio, sr = sf.read(io.BytesIO(z.read(wav)), dtype="float32")
    segs = segments(grid)
    total = sum(len(v) for v in tiers(grid).values())
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio16 = AF.resample(torch.from_numpy(audio), sr, SR).numpy()
    out_dir = DATA / "finka_fi" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, g in enumerate(segs):
        clip = audio16[int(g["start"] * SR) : int(g["end"] * SR)]
        if len(clip) < MIN_S * SR or float(np.abs(clip).max()) < 1e-4:
            continue
        sid = f"{rec}_{i:04d}"
        sf.write(out_dir / f"{sid}.wav", clip, SR, subtype="PCM_16")
        rows.append(
            {
                "id": sid,
                "split": split,
                "parish": rec.split("_")[0],
                "recording": rec,
                "seconds": round(len(clip) / SR, 2),
                "ref": g["ref"],
                "orig": g["orig"],
            }
        )
    return rows, Counter({"utterances": total, "kept_segments": len(rows)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=DATA / "finka_raw")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    zpath = args.raw / ZIP
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
    grids = {Path(n).stem: n for n in names if n.endswith(".TextGrid")}
    jobs = [(str(zpath), n, grids[Path(n).stem], Path(n).stem) for n in names if n.endswith(".wav")]
    jobs = [j for j in sorted(jobs) if j[3] in grids]
    print(f"{len(jobs)} recordings", flush=True)
    rows: list[dict] = []
    counts: Counter = Counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for job, (res, c) in zip(jobs, pool.map(cut_recording, jobs), strict=True):
            rows += res
            counts += c
            print(f"{job[3]}: {len(res)} segments", flush=True)

    out = DATA / "finka_fi"
    stats = {"source": {"recordings": len(jobs), **counts}}
    rng = random.Random(0)
    for split in ("train", "validation"):
        part = sorted((r for r in rows if r["split"] == split), key=lambda r: r["id"])
        if split == "validation":  # an even sample per informant; the rest is deleted
            keep = []
            for who in sorted(VALIDATION):
                rs = [r for r in part if who in informants(r["recording"])]
                keep += rng.sample(rs, min(VALIDATION_CAP, len(rs)))
            keep = sorted({r["id"]: r for r in keep}.values(), key=lambda r: r["id"])
            for r in part:
                if r not in keep:
                    (out / split / f"{r['id']}.wav").unlink(missing_ok=True)
            part = keep
        (out / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part), encoding="utf-8", newline="\n"
        )
        stats[split] = {
            "segments": len(part),
            "hours": round(sum(r["seconds"] for r in part) / 3600, 2),
            "words": sum(len(r["ref"].split()) for r in part),
            "parishes": dict(sorted(Counter(r["parish"] for r in part).items())),
        }
        print(f"{split}: {stats[split]['segments']} segments, {stats[split]['hours']} h", flush=True)
    (out / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
