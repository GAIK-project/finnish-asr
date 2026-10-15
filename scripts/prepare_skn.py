"""Cut Suomen kielen näytteitä (SKN) into ASR training and test segments (CPU job on LUMI).

SKN (Kotus / Kielipankki, CC BY 4.0, http://urn.fi/urn:nbn:fi:lb-2020112937) holds dialect
interviews from 50 Finnish municipalities, about two hours each. Every sentence has a time
span, a dialect transcript in phonetic notation and a normalised transcript in standard
Finnish word forms (lower-case except names, no punctuation). The normalised text is the
target here: a model that hears dialect and writes standard Finnish.

Segments: sentence annotations of all speakers in time order are joined while the gap is
under 1.5 s and the segment stays under 25 s; segments shorter than 2 s or with fewer than
3 words are dropped. Audio is resampled from 44.1 kHz to 16 kHz mono WAV.

Split by municipality, so no test interviewee or locality is seen in training (interviewer speech
is kept, and a few SKN fieldworkers recorded both test and training municipalities). The SKN numbers
are the series' publication order; every fifth one gives a test set in five of SKN's six main
dialect groups (no Ostrobothnian or Peräpohjola municipality, all nine are in training):
    test        SKN05, 10, 15, 20, 25, 30, 35, 40, 45, 50   (at most 60 segments each)
    validation  SKN03, SKN28                                 (at most 100 segments each)
    train       the other 38

    python prepare_skn.py --raw $FIASR_DATA/skn_raw [--workers 16]
    python prepare_skn.py --heldout       # the 12 test and validation municipalities, every segment

Output: $FIASR_DATA/skn_fi/{train,validation,test}/<id>.wav and <split>.jsonl
(id, ref = normalised text, orig = dialect text, municipality, seconds), and stats.json.

--heldout cuts only the test and validation municipalities, without the cap, into skn_fi/heldout/ and
heldout.jsonl (heldout_stats.json): the rest of SKN, for a model trained on all 50 municipalities
(finetune_whisper.py --data skn-all+open). The other splits are left as they are. Such a model has
heard the SKN test speakers, so the SKN test no longer measures it.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SR = 16000
MAX_S, MIN_S, MAX_GAP_S, MIN_WORDS = 25.0, 2.0, 1.5, 3
TEST = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50}
VALIDATION = {3, 28}
CAP = {"test": 60, "validation": 100}
DATA = Path(os.environ.get("FIASR_DATA", Path(__file__).resolve().parents[1] / "data"))


def split_of(number: int) -> str:
    return "test" if number in TEST else "validation" if number in VALIDATION else "train"


def sentences(eaf: bytes) -> list[dict]:
    """(start s, end s, normalised text, dialect text) for every sentence of every speaker.

    The dialect sentence (tier `XX-sentence`) is matched to the normalised one (`XX-sentence-norm`)
    by its time values: each tier has its own time-slot ids, so matching by id never succeeds
    (that bug left every dialect text empty before 14 Oct 2026). A normalised sentence without a
    dialect sentence at the same times (65 of 51 312) gets orig None.
    """
    root = ET.fromstring(eaf)
    ts = {t.get("TIME_SLOT_ID"): int(t.get("TIME_VALUE") or -1) for t in root.iter("TIME_SLOT")}
    tiers = {t.get("TIER_ID"): t for t in root.iter("TIER")}
    out = []
    for tid, tier in tiers.items():
        if not tid.endswith("-sentence-norm"):
            continue
        orig_tier = tiers.get(tid[: -len("-norm")])
        orig = {}
        if orig_tier is not None:
            for a in orig_tier.iter("ALIGNABLE_ANNOTATION"):
                key = (ts.get(a.get("TIME_SLOT_REF1"), -1), ts.get(a.get("TIME_SLOT_REF2"), -1))
                orig[key] = re.sub(r"\s+", " ", a.findtext("ANNOTATION_VALUE") or "").strip()
        for a in tier.iter("ALIGNABLE_ANNOTATION"):
            t1, t2 = ts.get(a.get("TIME_SLOT_REF1"), -1), ts.get(a.get("TIME_SLOT_REF2"), -1)
            text = re.sub(r"\s+", " ", a.findtext("ANNOTATION_VALUE") or "").strip()
            if t1 < 0 or t2 <= t1 or not text:
                continue
            out.append({"start": t1 / 1000, "end": t2 / 1000, "ref": text, "orig": orig.get((t1, t2))})
    return sorted(out, key=lambda s: s["start"])


def segments(sents: list[dict]) -> list[dict]:
    """Join sentences into segments; each keeps its sentences as parts (start, end, normalised text)."""
    segs, cur = [], None
    for s in sents:
        if cur and s["start"] - cur["end"] < MAX_GAP_S and max(cur["end"], s["end"]) - cur["start"] <= MAX_S:
            cur["end"] = max(cur["end"], s["end"])
            cur["ref"] += " " + s["ref"]
            # a segment's dialect text is complete or empty: one unmatched sentence empties it
            cur["orig"] = None if cur["orig"] is None or s["orig"] is None else cur["orig"] + " " + s["orig"]
            cur["parts"].append((s["start"], s["end"], s["ref"]))
        else:
            if cur:
                segs.append(cur)
            cur = dict(s, parts=[(s["start"], s["end"], s["ref"])])
    if cur:
        segs.append(cur)
    return [g for g in segs if MIN_S <= g["end"] - g["start"] <= MAX_S and len(g["ref"].split()) >= MIN_WORDS]


def process(job: tuple[str, str, str, str, bool]) -> list[dict]:
    """One recording: (zip path, wav member, eaf member, recording id, heldout) -> its segments, written to disk."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.functional as AF

    zpath, wav_name, eaf_name, rec, heldout = job
    number = int(rec[3:5])
    split = "heldout" if heldout else split_of(number)
    with zipfile.ZipFile(zpath) as z:
        segs = segments(sentences(z.read(eaf_name)))
        if not segs:
            return []
        audio, sr = sf.read(io.BytesIO(z.read(wav_name)), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio16 = AF.resample(torch.from_numpy(audio), sr, SR).numpy()
    out_dir = DATA / "skn_fi" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, g in enumerate(segs):
        clip = audio16[int(g["start"] * SR) : int(g["end"] * SR)]
        if len(clip) < MIN_S * SR or float(np.abs(clip).max()) < 1e-4:
            continue
        sid = f"{rec}_{i:04d}.wav"
        sf.write(out_dir / sid, clip, SR, subtype="PCM_16")
        rows.append(
            {
                "id": sid,
                "split": split,
                "municipality": rec.split("_", 1)[1],
                "number": number,
                "seconds": round(len(clip) / SR, 2),
                "ref": g["ref"],
                "orig": g["orig"] or "",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=DATA / "skn_raw")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--heldout", action="store_true", help="Only the test and validation municipalities, uncapped.")
    args = parser.parse_args()

    jobs = []
    for zpath in sorted(args.raw.glob("skn_*.zip")):
        with zipfile.ZipFile(zpath) as z:
            names = z.namelist()
        for wav in (n for n in names if n.endswith(".wav") and "/Media/" in n):
            rec = Path(wav).stem  # SKN07a_Lappajarvi
            eaf = [n for n in names if n.endswith(f"/Annotations/{rec}_full.eaf")]
            if eaf and (not args.heldout or int(rec[3:5]) in TEST | VALIDATION):
                jobs.append((str(zpath), wav, eaf[0], rec, args.heldout))
    print(f"{len(jobs)} recordings", flush=True)
    rows: list[dict] = []
    with ProcessPoolExecutor(args.workers) as pool:
        for job, res in zip(jobs, pool.map(process, jobs), strict=True):
            rows += res
            print(f"{job[3]}: {len(res)} segments", flush=True)

    if args.heldout:
        rows.sort(key=lambda r: r["id"])
        (DATA / "skn_fi" / "heldout.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n"
        )
        stats = {
            "segments": len(rows),
            "hours": round(sum(r["seconds"] for r in rows) / 3600, 2),
            "municipalities": sorted({r["municipality"] for r in rows}),
        }
        (DATA / "skn_fi" / "heldout_stats.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"heldout: {stats['segments']} segments, {stats['hours']} h", flush=True)
        return 0

    random.seed(0)
    stats = {}
    for split in ("train", "validation", "test"):
        part = [r for r in rows if r["split"] == split]
        if split in CAP:  # an even, reproducible sample per municipality; the rest is deleted
            keep = []
            for m in sorted({r["municipality"] for r in part}):
                ms = [r for r in part if r["municipality"] == m]
                keep += sorted(random.sample(ms, min(CAP[split], len(ms))), key=lambda r: r["id"])
            for r in part:
                if r not in keep:
                    (DATA / "skn_fi" / split / r["id"]).unlink(missing_ok=True)
            part = keep
        (DATA / "skn_fi" / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part), encoding="utf-8", newline="\n"
        )
        stats[split] = {
            "segments": len(part),
            "hours": round(sum(r["seconds"] for r in part) / 3600, 2),
            "municipalities": sorted({r["municipality"] for r in part}),
        }
        print(f"{split}: {stats[split]['segments']} segments, {stats[split]['hours']} h", flush=True)
    (DATA / "skn_fi" / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
