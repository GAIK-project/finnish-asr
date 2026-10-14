"""Cut a second dialect test set from the Finnish Dialect Syntax Archive (LA-murre), test only.

LA-murre (University of Turku / Kotus, via Kielipankki, CC BY-ND 4.0,
http://urn.fi/urn:nbn:fi:lb-2020112935) holds 142 hour-long dialect interviews from 132
localities, recorded separately from SKN. Interviewee speech is time-aligned in sentence-sized
chunks with a coarse **dialect** transcript: no standard-Finnish layer, unlike SKN. So WER here
is against dialect spellings, which charges a model that writes standard Finnish; compare models
with each other, not with the SKN numbers.

Localities: one per dialect region, none of them among SKN's 50 municipalities, so neither the
place nor the speaker can be in this project's SKN training data. Only the 10 WAV files are read,
by HTTP range requests into the six 10-14 GB zip packages, so nothing else is downloaded.

Segments: interviewee annotations joined while the gap is under 1.5 s and the segment stays
under 25 s; kept only when no interviewer annotation overlaps it (the reference would miss that
speech) and it has no unclear-speech mark "(", "/" or "§". Reference clean-up: "+" (proper-noun
mark) and "'" (palatalisation mark) removed; a word cut short ("niin-", "se...") keeps the part
that was said. At most 60 segments per locality, sampled with seed 0.

    python prepare_lamurre.py      # login node (needs the internet); writes $FIASR_DATA/lamurre_fi/

--train cuts TRAINING data from the other recordings (a CPU job; reads the zip packages that
`lumi.sh download lamurre` put in $FIASR_DATA/lamurre_raw). Left out: the 10 test recordings, a
second recording from a test locality, and recordings from the SKN test municipalities, so both
dialect tests stay unseen. Two recordings are set aside for validation. The dialect text of these
segments is rewritten in standard Finnish by prepare_targets.py before training. LA-murre is
CC BY-ND: a model trained on it is for internal comparison, not for publishing.

    python prepare_lamurre.py --train [--workers 12]

Everything this script writes ($FIASR_DATA/lamurre_fi, default data/lamurre_fi) and every file made
from it later (results/benchmarks/lamurre_fi/*.jsonl, results/judgements_dialect/**/*.jsonl,
results/labels/lamurre_*.jsonl) holds the corpus's texts and is gitignored: the licence does not
allow sharing modified copies. Only numbers made from them are committed.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SR = 16000
MAX_S, MIN_S, MAX_GAP_S, MIN_WORDS, CAP = 25.0, 2.0, 1.5, 3, 60
BASE = "https://www.kielipankki.fi/download/la-murre/audio/"
DATA = Path(os.environ.get("FIASR_DATA", Path(__file__).resolve().parents[1] / "data"))
# recording -> WAV package; one locality per region, none in SKN. Place names marked ? are read
# from the file code only; the others are also said in the interview.
RECORDINGS = {
    "hame_tamm1": "hame-kai",  # Häme: Tammela
    "kai_pos1": "hame-kai",  # Kainuu dialects: Posio
    "kare_muol1": "kare-karp",  # Karelian Isthmus: Muolaa
    "kess_lau1": "kess-pohe",  # Central Finland: Laukaa ?
    "kyml_iit1": "kess-pohe",  # Kymenlaakso: Iitti ?
    "perpl_sod1": "kess-pohe",  # Lapland: Sodankylä ?
    "pohe_laih1": "kess-pohe",  # South Ostrobothnia: Laihia
    "pohp_tem1": "pohki-sate",  # North Ostrobothnia: Temmes
    "savp_nil1": "satl-som",  # North Savo: Nilsiä ?
    "varp_raum1": "uusi-varp",  # Southwest Finland, north: Rauma
}
# never trained on: a second recording from a test locality, and SKN's test municipalities
# (Eurajoki, Rautu, Vihti, Mikkeli, Ikaalinen have a recording here too)
TRAIN_EXCLUDE = {"varp_raum2", "varp_eurj1", "kark_rautu1", "uusl_viht1", "save_mik1", "satp_ika1"}
VALIDATION = {"som_somero1", "jare_pihl1"}
VALIDATION_CAP = 100


class HttpFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests, enough for zipfile."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n: int = -1) -> bytes:
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        if end <= self.pos:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end - 1}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        self.pos += len(data)
        return data


def clean(text: str) -> str:
    words = [w.replace("+", "").replace("'", "").rstrip(".-") for w in text.split()]
    return " ".join(w for w in words if w)


def segments(eaf: bytes) -> list[dict]:
    root = ET.fromstring(eaf)
    ts = {t.get("TIME_SLOT_ID"): int(t.get("TIME_VALUE") or -1) for t in root.iter("TIME_SLOT")}
    spans: dict[str, list[tuple[float, float, str]]] = {"interviewee": [], "interviewer": []}
    for tier in root.iter("TIER"):
        kind = tier.get("LINGUISTIC_TYPE_REF") or ""
        who = "interviewee" if "interviewee" in kind else "interviewer" if "interviewer" in kind else None
        if not who:
            continue
        for a in tier.iter("ALIGNABLE_ANNOTATION"):
            t1, t2 = ts.get(a.get("TIME_SLOT_REF1"), -1), ts.get(a.get("TIME_SLOT_REF2"), -1)
            text = re.sub(r"\s+", " ", a.findtext("ANNOTATION_VALUE") or "").strip()
            if t1 >= 0 and t2 > t1 and text:
                spans[who].append((t1 / 1000, t2 / 1000, text))
    segs, cur = [], None
    for s, e, text in sorted(spans["interviewee"]):
        if cur and s - cur["end"] < MAX_GAP_S and max(cur["end"], e) - cur["start"] <= MAX_S:
            cur["end"], cur["orig"] = max(cur["end"], e), cur["orig"] + " " + text
        else:
            if cur:
                segs.append(cur)
            cur = {"start": s, "end": e, "orig": text}
    if cur:
        segs.append(cur)
    out = []
    for g in segs:
        overlap = any(s < g["end"] and e > g["start"] for s, e, _ in spans["interviewer"])
        ref = clean(g["orig"])
        if overlap or re.search(r"[(/§<0-9]", g["orig"]):
            continue
        if MIN_S <= g["end"] - g["start"] <= MAX_S and len(ref.split()) >= MIN_WORDS:
            out.append({**g, "ref": ref})
    return out


def cut_recording(job: tuple[str, str, str, bytes]) -> list[dict]:
    """One training recording: (zip path, wav member, recording id, eaf bytes) -> its segments on disk."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.functional as AF

    zpath, wav, rec, eaf = job
    split = "validation" if rec in VALIDATION else "train"
    segs = segments(eaf)
    with zipfile.ZipFile(zpath) as z:
        audio, sr = sf.read(io.BytesIO(z.read(wav)), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio16 = AF.resample(torch.from_numpy(audio), sr, SR).numpy()
    out_dir = DATA / "lamurre_fi" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, g in enumerate(segs):
        clip = audio16[int(g["start"] * SR) : int(g["end"] * SR)]
        if len(clip) < MIN_S * SR or float(np.abs(clip).max()) < 1e-4:
            continue
        sid = f"{rec}_{i:04d}.wav"
        sf.write(out_dir / sid, clip, SR, subtype="PCM_16")
        seconds = round(len(clip) / SR, 2)
        rows.append(
            {"id": sid, "split": split, "locality": rec, "seconds": seconds, "ref": g["ref"], "orig": g["orig"]}
        )
    return rows


def train(workers: int) -> int:
    raw = DATA / "lamurre_raw"
    eafs = zipfile.ZipFile(raw / "la-murre_eaf.zip")
    eaf_names = {Path(n).stem: n for n in eafs.namelist() if n.endswith(".eaf")}
    jobs = []
    for zpath in sorted(raw.glob("la-murre_wav_*.zip")):
        with zipfile.ZipFile(zpath) as z:
            for wav in (n for n in z.namelist() if n.endswith(".wav")):
                rec = Path(wav).stem
                if rec in eaf_names and rec not in RECORDINGS and rec not in TRAIN_EXCLUDE:
                    jobs.append((str(zpath), wav, rec, eafs.read(eaf_names[rec])))
    print(f"{len(jobs)} recordings", flush=True)
    rows: list[dict] = []
    with ProcessPoolExecutor(workers) as pool:
        for job, res in zip(jobs, pool.map(cut_recording, jobs), strict=True):
            rows += res
            print(f"{job[2]}: {len(res)} segments", flush=True)
    random.seed(0)
    for split in ("train", "validation"):
        part = [r for r in rows if r["split"] == split]
        if split == "validation":
            keep = []
            for rec in sorted(VALIDATION):
                rs = [r for r in part if r["locality"] == rec]
                keep += sorted(random.sample(rs, min(VALIDATION_CAP, len(rs))), key=lambda r: r["id"])
            for r in part:
                if r not in keep:
                    (DATA / "lamurre_fi" / split / r["id"]).unlink(missing_ok=True)
            part = keep
        (DATA / "lamurre_fi" / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part), encoding="utf-8", newline="\n"
        )
        print(f"{split}: {len(part)} segments, {sum(r['seconds'] for r in part) / 3600:.2f} h", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.train:
        return train(args.workers)

    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.functional as AF

    out_dir = DATA / "lamurre_fi" / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(BASE + "la-murre_eaf.zip") as r:
        eafs = zipfile.ZipFile(io.BytesIO(r.read()))
    eaf_names = {Path(n).stem: n for n in eafs.namelist() if n.endswith(".eaf")}
    random.seed(0)
    rows = []
    for rec, package in RECORDINGS.items():
        segs = segments(eafs.read(eaf_names[rec]))
        with zipfile.ZipFile(HttpFile(f"{BASE}la-murre_wav_{package}.zip")) as z:
            wav = next(n for n in z.namelist() if Path(n).name == f"{rec}.wav")
            audio, sr = sf.read(io.BytesIO(z.read(wav)), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio16 = AF.resample(torch.from_numpy(audio), sr, SR).numpy()
        kept = []
        for i, g in enumerate(segs):
            clip = audio16[int(g["start"] * SR) : int(g["end"] * SR)]
            if len(clip) >= MIN_S * SR and float(np.abs(clip).max()) >= 1e-4:
                kept.append((i, g, clip))
        for i, g, clip in sorted(random.sample(kept, min(CAP, len(kept))), key=lambda k: k[0]):
            sid = f"{rec}_{i:04d}.wav"
            sf.write(out_dir / sid, clip, SR, subtype="PCM_16")
            rows.append(
                {"id": sid, "locality": rec, "seconds": round(len(clip) / SR, 2), "ref": g["ref"], "orig": g["orig"]}
            )
        print(f"{rec}: {len(segs)} segments, {len(kept)} usable, kept {min(CAP, len(kept))}", flush=True)
    (DATA / "lamurre_fi" / "test.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n"
    )
    print(f"test: {len(rows)} segments, {sum(r['seconds'] for r in rows) / 3600:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
