"""Cut a seeded subsample of the Aalto Finnish Parliament ASR Corpus into training segments (CPU job on LUMI).

Why: training on dialects costs some standard Finnish. The `la-murre` revision is the best model on both
dialect tests but scores 11.4 % on VoxPopuli (European Parliament speech) against the published
model's 10.8 %. Plenary speech from the Finnish Parliament is the closest large match for that.

Corpus: Aalto Finnish Parliament ASR Corpus 2008-2020, version 2 (Aalto University, via Kielipankki,
http://urn.fi/urn:nbn:fi:lb-2022052002; resource group http://urn.fi/urn:nbn:fi:lb-2021081105). Copyright
holder: the Parliament of Finland. Audio: CLARIN PUB +BY +PRIV +ND +OTHER; transcripts: CLARIN PUB +BY
+PRIV (the corpus's LICENSE.txt). The 2015-2020 set has 1 780 h of plenary speech from sessions 1/2015 to
104/2020, cut by Aalto's aligner into utterances (16 kHz mono WAV) with a transcript each (.trn): lower
case, no punctuation, numbers as words, and <UNK> where a spoken word did not align to the session
minutes. The text is machine-extracted, not checked by a person; the dev and test sets are.

What is read: package fi-parliament-asr-2015-2020-part1.zip (55 GB, 594 h from sessions of every year
2015-2020) and devtest.zip, both downloaded by `lumi.sh download parl` into $FIASR_DATA/parl_raw.
    train       a seeded sample (seed 0) of --hours hours (default 60) from part 1: utterances of
                2-25 s with at least 3 words and no <UNK> or other <...> token, at most PER_SPEAKER
                utterances per speaker, so the hours spread over the members of parliament
    validation  --validation segments (default 200) of the hand-corrected 2016 dev set, same filters
The 2016 test and 2020 test sets are not read.

Speakers left out: everyone who could have spoken Finnish in the European Parliament plenary in
2009-2020 (VOXPOPULI_SPEAKERS: Finnish MEPs and commissioners of those years). VoxPopuli's test set is
that plenary speech, and a model that has heard the same voices here would be scored on familiar
speakers. Matched by name through the corpus's speaker-id-mapping.csv and dev-test-speakers.csv.

The text targets get punctuation and capitals from prepare_targets.py punct (the same prompt as SKN's)
before training: the corpus text has none, and training on it would teach the model to drop them.

    python prepare_parliament.py [--hours 60] [--validation 200] [--workers 12]

Output: $FIASR_DATA/parl_fi/{train,validation}/<id>.wav and <split>.jsonl (id, split, speaker, session,
seconds, ref), and stats.json. The audio is not published: the ND condition does not allow distributing
works that contain parts of it. The transcripts are PUB +BY +PRIV; this repository commits only the
statistics made from them (results/labels/parl_*.stats.json), not the per-segment texts.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import wave
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SR = 16000
MAX_S, MIN_S, MIN_WORDS = 25.0, 2.0, 3
PER_SPEAKER = 300
DATA = Path(os.environ.get("FIASR_DATA", Path(__file__).resolve().parents[1] / "data"))
PART = "fi-parliament-asr-2015-2020-part1.zip"
DEVTEST = "devtest.zip"
# Everyone who could have spoken Finnish in the European Parliament plenary in 2009-2020, i.e. in
# VoxPopuli's material: the MEPs elected from Finland (terms 2009-2014, 2014-2019, 2019-2024, until the end
# of 2020, substitutes included) and the Finnish commissioners. Sources: the European Parliament's MEP
# directory (https://www.europarl.europa.eu/meps/en/full-list) and Wikipedia's lists of members of the
# European Parliament for Finland, the EP open-data API (data.europarl.europa.eu/api/v2/meps,
# country FI), and for the two ministers the Finnish government's releases on their 2019 plenary
# speeches. 38 of these names are speakers in the corpus (9 % of part 1's utterances).
VOXPOPULI_SPEAKERS: tuple[str, ...] = (
    # MEPs from Finland 2009-2020, substitutes included
    "Alviina Alametsä", "Tarja Cronberg", "Sari Essayah", "Carl Haglund", "Jussi Halla-aho", "Satu Hassi",
    "Heidi Hautala", "Teuvo Hakkarainen", "Eero Heinäluoma", "Laura Huhtasaari", "Ville Itälä",
    "Liisa Jaakonsaari", "Anneli Jäätteenmäki", "Elsi Katainen", "Eija-Riitta Korhola",
    "Miapetra Kumpula-Natri", "Merja Kyllönen", "Henrik Lax", "Lasse Lehtinen", "Silvia Modig",
    "Riitta Myller", "Ville Niinistö", "Reino Paasilinna", "Riikka Pakarinen", "Riikka Manner",
    "Mauri Pekkarinen", "Sirpa Pietikäinen", "Samuli Pohjamo", "Mitro Repo", "Pirkko Ruohonen-Lerner",
    "Petri Sarvamaa", "Esko Seppänen", "Eva-Riitta Siitonen", "Timo Soini", "Alexander Stubb",
    "Hannu Takkula", "Sampo Terho", "Nils Torvalds", "Ari Vatanen", "Mirja Vehkaperä", "Henna Virkkunen",
    "Kyösti Virrankoski", "Paavo Väyrynen",
    # Finnish commissioners 2009-2020
    "Olli Rehn", "Jyrki Katainen", "Jutta Urpilainen",
    # ministers who spoke in the plenary during Finland's 2019 Council presidency
    "Antti Rinne", "Tytti Tuppurainen",
)  # fmt: skip


def norm_name(name: str) -> str:
    """'Anne-Mari Virolainen', 'anne_mari_virolainen' -> 'anne mari virolainen'."""
    return " ".join(re.sub(r"[-_.]", " ", name.casefold()).split())


def clean(text: str) -> str | None:
    """The transcript with whitespace collapsed; None if it has a <...> token (<UNK>: an unaligned word)."""
    words = text.split()
    if any(w.startswith("<") for w in words):
        return None
    return " ".join(words)


def eligible(seconds: float, text: str | None) -> bool:
    return text is not None and MIN_S <= seconds <= MAX_S and len(text.split()) >= MIN_WORDS


def select(items: list[dict], hours: float, per_speaker: int, seed: int = 0) -> list[dict]:
    """A seeded sample of up to `hours` of audio with at most `per_speaker` items per speaker, sorted by id."""
    pool = sorted(items, key=lambda r: r["id"])
    random.Random(seed).shuffle(pool)
    picked, per, total = [], Counter(), 0.0
    for r in pool:
        if total >= hours * 3600:
            break
        if per[r["speaker"]] < per_speaker:
            picked.append(r)
            per[r["speaker"]] += 1
            total += r["seconds"]
    return sorted(picked, key=lambda r: r["id"])


def read_texts(job: tuple[str, list[str]]) -> dict[str, str]:
    """(zip path, .trn member names) -> their texts; one process reads one chunk."""
    zpath, members = job
    with zipfile.ZipFile(zpath) as z:
        return {m: z.read(m).decode("utf-8") for m in members}


def write_wavs(zpath: Path, rows: list[dict], members: dict[str, str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        for r in rows:
            data = z.read(members[r["id"]])
            with wave.open(io.BytesIO(data)) as w:
                if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SR, 1, 2):
                    raise ValueError(f"{r['id']}: {w.getframerate()} Hz, {w.getnchannels()} ch, {w.getsampwidth()} B")
            (out_dir / f"{r['id']}.wav").write_bytes(data)


def speaker_names(z: zipfile.ZipFile, member: str, id_col: str, name_col: str) -> dict[str, str]:
    name = next(n for n in z.namelist() if n.endswith(member))
    rows = csv.DictReader(io.StringIO(z.read(name).decode("utf-8")), skipinitialspace=True)
    return {r[id_col].strip(): r[name_col].strip() for r in rows if r.get(id_col) and r.get(name_col)}


def train_items(zpath: Path, workers: int, excluded: set[str]) -> tuple[list[dict], dict[str, str], dict]:
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        mapping = speaker_names(z, "speaker-id-mapping.csv", "New speaker ID", "Name")
    out_ids = {i.zfill(5) for i, n in mapping.items() if norm_name(n) in excluded}
    wavs = {Path(n).stem: n for n in names if n.endswith(".wav")}
    trns = sorted(n for n in names if n.endswith(".trn") and Path(n).stem in wavs)
    chunks = [(str(zpath), trns[i : i + 5000]) for i in range(0, len(trns), 5000)]
    texts: dict[str, str] = {}
    with ProcessPoolExecutor(workers) as pool:
        for part in pool.map(read_texts, chunks):
            texts.update(part)
    items, dropped = [], Counter()
    for member, raw in texts.items():
        stem = Path(member).stem  # <mpid>-<session>-<year>-<start cs>-<end cs>
        mpid, session, year, start, end = stem.split("-")
        seconds = (int(end) - int(start)) / 100
        text = clean(raw)
        if mpid in out_ids:
            dropped["voxpopuli_speaker"] += 1
        elif text is None:
            dropped["unk_token"] += 1
        elif not eligible(seconds, text):
            dropped["length"] += 1
        else:
            row = {"id": stem, "split": "train", "speaker": mpid, "session": f"{session}/{year}", "seconds": seconds}
            items.append({**row, "ref": text})
    info = {
        "utterances": len(texts),
        "speakers": len({Path(m).stem.split("-")[0] for m in texts}),
        "excluded_speakers": sorted(mapping[i.lstrip("0")] for i in out_ids if i.lstrip("0") in mapping),
        "dropped": dict(dropped),
    }
    return items, wavs, info


def validation_items(zpath: Path, excluded: set[str]) -> tuple[list[dict], dict[str, str]]:
    items, wavs = [], {}
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            n = info.filename
            if "/2016-dev/" not in n or not n.endswith(".trn"):
                continue
            stem = Path(n).stem  # <first>_<last>_<sample id>
            speaker = stem.rsplit("_", 1)[0]
            wav = n[: -len(".trn")] + ".wav"
            if norm_name(speaker) in excluded or wav not in z.NameToInfo:
                continue
            seconds = round((z.getinfo(wav).file_size - 44) / (2 * SR), 2)
            text = clean(z.read(n).decode("utf-8"))
            if eligible(seconds, text):
                wavs[stem] = wav
                row = {"id": stem, "split": "validation", "speaker": speaker, "session": "2016-dev", "seconds": seconds}
                items.append({**row, "ref": text})
    return items, wavs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=DATA / "parl_raw")
    parser.add_argument("--hours", type=float, default=60.0)
    parser.add_argument("--validation", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not VOXPOPULI_SPEAKERS:
        raise SystemExit("VOXPOPULI_SPEAKERS is empty: fill it before cutting training data")
    excluded = {norm_name(n) for n in VOXPOPULI_SPEAKERS}
    out = DATA / "parl_fi"

    items, wavs, info = train_items(args.raw / PART, args.workers, excluded)
    train = select(items, args.hours, PER_SPEAKER)
    write_wavs(args.raw / PART, train, wavs, out / "train")
    val_items, val_wavs = validation_items(args.raw / DEVTEST, excluded)
    val = sorted(random.Random(0).sample(val_items, min(args.validation, len(val_items))), key=lambda r: r["id"])
    write_wavs(args.raw / DEVTEST, val, val_wavs, out / "validation")

    stats = {"source": {"package": PART, **info, "eligible": len(items)}}
    for split, rows in (("train", train), ("validation", val)):
        (out / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n"
        )
        stats[split] = {
            "segments": len(rows),
            "hours": round(sum(r["seconds"] for r in rows) / 3600, 2),
            "speakers": len({r["speaker"] for r in rows}),
            "words": sum(len(r["ref"].split()) for r in rows),
            "years": dict(sorted(Counter(r["session"].split("/")[-1] for r in rows).items())),
        }
        print(f"{split}: {stats[split]['segments']} segments, {stats[split]['hours']} h", flush=True)
    (out / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(stats["source"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
