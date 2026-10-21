"""Finnish parliament speech (Aalto Finnish Parliament ASR Corpus 2008-2020, v2) for testing and training.

The corpus (Kielipankki, http://urn.fi/urn:nbn:fi:lb-2022052002, CLARIN PUB +BY +PRIV +ND: free to use,
derived data not to be distributed) holds 16 kHz WAV segments of plenary sessions with lower-case
transcripts without punctuation; hesitations and repeated words are written out. `lumi.sh download parl`
puts devtest.zip and the 2015-2020 parts in $FIASR_DATA/parl_raw.

    python prepare_parl.py test            # parl_fi: 2016-test/unseen, 962 segments of speakers the
                                           # corpus's 2008-2016 training set does not have
    python prepare_parl.py train --hours H  # parl_train: H hours drawn (seed 0) from the 2015-2020 parts,
                                           # leaving out the parl_fi speakers and every 2020 session
                                           # in the corpus's own 2020 test set

Writes $FIASR_DATA/<set>/<split>/<id>.wav and <split>.jsonl (id, ref, speaker, seconds), the layout
run_benchmark.py and finetune_whisper.py read. Nothing here leaves LUMI or data/ (gitignored).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import wave
import zipfile
from contextlib import ExitStack
from pathlib import Path

DATA = Path(os.environ.get("FIASR_DATA", Path(__file__).resolve().parents[1] / "data"))
RAW = DATA / "parl_raw"
DEVTEST = "fi-parliament-asr/dev-test/"


def write(set_name: str, split: str, rows: list[dict]) -> None:
    path = DATA / set_name / f"{split}.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
    print(f"{set_name}/{split}: {len(rows)} segments, {sum(r['seconds'] for r in rows) / 3600:.2f} h", flush=True)


def seconds(path: Path) -> float:
    with wave.open(str(path)) as w:
        return round(w.getnframes() / w.getframerate(), 2)


def test_speakers(z: zipfile.ZipFile) -> set[str]:
    """Speaker names (as in file names) and parliament ids of the parl_fi test speakers."""
    names = {Path(n).name.rsplit("_", 1)[0] for n in z.read(DEVTEST + "unseen_test.list").decode().split()}
    ids = set()
    for line in z.read(DEVTEST + "dev-test-speakers.csv").decode().splitlines()[1:]:
        new_id, _, old_id, file_name = line.split(",")[:4]
        if file_name in names:
            ids.update({new_id.zfill(5), old_id.zfill(5)})
    return names | ids


def test() -> int:
    z = zipfile.ZipFile(RAW / "devtest.zip")
    out = DATA / "parl_fi" / "test"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in z.read(DEVTEST + "unseen_test.list").decode().split():
        wav = DEVTEST + name
        sid = Path(name).name
        (out / sid).write_bytes(z.read(wav))
        ref = z.read(wav.removesuffix(".wav") + ".trn").decode("utf-8").strip()
        speaker = sid.rsplit("_", 1)[0]
        rows.append({"id": sid, "ref": ref, "speaker": speaker, "seconds": seconds(out / sid)})
    write("parl_fi", "test", rows)
    return 0


def train(hours: float, output_name: str = "parl_train") -> int:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", output_name):
        raise ValueError("output name must be a single dataset directory name")
    destination = DATA / output_name
    if output_name != "parl_train":
        # Recovery attempts must never overwrite the original attempt or another recovery.
        destination.mkdir(parents=True, exist_ok=False)
    test_sessions = set()
    with zipfile.ZipFile(RAW / "devtest.zip") as z:
        held = test_speakers(z)
        for n in z.namelist():
            m = re.search(r"2020-test/(\d+)-2020/", n)
            if m:
                test_sessions.add(f"{int(m.group(1)):03d}-2020")
    pool = []
    for part in sorted(RAW.glob("fi-parliament-asr-2015-2020-part*.zip")):
        with zipfile.ZipFile(part) as z:
            for n in z.namelist():
                if not n.endswith(".wav"):
                    continue
                uid = Path(n).stem  # <mpid>-<session>-<year>-<start>-<end>
                mpid, session, year, start, end = uid.split("-")
                if mpid in held or f"{session}-{year}" in test_sessions:
                    continue
                pool.append((str(part), n, uid, (int(end) - int(start)) / 100))
    random.seed(0)
    random.shuffle(pool)
    out = destination / "train"
    out.mkdir(parents=True, exist_ok=True)
    # draw first, then read in the order the members lie in the zip: random reads in a 60 GB zip on
    # Lustre ran at 15 files a minute
    chosen, total = [], 0.0
    for item in pool:
        if total >= hours * 3600:
            break
        if 2.0 <= item[3] <= 28.0:
            chosen.append(item)
            total += item[3]
    rows, open_zips = [], {}
    offsets = {}
    for part in {c[0] for c in chosen}:
        with zipfile.ZipFile(part) as z:
            offsets[part] = {i.filename: i.header_offset for i in z.infolist()}
    with ExitStack() as stack:
        for part, n, uid, dur in sorted(chosen, key=lambda c: (c[0], offsets[c[0]][c[1]])):
            if part not in open_zips:
                open_zips[part] = stack.enter_context(zipfile.ZipFile(part))
            z = open_zips[part]
            ref = z.read(n.removesuffix(".wav") + ".trn").decode("utf-8").strip()
            if len(ref.split()) < 3:
                continue
            (out / f"{uid}.wav").write_bytes(z.read(n))
            rows.append({"id": f"{uid}.wav", "ref": ref, "speaker": uid.split("-")[0], "seconds": round(dur, 2)})
            if len(rows) % 500 == 0:
                print(f"extracted {len(rows)}/{len(chosen)} segments", flush=True)
    write(output_name, "train", rows)
    print(f"pool {len(pool)} segments; left out {len(held)} test speaker keys, {len(test_sessions)} sessions")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["test", "train"])
    parser.add_argument("--hours", type=float, default=100.0)
    parser.add_argument(
        "--output-name", default="parl_train", help="Fresh dataset directory name for a recovery attempt."
    )
    parser.add_argument("--workers", type=int, help="Ignored: prep.sbatch passes it to every preparer.")
    args = parser.parse_args()
    return test() if args.step == "test" else train(args.hours, args.output_name)


if __name__ == "__main__":
    raise SystemExit(main())
