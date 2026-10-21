"""A standard-Finnish layer for LA-murre, as a package for the Language Bank of Finland (Kielipankki).

Kielipankki may distribute modified LA-murre material and has offered to take in this project's
standard-Finnish rewrites as a new version of the corpus. This script collects them per recording with
the times of each segment, so the text can be put back beside the audio and the original annotation.

Segments: prepare_lamurre.py's (interviewee annotations joined while the gap is under 1.5 s and the
segment stays under 25 s; left out when an interviewer annotation overlaps it or it has an
unclear-speech mark), for all 142 recordings. The time is in seconds from the start of the recording's
WAV and EAF; the id is <recording>_<n>, n the segment's index in segments(), the same id the training
and test files use. Times come from the public EAF package alone, so no audio is read.

    uv run python scripts/lamurre_layer.py collect   # data/lamurre_layer/: segments.jsonl and todo.jsonl
    uv run python scripts/lamurre_layer.py package   # data/lamurre_layer/package/: one TSV per recording,
                                                     # README.md and stats.json

`collect` joins every *.targets.jsonl of LA-murre it finds (data/lamurre_fi/, results/labels/lamurre_*)
and writes todo.jsonl with the segments that have no standard-Finnish text yet; prepare_targets.py
normalise writes them (results/labels/lamurre_rest.targets.jsonl). Everything written here is LA-murre
text (CC BY-ND 4.0) and stays under data/ (gitignored); only stats.json's numbers may be quoted.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dialect_regions import lamurre_regions  # noqa: E402
from prepare_lamurre import BASE, RECORDINGS, TRAIN_EXCLUDE, VALIDATION, segments  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "lamurre_layer"
TARGETS = [
    ROOT / "data" / "lamurre_fi" / "train.targets.jsonl",
    ROOT / "data" / "lamurre_fi" / "validation.targets.jsonl",
    *sorted((ROOT / "results" / "labels").glob("lamurre_*.targets.jsonl")),
]


def role(rec: str) -> str:
    """How this project used the recording."""
    if rec in RECORDINGS:
        return "test"
    if rec in TRAIN_EXCLUDE:
        return "held out"
    return "validation" if rec in VALIDATION else "train"


def collect() -> int:
    with urllib.request.urlopen(BASE + "la-murre_eaf.zip") as r:
        eafs = zipfile.ZipFile(io.BytesIO(r.read()))
    targets: dict[str, dict] = {}
    for path in TARGETS:
        if path.exists():
            for row in map(json.loads, path.read_text(encoding="utf-8").splitlines()):
                targets.setdefault(row["id"].removesuffix(".wav"), row)
    rows, todo = [], []
    for name in sorted(n for n in eafs.namelist() if n.endswith(".eaf")):
        rec = Path(name).stem
        for i, g in enumerate(segments(eafs.read(name))):
            sid = f"{rec}_{i:04d}"
            t = targets.get(sid, {})
            row = {
                "id": sid,
                "recording": rec,
                "role": role(rec),
                "start": round(g["start"], 3),
                "end": round(g["end"], 3),
                "orig": g["orig"],
                "ref": g["ref"],
                "target": t.get("target", ""),
                "target_by": t.get("target_by", ""),
            }
            rows.append(row)
            if not row["target"]:
                todo.append({"id": sid, "ref": g["ref"]})
    OUT.mkdir(parents=True, exist_ok=True)
    for fname, part in (("segments.jsonl", rows), ("todo.jsonl", todo)):
        (OUT / fname).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part), encoding="utf-8", newline="\n"
        )
    recs = {r["recording"] for r in rows}
    print(f"{len(recs)} recordings, {len(rows)} segments, {len(rows) - len(todo)} with text, {len(todo)} to write")
    return 0


def package() -> int:
    rows = [json.loads(line) for line in (OUT / "segments.jsonl").read_text(encoding="utf-8").splitlines()]
    regions = lamurre_regions()
    pkg = OUT / "package"
    pkg.mkdir(parents=True, exist_ok=True)
    by_rec: dict[str, list[dict]] = {}
    for r in rows:
        by_rec.setdefault(r["recording"], []).append(r)
    head = "segment\tstart_s\tend_s\tdialect_transcript\tstandard_finnish\twritten_by\n"
    for rec, part in by_rec.items():
        lines = [
            "\t".join([r["id"], f"{r['start']:.3f}", f"{r['end']:.3f}", r["orig"], r["target"], r["target_by"]])
            for r in part
            if r["target"]
        ]
        (pkg / f"{rec}.tsv").write_text(head + "".join(f"{x}\n" for x in lines), encoding="utf-8", newline="\n")
    done = [r for r in rows if r["target"]]
    stats = {
        "recordings": len(by_rec),
        "segments": len(rows),
        "segments_with_standard_text": len(done),
        "hours_with_standard_text": round(sum(r["end"] - r["start"] for r in done) / 3600, 2),
        "written_by": dict(Counter(r["target_by"] for r in done)),
        "by_role": dict(Counter(r["role"] for r in done)),
        "by_region": dict(Counter(regions.get(r["recording"], "?") for r in done)),
    }
    (pkg / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["collect", "package"])
    args = parser.parse_args()
    return collect() if args.step == "collect" else package()


if __name__ == "__main__":
    raise SystemExit(main())
