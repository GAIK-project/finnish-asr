"""Download the Ajokortti webinar clip and its two Finnish references from gaik-toolkit.

    uv run python scripts/get_clip.py                          # the clip and both references
    uv run --extra clip python scripts/get_clip.py --prepare-audio   # also the 16 kHz copy for the GPU runners
    uv run python scripts/get_clip.py --ref 54218f5d           # from a fixed gaik-toolkit commit

The files are public in GAIK-project/gaik-toolkit, evaluation_layer/eval_methods/transcription_eval/data/:

    Ajokortti.mp3            180 s of a Finnish dental-education webinar, two speakers
    reference.txt            the University of Helsinki (HY) team's transcript, used for every score
    reference.reviewed.txt   the same with 14 corrections a listener made at the 42 disputed spans

They go to data/ajokortti/ (gitignored) and must match the SHA-256 values in fi_common.CLIP_FILES,
or the script stops and deletes the mismatching file. data/ajokortti/_download.json records the URLs,
hashes and time. --prepare-audio decodes the clip with ffmpeg (on PATH) to 16 kHz mono float32,
data/ajokortti/Ajokortti.16k.npy, which run_open_models.py and run_api_asr.py (Chirp chunks) read,
so the GPU node needs no ffmpeg. No API keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fi_common import AUDIO_16K, AUDIO_PATH, CLIP_DIR, CLIP_FILES, TOOLKIT_RAW, now_utc, rel  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def base_url(ref: str | None) -> str:
    return TOOLKIT_RAW if not ref else TOOLKIT_RAW.replace("/main/", f"/{ref}/", 1)


def download(ref: str | None, force: bool) -> dict:
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    record = {"source": base_url(ref), "files": {}}
    for name, expected in CLIP_FILES.items():
        path = CLIP_DIR / name
        url = base_url(ref) + name
        if path.exists() and not force and sha256(path) == expected:
            print(f"[skip] {name}: already here, SHA-256 matches")
        else:
            print(f"[get ] {url}", flush=True)
            with urllib.request.urlopen(url, timeout=120) as resp:
                path.write_bytes(resp.read())
            got = sha256(path)
            if got != expected:
                path.unlink()
                raise SystemExit(f"{name}: SHA-256 {got} differs from the expected {expected}; file deleted")
        record["files"][name] = {"url": url, "sha256": expected, "bytes": path.stat().st_size}
    record["downloaded_utc"] = now_utc()
    (CLIP_DIR / "_download.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    return record


def prepare_audio() -> None:
    """Decode the clip to 16 kHz mono float32 so the GPU node needs no ffmpeg (QAD-1's recipe)."""
    import numpy as np

    pcm = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(AUDIO_PATH), "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
        check=True,
        capture_output=True,
    ).stdout
    audio = np.frombuffer(pcm, dtype=np.float32)
    AUDIO_16K.parent.mkdir(parents=True, exist_ok=True)
    np.save(AUDIO_16K, audio)
    print(f"wrote {rel(AUDIO_16K)} ({len(audio) / 16000:.1f} s)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", help="gaik-toolkit commit or branch to download from (default: main).")
    parser.add_argument("--force", action="store_true", help="Download again even when the files match.")
    parser.add_argument("--prepare-audio", action="store_true", help="Also write the 16 kHz float32 .npy copy.")
    args = parser.parse_args()
    download(args.ref, args.force)
    if args.prepare_audio:
        prepare_audio()
    print(f"clip and references in {rel(CLIP_DIR)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
