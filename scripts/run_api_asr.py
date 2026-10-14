"""Transcribe the Ajokortti clip with the API arms that QAD-1 (GAIK's earlier run of this clip) did not run.

    elevenlabs_scribe-v2   ElevenLabs Scribe v2, language_code="fin", whole file, no diarization
    google_chirp-3         Google Cloud Speech-to-Text v2, model chirp_3, fi-FI, location eu.
                           Synchronous recognize takes at most 60 s, so the clip goes in the
                           same chunks as the chunked open models (run_open_models.chunk_bounds).

QAD-1 already has OpenAI (gpt-transcribe, gpt-4o(-mini)-transcribe, whisper-1), Gemini
3.1 Pro / 3.5 Flash and the Haaga-Helia WhisperX server; fi_common.load_arms() brings them in from
results/transcriptions/qad1/.

Usage (from the repo root):
    uv run python scripts/get_clip.py --prepare-audio        # once: the clip, and the 16 kHz copy Chirp needs
    uv run --extra api-asr python scripts/run_api_asr.py --list
    uv run --extra api-asr python scripts/run_api_asr.py [--arm ID] [--repeats 3] [--force]

Needs ELEVENLABS_API_KEY; GOOGLE_VERTEXAI_PROJECT + GOOGLE_APPLICATION_CREDENTIALS for Chirp
(the service account needs Cloud Speech access), in the repo-root .env (.env.example lists the
names). An arm whose keys are missing is skipped.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fi_common import APP_UNDER_TEST, ASR_DIR, AUDIO_PATH, ArmResult, rel, require_clip, save_arm  # noqa: E402
from provenance import build_run_metadata, load_repo_env, write_run_metadata  # noqa: E402
from run_open_models import SR, chunk_bounds, load_audio  # noqa: E402

load_repo_env(Path(__file__))

ELEVEN_MODEL = "scribe_v2"
CHIRP_LOCATION = "eu"


def run_elevenlabs() -> tuple[str, dict, dict]:
    import requests

    with open(AUDIO_PATH, "rb") as f:
        resp = requests.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
            data={"model_id": ELEVEN_MODEL, "language_code": "fin", "tag_audio_events": "false", "diarize": "false"},
            files={"file": (AUDIO_PATH.name, f, "audio/mpeg")},
            timeout=600,
        )
    resp.raise_for_status()
    body = resp.json()
    words = [
        {"word": w["text"], "start": w.get("start"), "end": w.get("end")}
        for w in body.get("words", [])
        if w.get("type") == "word"
    ]
    raw = {k: v for k, v in body.items() if k not in {"text", "words"}}
    return body["text"], {"words": words}, raw


def _wav_bytes(samples) -> bytes:
    import wave

    import numpy as np

    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)
    return buf.getvalue()


def run_chirp() -> tuple[str, dict, dict]:
    import google.auth
    import google.auth.transport.requests
    import requests

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(google.auth.transport.requests.Request())
    project = os.environ["GOOGLE_VERTEXAI_PROJECT"]
    url = (
        f"https://{CHIRP_LOCATION}-speech.googleapis.com/v2/projects/{project}"
        f"/locations/{CHIRP_LOCATION}/recognizers/_:recognize"
    )
    audio = load_audio()
    segs = []
    for s, e in chunk_bounds(audio):
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {creds.token}"},
            json={
                "config": {
                    "autoDecodingConfig": {},
                    "model": "chirp_3",
                    "languageCodes": ["fi-FI"],
                    "features": {"enableAutomaticPunctuation": True},
                },
                "content": base64.b64encode(_wav_bytes(audio[s:e])).decode(),
            },
            timeout=300,
        )
        resp.raise_for_status()
        text = " ".join(
            r["alternatives"][0]["transcript"].strip() for r in resp.json().get("results", []) if r.get("alternatives")
        )
        segs.append({"start": round(s / SR, 2), "end": round(e / SR, 2), "text": text})
    return " ".join(x["text"] for x in segs if x["text"]), {}, {"segments": segs}


ARMS = {
    "elevenlabs_scribe-v2": (
        "ElevenLabs Scribe v2",
        "elevenlabs",
        ELEVEN_MODEL,
        ("ELEVENLABS_API_KEY",),
        run_elevenlabs,
        {"language_code": "fin", "diarize": False, "tag_audio_events": False, "input": "mp3, whole file"},
    ),
    "google_chirp-3": (
        "Google Chirp 3",
        "google-speech-v2",
        "chirp_3",
        ("GOOGLE_VERTEXAI_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS"),
        run_chirp,
        {"language_codes": ["fi-FI"], "location": CHIRP_LOCATION, "punctuation": True, "input": "16 kHz WAV chunks"},
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append")
    parser.add_argument("--repeats", type=int, default=1, help="Runs per arm; extra runs are saved as <arm>__r<n>.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    selected = [a for a in ARMS if not args.arm or a in args.arm]
    if args.list:
        for a in selected:
            print(f"{a:24s} {ARMS[a][2]}")
        return 0
    require_clip("Ajokortti.mp3")
    failures = 0
    for arm_id in selected:
        label, provider, model, env, fn, params = ARMS[arm_id]
        missing = [e for e in env if not os.environ.get(e)]
        if missing:
            print(f"[skip] {arm_id}: missing {', '.join(missing)}")
            continue
        for n in range(1, args.repeats + 1):
            run_id = arm_id if n == 1 else f"{arm_id}__r{n}"
            if (ASR_DIR / f"{run_id}.json").exists() and not args.force:
                print(f"[skip] {run_id}: result exists (use --force)")
                continue
            t0 = time.perf_counter()
            try:
                text, extra, raw = fn()
            except Exception as exc:  # noqa: BLE001
                failures += 1
                detail = getattr(getattr(exc, "response", None), "text", "")[:300]
                print(f"[FAIL] {run_id}: {type(exc).__name__}: {str(exc)[:200]} {detail}")
                continue
            save_arm(
                ArmResult(
                    arm_id=run_id,
                    stage="asr",
                    provider=provider,
                    model=model,
                    text=text.strip(),
                    params=params,
                    input_ref=rel(AUDIO_PATH),
                    label=label,
                    words=extra.get("words") or None,
                    latency_s=round(time.perf_counter() - t0, 2),
                    raw=raw or None,
                    repeat=n,
                )
            )
            print(f"[ ok ] {run_id}: {len(text.split())} words")
    write_run_metadata(
        ASR_DIR,
        build_run_metadata(
            app_under_test=APP_UNDER_TEST,
            dataset_path=AUDIO_PATH,
            runner_path=Path(__file__),
            model_or_deployment=[ARMS[a][2] for a in selected],
            env_names=sorted({e for a in selected for e in ARMS[a][3]}),
        ),
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
