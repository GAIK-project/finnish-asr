"""Transcribe the test sets with the API services, as the open models were (run_benchmark.py).

The API arms were first run on the Ajokortti clip only; this puts them on the other test sets too, with
the clip runs' models and settings (gaik-evals run_asr.py, scripts/run_api_asr.py), one request per item.
--bench picks the set (results/benchmarks/<bench>/):

    skn_fi        SKN dialect test, 600 segments, from the published dataset (pinned revision)
    dental10_fi   ten 180 s dental-webinar clips (HY's QAD-1 set A, Ajokortti among them) with HY's Finnish
                  references, from data/dental10/ (gitignored: <name>.mp3 + <name>.txt)
    fleurs_fi, voxpopuli_fi, lamurre_fi
                  the audio the open models were run on, fetched from LUMI into data/<bench>/ (gitignored);
                  ids and references come from results/benchmarks/<bench>/hf_whisper-large-v3.jsonl

Arms:

    vertex_gemini-3.5-flash        Gemini 3.5 Flash on Vertex AI, the clip's prompt ("exactly as spoken"),
                                   thinking_level low
    vertex_gemini-3.5-flash-std    the same, with a prompt that asks for standard Finnish word forms,
                                   which is what the SKN reference has (an extra arm, labelled as such)
    vertex_gemini-3.1-pro          Gemini 3.1 Pro (preview), the clip's prompt, thinking_level low
    openai_gpt-transcribe          OpenAI gpt-transcribe, languages ["fi"]
    openai_gpt-4o-mini-transcribe  OpenAI gpt-4o-mini-transcribe-2025-12-15, language "fi"
    azure_gpt-4o-transcribe        OpenAI gpt-4o-transcribe through GAIK's Azure OpenAI resource, language "fi"
    azure_whisper                  OpenAI Whisper (whisper-1) through the same Azure resource, language "fi"
    elevenlabs_scribe-v2           ElevenLabs Scribe v2, language_code "fin", no diarization
    google_chirp-3                 Google Cloud Speech-to-Text v2, chirp_3, fi-FI, location eu; items over
                                   55 s go in 15-25 s pieces cut at the quietest point, as for the clip

Each arm writes results/benchmarks/<bench>/<arm>.jsonl ({"id", "ref", "hyp", "seconds"}: seconds is the
request's wall time, retries included) and <arm>.meta.json; scripts/score_benchmark.py scores them with
the open models. An item whose request fails is logged and left out, so a rerun picks it up. A call that
succeeds without any words (Chirp hearing no speech) is written with an empty transcript, which the
scorer counts as deletions.

Usage (from the repo root; the keys as for run_api_asr.py in the repo-root .env, and AZURE_API_KEY,
AZURE_ENDPOINT, AZURE_API_VERSION for the Azure arms):
    uv run --extra api-asr --extra judges --extra hub python scripts/run_api_benchmark.py --list
    uv run --extra api-asr --extra judges --extra hub python scripts/run_api_benchmark.py --bench B --arm ID --limit 20
    uv run --extra api-asr --extra judges --extra hub python scripts/run_api_benchmark.py --bench B [--arm ID] [--workers 4]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fi_common import RESULTS_DIR  # noqa: E402
from provenance import build_run_metadata, load_repo_env, write_run_metadata  # noqa: E402

load_repo_env(Path(__file__))

DATASET, REVISION = "GAIK-project/skn-finnish-dialects-asr", "ee98f5e16dcb0986a5514b787bee100962be3bf6"
DATA = RESULTS_DIR.parent / "data"
BENCHES = {
    "skn_fi": "SKN dialect test, 10 held-out municipalities, human standard-Finnish reference",
    "dental10_fi": "Ten 180 s Finnish dental-webinar clips (HY QAD-1 set A), HY's Finnish references",
    "fleurs_fi": "FLEURS Finnish test, read Wikipedia sentences",
    "voxpopuli_fi": "VoxPopuli Finnish test, European Parliament speech",
    "lamurre_fi": "LA-murre dialect test, 10 localities outside SKN, dialect reference",
}
LONG_S = 55.0  # Chirp's synchronous recognize takes at most 60 s
GEMINI_PROMPT = (  # the clip's prompt, gaik-evals run_asr.py
    "Transcribe this Finnish speech exactly as spoken, in Finnish. Do not translate, summarise or correct the "
    "speakers. Output only the transcript as plain text, without timestamps or speaker labels."
)
GEMINI_PROMPT_STD = (
    "Transcribe this Finnish speech in Finnish. The speaker may talk in a dialect: write every word in its "
    "standard Finnish form, in the order spoken. Do not translate, summarise, leave out or add words. Output "
    "only the transcript as plain text, without timestamps or speaker labels."
)
VERTEX_ENV = ("GOOGLE_VERTEXAI_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS")
AZURE_ENV = ("AZURE_API_KEY", "AZURE_ENDPOINT", "AZURE_API_VERSION")
_local = threading.local()


def load_items(bench: str, limit: int | None) -> list[dict]:
    """[{"id", "ref", "wav": audio bytes, "mime"}] in the order of the open models' result file."""
    if bench == "skn_fi":
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(DATASET, "data/test-00000-of-00001.parquet", repo_type="dataset", revision=REVISION)
        rows = pq.read_table(path, columns=["id", "audio", "text_normalized", "seconds"]).to_pylist()
        items = [
            {"id": r["id"] + ".wav", "ref": r["text_normalized"], "wav": r["audio"]["bytes"], "mime": "audio/wav"}
            for r in rows
        ]
    elif bench == "dental10_fi":
        d = DATA / "dental10"
        items = [
            {"id": p.name, "ref": p.with_suffix(".txt").read_text(encoding="utf-8").strip(), "wav": p.read_bytes(),
             "mime": "audio/mpeg"}
            for p in sorted(d.glob("*.mp3"))
        ]  # fmt: skip
        if len(items) != 10:
            raise SystemExit(f"expected 10 clips in {d}, found {len(items)}")
    else:
        ref_file = RESULTS_DIR / "benchmarks" / bench / "hf_whisper-large-v3.jsonl"
        if not ref_file.exists():
            raise SystemExit(f"{ref_file} is needed for the ids and references (LA-murre's stay local)")
        items = []
        for line in ref_file.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            wav = DATA / bench / row["id"]
            if not wav.exists():
                raise SystemExit(f"missing {wav}: fetch the {bench} audio from LUMI first")
            items.append({"id": row["id"], "ref": row["ref"], "wav": wav.read_bytes(), "mime": "audio/wav"})
    return items[:limit]


def pieces(item: dict) -> list[bytes]:
    """The item as 16 kHz WAV: one piece, or 15-25 s pieces cut at quiet points when longer than LONG_S."""
    import numpy as np

    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"],
        input=item["wav"], capture_output=True, check=True,
    ).stdout  # fmt: skip
    audio = np.frombuffer(pcm, dtype="<i2")
    bounds = [(0, len(audio))]
    if len(audio) > LONG_S * 16000:
        from run_open_models import chunk_bounds

        bounds = chunk_bounds(audio.astype(np.float32) / 32768)
    out = []
    for s, e in bounds:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(audio[s:e].tobytes())
        out.append(buf.getvalue())
    return out


def gemini(prompt: str, model: str = "gemini-3.5-flash"):
    def run(item: dict) -> tuple[str, dict]:
        from google import genai
        from google.genai import types

        if not hasattr(_local, "genai"):
            _local.genai = genai.Client(
                vertexai=True,
                project=os.environ["GOOGLE_VERTEXAI_PROJECT"],
                location=os.environ.get("GOOGLE_VERTEXAI_LOCATION", "global"),
            )
        resp = _local.genai.models.generate_content(
            model=model,
            contents=[prompt, types.Part.from_bytes(data=item["wav"], mime_type=item["mime"])],
            config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="low")),
        )
        reason = resp.candidates[0].finish_reason.name if resp.candidates and resp.candidates[0].finish_reason else None
        if reason != "STOP" or not (resp.text or "").strip():
            raise RuntimeError(f"finish_reason {reason}, {len(resp.text or '')} chars")
        usage = resp.usage_metadata.model_dump(exclude_none=True) if resp.usage_metadata else {}
        return resp.text.strip(), usage

    return run


def openai_arm(model: str, azure: bool = False, **params):
    """OpenAI's transcription API, directly or (azure=True) through the Azure OpenAI deployment named `model`."""

    def run(item: dict) -> tuple[str, dict]:
        from openai import AzureOpenAI, OpenAI

        key = "azure" if azure else "openai"
        if not hasattr(_local, key):
            client = (
                AzureOpenAI(
                    api_key=os.environ["AZURE_API_KEY"],
                    azure_endpoint=os.environ["AZURE_ENDPOINT"],
                    api_version=os.environ["AZURE_API_VERSION"],
                )
                if azure
                else OpenAI()
            )
            setattr(_local, key, client)
        extra = dict(params)
        extra_body = extra.pop("extra_body", None)
        resp = getattr(_local, key).audio.transcriptions.create(
            model=model, file=(item["id"], item["wav"], item["mime"]), extra_body=extra_body, **extra
        )
        data = resp.model_dump() if hasattr(resp, "model_dump") else {"text": str(resp)}
        return (data.get("text") or "").strip(), data.get("usage") or {}

    return run


def elevenlabs(item: dict) -> tuple[str, dict]:
    import requests

    resp = requests.post(
        "https://api.elevenlabs.io/v1/speech-to-text",
        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
        data={"model_id": "scribe_v2", "language_code": "fin", "tag_audio_events": "false", "diarize": "false"},
        files={"file": (item["id"], item["wav"], item["mime"])},
        timeout=300,
    )
    resp.raise_for_status()
    body = resp.json()
    return (body.get("text") or "").strip(), {"audio_duration_secs": body.get("audio_duration_secs")}


def chirp(item: dict) -> tuple[str, dict]:
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    if not hasattr(_local, "chirp"):
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        _local.chirp = AuthorizedSession(creds)  # refreshes the token itself during a long run
    project = os.environ["GOOGLE_VERTEXAI_PROJECT"]
    # Chirp decodes 16-bit PCM WAV, not FLEURS' and VoxPopuli's 32-bit float: pieces() converts with ffmpeg
    pcm16 = item["mime"] == "audio/wav" and item["wav"][20:22] == b"\x01\x00" and item["wav"][34:36] == b"\x10\x00"
    short = pcm16 and len(item["wav"]) < LONG_S * 32000
    texts = []
    for wav in [item["wav"]] if short else pieces(item):
        resp = _local.chirp.post(
            f"https://eu-speech.googleapis.com/v2/projects/{project}/locations/eu/recognizers/_:recognize",
            json={
                "config": {
                    "autoDecodingConfig": {},
                    "model": "chirp_3",
                    "languageCodes": ["fi-FI"],
                    "features": {"enableAutomaticPunctuation": True},
                },
                "content": base64.b64encode(wav).decode(),
            },
            timeout=300,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        # a result without a transcript means Chirp heard no speech there; that is its answer, not a failure
        texts += [r["alternatives"][0].get("transcript", "").strip() for r in results if r.get("alternatives")]
    return " ".join(t for t in texts if t).strip(), {}


# arm -> (model, provider, env names, call, params recorded in the meta file)
ARMS = {
    "vertex_gemini-3.5-flash": (
        "gemini-3.5-flash", "google-vertex", VERTEX_ENV, gemini(GEMINI_PROMPT),
        {"thinking_level": "low", "prompt": GEMINI_PROMPT},
    ),
    "vertex_gemini-3.5-flash-std": (
        "gemini-3.5-flash", "google-vertex", VERTEX_ENV, gemini(GEMINI_PROMPT_STD),
        {"thinking_level": "low", "prompt": GEMINI_PROMPT_STD},
    ),
    "vertex_gemini-3.1-pro": (
        "gemini-3.1-pro-preview", "google-vertex", VERTEX_ENV, gemini(GEMINI_PROMPT, "gemini-3.1-pro-preview"),
        {"thinking_level": "low", "prompt": GEMINI_PROMPT},
    ),
    "openai_gpt-transcribe": (
        "gpt-transcribe", "openai", ("OPENAI_API_KEY",), openai_arm("gpt-transcribe", extra_body={"languages": ["fi"]}),
        {"extra_body": {"languages": ["fi"]}},
    ),
    "openai_gpt-4o-mini-transcribe": (
        "gpt-4o-mini-transcribe-2025-12-15", "openai", ("OPENAI_API_KEY",),
        openai_arm("gpt-4o-mini-transcribe-2025-12-15", language="fi"), {"language": "fi"},
    ),
    "azure_gpt-4o-transcribe": (
        "gpt-4o-transcribe", "azure-openai", AZURE_ENV, openai_arm("gpt-4o-transcribe", azure=True, language="fi"),
        {"language": "fi", "deployment": "gpt-4o-transcribe"},
    ),
    "azure_whisper": (
        "whisper", "azure-openai", AZURE_ENV, openai_arm("whisper", azure=True, language="fi"),
        {"language": "fi", "deployment": "whisper"},
    ),
    "elevenlabs_scribe-v2": (
        "scribe_v2", "elevenlabs", ("ELEVENLABS_API_KEY",), elevenlabs,
        {"language_code": "fin", "diarize": False, "tag_audio_events": False},
    ),
    "google_chirp-3": (
        "chirp_3", "google-speech-v2", VERTEX_ENV, chirp,
        {"language_codes": ["fi-FI"], "location": "eu", "punctuation": True, "long_items": "15-25 s quiet-point pieces"},
    ),
}  # fmt: skip


def with_retries(fn, item: dict, tries: int = 5) -> tuple[str, dict]:
    for n in range(tries):
        try:
            return fn(item)
        except Exception as exc:  # noqa: BLE001  (rate limits and transient errors: back off and retry)
            if n == tries - 1:
                raise
            status = getattr(getattr(exc, "response", None), "status_code", None) or getattr(exc, "status_code", None)
            if status is not None and status < 500 and status not in (408, 409, 429):
                raise
            time.sleep(2 ** (n + 1))
    raise AssertionError("unreachable")


def run_arm(arm_id: str, bench: str, items: list[dict], workers: int, force: bool) -> int:
    model, provider, env, fn, params = ARMS[arm_id]
    out_dir = RESULTS_DIR / "benchmarks" / bench
    out = out_dir / f"{arm_id}.jsonl"
    if force and out.exists():
        out.unlink()
    done = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["id"]] = row
    todo = [it for it in items if it["id"] not in done]
    print(f"[run ] {arm_id} on {bench}: {len(todo)} to do, {len(done)} cached", flush=True)
    usage: dict[str, float] = {}
    failed = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool, out.open("a", encoding="utf-8", newline="\n") as f:

        def call(it: dict) -> tuple[dict, str, dict, float]:
            tb = time.perf_counter()
            hyp, use = with_retries(fn, it)
            return it, hyp, use, round(time.perf_counter() - tb, 3)

        for fut in as_completed([pool.submit(call, it) for it in todo]):
            try:
                it, hyp, use, secs = fut.result()
            except Exception as exc:  # noqa: BLE001  (one failed item must not lose the rest)
                failed += 1
                print(f"[FAIL] {arm_id}: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
                continue
            if not hyp:  # a successful call that returned no words: scored as all deletions
                print(f"[note] {arm_id}: {it['id']}: empty transcript", flush=True)
            row = {"id": it["id"], "ref": it["ref"], "hyp": hyp, "seconds": secs}
            done[it["id"]] = row
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            for k, v in use.items():
                if isinstance(v, int | float) and not isinstance(v, bool):
                    usage[k] = usage.get(k, 0) + v
    order = [it["id"] for it in items if it["id"] in done]
    out.write_text(
        "".join(json.dumps(done[i], ensure_ascii=False) + "\n" for i in order), encoding="utf-8", newline="\n"
    )
    meta_path = out_dir / f"{arm_id}.meta.json"
    old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() and not force else {}
    meta = {
        "arm_id": arm_id,
        "model": model,
        "provider": provider,
        "params": params,
        "benchmark": BENCHES[bench],
        **({"dataset": DATASET, "revision": REVISION} if bench == "skn_fi" else {}),
        "utterances": len(order),
        "failed_this_run": failed,
        "workers": workers,
        "transcribe_s": round(old.get("transcribe_s", 0) + time.perf_counter() - t0, 1),
        "usage": {k: old.get("usage", {}).get(k, 0) + v for k, v in usage.items()} or old.get("usage", {}),
        "gpu": None,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"[ ok ] {arm_id}: {len(order)} of {len(items)} in the file, {failed} failed this run", flush=True)
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", choices=list(BENCHES), default="skn_fi")
    parser.add_argument("--arm", action="append")
    parser.add_argument("--limit", type=int, help="Only the first N items (a canary run).")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Start the arm's file again.")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    selected = [a for a in ARMS if not args.arm or a in args.arm]
    if args.list:
        for a in selected:
            print(f"{a:32s} {ARMS[a][0]}")
        return 0
    items = load_items(args.bench, args.limit)
    out_dir = RESULTS_DIR / "benchmarks" / args.bench
    out_dir.mkdir(parents=True, exist_ok=True)
    failures, ran = 0, []
    for arm_id in selected:
        missing = [e for e in ARMS[arm_id][2] if not os.environ.get(e)]
        if missing:
            print(f"[skip] {arm_id}: missing {', '.join(missing)}")
            continue
        failures += run_arm(arm_id, args.bench, items, args.workers, args.force)
        ran.append(arm_id)
    if ran:
        write_run_metadata(
            out_dir,
            build_run_metadata(
                app_under_test=f"Finnish ASR model comparison ({BENCHES[args.bench]}, API services)",
                dataset_path=None,
                runner_path=Path(__file__),
                model_or_deployment=[ARMS[a][0] for a in ran],
                env_names=sorted({e for a in ran for e in ARMS[a][2]}),
                extra={
                    "bench": args.bench,
                    "arms": ran,
                    "limit": args.limit,
                    **({"dataset": DATASET, "revision": REVISION} if args.bench == "skn_fi" else {}),
                },
            ),
            name="_run_metadata_api.json",
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
