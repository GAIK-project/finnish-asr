"""Transcribe the SKN dialect test (600 segments, 2.5 h) with the API services, as the open models were.

The API arms were run on the Ajokortti clip only; this puts them on the dialect test too, with the clip
runs' models and settings (gaik-evals run_asr.py, scripts/run_api_asr.py), one request per segment:

    vertex_gemini-3.5-flash        Gemini 3.5 Flash on Vertex AI, the clip's prompt ("exactly as spoken"),
                                   thinking_level low
    vertex_gemini-3.5-flash-std    the same, with a prompt that asks for standard Finnish word forms,
                                   which is what the SKN reference has (an extra arm, labelled as such)
    openai_gpt-transcribe          OpenAI gpt-transcribe, languages ["fi"]
    openai_gpt-4o-mini-transcribe  OpenAI gpt-4o-mini-transcribe-2025-12-15, language "fi"
    elevenlabs_scribe-v2           ElevenLabs Scribe v2, language_code "fin", no diarization
    google_chirp-3                 Google Cloud Speech-to-Text v2, chirp_3, fi-FI, location eu

The audio is the test split of the published dataset GAIK-project/skn-finnish-dialects-asr (pinned
revision), the reference its human standard-Finnish text. Each arm writes
results/benchmarks/skn_fi/<arm>.jsonl ({"id", "ref", "hyp", "seconds"}: seconds is the request's
latency) and <arm>.meta.json; scripts/score_benchmark.py scores them with the open models. A segment
whose request fails is logged and left out, so a rerun picks it up. A call that succeeds without any
words (Chirp hearing no speech) is written with an empty transcript, which the scorer counts as deletions.

Usage (from the repo root; the keys as for run_api_asr.py, in the repo-root .env):
    uv run --extra api-asr --extra judges --extra hub python scripts/run_api_benchmark.py --list
    uv run --extra api-asr --extra judges --extra hub python scripts/run_api_benchmark.py --arm ID --limit 20
    uv run --extra api-asr --extra judges --extra hub python scripts/run_api_benchmark.py [--arm ID] [--workers 4]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fi_common import RESULTS_DIR  # noqa: E402
from provenance import build_run_metadata, load_repo_env, write_run_metadata  # noqa: E402

load_repo_env(Path(__file__))

DATASET, REVISION = "GAIK-project/skn-finnish-dialects-asr", "ee98f5e16dcb0986a5514b787bee100962be3bf6"
OUT_DIR = RESULTS_DIR / "benchmarks" / "skn_fi"
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
_local = threading.local()


def load_items(limit: int | None) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, "data/test-00000-of-00001.parquet", repo_type="dataset", revision=REVISION)
    rows = pq.read_table(path, columns=["id", "audio", "text_normalized", "seconds"]).to_pylist()
    items = [{"id": r["id"] + ".wav", "ref": r["text_normalized"], "wav": r["audio"]["bytes"]} for r in rows]
    return items[:limit]


def gemini(prompt: str):
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
            model="gemini-3.5-flash",
            contents=[prompt, types.Part.from_bytes(data=item["wav"], mime_type="audio/wav")],
            config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="low")),
        )
        reason = resp.candidates[0].finish_reason.name if resp.candidates and resp.candidates[0].finish_reason else None
        if reason != "STOP" or not (resp.text or "").strip():
            raise RuntimeError(f"finish_reason {reason}, {len(resp.text or '')} chars")
        usage = resp.usage_metadata.model_dump(exclude_none=True) if resp.usage_metadata else {}
        return resp.text.strip(), usage

    return run


def openai_arm(model: str, **params):
    def run(item: dict) -> tuple[str, dict]:
        from openai import OpenAI

        if not hasattr(_local, "openai"):
            _local.openai = OpenAI()
        extra = dict(params)
        extra_body = extra.pop("extra_body", None)
        resp = _local.openai.audio.transcriptions.create(
            model=model, file=(item["id"], item["wav"], "audio/wav"), extra_body=extra_body, **extra
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
        files={"file": (item["id"], item["wav"], "audio/wav")},
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
    resp = _local.chirp.post(
        f"https://eu-speech.googleapis.com/v2/projects/{project}/locations/eu/recognizers/_:recognize",
        json={
            "config": {
                "autoDecodingConfig": {},
                "model": "chirp_3",
                "languageCodes": ["fi-FI"],
                "features": {"enableAutomaticPunctuation": True},
            },
            "content": base64.b64encode(item["wav"]).decode(),
        },
        timeout=300,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    # a result without a transcript means Chirp heard no speech there; that is its answer, not a failure
    text = " ".join(r["alternatives"][0].get("transcript", "").strip() for r in results if r.get("alternatives"))
    return text.strip(), {}


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
    "openai_gpt-transcribe": (
        "gpt-transcribe", "openai", ("OPENAI_API_KEY",), openai_arm("gpt-transcribe", extra_body={"languages": ["fi"]}),
        {"extra_body": {"languages": ["fi"]}},
    ),
    "openai_gpt-4o-mini-transcribe": (
        "gpt-4o-mini-transcribe-2025-12-15", "openai", ("OPENAI_API_KEY",),
        openai_arm("gpt-4o-mini-transcribe-2025-12-15", language="fi"), {"language": "fi"},
    ),
    "elevenlabs_scribe-v2": (
        "scribe_v2", "elevenlabs", ("ELEVENLABS_API_KEY",), elevenlabs,
        {"language_code": "fin", "diarize": False, "tag_audio_events": False},
    ),
    "google_chirp-3": (
        "chirp_3", "google-speech-v2", VERTEX_ENV, chirp,
        {"language_codes": ["fi-FI"], "location": "eu", "punctuation": True},
    ),
}  # fmt: skip


def with_retries(fn, item: dict, tries: int = 5) -> tuple[str, dict]:
    for n in range(tries):
        try:
            return fn(item)
        except Exception as exc:  # noqa: BLE001  (rate limits and transient errors: back off and retry)
            if n == tries - 1:
                raise
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status < 500 and status not in (408, 409, 429):
                raise
            time.sleep(2 ** (n + 1))
    raise AssertionError("unreachable")


def run_arm(arm_id: str, items: list[dict], workers: int, force: bool) -> int:
    model, provider, env, fn, params = ARMS[arm_id]
    out = OUT_DIR / f"{arm_id}.jsonl"
    if force and out.exists():
        out.unlink()
    done = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["id"]] = row
    todo = [it for it in items if it["id"] not in done]
    print(f"[run ] {arm_id}: {len(todo)} to do, {len(done)} cached", flush=True)
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
            except Exception as exc:  # noqa: BLE001  (one failed segment must not lose the rest)
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
    out.write_text("".join(json.dumps(done[i], ensure_ascii=False) + "\n" for i in order), encoding="utf-8",
                   newline="\n")  # fmt: skip
    meta_path = OUT_DIR / f"{arm_id}.meta.json"
    old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() and not force else {}
    meta = {
        "arm_id": arm_id,
        "model": model,
        "provider": provider,
        "params": params,
        "benchmark": "SKN dialect test, 10 held-out municipalities, human standard-Finnish reference",
        "dataset": DATASET,
        "revision": REVISION,
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
    parser.add_argument("--arm", action="append")
    parser.add_argument("--limit", type=int, help="Only the first N segments (a canary run).")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Start the arm's file again.")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    selected = [a for a in ARMS if not args.arm or a in args.arm]
    if args.list:
        for a in selected:
            print(f"{a:32s} {ARMS[a][0]}")
        return 0
    items = load_items(args.limit)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures, ran = 0, []
    for arm_id in selected:
        missing = [e for e in ARMS[arm_id][2] if not os.environ.get(e)]
        if missing:
            print(f"[skip] {arm_id}: missing {', '.join(missing)}")
            continue
        failures += run_arm(arm_id, items, args.workers, args.force)
        ran.append(arm_id)
    if ran:
        write_run_metadata(
            OUT_DIR,
            build_run_metadata(
                app_under_test="Finnish ASR model comparison (SKN dialect test, API services)",
                dataset_path=None,
                runner_path=Path(__file__),
                model_or_deployment=[ARMS[a][0] for a in ran],
                env_names=sorted({e for a in ran for e in ARMS[a][2]}),
                extra={"dataset": DATASET, "revision": REVISION, "arms": ran, "limit": args.limit},
            ),
            name="_run_metadata_api.json",
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
