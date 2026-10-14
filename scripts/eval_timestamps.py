r"""Timestamps and long-form decoding on dialect speech: does training with timestamp tokens help?

Two steps:

    run    (GPU job on LUMI)  bash lumi/lumi.sh run timestamps hf --arm <arm> [--arm ...] [--force]
    score  (anywhere)         uv run python scripts/eval_timestamps.py score

run decodes, per arm, (A) the 600 SKN test segments with return_timestamps=True, and (B) the ten
long-form windows of results/labels/skn_longform.jsonl (about 4 minutes each from the ten test
municipalities, cut from SKN's WAVs) twice: Whisper's own sequential long-form decoding with
timestamps, and the quiet-point chunks of the model card (15-25 s, no timestamps). Raw outputs go to
results/timestamps/<arm>.json, with the GPU and library versions; nothing is scored on LUMI. On LUMI
the test segments come from $FIASR_DATA/skn_fi (prepare_skn.py) and the windows from
$FIASR_DATA/skn_longform.jsonl (lumi.sh push-labels copies results/labels/skn_longform.jsonl there).

score compares with SKN's human annotation:
- WER of every decode (lower-cased, punctuation removed, as score_benchmark.py);
- boundaries: SKN times every sentence. For each sentence start inside a segment or window, the
  distance to the nearest timestamp the model put there; reported as the share within 0.5 s and
  1.0 s and the median distance;
- loops: a decode is flagged when one 5-word sequence occurs four times or more and more often than
  in the reference.
Output: results/timestamps/scores.json, _run_metadata.json and a table. SKN is CC BY 4.0 (Kotus,
via Kielipankki); the raw outputs quote only SKN test material.

Arms: hf_whisper-large-v3 (base), hub_whisper-large-v3-fi-dialects (the published model, downloaded
from the Hub on the login node first), and ft_<name> for any fine-tuned model in $FIASR_MODELS.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import statistics
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from provenance import build_run_metadata, write_run_metadata  # noqa: E402  (standard library only)

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT
DATA = Path(os.environ.get("FIASR_DATA", ROOT / "data"))
MODELS = Path(os.environ.get("FIASR_MODELS", ROOT / "models"))
OUT = PROJECT / "results" / "timestamps"
LABELS = PROJECT / "results" / "labels"
SR = 16000
CHUNK = {"min_s": 15.0, "max_s": 25.0, "window_s": 0.1}
KW = {"language": "fi", "task": "transcribe"}


def arms() -> dict[str, str]:
    out = {
        "hf_whisper-large-v3": "openai/whisper-large-v3",
        "hub_whisper-large-v3-fi-dialects": "GAIK-project/whisper-large-v3-fi-dialects",
    }
    for d in sorted(MODELS.glob("*/config.json")):
        out[f"ft_{d.parent.name}"] = str(d.parent)
    return out


def public_name(arm: str, model: str) -> str:
    """What the result files record: the Hub id, or for a local fine-tune only its directory name (no scratch path)."""
    return Path(model).name if arm.startswith("ft_") else model


def quiet_bounds(n: int, audio) -> list[tuple[int, int]]:
    import numpy as np

    win, start, bounds = int(CHUNK["window_s"] * SR), 0, []
    while n - start > CHUNK["max_s"] * SR:
        lo, hi = start + int(CHUNK["min_s"] * SR), start + int(CHUNK["max_s"] * SR)
        frames = (hi - lo) // win
        energy = (audio[lo : lo + frames * win].reshape(frames, win) ** 2).mean(axis=1)
        cut = lo + int(np.argmin(energy)) * win + win // 2
        bounds.append((start, cut))
        start = cut
    return bounds + [(start, n)]


def longform_audio(row: dict):
    """The window's audio at 16 kHz mono, cut once from SKN's WAV and cached."""
    import numpy as np
    import soundfile as sf

    cache = DATA / "skn_longform" / f"{row['recording']}.wav"
    if not cache.exists():
        import torch
        import torchaudio.functional as AF

        for zpath in sorted((DATA / "skn_raw").glob("skn_*.zip")):
            with zipfile.ZipFile(zpath) as z:
                names = [n for n in z.namelist() if n.endswith(f"/Media/{row['recording']}.wav")]
                if names:
                    audio, sr = sf.read(io.BytesIO(z.read(names[0])), dtype="float32")
                    break
        else:
            raise FileNotFoundError(row["recording"])
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio[int(row["start"] * sr) : int(row["end"] * sr)]
        audio = AF.resample(torch.from_numpy(audio), sr, SR).numpy()
        cache.parent.mkdir(parents=True, exist_ok=True)
        sf.write(cache, audio, SR, subtype="PCM_16")
    audio, _ = sf.read(cache, dtype="float32")
    return np.asarray(audio, dtype="float32")


def chunks_of(res: dict) -> list[dict]:
    return [
        {"text": c["text"].strip(), "start": c["timestamp"][0], "end": c["timestamp"][1]} for c in res.get("chunks", [])
    ]


def run(arm_ids: list[str], force: bool) -> int:
    import soundfile as sf
    import torch
    import transformers
    from transformers import pipeline

    table = arms()
    OUT.mkdir(parents=True, exist_ok=True)
    test = [json.loads(x) for x in (DATA / "skn_fi" / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    lf_path = DATA / "skn_longform.jsonl" if (DATA / "skn_longform.jsonl").exists() else LABELS / "skn_longform.jsonl"
    windows = [json.loads(x) for x in lf_path.read_text(encoding="utf-8").splitlines()]
    for arm in arm_ids:
        out_path = OUT / f"{arm}.json"
        if out_path.exists() and not force:
            print(f"[skip] {arm}: {out_path.name} exists", flush=True)
            continue
        model = table[arm]
        asr = pipeline("automatic-speech-recognition", model=model, dtype=torch.float16, device="cuda")
        t0 = time.perf_counter()
        segs = {}
        for i in range(0, len(test), 8):
            batch = test[i : i + 8]
            audio = [sf.read(DATA / "skn_fi" / "test" / r["id"], dtype="float32")[0] for r in batch]
            res = asr(
                [{"raw": a, "sampling_rate": SR} for a in audio],
                batch_size=8,
                return_timestamps=True,
                generate_kwargs=KW,
            )
            for r, o in zip(batch, res, strict=True):
                segs[r["id"].removesuffix(".wav")] = {"text": o["text"].strip(), "chunks": chunks_of(o)}
        print(f"[run] {arm}: {len(segs)} test segments with timestamps, {time.perf_counter() - t0:.0f} s", flush=True)
        longform = {}
        for row in windows:
            audio = longform_audio(row)
            seq = asr({"raw": audio, "sampling_rate": SR}, return_timestamps=True, generate_kwargs=KW)
            bounds = quiet_bounds(len(audio), audio)
            pieces = asr(
                [{"raw": audio[s:e], "sampling_rate": SR} for s, e in bounds], batch_size=8, generate_kwargs=KW
            )
            longform[row["recording"]] = {
                "sequential": {"text": seq["text"].strip(), "chunks": chunks_of(seq)},
                "chunks": {
                    "text": " ".join(p["text"].strip() for p in pieces),
                    "pieces": [
                        {"start": round(s / SR, 2), "end": round(e / SR, 2), "text": p["text"].strip()}
                        for (s, e), p in zip(bounds, pieces, strict=True)
                    ],
                },
            }
            print(f"[run] {arm}: {row['recording']} done", flush=True)
        record = {
            "arm": arm,
            "model": public_name(arm, model),
            "settings": {"generate": KW, "chunks": CHUNK, "dtype": "float16", "batch_size": 8},
            "gpu": torch.cuda.get_device_name(0),
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
            "test_segments": segs,
            "longform": longform,
        }
        out_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        del asr
        torch.cuda.empty_cache()
    write_run_metadata(
        OUT,
        build_run_metadata(
            app_under_test="Finnish dialect ASR: timestamps and long-form decoding (GPU decode)",
            dataset_path=None,
            runner_path=Path(__file__),
            model_or_deployment=[public_name(a, table[a]) for a in arm_ids],
            extra={
                "arms": arm_ids,
                "host": platform.node(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "gpu": torch.cuda.get_device_name(0),
                "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
            },
        ),
        name="_run_metadata_run.json",
    )
    return 0


def nearest_hits(ref_starts: list[float], pred_starts: list[float]) -> list[float]:
    """For each reference sentence start, the distance to the nearest predicted timestamp (inf if none)."""
    return [min((abs(r - p) for p in pred_starts), default=float("inf")) for r in ref_starts]


def loops(text: str, ref: str) -> bool:
    def grams(t: str) -> Counter:
        w = t.lower().split()
        return Counter(tuple(w[i : i + 5]) for i in range(len(w) - 4))

    hyp, base = grams(text), grams(ref)
    return any(c >= 4 and c > base.get(g, 0) for g, c in hyp.items())


def wer(refs: list[str], hyps: list[str]) -> float:
    import jiwer

    from score_benchmark import transform

    hyps = [h if transform(False)(h)[0] else "-" for h in hyps]
    w = jiwer.process_words(refs, hyps, transform(False), transform(False))
    return round((w.substitutions + w.deletions + w.insertions) / (w.hits + w.substitutions + w.deletions) * 100, 2)


def boundary_stats(dist: list[float]) -> dict:
    found = [d for d in dist if d != float("inf")]
    return {
        "sentence_starts": len(dist),
        "within_0_5s": round(sum(d <= 0.5 for d in dist) / len(dist), 3) if dist else None,
        "within_1s": round(sum(d <= 1.0 for d in dist) / len(dist), 3) if dist else None,
        "median_s": round(statistics.median(found), 2) if found else None,
    }


def score() -> int:
    parts = {}
    for line in (LABELS / "skn_parts.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        parts[r["id"]] = r["parts"]
    refs = {}
    bench = PROJECT / "results" / "benchmarks" / "skn_fi" / "hf_whisper-large-v3.jsonl"
    for line in bench.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        refs[r["id"].removesuffix(".wav")] = r["ref"]
    windows = {
        r["recording"]: r
        for r in map(json.loads, (LABELS / "skn_longform.jsonl").read_text(encoding="utf-8").splitlines())
    }
    result, models, gpus = {}, {}, {}
    for path in sorted(OUT.glob("*.json")):
        if path.name.startswith("_") or path.name == "scores.json":
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        models[d["arm"]], gpus[d["arm"]] = d["model"], d.get("gpu")
        ids = sorted(d["test_segments"])
        seg_dist, with_times = [], 0
        for sid in ids:
            chunks = d["test_segments"][sid]["chunks"]
            pred = [c["start"] for c in chunks if c["start"] is not None]
            with_times += bool(pred)
            ref_starts = [p[0] for p in parts[sid][1:]]  # sentence starts after the first
            if ref_starts:
                seg_dist += nearest_hits(ref_starts, pred[1:] if len(pred) > 1 else [])
        entry = {
            "test_segments": {
                "segments": len(ids),
                "wer_with_timestamps": wer([refs[i] for i in ids], [d["test_segments"][i]["text"] for i in ids]),
                "segments_with_timestamps": with_times,
                "boundaries": boundary_stats(seg_dist),
            },
            "longform": {},
        }
        lf = d["longform"]
        recs = sorted(lf)
        for mode in ("sequential", "chunks"):
            texts = [lf[r][mode]["text"] for r in recs]
            row = {
                "wer": wer([windows[r]["ref"] for r in recs], texts),
                "windows_with_loop": sum(loops(t, windows[r]["ref"]) for r, t in zip(recs, texts, strict=True)),
            }
            if mode == "sequential":
                dist = []
                for r in recs:
                    pred = [c["start"] for c in lf[r][mode]["chunks"] if c["start"] is not None]
                    dist += nearest_hits([p[0] for p in windows[r]["parts"]], pred)
                row["boundaries"] = boundary_stats(dist)
            entry["longform"][mode] = row
        entry["longform"]["windows"] = len(recs)
        result[d["arm"]] = entry
    (OUT / "scores.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_run_metadata(
        OUT,
        build_run_metadata(
            app_under_test="Finnish dialect ASR: timestamps and long-form decoding",
            dataset_path=LABELS / "skn_longform.jsonl",
            runner_path=Path(__file__),
            model_or_deployment=[models[a] for a in result],
            extra={"arms": list(result), "decoded_on": gpus, "chunks": CHUNK, "generate": KW},
        ),
    )
    head = ("segWER", "seg<1s", "LF seq WER", "loops", "LF<1s", "LF chunk WER")
    print(f"{'arm':40s} " + " ".join(f"{h:>{max(6, len(h))}s}" for h in head))
    for arm, e in result.items():
        t, s, c = e["test_segments"], e["longform"]["sequential"], e["longform"]["chunks"]
        print(
            f"{arm:40s} {t['wer_with_timestamps']:6.1f} {t['boundaries']['within_1s'] or 0:6.2f} "
            f"{s['wer']:10.1f} {s['windows_with_loop']:5d} {s['boundaries']['within_1s'] or 0:6.2f} {c['wer']:12.1f}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", nargs="?", default="run", choices=["run", "score"])
    parser.add_argument("--arm", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--env", help="Ignored: lumi/job.sbatch passes it to every runner.")
    args = parser.parse_args()
    if args.cmd == "score":
        return score()
    return run(args.arm or list(arms()), args.force)


if __name__ == "__main__":
    raise SystemExit(main())
