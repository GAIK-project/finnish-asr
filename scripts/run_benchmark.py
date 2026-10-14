"""Run the open models on public Finnish test sets on a GPU (LUMI).

    fleurs_fi      google/fleurs fi_fi test, 918 recordings, CC-BY-4.0. Wikipedia sentences
                   read aloud, about 10 s each.
    voxpopuli_fi   facebook/voxpopuli fi test, 478 recordings, CC0. European Parliament
                   speeches: prepared but real speech, formal Finnish.
    skn_fi         Suomen kielen näytteitä, 10 municipalities held out of training, 600 dialect
                   segments of 2-25 s; reference = SKN's normalised standard-Finnish text.
                   Made by prepare_skn.py; --download does not fetch it.
    lamurre_fi     LA-murre (Finnish Dialect Syntax Archive), 10 localities outside SKN, test only;
                   reference = coarse dialect transcript. Made by prepare_lamurre.py.
FLEURS and VoxPopuli are sets that model cards report, so these numbers can be put next to published
ones. Neither is spontaneous or dialect speech.

Each model transcribes every utterance on its own (no chunking needed), Finnish forced
where the model takes a language. Output is one JSON line per utterance, written as it
goes, so a job that hits its time limit resumes where it stopped.

    python run_benchmark.py --list
    python run_benchmark.py --download --env hf     # login node: weights and the test set
    python run_benchmark.py --env hf                # GPU node
    python run_benchmark.py --env hf --limit 50     # a quick sample
    python run_benchmark.py --env hf --bench voxpopuli_fi

Output: results/benchmarks/<bench>/<arm_id>.jsonl (id, ref, hyp, seconds), <arm_id>.meta.json and
_run_metadata_<env>.json. Score locally with score_benchmark.py. The lamurre_fi *.jsonl files hold
LA-murre's texts (CC BY-ND 4.0) and are gitignored: only its scores.json and *.meta.json are published.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fi_common import RESULTS_DIR  # noqa: E402
from provenance import build_run_metadata, write_run_metadata  # noqa: E402  (standard library only)
from run_open_models import RASMUS, _device, _hf_file, lib_versions  # noqa: E402

SR = 16000
BENCHES = {
    "fleurs_fi": "google/fleurs fi_fi test",
    "voxpopuli_fi": "facebook/voxpopuli fi test",
    "skn_fi": "SKN dialect test: 10 held-out municipalities, normalised text (prepare_skn.py)",
    "lamurre_fi": "LA-murre dialect test: 10 localities outside SKN, dialect text (prepare_lamurre.py)",
}
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("FIASR_DATA", ROOT / "data"))
MODELS = Path(os.environ.get("FIASR_MODELS", ROOT / "models"))  # fine-tuned models (finetune_whisper.py)
BATCH = 16

# arm_id -> (env, model, recipe)
ARMS: dict[str, tuple[str, str, str]] = {
    "hf_whisper-large-v3": ("hf", "openai/whisper-large-v3", "whisper"),
    "hf_whisper-large-v3-turbo": ("hf", "openai/whisper-large-v3-turbo", "whisper"),
    "hf_whisper-large-finnish-v3": ("hf", "Finnish-NLP/whisper-large-finnish-v3", "whisper"),
    "ft_whisper-large-v3-fi-pilot": ("hf", str(MODELS / "whisper-large-v3-fi-pilot"), "whisper"),
    "ft_whisper-large-v3-fi-skn": ("hf", str(MODELS / "whisper-large-v3-fi-skn"), "whisper"),
    "hf_wav2vec2-xlsr-1b-finnish-lm-v2": ("hf", "Finnish-NLP/wav2vec2-xlsr-1b-finnish-lm-v2", "pipeline"),
    "hf_parakeet-tdt-0.6b-v3": ("hf", "nvidia/parakeet-tdt-0.6b-v3", "pipeline"),
    "hf_orukeet": ("hf", "oruk/orukeet", "pipeline"),
    "qwen_qwen3-asr-1.7b": ("qwen", "Qwen/Qwen3-ASR-1.7B", "qwen"),
    "nemo_canary-1b-v2": ("nemo", "nvidia/canary-1b-v2", "canary"),
    "nemo_rasmus-canary-fi-v2": ("nemo", f"{RASMUS}:models/canary-finnish-v2.nemo", "canary"),
    "nemo_rasmus-canary-fi-v2-kenlm": ("nemo", f"{RASMUS}:models/canary-finnish-v2.nemo", "canary_kenlm"),
}
# every other fine-tuned model in $FIASR_MODELS is an arm too: ft_<directory name>
for _d in sorted(MODELS.glob("*/config.json")):
    ARMS.setdefault(f"ft_{_d.parent.name}", ("hf", str(_d.parent), "whisper"))

WHISPER_FILES = ["*.json", "*.txt", "model.safetensors"]
EXTRA_DOWNLOADS = {"openai/whisper-large-v3": WHISPER_FILES, "Finnish-NLP/whisper-large-finnish-v3": WHISPER_FILES}


def download_data() -> None:
    from huggingface_hub import hf_hub_download

    d = DATA_ROOT / "fleurs_fi"
    d.mkdir(parents=True, exist_ok=True)
    tsv = hf_hub_download("google/fleurs", "data/fi_fi/test.tsv", repo_type="dataset")
    (d / "test.tsv").write_bytes(Path(tsv).read_bytes())
    if not (d / "test").exists():
        tar = hf_hub_download("google/fleurs", "data/fi_fi/audio/test.tar.gz", repo_type="dataset")
        with tarfile.open(tar) as tf:
            tf.extractall(d, filter="data")

    import pyarrow.parquet as pq

    d = DATA_ROOT / "voxpopuli_fi"
    (d / "test").mkdir(parents=True, exist_ok=True)
    src = hf_hub_download(
        "facebook/voxpopuli", "fi/test/0000.parquet", repo_type="dataset", revision="refs/convert/parquet"
    )
    rows = []
    table = pq.read_table(src, columns=["audio_id", "audio", "raw_text", "normalized_text"])
    for i, row in enumerate(table.to_pylist()):
        name = f"{i:04d}.wav"  # the original names contain colons
        (d / "test" / name).write_bytes(row["audio"]["bytes"])
        rows.append({"id": name, "audio_id": row["audio_id"], "ref": row["raw_text"] or row["normalized_text"]})
    (d / "test.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n"
    )
    for bench in ("fleurs_fi", "voxpopuli_fi"):
        print(f"{bench}: {len(load_items(bench))} utterances in {DATA_ROOT / bench}")


def load_items(bench: str) -> list[dict]:
    d = DATA_ROOT / bench
    items = []
    if bench == "fleurs_fi":  # tsv: id, file name, raw transcription, normalised transcription, ...
        with open(d / "test.tsv", encoding="utf-8", newline="") as f:
            for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                if (d / "test" / row[1]).exists():
                    items.append({"id": row[1], "ref": row[2], "path": str(d / "test" / row[1])})
    else:
        for line in (d / "test.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["ref"].strip():
                items.append({"id": row["id"], "ref": row["ref"], "path": str(d / "test" / row["id"])})
    return items


def read_audio(path: str):
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        raise ValueError(f"{path}: {sr} Hz, expected {SR}")
    return audio


def make_transcriber(arm_id: str):
    """Load the model once; return fn(list of paths) -> list of texts."""
    _, model, recipe = ARMS[arm_id]
    dev = _device()
    if recipe in {"whisper", "pipeline"}:
        import torch
        from transformers import pipeline

        kw = {"dtype": torch.float16} if recipe == "whisper" and dev == "cuda" else {}
        if recipe == "whisper":
            kw["chunk_length_s"] = 30  # a few clips run past Whisper's 30 s window, which a batch cannot mix in
        asr = pipeline("automatic-speech-recognition", model=model, device=dev, **kw)
        gen = {"generate_kwargs": {"language": "fi", "task": "transcribe"}} if recipe == "whisper" else {}

        def run(paths: list[str]) -> list[str]:
            inputs = [{"raw": read_audio(p), "sampling_rate": SR} for p in paths]
            return [r["text"].strip() for r in asr(inputs, batch_size=len(inputs), **gen)]

        return run
    if recipe == "qwen":
        import torch
        from qwen_asr import Qwen3ASRModel

        m = Qwen3ASRModel.from_pretrained(
            model, dtype=torch.bfloat16, device_map="cuda:0", max_inference_batch_size=BATCH, max_new_tokens=512
        )

        def run(paths: list[str]) -> list[str]:
            res = m.transcribe(audio=[(read_audio(p), SR) for p in paths], language=["Finnish"] * len(paths))
            return [r.text.strip() for r in res]

        return run

    from nemo.collections.asr.models import ASRModel, EncDecMultiTaskModel

    asr = EncDecMultiTaskModel.restore_from(_hf_file(model)) if ":" in model else ASRModel.from_pretrained(model)
    asr = asr.eval().to(dev)
    lm = recipe == "canary_kenlm"  # the model card decodes one utterance at a time with the LM
    if recipe == "canary_kenlm":
        from omegaconf import OmegaConf

        beam = {"beam_size": 5, "ngram_lm_model": _hf_file(f"{RASMUS}:models/kenlm_5M.nemo"), "ngram_lm_alpha": 0.2}
        asr.change_decoding_strategy(decoding_cfg=OmegaConf.create({"strategy": "beam", "beam": beam, "batch_size": 1}))

    def run(paths: list[str]) -> list[str]:
        out = asr.transcribe(
            audio=paths,
            batch_size=1 if lm else len(paths),
            taskname="asr",
            source_lang="fi",
            target_lang="fi",
            pnc="yes",
        )
        return [(o.text if hasattr(o, "text") else str(o)).strip() for o in out]

    return run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=["hf", "qwen", "nemo"])
    parser.add_argument("--arm", action="append")
    parser.add_argument("--bench", choices=list(BENCHES), default="fleurs_fi")
    parser.add_argument("--limit", type=int, help="Only the first N utterances.")
    parser.add_argument("--force", action="store_true", help="Start an arm's output file again.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--download", action="store_true", help="Fetch the test set and missing weights, then exit.")
    args = parser.parse_args()

    selected = [a for a in ARMS if (not args.arm or a in args.arm) and (not args.env or ARMS[a][0] == args.env)]
    if not args.list:  # a fine-tuned arm exists only once its model directory does
        selected = [a for a in selected if not a.startswith("ft_") or Path(ARMS[a][1]).exists()]
    if args.list:
        for a in selected:
            print(f"{a:36s} {ARMS[a][0]:5s} {ARMS[a][1]}")
        return 0
    if args.download:
        from huggingface_hub import snapshot_download

        download_data()
        for repo, patterns in EXTRA_DOWNLOADS.items():
            if any(ARMS[a][1] == repo for a in selected):
                print(f"downloading {repo} ...", flush=True)
                snapshot_download(repo, allow_patterns=patterns)
        print("done")
        return 0

    items = load_items(args.bench)[: args.limit]
    out_dir = RESULTS_DIR / "benchmarks" / args.bench
    out_dir.mkdir(parents=True, exist_ok=True)
    versions = lib_versions()
    print(f"device={_device()} {versions} utterances={len(items)}", flush=True)
    failures = 0
    for arm_id in selected:
        out = out_dir / f"{arm_id}.jsonl"
        if args.force and out.exists():
            out.unlink()
        done = (
            {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()} if out.exists() else set()
        )
        todo = [it for it in items if it["id"] not in done]
        if not todo:
            print(f"[skip] {arm_id}: all {len(items)} done")
            continue
        print(f"[run ] {arm_id}: {len(todo)} to do ...", flush=True)
        try:
            t0 = time.perf_counter()
            run = make_transcriber(arm_id)
            load_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            with open(out, "a", encoding="utf-8", newline="\n") as f:
                for i in range(0, len(todo), BATCH):
                    chunk = todo[i : i + BATCH]
                    tb = time.perf_counter()
                    hyps = run([it["path"] for it in chunk])
                    per = round((time.perf_counter() - tb) / len(chunk), 3)
                    for it, hyp in zip(chunk, hyps, strict=True):
                        row = {"id": it["id"], "ref": it["ref"], "hyp": hyp, "seconds": per}
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
            import torch

            meta = {
                "arm_id": arm_id,
                "model": Path(ARMS[arm_id][1]).name if arm_id.startswith("ft_") else ARMS[arm_id][1],
                "recipe": ARMS[arm_id][2],
                "benchmark": BENCHES[args.bench],
                "utterances": len(items),
                "batch_size": BATCH,
                "load_s": round(load_s, 1),
                "transcribe_s": round(time.perf_counter() - t1, 1),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "host": platform.node(),
                "versions": versions,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            }
            (out_dir / f"{arm_id}.meta.json").write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n"
            )
            print(f"[ ok ] {arm_id}: {meta['transcribe_s']} s for {len(todo)} utterances", flush=True)
        except Exception as exc:  # noqa: BLE001 - one model failing must not stop the others
            failures += 1
            print(f"[FAIL] {arm_id}: {type(exc).__name__}: {str(exc)[:500]}", flush=True)
    write_run_metadata(
        out_dir,
        build_run_metadata(
            app_under_test=f"Finnish ASR public benchmarks: {BENCHES[args.bench]}",
            dataset_path=None,
            runner_path=Path(__file__),
            model_or_deployment=[Path(ARMS[a][1]).name if a.startswith("ft_") else ARMS[a][1] for a in selected],
            extra={
                "bench": args.bench,
                "env": args.env,
                "utterances": len(items),
                "host": platform.node(),
                "versions": versions,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
        ),
        name=f"_run_metadata_{args.env or 'all'}.json",
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
