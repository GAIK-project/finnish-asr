"""Run the open Finnish-capable ASR models on a GPU (LUMI or Roihu) and save them as arms.

Every arm gets the same 180 s Ajokortti clip, decoded once to 16 kHz mono float32
(data/ajokortti/Ajokortti.16k.npy, from `uv run --extra clip python scripts/get_clip.py --prepare-audio`;
`lumi.sh push` copies it to LUMI), with Finnish forced wherever the model takes a language. No prompt,
word list or language model built from the reference.

Arms need one of three Python environments, because their libraries pin different
transformers versions (lumi/setup.sh builds them):

    env   arm                                   model
    hf    hf_whisper-large-v3-turbo             openai/whisper-large-v3-turbo
    hf    hf_wav2vec2-xlsr-1b-finnish-lm-v2     Finnish-NLP/wav2vec2-xlsr-1b-finnish-lm-v2 (greedy CTC, no KenLM)
    hf    hf_parakeet-tdt-0.6b-v3               nvidia/parakeet-tdt-0.6b-v3 (transformers ParakeetForTDT)
    hf    hf_orukeet                            oruk/orukeet (Parakeet v3 fine-tune, transformers export)
    qwen  qwen_qwen3-asr-1.7b                   Qwen/Qwen3-ASR-1.7B (qwen-asr package, language="Finnish")
    qwen  qwen_qwen3-asr-0.6b                   Qwen/Qwen3-ASR-0.6B
    nemo  nemo_canary-1b-v2                     nvidia/canary-1b-v2 (NeMo, fi -> fi, pnc)
    nemo  nemo_rasmus-canary-fi-v2              RASMUS/Finnish-ASR-Canary-v2 R2, greedy
    nemo  nemo_rasmus-canary-fi-v2-kenlm        the same with its 5M-sentence Finnish KenLM (beam 5, alpha 0.2)

Long audio: Whisper and Qwen3-ASR take the whole clip (their own long-form handling).
The others get the same chunks: cut at the quietest 100 ms between 15 and 25 s into
each chunk (energy based, no VAD model, so nothing has to be downloaded on the GPU
node). The chunk boundaries are saved with each result.

Usage:
    python run_open_models.py --list
    python run_open_models.py --download --env hf      # login node: weights into $HF_HOME
    python run_open_models.py --env hf                 # GPU node: every hf arm without a result
    python run_open_models.py --arm qwen_qwen3-asr-1.7b --force

Output: results/transcriptions/<arm_id>.json and results/gpu_run/_run_metadata_<env>.json (GPU, host,
library versions, Slurm job id).
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fi_common import (  # noqa: E402
    APP_UNDER_TEST,
    ASR_DIR,
    AUDIO_16K,
    AUDIO_PATH,
    RESULTS_DIR,
    ArmResult,
    rel,
    save_arm,
)
from provenance import build_run_metadata, write_run_metadata  # noqa: E402  (standard library only)

SR = 16000
RASMUS = "RASMUS/Finnish-ASR-Canary-v2"
MODELS = Path(os.environ.get("FIASR_MODELS", Path(__file__).resolve().parents[1] / "models"))  # finetune_whisper.py

# arm_id -> (env, model, recipe, label)
ARMS: dict[str, tuple[str, str, str, str]] = {
    "hf_whisper-large-v3-turbo": ("hf", "openai/whisper-large-v3-turbo", "whisper", "Whisper large-v3 turbo"),
    "ft_whisper-large-v3-fi-pilot": (
        "hf",
        str(MODELS / "whisper-large-v3-fi-pilot"),
        "whisper",
        "Whisper large-v3 + LoRA on FLEURS and VoxPopuli (pilot)",
    ),
    "ft_whisper-large-v3-fi-skn": (
        "hf",
        str(MODELS / "whisper-large-v3-fi-skn"),
        "whisper",
        "Whisper large-v3 + LoRA on SKN dialects and open data",
    ),
    "hf_wav2vec2-xlsr-1b-finnish-lm-v2": (
        "hf",
        "Finnish-NLP/wav2vec2-xlsr-1b-finnish-lm-v2",
        "ctc",
        "Finnish-NLP wav2vec2 XLS-R 1B (no LM)",
    ),
    "hf_parakeet-tdt-0.6b-v3": ("hf", "nvidia/parakeet-tdt-0.6b-v3", "tdt", "Parakeet TDT 0.6B v3"),
    "hf_orukeet": ("hf", "oruk/orukeet", "tdt", "Orukeet (Parakeet v3 fine-tune)"),
    "qwen_qwen3-asr-1.7b": ("qwen", "Qwen/Qwen3-ASR-1.7B", "qwen", "Qwen3-ASR 1.7B"),
    "qwen_qwen3-asr-0.6b": ("qwen", "Qwen/Qwen3-ASR-0.6B", "qwen", "Qwen3-ASR 0.6B"),
    "nemo_canary-1b-v2": ("nemo", "nvidia/canary-1b-v2", "canary", "Canary 1B v2"),
    "nemo_rasmus-canary-fi-v2": (
        "nemo",
        f"{RASMUS}:models/canary-finnish-v2.nemo",
        "canary",
        "RASMUS Finnish Canary v2 (greedy)",
    ),
    "nemo_rasmus-canary-fi-v2-kenlm": (
        "nemo",
        f"{RASMUS}:models/canary-finnish-v2.nemo",
        "canary_kenlm",
        "RASMUS Finnish Canary v2 + KenLM",
    ),
}
# every other fine-tuned model in $FIASR_MODELS is an arm too: ft_<directory name>
for _d in sorted(MODELS.glob("*/config.json")):
    ARMS.setdefault(f"ft_{_d.parent.name}", ("hf", str(_d.parent), "whisper", f"Fine-tuned {_d.parent.name}"))
# The same Whisper models fed the energy chunks instead of Whisper's own long-form loop: a
# fine-tune trained without timestamp tokens can fall into a repetition loop in that loop.
ARMS["hf_whisper-large-v3-chunks"] = ("hf", "openai/whisper-large-v3", "whisper_chunks", "Whisper large-v3, chunked")
for _a in [a for a in ARMS if a.startswith("ft_")]:
    ARMS[f"{_a}-chunks"] = ("hf", ARMS[_a][1], "whisper_chunks", f"{ARMS[_a][3]}, chunked")

CHUNK = {"min_s": 15.0, "max_s": 25.0, "window_s": 0.1}

RECIPES = {
    "whisper": {
        "language": "fi",
        "task": "transcribe",
        "dtype": "float16",
        "long_form": "sequential, return_timestamps=True",
    },
    "whisper_chunks": {"language": "fi", "task": "transcribe", "dtype": "float16", "chunks": CHUNK},
    # The repo ships a KenLM for pyctcdecode, but kenlm does not build in LUMI's container, so greedy CTC.
    "ctc": {"decoder": "greedy CTC (the repo's KenLM is not used)", "chunks": CHUNK},
    "tdt": {
        "decoder": "greedy TDT (transformers pipeline)",
        "language": "auto (model has no language input)",
        "chunks": CHUNK,
    },
    "qwen": {"language": "Finnish", "dtype": "bfloat16", "max_new_tokens": 4096, "backend": "transformers"},
    "canary": {"source_lang": "fi", "target_lang": "fi", "pnc": "yes", "decoding": "greedy", "chunks": CHUNK},
    "canary_kenlm": {
        "source_lang": "fi",
        "target_lang": "fi",
        "pnc": "yes",
        "decoding": {
            "strategy": "beam",
            "beam_size": 5,
            "ngram_lm_model": f"{RASMUS}:models/kenlm_5M.nemo",
            "ngram_lm_alpha": 0.2,
        },
        "chunks": CHUNK,
    },
}

DOWNLOADS = {
    "openai/whisper-large-v3-turbo": ["*.json", "*.txt", "model.safetensors"],
    "Finnish-NLP/wav2vec2-xlsr-1b-finnish-lm-v2": ["*.json", "*.txt", "model.safetensors", "language_model/*"],
    "nvidia/parakeet-tdt-0.6b-v3": ["*.json", "*.model", "model.safetensors"],
    "oruk/orukeet": ["*.json", "model.safetensors"],
    "Qwen/Qwen3-ASR-1.7B": None,
    "Qwen/Qwen3-ASR-0.6B": None,
    "nvidia/canary-1b-v2": ["*.nemo", "*.json"],
    RASMUS: ["models/canary-finnish-v2.nemo", "models/kenlm_5M.nemo"],
}


def load_audio():
    import numpy as np

    if not AUDIO_16K.exists():
        hint = "uv run --extra clip python scripts/get_clip.py --prepare-audio"
        raise SystemExit(f"missing {rel(AUDIO_16K)}: run {hint}")
    return np.load(AUDIO_16K).astype("float32")


def chunk_bounds(audio) -> list[tuple[int, int]]:
    """Split at the quietest window between min_s and max_s into each chunk."""
    import numpy as np

    win = int(CHUNK["window_s"] * SR)
    bounds, start, n = [], 0, len(audio)
    while n - start > CHUNK["max_s"] * SR:
        lo, hi = start + int(CHUNK["min_s"] * SR), start + int(CHUNK["max_s"] * SR)
        seg = audio[lo:hi]
        frames = len(seg) // win
        energy = (seg[: frames * win].reshape(frames, win) ** 2).mean(axis=1)
        cut = lo + int(np.argmin(energy)) * win + win // 2
        bounds.append((start, cut))
        start = cut
    bounds.append((start, n))
    return bounds


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _join(parts: list[str]) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip())


def run_whisper(model: str, audio) -> tuple[str, dict]:
    import torch
    from transformers import pipeline

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        dtype=torch.float16 if _device() == "cuda" else torch.float32,
        device=_device(),
    )
    res = asr(
        {"raw": audio, "sampling_rate": SR},
        return_timestamps=True,
        generate_kwargs={"language": "fi", "task": "transcribe"},
    )
    chunks = [
        {"text": c["text"].strip(), "start": c["timestamp"][0], "end": c["timestamp"][1]} for c in res.get("chunks", [])
    ]
    return res["text"].strip(), {"chunks": chunks}


def run_whisper_chunks(model: str, audio) -> tuple[str, dict]:
    import torch
    from transformers import pipeline

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        dtype=torch.float16 if _device() == "cuda" else torch.float32,
        device=_device(),
    )
    segs = []
    for s, e in chunk_bounds(audio):
        res = asr({"raw": audio[s:e], "sampling_rate": SR}, generate_kwargs={"language": "fi", "task": "transcribe"})
        segs.append({"start": round(s / SR, 2), "end": round(e / SR, 2), "text": res["text"].strip()})
    return _join([x["text"] for x in segs]), {"segments": segs}


def run_chunked_hf(model: str, audio) -> tuple[str, dict]:
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=model, device=_device())
    segs = []
    for s, e in chunk_bounds(audio):
        text = asr({"raw": audio[s:e], "sampling_rate": SR})["text"]
        segs.append({"start": round(s / SR, 2), "end": round(e / SR, 2), "text": text.strip()})
    return _join([x["text"] for x in segs]), {"segments": segs}


def run_qwen(model: str, audio) -> tuple[str, dict]:
    import torch
    from qwen_asr import Qwen3ASRModel

    m = Qwen3ASRModel.from_pretrained(
        model,
        dtype=torch.bfloat16,
        device_map="cuda:0" if _device() == "cuda" else "cpu",
        max_inference_batch_size=8,
        max_new_tokens=RECIPES["qwen"]["max_new_tokens"],
    )
    res = m.transcribe(audio=(audio, SR), language="Finnish")
    return res[0].text.strip(), {"detected_language": getattr(res[0], "language", None)}


def _hf_file(spec: str) -> str:
    from huggingface_hub import hf_hub_download

    repo, filename = spec.split(":", 1)
    return hf_hub_download(repo, filename)


def run_canary(model: str, audio, kenlm: bool) -> tuple[str, dict]:
    import soundfile as sf
    from nemo.collections.asr.models import ASRModel, EncDecMultiTaskModel

    if ":" in model:
        asr = EncDecMultiTaskModel.restore_from(_hf_file(model))
    else:
        asr = ASRModel.from_pretrained(model)
    asr = asr.eval().to(_device())
    if kenlm:
        from omegaconf import OmegaConf

        dec = dict(RECIPES["canary_kenlm"]["decoding"])
        lm = _hf_file(dec.pop("ngram_lm_model"))
        asr.change_decoding_strategy(
            decoding_cfg=OmegaConf.create(
                {
                    "strategy": "beam",
                    "beam": {
                        "beam_size": dec["beam_size"],
                        "ngram_lm_model": lm,
                        "ngram_lm_alpha": dec["ngram_lm_alpha"],
                    },
                    "batch_size": 1,
                }
            )
        )
    segs = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (s, e) in enumerate(chunk_bounds(audio)):
            path = os.path.join(tmp, f"chunk_{i}.wav")
            sf.write(path, audio[s:e], SR)
            out = asr.transcribe(audio=[path], taskname="asr", source_lang="fi", target_lang="fi", pnc="yes")
            text = out[0].text if hasattr(out[0], "text") else str(out[0])
            segs.append({"start": round(s / SR, 2), "end": round(e / SR, 2), "text": text.strip()})
    return _join([x["text"] for x in segs]), {"segments": segs}


def run_arm(arm_id: str, audio) -> tuple[str, dict]:
    _, model, recipe, _ = ARMS[arm_id]
    if recipe == "whisper":
        return run_whisper(model, audio)
    if recipe == "whisper_chunks":
        return run_whisper_chunks(model, audio)
    if recipe in {"ctc", "tdt"}:
        return run_chunked_hf(model, audio)
    if recipe == "qwen":
        return run_qwen(model, audio)
    return run_canary(model, audio, kenlm=recipe == "canary_kenlm")


def download(arm_ids: list[str]) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    repos = sorted({ARMS[a][1].split(":", 1)[0] for a in arm_ids if not a.startswith("ft_")})
    for repo in repos:
        print(f"downloading {repo} ...", flush=True)
        if repo == RASMUS:  # allow_patterns did not match its models/ files (0 fetched), so fetch them by name
            for name in DOWNLOADS[repo]:
                hf_hub_download(repo, name)
        else:
            snapshot_download(repo, allow_patterns=DOWNLOADS[repo])
    print("done")


def lib_versions() -> dict:
    out = {"python": platform.python_version()}
    for name in ("torch", "transformers", "qwen_asr", "nemo", "pyctcdecode"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception:  # noqa: BLE001 - absent in this env
            pass
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=["hf", "qwen", "nemo"], help="Only the arms of this environment.")
    parser.add_argument("--arm", action="append", help="Only these arm ids (repeatable).")
    parser.add_argument("--force", action="store_true", help="Re-run arms that already have a result file.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--download", action="store_true", help="Fetch model weights into $HF_HOME and exit.")
    args = parser.parse_args()

    selected = [a for a in ARMS if (not args.arm or a in args.arm) and (not args.env or ARMS[a][0] == args.env)]
    if not args.list:  # a fine-tuned arm exists only once its model directory does
        selected = [a for a in selected if not a.startswith("ft_") or Path(ARMS[a][1]).exists()]
    if args.list:
        for a in selected:
            env, model, recipe, _ = ARMS[a]
            print(f"{a:36s} {env:5s} {model:58s} {recipe}")
        return 0
    if args.download:
        download(selected)
        return 0

    import torch

    audio = load_audio()
    versions = lib_versions()
    print(
        f"device={_device()} {versions} "
        f"chunks={[(round(s / SR, 1), round(e / SR, 1)) for s, e in chunk_bounds(audio)]}",
        flush=True,
    )
    failures = 0
    for arm_id in selected:
        env, model, recipe, label = ARMS[arm_id]
        if (ASR_DIR / f"{arm_id}.json").exists() and not args.force:
            print(f"[skip] {arm_id}: result exists (use --force)")
            continue
        print(f"[run ] {arm_id} ...", flush=True)
        t0 = time.perf_counter()
        try:
            text, raw = run_arm(arm_id, audio)
        except Exception as exc:  # noqa: BLE001 - one model failing must not stop the others
            failures += 1
            print(f"[FAIL] {arm_id}: {type(exc).__name__}: {str(exc)[:500]}", flush=True)
            continue
        res = ArmResult(
            arm_id=arm_id,
            stage="asr",
            provider="huggingface-gpu",
            model=Path(model).name if arm_id.startswith("ft_") else model,
            text=text.strip(),
            params={"recipe": recipe, **RECIPES[recipe], "env": env, "device": _device(), "versions": versions},
            input_ref=rel(AUDIO_PATH),
            source="run",
            label=label,
            latency_s=round(time.perf_counter() - t0, 2),
            raw=raw or None,
            notes="latency_s includes model loading.",
        )
        save_arm(res)
        print(f"[ ok ] {arm_id}: {len(res.text.split())} words, {res.latency_s}s", flush=True)

    write_run_metadata(
        RESULTS_DIR / "gpu_run",
        build_run_metadata(
            app_under_test=APP_UNDER_TEST,
            dataset_path=AUDIO_16K,
            runner_path=Path(__file__),
            model_or_deployment=[Path(ARMS[a][1]).name if a.startswith("ft_") else ARMS[a][1] for a in selected],
            extra={
                "env": args.env,
                "arms": selected,
                "host": platform.node(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "versions": versions,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
        ),
        name=f"_run_metadata_{args.env or 'all'}.json",
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
