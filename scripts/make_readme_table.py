"""Write the clip results table between the readme-table markers in README.md.

Reads results/scores.json (score_asr.py) and results/judgements/ (judge_asr.py); no API keys.
Sorted by the mean of the two judges' error points.

Usage: uv run python scripts/make_readme_table.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fi_common import JUDGE_DIR, PROJECT_ROOT, SCORES_PATH  # noqa: E402

START, END = "<!-- readme-table:start -->", "<!-- readme-table:end -->"
JUDGES = ("claude-opus-5-5", "gemini-3.1-pro")

# arm_id -> (label, where it runs)
ROWS = {
    "vertex_gemini-3.5-flash": ("Gemini 3.5 Flash", "API"),
    "elevenlabs_scribe-v2": ("ElevenLabs Scribe v2", "API"),
    "google_chirp-3": ("Google Chirp 3", "API"),
    "hf_whisper-large-v3": ("Whisper large-v3", "open"),
    "hf_whisper-large-v3-turbo": ("Whisper large-v3 turbo", "open"),
    "ft_whisper-large-v3-fi-pilot": ("This project's pilot: Whisper large-v3 + LoRA on FLEURS and VoxPopuli", "open"),
    "ft_whisper-large-v3-fi-skn": (
        "This project, round 1: Whisper large-v3 + LoRA on SKN dialects, FLEURS and VoxPopuli",
        "open",
    ),
    "hf_whisper-large-v3-chunks": ("Whisper large-v3, chunked decoding", "open"),
    "ft_whisper-large-v3-fi-skn-chunks": ("This project, round 1, chunked decoding", "open"),
    "ft_whisper-large-v3-fi-skn-punct-best-chunks": (
        "This project, round 2 (published), chunked decoding",
        "open",
    ),
    "ft_whisper-large-v3-fi-skn-lamurre-best-chunks": ("This project, round 2 internal (+ LA-murre), chunked", "open"),
    "openai_whisper-1": ("OpenAI whisper-1", "API"),
    "openai_gpt-4o-mini-transcribe": ("OpenAI gpt-4o-mini-transcribe", "API"),
    "openai_gpt-transcribe": ("OpenAI gpt-transcribe", "API"),
    "hh_whisperx_fi": ("WhisperX + Finnish-NLP whisper-large-finnish-v3 (Haaga-Helia WhisperX server)", "open"),
    "hf_whisper-large-finnish-v3": ("Finnish-NLP whisper-large-finnish-v3 (plain transformers)", "open"),
    "openai_gpt-4o-transcribe": ("OpenAI gpt-4o-transcribe", "API"),
    "hf_orukeet": ("Orukeet (Parakeet v3 fine-tune)", "open"),
    "nemo_rasmus-canary-fi-v2": ("RASMUS Finnish-ASR-Canary-v2, greedy", "open"),
    "nemo_rasmus-canary-fi-v2-kenlm": ("RASMUS Finnish-ASR-Canary-v2 + KenLM", "open"),
    "vertex_gemini-3.1-pro": ("Gemini 3.1 Pro", "API"),
    "nemo_canary-1b-v2": ("NVIDIA Canary 1B v2", "open"),
    "hf_parakeet-tdt-0.6b-v3": ("NVIDIA Parakeet TDT 0.6B v3", "open"),
    "hf_wav2vec2-xlsr-1b-finnish-lm-v2": ("Finnish-NLP wav2vec2 XLS-R 1B (no LM)", "open"),
    "qwen_qwen3-asr-1.7b": ("Qwen3-ASR 1.7B", "open"),
    "qwen_qwen3-asr-0.6b": ("Qwen3-ASR 0.6B", "open"),
}


def judge(judge_id: str, arm_id: str) -> dict:
    return json.loads((JUDGE_DIR / judge_id / f"{arm_id}.json").read_text(encoding="utf-8"))["scores"]


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    scores = {a["arm_id"]: a for a in json.loads(SCORES_PATH.read_text(encoding="utf-8"))["arms"]}
    rows = []
    for arm_id, (label, kind) in ROWS.items():
        s = scores[arm_id]
        j = [judge(jid, arm_id) for jid in JUDGES]
        rows.append((sum(x["error_points_per_100w"] for x in j) / 2, label, kind, s, j))
    rows.sort(key=lambda r: r[0])
    lines = [
        "| Model | Type | WER % | CER % | Claude pts | Gemini pts | Names (Claude / Gemini) |",
        "|---|---|--:|--:|--:|--:|--:|",
    ]
    for _, label, kind, s, j in rows:
        m = s["metrics"]
        rng = (
            f" ({s['WER_range'][0]:.1f}–{s['WER_range'][1]:.1f})"
            if s["WER_range"] and s["WER_range"][0] != s["WER_range"][1]
            else ""
        )
        names = " / ".join(f"{x['entities_ok']}" for x in j)
        lines.append(
            f"| {label} | {kind} | {m['WER']:.1f}{rng} | {m['CER']:.1f} | "
            f"{j[0]['error_points_per_100w']:.1f} | {j[1]['error_points_per_100w']:.1f} | {names} of 14 |"
        )
    block = "\n".join(lines)
    readme = PROJECT_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    a, b = text.index(START) + len(START), text.index(END)
    readme.write_text(text[:a] + "\n" + block + "\n" + text[b:], encoding="utf-8", newline="\n")
    print(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
