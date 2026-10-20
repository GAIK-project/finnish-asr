"""Score the public-benchmark runs (run_benchmark.py): results/benchmarks/<bench>/scores.json.

WER and CER are corpus-level (summed errors over summed reference words or characters)
after the usual model-card normalisation: lower-case, punctuation removed, whitespace
collapsed. That is what the RASMUS model card and most Finnish model cards use, so the
numbers are comparable in kind; small differences in normalisation still move them.
Seconds per utterance are GPU wall time in batches of 16, model loading excluded; for the API
arms (run_api_benchmark.py) they are request wall time with 4 parallel workers, retries included.

LA-murre's per-utterance files (lamurre_fi/*.jsonl) hold the corpus's texts, which its licence
(CC BY-ND 4.0) does not let this repository share, so they are not committed: a benchmark without
*.jsonl keeps its committed scores.json as it is. Rebuild the files with prepare_lamurre.py and
run_benchmark.py --bench lamurre_fi to rescore it. The same holds for the ten dental-webinar clips
(dental10_fi/*.jsonl): their Finnish references are the University of Helsinki's, from a private
repository, so only the numbers are committed.

Also writes one table with the public test sets between the benchmark-table markers in README.md, the
dental clips' table (with the blind judges of judge_dental.py) between the dental-table markers, and a
secret-free _score_run_metadata.json beside each rescored scores.json (built before anything is
written, so its repo_dirty says whether the inputs were committed).

Usage: uv run python scripts/score_benchmark.py
No API keys needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jiwer  # noqa: E402

from fi_common import PROJECT_ROOT, RESULTS_DIR, now_utc  # noqa: E402
from provenance import build_run_metadata, write_run_metadata  # noqa: E402

BENCHES = {
    "fleurs_fi": "google/fleurs fi_fi test",
    "voxpopuli_fi": "facebook/voxpopuli fi test",
    "skn_fi": "SKN dialect test, 10 held-out municipalities",
    "dental10_fi": "Ten 180 s dental-webinar clips (HY QAD-1 set A), HY's Finnish references",
    "lamurre_fi": "LA-murre dialect test, 10 localities outside SKN, dialect reference",
}
START, END = "<!-- benchmark-table:start -->", "<!-- benchmark-table:end -->"
DENTAL_START, DENTAL_END = "<!-- dental-table:start -->", "<!-- dental-table:end -->"
# training-run variants kept in scores.json but not in the README table (the README's training section
# compares them): 4-epoch runs, whose last epoch overfit, and the longer run on round 1's data
HIDDEN = {"ft_whisper-large-v3-fi-skn-r64e4", "ft_whisper-large-v3-fi-skn-punct", "ft_whisper-large-v3-fi-skn-lamurre"}

# Bold: this project's models; the README note under the table says what each was trained on
LABELS = {
    "ft_whisper-large-v3-fi-pilot": "**Ours: pilot, FLEURS + VoxPopuli only**",
    "ft_whisper-large-v3-fi-skn": "**Ours: round 1**",
    "ft_whisper-large-v3-fi-skn-punct-best": "**Ours: published model (main, v1.0)**",
    "ft_whisper-large-v3-fi-skn-punct-ts": "**Ours: timestamps version**",
    "ft_whisper-large-v3-fi-skn-lamurre-best": "**Ours: la-murre version**",
    "hf_whisper-large-v3": "Whisper large-v3",
    "hf_whisper-large-v3-turbo": "Whisper large-v3 turbo",
    "hf_whisper-large-finnish-v3": "Finnish-NLP whisper-large-finnish-v3",
    "hf_wav2vec2-xlsr-1b-finnish-lm-v2": "Finnish-NLP wav2vec2 XLS-R 1B (no LM)",
    "hf_parakeet-tdt-0.6b-v3": "NVIDIA Parakeet TDT 0.6B v3",
    "hf_orukeet": "Orukeet (Parakeet v3 fine-tune)",
    "qwen_qwen3-asr-1.7b": "Qwen3-ASR 1.7B",
    "nemo_canary-1b-v2": "NVIDIA Canary 1B v2",
    "nemo_rasmus-canary-fi-v2": "RASMUS Finnish-ASR-Canary-v2, greedy",
    "nemo_rasmus-canary-fi-v2-kenlm": "RASMUS Finnish-ASR-Canary-v2 + KenLM",
    "vertex_gemini-3.5-flash": "Gemini 3.5 Flash (API, asked to write as spoken)",
    "vertex_gemini-3.1-pro": "Gemini 3.1 Pro (API)",
    "azure_gpt-4o-transcribe": "OpenAI gpt-4o-transcribe (API, via Azure)",
    "azure_whisper": "OpenAI Whisper (API, via Azure)",
    "hf_whisper-large-v3-chunks": "Whisper large-v3, in 15–25 s pieces",
    "ft_whisper-large-v3-fi-skn-punct-best-chunks": "**Ours: published model, in 15–25 s pieces**",
    "ft_whisper-large-v3-fi-skn-punct-ts-chunks": "**Ours: timestamps version, in pieces**",
    "ft_whisper-large-v3-fi-skn-punct-ts-fallback": "**Ours: timestamps version, long audio in one go**",
    "ft_whisper-large-v3-fi-skn-lamurre-best-chunks": "**Ours: la-murre version, in pieces**",
    "ft_whisper-large-v3-fi-skn-all": "**Ours: all-skn version**",
    "ft_whisper-large-v3-fi-skn-all-chunks": "**Ours: all-skn version, in pieces**",
    "vertex_gemini-3.5-flash-std": "Gemini 3.5 Flash (API, asked for standard Finnish)",
    "google_chirp-3": "Google Chirp 3 (API)",
    "openai_gpt-transcribe": "OpenAI gpt-transcribe (API)",
    "openai_gpt-4o-mini-transcribe": "OpenAI gpt-4o-mini-transcribe (API)",
    "elevenlabs_scribe-v2": "ElevenLabs Scribe v2 (API)",
}


def transform(chars: bool) -> jiwer.Compose:
    return jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveWhiteSpace(replace_by_space=True),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfChars() if chars else jiwer.ReduceToListOfListOfWords(),
        ]
    )


def score_rows(bench: str) -> list[dict]:
    """One row per arm from the benchmark's *.jsonl files, best WER first."""
    out_dir = RESULTS_DIR / "benchmarks" / bench
    rows = []
    for path in sorted(out_dir.glob("*.jsonl")):
        recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not recs:  # a run that failed before its first item
            continue
        refs = [r["ref"] for r in recs]
        hyps = [r["hyp"] if transform(False)(r["hyp"])[0] else "-" for r in recs]  # jiwer rejects an empty hypothesis
        w = jiwer.process_words(refs, hyps, transform(False), transform(False))
        c = jiwer.process_characters(refs, hyps, transform(True), transform(True))
        ref_words = w.hits + w.substitutions + w.deletions
        ref_chars = c.hits + c.substitutions + c.deletions
        meta_path = out_dir / f"{path.stem}.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        rows.append(
            {
                "arm_id": path.stem,
                "label": LABELS.get(path.stem, path.stem),
                "utterances": len(recs),
                "WER": round((w.substitutions + w.deletions + w.insertions) / ref_words * 100, 4),
                "CER": round((c.substitutions + c.deletions + c.insertions) / ref_chars * 100, 4),
                "SUB": round(w.substitutions / ref_words * 100, 4),
                "DEL": round(w.deletions / ref_words * 100, 4),
                "INS": round(w.insertions / ref_words * 100, 4),
                "ref_words": ref_words,
                "empty_hyps": sum(1 for r in recs if not r["hyp"].strip()),
                "seconds_per_utterance": round(sum(r["seconds"] for r in recs) / len(recs), 3),
                "model": meta.get("model"),
                "gpu": meta.get("gpu"),
            }
        )
    rows.sort(key=lambda r: r["WER"])
    return rows


def score_metadata(bench: str, rows: list[dict]) -> dict:
    out_dir = RESULTS_DIR / "benchmarks" / bench
    return build_run_metadata(
        app_under_test=f"Finnish ASR public benchmarks: {BENCHES[bench]}",
        dataset_path=out_dir,
        runner_path=Path(__file__),
        model_or_deployment=[r["model"] for r in rows if r["model"]],
        extra={"arms": [r["arm_id"] for r in rows], "utterances": sorted({r["utterances"] for r in rows})},
    )


def write_scores(bench: str, rows: list[dict], meta: dict) -> None:
    out_dir = RESULTS_DIR / "benchmarks" / bench
    (out_dir / "scores.json").write_text(
        json.dumps({"generated_utc": now_utc(), "benchmark": BENCHES[bench], "arms": rows}, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_run_metadata(out_dir, meta, name="_score_run_metadata.json")


def score(bench: str, write: bool = True) -> list[dict]:
    """Rescore a benchmark and write its scores.json; without *.jsonl, the committed scores.json rows."""
    out_dir = RESULTS_DIR / "benchmarks" / bench
    rows = score_rows(bench)
    if not rows:
        committed = out_dir / "scores.json"
        return json.loads(committed.read_text(encoding="utf-8"))["arms"] if committed.exists() else []
    if write:
        write_scores(bench, rows, score_metadata(bench, rows))
    return rows


def dental_table(de: dict) -> str:
    """The dental clips: WER, its inserted words, and the two blind judges' meaning-error points (judge_dental.py)."""
    path = RESULTS_DIR / "judgements_dental" / "summary.json"
    judges = json.loads(path.read_text(encoding="utf-8"))["judges"] if path.exists() else {}
    names = {"claude-opus-5-5": "Claude Opus 5.5", "gemini-3.1-pro": "Gemini 3.1 Pro"}
    head = " | ".join(f"Meaning points, {names.get(j, j)}" for j in judges)
    lines = [f"| Model | WER % | Inserted words % | {head} |", "|---|--:|--:|" + "--:|" * len(judges)]
    for a in sorted(set(de) - HIDDEN, key=lambda a: de[a]["WER"]):
        pts = " | ".join(f"{judges[j][a]['points_per_100w']:.1f}" if a in judges[j] else "–" for j in judges)
        label = LABELS.get(a, a)
        if a.startswith("ft_") and not a.endswith(("-chunks", "-fallback")):
            label += ", whole clip in one go"  # these repeat themselves on 180 s audio: the inserted words
        lines.append(f"| {label} | {de[a]['WER']:.1f} | {de[a]['INS']:.1f} | {pts} |")
    return "\n".join(lines)


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    fresh = {b: score_rows(b) for b in BENCHES}
    metas = {b: score_metadata(b, rows) for b, rows in fresh.items() if rows}  # all before the first write
    for b, meta in metas.items():
        write_scores(b, fresh[b], meta)
    scores = {b: {r["arm_id"]: r for r in (fresh[b] or score(b, write=False))} for b in BENCHES}
    fl, vp, sk, la = scores["fleurs_fi"], scores["voxpopuli_fi"], scores["skn_fi"], scores["lamurre_fi"]
    de = scores["dental10_fi"]
    arms = sorted((set(fl) | set(vp) | set(sk)) - HIDDEN, key=lambda a: (sk.get(a, fl.get(a, {})).get("WER", 999), a))

    def cell(table: dict, arm: str, key: str) -> str:
        return f"{table[arm][key]:.1f}" if arm in table else "–"

    lines = [
        "| Model | SKN dialects WER % | LA-murre dialects WER % | FLEURS WER % | VoxPopuli WER % "
        "| FLEURS CER % | s / clip |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    for a in arms:
        speed = f"{fl[a]['seconds_per_utterance']:.2f}" if a in fl else "–"
        lines.append(
            f"| {LABELS.get(a, a)} | {cell(sk, a, 'WER')} | {cell(la, a, 'WER')} | {cell(fl, a, 'WER')} | "
            f"{cell(vp, a, 'WER')} | {cell(fl, a, 'CER')} | {speed} |"
        )
    block = "\n".join(lines)
    dental = dental_table(de)
    readme = PROJECT_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    for start, end, body in ((START, END, block), (DENTAL_START, DENTAL_END, dental)):
        if start in text:
            a, b = text.index(start) + len(start), text.index(end)
            text = text[:a] + "\n" + body + "\n" + text[b:]
    readme.write_text(text, encoding="utf-8", newline="\n")
    print(block + "\n\n" + dental)
    counts = {b: sorted({r["utterances"] for r in t.values()}) for b, t in scores.items()}
    print(f"\nutterances per arm: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
