"""Score every clip transcript against HY's Finnish reference: results/scores.json.

WER/CER use the vendored QAD-1 scorer (asr_scoring.py: HY's normalisation, lower-case, all Unicode
punctuation removed, split on spaces), so the numbers line up with the published QAD-1 and HY ones.
Arms with repeat runs (<arm>__r<n>) also get the min-max WER over their runs.
WER_nl reads line breaks as spaces (whitespace_merge), which matters for the few models that put
each segment on its own line; the main WER keeps HY's behaviour.

--reviewed scores against reference.reviewed.txt instead: HY's text with the 14 changes a listener
made at the 42 disputed spans, written to results/scores.reviewed.json with each arm's WER against
HY's text beside it.

--decoding scores results/clip/decoding_check.json, the published model downloaded back from the Hub
and decoded six ways (quiet-point pieces, the pipeline's chunk_length_s 20 and 30, each with and
without no_repeat_ngram_size=6), against both references: results/clip/decoding_scores.json, the
source of the model card's "Settings to avoid" table.

Each run also writes a secret-free _score*_run_metadata.json, built before anything is written, so
its repo_dirty says whether the inputs were committed.

Usage: uv run python scripts/score_asr.py [--reviewed | --decoding]
Needs the clip files (uv run python scripts/get_clip.py). No API keys.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asr_scoring as qs  # noqa: E402
import fi_common as FI  # noqa: E402
from provenance import build_run_metadata, write_run_metadata  # noqa: E402

DECODING_PATH = FI.RESULTS_DIR / "clip" / "decoding_check.json"
DECODING_SCORES_PATH = FI.RESULTS_DIR / "clip" / "decoding_scores.json"


def repeat_texts() -> dict[str, list[str]]:
    """base arm id -> texts of every run (primary first), from this repository's and QAD-1's arms."""
    out: dict[str, list[str]] = {}
    for d in (FI.ASR_DIR, FI.QAD1_ASR_DIR):
        for p in sorted(d.glob("*.json")):
            if p.name.startswith("_"):
                continue
            a = FI.load_arm(p)
            out.setdefault(a.base_arm_id, []).append(a.text)
    return out


def score(reviewed: bool = False) -> dict:
    """The scores.json document (without writing it)."""
    ref_path = FI.GT_REVIEWED_PATH if reviewed else FI.GT_FI_PATH
    original = {}
    if reviewed and FI.SCORES_PATH.exists():
        original = {
            r["arm_id"]: r["metrics"]["WER"] for r in json.loads(FI.SCORES_PATH.read_text(encoding="utf-8"))["arms"]
        }
    gt = FI.read_text(ref_path)
    runs = repeat_texts()
    rows = []
    for arm in FI.load_arms():
        m, _ = qs.asr_metrics(gt, arm.text)
        ws = qs.whitespace_merge(gt, arm.text)
        wers = [qs.asr_metrics(gt, t)[0]["WER"] for t in runs.get(arm.arm_id, [])]
        rows.append(
            {
                "arm_id": arm.arm_id,
                "label": arm.label or arm.arm_id,
                "model": arm.model,
                "provider": arm.provider,
                "origin": "qad1" if FI.QAD1_TAG in arm.notes else "this project",
                "metrics": m,
                "WER_nl": ws["WER"] if ws else m["WER"],
                "runs": len(wers),
                "WER_range": [min(wers), max(wers)] if len(wers) > 1 else None,
                "latency_s": arm.latency_s,
                **({"WER_against_hy": original.get(arm.arm_id)} if reviewed else {}),
            }
        )
    rows.sort(key=lambda r: r["metrics"]["WER"])
    return {
        "generated_utc": FI.now_utc(),
        "reference": FI.rel(ref_path),
        "normalization": qs.NORMALIZATION,
        "arms": rows,
    }


def score_decoding() -> dict:
    """WER and CER of every decoding in decoding_check.json against HY's and the reviewed reference."""
    doc = json.loads(DECODING_PATH.read_text(encoding="utf-8"))
    refs = {"hy": FI.read_text(FI.GT_FI_PATH), "reviewed": FI.read_text(FI.GT_REVIEWED_PATH)}
    rows = {}
    for name, d in doc["decodings"].items():
        metrics = {ref: qs.asr_metrics(text, d["text"])[0] for ref, text in refs.items()}
        rows[name] = {
            "settings": d["settings"],
            "WER": {ref: m["WER"] for ref, m in metrics.items()},
            "CER": {ref: m["CER"] for ref, m in metrics.items()},
            "hyp_words": len(d["text"].split()),
        }
    return {
        "generated_utc": FI.now_utc(),
        "model": doc["model"],
        "model_revision": doc["model_revision"],
        "references": {"hy": FI.rel(FI.GT_FI_PATH), "reviewed": FI.rel(FI.GT_REVIEWED_PATH)},
        "normalization": qs.NORMALIZATION,
        "decodings": rows,
    }


def write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def main_decoding() -> int:
    FI.require_clip("reference.txt", "reference.reviewed.txt")
    meta = build_run_metadata(
        app_under_test=FI.APP_UNDER_TEST + ": decoding settings of the published model",
        dataset_path=DECODING_PATH,
        runner_path=Path(__file__),
        model_or_deployment=json.loads(DECODING_PATH.read_text(encoding="utf-8"))["model"],
        extra={
            "output": FI.rel(DECODING_SCORES_PATH),
            "reference_sha256": {n: FI.CLIP_FILES[n] for n in ("reference.txt", "reference.reviewed.txt")},
        },
    )
    doc = score_decoding()
    write_json(DECODING_SCORES_PATH, doc)
    write_run_metadata(DECODING_SCORES_PATH.parent, meta, name="_score_decoding_run_metadata.json")
    print(f"{'decoding':24s} {'WER HY':>7s} {'WER rev':>7s}  settings")
    for name, r in doc["decodings"].items():
        print(f"{name:24s} {r['WER']['hy']:7.2f} {r['WER']['reviewed']:7.2f}  {r['settings']}")
    print(f"\nwrote {FI.rel(DECODING_SCORES_PATH)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--reviewed", action="store_true", help="Score against the listener-reviewed reference.")
    group.add_argument(
        "--decoding", action="store_true", help="Score results/clip/decoding_check.json against both references."
    )
    args = parser.parse_args()
    if args.decoding:
        return main_decoding()
    FI.require_clip("reference.reviewed.txt" if args.reviewed else "reference.txt")
    out_path = FI.SCORES_PATH.with_name("scores.reviewed.json") if args.reviewed else FI.SCORES_PATH
    ref_path = FI.GT_REVIEWED_PATH if args.reviewed else FI.GT_FI_PATH
    meta = build_run_metadata(
        app_under_test=FI.APP_UNDER_TEST,
        dataset_path=ref_path,
        runner_path=Path(__file__),
        extra={
            "output": FI.rel(out_path),
            "arms": len(FI.load_arms()),
            "reference_sha256": FI.CLIP_FILES[ref_path.name],
        },
    )
    doc = score(args.reviewed)
    rows = doc["arms"]
    write_json(out_path, doc)
    write_run_metadata(
        FI.RESULTS_DIR, meta, name="_score_reviewed_run_metadata.json" if args.reviewed else "_score_run_metadata.json"
    )
    print(f"{'arm':42s} {'WER':>6s} {'CER':>6s} {'WER_nl':>6s} {'range':>13s} {'HY ref':>7s}  origin")
    for r in rows:
        rng = f"{r['WER_range'][0]:.1f}-{r['WER_range'][1]:.1f}" if r["WER_range"] else ""
        hy = f"{r['WER_against_hy']:.1f}" if r.get("WER_against_hy") is not None else ""
        print(
            f"{r['arm_id']:42s} {r['metrics']['WER']:6.1f} {r['metrics']['CER']:6.1f} "
            f"{r['WER_nl']:6.1f} {rng:>13s} {hy:>7s}  {r['origin']}"
        )
    print(f"\nwrote {FI.rel(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
