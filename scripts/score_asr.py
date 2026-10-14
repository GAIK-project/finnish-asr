"""Score every clip transcript against HY's Finnish reference: results/scores.json.

WER/CER use the vendored QAD-1 scorer (asr_scoring.py: HY's normalisation, lower-case, all Unicode
punctuation removed, split on spaces), so the numbers line up with the published QAD-1 and HY ones.
Arms with repeat runs (<arm>__r<n>) also get the min-max WER over their runs.
WER_nl reads line breaks as spaces (whitespace_merge), which matters for the few models that put
each segment on its own line; the main WER keeps HY's behaviour.

--reviewed scores against reference.reviewed.txt instead: HY's text with the 14 changes a listener
made at the 42 disputed spans, written to results/scores.reviewed.json with each arm's WER against
HY's text beside it.

Usage: uv run python scripts/score_asr.py [--reviewed]
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reviewed", action="store_true", help="Score against the listener-reviewed reference.")
    args = parser.parse_args()
    FI.require_clip("reference.reviewed.txt" if args.reviewed else "reference.txt")
    out_path = FI.SCORES_PATH.with_name("scores.reviewed.json") if args.reviewed else FI.SCORES_PATH
    doc = score(args.reviewed)
    rows = doc["arms"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    ref_path = FI.GT_REVIEWED_PATH if args.reviewed else FI.GT_FI_PATH
    write_run_metadata(
        FI.RESULTS_DIR,
        build_run_metadata(
            app_under_test=FI.APP_UNDER_TEST,
            dataset_path=ref_path,
            runner_path=Path(__file__),
            extra={"output": FI.rel(out_path), "arms": len(rows), "reference_sha256": FI.CLIP_FILES[ref_path.name]},
        ),
        name="_score_reviewed_run_metadata.json" if args.reviewed else "_score_run_metadata.json",
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
