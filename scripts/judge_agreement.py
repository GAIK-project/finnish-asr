r"""Do the LA-murre meaning judges agree? Claude Opus 5.5 against the open models run on CSC Aitta.

    uv run python scripts/judge_agreement.py [--round r2]

Reads every judge's cache of one round in results/judgements_dialect/lamurre_fi/
(<judge>-<round>.jsonl, written by judge_dialect.py; all judges saw the same segments, labels and
order) and writes agreement-<round>.json there, numbers only. No keys, no API calls.

The caches quote LA-murre's texts, which its licence (CC BY-ND 4.0) does not let this repository
share, so they are not committed: to rerun this, rebuild the LA-murre test from Kielipankki
(prepare_lamurre.py), transcribe it (run_benchmark.py --bench lamurre_fi) and judge it
(judge_dialect.py). The committed agreement-r2.json is the result of that run. For each judge:

- meaning-error points per 100 reference words per arm (critical 10, major 5, minor 1), the order
  of the arms, and the sign test of the published model against base Whisper;
- against Claude, on the segments both judged: how often the two judges prefer the same arm in
  each pair of arms where both see a difference (pairwise agreement), and the rank correlation of
  their per-segment points for each arm (Spearman).

Judges run on a sample (judge_dialect.py --per-locality N, see SAMPLED) count only the segments of
that seeded draw, so their numbers do not depend on segments judged before a run was narrowed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fi_common import RESULTS_DIR, now_utc, rel  # noqa: E402
from provenance import build_run_metadata, write_run_metadata  # noqa: E402

OUT_DIR = RESULTS_DIR / "judgements_dialect" / "lamurre_fi"
POINTS = {"critical": 10, "major": 5, "minor": 1}
REFERENCE = "claude-opus-5-5"
BASE, PUBLISHED = "hf_whisper-large-v3", "ft_whisper-large-v3-fi-skn-punct-best"
SAMPLED = {"aitta-gemma-4-31b-it": 20, "aitta-qwen3-6-27b": 20}  # judge -> its --per-locality


def seeded_sample(ids: set[str], per_locality: int) -> set[str]:
    """The segments judge_dialect.py --per-locality draws (seed 0), rebuilt from the reference judge's ids."""
    by_loc: dict[str, list[str]] = {}
    for seg in sorted(ids):
        by_loc.setdefault(seg.rsplit("_", 1)[0], []).append(seg)
    rng = random.Random(0)
    return {seg for loc in sorted(by_loc) for seg in rng.sample(by_loc[loc], min(per_locality, len(by_loc[loc])))}


def load(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows[row["id"]] = row
    return rows


def points(row: dict, arm: str) -> int:
    return sum(row["arms"][arm][k] * p for k, p in POINTS.items())


def sign_test(better: int, worse: int) -> float:
    n, k = better + worse, min(better, worse)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(a: list[float], b: list[float]) -> float | None:
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    va, vb = sum((x - ma) ** 2 for x in ra), sum((y - mb) ** 2 for y in rb)
    return round(cov / math.sqrt(va * vb), 3) if va and vb else None


def judge_summary(rows: dict[str, dict], arms: list[str]) -> dict:
    words = sum(r["ref_words"] for r in rows.values())
    per_arm = {a: round(sum(points(r, a) for r in rows.values()) / words * 100, 2) for a in arms}
    better = sum(1 for r in rows.values() if points(r, PUBLISHED) < points(r, BASE))
    worse = sum(1 for r in rows.values() if points(r, PUBLISHED) > points(r, BASE))
    return {
        "segments": len(rows),
        "points_per_100w": per_arm,
        "order_best_first": sorted(arms, key=per_arm.get),
        "published_vs_base": {"fewer_points": better, "more_points": worse, "sign_test_p": sign_test(better, worse)},
    }


def agreement(ref: dict[str, dict], other: dict[str, dict], arms: list[str]) -> dict:
    common = sorted(set(ref) & set(other))
    same = differ = 0
    for seg in common:
        for a, b in itertools.combinations(arms, 2):
            d1 = points(ref[seg], a) - points(ref[seg], b)
            d2 = points(other[seg], a) - points(other[seg], b)
            if d1 and d2:
                same += (d1 > 0) == (d2 > 0)
                differ += (d1 > 0) != (d2 > 0)
    return {
        "segments_both_judged": len(common),
        "pairwise_same_preference": round(same / (same + differ), 3) if same + differ else None,
        "pairs_compared": same + differ,
        "spearman_per_arm": {
            a: spearman([points(ref[s], a) for s in common], [points(other[s], a) for s in common]) for a in arms
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round", default="r2")
    args = parser.parse_args()
    caches = sorted(OUT_DIR.glob(f"*-{args.round}.jsonl"))
    judges = {p.name.removesuffix(f"-{args.round}.jsonl"): load(p) for p in caches}
    if REFERENCE not in judges:
        print(f"no {REFERENCE}-{args.round}.jsonl in {rel(OUT_DIR)} (the caches stay local)")
        return 1
    for name, per_locality in SAMPLED.items():
        if name in judges:
            keep = seeded_sample(set(judges[REFERENCE]), per_locality)
            judges[name] = {seg: row for seg, row in judges[name].items() if seg in keep}
    arms = list(next(iter(judges[REFERENCE].values()))["arms"])
    result = {"generated_utc": now_utc(), "round": args.round, "reference_judge": REFERENCE, "judges": {}}
    for name, rows in judges.items():
        entry = judge_summary(rows, arms)
        if name != REFERENCE:
            entry["against_reference"] = agreement(judges[REFERENCE], rows, arms)
        result["judges"][name] = entry
    (OUT_DIR / f"agreement-{args.round}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    write_run_metadata(
        OUT_DIR,
        build_run_metadata(
            app_under_test="Finnish ASR model comparison (LA-murre dialect test, judge agreement)",
            dataset_path=OUT_DIR,
            runner_path=Path(__file__),
            model_or_deployment=sorted(judges),
            extra={"round": args.round, "segments": {name: len(rows) for name, rows in judges.items()}},
        ),
        name=f"_run_metadata-agreement-{args.round}.json",
    )
    for name, e in result["judges"].items():
        pts = "  ".join(f"{a.split('_', 1)[1][-18:]} {v:5.1f}" for a, v in e["points_per_100w"].items())
        pv = e["published_vs_base"]
        tail = f"published<base {pv['fewer_points']}/{pv['more_points']} p={pv['sign_test_p']:.1g}"
        print(f"{name:32s} n={e['segments']:3d}  {pts}  {tail}")
        if "against_reference" in e:
            ag = e["against_reference"]
            print(
                f"{'':32s} same preference as Claude {ag['pairwise_same_preference']} of {ag['pairs_compared']} pairs; "
                f"Spearman {ag['spearman_per_arm']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
