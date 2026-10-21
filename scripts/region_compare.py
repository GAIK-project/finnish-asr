"""Compare arms of one benchmark segment by segment, overall and per dialect region, with a paired bootstrap.

For the region-prompt experiment (results/finetune/region-plan.md): which of two arms has the lower WER
on the same segments, and is the gap larger than resampling the segments would make it? WER is
corpus-level with score_benchmark.py's normalisation. The bootstrap draws the segments with replacement
(2 000 draws, seed 0) and recomputes both arms' corpus WER on each draw; the interval is the 2.5th to
97.5th percentile of the difference. Only segments both arms have are compared.

    uv run python scripts/region_compare.py --bench skn_fi --arm A --arm B [--arm C ...]

Pairs are every later arm against the first. Writes results/finetune/region/<bench>.<first arm>.json
(numbers only, so also safe for LA-murre).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jiwer  # noqa: E402

from dialect_regions import lamurre_regions, region_of  # noqa: E402
from fi_common import RESULTS_DIR  # noqa: E402
from score_benchmark import transform  # noqa: E402

DRAWS, SEED = 2000, 0


def load(bench: str, arm: str) -> dict[str, dict]:
    path = RESULTS_DIR / "benchmarks" / bench / f"{arm}.jsonl"
    return {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}


def counts(ref: str, hyp: str) -> tuple[int, int]:
    """(errors, reference words) of one segment."""
    hyp = hyp if transform(False)(hyp)[0] else "-"  # jiwer rejects an empty hypothesis
    w = jiwer.process_words(ref, hyp, transform(False), transform(False))
    return w.substitutions + w.deletions + w.insertions, w.hits + w.substitutions + w.deletions


def wer(seg: list[tuple[int, int]]) -> float:
    return 100 * sum(e for e, _ in seg) / max(1, sum(n for _, n in seg))


def compare(bench: str, arms: list[str], lamurre: dict[str, str]) -> dict:
    runs = [load(bench, a) for a in arms]
    ids = sorted(set.intersection(*(set(r) for r in runs)))
    per = {a: [counts(run[i]["ref"], run[i]["hyp"]) for i in ids] for a, run in zip(arms, runs, strict=True)}
    regions = [region_of(i, lamurre) or "-" for i in ids]
    out = {"bench": bench, "segments": len(ids), "arms": {}, "pairs": {}}
    for a in arms:
        by_region = {}
        for reg in sorted(set(regions)):
            seg = [c for c, r in zip(per[a], regions, strict=True) if r == reg]
            by_region[reg] = {"segments": len(seg), "WER": round(wer(seg), 2)}
        out["arms"][a] = {"WER": round(wer(per[a]), 2), "by_region": by_region}
    rng = random.Random(SEED)
    draws = [[rng.randrange(len(ids)) for _ in ids] for _ in range(DRAWS)]
    first = arms[0]
    for b in arms[1:]:
        diffs = sorted(wer([per[b][k] for k in d]) - wer([per[first][k] for k in d]) for d in draws)
        out["pairs"][f"{b} - {first}"] = {
            "WER_difference": round(wer(per[b]) - wer(per[first]), 2),
            "interval_95": [round(diffs[int(0.025 * DRAWS)], 2), round(diffs[int(0.975 * DRAWS) - 1], 2)],
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--arm", action="append", required=True)
    args = parser.parse_args()
    if len(args.arm) < 2:
        parser.error("two arms or more")
    res = compare(args.bench, args.arm, lamurre_regions())
    out = RESULTS_DIR / "finetune" / "region" / f"{args.bench}.{args.arm[0]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v["WER"] for k, v in res["arms"].items()}, indent=1))
    print(json.dumps(res["pairs"], indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
