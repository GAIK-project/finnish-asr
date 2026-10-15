"""The blind meaning judges of judge_asr.py on the ten dental-webinar clips (dental10_fi).

Word error rate on these clips mostly measures fillers: HY's references leave out "niinku", "tota",
"öö" and the like, so a model that writes what is said (ElevenLabs, Gemini, this project's dialect
model) collects 10-14 points of insertions. The judges of judge_asr.py ignore fillers, spoken forms,
compounds and punctuation and count only what changes the meaning. Same system prompt, answer schema
and judges (Claude Opus 5.5, Gemini 3.1 Pro); the clips have no hand-made name list, so the entity
part is left empty.

Score per arm: meaning-error points per 100 reference words over all ten clips (critical 10,
major 5, minor 1), and the per-clip points (clips numbered, not named). The judge records quote HY's references, which come
from a private repository, so they stay in results/judgements_dental/<judge>/ (gitignored); only
summary.json is committed.

Usage (from the repo root; the transcripts from run_open_models.py --bench dental10_fi and
run_api_benchmark.py --bench dental10_fi, keys as for judge_asr.py):
    uv run --extra judges python scripts/judge_dental.py [--judge claude-opus-5-5] [--arm ID] [--workers 4]
    uv run python scripts/judge_dental.py --report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import judge_asr as ja  # noqa: E402
from fi_common import RESULTS_DIR, now_utc  # noqa: E402
from provenance import build_run_metadata, write_run_metadata  # noqa: E402

BENCH = RESULTS_DIR / "benchmarks" / "dental10_fi"
OUT = RESULTS_DIR / "judgements_dental"
ARMS = [
    "hf_whisper-large-v3",
    "ft_whisper-large-v3-fi-skn-punct-best-chunks",
    "ft_whisper-large-v3-fi-skn-all-chunks",
    "azure_gpt-4o-transcribe",
    "vertex_gemini-3.5-flash",
    "elevenlabs_scribe-v2",
    "google_chirp-3",
]
POINTS = {"critical": 10, "major": 5, "minor": 1}
NO_ENTITIES = "(none for this recording: return an empty entities list)"
PROMPT_SHA = hashlib.sha256((ja.SYSTEM_PROMPT + ja.USER_TEMPLATE + NO_ENTITIES).encode()).hexdigest()


def items(arms: list[str]) -> list[tuple[str, dict]]:
    out = []
    for arm in arms:
        path = BENCH / f"{arm}.jsonl"
        if not path.exists():
            print(f"[skip] {arm}: no {path.name}")
            continue
        out += [(arm, json.loads(x)) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return out


def cache_path(judge: str, arm: str, clip: str) -> Path:
    return OUT / judge / arm / (clip.rsplit(".", 1)[0] + ".json")


def judge_one(judge: str, arm: str, row: dict) -> dict:
    j = ja.JUDGES[judge]
    prompt = ja.USER_TEMPLATE.format(entities=NO_ENTITIES, reference=row["ref"], hypothesis=row["hyp"])
    call = ja.call_claude if judge.startswith("claude") else ja.call_gemini
    for attempt in range(ja.RETRIES + 1):
        try:
            t0 = time.perf_counter()
            parsed, usage, model = call(j, prompt)
            ja.AsrJudgement.model_validate(parsed)
            break
        except Exception:  # noqa: BLE001  (a malformed answer or a transient error: ask again)
            if attempt == ja.RETRIES:
                raise
            time.sleep(5 * (attempt + 1))
    ref_words = len(row["ref"].split())
    pts = sum(POINTS[e["severity"]] for e in parsed["errors"])
    rec = {
        "arm_id": arm,
        "clip": row["id"],
        "judge": judge,
        "judge_model": model,
        "prompt_sha256": PROMPT_SHA,
        "hyp_sha256": hashlib.sha256(row["hyp"].encode()).hexdigest(),
        "ref_words": ref_words,
        "error_points": pts,
        "counts": {s: sum(1 for e in parsed["errors"] if e["severity"] == s) for s in POINTS},
        "usage": usage,
        "latency_s": round(time.perf_counter() - t0, 1),
        "judgement": parsed,
    }
    p = cache_path(judge, arm, row["id"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return rec


def summarise() -> dict:
    out: dict = {"generated_utc": now_utc(), "prompt_sha256": PROMPT_SHA, "judges": {}}
    for jdir in sorted(p for p in OUT.iterdir() if p.is_dir()) if OUT.exists() else []:
        arms = {}
        for adir in sorted(p for p in jdir.iterdir() if p.is_dir()):
            recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(adir.glob("*.json"))]
            recs = [r for r in recs if r["prompt_sha256"] == PROMPT_SHA]
            if not recs:
                continue
            words = sum(r["ref_words"] for r in recs)
            arms[adir.name] = {
                "clips": len(recs),
                "points_per_100w": round(sum(r["error_points"] for r in recs) / words * 100, 2),
                # clips by number in file-name order: the file names carry the speakers' names
                "per_clip": {f"clip{i:02d}": round(r["error_points"] / r["ref_words"] * 100, 2)
                             for i, r in enumerate(sorted(recs, key=lambda r: r["clip"]), 1)},  # fmt: skip
                **{s: sum(r["counts"][s] for r in recs) for s in POINTS},
                "usage": {k: sum(r["usage"].get(k, 0) for r in recs) for k in ("input_tokens", "output_tokens")},
            }
        out["judges"][jdir.name] = arms
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge", choices=list(ja.JUDGES), action="append")
    parser.add_argument("--arm", action="append")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    judges = args.judge or list(ja.JUDGES)
    todo = []
    if not args.report:
        for judge in judges:
            missing = [e for e in ja.JUDGES[judge]["env"] if not os.environ.get(e)]
            if missing:
                print(f"[skip] {judge}: missing {', '.join(missing)}")
                continue
            for arm, row in items(args.arm or ARMS):
                p = cache_path(judge, arm, row["id"])
                if p.exists():
                    old = json.loads(p.read_text(encoding="utf-8"))
                    if old["prompt_sha256"] == PROMPT_SHA and old["hyp_sha256"] == hashlib.sha256(row["hyp"].encode()).hexdigest():
                        continue
                todo.append((judge, arm, row))
    print(f"{len(todo)} judgements to make", flush=True)
    failed = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(judge_one, *t): t for t in todo}
        for fut in as_completed(futs):
            judge, arm, row = futs[fut]
            try:
                rec = fut.result()
                print(f"[ ok ] {judge} {arm} {row['id'][:30]}: {rec['error_points']} pts", flush=True)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"[FAIL] {judge} {arm} {row['id'][:30]}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
    summary = summarise()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    if todo:
        write_run_metadata(
            OUT,
            build_run_metadata(
                app_under_test="Finnish ASR model comparison (ten dental-webinar clips, blind meaning judges)",
                dataset_path=None,
                runner_path=Path(__file__),
                model_or_deployment=[ja.JUDGES[j]["model"] for j in judges],
                env_names=sorted({e for j in judges for e in ja.JUDGES[j]["env"]}),
                extra={"prompt_sha256": PROMPT_SHA, "arms": args.arm or ARMS, "failed": failed},
            ),
        )
    for judge, arms in summary["judges"].items():
        for arm, s in sorted(arms.items(), key=lambda kv: kv[1]["points_per_100w"]):
            print(f"{judge:16s} {arm:46s} {s['points_per_100w']:6.2f} pts/100w  crit {s['critical']:3d} "
                  f"major {s['major']:3d} minor {s['minor']:3d}  ({s['clips']} clips)")  # fmt: skip
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
