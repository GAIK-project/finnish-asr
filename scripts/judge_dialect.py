"""Blind LLM judge for the LA-murre dialect test: which model keeps the meaning?

LA-murre's reference is a transcript in dialect spelling, so WER charges a model for writing
standard Finnish, which is what a user wants. This judge reads the dialect
reference and the transcripts of several models for the same segment, labelled A, B, C in an
order shuffled per segment, and counts only meaning errors: a wrong word, name or number,
missing content, invented content. Standard against dialect forms, digits against number
words, punctuation, case and fillers are explicitly not errors.

Sample: --per-locality N segments per locality (seed 0); the default takes all 583. One judge, Claude Opus 5.5; the
judge is a model and is noisy, so read small gaps as ties. The reference is a human transcript
from the corpus; nothing here changes it.

Score per arm: meaning-error points per 100 reference words (critical 10, major 5, minor 1,
as in judge_asr.py), and how often the arm had the fewest points of the segment.

Rounds: the default round judges base Whisper, round 1 and Finnish-NLP. --round NAME --arms A,B,..
judges another set of up to six arms in one call per segment; each round has its own cache and
summary, so rounds are compared through the arms they share. The committed rounds: the default
(summary.json: Whisper large-v3, round 1, Finnish-NLP), r2 (summary-r2*.json: Whisper large-v3,
round 1, the published model, the internal model; also judged by open models on Aitta) and r3
(summary-r3.json: Whisper large-v3, the published model, the model trained with timestamp tokens).

Cache: results/judgements_dialect/lamurre_fi/<judge>[-<round>].jsonl, one line per segment;
summary[-<round>].json for Claude, summary[-<round>]-<judge>.json for any other judge.

Second judges: --provider azure --judge-model <deployment> runs an OpenAI model deployed in Azure
(round r4: gpt-6.1-sol, through GAIK's Azure). --provider aitta --judge-model <id> runs an open model on CSC Aitta (CSC's
OpenAI-compatible LLM service for Finnish research) with the same prompt, schema, labels and order,
so its verdicts can be compared segment by segment with Claude's (judge_agreement.py). Needs
AITTA_API_TOKEN. Aitta's terms forbid personal or sensitive data; the LA-murre texts are a
published research corpus.

Prompt versions: every LA-murre judgement in results/ was made with dialect-judge-v1. The recorded
prompt_sha256 covers the prompt and the answer schema, and the schema lists the labels, so it
depends on the number of arms: 8489bd83... for the three-arm rounds (the default round and r3) and
10a38347... for the four-arm round r2 (Claude and the Aitta judges alike). v1's three example word
pairs in the first paragraph were taken from the LA-murre corpus, which this repository may not
share (CC BY-ND 4.0), so this public copy, dialect-judge-v2, gives example pairs from SKN instead
(segment SKN45a_Heinola_0097, CC BY 4.0) and is otherwise identical. Its prompt hashes differ, so it
does not reuse v1 caches.

Inputs and outputs stay local. The input is results/benchmarks/lamurre_fi/<arm>.jsonl
(prepare_lamurre.py, then run_benchmark.py --bench lamurre_fi) and every cache line quotes the
reference: both are gitignored. Only the summaries and the run metadata (numbers) are committed.

Usage (from the repo root):
    uv run --extra judges python scripts/judge_dialect.py [--per-locality 60] [--report]
    uv run --extra judges python scripts/judge_dialect.py --round r2 --arms hf_whisper-large-v3,...
    uv run --extra judges python scripts/judge_dialect.py --round r2 --arms ... --provider aitta
        --judge-model openai/gpt-oss-120b
Needs ANTHROPIC_FOUNDRY_RESOURCE and ANTHROPIC_FOUNDRY_API_KEY, or ANTHROPIC_API_KEY (--provider api),
or AITTA_API_TOKEN (--provider aitta), in the repo-root .env.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pydantic import BaseModel, create_model  # noqa: E402

from fi_common import RESULTS_DIR, now_utc  # noqa: E402
from provenance import build_run_metadata, load_repo_env, write_run_metadata  # noqa: E402

load_repo_env(Path(__file__))

BENCH = "lamurre_fi"
ARMS = ["hf_whisper-large-v3", "ft_whisper-large-v3-fi-skn", "hf_whisper-large-finnish-v3"]
JUDGE_ID, JUDGE_MODEL = "claude-opus-5-5", "claude-opus-5-5"
ENV = ("ANTHROPIC_FOUNDRY_RESOURCE", "ANTHROPIC_FOUNDRY_API_KEY")
ENV_API = ("ANTHROPIC_API_KEY",)
ENV_AITTA = ("AITTA_API_TOKEN",)
ENV_AZURE = ("AZURE_ENDPOINT", "AZURE_API_KEY")
AITTA_URL = "https://aitta-api.csc.fi/openai/v1"
PROVIDER = "foundry"
ROUND = ""
LABELS = "ABC"
PROMPT_VERSION = "dialect-judge-v2"  # v1 (every published result) differed only in the example word pairs
POINTS = {"critical": 10, "major": 5, "minor": 1}
OUT_DIR = RESULTS_DIR / "judgements_dialect" / BENCH

SYSTEM_PROMPT = """You compare automatic transcripts of Finnish dialect speech with a human reference.

The reference is a linguist's transcript in DIALECT spelling (for example "miekii" = "minäkin",
"viej juomaakii" = "vien juomaakin", "irvistie" = "irvistää"). The transcripts should
say the same thing; writing it in standard Finnish is correct and wanted.

For each transcript count only MEANING errors against the reference:
- a wrong word, name or number that changes what is said
- content that is missing
- content that was not said
Severity: critical = the segment's main point is lost or reversed; major = a content word,
name or number is wrong or a clause is missing; minor = a small slip a reader can see through.

NOT errors: dialect against standard forms, digits against number words, punctuation, case,
compound spacing, dropped fillers or repetitions ("niin niin", "tuota"), a cut-off word left out.
Judge each transcript on its own; the labels carry no information."""


def judgement_model(labels: str) -> type[BaseModel]:
    """The answer schema; for three arms it is exactly the schema of the first round, so its cache stays valid."""
    verdict = create_model(
        "Verdict",
        label=(Literal[tuple(labels)], ...),
        critical=(int, ...),
        major=(int, ...),
        minor=(int, ...),
        note=(str, ...),
    )
    return create_model("SegmentJudgement", transcripts=(list[verdict], ...))


def _strict(schema: dict) -> dict:
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
        for v in schema.values():
            _strict(v) if isinstance(v, dict) else [_strict(x) for x in v] if isinstance(v, list) else None
    return schema


def configure(arms: list[str], round_name: str, provider: str, judge_model: str = "") -> None:
    global ARMS, LABELS, ROUND, PROVIDER, SegmentJudgement, JSON_SCHEMA, PROMPT_SHA, JUDGE_ID, JUDGE_MODEL
    ARMS, LABELS, ROUND, PROVIDER = arms, "ABCDEF"[: len(arms)], round_name, provider
    if provider in ("aitta", "azure"):
        JUDGE_MODEL = judge_model
        JUDGE_ID = f"{provider}-" + judge_model.split("/")[-1].lower().replace(".", "-")
    SegmentJudgement = judgement_model(LABELS)
    JSON_SCHEMA = _strict(SegmentJudgement.model_json_schema())
    PROMPT_SHA = hashlib.sha256((SYSTEM_PROMPT + json.dumps(JSON_SCHEMA, sort_keys=True)).encode()).hexdigest()


configure(ARMS, ROUND, PROVIDER)


def load_sample(per_locality: int) -> list[dict]:
    d = RESULTS_DIR / "benchmarks" / BENCH
    missing = [arm for arm in ARMS if not (d / f"{arm}.jsonl").exists()]
    if missing:
        raise SystemExit(
            f"no {', '.join(missing)} in results/benchmarks/{BENCH}/: LA-murre's per-segment files are not in "
            "this repository (CC BY-ND); rebuild them with prepare_lamurre.py and run_benchmark.py --bench lamurre_fi"
        )
    hyps = {}
    for arm in ARMS:
        for line in (d / f"{arm}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            hyps.setdefault(row["id"], {"id": row["id"], "ref": row["ref"], "hyp": {}})["hyp"][arm] = row["hyp"]
    by_loc: dict[str, list[dict]] = {}
    for item in hyps.values():
        by_loc.setdefault(item["id"].rsplit("_", 1)[0], []).append(item)
    rng = random.Random(0)
    sample = []
    for loc in sorted(by_loc):
        items = sorted(by_loc[loc], key=lambda i: i["id"])
        sample += sorted(rng.sample(items, min(per_locality, len(items))), key=lambda i: i["id"])
    return sample


def judge(item: dict) -> dict:
    for _ in range(2):  # an incomplete answer (a verdict left out): try again
        try:
            return judge_once(item)
        except (RuntimeError, ValueError):
            pass
    return judge_once(item)


def judge_once(item: dict) -> dict:
    import anthropic

    order = ARMS[:]
    random.Random(item["id"]).shuffle(order)
    labels = dict(zip(LABELS, order, strict=True))
    prompt = f"Reference (dialect transcript):\n{item['ref']}\n\n" + "\n\n".join(
        f"Transcript {label}:\n{item['hyp'][arm] or '(empty)'}" for label, arm in labels.items()
    )
    if len(labels) > 3:
        prompt += f"\n\nReturn exactly {len(labels)} verdicts, one for each of {', '.join(labels)}."
    if PROVIDER in ("aitta", "azure"):
        return judge_aitta(item, labels, prompt)
    if PROVIDER == "api":
        client = anthropic.Anthropic(api_key=os.environ[ENV_API[0]], max_retries=3, timeout=300)
    else:
        client = anthropic.AnthropicFoundry(
            resource=os.environ[ENV[0]], api_key=os.environ[ENV[1]], max_retries=3, timeout=300
        )
    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        thinking={"type": "adaptive"},
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": JSON_SCHEMA}},
    )
    if resp.stop_reason != "end_turn":
        raise RuntimeError(f"stop_reason={resp.stop_reason}")
    out = SegmentJudgement.model_validate_json("".join(b.text for b in resp.content if b.type == "text"))
    verdicts = {v.label: v for v in out.transcripts}
    if set(verdicts) != set(labels):
        raise RuntimeError(f"labels {sorted(verdicts)}")
    return {
        "id": item["id"],
        "ref": item["ref"],
        "ref_words": len(item["ref"].split()),
        "prompt_sha256": PROMPT_SHA,
        "judge_model": resp.model,
        "usage": {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens},
        "arms": {
            arm: {"label": label, "hyp": item["hyp"][arm], **verdicts[label].model_dump(exclude={"label"})}
            for label, arm in labels.items()
        },
    }


def judge_aitta(item: dict, labels: dict[str, str], prompt: str) -> dict:
    """The same judgement on an open model served by CSC Aitta, or an OpenAI model deployed in Azure (its
    v1 endpoint; the deployment name is the model), with the answer schema enforced."""
    from openai import OpenAI

    schema = {"type": "json_schema", "json_schema": {"name": "SegmentJudgement", "schema": JSON_SCHEMA, "strict": True}}
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    if PROVIDER == "azure":  # a reasoning model: no temperature, and its reasoning counts toward the token limit
        base = os.environ[ENV_AZURE[0]].rstrip("/") + "/openai/v1"
        client = OpenAI(base_url=base, api_key=os.environ[ENV_AZURE[1]], timeout=900, max_retries=5)
        resp = client.chat.completions.create(
            model=JUDGE_MODEL, messages=messages, response_format=schema, max_completion_tokens=16000
        )
    else:
        client = OpenAI(base_url=AITTA_URL, api_key=os.environ[ENV_AITTA[0]], timeout=900, max_retries=5)
        resp = client.chat.completions.create(
            model=JUDGE_MODEL, messages=messages, response_format=schema, max_tokens=6000, temperature=0
        )
    choice = resp.choices[0]
    if choice.finish_reason != "stop":
        raise RuntimeError(f"finish_reason={choice.finish_reason}")
    out = SegmentJudgement.model_validate_json(choice.message.content or "")
    verdicts = {v.label: v for v in out.transcripts}
    if set(verdicts) != set(labels):
        raise RuntimeError(f"labels {sorted(verdicts)}")
    usage = resp.usage
    return {
        "id": item["id"],
        "ref": item["ref"],
        "ref_words": len(item["ref"].split()),
        "prompt_sha256": PROMPT_SHA,
        "judge_model": resp.model or JUDGE_MODEL,
        "usage": {
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
        },
        "arms": {
            arm: {"label": label, "hyp": item["hyp"][arm], **verdicts[label].model_dump(exclude={"label"})}
            for label, arm in labels.items()
        },
    }


def summarise(rows: list[dict]) -> dict:
    words = sum(r["ref_words"] for r in rows)
    arms = {}
    for arm in ARMS:
        counts = {s: sum(r["arms"][arm][s] for r in rows) for s in POINTS}
        pts = [sum(r["arms"][arm][s] * p for s, p in POINTS.items()) for r in rows]
        best = sum(
            1
            for r, own in zip(rows, pts, strict=True)
            if own == min(sum(r["arms"][a][s] * p for s, p in POINTS.items()) for a in ARMS)
        )
        arms[arm] = {
            **counts,
            "error_points_per_100w": round(sum(pts) / words * 100, 2),
            "segments_without_errors": sum(1 for p in pts if p == 0),
            "segments_best_or_tied": best,
        }
    usage = {k: sum(r["usage"][k] for r in rows) for k in ("input_tokens", "output_tokens")}
    return {
        "generated_utc": now_utc(),
        "benchmark": BENCH,
        "judge": JUDGE_MODEL,
        "prompt_version": PROMPT_VERSION,
        "segments": len(rows),
        "ref_words": words,
        "usage": usage,
        "arms": arms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-locality", type=int, default=60)
    parser.add_argument("--report", action="store_true", help="Only print the summary of cached judgements.")
    parser.add_argument("--round", default="", help="Name of a round with its own arms, cache and summary.")
    parser.add_argument("--arms", help="Comma-separated arms for --round (2 to 6).")
    parser.add_argument("--provider", choices=["foundry", "api", "aitta", "azure"], default="foundry")
    parser.add_argument("--judge-model", default="", help="Aitta model id, or Azure deployment name.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.provider in ("aitta", "azure") and not args.judge_model:
        parser.error(f"--provider {args.provider} needs --judge-model")
    if args.round:
        if not args.arms or not 2 <= len(args.arms.split(",")) <= 6:
            parser.error("--round needs --arms with 2 to 6 arm ids")
        configure(args.arms.split(","), args.round, args.provider, args.judge_model)
    else:
        configure(ARMS, "", args.provider, args.judge_model)
    env = {"api": ENV_API, "aitta": ENV_AITTA, "azure": ENV_AZURE}.get(PROVIDER, ENV)
    suffix = f"-{ROUND}" if ROUND else ""
    judge_suffix = "" if PROVIDER not in ("aitta", "azure") else f"-{JUDGE_ID}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUT_DIR / f"{JUDGE_ID}{suffix}.jsonl"
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["prompt_sha256"] == PROMPT_SHA:
                done[row["id"]] = row
    sample = load_sample(args.per_locality)
    todo = [] if args.report else [i for i in sample if i["id"] not in done]
    if todo and any(not os.environ.get(e) for e in env):
        print(f"[skip] missing {', '.join(e for e in env if not os.environ.get(e))}")
        todo = []
    print(f"{len(sample)} segments in the sample, {len(todo)} to judge")
    failures = 0
    with ThreadPoolExecutor(args.workers) as pool, cache_path.open("a", encoding="utf-8", newline="\n") as cache:
        for fut in as_completed({pool.submit(judge, i): i for i in todo}):
            try:
                row = fut.result()
            except Exception as exc:  # noqa: BLE001  (one failed segment must not lose the rest)
                failures += 1
                print(f"[FAIL] {type(exc).__name__}: {exc}")
                continue
            done[row["id"]] = row
            cache.write(json.dumps(row, ensure_ascii=False) + "\n")
            cache.flush()
    rows = [done[i["id"]] for i in sample if i["id"] in done]
    if not rows:
        return 1
    summary = summarise(rows)
    summary["arms_judged_together"] = ARMS
    (OUT_DIR / f"summary{suffix}{judge_suffix}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    if todo:  # one sidecar per round and judge: _run_metadata[-<round>][-<second judge>].json
        write_run_metadata(
            OUT_DIR,
            build_run_metadata(
                app_under_test="Finnish ASR model comparison (LA-murre dialect test, meaning judge)",
                dataset_path=RESULTS_DIR / "benchmarks" / BENCH,
                runner_path=Path(__file__),
                model_or_deployment=[JUDGE_MODEL],
                env_names=sorted(env),
                extra={
                    "prompt_version": PROMPT_VERSION,
                    "prompt_sha256": PROMPT_SHA,
                    "segments": len(rows),
                    "round": ROUND or "1",
                    "provider": PROVIDER,
                },
            ),
            name=f"_run_metadata{suffix}{judge_suffix}.json",
        )
    print(f"{len(rows)} segments, {summary['ref_words']} reference words, tokens {summary['usage']}")
    for arm, s in sorted(summary["arms"].items(), key=lambda kv: kv[1]["error_points_per_100w"]):
        print(
            f"{arm:32s} {s['error_points_per_100w']:6.2f} pts/100w  crit {s['critical']:3d} major {s['major']:3d} "
            f"minor {s['minor']:3d}  clean {s['segments_without_errors']:3d}  best/tied {s['segments_best_or_tied']:3d}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
