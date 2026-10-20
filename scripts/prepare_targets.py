"""Training targets for the dialect fine-tune, written by an LLM from human transcripts.

Two text-only jobs for Claude Opus 5.5. Neither hears audio and neither touches a test set:
the outputs are TRAINING and VALIDATION targets only, never ground truth for scoring.

  punct      SKN's normalised transcript has no punctuation and almost no capitals, so a model
             trained on it stops writing them. The LLM adds punctuation and capitals and must
             keep every word: an answer whose words differ from the input is thrown away and
             the segment keeps its plain text.
  normalise  LA-murre has only a dialect transcript. The LLM rewrites it in standard Finnish
             word forms with punctuation, word for word, without rephrasing. An answer whose
             word count is off by more than 30 % is thrown away and the segment is dropped from
             training. Only the `la-murre` revision was trained on these targets.

Input: a JSONL of segments (prepare_skn.py: field "ref"; prepare_lamurre.py --train: field "ref").
Output: the same rows with "target" added, in results/labels/<name>.targets.jsonl, plus
<name>.targets.stats.json. Calls are cached per batch in <name>.targets.cache.jsonl.
The SKN outputs (CC BY 4.0) are committed; every lamurre_* output except the .stats.json files is
gitignored, because LA-murre (CC BY-ND 4.0) may not be shared in modified form.

Prompt versions: the punct prompt is the one that made results/labels/skn_*.targets.jsonl
(prompt_sha256 3f5919ad...). The normalise prompt that made the `la-murre` revision's targets
(prompt_sha256 88232dbd...) gave four example word pairs from the LA-murre corpus; this public copy
gives example pairs from SKN instead (segment SKN45a_Heinola_0097, CC BY 4.0), so its hash differs.

Usage (from the repo root):
    uv run --extra judges python scripts/prepare_targets.py punct results/labels/skn_train.jsonl
    uv run --extra judges python scripts/prepare_targets.py normalise results/labels/lamurre_train.jsonl
    add --limit 100 for a trial. Needs ANTHROPIC_FOUNDRY_RESOURCE and ANTHROPIC_FOUNDRY_API_KEY in the
    repo-root .env, or ANTHROPIC_API_KEY with --provider api.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pydantic import BaseModel  # noqa: E402

from fi_common import PROJECT_ROOT, now_utc  # noqa: E402
from provenance import build_run_metadata, load_repo_env, write_run_metadata  # noqa: E402

load_repo_env(Path(__file__))

MODEL = "claude-opus-5-5"
ENV = ("ANTHROPIC_FOUNDRY_RESOURCE", "ANTHROPIC_FOUNDRY_API_KEY")
ENV_API = ("ANTHROPIC_API_KEY",)
PROVIDER = "foundry"
BATCH = 20
PROMPTS = {
    "punct": """You add punctuation and capital letters to transcripts of spoken Finnish.

Each numbered line is one stretch of an interview, already in standard Finnish word forms but
without punctuation. Return every line with sentence punctuation (. , ? !) and capitals at
sentence starts and in names. Rules:
- Keep EVERY word exactly as it is and in the same order. Do not add, remove, correct or
  merge words, even when a word looks wrong, is repeated or is cut off.
- Do not turn number words into digits.
- A lone "-" marks a break in the speech: leave it out.
- Speech is often unfinished: end the line with a full stop anyway.""",
    "normalise": """You rewrite dialect transcripts of spoken Finnish in standard Finnish spelling.

Each numbered line is one stretch of an interview, written by a linguist in dialect spelling
(for example "miekii" = minäkin, "viej" = vien, "juomaakii" = juomaakin, "irvistie" = irvistää).
Return every line in standard Finnish word forms with sentence punctuation and capitals. Rules:
- Word for word: each dialect word becomes its standard form, in the same order. Do not
  rephrase, summarise, complete unfinished sentences or drop repetitions and fillers.
- Keep a dialect word that has no standard counterpart as it is.
- Write numbers as words.
- If a word is unintelligible to you, keep it as written.""",
}


class Line(BaseModel):
    n: int
    text: str


class Lines(BaseModel):
    lines: list[Line]


SCHEMA = Lines.model_json_schema()
SCHEMA["additionalProperties"] = False
SCHEMA["$defs"]["Line"]["additionalProperties"] = False


def words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower().replace("-", " ")).split()


def accept(mode: str, source: str, target: str) -> bool:
    a, b = words(source), words(target)
    if not b:
        return False
    if mode == "punct":
        return a == b
    return 0.7 <= len(b) / max(1, len(a)) <= 1.3


def call(mode: str, batch: list[dict]) -> dict:
    import anthropic

    if PROVIDER == "api":
        client = anthropic.Anthropic(api_key=os.environ[ENV_API[0]], max_retries=4, timeout=600)
    else:
        client = anthropic.AnthropicFoundry(
            resource=os.environ[ENV[0]], api_key=os.environ[ENV[1]], max_retries=4, timeout=600
        )
    prompt = "\n".join(f"{i + 1}. {row['ref']}" for i, row in enumerate(batch))
    resp = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=PROMPTS[mode],
        messages=[
            {"role": "user", "content": prompt + f"\n\nReturn exactly {len(batch)} lines, n = 1 to {len(batch)}."}
        ],
        thinking={"type": "adaptive"},
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
    )
    if resp.stop_reason != "end_turn":
        raise RuntimeError(f"stop_reason={resp.stop_reason}")
    out = Lines.model_validate_json("".join(b.text for b in resp.content if b.type == "text"))
    texts = {line.n: line.text.strip() for line in out.lines}
    return {
        "ids": [row["id"] for row in batch],
        "targets": [texts.get(i + 1, "") for i in range(len(batch))],
        "usage": {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=list(PROMPTS))
    parser.add_argument("input", type=Path, help="JSONL, relative to the repository root or absolute.")
    parser.add_argument(
        "--provider", choices=["foundry", "api"], default="foundry", help="Claude via Azure Foundry or Anthropic API."
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    global PROVIDER
    PROVIDER = args.provider
    env = ENV_API if PROVIDER == "api" else ENV

    src = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows[: args.limit] if args.limit else rows
    stem = src.with_suffix("")
    cache_path = Path(f"{stem}.targets.cache.jsonl")
    prompt_sha = hashlib.sha256(PROMPTS[args.mode].encode()).hexdigest()
    done: dict[str, str] = {}
    usage = {"input_tokens": 0, "output_tokens": 0}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["prompt_sha256"] == prompt_sha:
                done.update(zip(rec["ids"], rec["targets"], strict=True))
                usage = {k: usage[k] + rec["usage"][k] for k in usage}
    todo = [r for r in rows if r["id"] not in done]
    batches = [todo[i : i + BATCH] for i in range(0, len(todo), BATCH)]
    if batches and any(not os.environ.get(e) for e in env):
        print(f"[skip] missing {', '.join(e for e in env if not os.environ.get(e))}")
        return 1
    print(f"{len(rows)} segments, {len(todo)} to do in {len(batches)} calls", flush=True)
    failures = 0
    with ThreadPoolExecutor(args.workers) as pool, cache_path.open("a", encoding="utf-8", newline="\n") as cache:
        for n, fut in enumerate(as_completed([pool.submit(call, args.mode, b) for b in batches]), 1):
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001  (a failed batch is retried on the next run)
                failures += 1
                print(f"[FAIL] {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                continue
            done.update(zip(rec["ids"], rec["targets"], strict=True))
            usage = {k: usage[k] + rec["usage"][k] for k in usage}
            cache.write(json.dumps({**rec, "prompt_sha256": prompt_sha}, ensure_ascii=False) + "\n")
            cache.flush()
            if n % 50 == 0:
                print(f"{n}/{len(batches)} calls, tokens {usage}", flush=True)

    out_rows, accepted = [], 0
    for r in rows:
        target = done.get(r["id"], "")
        ok = accept(args.mode, r["ref"], target)
        accepted += ok
        if ok:
            out_rows.append({**r, "target": target, "target_by": MODEL})
        elif args.mode == "punct":  # keep the segment with its human text
            out_rows.append({**r, "target": r["ref"], "target_by": "source"})
    Path(f"{stem}.targets.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8", newline="\n"
    )
    stats = {
        "generated_utc": now_utc(),
        "mode": args.mode,
        "model": MODEL,
        "prompt_sha256": prompt_sha,
        "segments": len(rows),
        "llm_target_accepted": accepted,
        "rows_written": len(out_rows),
        "failed_calls": failures,
        "usage": usage,
    }
    Path(f"{stem}.targets.stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8", newline="\n")
    if batches:
        write_run_metadata(
            src.parent,
            build_run_metadata(
                app_under_test="Finnish ASR dialect fine-tune: LLM-written training targets",
                dataset_path=src,
                runner_path=Path(__file__),
                model_or_deployment=[MODEL],
                env_names=sorted(env),
                extra={"mode": args.mode, "prompt_sha256": prompt_sha, "provider": PROVIDER},
            ),
        )
    print(json.dumps(stats, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
