"""LLM-as-judge for Finnish transcripts: which differences from the ground truth matter?

WER charges a dropped filler, a colloquial form ("mä" for "minä") or a compound written
as two words as much as a wrong clinical term or a wrong name. A reader of the
transcript cares about the second kind. Two blind judges read HY's Finnish ground
truth and one transcript (no model name) and return:

  errors     every meaning-relevant difference: wrong word or term, wrong name, wrong
             number, omission, hallucinated addition; severity critical / major / minor.
             Style differences (fillers, repetitions, spoken vs written forms, compound
             spacing, punctuation, case, digits vs words) are explicitly NOT errors.
  entities   each of the clip's names and brands: correct / variant / wrong / missing.
             "variant" = recognisably the right entity with a spelling slip.
  ref_doubts places where the transcript looks right and the ground truth looks wrong;
             reported, never scored (QAD-1 found 42 such spots by voting, without listening).
  usability  0-100: could a reader use this transcript in place of the reference?

Score per arm: meaning-error points per 100 reference words (critical 10, major 5,
minor 1, as in QAD-1's translation MQM), and entities correct out of the list.

Judges: Claude Opus 5.5 (Azure Foundry) and Gemini 3.1 Pro (Vertex), the QAD-1 judge
pair. Gemini also judges the Gemini ASR arms (vertex_gemini-*), blind; check those
against Claude for self-preference. The recorded results used Azure Foundry for Claude;
--claude-provider api runs the same model, prompt and schema with an Anthropic API key.

Cache: results/judgements/<judge>/<arm>.json, skipped while the transcript, prompt
version and judge model are unchanged. Only the arms in results/transcriptions/ (and qad1/)
are judged; HY's own tool transcripts are not part of this repository.

Usage (from the repo root, after uv run python scripts/get_clip.py):
    uv run --extra judges python scripts/judge_asr.py [--judge claude-opus-5-5] [--arm ID] [--force]
    uv run python scripts/judge_asr.py --report
Keys (repo-root .env): ANTHROPIC_FOUNDRY_RESOURCE + ANTHROPIC_FOUNDRY_API_KEY (or ANTHROPIC_API_KEY
with --claude-provider api), GOOGLE_VERTEXAI_PROJECT + GOOGLE_APPLICATION_CREDENTIALS.
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
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pydantic import BaseModel  # noqa: E402

from fi_common import APP_UNDER_TEST, GT_FI_PATH, JUDGE_DIR, load_arms, now_utc, read_text, require_clip  # noqa: E402
from provenance import build_run_metadata, load_repo_env, write_run_metadata  # noqa: E402

load_repo_env(Path(__file__))

PROMPT_VERSION = "asr-judge-v1"
SEVERITY_POINTS = {"critical": 10, "major": 5, "minor": 1}
MAX_WORKERS = 4
RETRIES = 2

# Canonical spellings (the GT itself writes "Densply Sirona"); inflected forms count as correct.
ENTITIES = [
    "Timo Suojärvi",
    "Martta Martola",
    "Peter",
    "Ilkka Pallonen",
    "Virpi Myller",
    "Clarion (hotelli)",
    "Straumann",
    "Dentsply Sirona",
    "Nobel Biocare",
    "Camlog-Conelog",
    "Implantona Oy",
    "Futudent",
    "peri-implantiitti",
    "implanttiajokortti / ajokortti (kurssin nimi)",
]

SYSTEM_PROMPT = """\
You evaluate a Finnish speech-to-text transcript against a reference transcript of the same
180-second recording (a Finnish dental-education webinar, two speakers). You see text only,
never the audio, and you are not told which system produced the transcript.

Your job is to find the differences that change what a reader learns, not to count
spelling differences. Read both texts completely, align them in your head, then report.

ERRORS (scored). Report each place where the transcript, read on its own, would mislead or
lose information compared with the reference:
- wrong_word: a different word or term, or an inflection that changes the meaning
- wrong_name: a person, company, product, place or course name that is wrong or garbled
  beyond recognition (a small spelling slip in a recognisable name is minor)
- wrong_number: a different number, quantity, duration or count
- omission: content missing from the transcript (not fillers)
- addition: content in the transcript that is not in the reference (hallucination or
  repeated sentences), not fillers
Severity:
- critical: a reader would come away with a false fact (wrong number, wrong person/company,
  negation flipped, a clinical term replaced by a different one), or a whole sentence or
  more is missing or invented
- major: the meaning of a phrase is lost or garbled, a name is unrecognisable, a clause is missing
- minor: the meaning survives but a word is wrong, a name is misspelled yet recognisable,
  or a short meaningful word is missing

NOT ERRORS (never report these): fillers and false starts ("niinku", "tota", "elikkä",
"semmoinen") present or absent; repetitions; spoken vs standard Finnish forms ("mä"/"minä",
"on tärkeä"/"on tärkeää", "tiiä"/"tiedä"); a word split or joined as a compound
("implantti hoito"/"implanttihoito"); punctuation, capitalisation, hyphenation; digits vs
number words; an equivalent inflection or word order that says the same thing.

REFERENCE DOUBTS (not scored). The reference was made by correcting machine output and may
contain mistakes. When the transcript is plausible and the reference looks wrong (a
misspelled brand, a word that makes no sense in context), list it under ref_doubts and do
not count it as an error.

ENTITIES. For each listed entity say whether the transcript has it: correct (any inflected
form, right spelling), variant (recognisably the same entity, spelling slip), wrong (a
different or garbled name), missing (not mentioned where the reference mentions it).
Judge against the canonical spelling given, not against the reference's spelling.

Quote spans exactly as they appear in the texts. Write explanations in English, briefly.
usability: 0-100, how well the transcript could replace the reference for a reader who
needs the facts (names, dates, numbers, course content). 100 = no meaning lost."""

USER_TEMPLATE = """\
ENTITIES TO CHECK:
{entities}

REFERENCE TRANSCRIPT:
<<<
{reference}
>>>

TRANSCRIPT TO EVALUATE:
<<<
{hypothesis}
>>>"""


class AsrError(BaseModel):
    category: Literal["wrong_word", "wrong_name", "wrong_number", "omission", "addition"]
    severity: Literal["critical", "major", "minor"]
    reference_span: str
    transcript_span: str
    explanation: str


class EntityCheck(BaseModel):
    entity: str
    status: Literal["correct", "variant", "wrong", "missing"]
    transcript_form: str


class RefDoubt(BaseModel):
    reference_span: str
    transcript_span: str
    explanation: str


class AsrJudgement(BaseModel):
    errors: list[AsrError]
    entities: list[EntityCheck]
    ref_doubts: list[RefDoubt]
    usability: int
    summary: str


def _strict(schema: dict[str, Any]) -> dict[str, Any]:
    """Pydantic JSON schema -> the strict form Claude's structured outputs want."""
    defs = schema.pop("$defs", {})

    def fix(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return fix(dict(defs[node["$ref"].split("/")[-1]]))
            node = {k: fix(v) for k, v in node.items() if k != "title"}
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            return node
        if isinstance(node, list):
            return [fix(x) for x in node]
        return node

    return fix(schema)


JSON_SCHEMA = _strict(AsrJudgement.model_json_schema())

JUDGES = {
    "claude-opus-5-5": {
        "provider": "azure-foundry-anthropic",
        "model": "claude-opus-5-5",
        "env": ("ANTHROPIC_FOUNDRY_RESOURCE", "ANTHROPIC_FOUNDRY_API_KEY"),
        "params": {"max_tokens": 16000, "thinking": "adaptive", "effort": "high"},
    },
    "gemini-3.1-pro": {
        "provider": "google-vertex",
        "model": "gemini-3.1-pro-preview",
        "env": ("GOOGLE_VERTEXAI_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS"),
        "params": {"max_output_tokens": 32768, "thinking_level": "high"},
    },
}


class JudgeOutputError(RuntimeError):
    pass


CLAUDE_API_ENV = ("ANTHROPIC_API_KEY",)


def call_claude(j: dict, prompt: str) -> tuple[dict, dict, str]:
    import anthropic

    if j.get("via_api"):
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=0, timeout=900)
    else:
        client = anthropic.AnthropicFoundry(
            resource=os.environ["ANTHROPIC_FOUNDRY_RESOURCE"],
            api_key=os.environ["ANTHROPIC_FOUNDRY_API_KEY"],
            max_retries=0,
            timeout=900,
        )
    resp = client.messages.create(
        model=j["model"],
        max_tokens=j["params"]["max_tokens"],
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        thinking={"type": "adaptive"},
        output_config={"effort": j["params"]["effort"], "format": {"type": "json_schema", "schema": JSON_SCHEMA}},
    )
    if resp.stop_reason != "end_turn":
        raise JudgeOutputError(f"stop_reason={resp.stop_reason}")
    text = "".join(b.text for b in resp.content if b.type == "text")
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    return json.loads(text), usage, resp.model


def call_gemini(j: dict, prompt: str) -> tuple[dict, dict, str]:
    from google import genai
    from google.genai import types

    client = genai.Client(
        vertexai=True,
        project=os.environ["GOOGLE_VERTEXAI_PROJECT"],
        location=os.environ.get("GOOGLE_VERTEXAI_LOCATION", "global"),
    )
    resp = client.models.generate_content(
        model=j["model"],
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=AsrJudgement,
            max_output_tokens=j["params"]["max_output_tokens"],
            thinking_config=types.ThinkingConfig(thinking_level=j["params"]["thinking_level"]),
        ),
    )
    finish = str(resp.candidates[0].finish_reason) if resp.candidates else "NO_CANDIDATES"
    if not finish.endswith("STOP"):
        raise JudgeOutputError(f"finish_reason={finish}")
    meta = resp.usage_metadata
    usage = {"input_tokens": meta.prompt_token_count, "output_tokens": meta.candidates_token_count}
    return json.loads(resp.text or ""), usage, resp.model_version or j["model"]


CALLERS = {"azure-foundry-anthropic": call_claude, "google-vertex": call_gemini}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


PROMPT_SHA = sha(SYSTEM_PROMPT + USER_TEMPLATE + json.dumps(JSON_SCHEMA, sort_keys=True) + json.dumps(ENTITIES))


def out_path(judge_id: str, arm_id: str) -> Path:
    return JUDGE_DIR / judge_id / f"{arm_id}.json"


def roll_up(ans: dict, ref_words: int) -> dict:
    AsrJudgement.model_validate(ans)
    points = sum(SEVERITY_POINTS[e["severity"]] for e in ans["errors"])
    counts = {s: sum(1 for e in ans["errors"] if e["severity"] == s) for s in SEVERITY_POINTS}
    ent = {s: sum(1 for e in ans["entities"] if e["status"] == s) for s in ("correct", "variant", "wrong", "missing")}
    return {
        "error_points_per_100w": round(points / ref_words * 100, 2),
        "error_counts": counts,
        "entities": ent,
        "entities_ok": ent["correct"] + ent["variant"],
        "entities_total": len(ans["entities"]),
        "ref_doubts": len(ans["ref_doubts"]),
        "usability": max(0, min(100, int(ans["usability"]))),
    }


def judge_one(judge_id: str, arm, gt: str, ref_words: int) -> dict:
    j = JUDGES[judge_id]
    prompt = USER_TEMPLATE.format(entities="\n".join(f"- {e}" for e in ENTITIES), reference=gt, hypothesis=arm.text)
    last: Exception | None = None
    for attempt in range(RETRIES + 1):
        try:
            t0 = time.perf_counter()
            ans, usage, model_version = CALLERS[j["provider"]](j, prompt)
            rec = {
                "arm_id": arm.arm_id,
                "label": arm.label,
                "judge_id": judge_id,
                "judge_model": model_version,
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": PROMPT_SHA,
                "text_sha256": sha(arm.text),
                "created_at_utc": now_utc(),
                "latency_s": round(time.perf_counter() - t0, 1),
                "usage": usage,
                "attempts": attempt + 1,
                "scores": roll_up(ans, ref_words),
                "judgement": ans,
            }
            path = out_path(judge_id, arm.arm_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return rec
        except Exception as exc:  # noqa: BLE001 - retried, then reported
            last = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{judge_id}/{arm.arm_id}: {type(last).__name__}: {str(last)[:300]}")


def cached(judge_id: str, arm) -> bool:
    p = out_path(judge_id, arm.arm_id)
    if not p.exists():
        return False
    r = json.loads(p.read_text(encoding="utf-8"))
    return r.get("text_sha256") == sha(arm.text) and r.get("prompt_sha256") == PROMPT_SHA


def report(arms) -> None:
    print(f"{'arm':40s}" + "".join(f" {j[:12]:>26s}" for j in JUDGES))
    print(f"{'':40s}" + "".join(f" {'pts/100w ent use doubt':>26s}" for _ in JUDGES))
    for arm in arms:
        cells = []
        for jid in JUDGES:
            p = out_path(jid, arm.arm_id)
            if p.exists():
                s = json.loads(p.read_text(encoding="utf-8"))["scores"]
                cells.append(
                    f"{s['error_points_per_100w']:8.1f} {s['entities_ok']:2d}/{s['entities_total']:<2d} "
                    f"{s['usability']:3d} {s['ref_doubts']:5d}"
                )
            else:
                cells.append(f"{'-':>26s}")
        print(f"{arm.arm_id:40s}" + "".join(f" {c:>26s}" for c in cells))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge", action="append", choices=list(JUDGES))
    parser.add_argument("--arm", action="append")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument(
        "--claude-provider",
        choices=["foundry", "api"],
        default="foundry",
        help="Claude through Azure Foundry (as recorded) or the Anthropic API (ANTHROPIC_API_KEY).",
    )
    args = parser.parse_args()
    if args.claude_provider == "api":
        JUDGES["claude-opus-5-5"] = {
            **JUDGES["claude-opus-5-5"],
            "provider": "azure-foundry-anthropic",
            "via_api": True,
            "env": CLAUDE_API_ENV,
        }

    arms = [a for a in load_arms() if not args.arm or a.arm_id in args.arm]
    if args.report:
        report(arms)
        return 0
    require_clip("reference.txt")
    gt = read_text(GT_FI_PATH)
    ref_words = len(gt.split())
    judges = args.judge or list(JUDGES)
    jobs = []
    for jid in judges:
        missing = [e for e in JUDGES[jid]["env"] if not os.environ.get(e)]
        if missing:
            print(f"[skip] {jid}: missing {', '.join(missing)}")
            continue
        jobs += [(jid, a) for a in arms if args.force or not cached(jid, a)]
    print(f"{len(jobs)} judgements to run")
    failures = 0
    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        futs = {pool.submit(judge_one, jid, a, gt, ref_words): (jid, a.arm_id) for jid, a in jobs}
        for f in as_completed(futs):
            try:
                r = f.result()
                s = r["scores"]
                print(
                    f"[ ok ] {r['judge_id']:16s} {r['arm_id']:40s} "
                    f"{s['error_points_per_100w']:5.1f} pts/100w, usability {s['usability']}"
                )
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"[FAIL] {exc}")
    write_run_metadata(
        JUDGE_DIR,
        build_run_metadata(
            app_under_test=APP_UNDER_TEST,
            dataset_path=GT_FI_PATH,
            runner_path=Path(__file__),
            model_or_deployment=[JUDGES[j]["model"] for j in judges],
            env_names=sorted({e for j in judges for e in JUDGES[j]["env"]}),
            extra={
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": PROMPT_SHA,
                "ref_words_whitespace": ref_words,
                "claude_provider": args.claude_provider,
            },
        ),
    )
    report(arms)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
