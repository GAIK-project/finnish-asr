r"""One HTML page to listen to the audio and compare, side by side, what the models wrote.

    uv run --extra hub python scripts/build_compare.py
    uv run --extra hub python scripts/build_compare.py --parquet <SKN test parquet>

Writes results/compare.html (one file: inline CSS and JS, audio embedded as base64, no network
requests) and results/_compare_run_metadata.json. No API keys, no GPU. Both files are gitignored: the
page embeds the Ajokortti clip, which is published only in gaik-toolkit, so build it locally after
`uv run python scripts/get_clip.py`. Two sections:

A. The Ajokortti webinar clip (180 s, HY's reference): one row per reference sentence with a play
   button for its span, and what Whisper large-v3 (chunked), our published model, Gemini 3.5 Flash and
   ElevenLabs Scribe v2 (repeat 1) wrote for it. A model's span comes from aligning its whole transcript
   to the reference tokens under HY's normalisation with line breaks read as spaces, the README's WER,
   so each model's errors over all rows add up to its WER (checked). Inserted words on a sentence
   boundary go to the sentence the model's own punctuation puts them in. QAD-1's 42 reference doubts
   are marked in the reference. Sentence spans, token times and doubts come from
   results/clip/gt_alignment.json, a frozen copy of QAD-1's alignment of the public reference.
B. The SKN dialect test: the 20 segments of docs/DIALECT-EXAMPLES.md (the same rule, imported from
   dialect_examples.py), with audio from the public dataset GAIK-project/skn-finnish-dialects-asr
   (test parquet, 237 MB). Without --parquet it is downloaded into a temporary folder that is deleted
   afterwards; audio is re-encoded to 16 kHz mono MP3 when ffmpeg is on PATH, otherwise kept as WAV.

The LA-murre meaning judge is not on this page: its segments are LA-murre text (CC BY-ND 4.0).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asr_scoring  # noqa: E402
import dialect_examples as dx  # noqa: E402
from fi_common import (  # noqa: E402
    ASR_DIR,
    AUDIO_PATH,
    GT_ALIGNMENT_PATH,
    GT_FI_PATH,
    JUDGE_DIR,
    QAD1_ASR_DIR,
    RESULTS_DIR,
    SCORES_PATH,
    load_arms,
    rel,
    require_clip,
)
from provenance import build_run_metadata, write_run_metadata  # noqa: E402

OUT = RESULTS_DIR / "compare.html"
META = RESULTS_DIR / "_compare_run_metadata.json"

# (arm id, label on the page, detail line)
CLIP_ARMS = [
    ("hf_whisper-large-v3-chunks", "Whisper large-v3", "openai/whisper-large-v3, paloittain"),
    (
        "ft_whisper-large-v3-fi-skn-punct-best-chunks",
        "Oma malli (julkaistu)",
        "GAIK-project/whisper-large-v3-fi-dialects, paloittain",
    ),
    ("vertex_gemini-3.5-flash", "Gemini 3.5 Flash", "gemini-3.5-flash, ajo 1/3"),
    ("elevenlabs_scribe-v2", "ElevenLabs Scribe v2", "scribe_v2, ajo 1/3"),
]
JUDGES = [("claude-opus-5-5", "Claude Opus 5.5"), ("gemini-3.1-pro", "Gemini 3.1 Pro")]
PAD_S = 0.2  # play a sentence from 0.2 s before its first word to 0.2 s after its last

HF_DATA_REPO = "GAIK-project/skn-finnish-dialects-asr"
HF_TEST_FILE = "data/test-00000-of-00001.parquet"
SKN_ARMS = [(dx.BASE, "Whisper large-v3"), (dx.OURS, "Oma malli (julkaistu)")]

SKN_CREDIT = "Suomen kielen näytteitä, Kotimaisten kielten keskus (Kotus), via Kielipankki, CC BY 4.0"
SENTENCE_END = re.compile(r"[.?!][\"'»”)\]]*$")

CLIP_NORM = asr_scoring.ws_fixed_transform(False)  # HY's normalisation, every whitespace read as a space (WER_nl)
SKN_NORM = dx.transform(False)  # lower-case, punctuation removed (the SKN WER)


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def mmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"


def data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def display_words(text: str, norm: jiwer.Compose) -> tuple[list[str], list[str]]:
    """(words as written, their normalised tokens), one each; punctuation-only pieces join a neighbour."""
    shown: list[str] = []
    toks: list[str] = []
    lead = ""
    for piece in text.split():
        t = [x for x in norm(piece)[0] if x]
        if not t:
            if shown:
                shown[-1] += " " + piece
            else:
                lead += piece + " "
            continue
        assert len(t) == 1, piece
        shown.append(lead + piece)
        toks.append(t[0])
        lead = ""
    assert toks == [x for x in norm(text)[0] if x], "per-word and whole-text normalisation differ"
    return shown, toks


def align(ref: list[str], hyp: list[str]) -> list[dict]:
    """QAD-1's alignment rows (equal/sub/del/ins), each hypothesis row with its word index "h"."""
    if not hyp:
        return [{"op": "del", "ref": r, "hyp": None, "ref_idx": i} for i, r in enumerate(ref)]
    split = jiwer.ReduceToListOfListOfWords()
    out = jiwer.process_words(" ".join(ref), " ".join(hyp), split, split)
    rows = asr_scoring.alignment_rows(ref, hyp, out.alignments[0])
    h = 0
    for r in rows:
        if r["op"] != "del":
            r["h"] = h
            h += 1
    assert h == len(hyp)
    return rows


def n_errors(rows: list[dict]) -> int:
    return sum(r["op"] != "equal" for r in rows)


def by_sentence(rows: list[dict], shown: list[str], bounds: list[tuple[int, int]]) -> list[list[dict]]:
    """Alignment rows per reference sentence [start, end); insertions on a boundary follow the punctuation."""

    def sentence_of(i: int) -> int:
        return next(k for k, (s, e) in enumerate(bounds) if s <= i < e)

    out: list[list[dict]] = [[] for _ in bounds]
    i = 0
    while i < len(rows):
        r = rows[i]
        if r["op"] != "ins":
            out[sentence_of(r["ref_idx"])].append(r)
            i += 1
            continue
        j = i
        while j < len(rows) and rows[j]["op"] == "ins" and rows[j]["ref_idx"] == r["ref_idx"]:
            j += 1
        run, p, i = rows[i:j], r["ref_idx"], j
        if p == 0:
            out[0] += run
        elif p >= bounds[-1][1]:
            out[-1] += run
        elif bounds[k := sentence_of(p)][0] != p:
            out[k] += run  # inside a sentence
        else:  # between sentence k-1 and k: words up to the first sentence end close k-1
            prev = shown[run[0]["h"] - 1] if run[0]["h"] > 0 else "."
            cut = 0
            if not SENTENCE_END.search(prev):
                cut = next((m + 1 for m, x in enumerate(run) if SENTENCE_END.search(shown[x["h"]])), len(run))
            out[k - 1] += run[:cut]
            out[k] += run[cut:]
    return out


def render_hyp(rows: list[dict], shown: list[str]) -> str:
    parts = []
    for r in rows:
        if r["op"] == "equal":
            parts.append(esc(shown[r["h"]]))
        elif r["op"] == "sub":
            parts.append(f'<span class="sub" title="Referenssissä: {esc(r["ref"])}">{esc(shown[r["h"]])}</span>')
        elif r["op"] == "ins":
            parts.append(f'<span class="ins" title="Ei referenssissä">{esc(shown[r["h"]])}</span>')
        else:
            parts.append(f'<del title="Puuttuu: {esc(r["ref"])}">{esc(r["ref"])}</del>')
    return " ".join(parts) or '<span class="muted">(ei mitään)</span>'


# ---------------------------------------------------------------------------
# A. Ajokortti clip
# ---------------------------------------------------------------------------


def flag_title(f: dict) -> str:
    alts = ", ".join(f"{a['text'] or '(ei mitään)'} ({len(a['arms'])})" for a in f["alternatives"])
    return f"QAD-1 #{f['flag_id']}: {f['n_disagree']}/{f['n_systems']} järjestelmää kirjoitti toisin: {alts}"


def render_gt(sent: dict, gt_tokens: list[str], flags: list[dict], last: bool) -> tuple[str, list[dict]]:
    """The reference sentence with flagged words marked, and the flags that touch it."""
    ts, te = sent["token_start"], sent["token_end"]
    shown, toks = display_words(sent["text"], CLIP_NORM)
    if toks != gt_tokens[ts:te]:  # cannot map the written words: show the tokens
        shown = gt_tokens[ts:te]
    mine = [f for f in flags if (ts <= f["ref_start"] < te) or (last and f["ref_start"] == te)]
    mine += [f for f in flags if f["ref_start"] < ts < f["ref_end"] and f not in mine]
    parts = []
    for i in range(ts, te + (1 if last else 0)):
        for f in flags:
            if f["gt_missing"] and f["ref_start"] == i:
                parts.append(f'<mark class="flag gap" title="{esc(flag_title(f))}">⁁</mark>')
        if i == te:
            break
        word = esc(shown[i - ts])
        hit = [f for f in flags if f["ref_start"] <= i < f["ref_end"]]
        parts.append(f'<mark class="flag" title="{esc(flag_title(hit[0]))}">{word}</mark>' if hit else word)
    return " ".join(parts), sorted(mine, key=lambda f: f["ref_start"])


def section_clip() -> tuple[str, dict]:
    q = json.loads(GT_ALIGNMENT_PATH.read_text(encoding="utf-8"))
    sents, gt_tok_rows, flags = q["gt_sentences"], q["gt_tokens"], q["gt_audit"]["flags"]
    gt_tokens = [t["token"] for t in gt_tok_rows]
    gt_text = GT_FI_PATH.read_text(encoding="utf-8")
    assert [x for x in CLIP_NORM(gt_text)[0] if x] == gt_tokens, "GT tokens differ from the normalised GT"
    bounds = [(s["token_start"], s["token_end"]) for s in sents]
    assert bounds[0][0] == 0 and bounds[-1][1] == len(gt_tokens)
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:], strict=False)), "sentences are not contiguous"

    scores = {a["arm_id"]: a for a in json.loads(SCORES_PATH.read_text(encoding="utf-8"))["arms"]}
    arms = {a.arm_id: a for a in load_arms()}
    models = []
    for arm_id, label, detail in CLIP_ARMS:
        arm = arms[arm_id]
        assert arm.repeat == 1, arm_id
        shown, toks = display_words(arm.text, CLIP_NORM)
        rows = align(gt_tokens, toks)
        wer = round(n_errors(rows) / len(gt_tokens) * 100, 2)
        sc = scores[arm_id]
        if abs(wer - sc["WER_nl"]) > 0.005:
            raise SystemExit(f"{arm_id}: aligned WER {wer} differs from scores.json WER_nl {sc['WER_nl']}")
        judged = {}
        for jid, _ in JUDGES:
            j = json.loads((JUDGE_DIR / jid / f"{arm_id}.json").read_text(encoding="utf-8"))["scores"]
            judged[jid] = j
        models.append(
            {
                "arm_id": arm_id,
                "label": label,
                "detail": detail,
                "shown": shown,
                "per_sentence": by_sentence(rows, shown, bounds),
                "ops": {op: sum(r["op"] == op for r in rows) for op in ("sub", "del", "ins")},
                "wer": sc["WER_nl"],
                "wer_range": sc.get("WER_range"),
                "judged": judged,
            }
        )
        assert sum(n_errors(s) for s in models[-1]["per_sentence"]) == n_errors(rows)

    # summary table
    head = "".join(f"<th>{esc(name)}<br><small>pist. / 100 sanaa</small></th>" for _, name in JUDGES)
    trs = []
    for m in models:
        rng = f"<br><small>ajot {m['wer_range'][0]:.1f}–{m['wer_range'][1]:.1f}</small>" if m["wer_range"] else ""
        ops = m["ops"]
        judge_cells = "".join(
            f"<td>{m['judged'][jid]['error_points_per_100w']:.1f}"
            f"<br><small>nimet {m['judged'][jid]['entities_ok']}/{m['judged'][jid]['entities_total']}</small></td>"
            for jid, _ in JUDGES
        )
        trs.append(
            f"<tr><th scope=row>{esc(m['label'])}<br><small>{esc(m['detail'])}</small></th>"
            f"<td>{m['wer']:.1f}{rng}</td><td>{ops['sub']} / {ops['del']} / {ops['ins']}</td>{judge_cells}</tr>"
        )
    table = (
        '<div class="scroll"><table><thead><tr><th>Malli</th><th>WER %</th><th>S / D / I</th>'
        f"{head}</tr></thead><tbody>{''.join(trs)}</tbody></table></div>"
    )

    toggles = "".join(
        f'<label><input type="checkbox" data-arm="{k}" checked> {esc(m["label"])}</label>' for k, m in enumerate(models)
    )
    rows_html = []
    for k, s in enumerate(sents):
        gt_html, mine = render_gt(s, gt_tokens, flags, last=k == len(sents) - 1)
        n_ref = s["token_end"] - s["token_start"]
        if s.get("start") is not None and s.get("end") is not None:
            t0, t1 = s["start"], s["end"]  # the row is highlighted over its own span, played with padding
            play = (
                f'<button class="play" type="button" data-start="{max(0.0, t0 - PAD_S):.2f}" '
                f'data-end="{t1 + PAD_S:.2f}" '
                f'aria-label="Toista lause {k + 1} ({mmss(s["start"])}–{mmss(s["end"])})">▶</button>'
                f'<span class="time">{mmss(s["start"])}–{mmss(s["end"])}</span>'
            )
        else:
            t0, t1, play = -1, -1, '<span class="time">ei aikaa</span>'
        lines = []
        for m_i, m in enumerate(models):
            part = m["per_sentence"][k]
            e = n_errors(part)
            lines.append(
                f'<div class="line m{m_i}"><div class="who">{esc(m["label"])}</div>'
                f'<div class="txt">{render_hyp(part, m["shown"])}</div>'
                f'<div class="err{" zero" if e == 0 else ""}">{e} / {n_ref}</div></div>'
            )
        doubts = ""
        if mine:
            items = "; ".join(
                f"<b>{esc(f['gt_text'] or '(puuttuu)')}</b> → "
                + ", ".join(f"{esc(a['text'] or '(ei mitään)')} ({len(a['arms'])})" for a in f["alternatives"])
                for f in mine
            )
            doubts = f'<div class="doubts">QAD-1:n epäilyt: {items}</div>'
        rows_html.append(
            f'<article class="row" data-start="{t0:.2f}" data-end="{t1:.2f}">'
            f'<div class="rowhead"><span class="n">{k + 1}</span>{play}</div>'
            f'<div class="line gt"><div class="who">Referenssi</div><div class="txt">{gt_html}</div>'
            f'<div class="err">{n_ref} sanaa</div></div>{doubts}{"".join(lines)}</article>'
        )

    audio = AUDIO_PATH.read_bytes()
    body = f"""
<section id="a">
<h2>A. Webinaarileike (Ajokortti)</h2>
<p>180 s webinaarileike, jolla QAD-1 testasi litterointia. Jokainen rivi on yksi referenssilause: ▶ soittaa sen
kohdan. Mallin teksti lauseelle saadaan kohdistamalla koko litteraatti referenssin sanoihin (pienet kirjaimet,
välimerkit pois, rivinvaihdot välilyönteinä, kuten README:n WER), joten rivien virheet summautuvat mallin
WER-lukuun. Lauseen rajalle osuvat lisäsanat menevät sille lauseelle, johon mallin oma välimerkitys ne sijoittaa.
Numero oikealla: virheet / referenssin sanat.</p>
<p class="note"><b>Referenssi on HY:n litteraatti; kuulija tarkisti siitä vain kiistanalaiset kohdat
(<code>reference.reviewed.txt</code>, ei käytössä tällä sivulla).</b> QAD-1 merkitsi {len(flags)}
kohtaa, joissa vähintään {q["gt_audit"]["threshold"]} sen {len(q["gt_audit"]["systems"])} järjestelmästä
kirjoitti toisin (<mark class="flag">näin</mark>, ⁁ = referenssistä ehkä puuttuu sana; vaihtoehdot rivin alla
ja hiiren alla). Lauseiden ajat ovat QAD-1:n kohdistamia sana-aikoja, joten alku ja loppu voivat heittää.</p>
{table}
<p class="small">WER: mallin ajo 1 (Geminillä ja ElevenLabsilla kolme ajoa, vaihteluväli alla). S / D / I =
korvatut, puuttuvat ja lisätyt sanat koko leikkeellä. Tuomarit lukivat referenssin ja litteraatin tietämättä
mallia ja laskivat vain merkitysvirheet (kriittinen 10, vakava 5, lievä 1 pistettä per 100 referenssisanaa);
nimet = oikein kirjoitetut 14 nimestä ja termistä.</p>
<div class="player" id="player">
<audio id="clip" controls preload="metadata" src="{data_uri(audio, "audio/mpeg")}"
 aria-label="Ajokortti-leike, 180 sekuntia"></audio>
<div class="toggles">{toggles}</div>
</div>
<div id="rows">{"".join(rows_html)}</div>
</section>"""
    meta = {
        "sentences": len(sents),
        "gt_flags": len(flags),
        "arms": {m["arm_id"]: {"WER_nl": m["wer"], **m["ops"]} for m in models},
        "audio_bytes": len(audio),
    }
    return body, meta


# ---------------------------------------------------------------------------
# B. SKN dialect test
# ---------------------------------------------------------------------------


def to_mp3(wav: bytes) -> tuple[bytes, str]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return wav, "audio/wav"
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", "16000"]
    cmd += ["-b:a", "48k", "-f", "mp3", "pipe:1"]
    return subprocess.run(cmd, input=wav, capture_output=True, check=True).stdout, "audio/mpeg"


def skn_rows(ids: list[str], parquet: Path | None) -> tuple[dict[str, dict], dict]:
    """id -> {audio, municipality, seconds, text_dialect} for the picked segments, and the source record."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("pyarrow is needed: uv run --extra hub python scripts/build_compare.py") from None
    tmp, source = None, {}
    try:
        if parquet is None:
            tmp = Path(tempfile.mkdtemp(prefix="skn_compare_"))
            os.environ["HF_XET_CACHE"] = str(tmp / "xet")  # keep the Xet chunk cache in the temp folder too
            from huggingface_hub import hf_hub_download

            print(f"downloading {HF_DATA_REPO}/{HF_TEST_FILE} into {tmp}", flush=True)
            parquet = Path(hf_hub_download(HF_DATA_REPO, HF_TEST_FILE, repo_type="dataset", cache_dir=tmp))
        parts = Path(parquet).parts
        revision = parts[parts.index("snapshots") + 1] if "snapshots" in parts else None
        source.update(repo=HF_DATA_REPO, file=HF_TEST_FILE, revision=revision, sha256=sha256(parquet))
        cols = ["audio", "id", "municipality", "seconds", "text_dialect"]
        table = pq.read_table(parquet, columns=cols, filters=[("id", "in", ids)])
        rows = {r["id"]: r for r in table.to_pylist()}
        del table
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"temp folder {'still exists: ' if tmp.exists() else 'deleted: '}{tmp}", flush=True)
            source["temp_folder_deleted"] = not tmp.exists()
    missing = sorted(set(ids) - set(rows))
    if missing:
        raise SystemExit(f"not in the parquet: {missing}")
    return rows, source


def section_skn(parquet: Path | None) -> tuple[str, dict]:
    hyps = {arm: dx.load(arm) for arm, _ in SKN_ARMS}
    base = hyps[dx.BASE]
    picked = [(place, seg) for place, segs in sorted(dx.by_place(base).items()) for seg in dx.picks(base, segs)]
    data, source = skn_rows([seg.removesuffix(".wav") for _, seg in picked], parquet)

    totals = {}
    all_refs = [base[s]["ref"] for s in sorted(base)]
    for arm, _ in SKN_ARMS:
        e, n = dx.errors(all_refs, [hyps[arm][s]["hyp"] for s in sorted(base)])
        totals[arm] = e / n * 100
    cards, audio_bytes, mime = [], 0, ""
    for place, seg in picked:
        row = data[seg.removesuffix(".wav")]
        audio, mime = to_mp3(row["audio"]["bytes"])
        audio_bytes += len(audio)
        ref = base[seg]["ref"]
        ref_shown, ref_toks = display_words(ref, SKN_NORM)
        lines = []
        dialect = (row.get("text_dialect") or "").strip()
        if dialect:
            lines.append(
                '<div class="line dia"><div class="who">Murrekirjoitus</div>'
                f'<div class="txt">{esc(dialect)}</div><div class="err"></div></div>'
            )
        lines.append(
            f'<div class="line gt"><div class="who">Ihminen (yleiskieli)</div><div class="txt">{esc(ref)}</div>'
            f'<div class="err">{len(ref_toks)} sanaa</div></div>'
        )
        for arm, label in SKN_ARMS:
            hyp = hyps[arm][seg]["hyp"].strip()
            shown, toks = display_words(hyp, SKN_NORM)
            rows = align(ref_toks, toks)
            e, words = dx.errors([ref], [hyp])
            assert (e, words) == (n_errors(rows), len(ref_toks)), seg
            lines.append(
                f'<div class="line"><div class="who">{esc(label)}</div><div class="txt">{render_hyp(rows, shown)}</div>'
                f'<div class="err{" zero" if e == 0 else ""}">{e} / {words}</div></div>'
            )
        cards.append(
            f'<article class="row"><div class="rowhead"><b>{esc(row["municipality"] or place)}</b>'
            f'<code>{esc(seg.removesuffix(".wav"))}</code><span class="time">{row["seconds"]:.1f} s</span></div>'
            f'<audio controls preload="none" src="{data_uri(audio, mime)}" '
            f'aria-label="{esc(row["municipality"] or place)}, {esc(seg.removesuffix(".wav"))}"></audio>'
            f"{''.join(lines)}</article>"
        )
    packed = ", pakattu 16 kHz mono-MP3:ksi" if mime == "audio/mpeg" else ""
    body = f"""
<section id="b">
<h2>B. Murretesti (SKN)</h2>
<p>SKN-murretestin 20 segmenttiä, samat kuin <code>docs/DIALECT-EXAMPLES.md</code>:ssä: kunnittain niistä
segmenteistä, joissa on 12–40 referenssisanaa, id-järjestyksessä kolmanneksen ja kahden kolmanneksen kohdalla
olevat (valittu sijainnin, ei tuloksen mukaan). Testikunnat eivät ole koulutusdatassa. Vertailukohta on SKN:n
ihmisen tekemä yleiskielinen litteraatti; numero oikealla: sanavirheet / referenssin sanat (pienet kirjaimet,
välimerkit pois). Koko testissä ({len(base)} segmenttiä) WER on Whisper large-v3:lla
{totals[dx.BASE]:.1f} % ja omalla mallilla {totals[dx.OURS]:.1f} %.</p>
{"".join(cards)}
<p class="small">Ääni ja teksti: {esc(SKN_CREDIT)}. Ääni datasetista
<code>{esc(HF_DATA_REPO)}</code> (split <code>test</code>){packed}.</p>
</section>"""
    meta = {
        "source": source,
        "segments": [seg for _, seg in picked],
        "audio_mime": mime,
        "audio_bytes": audio_bytes,
        "dialect_text_present": sum(
            bool((data[s.removesuffix(".wav")].get("text_dialect") or "").strip()) for _, s in picked
        ),
    }
    return body, meta


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

CSS = """
:root{color-scheme:light dark;--bg:#fafaf9;--card:#fff;--fg:#1c1c1e;--muted:#62636b;--line:#e2e2df;
--accent:#2453c2;--sub-bg:#fbe3b8;--sub-fg:#5c3600;--ins-bg:#d9e5ff;--ins-fg:#173a8a;--del:#b3261e;
--flag:#b06a00;--now:#fff4cc;--gt:#f2f2ef}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--card:#1c1c20;--fg:#e9e9ec;--muted:#a3a4ad;
--line:#303036;--accent:#93b4ff;--sub-bg:#5b3d0c;--sub-fg:#ffe1b0;--ins-bg:#20335f;--ins-fg:#d3e0ff;
--del:#ff8f86;--flag:#f2b45c;--now:#2e2a18;--gt:#232328}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:62rem;margin:0 auto;padding:1rem 1rem 4rem}
h1{font-size:1.6rem;line-height:1.25;margin:.5rem 0}
h2{font-size:1.3rem;margin:2.5rem 0 .5rem;padding-top:.5rem;border-top:1px solid var(--line)}
p{margin:.5rem 0}
a{color:var(--accent)}
code{font-size:.85em;overflow-wrap:anywhere}
.small,small{font-size:.85rem;color:var(--muted)}
.muted{color:var(--muted)}
.note{border-left:3px solid var(--flag);padding:.25rem .75rem}
nav{display:flex;flex-wrap:wrap;gap:.25rem 1rem}
.legend{display:flex;flex-wrap:wrap;gap:.25rem 1rem;font-size:.9rem;margin:.75rem 0}
.scroll{overflow-x:auto;margin:.75rem 0}
table{border-collapse:collapse;font-size:.92rem;min-width:100%}
th,td{border-bottom:1px solid var(--line);padding:.4rem .6rem;text-align:right;vertical-align:top}
th:first-child,td:first-child{text-align:left}
thead th{font-weight:600;vertical-align:bottom}
.sub{background:var(--sub-bg);color:var(--sub-fg);border-radius:3px;padding:0 2px}
.ins{background:var(--ins-bg);color:var(--ins-fg);border-radius:3px;padding:0 2px}
del{color:var(--del);font-size:.85em;opacity:.85}
mark.flag{background:none;color:inherit;text-decoration:underline dashed var(--flag) 2px;text-underline-offset:3px}
mark.gap{color:var(--flag);font-weight:700;text-decoration:none}
.player{position:sticky;top:0;z-index:2;background:var(--bg);padding:.5rem 0;border-bottom:1px solid var(--line)}
.player audio{width:100%}
.toggles{display:flex;flex-wrap:wrap;gap:.25rem 1rem;font-size:.9rem}
.row{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:.6rem .8rem;margin:.75rem 0}
.row.now{background:var(--now)}
.rowhead{display:flex;flex-wrap:wrap;align-items:center;gap:.5rem;margin-bottom:.35rem;font-size:.9rem}
.rowhead .n{font-weight:600;color:var(--muted)}
.row audio{width:100%;height:2.5rem;margin:.25rem 0}
.time{color:var(--muted);font-variant-numeric:tabular-nums}
button.play{font:inherit;border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:6px;
padding:.1rem .6rem;cursor:pointer}
button.play[aria-pressed=true]{background:var(--accent);color:var(--bg);border-color:var(--accent)}
.line{display:grid;grid-template-columns:11rem 1fr 4.5rem;gap:.75rem;padding:.3rem 0;border-top:1px solid var(--line)}
.line:first-of-type{border-top:0}
.line.gt,.line.dia{background:var(--gt);border-radius:4px;padding:.3rem .4rem;margin:0 -.4rem}
.who{font-size:.85rem;color:var(--muted);font-weight:600}
.err{text-align:right;font-size:.85rem;color:var(--muted);font-variant-numeric:tabular-nums}
.err.zero{opacity:.6}
.doubts{font-size:.8rem;color:var(--muted);margin:.15rem 0 .25rem}
.judge{font-size:.85rem;color:var(--muted);margin-top:.2rem}
.badge{display:inline-block;min-width:1.4rem;text-align:center;border:1px solid var(--line);border-radius:4px;
color:var(--fg);font-weight:700}
details{margin:.75rem 0}
summary{cursor:pointer}
pre{white-space:pre-wrap;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:.75rem;
font-size:.85rem}
.hide0 .m0,.hide1 .m1,.hide2 .m2,.hide3 .m3{display:none}
@media (max-width:640px){.line{grid-template-columns:1fr auto;gap:0 .5rem}.line .who{grid-column:1}
.line .err{grid-column:2;grid-row:1}.line .txt{grid-column:1/-1}}
"""

JS = """
(() => {
  const clip = document.getElementById('clip');
  const rows = [...document.querySelectorAll('#rows .row')];
  let stopAt = null, active = null;
  const setActive = b => {
    if (active) active.setAttribute('aria-pressed', 'false');
    active = b;
    if (b) b.setAttribute('aria-pressed', 'true');
  };
  document.querySelectorAll('button.play').forEach(b => {
    b.setAttribute('aria-pressed', 'false');
    b.addEventListener('click', () => {
      if (active === b && !clip.paused) { clip.pause(); return; }
      stopAt = parseFloat(b.dataset.end);
      clip.currentTime = parseFloat(b.dataset.start);
      clip.play();
      setActive(b);
    });
  });
  const tick = () => {
    const t = clip.currentTime;
    if (stopAt !== null && t >= stopAt) clip.pause();
    rows.forEach(r => r.classList.toggle('now', !clip.paused && t >= +r.dataset.start && t < +r.dataset.end));
    if (!clip.paused) requestAnimationFrame(tick);
  };
  clip.addEventListener('play', () => requestAnimationFrame(tick));
  clip.addEventListener('pause', () => { stopAt = null; setActive(null); tick(); });
  const box = document.getElementById('a');
  document.querySelectorAll('.toggles input').forEach(c =>
    c.addEventListener('change', () => box.classList.toggle('hide' + c.dataset.arm, !c.checked)));
  // one player at a time across the page
  document.addEventListener('play', e => document.querySelectorAll('audio').forEach(a => {
    if (a !== e.target) a.pause();
  }), true);
})();
"""


def page(sections: list[str]) -> str:
    return f"""<!doctype html>
<html lang="fi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Puheentunnistus rinnakkain</title>
<link rel="icon" href="data:,">
<style>{CSS}</style>
</head>
<body>
<main>
<h1>Suomen puheentunnistus: mallit rinnakkain</h1>
<p>Kuuntele ääni ja vertaa, mitä mallit kirjoittivat. Repo: <code>GAIK-project/finnish-asr</code>,
sivun tekee <code>scripts/build_compare.py</code>. Sivu toimii ilman verkkoyhteyttä. Webinaarileike on
julkaistu GAIK-projektin gaik-toolkit-repossa; jaa leike sieltä, älä tätä sivua.</p>
<nav><a href="#a">A. Webinaarileike</a><a href="#b">B. Murretesti (SKN)</a></nav>
<div class="legend"><span><span class="sub">korvattu</span> eri sana kuin referenssissä</span>
<span><span class="ins">lisätty</span> ei referenssissä</span><span><del>puuttuu</del> referenssin sana jäi pois</span>
<span>Isot kirjaimet ja välimerkit eivät ole virheitä.</span></div>
{"".join(sections)}
</main>
<script>{JS}</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", type=Path, help="SKN test parquet already on disk (skips the download).")
    args = parser.parse_args()

    require_clip("Ajokortti.mp3", "reference.txt")
    a_html, a_meta = section_clip()
    b_html, b_meta = section_skn(args.parquet)
    text = page([a_html, b_html])
    OUT.write_text(text, encoding="utf-8", newline="\n")
    size = OUT.stat().st_size
    print(f"wrote {rel(OUT)}: {size / 1e6:.2f} MB")

    inputs = [AUDIO_PATH, GT_FI_PATH, GT_ALIGNMENT_PATH, SCORES_PATH]
    for arm, _, _ in CLIP_ARMS:  # the transcript file, this repository's or QAD-1's
        own = ASR_DIR / f"{arm}.json"
        inputs.append(own if own.exists() else QAD1_ASR_DIR / f"{arm}.json")
    inputs += [JUDGE_DIR / j / f"{arm}.json" for arm, _, _ in CLIP_ARMS for j, _ in JUDGES]
    inputs += [dx.BENCH / f"{arm}.jsonl" for arm, _ in SKN_ARMS]
    meta = build_run_metadata(
        app_under_test="Finnish ASR model comparison (listening page: Ajokortti clip, SKN)",
        dataset_path=GT_ALIGNMENT_PATH,
        runner_path=Path(__file__),
        model_or_deployment=[arm for arm, _, _ in CLIP_ARMS],
        env_names=[],
        extra={
            "output": rel(OUT),
            "output_bytes": size,
            "inputs_sha256": {rel(p): sha256(p) for p in inputs},
            "clip": a_meta,
            "skn": b_meta,
        },
    )
    write_run_metadata(RESULTS_DIR, meta, name=META.name)
    print(f"wrote {rel(META)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
