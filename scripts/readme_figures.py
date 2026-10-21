r"""The README's explainer figures, drawn as static SVG with text as text.

    uv run python scripts/readme_figures.py

Writes to docs/img/:

- example.svg: one real SKN test segment, base Whisper against our model (as in the story);
- punctuation.svg: what Opus did to the training text, on one real SKN segment;
- data-sources.svg: where the audio and the texts come from, with the sources' logos;
- pipeline.svg: how the dialect model was made, six steps; the two language-model steps in brown;
- dialect-ranking.svg: every open model run on the SKN dialect test, from results/benchmarks/skn_fi;
- data-split.svg: SKN's 50 municipalities split into training, validation and test, and LA-murre
  as a test-only corpus;
- two-measures.svg: word error rate against the blind LLM judge, on one constructed example;
- clip-ranking.svg: the ten best set-ups on the Ajokortti clip and our published model, by the mean
  of the two judges' meaning-error points, from results/scores.json and results/judgements/;
- which-version.svg: when to use the main model and when its timestamps revision.

The numbers in the explainer figures are this README's own; the two rankings read the results.
Icons are Lucide (ISC), from docs/story/src/assets/icons; logos are their owners' trademarks, from
svgl and Iconify (docs/img/logos, see NOTICE.md), used only to name the data sources and tools. No keys,
no network.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fi_common import PROJECT_ROOT, SCORES_PATH  # noqa: E402
from make_readme_table import JUDGES, ROWS, judge  # noqa: E402

IMG = PROJECT_ROOT / "docs" / "img"
ICONS = PROJECT_ROOT / "docs" / "story" / "src" / "assets" / "icons"
LOGOS = IMG / "logos"
FONT = "system-ui, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"
INK, MUTED, LINE, BG = "#1f2328", "#59636e", "#d8dee4", "#ffffff"
OURS, OURS_SOFT = "#2f6fb3", "#e8f0fa"
LLM, LLM_SOFT = "#8a5a2b", "#f6eee4"
OPEN, API, BASE = "#3f8f5b", "#9aa4ae", "#9aa4ae"
WARN = "#b3402e"


def svg(w: int, h: int, label: str, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'font-family="{FONT}" role="img" aria-label="{escape(label)}">\n'
        f'<rect width="{w}" height="{h}" rx="12" fill="{BG}"/>\n' + "\n".join(body) + "\n</svg>\n"
    )


def text(x: float, y: float, s: str, size: float = 14, weight: int = 400, fill: str = INK, anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{escape(s)}</text>'
    )


def icon(name: str, x: float, y: float, size: float, color: str) -> str:
    """A Lucide icon's shapes, scaled from its 24 px grid, stroked in one colour."""
    src = (ICONS / f"{name}.svg").read_text(encoding="utf-8")
    inner = re.sub(r"^.*?<svg[^>]*>|</svg>\s*$", "", src.strip(), flags=re.S)
    inner = re.sub(r"</?g[^>]*>", "", inner)
    k = size / 24
    return (
        f'<g transform="translate({x:.1f} {y:.1f}) scale({k:.3f})" fill="none" stroke="{color}" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{inner}</g>'
    )


def arrow(x1: float, y1: float, x2: float, y2: float, color: str = MUTED) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="2" '
        f'marker-end="url(#arrow)"/>'
    )


ARROW_DEF = (
    f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
    f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{MUTED}"/></marker></defs>'
)


def logo(name: str, x: float, y: float, h: float, uid: str = "") -> tuple[str, float]:
    """A logo from docs/img/logos as a nested <svg>, h px tall; returns (markup, width).

    Element ids get the logo's name and uid as a prefix, so two logos (or the same one twice) in one
    figure cannot share a gradient id."""
    src = (LOGOS / f"{name}.svg").read_text(encoding="utf-8")
    src = re.sub(r"<!--.*?-->|<title>.*?</title>|<desc>.*?</desc>", "", src, flags=re.S)
    root = re.search(r"<svg\b[^>]*>", src).group(0)
    vb = re.search(r'viewBox="([^"]+)"', root).group(1).split()
    vw, vh = float(vb[2]), float(vb[3])
    inner = src[src.index(root) + len(root) : src.rindex("</svg>")]
    p = f"{name}{uid}-"
    inner = re.sub(r'id="([^"]+)"', lambda m: f'id="{p}{m.group(1)}"', inner)
    inner = re.sub(r"url\(#([^)]+)\)", lambda m: f"url(#{p}{m.group(1)})", inner)
    inner = re.sub(r'href="#([^"]+)"', lambda m: f'href="#{p}{m.group(1)}"', inner)
    w = h * vw / vh
    return (
        f'<svg x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" viewBox="{" ".join(vb)}" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink">{inner}</svg>',
        w,
    )


def words(x: float, y: float, parts: list[tuple[str, bool]], size: float = 16, bad: str = WARN) -> str:
    """One line of text in which the flagged parts are bold and red."""
    spans = "".join(
        f'<tspan font-weight="700" fill="{bad}">{escape(t)}</tspan>' if flag else escape(t) for t, flag in parts
    )
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{INK}">{spans}</text>'


def pipeline() -> str:
    # (icon, or "logo:<name>"), title, colours, at most three short lines
    steps = [
        ("audio-lines", "Dialect speech", OURS, OURS_SOFT,
         ["62 h of old interviews from 38", "municipalities (SKN), with read", "and parliament speech added."]),
        ("logo:claude", "Opus adds punctuation", LLM, LLM_SOFT,
         ["Full stops and capitals for the", "training text. Not one word may", "change."]),
        ("cpu", "Training on LUMI", OURS, OURS_SOFT,
         ["Whisper large-v3 learns from it", "(LoRA). One AMD GPU on CSC's", "LUMI, 3 h 8 min."]),
        ("chart-column", "Test: word errors", OURS, OURS_SOFT,
         ["600 segments from 10 municipalities", "the model never heard, plus the", "standard-Finnish sets."]),
        ("logo:claude", "Test: blind LLM judge", LLM, LLM_SOFT,
         ["A second dialect corpus. Opus counts", "only errors that change the", "meaning; 3 open models agree."]),
        ("logo:hugging-face", "Published", OURS, OURS_SOFT,
         ["Model and training data on", "Hugging Face, free to use", "(Apache 2.0, CC BY 4.0)."]),
    ]
    W, BW, BH, GAP, X0, Y1, Y2 = 1000, 300, 150, 40, 20, 64, 278
    body = [ARROW_DEF, text(X0, 38, "How we made it", 19, 700)]
    body.append(text(W - X0, 38, "brown = a step where a language model (Claude Opus 5.5) helped", 13, 400, LLM, "end"))
    for i, (ic, title, color, soft, lines) in enumerate(steps):
        col, row = i % 3, i // 3
        x, y = X0 + col * (BW + GAP), (Y1, Y2)[row]
        body.append(f'<rect x="{x}" y="{y}" width="{BW}" height="{BH}" rx="10" fill="{soft}" stroke="{color}" stroke-width="1.5"/>')
        body.append(f'<circle cx="{x + 26}" cy="{y + 28}" r="15" fill="{color}"/>')
        body.append(text(x + 26, y + 33.5, str(i + 1), 15, 700, "#ffffff", "middle"))
        if ic.startswith("logo:"):
            body.append(logo(ic[5:], x + 50, y + 16, 24, f"p{i}")[0])
        else:
            body.append(icon(ic, x + 50, y + 16, 24, color))
        body.append(text(x + 84, y + 34, title, 17, 700, INK))
        for k, ln in enumerate(lines):
            body.append(text(x + 18, y + 74 + k * 23, ln, 15, 400, INK))
        if col < 2:
            body.append(arrow(x + BW + 4, y + BH / 2, x + BW + GAP - 4, y + BH / 2))
    # elbow from step 3 down to step 4
    x3, x4 = X0 + 2 * (BW + GAP) + BW / 2, X0 + BW / 2
    ymid = (Y1 + BH + Y2) / 2
    body.append(
        f'<path d="M{x3:.1f} {Y1 + BH + 4} V{ymid:.1f} H{x4:.1f} V{Y2 - 5}" fill="none" stroke="{MUTED}" '
        f'stroke-width="2" marker-end="url(#arrow)"/>'
    )
    return svg(W, Y2 + BH + 22, "How the dialect model was made, in six steps", body)


def punctuation() -> str:
    """SKN03a_Palkane_0021, a validation segment, as in the story's data section."""
    W, H = 1000, 372
    body = [ARROW_DEF, text(24, 38, "What the language model did to the training text: punctuation, nothing else", 19, 700)]
    body.append(f'<rect x="24" y="58" width="{W - 48}" height="70" rx="10" fill="#f6f8fa" stroke="{LINE}"/>')
    body.append(text(42, 82, "Before: a person's transcript from SKN (Pälkäne, a validation segment)", 14, 700, MUTED))
    body.append(text(42, 112, "vieläkö osaisitte kyllä minä niin kauan osaan kun minä elän", 17, 400))
    mark, w = logo("claude", 42, 146, 26, "pu")
    body.append(mark)
    body.append(text(42 + w + 10, 165, "Claude Opus 5.5 adds full stops, question marks, commas and capitals", 15, 700, LLM))
    body.append(arrow(W - 60, 134, W - 60, 190))
    body.append(f'<rect x="24" y="196" width="{W - 48}" height="70" rx="10" fill="{LLM_SOFT}" stroke="{LLM}" stroke-width="1.5"/>')
    body.append(text(42, 220, "After: the target text the model learns from", 14, 700, LLM))
    body.append(words(42, 250, [("Vieläkö osaisitte", False), ("?", True), (" ", False), ("K", True),
                                ("yllä minä niin kauan osaan kun minä elän", False), (".", True)], 17, LLM))
    body.append(icon("shield-check", 24, 284, 24, OPEN))
    body.append(text(58, 302, "The check: the words must be exactly the person's. 15 024 of 15 042 passed; the other 18", 14, 400))
    body.append(text(58, 322, "stayed as they were. Opus never heard audio and never wrote a test reference.", 14, 400))
    body.append(text(24, 356, "Why: SKN's text has no punctuation, so without this the model learned to write without full stops.", 13, 400, MUTED))
    return svg(W, H, "Opus added only punctuation and capitals to the training text; the words had to stay the same", body)


def data_sources() -> str:
    W, H = 1000, 270
    body = [text(24, 38, "Where it comes from", 19, 700)]
    cards = [
        ("logo:openai", "Whisper large-v3", ["The starting model", "OpenAI · Apache 2.0"], "starting point", INK),
        ("database", "SKN", ["Dialect interviews, 1960s–70s", "Kotus, Kielipankki · CC BY 4.0"], "training + test", OURS),
        ("logo:google", "FLEURS", ["Read sentences", "Google · CC BY 4.0"], "training + test", OURS),
        ("logo:meta", "VoxPopuli", ["Parliament speech", "Meta · CC0"], "training + test", OURS),
        ("map-pin", "LA-murre", ["Dialect interviews, 132 places", "Univ. of Turku · CC BY-ND"], "test + variant", INK),
    ]
    CW, gap, y = 180, 9, 60
    for i, (ic, name, lines, role, color) in enumerate(cards):
        x = 24 + i * (CW + gap)
        body.append(f'<rect x="{x}" y="{y}" width="{CW}" height="186" rx="10" fill="{BG}" stroke="{LINE}" stroke-width="1.5"/>')
        if ic.startswith("logo:"):
            w = logo(ic[5:], 0, 0, 30, f"d{i}")[1]
            body.append(logo(ic[5:], x + (CW - w) / 2, y + 18, 30, f"d{i}")[0])
        else:
            body.append(icon(ic, x + CW / 2 - 15, y + 18, 30, color))
        body.append(text(x + CW / 2, y + 76, name, 17, 700, INK, "middle"))
        for k, ln in enumerate(lines):
            body.append(text(x + CW / 2, y + 102 + k * 20, ln, 12.5, 400, MUTED, "middle"))
        pill = OURS if role.startswith("training") else (MUTED if role == "starting point" else INK)
        body.append(f'<rect x="{x + 30}" y="{y + 146}" width="{CW - 60}" height="24" rx="12" fill="{pill}"/>')
        body.append(text(x + CW / 2, y + 162.5, role, 13, 700, "#ffffff", "middle"))
    return svg(W, H, "Sources: Whisper large-v3 by OpenAI, SKN, FLEURS by Google, VoxPopuli by Meta, LA-murre", body)


LEADERBOARD_OURS = {"ft_whisper-large-v3-fi-parl"}  # the front page compares the main model only
DIALECT_SHORT = {  # arm -> (label, group); arms missing from scores.json are left out
    "ft_whisper-large-v3-fi-parl": ("Ours, main model", "ours"),
    "ft_whisper-large-v3-fi-skn-punct-best": ("Ours, first model (v1.0)", "ours"),
    "vertex_gemini-3.5-flash": ("Gemini 3.5 Flash, asked to write as spoken", "API"),
    "vertex_gemini-3.5-flash-std": ("Gemini 3.5 Flash, asked for standard Finnish", "API"),
    "google_chirp-3": ("Google Chirp 3", "API"),
    "elevenlabs_scribe-v2": ("ElevenLabs Scribe v2", "API"),
    "azure_gpt-4o-transcribe": ("OpenAI gpt-4o-transcribe", "API"),
    "hf_whisper-large-v3": ("Whisper large-v3", "open"),
    "hf_whisper-large-v3-turbo": ("Whisper large-v3 turbo", "open"),
    "nemo_canary-1b-v2": ("NVIDIA Canary 1B v2", "open"),
    "nemo_rasmus-canary-fi-v2-kenlm": ("RASMUS Finnish Canary v2 + KenLM", "open"),
    "nemo_rasmus-canary-fi-v2": ("RASMUS Finnish Canary v2", "open"),
    "hf_whisper-large-finnish-v3": ("Finnish-NLP whisper-large-finnish-v3", "open"),
    "qwen_qwen3-asr-1.7b": ("Qwen3-ASR 1.7B", "open"),
}


def dialect_ranking() -> str:
    path = PROJECT_ROOT / "results" / "benchmarks" / "skn_fi" / "scores.json"
    wer = {a["arm_id"]: a["WER"] for a in json.loads(path.read_text(encoding="utf-8"))["arms"]}
    rows = sorted((wer[a], label, group, a) for a, (label, group) in DIALECT_SHORT.items() if a in wer)
    W, ROW, TOP, LX, BX, BW = 1000, 34, 112, 24, 400, 500
    vmax = 70.0
    colors = {"API": API, "open": OPEN, "ours": OURS}
    body = [text(24, 36, "Dialect speech: our model against the others", 19, 700)]
    body.append(text(24, 60, "Word error rate %, 600 segments from 10 municipalities left out of our training. Lower is better.", 14, 400, MUTED))
    lx = 24
    for name, color in (("paid API service", API), ("open model, runs on your own GPU", OPEN), ("this project", OURS)):
        body.append(f'<rect x="{lx}" y="76" width="14" height="14" rx="3" fill="{color}"/>')
        body.append(text(lx + 20, 88, name, 13, 400, INK))
        lx += 34 + 7.2 * len(name)
    bottom = TOP + len(rows) * ROW
    for t in range(0, 71, 10):
        x = BX + BW * t / vmax
        body.append(f'<line x1="{x:.1f}" y1="{TOP - 6}" x2="{x:.1f}" y2="{bottom}" stroke="{LINE}"/>')
        body.append(text(x, bottom + 16, str(t), 11, 400, MUTED, "middle"))
    for i, (v, label, group, arm) in enumerate(rows):
        y = TOP + i * ROW
        if group == "ours":
            body.append(f'<rect x="12" y="{y}" width="{W - 24}" height="{ROW}" rx="6" fill="{OURS_SOFT}"/>')
        body.extend(label_with_logo(arm, label, LX, y + 22, group == "ours", f"d{i}"))
        body.append(f'<rect x="{BX}" y="{y + 7}" width="{BW * v / vmax:.1f}" height="{ROW - 14}" rx="3" fill="{colors[group]}"/>')
        body.append(text(BX + BW * v / vmax + 8, y + 22, f"{v:.1f}", 13, 700, INK))
    if "google_chirp-3" in wer:  # README footnote under the benchmark table
        body.append(text(24, bottom + 40, "Google Chirp 3 repeated “se on” 1 262 times in one 50-word segment; without that segment it scores 35.0 %.", 12.5, 400, MUTED))
        bottom += 26
    best_other = min(r for r in rows if r[2] != "ours")
    alt = f"Dialect test word error rate: our model {rows[0][0]:.1f} %, the best other model {best_other[1]} {best_other[0]:.1f} %"
    return svg(W, bottom + 30, alt, body)


def data_split() -> str:
    W, H = 1000, 360
    body = [text(24, 38, "Split by municipality, so the test is speech the model never heard", 19, 700)]
    cats = [(38, OURS, "#ffffff"), (2, LLM, "#ffffff"), (10, INK, "#ffffff")]
    size, gap, cols, x0, y0 = 30, 6, 10, 24, 64
    k = 0
    for n, color, _ in cats:
        for _ in range(n):
            r, c = divmod(k, cols)
            body.append(f'<rect x="{x0 + c * (size + gap)}" y="{y0 + r * (size + gap)}" width="{size}" height="{size}" rx="5" fill="{color}"/>')
            k += 1
    body.append(text(x0, y0 + 5 * (size + gap) + 16, "SKN: 50 municipalities, about 2 h of interviews each", 13, 400, MUTED))
    lx = x0 + cols * (size + gap) + 34
    legend = [
        (OURS, "Training: 38 municipalities", "15 042 segments, 62 h, with FLEURS and VoxPopuli train added"),
        (LLM, "Validation: 2 municipalities", "Pälkäne and Sotkamo; picks the best of three epochs"),
        (INK, "Test: 10 municipalities", "600 segments; five of SKN's six main dialect groups"),
    ]
    for i, (color, head, sub) in enumerate(legend):
        y = y0 + 6 + i * 58
        body.append(f'<rect x="{lx}" y="{y}" width="18" height="18" rx="4" fill="{color}"/>')
        body.append(text(lx + 30, y + 15, head, 16, 700))
        body.append(text(lx + 30, y + 37, sub, 14, 400, MUTED))
    yb = 270
    body.append(f'<rect x="24" y="{yb}" width="{W - 48}" height="66" rx="10" fill="#f6f8fa" stroke="{LINE}"/>')
    body.append(icon("map-pin", 40, yb + 21, 24, INK))
    body.append(text(76, yb + 28, "A second check: LA-murre, another dialect corpus", 16, 700))
    body.append(text(76, yb + 50, "10 localities not in SKN, 583 segments, kept out of all training. Scored by the blind LLM judge.", 14, 400, MUTED))
    return svg(W, H, "SKN's 50 municipalities split into 38 for training, 2 for validation and 10 for the test; LA-murre as a second test corpus", body)


def two_measures() -> str:
    W, H = 1000, 400
    body = [text(24, 38, "Two ways to score a transcript", 19, 700)]
    body.append(text(W - 24, 38, "illustration, not from the test data", 13, 400, MUTED, "end"))
    ref = "Kyllä minä niin kauan osaan kun minä elän."
    body.append(f'<rect x="24" y="58" width="{W - 48}" height="64" rx="10" fill="#f6f8fa" stroke="{LINE}"/>')
    body.append(text(44, 84, "Reference (what was said)", 13, 700, MUTED))
    body.append(text(44, 108, ref, 18, 400) + text(W - 44, 108, "“Yes, I will know it as long as I live.”", 14, 400, MUTED, "end"))

    rows = [
        ("A", [("Kyllä ", False), ("mä", True), (" niin kauan osaan kun ", False), ("mä", True), (" elän.", False)],
         "2 of 8 words differ: 25 %", "Same meaning, spoken form: 0 points", OPEN),
        ("B", [("Kyllä minä niin kauan osaan kun minä ", False), ("en", True), (".", False)],
         "1 of 8 words differs: 12.5 %", "Meaning lost: a critical error", WARN),
    ]
    body.append(text(560, 156, "Word error rate", 15, 700, OURS))
    body.append(text(780, 156, "Blind LLM judge", 15, 700, LLM))
    for i, (tag, parts, wer, jud, jcolor) in enumerate(rows):
        y = 200 + i * 78
        body.append(f'<circle cx="40" cy="{y - 6}" r="13" fill="{INK}"/>')
        body.append(text(40, y - 1, tag, 14, 700, "#ffffff", "middle"))
        body.append(words(64, y, parts, 18))
        body.append(text(560, y - 4, wer.split(":")[1].strip(), 22, 700, OURS))
        body.append(text(560, y + 18, wer.split(":")[0], 13, 400, MUTED))
        body.append(text(780, y - 4, jud.split(":")[1].strip(), 15, 700, jcolor))
        body.append(text(780, y + 18, jud.split(":")[0], 13, 400, MUTED))
    body.append(text(24, 362, "WER counts every word that differs, so A looks worse. The judge counts only changes in meaning:", 14, 400, INK))
    body.append(text(24, 384, "critical 10, major 5, minor 1 point per 100 reference words. Lower is better for both.", 14, 400, INK))
    return svg(W, H, "Word error rate counts every differing word; the blind LLM judge counts only meaning errors", body)


SHORT = {
    "vertex_gemini-3.5-flash": "Gemini 3.5 Flash",
    "elevenlabs_scribe-v2": "ElevenLabs Scribe v2",
    "google_chirp-3": "Google Chirp 3",
    "hf_whisper-large-v3": "Whisper large-v3",
    "hf_whisper-large-v3-turbo": "Whisper large-v3 turbo",
    "ft_whisper-large-v3-fi-pilot": "Ours, pilot (FLEURS + VoxPopuli)",
    "ft_whisper-large-v3-fi-skn": "Ours, round 1",
    "hf_whisper-large-v3-chunks": "Whisper large-v3, in pieces",
    "ft_whisper-large-v3-fi-skn-chunks": "Ours, round 1, in pieces",
    "ft_whisper-large-v3-fi-skn-punct-best-chunks": "Ours, first model (v1.0), in pieces",
    "ft_whisper-large-v3-fi-parl-chunks": "Ours, main model, in pieces",
    "ft_whisper-large-v3-fi-skn-lamurre-best-chunks": "Ours, + LA-murre, in pieces",
    "openai_whisper-1": "OpenAI whisper-1",
    "openai_gpt-4o-mini-transcribe": "OpenAI gpt-4o-mini-transcribe",
    "openai_gpt-transcribe": "OpenAI gpt-transcribe",
    "hh_whisperx_fi": "WhisperX + Finnish-NLP (Haaga-Helia server)",
    "hf_whisper-large-finnish-v3": "Finnish-NLP whisper-large-finnish-v3",
    "openai_gpt-4o-transcribe": "OpenAI gpt-4o-transcribe",
    "hf_orukeet": "Orukeet (Parakeet v3 fine-tune)",
    "nemo_rasmus-canary-fi-v2": "RASMUS Finnish Canary v2",
    "nemo_rasmus-canary-fi-v2-kenlm": "RASMUS Finnish Canary v2 + KenLM",
    "vertex_gemini-3.1-pro": "Gemini 3.1 Pro",
    "nemo_canary-1b-v2": "NVIDIA Canary 1B v2",
    "hf_parakeet-tdt-0.6b-v3": "NVIDIA Parakeet TDT 0.6B v3",
    "hf_wav2vec2-xlsr-1b-finnish-lm-v2": "Finnish-NLP wav2vec2 XLS-R 1B",
    "qwen_qwen3-asr-1.7b": "Qwen3-ASR 1.7B",
    "qwen_qwen3-asr-0.6b": "Qwen3-ASR 0.6B",
}


def clip_ranking() -> str:
    scores = {a["arm_id"]: a for a in json.loads(SCORES_PATH.read_text(encoding="utf-8"))["arms"]}
    rows = []
    for arm_id, (_, kind) in ROWS.items():
        pts = sum(judge(j, arm_id)["error_points_per_100w"] for j in JUDGES) / len(JUDGES)
        group = "ours" if arm_id.startswith("ft_") else kind
        rows.append((pts, SHORT[arm_id], group, scores[arm_id]["metrics"]["WER"], arm_id))
    rows.sort(key=lambda r: r[0])
    ranked = list(enumerate(rows, 1))
    ours = "ft_whisper-large-v3-fi-parl-chunks"
    shown = ranked[:10] + [None] + [r for r in ranked[10:] if r[1][4] == ours]
    W, ROW, TOP, LX, BX, BW = 1000, 28, 108, 360, 372, 470
    vmax = 10.0
    colors = {"API": API, "open": OPEN, "ours": OURS}
    body = [text(24, 36, f"Standard spoken Finnish: the 10 best of {len(rows)} set-ups on a webinar clip, and ours", 19, 700)]
    body.append(text(24, 60, "Meaning-error points per 100 words, mean of two blind judges (Claude, Gemini). Lower is better.", 14, 400, MUTED))
    lx = 24
    for name, color in (("paid API service", API), ("open model, runs on your own GPU", OPEN), ("this project", OURS)):
        body.append(f'<rect x="{lx}" y="76" width="14" height="14" rx="3" fill="{color}"/>')
        body.append(text(lx + 20, 88, name, 13, 400, INK))
        lx += 34 + 7.2 * len(name)
    body.append(text(W - 24, 88, "WER %", 13, 700, MUTED, "end"))
    bottom = TOP + len(shown) * ROW
    for t in range(0, 11, 2):
        x = BX + BW * t / vmax
        body.append(f'<line x1="{x:.1f}" y1="{TOP - 6}" x2="{x:.1f}" y2="{bottom}" stroke="{LINE}"/>')
        body.append(text(x, bottom + 16, str(t), 11, 400, MUTED, "middle"))
    for i, item in enumerate(shown):
        y = TOP + i * ROW
        if item is None:
            body.append(text(LX, y + 17, f"… {ranked[-1][0] - 11} more set-ups in the full table", 13, 400, MUTED, "end"))
            continue
        rank, (pts, label, group, wer, _) = item
        weight = 700 if group == "ours" else 400
        body.append(text(LX, y + 17, f"{rank}. {label}", 14, weight, INK, "end"))
        body.append(f'<rect x="{BX}" y="{y + 4}" width="{max(BW * pts / vmax, 2):.1f}" height="{ROW - 9}" rx="3" fill="{colors[group]}"/>')
        body.append(text(BX + BW * pts / vmax + 6, y + 17, f"{pts:.1f}", 13, 700, INK))
        body.append(text(W - 24, y + 17, f"{wer:.1f}", 13, 400, MUTED, "end"))
    return svg(W, bottom + 30, "The ten best speech-to-text set-ups on the Ajokortti webinar clip, and our main model", body)


def which_version() -> str:
    W, H = 1000, 330
    body = [text(24, 38, "Which version of the model?", 19, 700)]
    body.append(text(24, 62, "Do you need sentence timestamps, or long audio transcribed without cutting it yourself?", 15, 400, INK))
    cards = [
        ("No: main (tag v1.0), the default", OURS, OURS_SOFT, 'revision="main"',
         ["Cut the audio into 15–25 s pieces (code in the model card).",
          "The better one on the webinar and on VoxPopuli.",
          "Its own timestamps are not usable."],
         "SKN dialects 21.1 % · FLEURS 7.6 % · VoxPopuli 10.8 %"),
        ("Yes: the timestamps revision", LLM, LLM_SOFT, 'revision="timestamps"',
         ["Sentence start times: 58 % within 1 s of SKN's.",
          "Long audio in one go, always with temperature fallback.",
          "Same dialect accuracy, slightly worse on VoxPopuli."],
         "SKN dialects 21.4 % · FLEURS 7.4 % · VoxPopuli 11.2 %"),
    ]
    CW, gap, y = 466, 20, 84
    for i, (head, color, soft, code, lines, nums) in enumerate(cards):
        x = 24 + i * (CW + gap)
        body.append(f'<rect x="{x}" y="{y}" width="{CW}" height="226" rx="10" fill="{soft}" stroke="{color}" stroke-width="1.5"/>')
        body.append(icon("package" if i == 0 else "audio-lines", x + 18, y + 18, 24, color))
        body.append(text(x + 52, y + 37, head, 17, 700, INK))
        body.append(
            f'<text x="{x + 18}" y="{y + 70}" font-size="14" font-family="ui-monospace, Consolas, monospace" fill="{color}">'
            f"{escape(code)}</text>"
        )
        for k, ln in enumerate(lines):
            body.append(text(x + 18, y + 102 + k * 24, "• " + ln, 14, 400, INK))
        body.append(f'<line x1="{x + 18}" y1="{y + 180}" x2="{x + CW - 18}" y2="{y + 180}" stroke="{LINE}"/>')
        body.append(text(x + 18, y + 206, nums, 14, 700, color))
    return svg(W, H, "Use the main model by default; use the timestamps revision for sentence times or long audio", body)


R1_SOFT, DEEP = "#8fb6e0", "#1d3f6e"


def card(x: float, y: float, w: float, h: float, head: str, color: str, soft: str) -> list[str]:
    """A rounded card with a coloured header strip; returns its shapes."""
    return [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{soft}" stroke="{color}" stroke-width="1.5"/>',
        f'<path d="M{x} {y + 34} V{y + 10} a10 10 0 0 1 10 -10 H{x + w - 10} a10 10 0 0 1 10 10 V{y + 34} Z" fill="{color}"/>',
        text(x + 14, y + 23, head, 14, 700, "#ffffff"),
    ]


def mark(name: str, x: float, y: float, h: float, uid: str, color: str = INK) -> tuple[str, float]:
    """A logo when docs/img/logos has it, else a Lucide icon of the same name, else nothing."""
    if (LOGOS / f"{name}.svg").exists():
        return logo(name, x, y, h, uid)
    if (ICONS / f"{name}.svg").exists():
        return icon(name, x, y, h, color), h
    return "", 0.0


def journey() -> str:
    """The rounds of training, from the base model to the two published versions (numbers: README)."""
    W, H = 1000, 640
    body = [ARROW_DEF, text(24, 38, "How the model got better, round by round", 19, 700)]
    body.append(text(W - 24, 38, "dialect test = words wrong on 600 segments from 10 unheard municipalities", 13, 400, MUTED, "end"))
    top = [
        ("Start", "openai", BASE, "#f3f4f6", ["Whisper large-v3 by OpenAI,", "as it is"], None, "40.4 %", "dialect test", ""),
        ("Pilot", "audio-lines", BASE, "#f3f4f6", ["+ read sentences (FLEURS)", "+ parliament speech", "   (VoxPopuli)"], "51 min",
         "–", "not run on the dialect test", "Read speech is not enough"),
        ("Round 1", "map-pin", R1_SOFT, OURS_SOFT, ["+ 62 h of dialect", "   interviews (SKN)"], "2 h 2 min", "22.2 %", "dialect test",
         "Almost halved; no full stops"),
        ("Round 2", "logo:claude", OURS, OURS_SOFT, ["+ Opus adds punctuation", "+ bigger LoRA, best of", "   3 epochs"], "3 h 8 min",
         "21.1 %", "dialect test", "Published as v1.0"),
    ]
    CW, gap, y0, CH = 220, 30, 62, 262
    xs = [24 + i * (CW + gap) for i in range(4)]
    for i, (head, ic, color, soft, lines, lumi, big, unit, note) in enumerate(top):
        x = xs[i]
        body.extend(card(x, y0, CW, CH, head, color if color != R1_SOFT else OURS, soft))
        if ic.startswith("logo:"):
            m, _ = mark(ic[5:], x + CW - 38, y0 + 6, 22, f"j{i}")
            body.append(f'<rect x="{x + CW - 42}" y="{y0 + 4}" width="30" height="26" rx="6" fill="#ffffff"/>' + m)
        for k, ln in enumerate(lines):
            body.append(text(x + 14, y0 + 60 + k * 20, ln, 14, 400, INK))
        if lumi:
            body.append(icon("cpu", x + 14, y0 + 126, 16, MUTED))
            body.append(text(x + 36, y0 + 139, f"LUMI, {lumi}", 13, 400, MUTED))
        body.append(text(x + 14, y0 + 188, big, 30, 700, OURS if i >= 2 else INK))
        body.append(text(x + 14, y0 + 208, unit, 13, 400, MUTED))
        if note:
            body.append(f'<line x1="{x + 14}" y1="{y0 + 222}" x2="{x + CW - 14}" y2="{y0 + 222}" stroke="{LINE}"/>')
            body.append(text(x + 14, y0 + 245, note, 12.5, 700, INK))
        if i < 3:
            body.append(arrow(x + CW + 3, y0 + CH / 2, x + CW + gap - 3, y0 + CH / 2))
    # two branches from round 2
    BW, by, BH = 340, 400, 214
    bx = [W / 2 - BW - 12, W / 2 + 12]
    branches = [
        ("LA-murre version", DEEP, "#e9eef6", ["+ 56 h more dialect interviews (LA-murre);", "Opus rewrote their dialect spelling", "into standard Finnish"],
         "4 h 57 min", "19.7 %", "the best on dialects:", "published as the la-murre branch"),
        ("Timestamps version", LLM, LLM_SOFT, ["+ Whisper's timestamp tokens on half", "of the SKN training segments"],
         "3 h 6 min", "21.4 %", "and sentence times that work:", "published as the timestamps branch"),
    ]
    cx3 = xs[3] + CW / 2
    for j, (head, color, soft, lines, lumi, big, n1, n2) in enumerate(branches):
        x = bx[j]
        body.extend(card(x, by, BW, BH, head, color, soft))
        for k, ln in enumerate(lines):
            body.append(text(x + 14, by + 60 + k * 20, ln, 14, 400, INK))
        body.append(icon("cpu", x + 14, by + 112 if len(lines) == 3 else by + 92, 16, MUTED))
        body.append(text(x + 36, (by + 125) if len(lines) == 3 else by + 105, f"LUMI, {lumi}", 13, 400, MUTED))
        body.append(text(x + 14, by + 170, big, 30, 700, color))
        body.append(text(x + 128, by + 154, n1, 13, 700, INK))
        for k, ln in enumerate(textwrap.wrap(n2, 32)):
            body.append(text(x + 128, by + 172 + k * 18, ln, 12.5, 400, MUTED))
        tx = x + BW / 2
        body.append(
            f'<path d="M{cx3:.1f} {y0 + CH + 3} V{(y0 + CH + by) / 2:.1f} H{tx:.1f} V{by - 5}" fill="none" stroke="{MUTED}" '
            f'stroke-width="2" marker-end="url(#arrow)"/>'
        )
    body.append(text(24, H - 14, "Same base model and method in every round; the version to keep was picked by validation loss, never by test scores.", 13, 400, MUTED))
    return svg(W, H, "Training rounds: pilot, round 1 with dialect interviews, round 2 published, a version with LA-murre and a timestamps version", body)


DIALECT_JUDGES = [  # the judges of agreement-r4.json that are drawn, in this order
    ("claude-opus-5-5", "Claude Opus 5.5", "claude", "the main judge"),
    ("azure-gpt-6-1-sol", "GPT-6.1-sol", "openai", "via Azure"),
    ("aitta-gpt-oss-120b", "gpt-oss-120b", "openai", "on CSC Aitta"),
    ("aitta-gemma-4-31b-it", "Gemma 4 31B", "google", "on CSC Aitta"),
    ("aitta-qwen3-6-27b", "Qwen3.6 27B", "qwen", "on CSC Aitta"),
]
JUDGED = [  # the fifth round: six transcripts of each LA-murre segment judged together
    ("azure_gpt-4o-transcribe", "gpt-4o-transcribe", "#6b7280"),
    ("vertex_gemini-3.5-flash-std", "Gemini 3.5 Flash", LLM),
    ("hf_whisper-large-v3", "Whisper large-v3", BASE),
    ("ft_whisper-large-v3-fi-skn-punct-best", "Ours, first model (v1.0)", R1_SOFT),
    ("ft_whisper-large-v3-fi-skn-lamurre-best", "Ours, + LA-murre", DEEP),
    ("ft_whisper-large-v3-fi-parl", "Ours, main model", OURS),
]


def judges() -> str:
    """Three blind judges on LA-murre, from results/judgements_dialect/lamurre_fi/agreement-r5.json."""
    path = PROJECT_ROOT / "results" / "judgements_dialect" / "lamurre_fi" / "agreement-r5.json"
    data = json.loads(path.read_text(encoding="utf-8"))["judges"]
    W, H = 1000, 452
    body = [text(24, 38, "Dialect speech, several judges: our models make far fewer meaning errors", 19, 700)]
    body.append(text(24, 62, "Meaning-error points per 100 words on the second dialect corpus (LA-murre), lower is better. Each judge is", 14, 400, MUTED))
    body.append(text(24, 82, "stricter or milder, so compare bars inside one panel. Every judge puts the API services and base Whisper far behind ours.", 14, 400, MUTED))
    drawn = [j for j in DIALECT_JUDGES if j[0] in data]
    gap, py, ph = 14, 100, 300
    PW = (W - 48 - gap * (len(drawn) - 1)) / len(drawn)
    vmax = 40.0
    for i, (jid, name, lg, where) in enumerate(drawn):
        x = 24 + i * (PW + gap)
        d = data[jid]
        body.append(f'<rect x="{x:.1f}" y="{py}" width="{PW:.1f}" height="{ph}" rx="10" fill="#f6f8fa" stroke="{LINE}"/>')
        m, w = mark(lg, x + 12, py + 12, 22, f"g{i}")
        body.append(m)
        body.append(text(x + 12 + (w + 8 if w else 0), py + 29, name, 15, 700, INK))
        sub = f"{d['segments']} segments, {where}"
        body.append(text(x + 12, py + 50, sub, 12, 400, MUTED))
        if "against_reference" in d:
            pct = round(d["against_reference"]["pairwise_same_preference"] * 100)
            body.append(text(x + 12, py + 68, f"agrees with Claude in {pct} % of pairs", 12, 700, OPEN))
        else:
            body.append(text(x + 12, py + 68, "the reference for the others", 12, 700, MUTED))
        base_y, top_y = py + ph - 28, py + 96
        bw, bgap = 26, 7
        for k, (arm, _, color) in enumerate(JUDGED):
            v = d["points_per_100w"][arm]
            hgt = (base_y - top_y) * v / vmax
            bx = x + (PW - len(JUDGED) * bw - (len(JUDGED) - 1) * bgap) / 2 + k * (bw + bgap)
            body.append(f'<rect x="{bx:.1f}" y="{base_y - hgt:.1f}" width="{bw}" height="{hgt:.1f}" rx="3" fill="{color}"/>')
            body.append(text(bx + bw / 2, base_y - hgt - 6, f"{v:.0f}", 12, 700, INK, "middle"))
        body.append(f'<line x1="{x + 14}" y1="{base_y}" x2="{x + PW - 14:.1f}" y2="{base_y}" stroke="{MUTED}"/>')
    lx = 24
    for _, label, color in JUDGED:
        body.append(f'<rect x="{lx}" y="{py + ph + 22}" width="14" height="14" rx="3" fill="{color}"/>')
        body.append(text(lx + 20, py + ph + 34, label, 13, 400, INK))
        lx += 40 + 7.0 * len(label)
    return svg(W, H, "Blind LLM judges on LA-murre: gpt-4o-transcribe, Gemini 3.5 Flash and base Whisper far behind our three fine-tuned models", body)


MAKER = {  # arm id prefix -> logo file in docs/img/logos (or a Lucide icon); first match wins
    "vertex_gemini": "gemini", "vertex_": "google", "google_": "google", "elevenlabs_": "elevenlabs", "openai_": "openai",
    "azure_": "openai",
    "hf_whisper-large-v3": "openai", "hf_whisper-large-finnish-v3": "hugging-face", "hh_whisperx": "hugging-face",
    "hf_orukeet": "hugging-face", "hf_wav2vec2": "hugging-face", "nemo_rasmus": "hugging-face",
    "nemo_canary": "nvidia", "hf_parakeet": "nvidia", "qwen_": "qwen", "ft_": "",
}


def maker(arm_id: str) -> str:
    return next((v for k, v in MAKER.items() if arm_id.startswith(k)), "")


def gaik_mark(x: float, y: float, h: float) -> str:
    """GAIK's letter mark (docs/img/logos/gaik-mark.png, from the GAIK brand files) embedded as a data URI."""
    import base64
    import struct

    raw = (LOGOS / "gaik-mark.png").read_bytes()
    pw, ph = struct.unpack(">II", raw[16:24])  # the PNG header's width and height
    w = h * pw / ph
    data = base64.b64encode(raw).decode()
    return f'<image x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" href="data:image/png;base64,{data}"/>'


def label_with_logo(arm_id: str, label: str, x: float, y: float, bold: bool, uid: str) -> list[str]:
    """A row label with the maker's logo in a column to its left (our models get a blue dot)."""
    out = [text(x + 30, y, label, 14, 700 if bold else 400, INK)]
    lx = x
    name = maker(arm_id)
    if arm_id.startswith("ft_"):  # this project: the GAIK mark
        out.append(gaik_mark(lx, y - 15, 18))
    elif name:
        m, w = mark(name, lx, y - 15, 18, uid)
        out.append(m)
    return out


WEBINAR_LABELS = {**SHORT, "ft_whisper-large-v3-fi-parl-chunks": "Ours, main model (in pieces)", "ft_whisper-large-v3-fi-skn-punct-best-chunks": "Ours, first model (in pieces)",
                  "hh_whisperx_fi": "WhisperX + Finnish-NLP model", "hf_whisper-large-finnish-v3": "Finnish-NLP whisper-large-finnish-v3"}


def webinar() -> str:
    """The Ajokortti clip, two scores side by side: meaning-error points (mean of two judges) and WER."""
    scores = {a["arm_id"]: a for a in json.loads(SCORES_PATH.read_text(encoding="utf-8"))["arms"]}
    rows = []
    for arm_id, (_, kind) in ROWS.items():
        pts = sum(judge(j, arm_id)["error_points_per_100w"] for j in JUDGES) / len(JUDGES)
        rows.append((pts, scores[arm_id]["metrics"]["WER"], arm_id))
    by_pts = sorted(rows)
    by_wer = sorted(rows, key=lambda r: r[1])
    keep = {r[2] for r in by_pts[:7]} | {r[2] for r in by_wer[:4]} | {"ft_whisper-large-v3-fi-parl-chunks"}
    keep -= {"ft_whisper-large-v3-fi-skn-lamurre-best-chunks", "ft_whisper-large-v3-fi-pilot"}  # LA-murre / pilot: in the full table
    shown = [r for r in by_pts if r[2] in keep]
    def tied_rank(i: int) -> dict[str, int]:  # equal values at one decimal share a rank
        return {r[2]: 1 + sum(round(o[i], 1) < round(r[i], 1) for o in rows) for r in rows}

    rank_pts, rank_wer = tied_rank(0), tied_rank(1)
    W, ROW, TOP, LX = 1000, 34, 128, 24
    P1, P1W, P2, P2W = 370, 250, 690, 220
    body = [text(24, 36, "Standard spoken Finnish: two scores, two different winners", 19, 700)]
    body.append(text(24, 60, f"A 180 s dental webinar with many brand names (the Ajokortti clip), {len(rows)} set-ups. Lower is better in both.", 14, 400, MUTED))
    body.append(text(P1, 98, "Meaning-error points per 100 words", 14, 700, LLM))
    body.append(text(P1, 115, "two blind LLM judges (Claude, Gemini)", 12, 400, MUTED))
    body.append(text(P2, 98, "Word error rate %", 14, 700, OURS))
    body.append(text(P2, 115, "every differing word counts", 12, 400, MUTED))
    vmax1, vmax2 = 10.0, 22.0
    for i, (pts, wer, arm_id) in enumerate(shown):
        y = TOP + i * ROW
        ours = arm_id.startswith("ft_")
        if ours:
            body.append(f'<rect x="12" y="{y - 2}" width="{W - 24}" height="{ROW}" rx="6" fill="{OURS_SOFT}"/>')
        body.extend(label_with_logo(arm_id, WEBINAR_LABELS[arm_id], LX, y + 20, ours, f"w{i}"))
        for x0, pw, v, vmax, rank, color in ((P1, P1W, pts, vmax1, rank_pts[arm_id], LLM), (P2, P2W, wer, vmax2, rank_wer[arm_id], OURS)):
            bw = max(pw * min(v, vmax) / vmax, 2)
            body.append(f'<rect x="{x0}" y="{y + 7}" width="{bw:.1f}" height="{ROW - 14}" rx="3" fill="{color}" opacity="{1 if rank == 1 else 0.55}"/>')
            tag = f"{v:.1f}" + ("  best" if rank == 1 else f"  #{rank}")
            body.append(text(x0 + bw + 6, y + 21, tag, 13, 700 if rank == 1 else 400, INK))
    bottom = TOP + len(shown) * ROW
    body.append(text(24, bottom + 26, "Gemini 3.5 Flash makes the fewest meaning errors. Our model's WER is near the top, but its few errors hit brand names and", 13, 400, INK))
    body.append(text(24, bottom + 44, "dental terms, which change the meaning. #n = rank among all set-ups; the rest are in the full table.", 13, 400, INK))
    return svg(W, bottom + 58, "Webinar clip: meaning-error points and word error rate side by side", body)


def marked(ref: str, hyp: str) -> tuple[list[tuple[str, bool]], int, int]:
    """hyp's words with the ones WER counts as wrong flagged; the error count (S+D+I) and the reference length.

    Words are compared as the scorer does: lower-cased, every punctuation character removed."""
    import jiwer

    def norm(t: str) -> str:
        return jiwer.RemovePunctuation()(t.lower()).strip()

    pairs = [(t, norm(t)) for t in hyp.split()]
    ref_words = [w for w in (norm(w) for w in ref.split()) if w]
    out = jiwer.process_words(" ".join(ref_words), " ".join(n for _, n in pairs if n))
    bad = {i for a in out.alignments[0] if a.type in ("substitute", "insert") for i in range(a.hyp_start_idx, a.hyp_end_idx)}
    parts, k = [], 0
    for t, n in pairs:
        parts.append((t, bool(n) and k in bad))
        k += bool(n)
    spaced = [(t + (" " if i < len(parts) - 1 else ""), f) for i, (t, f) in enumerate(parts)]
    return spaced, out.substitutions + out.deletions + out.insertions, len(ref_words)


def bench_row(arm: str, seg: str) -> dict | None:
    path = PROJECT_ROOT / "results" / "benchmarks" / "skn_fi" / f"{arm}.jsonl"
    if not path.exists():
        return None
    return next((r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r["id"] == seg), None)


EXAMPLE_SEG = "SKN45a_Heinola_0097.wav"
EXAMPLE_ARMS = [  # arm, label, maker logo, colour
    ("hf_whisper-large-v3", "Whisper large-v3 (OpenAI), as it is", "openai", BASE),
    ("vertex_gemini-3.5-flash-std", "Gemini 3.5 Flash (Google), asked for standard Finnish", "gemini", LLM),
    ("ft_whisper-large-v3-fi-skn-punct-best", "Our model", "", OURS),
]


def example() -> str:
    """SKN45a_Heinola_0097, a test segment: each model's output from results/benchmarks/skn_fi, errors marked."""
    rows = [(a, lab, lg, col, bench_row(a, EXAMPLE_SEG)) for a, lab, lg, col in EXAMPLE_ARMS]
    rows = [r for r in rows if r[4]]
    ref = rows[0][4]["ref"]
    W = 1000
    H = 104 + 90 * (len(rows) + 1) + 30
    body = [text(24, 38, "One real test sentence, from a municipality the model never heard", 19, 700)]
    body.append(icon("audio-lines", 24, 56, 22, INK))
    body.append(text(54, 73, "Heinola · SKN test segment SKN45a_Heinola_0097 · an older speaker's dialect", 14, 400, MUTED))
    y = 104
    body.append(f'<rect x="24" y="{y}" width="{W - 48}" height="76" rx="10" fill="#f6f8fa" stroke="{MUTED}"/>')
    body.append(text(42, y + 26, "What was said (a person's standard-Finnish transcript)", 14, 700, MUTED))
    body.append(text(42, y + 56, ref, 16))
    for i, (arm, lab, lg, col, row) in enumerate(rows):
        y += 90
        parts, errs, n = marked(ref, row["hyp"])
        ours = arm.startswith("ft_")
        body.append(f'<rect x="24" y="{y}" width="{W - 48}" height="76" rx="10" fill="{OURS_SOFT if ours else BG}" stroke="{col}" stroke-width="2"/>')
        x0 = 42
        if lg:
            m, w = mark(lg, 42, y + 11, 20, f"e{i}")
            body.append(m)
            x0 = 42 + w + 8
        body.append(text(x0, y + 26, lab, 14, 700, OURS if ours else INK))
        body.append(text(W - 42, y + 26, f"{errs} of {n} words wrong", 15, 700, OURS if ours else WARN, "end"))
        body.append(words(42, y + 56, parts, 16))
    body.append(text(24, y + 102, "In English: “Even when I sometimes bring it something to drink, it doesn't bite, but it grimaces so you'd think it bites now.”", 13, 400, MUTED))
    return svg(W, H, "A Heinola test segment: Whisper large-v3, Gemini 3.5 Flash and our model against a person's transcript", body)


# The brand-name sentence of the Ajokortti clip (78.3-84.0 s, results/clip/gt_alignment.json): each arm's text as
# stored in results/transcriptions/, and the Claude judge's verdicts from results/judgements/claude-opus-5-5/.
BRAND_REF = "Straumann, Densply Sirona, Nobel Biocare ja Camlog-Conelog eli Implantona Oy."
BRAND_ROWS = [
    ("Gemini 3.5 Flash (Google)", "gemini", "Straumann, Dentsply Sirona, Nobel Biocare ja Camlog Conelog eli Implantona Oy.",
     set(), [("no meaning errors", OPEN)]),
    ("Our model", "", "Straumann, Densplay-sirvana, Nobelbayer, ja Camlock-Connellock eli implanttoona Oy.",
     {"Densplay-sirvana,", "Nobelbayer,", "Camlock-Connellock", "implanttoona"},
     [("major: “Nobelbayer” is a different name", WARN), ("3 × minor: misspelled, still recognisable", "#b7791f")]),
    ("Whisper large-v3 (OpenAI)", "openai", "Straumanni, Dentsply Sirona, Noble Biocare ja Camlock Conelog, eli implanttoona OY.",
     {"Straumanni,", "Noble", "Camlock", "implanttoona"}, [("1 × minor: names misspelled, still recognisable", "#b7791f")]),
]


def judge_example() -> str:
    W, H = 1000, 470
    body = [text(24, 38, "What the judge does: one real sentence from the webinar (78–84 s)", 19, 700)]
    body.append(text(24, 62, "The judge reads the reference and each transcript without knowing which model wrote it, and scores only", 14, 400, MUTED))
    body.append(text(24, 82, "the errors that change the meaning: critical 10, major 5, minor 1 point.", 14, 400, MUTED))
    body.append(f'<rect x="24" y="100" width="{W - 48}" height="62" rx="10" fill="#f6f8fa" stroke="{MUTED}"/>')
    body.append(text(42, 123, "Reference: four implant brands", 13, 700, MUTED))
    body.append(text(42, 148, BRAND_REF, 16))
    y = 176
    for i, (lab, lg, hyp, bad, verdicts) in enumerate(BRAND_ROWS):
        ours = lg == ""
        body.append(f'<rect x="24" y="{y}" width="{W - 48}" height="88" rx="10" fill="{OURS_SOFT if ours else BG}" stroke="{OURS if ours else LINE}" stroke-width="{2 if ours else 1.5}"/>')
        x0 = 42
        if lg:
            m, w = mark(lg, 42, y + 10, 20, f"b{i}")
            body.append(m)
            x0 = 42 + w + 8
        body.append(text(x0, y + 25, lab, 14, 700, OURS if ours else INK))
        toks = hyp.split()
        body.append(words(42, y + 52, [(t + (" " if k < len(toks) - 1 else ""), t in bad) for k, t in enumerate(toks)], 16))
        cx = 42
        for label, color in verdicts:
            w = 7.0 * len(label) + 22
            body.append(f'<rect x="{cx}" y="{y + 62}" width="{w:.0f}" height="20" rx="10" fill="{color}"/>')
            body.append(text(cx + 11, y + 76, label, 12, 700, "#ffffff"))
            cx += w + 8
        y += 98
    body.append(text(24, y + 10, "Verdicts of the main judge, Claude Opus 5.5; the second judge, Gemini 3.1 Pro, also rates “Nobelbayer” major.", 13, 400, INK))
    body.append(text(24, y + 28, "Red: the brand names the judge looked at. Hear the sentence on the listening page.", 13, 400, INK))
    return svg(W, y + 42, "A brand-name sentence from the webinar: Gemini 3.5 Flash, our model and Whisper large-v3 with the judge's verdicts", body)


def leaderboard() -> str:
    """Results at a glance: dialect speech (SKN WER) and standard Finnish (FLEURS + VoxPopuli, pooled WER)."""
    def arms(bench: str) -> dict:
        path = PROJECT_ROOT / "results" / "benchmarks" / bench / "scores.json"
        return {a["arm_id"]: a for a in json.loads(path.read_text(encoding="utf-8"))["arms"]}
    skn = {a: r["WER"] for a, r in arms("skn_fi").items()}
    left = sorted((skn[a], lab, grp, a) for a, (lab, grp) in DIALECT_SHORT.items() if a in skn and (grp != "ours" or a in LEADERBOARD_OURS))
    fl, vp = arms("fleurs_fi"), arms("voxpopuli_fi")
    std = []
    for a, (lab, grp) in DIALECT_SHORT.items():  # pooled: all errors over all reference words of both sets
        if a in fl and a in vp and a != "ft_whisper-large-v3-fi-skn-punct-ts":
            err = sum(t[a]["WER"] * t[a]["ref_words"] for t in (fl, vp))
            std.append((err / (fl[a]["ref_words"] + vp[a]["ref_words"]), lab, grp, a))
    std.sort()
    N = 6
    ours_rank = next(i for i, r in enumerate(std) if r[3] == "ft_whisper-large-v3-fi-parl")
    right = std[:N] if ours_rank < N else std[: N - 2] + [None, std[ours_rank]]
    orank_right = None if ours_rank < N else ours_rank
    W, PW, ROW, TOP = 1000, 466, 34, 118
    short = {"nemo_rasmus-canary-fi-v2-kenlm": "RASMUS Canary v2 + KenLM", "nemo_rasmus-canary-fi-v2": "RASMUS Canary v2",
             "ft_whisper-large-v3-fi-parl-chunks": "Ours, main model", "vertex_gemini-3.5-flash-std": "Gemini 3.5 Flash, std Finnish", "vertex_gemini-3.5-flash": "Gemini 3.5 Flash, as spoken",
             "hf_whisper-large-v3-chunks": "Whisper large-v3, in pieces"}
    colors = {"API": API, "open": OPEN, "ours": OURS}
    body = [text(24, 36, "Results at a glance", 19, 700)]
    panels = [
        (24, "Dialect speech", "words wrong, %: 600 interview segments", left[:N], 45.0, "{:.1f} %", None),
        (24 + PW + 20, "Standard Finnish", "words wrong, %: FLEURS + VoxPopuli, 1 396 segments", right, 15.0, "{:.1f} %", orank_right),
    ]
    for px, title, sub, rows, vmax, fmt, orank in panels:
        body.append(f'<rect x="{px}" y="52" width="{PW}" height="{TOP - 52 + N * ROW + 26}" rx="10" fill="#f6f8fa" stroke="{LINE}"/>')
        body.append(text(px + 16, 78, title, 16, 700, INK))
        body.append(text(px + 16, 98, sub, 12.5, 400, MUTED))
        for i, r in enumerate(rows):
            y = TOP + i * ROW
            if r is None:
                body.append(text(px + 46, y + 20, "…", 14, 700, MUTED))
                continue
            v, lab, grp, arm = r
            rank = (orank + 1) if (orank is not None and i == len(rows) - 1) else i + 1
            if grp == "ours":
                body.append(f'<rect x="{px + 6}" y="{y + 1}" width="{PW - 12}" height="{ROW - 2}" rx="6" fill="{OURS_SOFT}"/>')
            body.append(text(px + 30, y + 22, f"{rank}.", 14, 700, MUTED, "end"))
            lab = short.get(arm, lab)
            body.extend(label_with_logo(arm, lab if len(lab) <= 32 else lab[:31] + "…", px + 38, y + 22, grp == "ours", f"l{px}{i}"))
            bx, bw = px + 290, PW - 290 - 60
            w = max(bw * min(v, vmax) / vmax, 2)
            body.append(f'<rect x="{bx}" y="{y + 9}" width="{w:.1f}" height="{ROW - 18}" rx="3" fill="{colors[grp]}"/>')
            body.append(text(bx + w + 6, y + 22, fmt.format(v), 13, 700, INK))
    bottom = TOP + N * ROW + 26 + 52 - 52
    y = bottom + 34
    lx = 24
    for name, color in (("paid API service", API), ("open model, runs on your own GPU", OPEN), ("this project", OURS)):
        body.append(f'<rect x="{lx}" y="{y - 12}" width="14" height="14" rx="3" fill="{color}"/>')
        body.append(text(lx + 20, y, name, 13, 400, INK))
        lx += 34 + 7.2 * len(name)
    body.append(text(W - 24, y, "lower is better in both", 13, 400, MUTED, "end"))
    y += 24
    body.append(text(24, y, "Standard Finnish: our model and the Finnish-NLP and RASMUS fine-tunes were trained on these sets' training splits.", 12.5, 400, MUTED))
    return svg(W, y + 16, "Results at a glance: best models on dialect speech and on standard Finnish public test sets", body)


def bench_wer(bench: str) -> dict[str, float]:
    path = PROJECT_ROOT / "results" / "benchmarks" / bench / "scores.json"
    if not path.exists():
        return {}
    return {a["arm_id"]: a["WER"] for a in json.loads(path.read_text(encoding="utf-8"))["arms"]}


DENTAL_LABELS = {  # arm -> (label, group); arms missing from dental10_fi/scores.json are left out
    "ft_whisper-large-v3-fi-skn-punct-best-chunks": ("Ours, first model (in pieces)", "ours"),
    "ft_whisper-large-v3-fi-skn-all-chunks": ("Ours, all of SKN (in pieces)", "ours"),
    "ft_whisper-large-v3-fi-skn-punct-ts-fallback": ("Ours, timestamps version (in one go)", "ours"),
    "azure_gpt-4o-transcribe": ("OpenAI gpt-4o-transcribe", "API"),
    "azure_whisper": ("OpenAI Whisper (API)", "API"),
    "elevenlabs_scribe-v2": ("ElevenLabs Scribe v2", "API"),
    "google_chirp-3": ("Google Chirp 3", "API"),
    "vertex_gemini-3.5-flash": ("Gemini 3.5 Flash", "API"),
    "vertex_gemini-3.1-pro": ("Gemini 3.1 Pro", "API"),
    "hf_whisper-large-v3": ("Whisper large-v3", "open"),
    "hf_whisper-large-v3-chunks": ("Whisper large-v3 (in pieces)", "open"),
    "hf_whisper-large-v3-turbo": ("Whisper large-v3 turbo", "open"),
    "hf_whisper-large-finnish-v3": ("Finnish-NLP whisper-large-finnish-v3", "open"),
    "nemo_rasmus-canary-fi-v2-kenlm": ("RASMUS Finnish Canary v2 + KenLM", "open"),
    "nemo_canary-1b-v2": ("NVIDIA Canary 1B v2", "open"),
    "hf_orukeet": ("Orukeet (Parakeet v3 fine-tune)", "open"),
    "hf_parakeet-tdt-0.6b-v3": ("NVIDIA Parakeet TDT 0.6B v3", "open"),
    "hf_wav2vec2-xlsr-1b-finnish-lm-v2": ("Finnish-NLP wav2vec2 XLS-R 1B", "open"),
    "qwen_qwen3-asr-1.7b": ("Qwen3-ASR 1.7B", "open"),
}


def dental_webinars() -> str:
    """The ten dental-webinar clips: meaning-error points (mean of two blind judges) and WER side by side."""
    summary = json.loads((PROJECT_ROOT / "results" / "judgements_dental" / "summary.json").read_text(encoding="utf-8"))["judges"]
    wer = bench_wer("dental10_fi")
    judged = set.intersection(*(set(arms) for arms in summary.values()))
    rows = sorted((sum(summary[j][a]["points_per_100w"] for j in summary) / len(summary), wer[a], a)
                  for a in judged if a in wer and a in DENTAL_LABELS)  # fmt: skip
    best_pts, best_wer = min(r[0] for r in rows), min(r[1] for r in rows)
    W, ROW, TOP, LX = 1000, 34, 128, 24
    P1, P1W, P2, P2W = 370, 250, 690, 220
    body = [text(24, 36, "Dental webinars: ten clips, 30 minutes of standard spoken Finnish", 19, 700)]
    body.append(text(24, 60, "Against the University of Helsinki's transcripts, which leave out fillers (“niinku”, “tota”); so word error rate", 14, 400, MUTED))
    body.append(text(24, 79, "favours models that drop them too. The judges ignore fillers and count only meaning. Lower is better in both.", 14, 400, MUTED))
    body.append(text(P1, 102, "Meaning-error points per 100 words", 14, 700, LLM))
    body.append(text(P1, 119, "two blind LLM judges (Claude, Gemini)", 12, 400, MUTED))
    body.append(text(P2, 102, "Word error rate %", 14, 700, OURS))
    body.append(text(P2, 119, "every differing word counts", 12, 400, MUTED))
    vmax1, vmax2 = 8.0, 30.0
    for i, (pts, w, arm) in enumerate(rows):
        y = TOP + i * ROW
        ours = arm.startswith("ft_")
        if ours:
            body.append(f'<rect x="12" y="{y - 2}" width="{W - 24}" height="{ROW}" rx="6" fill="{OURS_SOFT}"/>')
        body.extend(label_with_logo(arm, DENTAL_LABELS[arm][0], LX, y + 20, ours, f"d{i}"))
        for x0, pw, v, vmax, best, color in ((P1, P1W, pts, vmax1, best_pts, LLM), (P2, P2W, w, vmax2, best_wer, OURS)):
            bw = max(pw * min(v, vmax) / vmax, 2)
            top = round(v, 1) == round(best, 1)
            body.append(f'<rect x="{x0}" y="{y + 7}" width="{bw:.1f}" height="{ROW - 14}" rx="3" fill="{color}" opacity="{1 if top else 0.55}"/>')
            body.append(text(x0 + bw + 6, y + 21, f"{v:.1f}" + ("  best" if top else ""), 13, 700 if top else 400, INK))
    bottom = TOP + len(rows) * ROW
    body.append(text(24, bottom + 26, "Whisper-style models win on word error rate; Gemini 3.5 Flash and ElevenLabs make the fewest meaning errors. Our model is", 13, 400, INK))
    body.append(text(24, bottom + 44, "built for dialects and is not the pick here. The transcripts were not checked by a listener (see the README).", 13, 400, INK))
    return svg(W, bottom + 58, "Ten dental-webinar clips: meaning-error points and word error rate side by side", body)


ACROSS = [  # arm per dataset, label, colour, logo
    ({"*": "hf_whisper-large-v3"}, "Whisper large-v3 (OpenAI, open)", BASE, "openai"),
    ({"*": "ft_whisper-large-v3-fi-parl"}, "Ours, main model", OURS, ""),
    ({"*": "azure_gpt-4o-transcribe"}, "OpenAI gpt-4o-transcribe (API)", "#6b7280", "openai"),
    ({"*": "vertex_gemini-3.5-flash", "skn_fi": "vertex_gemini-3.5-flash-std", "lamurre_fi": "vertex_gemini-3.5-flash-std"},
     "Gemini 3.5 Flash (API)", LLM, "gemini"),
]
ACROSS_SETS = [
    ("skn_fi", "Dialect interviews", "SKN, 600 segments"),
    ("fleurs_fi", "Read sentences", "FLEURS, 918"),
    ("voxpopuli_fi", "Parliament speech", "VoxPopuli, 478"),
]


def across_sets() -> str:
    """Four models on three test sets, word error rate (LA-murre: judges.svg; the dental clips: dental-webinars.svg)."""
    wer = {b: bench_wer(b) for b, _, _ in ACROSS_SETS}
    W = 1000
    body = [text(24, 36, "Four models, three kinds of Finnish speech", 19, 700)]
    body.append(text(24, 60, "Word error rate %, lower is better. Gemini on the dialect set was asked for standard Finnish word forms.", 14, 400, MUTED))
    GW, gap, gy, gh = 310, 15, 84, 270
    vmax = 45.0
    for gi, (bench, title, sub) in enumerate(ACROSS_SETS):
        x = 24 + gi * (GW + gap)
        body.append(f'<rect x="{x}" y="{gy}" width="{GW}" height="{gh}" rx="10" fill="#f6f8fa" stroke="{LINE}"/>')
        body.append(text(x + 12, gy + 26, title, 15, 700, INK))
        body.append(text(x + 12, gy + 45, sub, 12, 400, MUTED))
        base_y, top_y = gy + gh - 22, gy + 70
        bw, bgap = 52, 16
        vals = [wer[bench].get(arms.get(bench, arms["*"])) for arms, _, _, _ in ACROSS]
        best = min(v for v in vals if v is not None) if any(v is not None for v in vals) else None
        for k, ((_, _, color, _), v) in enumerate(zip(ACROSS, vals, strict=True)):
            bx = x + 24 + k * (bw + bgap)
            if v is None:
                body.append(text(bx + bw / 2, base_y - 6, "–", 13, 400, MUTED, "middle"))
                continue
            hgt = (base_y - top_y) * min(v, vmax) / vmax
            body.append(f'<rect x="{bx}" y="{base_y - hgt:.1f}" width="{bw}" height="{hgt:.1f}" rx="3" fill="{color}"/>')
            body.append(text(bx + bw / 2, base_y - hgt - 6, f"{v:.1f}", 12.5, 700 if v == best else 400, INK, "middle"))
        body.append(f'<line x1="{x + 12}" y1="{base_y}" x2="{x + GW - 12}" y2="{base_y}" stroke="{MUTED}"/>')
    ly = gy + gh + 30
    lx = 24
    for i, (_, label, color, lg) in enumerate(ACROSS):
        body.append(f'<rect x="{lx}" y="{ly - 12}" width="14" height="14" rx="3" fill="{color}"/>')
        lx += 20
        if lg:
            m, w = mark(lg, lx, ly - 14, 16, f"a{i}")
            body.append(m)
            lx += w + 6
        body.append(text(lx, ly, label, 13, 700 if color == OURS else 400, INK))
        lx += 26 + 7.0 * len(label)
    body.append(text(24, ly + 26, "The dialect test's municipalities were kept out of our training; the API services' training data is unknown.", 12.5, 400, MUTED))
    return svg(W, ly + 40, "Whisper large-v3, our model, gpt-4o-transcribe and Gemini 3.5 Flash on three test sets", body)


def all_sets() -> str:
    """Every model with all three public test sets: one row each, WER per set side by side (best per set bold)."""
    def wer(bench: str) -> dict[str, float]:
        return bench_wer(bench)
    sets = [("skn_fi", "Dialect interviews", "SKN, 600 segments", 70.0),
            ("fleurs_fi", "Read sentences", "FLEURS, 918", 30.0),
            ("voxpopuli_fi", "Parliament speech", "VoxPopuli, 478", 30.0)]
    w = {b: wer(b) for b, *_ in sets}
    labels = {a: lab for a, (lab, _) in DIALECT_SHORT.items()}
    labels.update({"hf_wav2vec2-xlsr-1b-finnish-lm-v2": "Finnish-NLP wav2vec2 XLS-R 1B", "hf_parakeet-tdt-0.6b-v3": "NVIDIA Parakeet TDT 0.6B v3",
                   "hf_orukeet": "Orukeet (Parakeet v3 fine-tune)", "ft_whisper-large-v3-fi-parl": "Ours, main model"})
    group = {a: g for a, (_, g) in DIALECT_SHORT.items()}
    rows = [a for a in labels if all(a in w[b] for b, *_ in sets) and a != "ft_whisper-large-v3-fi-skn-punct-ts"]
    rows.sort(key=lambda a: w["skn_fi"][a])
    best = {b: min(w[b][a] for a in rows) for b, *_ in sets}
    colors = {"API": API, "open": OPEN, "ours": OURS}
    W, ROW, TOP, LX, CX, CW = 1000, 30, 128, 24, 360, 200
    body = [text(24, 36, "Every model on the three public test sets", 19, 700),
            text(24, 60, "Word error rate %, lower is better; sorted by the dialect test. Bold: best on that set.", 14, 400, MUTED)]
    for k, (b, title, sub, vmax) in enumerate(sets):
        x = CX + k * (CW + 10)
        body.append(text(x, 92, title, 14, 700, INK))
        body.append(text(x, 109, sub, 12, 400, MUTED))
    for i, a in enumerate(rows):
        y = TOP + i * ROW
        grp = "ours" if a.startswith("ft_") else group.get(a, "open")
        if grp == "ours":
            body.append(f'<rect x="12" y="{y}" width="{W - 24}" height="{ROW}" rx="6" fill="{OURS_SOFT}"/>')
        body.extend(label_with_logo(a, labels[a], LX, y + 20, grp == "ours", f"s{i}"))
        for k, (b, _, _, vmax) in enumerate(sets):
            x = CX + k * (CW + 10)
            v = w[b][a]
            bw = (CW - 52) * min(v, vmax) / vmax
            body.append(f'<rect x="{x}" y="{y + 9}" width="{bw:.1f}" height="{ROW - 18}" rx="3" fill="{colors[grp]}"/>')
            top = round(v, 1) == round(best[b], 1)
            body.append(text(x + bw + 5, y + 20, f"{v:.1f}", 12.5, 700 if top else 400, INK))
    bottom = TOP + len(rows) * ROW
    y = bottom + 30
    lx = 24
    for name, color in (("paid API service", API), ("open model, runs on your own GPU", OPEN), ("this project", OURS)):
        body.append(f'<rect x="{lx}" y="{y - 12}" width="14" height="14" rx="3" fill="{color}"/>')
        body.append(text(lx + 20, y, name, 13, 400, INK))
        lx += 34 + 7.2 * len(name)
    body.append(text(24, y + 24, "Our model and the Finnish-NLP and RASMUS fine-tunes were trained on FLEURS and VoxPopuli training data. Gemini as spoken here; asked for standard Finnish it gets 34.7 % on SKN.", 12, 400, MUTED))
    return svg(W, y + 38, "Word error rate of every model on SKN dialects, FLEURS and VoxPopuli", body)


FIGURES = {
    "example.svg": example,
    "punctuation.svg": punctuation,
    "data-sources.svg": data_sources,
    "pipeline.svg": pipeline,
    "dialect-ranking.svg": dialect_ranking,
    "data-split.svg": data_split,
    "two-measures.svg": two_measures,
    "clip-ranking.svg": clip_ranking,
    "which-version.svg": which_version,
    "journey.svg": journey,
    "judges.svg": judges,
    "webinar.svg": webinar,
    "judge-example.svg": judge_example,
    "leaderboard.svg": leaderboard,
    "dental-webinars.svg": dental_webinars,
    "across-sets.svg": across_sets,
    "all-sets.svg": all_sets,
}


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    IMG.mkdir(parents=True, exist_ok=True)
    for name, draw in FIGURES.items():
        (IMG / name).write_text(draw(), encoding="utf-8", newline="\n")
        print(f"wrote docs/img/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
