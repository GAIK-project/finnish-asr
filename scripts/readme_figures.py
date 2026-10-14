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
svgl (docs/img/logos), used only to name the data sources and tools. No keys, no network.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
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


def example() -> str:
    """SKN45a_Heinola_0097, a test segment; the outputs and error counts are the story's (section 04)."""
    W, H = 1000, 404
    body = [text(24, 38, "One real test sentence, from a municipality the model never heard", 19, 700)]
    body.append(icon("audio-lines", 24, 56, 22, INK))
    body.append(text(54, 73, "Heinola · SKN test segment SKN45a_Heinola_0097 · an older speaker's dialect", 14, 400, MUTED))
    rows = [
        ("What was said (a person's standard-Finnish transcript)", MUTED, None,
         [("vaikka minäkin välistä vien juomaakin sille niin ei se pure mutta se irvistää että luulisi että nyt se puree", False)]),
        ("Whisper large-v3 (OpenAI), as it is", BASE, "8 of 19 words wrong",
         [("aika miekivälistäviä", True), (" juomaakin sille, niin ei se pure. ", False), ("Mut", True), (" se ", False),
          ("hirvisti", True), (", että ", False), ("luulis", True), (", että nyt se ", False), ("puri", True), (".", False)]),
        ("Our model", OURS, "1 of 19 words wrong",
         [("Vaikka minäkin välistä vien juomaakin sille, niin ei se pure, mutta se ", False), ("irvisti", True),
          (", että luulisi, että nyt se puree.", False)]),
    ]
    y = 104
    for k, (head, color, score, parts) in enumerate(rows):
        fill = "#f6f8fa" if k == 0 else BG
        body.append(f'<rect x="24" y="{y}" width="{W - 48}" height="76" rx="10" fill="{fill}" stroke="{color}" stroke-width="{1 if k == 0 else 2}"/>')
        body.append(text(42, y + 26, head, 14, 700, color if k else MUTED))
        if score:
            body.append(text(W - 42, y + 26, score, 15, 700, WARN if k == 1 else OURS, "end"))
        body.append(words(42, y + 56, parts, 16))
        y += 90
    body.append(text(24, y + 12, "In English: “Even when I sometimes bring it something to drink, it doesn't bite, but it grimaces so you'd think it bites now.”", 13, 400, MUTED))
    return svg(W, H, "A Heinola test segment: Whisper large-v3 gets 8 of 19 words wrong, our model 1", body)


def punctuation() -> str:
    """SKN03a_Palkane_0021, a validation segment, as in the story's data section."""
    W, H = 1000, 372
    body = [ARROW_DEF, text(24, 38, "What the language model did to the training text: punctuation, nothing else", 19, 700)]
    body.append(f'<rect x="24" y="58" width="{W - 48}" height="70" rx="10" fill="#f6f8fa" stroke="{LINE}"/>')
    body.append(text(42, 82, "Before: a person's transcript from SKN (Pälkäne)", 14, 700, MUTED))
    body.append(text(42, 112, "vieläkö osaisitte kyllä minä niin kauan osaan kun minä elän", 17, 400))
    mark, w = logo("claude", 42, 146, 26, "pu")
    body.append(mark)
    body.append(text(42 + w + 10, 165, "Claude Opus 5.5 adds full stops, question marks, commas and capitals", 15, 700, LLM))
    body.append(arrow(W - 60, 134, W - 60, 190))
    body.append(f'<rect x="24" y="196" width="{W - 48}" height="70" rx="10" fill="{LLM_SOFT}" stroke="{LLM}" stroke-width="1.5"/>')
    body.append(text(42, 220, "After: the training target", 14, 700, LLM))
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
        ("map-pin", "LA-murre", ["Dialect interviews, 10 places", "Univ. of Turku · CC BY-ND"], "test only", INK),
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


DIALECT_SHORT = {
    "ft_whisper-large-v3-fi-skn-punct-best": ("Ours (published)", "ours"),
    "ft_whisper-large-v3-fi-skn-punct-ts": ("Ours, timestamps version", "ours"),
    "hf_whisper-large-v3": ("Whisper large-v3 (OpenAI)", "open"),
    "hf_whisper-large-v3-turbo": ("Whisper large-v3 turbo (OpenAI)", "open"),
    "nemo_canary-1b-v2": ("Canary 1B v2 (NVIDIA)", "open"),
    "nemo_rasmus-canary-fi-v2-kenlm": ("Finnish Canary v2 + KenLM (RASMUS)", "open"),
    "nemo_rasmus-canary-fi-v2": ("Finnish Canary v2 (RASMUS)", "open"),
    "hf_whisper-large-finnish-v3": ("whisper-large-finnish-v3 (Finnish-NLP)", "open"),
    "qwen_qwen3-asr-1.7b": ("Qwen3-ASR 1.7B (Alibaba)", "open"),
}


def dialect_ranking() -> str:
    path = PROJECT_ROOT / "results" / "benchmarks" / "skn_fi" / "scores.json"
    wer = {a["arm_id"]: a["WER"] for a in json.loads(path.read_text(encoding="utf-8"))["arms"]}
    rows = sorted((wer[a], label, group) for a, (label, group) in DIALECT_SHORT.items())
    W, ROW, TOP, LX, BX, BW = 1000, 34, 96, 330, 342, 560
    vmax = 70.0
    body = [text(24, 36, "Dialect test: our model against the other open models", 19, 700)]
    body.append(text(24, 60, "Word error rate %, 600 segments from 10 municipalities left out of our training. Lower is better.", 14, 400, MUTED))
    body.append(text(24, 80, "Paid API services were not run on the dialect test.", 13, 400, MUTED))
    for t in range(0, 71, 10):
        x = BX + BW * t / vmax
        body.append(f'<line x1="{x:.1f}" y1="{TOP - 6}" x2="{x:.1f}" y2="{TOP + len(rows) * ROW}" stroke="{LINE}"/>')
        body.append(text(x, TOP + len(rows) * ROW + 16, str(t), 11, 400, MUTED, "middle"))
    for i, (v, label, group) in enumerate(rows):
        y = TOP + i * ROW
        body.append(text(LX, y + 19, label, 14, 700 if group == "ours" else 400, INK, "end"))
        fill = OURS if group == "ours" else BASE
        body.append(f'<rect x="{BX}" y="{y + 5}" width="{BW * v / vmax:.1f}" height="{ROW - 10}" rx="3" fill="{fill}"/>')
        body.append(text(BX + BW * v / vmax + 8, y + 19, f"{v:.1f}", 13, 700, INK))
    return svg(W, TOP + len(rows) * ROW + 30, "SKN dialect test word error rate: our model 21.1 %, the best other open model 40.4 %", body)


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
    body.append(text(76, yb + 50, "10 other localities, 583 segments, test only. Scored by the blind LLM judge.", 14, 400, MUTED))
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
    "ft_whisper-large-v3-fi-skn-punct-best-chunks": "Ours, published, in pieces",
    "ft_whisper-large-v3-fi-skn-lamurre-best-chunks": "Ours, internal (+ LA-murre), in pieces",
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
    ours = "ft_whisper-large-v3-fi-skn-punct-best-chunks"
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
    return svg(W, bottom + 30, "The ten best speech-to-text set-ups on the Ajokortti webinar clip, and our published model", body)


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
