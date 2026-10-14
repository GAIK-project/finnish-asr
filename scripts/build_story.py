r"""The explainer pages about our own dialect model, in Finnish and English.

    uv run python scripts/build_story.py

Reads docs/story/src/ (style.css, body.fi.html, body.en.html, assets/) and writes
docs/story/story.fi.html and docs/story/story.en.html: one file each, with the image, the two
audio clips and the icons inlined, so they open without a network connection. The two charts
are drawn here to scale from the numbers below, which come from
results/finetune/whisper-large-v3-fi-skn-punct-best/train_log.json (epochs 1-3),
results/finetune/whisper-large-v3-fi-skn-punct/train_log.json (epoch 4 of the same recipe) and
docs/DIALECT-EXAMPLES.md (WER per municipality). The judge's instructions are read from
judge_dialect.py as text, without importing it.

Assets: hero.webp was generated with openai/gpt-image-2.5-sunburst on Replicate (an
illustration, no real place or person); the two clips are SKN segments from the public dataset
GAIK-project/skn-finnish-dialects-asr (CC BY 4.0), re-encoded as 48 kbit/s MP3; the icons are
Lucide (ISC licence) from Iconify, and Lucide's database and link icons are derived from Feather (MIT
licence). The footer of each page gets {{credits}} from CREDITS below: the asset credits, the page's
licence and LUMI's acknowledgement, whose wording LUMI prescribes in English
(https://lumi-supercomputer.eu/acknowledgement/, the template for awards by LUMI consortium countries,
filled in for access through CSC).
"""

from __future__ import annotations

import argparse
import base64
import html
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "docs" / "story" / "src"
OUT = PROJECT / "docs" / "story"

LOSS = [("before", 0.5382), ("1", 0.2132), ("2", 0.2063), ("3", 0.2088), ("4*", 0.2169)]
WER = [
    ("Alastaro", 35.7, 17.6),
    ("Artjärvi", 44.4, 23.1),
    ("Eurajoki", 43.7, 22.0),
    ("Heinola", 36.1, 19.9),
    ("Hinnerjoki", 51.8, 29.0),
    ("Ikaalinen", 31.3, 15.0),
    ("Jaala", 37.3, 18.2),
    ("Mikkeli", 38.5, 19.9),
    ("Rautu", 50.4, 31.6),
    ("Vihti", 32.8, 14.4),
    ("ALL", 40.4, 21.1),
]
LUMI_ACK = (
    "We acknowledge CSC – IT Center for Science, Finland for awarding this project access to the LUMI "
    "supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CSC (Finland) and the LUMI consortium "
    "through CSC – IT Center for Science, Finland."
)
LUCIDE = '<a href="https://lucide.dev/">Lucide</a>'
FEATHER = '<a href="https://feathericons.com/">Feather</a>'
CC_BY = '<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>'
CREDITS = {
    "fi": (
        f"Kuvitus: tekoälyn tekemä (GPT Image 2.5). Äänet: kaksi SKN-pätkää, {CC_BY}. Kuvakkeet: {LUCIDE} "
        f"(ISC-lisenssi); tietokanta- ja linkkikuvake pohjautuvat {FEATHER}-kuvakkeisiin (MIT-lisenssi). "
        f"Tämä sivu: GAIK-hanke, Haaga-Helia, {CC_BY}. LUMIn edellyttämä kiitos: <i>{LUMI_ACK}</i>"
    ),
    "en": (
        f"Illustration: AI-generated (GPT Image 2.5). Audio: two SKN segments, {CC_BY}. Icons: {LUCIDE} "
        f"(ISC licence); the database and link icons are derived from {FEATHER} (MIT licence). "
        f"This page: GAIK project, Haaga-Helia, {CC_BY}. {LUMI_ACK}"
    ),
}
TEXT = {
    "fi": {
        "lang": "fi",
        "before": "ennen koulutusta",
        "epoch": "epookki",
        "kept": "talteen",
        "all": "Kaikki 600",
        "loss_label": "Validointihäviö: ennen 0,538, epookit 1–4: 0,213, 0,206, 0,209, 0,217",
        "wer_label": "WER kunnittain, Whisper large-v3 ja oma malli",
        "decimal": ",",
    },
    "en": {
        "lang": "en",
        "before": "before training",
        "epoch": "epoch",
        "kept": "kept",
        "all": "All 600",
        "loss_label": "Validation loss: before 0.538, epochs 1-4: 0.213, 0.206, 0.209, 0.217",
        "wer_label": "WER by municipality, Whisper large-v3 and our model",
        "decimal": ".",
    },
}


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def icon(name: str) -> str:
    svg = (SRC / "assets" / "icons" / f"{name}.svg").read_text(encoding="utf-8")
    return svg.replace("<svg ", '<svg class="ico" aria-hidden="true" focusable="false" ', 1)


def loss_chart(t: dict) -> str:
    w, h, left, right, top, bottom, ymax = 640, 260, 56, 20, 20, 44, 0.6

    def x(i: int) -> float:
        return left + i * (w - left - right) / (len(LOSS) - 1)

    def y(v: float) -> float:
        return top + (1 - v / ymax) * (h - top - bottom)

    parts = []
    for v in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        parts.append(
            f'<line class="gridl" x1="{left}" x2="{w - right}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
            f'<text class="m" x="{left - 8}" y="{y(v) + 4:.1f}" text-anchor="end">'
            f"{f'{v:.1f}'.replace('.', t['decimal'])}</text>"
        )
    path = " ".join(f"{'M' if i == 0 else 'L'}{x(i):.1f},{y(v):.1f}" for i, (_, v) in enumerate(LOSS))
    for i, (label, v) in enumerate(LOSS):
        kept = label == "2"
        parts.append(
            f'<circle class="dot{" keep" if kept else ""}" cx="{x(i):.1f}" cy="{y(v):.1f}" r="{6 if kept else 4.5}"/>'
        )
        value = f"{v:.3f}".replace(".", t["decimal"])
        if label == "before":  # right of the dot, clear of the axis labels
            parts.append(f'<text x="{x(i) + 10:.1f}" y="{y(v) + 4:.1f}" text-anchor="start">{value}</text>')
        else:
            parts.append(f'<text x="{x(i):.1f}" y="{y(v) - 12:.1f}" text-anchor="middle">{value}</text>')
        name = t["before"] if label == "before" else f"{t['epoch']} {label}"
        parts.append(f'<text class="m" x="{x(i):.1f}" y="{h - bottom + 20}" text-anchor="middle">{name}</text>')
    parts.append(
        f'<text x="{x(2):.1f}" y="{y(0.2063) + 26:.1f}" text-anchor="middle" '
        f'style="fill:var(--tape);font-weight:600">{t["kept"]}</text>'
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{t["loss_label"]}">'
        f'{"".join(parts[:7])}<path class="ln" d="{path}"/>{"".join(parts[7:])}</svg>'
    )


def wer_chart(t: dict) -> str:
    w, left, right, top, row, bar, xmax = 640, 96, 46, 24, 34, 12, 60
    h = top + row * len(WER) + 10

    def x(v: float) -> float:
        return left + v / xmax * (w - left - right)

    parts = []
    for v in range(0, xmax + 1, 10):
        parts.append(
            f'<line class="gridl" x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top - 6}" y2="{h - 6}"/>'
            f'<text class="m" x="{x(v):.1f}" y="{top - 10}" text-anchor="middle">{v} %</text>'
        )
    for i, (place, base, ours) in enumerate(WER):
        y = top + i * row
        total = place == "ALL"
        name = t["all"] if total else place
        weight = ' style="font-weight:700"' if total else ""
        parts.append(f'<text x="{left - 10}" y="{y + bar + 5}" text-anchor="end"{weight}>{name}</text>')
        parts.append(
            f'<rect class="b-base" x="{left}" y="{y}" width="{x(base) - left:.1f}" height="{bar}" rx="2"/>'
            f'<text class="m" x="{x(base) + 5:.1f}" y="{y + bar - 2}">{str(base).replace(".", t["decimal"])}</text>'
        )
        parts.append(
            f'<rect class="b-ours" x="{left}" y="{y + bar + 2}" width="{x(ours) - left:.1f}" height="{bar}" rx="2"/>'
            f'<text x="{x(ours) + 5:.1f}" y="{y + 2 * bar}">{str(ours).replace(".", t["decimal"])}</text>'
        )
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{t["wer_label"]}">{"".join(parts)}</svg>'


def build(lang: str) -> Path:
    t = TEXT[lang]
    body = (SRC / f"body.{lang}.html").read_text(encoding="utf-8")
    title, body = body.split("\n", 1)
    css = (SRC / "style.css").read_text(encoding="utf-8")
    judge = (PROJECT / "scripts" / "judge_dialect.py").read_text(encoding="utf-8")
    prompt = re.search(r'SYSTEM_PROMPT = """(.*?)"""', judge, re.S).group(1)
    body = re.sub(r"\{\{icon:([a-z-]+)\}\}", lambda m: icon(m.group(1)), body)
    body = body.replace("{{img:hero}}", "data:image/webp;base64," + b64(SRC / "assets" / "hero.webp"))
    body = body.replace("{{audio:palkane}}", "data:audio/mpeg;base64," + b64(SRC / "assets" / "palkane_0021.mp3"))
    body = body.replace("{{audio:heinola}}", "data:audio/mpeg;base64," + b64(SRC / "assets" / "heinola_0097.mp3"))
    body = body.replace("{{prompt}}", html.escape(prompt))
    body = body.replace("{{credits}}", CREDITS[lang])
    body = body.replace("{{chart:loss}}", loss_chart(t)).replace("{{chart:wer}}", wer_chart(t))
    left = re.findall(r"\{\{[^}]+\}\}", body)
    if left:
        raise SystemExit(f"{lang}: unfilled placeholders {left}")
    fonts = (
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Young+Serif'
        '&family=Source+Sans+3:ital,wght@0,400;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;600&display=swap">\n'
    )
    page = (
        f'<!doctype html>\n<html lang="{t["lang"]}">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        f"{title}\n{fonts}<style>\n{css}</style>\n</head>\n<body>\n{body}</body>\n</html>\n"
    )
    out = OUT / f"story.{lang}.html"
    out.write_text(page, encoding="utf-8", newline="\n")
    return out


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    for lang in ("fi", "en"):
        out = build(lang)
        print(f"wrote {out.relative_to(PROJECT)}: {out.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
