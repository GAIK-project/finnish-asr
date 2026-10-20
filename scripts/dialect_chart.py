r"""README chart: how much the dialect fine-tune helps, on two dialect corpora.

    uv run python scripts/dialect_chart.py

Draws docs/img/dialect-finetune.svg, two panels of horizontal bars, lower is better:

- left: word error rate on the SKN dialect test (600 segments, 10 municipalities held out of
  training), from results/benchmarks/skn_fi/scores.json;
- right: meaning-error points per 100 words on the LA-murre test (10 localities in a corpus the
  SKN models never saw), from the blind judge's round 2, results/judgements_dialect/lamurre_fi/
  summary-r2.json. Its arms were judged together in one call per segment, so they share a scale.
  The LA-murre model (the la-murre revision) was trained on LA-murre recordings from other
  localities; the 10 test ones are held out of all training.

Static SVG with text as text, no matplotlib and no keys.
"""

from __future__ import annotations

import argparse
import json
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fi_common import PROJECT_ROOT, RESULTS_DIR  # noqa: E402

OUT = PROJECT_ROOT / "docs" / "img" / "dialect-finetune.svg"
# arm -> (label, ours)
ARMS = {
    "hf_whisper-large-finnish-v3": ("Finnish-NLP whisper-large-finnish-v3", False),
    "hf_whisper-large-v3": ("Whisper large-v3", False),
    "ft_whisper-large-v3-fi-skn": ("Ours, round 1", True),
    "ft_whisper-large-v3-fi-skn-punct-best": ("Ours, round 2 (published)", True),
    "ft_whisper-large-v3-fi-skn-lamurre-best": ("Ours, round 2 + LA-murre", True),
}
W, ROW, TOP, LABEL_W, PANEL_W, GAP = 1000, 34, 92, 270, 290, 60
INK, MUTED, BASE, OURS, GRID = "#1f2328", "#59636e", "#9aa4ae", "#2f6fb3", "#d8dee4"


def panel(x0: float, title: str, sub: str, values: dict[str, float], vmax: float) -> list[str]:
    out = [
        f'<text x="{x0}" y="{TOP - 46}" font-size="15" font-weight="600" fill="{INK}">{escape(title)}</text>',
        f'<text x="{x0}" y="{TOP - 26}" font-size="12" fill="{MUTED}">{escape(sub)}</text>',
    ]
    for tick in range(0, int(vmax) + 1, 10):
        x = x0 + PANEL_W * tick / vmax
        out.append(f'<line x1="{x:.1f}" y1="{TOP - 8}" x2="{x:.1f}" y2="{TOP + ROW * len(ARMS)}" stroke="{GRID}"/>')
        out.append(
            f'<text x="{x:.1f}" y="{TOP + ROW * len(ARMS) + 16}" font-size="11" fill="{MUTED}" '
            f'text-anchor="middle">{tick}</text>'
        )
    for i, arm in enumerate(ARMS):
        y = TOP + i * ROW
        if arm not in values:
            out.append(f'<text x="{x0}" y="{y + 19}" font-size="12" fill="{MUTED}">not judged in this round</text>')
            continue
        v, colour = values[arm], OURS if ARMS[arm][1] else BASE
        w = PANEL_W * v / vmax
        out.append(f'<rect x="{x0}" y="{y + 6}" width="{w:.1f}" height="{ROW - 12}" rx="3" fill="{colour}"/>')
        out.append(f'<text x="{x0 + w + 6:.1f}" y="{y + 22}" font-size="12" fill="{INK}">{v:.1f}</text>')
    return out


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    skn = json.loads((RESULTS_DIR / "benchmarks" / "skn_fi" / "scores.json").read_text(encoding="utf-8"))
    wer = {a["arm_id"]: a["WER"] for a in skn["arms"] if a["arm_id"] in ARMS}
    judge_path = RESULTS_DIR / "judgements_dialect" / "lamurre_fi" / "summary-r2.json"
    judged = json.loads(judge_path.read_text(encoding="utf-8"))
    pts = {a: s["error_points_per_100w"] for a, s in judged["arms"].items() if a in ARMS}
    missing = [a for a in ARMS if a not in wer]
    if missing:
        raise SystemExit(f"no SKN score for {missing}")

    height = TOP + ROW * len(ARMS) + 64
    x1 = 20 + LABEL_W
    x2 = x1 + PANEL_W + GAP
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" height="{height}" '
        'font-family="system-ui, -apple-system, Segoe UI, sans-serif" role="img" '
        'aria-label="Dialect fine-tune: SKN word error rate and LA-murre meaning errors, lower is better">',
        f'<rect width="{W}" height="{height}" fill="#ffffff"/>',
    ]
    for i, (label, ours) in enumerate(ARMS.values()):
        y = TOP + i * ROW
        weight = "600" if ours else "400"
        svg.append(
            f'<text x="{x1 - 12}" y="{y + 22}" font-size="13" font-weight="{weight}" fill="{INK}" '
            f'text-anchor="end">{escape(label)}</text>'
        )
    svg += panel(x1, "SKN dialect test: word error rate %", "600 segments, 10 held-out municipalities", wer, 60)
    svg += panel(
        x2,
        "LA-murre: meaning-error points / 100 words",
        f"{judged['segments']} segments, 10 localities held out of all training, blind judge",
        pts,
        30,
    )
    svg.append(
        f'<text x="20" y="{height - 14}" font-size="11" fill="{MUTED}">Lower is better. Blue: this project\'s '
        "LoRA fine-tunes of Whisper large-v3. The LA-murre model was trained on other LA-murre localities; "
        "Finnish-NLP was not in this judge round.</text>"
    )
    svg.append("</svg>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(svg) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(PROJECT_ROOT)}: SKN {wer}, LA-murre {pts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
