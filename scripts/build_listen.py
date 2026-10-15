"""The listening page: short real recordings, and what each model wrote, side by side.

    uv run --extra hub python scripts/build_listen.py

Writes docs/listen.html (served at https://gaik-project.github.io/finnish-asr/listen.html) and
docs/listen/*.mp3: a few SKN dialect test segments (municipalities kept out of our training), their
audio cut from the published dataset's test split (CC BY 4.0, Kotus via Kielipankki), and each model's
text from results/benchmarks/skn_fi/<arm>.jsonl with the words that word error rate counts as wrong marked.

No keys. Needs ffmpeg on PATH for the mp3s.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fi_common import PROJECT_ROOT  # noqa: E402
from readme_figures import marked  # noqa: E402

DOCS = PROJECT_ROOT / "docs"
OUT, AUDIO = DOCS / "listen.html", DOCS / "listen"
BENCH = PROJECT_ROOT / "results" / "benchmarks" / "skn_fi"
DATASET, REVISION = "GAIK-project/skn-finnish-dialects-asr", "ee98f5e16dcb0986a5514b787bee100962be3bf6"
SEGMENTS = [  # id, place, English gloss
    ("SKN45a_Heinola_0097", "Heinola", "Even when I sometimes bring it something to drink, it doesn't bite, but it grimaces so you'd think it bites now."),
    ("SKN50a_Eurajoki_0203", "Eurajoki", ""),
    ("SKN25a_Alastaro_0156", "Alastaro", ""),
    ("SKN15a_Rautu_0072", "Rautu", "The bus went in the mornings, and in the evenings you could already come back on the same bus."),
    ("SKN20b_Ikaalinen_0245", "Ikaalinen", "And they must be different medicines, then; they are not like the medicines for people."),
    ("SKN35a_Artjarvi_0081", "Artjärvi", ""),
]
DIALECT_ARMS = [  # arm, label, kind
    ("hf_whisper-large-v3", "Whisper large-v3 (OpenAI), as it is", "open"),
    ("vertex_gemini-3.5-flash", "Gemini 3.5 Flash (Google), asked to write as spoken", "api"),
    ("vertex_gemini-3.5-flash-std", "Gemini 3.5 Flash, asked for standard Finnish", "api"),
    ("google_chirp-3", "Google Chirp 3", "api"),
    ("azure_gpt-4o-transcribe", "OpenAI gpt-4o-transcribe", "api"),
    ("elevenlabs_scribe-v2", "ElevenLabs Scribe v2", "api"),
    ("ft_whisper-large-v3-fi-skn-punct-best", "Our model", "ours"),
]

CSS = """
:root { --bg:#f6f7f4; --surface:#fff; --ink:#1d2422; --muted:#5b6662; --line:#d9dfdb; --ours:#2f6fb3; --ours-soft:#e8f0fa;
  --api:#8a5a2b; --bad:#b3402e; --ok:#3f8f5b; }
@media (prefers-color-scheme: dark) { :root { --bg:#141917; --surface:#1c2320; --ink:#e7ece9; --muted:#a3afaa; --line:#2f3a36;
  --ours:#7fb2e5; --ours-soft:#1d2b3a; --api:#e0a86f; --bad:#f08b7a; --ok:#7cc4a8; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width: 860px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 30px; line-height:1.15; margin: 0 0 8px; } h2 { font-size: 22px; margin: 40px 0 6px; }
p.lede, p.note { color: var(--muted); margin: 0 0 16px; }
nav a { margin-right: 14px; color: var(--ours); }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 16px; margin: 16px 0; }
.card h3 { margin: 0 0 8px; font-size: 17px; } .card h3 span { color: var(--muted); font-weight: 400; font-size: 14px; }
audio { width: 100%; margin: 4px 0 10px; }
.ref { background: var(--bg); border-radius: 8px; padding: 8px 12px; margin-bottom: 10px; }
.k { display:block; font-size: 13px; font-weight: 700; color: var(--muted); }
.row { border-top: 1px solid var(--line); padding: 8px 0; } .row.ours { background: var(--ours-soft); margin: 0 -16px; padding: 8px 16px; }
.row .k { display:flex; justify-content: space-between; gap: 12px; } .row.ours .k { color: var(--ours); }
.row.api .k { color: var(--api); }
mark { background: none; color: var(--bad); font-weight: 700; }
.score { white-space: nowrap; } .score.good { color: var(--ok); } .score.bad { color: var(--bad); }
.verdict { font-size: 14px; color: var(--muted); margin-top: 2px; }
.en { font-size: 14px; color: var(--muted); font-style: italic; margin: 6px 0 0; }
footer { margin-top: 48px; font-size: 13px; color: var(--muted); }
"""


def marked_html(ref: str, hyp: str) -> tuple[str, int, int]:
    parts, errs, n = marked(ref, hyp)
    return "".join(f"<mark>{escape(t.strip())}</mark>{' ' if t.endswith(' ') else ''}" if bad else escape(t) for t, bad in parts), errs, n


def cut_mp3s(ids: list[str]) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, "data/test-00000-of-00001.parquet", repo_type="dataset", revision=REVISION)
    rows = {r["id"]: r for r in pq.read_table(path, columns=["id", "audio"]).to_pylist() if r["id"] in ids}
    AUDIO.mkdir(parents=True, exist_ok=True)
    for seg in ids:
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "in.wav"
            wav.write_bytes(rows[seg]["audio"]["bytes"])
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-ac", "1", "-b:a", "64k", str(AUDIO / f"{seg}.mp3")],
                           check=True)  # fmt: skip


def dialect_card(seg: str, place: str, gloss: str) -> str:
    rows = {}
    for arm, _, _ in DIALECT_ARMS:
        path = BENCH / f"{arm}.jsonl"
        if path.exists():
            rows[arm] = next((json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if json.loads(x)["id"] == seg + ".wav"), None)
    ref = rows["hf_whisper-large-v3"]["ref"]
    out = [f'<div class="card"><h3>{escape(place)} <span>· SKN test segment {seg}</span></h3>',
           f'<audio controls preload="none" src="listen/{seg}.mp3"></audio>',
           f'<div class="ref"><span class="k">What was said (a person\'s standard-Finnish transcript)</span>{escape(ref)}</div>']
    for arm, label, kind in DIALECT_ARMS:
        row = rows.get(arm)
        if not row:
            continue
        html, errs, n = marked_html(ref, row["hyp"])
        good = errs <= n * 0.15
        out.append(f'<div class="row {kind}"><span class="k">{escape(label)}<span class="score {"good" if good else "bad"}">'
                   f'{errs} of {n} words wrong</span></span>{html}</div>')
    if gloss:
        out.append(f'<p class="en">In English: “{escape(gloss)}”</p>')
    out.append("</div>")
    return "\n".join(out)


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    cut_mp3s([s for s, _, _ in SEGMENTS])
    body = [
        "<h1>Listen and compare</h1>",
        '<p class="lede">Short real recordings, and what each model wrote. Red words are wrong or extra; the count also includes words the model left out. '
        "Our model is a Whisper large-v3 fine-tuned on Finnish dialect interviews by the GAIK project. The examples were picked to show where the models differ on dialect speech; every model's full scores, standard Finnish included, are in the <a href=\"https://github.com/GAIK-project/finnish-asr\">repository</a>.</p>",
        '<nav><a href="https://github.com/GAIK-project/finnish-asr">Code and results</a>'
        '<a href="story/story.en.html">The story (English)</a><a href="story/story.fi.html">Tarina (suomeksi)</a>'
        '<a href="https://huggingface.co/GAIK-project/whisper-large-v3-fi-dialects">The model</a></nav>',
        "<h2>Dialect speech</h2>",
        '<p class="note">Interviews recorded in the 1960s–70s, from municipalities kept out of our training. The reference is a '
        "person's transcript in standard Finnish word forms, so a model is right when it hears the dialect and writes standard Finnish.</p>",
        *[dialect_card(*s) for s in SEGMENTS],
        '<footer>Dialect audio: Suomen kielen näytteitä (SKN), Kotus, via <a href="http://urn.fi/urn:nbn:fi:lb-2020112937">Kielipankki</a>, '
        'CC BY 4.0, from the test split of <a href="https://huggingface.co/datasets/GAIK-project/skn-finnish-dialects-asr">'
        'GAIK-project/skn-finnish-dialects-asr</a>.'
        "Built by scripts/build_listen.py from the repository's results. "
        'GAIK project, Haaga-Helia University of Applied Sciences, co-funded by the European Union (ERDF).</footer>',
    ]
    html = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Listen and compare · Finnish dialect speech-to-text</title>\n<style>{CSS}</style>\n</head>\n<body>\n<main>\n"
        + "\n".join(body)
        + "\n</main>\n</body>\n</html>\n"
    )
    OUT.write_text(html, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(PROJECT_ROOT)} and {len(SEGMENTS)} mp3s in {AUDIO.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
