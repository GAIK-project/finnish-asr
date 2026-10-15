"""The listening page: short real recordings, and what each model wrote, side by side.

    uv run --extra hub python scripts/build_listen.py

Writes docs/listen.html (served at https://gaik-project.github.io/finnish-asr/listen.html) and
docs/listen/*.mp3. Two parts:

- dialect speech: a few SKN test segments (municipalities kept out of our training). Their audio is
  cut from the published dataset's test split (CC BY 4.0, Kotus via Kielipankki); each model's text
  comes from results/benchmarks/skn_fi/<arm>.jsonl, with the words that word error rate counts as
  wrong marked;
- standard spoken Finnish: spans of the Ajokortti webinar clip, played straight from gaik-toolkit (the
  clip is not copied here), with each model's text for the span as stored in results/transcriptions/
  and the two judges' verdicts (Claude Opus 5.5, Gemini 3.1 Pro) from results/judgements/. The span texts are checked against those files.

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
CLIP_URL = (
    "https://raw.githubusercontent.com/GAIK-project/gaik-toolkit/main/"
    "evaluation_layer/eval_methods/transcription_eval/data/Ajokortti.mp3"
)

SEGMENTS = [  # id, place, English gloss
    ("SKN45a_Heinola_0097", "Heinola", "Even when I sometimes bring it something to drink, it doesn't bite, but it grimaces so you'd think it bites now."),
    ("SKN50a_Eurajoki_0203", "Eurajoki", ""),
    ("SKN25a_Alastaro_0156", "Alastaro", ""),
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

# Ajokortti spans: start, end (s), title, reference text (HY), rows (label, kind, transcript file, text, verdict)
SPANS = [
    (75.5, 83.7, "Four implant brands",
     "ja sen jälkeen siellä on neljä implanttibrändiä mukana. Straumann, Densply Sirona, Nobel Biocare ja Camlog-Conelog eli Implantona Oy.",
     "",
     [
         ("Gemini 3.5 Flash (Google)", "api", "qad1/vertex_gemini-3.5-flash.json",
          "Ja sen jälkeen siellä on neljä implanttibrändiä mukana: Straumann, Dentsply Sirona, Nobel Biocare ja Camlog Conelog eli Implantona Oy.",
          "Both judges: no meaning errors. Both also note that the reference misspells “Dentsply”; “Camlog Conelog” is how the listener-reviewed reference writes it, so these red words are correct."),
         ("Our model", "ours", "ft_whisper-large-v3-fi-skn-punct-best-chunks.json",
          "ja sen jälkeen siellä on neljä implanttibrändiä mukana, Straumann, Densplay-sirvana, Nobelbayer, ja Camlock-Connellock eli implanttoona Oy.",
          "Both judges: “Nobelbayer” is a major error (a different name). Claude rates the other brands minor (misspelled but recognisable); Gemini 3.1 Pro also rates “Densplay-sirvana” major."),
         ("Whisper large-v3 (OpenAI)", "open", "qad1/hf_whisper-large-v3.json",
          "Sen jälkeen siellä on neljä implanttebrändiä mukana, Straumanni, Dentsply Sirona, Noble Biocare ja Camlock Conelog, eli implanttoona OY.",
          "Claude: one minor error, the names misspelled but recognisable. Gemini 3.1 Pro: no error here."),
     ]),
    (8.3, 15.5, "A lecturer and a profession",
     "Sitten on parodontologi Martta Martolan luento perimplantiitista ja sitten Implantticaseja, semmoinen vodcast, jossa Peterin kanssa",
     "",
     [
         ("Gemini 3.5 Flash (Google)", "api", "qad1/vertex_gemini-3.5-flash.json",
          "Sitten on parodontologi Martta Martola luento peri-implantiitista. Ja sitten implanttikeissejä semmoinen podcast, jossa Petteri kanssa",
          "Both judges: “podcast” and “Petteri” are minor slips. “peri-implantiitista” is the correct spelling; both judges note that the reference misspells it."),
         ("Our model", "ours", "ft_whisper-large-v3-fi-skn-punct-best-chunks.json",
          "Sitten on Pardentolki Martta Marttojen luento periimplanttiitista ja sitten ImplanttiCase ja semmoinen vodcast, jossa Peterin kanssa",
          "Both judges: the surname “Marttojen” is a major error. “Pardentolki” (periodontist): minor for Claude, major for Gemini 3.1 Pro."),
         ("Whisper large-v3 (OpenAI)", "open", "qad1/hf_whisper-large-v3.json",
          "Sitten on parodontologi Martta Martola luento perimplantiitista ja sitten implanttikeissä ja semmoinen vodcast, jossa Peeterin kanssa",
          "“implanttikeissä” for “Implantticaseja”: minor for Claude, major for Gemini 3.1 Pro."),
     ]),
    (147.9, 151.8, "“Six to nine months”",
     "Kesto on noin kuusi-yhdeksän kuukautta, mutta nyt minä annan Peterille puheenvuoron.",
     "A listener hears “kuusi viiva yhdeksän” and “mä” here; the reference writes “kuusi-yhdeksän” and “minä”.",
     [
         ("Gemini 3.5 Flash (Google)", "api", "qad1/vertex_gemini-3.5-flash.json",
          "Kesto noin 6-9 kuukautta. Mutta nyt mä annan Petterille puheenvuoron.",
          "Neither judge counts digits or the spoken “mä” as errors; WER does."),
         ("Our model", "ours", "ft_whisper-large-v3-fi-skn-punct-best-chunks.json",
          "Kesto on noin kuusi vai yhdeksän kuukautta, mutta nyt minä annan Peterille puheenvuoron,",
          "“kuusi vai yhdeksän” (six or nine) for “kuusi-yhdeksän” (six to nine): minor for Claude, major for Gemini 3.1 Pro."),
         ("Whisper large-v3 (OpenAI)", "open", "qad1/hf_whisper-large-v3.json",
          "Kesto on noin 6-9 kuukautta, mutta nyt mä annan Peterille puheenvuoron.",
          "No meaning error for either judge."),
     ]),
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


def span_card(start: float, end: float, title: str, ref: str, note: str, rows: list) -> str:
    reference = (PROJECT_ROOT / "data" / "ajokortti" / "reference.txt").read_text(encoding="utf-8")
    if ref not in reference:
        raise SystemExit(f"span reference not in reference.txt: {ref[:60]}")
    out = [f'<div class="card"><h3>{escape(title)} <span>· Ajokortti webinar, {start:.1f}–{end:.1f} s</span></h3>',
           f'<audio controls preload="none" src="{CLIP_URL}#t={start},{end}"></audio>',
           f'<div class="ref"><span class="k">Reference (University of Helsinki transcript)</span>{escape(ref)}'
           + (f'<div class="verdict">{escape(note)}</div>' if note else "") + "</div>"]
    for label, kind, path, hyp, verdict in rows:
        stored = json.loads((PROJECT_ROOT / "results" / "transcriptions" / path).read_text(encoding="utf-8"))["text"]
        if hyp not in stored:
            raise SystemExit(f"span text not in {path}: {hyp[:60]}")
        html, errs, n = marked_html(ref, hyp)
        out.append(f'<div class="row {kind}"><span class="k">{escape(label)}<span class="score">{errs} of {n} words differ</span></span>'
                   f'{html}<div class="verdict">{escape(verdict)}</div></div>')
    out.append("</div>")
    return "\n".join(out)


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    cut_mp3s([s for s, _, _ in SEGMENTS])
    body = [
        "<h1>Listen and compare</h1>",
        '<p class="lede">Short real recordings, and what each model wrote. Red words are wrong or extra; the count also includes words the model left out. '
        "Our model is a Whisper large-v3 fine-tuned on Finnish dialect interviews by the GAIK project.</p>",
        '<nav><a href="https://github.com/GAIK-project/finnish-asr">Code and results</a>'
        '<a href="story/story.en.html">The story (English)</a><a href="story/story.fi.html">Tarina (suomeksi)</a>'
        '<a href="https://huggingface.co/GAIK-project/whisper-large-v3-fi-dialects">The model</a></nav>',
        "<h2>Dialect speech</h2>",
        '<p class="note">Interviews recorded in the 1960s–70s, from municipalities kept out of our training. The reference is a '
        "person's transcript in standard Finnish word forms, so a model is right when it hears the dialect and writes standard Finnish.</p>",
        *[dialect_card(*s) for s in SEGMENTS],
        "<h2>Standard spoken Finnish: a dental webinar</h2>",
        '<p class="note">The Ajokortti clip: 180 s of a public dental-education webinar with many names and brands. Here word errors '
        "and meaning errors part ways: a model can spell many words differently and still keep the meaning, or miss one name and change it. "
        "Meaning errors were rated by two LLM judges, Claude Opus 5.5 and Gemini 3.1 Pro, who read the reference and each "
        "transcript without knowing which model wrote it (minor, major or critical).</p>",
        *[span_card(*s) for s in SPANS],
        '<footer>Dialect audio: Suomen kielen näytteitä (SKN), Kotus, via <a href="http://urn.fi/urn:nbn:fi:lb-2020112937">Kielipankki</a>, '
        'CC BY 4.0, from the test split of <a href="https://huggingface.co/datasets/GAIK-project/skn-finnish-dialects-asr">'
        'GAIK-project/skn-finnish-dialects-asr</a>. Webinar: QAdental (Suomen Kotilääkäripalvelu Oy), the Ajokortti clip in '
        '<a href="https://github.com/GAIK-project/gaik-toolkit/tree/main/evaluation_layer/eval_methods/transcription_eval">gaik-toolkit</a>, '
        "reference transcript by the University of Helsinki. Built by scripts/build_listen.py from the repository's results. "
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
