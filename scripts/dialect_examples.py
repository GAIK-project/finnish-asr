r"""Side-by-side examples from the SKN dialect test: the human transcript, Whisper large-v3 and our model.

    uv run python scripts/dialect_examples.py

Writes docs/DIALECT-EXAMPLES.md from results/benchmarks/skn_fi/*.jsonl, no keys and no GPU. The
examples are picked by a fixed rule, not by result: per municipality, of the segments with 12-40
reference words in id order, the ones one third and two thirds of the way through.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fi_common import PROJECT_ROOT, RESULTS_DIR  # noqa: E402
from score_benchmark import transform  # noqa: E402

OUT = PROJECT_ROOT / "docs" / "DIALECT-EXAMPLES.md"
BENCH = RESULTS_DIR / "benchmarks" / "skn_fi"
BASE, OURS = "hf_whisper-large-v3", "ft_whisper-large-v3-fi-skn-punct-best"
HF_MODEL = "https://huggingface.co/GAIK-project/whisper-large-v3-fi-dialects"
HF_DATA = "https://huggingface.co/datasets/GAIK-project/skn-finnish-dialects-asr"


def load(arm: str) -> dict[str, dict]:
    lines = (BENCH / f"{arm}.jsonl").read_text(encoding="utf-8").splitlines()
    return {r["id"]: r for r in map(json.loads, lines)}


def errors(refs: list[str], hyps: list[str]) -> tuple[int, int]:
    hyps = [h if transform(False)(h)[0] else "-" for h in hyps]  # jiwer rejects an empty hypothesis
    w = jiwer.process_words(refs, hyps, transform(False), transform(False))
    return w.substitutions + w.deletions + w.insertions, w.hits + w.substitutions + w.deletions


def by_place(rows: dict[str, dict]) -> dict[str, list[str]]:
    """Municipality -> its segment ids, in id order."""
    places: dict[str, list[str]] = {}
    for seg in sorted(rows):
        places.setdefault(re.match(r"SKN\d+[a-z]?_(.+)_\d+\.wav", seg).group(1), []).append(seg)
    return places


def picks(rows: dict[str, dict], segs: list[str]) -> tuple[str, str]:
    """The fixed example rule: of the segments with 12-40 reference words, the ones 1/3 and 2/3 through."""
    mid = [s for s in segs if 12 <= len(rows[s]["ref"].split()) <= 40]
    return mid[len(mid) // 3], mid[2 * len(mid) // 3]


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    base, ours = load(BASE), load(OURS)
    places = by_place(base)

    out = [
        "# Dialect test, side by side",
        "",
        f"What the models write for the same audio: segments from the SKN dialect test ([dataset]({HF_DATA}),",
        "split `test`), 10 municipalities that are not in the training data. **Human** is SKN's transcript in",
        "standard Finnish word forms; **Whisper** is plain openai/whisper-large-v3; **Ours** is the",
        f"[published model]({HF_MODEL}).",
        "The numbers are word errors / reference words after lower-casing and removing punctuation.",
        "",
        "The examples are picked by position, not by result: per municipality, of the segments with 12-40",
        "reference words in id order, the ones one third and two thirds of the way through. Find a segment's",
        "audio and dialect-spelling transcript in the dataset viewer by its id. Written by",
        "[`dialect_examples.py`](../scripts/dialect_examples.py).",
        "",
        "| Municipality | Segments | Whisper large-v3 WER % | Ours WER % |",
        "|---|--:|--:|--:|",
    ]
    examples = []
    for place, segs in sorted(places.items()):
        refs = [base[s]["ref"] for s in segs]
        eb, n = errors(refs, [base[s]["hyp"] for s in segs])
        eo, _ = errors(refs, [ours[s]["hyp"] for s in segs])
        out.append(f"| {place} | {len(segs)} | {eb / n * 100:.1f} | {eo / n * 100:.1f} |")
        for seg in picks(base, segs):
            ref = base[seg]["ref"]
            b, words = errors([ref], [base[seg]["hyp"]])
            o, _ = errors([ref], [ours[seg]["hyp"]])
            examples += [
                f"**{place}** · `{seg.removesuffix('.wav')}`",
                "",
                "| | Text | Errors |",
                "|---|---|--:|",
                f"| Human | {ref} | |",
                f"| Whisper | {base[seg]['hyp'].strip()} | {b} / {words} |",
                f"| Ours | {ours[seg]['hyp'].strip()} | {o} / {words} |",
                "",
            ]
    all_refs = [base[s]["ref"] for s in sorted(base)]
    eb, n = errors(all_refs, [base[s]["hyp"] for s in sorted(base)])
    eo, _ = errors(all_refs, [ours[s]["hyp"] for s in sorted(base)])
    out.append(f"| **All** | {len(base)} | **{eb / n * 100:.1f}** | **{eo / n * 100:.1f}** |")
    out += ["", "## Examples", ""] + examples
    out += [
        "Source: Suomen kielen näytteitä, Kotimaisten kielten keskus (Kotus), via Kielipankki – The Language",
        "Bank of Finland, CC BY 4.0.",
    ]
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(PROJECT_ROOT)}: {len(examples) // 8} examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
