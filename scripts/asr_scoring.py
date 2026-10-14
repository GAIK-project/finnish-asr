"""WER and CER for the Ajokortti clip, with the University of Helsinki (HY) team's normalisation.

The scorer of QAD-1, GAIK's re-run of HY's evaluation of this clip (in GAIK's internal evaluation
repository), copied here unchanged in the parts this repository uses, so that its numbers match
QAD-1's and HY's published ones (gaik-toolkit, evaluation_layer/eval_methods/transcription_eval).

Normalisation, as in HY's error_analysis.py: jiwer with lower-casing, removal of every Unicode
punctuation character (hyphens included), collapsing of runs of two or more whitespace characters,
strip, and a split on the space character. WER, SUB, DEL and INS are percentages of reference words;
CER counts characters including spaces. spelling_rate is HY's "spelling error" rate: substitutions
whose Levenshtein distance / max(len) <= 0.4. CER_nospace also deletes all whitespace, so compound
splits ("implantti keissä" vs "implanttikeissä") cost nothing.

A quirk inherited from HY: RemoveMultipleSpaces only collapses runs of two or more whitespace
characters and the split is on " " alone, so a single line break glues two words into one token. The
main metrics keep that behaviour, because they must match HY's numbers; whitespace_merge() reports
the glued tokens and the WER/CER with line breaks read as spaces.

The numbers in results/ were made with jiwer 4.0.0.
"""

from __future__ import annotations

from typing import Any

import jiwer
from rapidfuzz.distance import Levenshtein

SPELLING_MAX_NORM_DISTANCE = 0.4  # HY's threshold

NORMALIZATION = (
    "HY error_analysis.py: jiwer ToLowerCase, RemovePunctuation (every Unicode P* character, hyphens included), "
    "RemoveMultipleSpaces (runs of 2+ whitespace -> one space), Strip, split on ' '. CER uses the same steps with a "
    "character split (spaces count). CER_nospace also removes all whitespace. A single line break is not collapsed, "
    "so it glues two words into one token, as in HY's scorer; see each arm's whitespace_merge."
)


def word_transform() -> jiwer.Compose:
    return jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )


def char_transform() -> jiwer.Compose:
    return jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfChars(),
        ]
    )


def _nospace_char_transform() -> jiwer.Compose:
    return jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveWhiteSpace(replace_by_space=False),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfChars(),
        ]
    )


def ws_fixed_transform(chars: bool) -> jiwer.Compose:
    """HY's transform, except that every whitespace character counts as a space."""
    return jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveWhiteSpace(replace_by_space=True),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfChars() if chars else jiwer.ReduceToListOfListOfWords(),
        ]
    )


_ws_fixed_transform = ws_fixed_transform  # QAD-1's name


def tokens(text: str) -> list[str]:
    """Reference/hypothesis tokens under HY's word normalisation."""
    return word_transform()(text)[0]


def is_spelling_error(ref_word: str, hyp_word: str) -> bool:
    max_len = max(len(ref_word), len(hyp_word))
    if max_len == 0:
        return False
    return Levenshtein.distance(ref_word, hyp_word) / max_len <= SPELLING_MAX_NORM_DISTANCE


def alignment_rows(ref_words: list[str], hyp_words: list[str], chunks: list[Any]) -> list[dict[str, Any]]:
    """jiwer alignment chunks -> one row per word operation, in order.

    An "ins" row carries the index of the next reference token as ref_idx.
    """
    rows: list[dict[str, Any]] = []
    for c in chunks:
        r = ref_words[c.ref_start_idx : c.ref_end_idx]
        h = hyp_words[c.hyp_start_idx : c.hyp_end_idx]
        if c.type == "equal":
            pairs = enumerate(zip(r, h, strict=True))
            rows += [{"op": "equal", "ref": a, "hyp": b, "ref_idx": c.ref_start_idx + i} for i, (a, b) in pairs]
        elif c.type == "substitute":
            m = min(len(r), len(h))
            rows += [{"op": "sub", "ref": r[i], "hyp": h[i], "ref_idx": c.ref_start_idx + i} for i in range(m)]
            rows += [{"op": "del", "ref": r[i], "hyp": None, "ref_idx": c.ref_start_idx + i} for i in range(m, len(r))]
            rows += [{"op": "ins", "ref": None, "hyp": h[i], "ref_idx": c.ref_end_idx} for i in range(m, len(h))]
        elif c.type == "delete":
            rows += [{"op": "del", "ref": a, "hyp": None, "ref_idx": c.ref_start_idx + i} for i, a in enumerate(r)]
        elif c.type == "insert":
            rows += [{"op": "ins", "ref": None, "hyp": b, "ref_idx": c.ref_start_idx} for b in h]
        else:  # pragma: no cover - jiwer has only the four types
            raise ValueError(f"unknown alignment chunk type {c.type!r}")
    return rows


def word_alignment(ref_text: str, hyp_text: str) -> tuple[Any, list[dict[str, Any]]]:
    """(jiwer WordOutput, alignment rows) under HY's normalisation."""
    out = jiwer.process_words(ref_text, hyp_text, word_transform(), word_transform())
    return out, alignment_rows(out.references[0], out.hypotheses[0], out.alignments[0])


def _cer(ref_text: str, hyp_text: str, transform: jiwer.Compose) -> float:
    out = jiwer.process_characters(ref_text, hyp_text, transform, transform)
    ref_chars = out.hits + out.substitutions + out.deletions
    return (out.substitutions + out.deletions + out.insertions) / ref_chars if ref_chars else 0.0


def asr_metrics(ref_text: str, hyp_text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """HY-compatible metrics plus the word alignment rows. Arithmetic mirrors HY's evaluate_model()."""
    word_out, rows = word_alignment(ref_text, hyp_text)
    ref_words = word_out.hits + word_out.substitutions + word_out.deletions
    spelling = sum(1 for row in rows if row["op"] == "sub" and is_spelling_error(row["ref"], row["hyp"]))
    wer = (word_out.substitutions + word_out.deletions + word_out.insertions) / ref_words
    metrics = {
        "WER": round(wer * 100, 2),
        "CER": round(_cer(ref_text, hyp_text, char_transform()) * 100, 2),
        "SUB": round(word_out.substitutions / ref_words * 100, 2),
        "DEL": round(word_out.deletions / ref_words * 100, 2),
        "INS": round(word_out.insertions / ref_words * 100, 2),
        "spelling_rate": round(spelling / ref_words * 100, 2),
        "CER_nospace": round(_cer(ref_text, hyp_text, _nospace_char_transform()) * 100, 2),
        "hits": word_out.hits,
        "substitutions": word_out.substitutions,
        "deletions": word_out.deletions,
        "insertions": word_out.insertions,
        "ref_words": ref_words,
        "hyp_words": len(word_out.hypotheses[0]),
    }
    return metrics, rows


def whitespace_merge(ref_text: str, hyp_text: str) -> dict[str, Any] | None:
    """Tokens that HY's split glues across a single line break, and the metrics without that."""
    glued = [t for t in tokens(hyp_text) + tokens(ref_text) if any(ch.isspace() for ch in t)]
    if not glued:
        return None
    wt, ct = ws_fixed_transform(chars=False), ws_fixed_transform(chars=True)
    out = jiwer.process_words(ref_text, hyp_text, wt, wt)
    ref_words = out.hits + out.substitutions + out.deletions
    wer = (out.substitutions + out.deletions + out.insertions) / ref_words
    return {
        "merged_tokens": glued,
        "WER": round(wer * 100, 2),
        "CER": round(_cer(ref_text, hyp_text, ct) * 100, 2),
        "note": "WER/CER with every whitespace character read as a space; the main metrics keep HY's behaviour.",
    }
