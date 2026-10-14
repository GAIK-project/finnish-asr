"""Timestamp targets for the fine-tune, SKN sentence parts, and SKN's annotation reader."""

import re

import finetune_whisper as ft
import prepare_skn
import prepare_timestamps as pt

STAMP = re.compile(r"<\|(\d+\.\d\d)\|>")


def test_timestamped_puts_one_pair_per_sentence_on_the_whisper_grid() -> None:
    out = ft.timestamped("Minä menin kotiin. Ja söin leipää.", [[0.0, 1.234, 3], [1.5, 4.0, 3]])
    assert out == "<|0.00|> Minä menin kotiin.<|1.24|><|1.50|> Ja söin leipää.<|4.00|>"
    times = [float(t) for t in STAMP.findall(out)]
    assert times == sorted(times)
    assert all(abs(t / 0.02 - round(t / 0.02)) < 1e-6 for t in times)


def test_timestamped_keeps_a_dash_with_the_word_before() -> None:
    out = ft.timestamped("Minä - menin", [[0.0, 1.0, 1], [1.2, 2.0, 1]])
    assert out == "<|0.00|> Minä -<|1.00|><|1.20|> menin<|2.00|>"


def test_timestamped_refuses_when_the_words_do_not_line_up() -> None:
    assert ft.timestamped("yksi kaksi kolme", [[0.0, 1.0, 2], [1.0, 2.0, 2]]) is None
    assert ft.timestamped("yksi kaksi", None) is None
    assert ft.timestamped("yksi kaksi", []) is None


def test_timestamped_never_goes_backwards_or_collapses_a_pair() -> None:
    out = ft.timestamped("a b c", [[0.0, 1.0, 1], [0.9, 0.9, 1], [2.0, 3.0, 1]])
    times = [float(t) for t in STAMP.findall(out)]
    assert times == sorted(times)
    starts, ends = times[0::2], times[1::2]
    assert all(e > s for s, e in zip(starts, ends, strict=True))


def test_merged_parts_merge_overlapping_sentences_and_clip_to_the_window() -> None:
    sents = [(0.0, 1.0, "a b"), (0.5, 2.0, "c"), (3.0, 4.0, "d e f")]
    assert pt.merged_parts(sents, 0.0, 5.0) == [[0.0, 2.0, 3], [3.0, 4.0, 3]]
    assert pt.merged_parts([(0.5, 1.5, "a"), (2.5, 9.0, "b")], 1.0, 2.0) == [[0.0, 0.5, 1], [1.5, 2.0, 1]]
    assert pt.words("yksi - kaksi") == 2


EAF = """<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT>
  <TIME_ORDER>
    <TIME_SLOT TIME_SLOT_ID="n1" TIME_VALUE="0"/><TIME_SLOT TIME_SLOT_ID="n2" TIME_VALUE="1500"/>
    <TIME_SLOT TIME_SLOT_ID="n3" TIME_VALUE="2000"/><TIME_SLOT TIME_SLOT_ID="n4" TIME_VALUE="4000"/>
    <TIME_SLOT TIME_SLOT_ID="n5" TIME_VALUE="10000"/><TIME_SLOT TIME_SLOT_ID="n6" TIME_VALUE="13000"/>
    <TIME_SLOT TIME_SLOT_ID="d1" TIME_VALUE="0"/><TIME_SLOT TIME_SLOT_ID="d2" TIME_VALUE="1500"/>
    <TIME_SLOT TIME_SLOT_ID="d3" TIME_VALUE="2000"/><TIME_SLOT TIME_SLOT_ID="d4" TIME_VALUE="4000"/>
  </TIME_ORDER>
  <TIER TIER_ID="AA-sentence-norm">
    <ANNOTATION><ALIGNABLE_ANNOTATION TIME_SLOT_REF1="n1" TIME_SLOT_REF2="n2">
      <ANNOTATION_VALUE>minä menin kotiin</ANNOTATION_VALUE></ALIGNABLE_ANNOTATION></ANNOTATION>
    <ANNOTATION><ALIGNABLE_ANNOTATION TIME_SLOT_REF1="n3" TIME_SLOT_REF2="n4">
      <ANNOTATION_VALUE>ja söin  leipää</ANNOTATION_VALUE></ALIGNABLE_ANNOTATION></ANNOTATION>
    <ANNOTATION><ALIGNABLE_ANNOTATION TIME_SLOT_REF1="n5" TIME_SLOT_REF2="n6">
      <ANNOTATION_VALUE>se oli hyvä päivä</ANNOTATION_VALUE></ALIGNABLE_ANNOTATION></ANNOTATION>
  </TIER>
  <TIER TIER_ID="AA-sentence">
    <ANNOTATION><ALIGNABLE_ANNOTATION TIME_SLOT_REF1="d1" TIME_SLOT_REF2="d2">
      <ANNOTATION_VALUE>mie mänin kotii</ANNOTATION_VALUE></ALIGNABLE_ANNOTATION></ANNOTATION>
    <ANNOTATION><ALIGNABLE_ANNOTATION TIME_SLOT_REF1="d3" TIME_SLOT_REF2="d4">
      <ANNOTATION_VALUE>ja söin leipää</ANNOTATION_VALUE></ALIGNABLE_ANNOTATION></ANNOTATION>
  </TIER>
</ANNOTATION_DOCUMENT>
"""  # invented sentences in SKN's tier layout: each tier has its own time-slot ids


def test_sentences_match_the_dialect_tier_by_time_value_not_by_slot_id() -> None:
    sents = prepare_skn.sentences(EAF.encode("utf-8"))
    assert [(s["start"], s["end"], s["ref"], s["orig"]) for s in sents] == [
        (0.0, 1.5, "minä menin kotiin", "mie mänin kotii"),
        (2.0, 4.0, "ja söin leipää", "ja söin leipää"),
        (10.0, 13.0, "se oli hyvä päivä", None),
    ]


def test_segments_join_close_sentences_and_keep_their_parts() -> None:
    segs = prepare_skn.segments(prepare_skn.sentences(EAF.encode("utf-8")))
    assert [(g["start"], g["end"], g["ref"], g["orig"]) for g in segs] == [
        (0.0, 4.0, "minä menin kotiin ja söin leipää", "mie mänin kotii ja söin leipää"),
        (10.0, 13.0, "se oli hyvä päivä", None),
    ]
    assert segs[0]["parts"] == [(0.0, 1.5, "minä menin kotiin"), (2.0, 4.0, "ja söin leipää")]
    assert prepare_skn.split_of(5) == "test" and prepare_skn.split_of(3) == "validation"
    assert prepare_skn.split_of(7) == "train"
