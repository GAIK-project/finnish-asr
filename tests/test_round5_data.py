"""Round 5 data preparation: the parliament sample, the Border Karelia segments and --data parsing."""

import argparse

import pytest

import finetune_whisper as FT
import prepare_finka as FK
import prepare_parliament as PP


def test_parliament_text_with_an_unaligned_word_is_dropped() -> None:
    assert PP.clean("arvoisa  puhemies\n") == "arvoisa puhemies"
    assert PP.clean("<UNK> erittäin vaikeat ajat") is None
    assert PP.eligible(5.0, "kolme sanaa tässä")
    assert not PP.eligible(1.5, "kolme sanaa tässä")
    assert not PP.eligible(5.0, "kaksi sanaa")


def test_parliament_sample_is_seeded_and_capped_per_speaker() -> None:
    items = [{"id": f"{s}-{i:03d}", "speaker": s, "seconds": 10.0} for s in "abc" for i in range(100)]
    one = PP.select(items, hours=0.5, per_speaker=70)
    assert one == PP.select(items, hours=0.5, per_speaker=70)
    assert sum(r["seconds"] for r in one) == 1800
    assert max(sum(r["speaker"] == s for r in one) for s in "abc") <= 70
    assert len(PP.select(items, hours=100, per_speaker=40)) == 120


def test_voxpopuli_speakers_match_corpus_names() -> None:
    names = {PP.norm_name(n) for n in PP.VOXPOPULI_SPEAKERS}
    assert PP.norm_name("anne_mari_virolainen") == PP.norm_name("Anne-Mari Virolainen")
    assert PP.norm_name("Miapetra Kumpula-Natri") in names
    assert len(PP.VOXPOPULI_SPEAKERS) == len(set(PP.VOXPOPULI_SPEAKERS))


def test_finka_marks_are_cleaned_and_unclear_speech_dropped() -> None:
    assert FK.clean("semmosta se ol´.") == "semmosta se ol."
    assert FK.clean("niitä pit*tää siell_ei kes- ver´suon-nen") == "niitä pittää siell ei kes versuonnen"
    assert FK.clean("sano ”em mie ota”") == "sano em mie ota"
    assert FK.clean("kaks, tarpojia, (tuot), kaks") is None
    assert FK.clean("se oli @ sitte") is None


def test_finka_informants_from_file_names() -> None:
    assert FK.informants("Korpiselka_07nB_08mB_SKNA_3700_1b") == {"Korpiselka_07", "Korpiselka_08"}
    assert FK.informants("Ilomantsi_13nE_SKNA_13818_1a") == {"Ilomantsi_13"}


GRID = """File type = "ooTextFile"
Object class = "TextGrid"
item []:
    item [1]:
        class = "IntervalTier"
        name = "MJ"
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 1.0
            text = ""
        intervals [2]:
            xmin = 10.0
            xmax = 12.0
            text = "No mimmoin se nuotta ol?"
    item [2]:
        class = "IntervalTier"
        name = "VV"
        intervals: size = 3
        intervals [1]:
            xmin = 1.0
            xmax = 4.0
            text = "se oli, semmone ´iso nuotta."
        intervals [2]:
            xmin = 4.5
            xmax = 7.0
            text = "ja sillä pyyvettih."
        intervals [3]:
            xmin = 11.0
            xmax = 14.0
            text = "no se oli pitkä."
    item [3]:
        class = "IntervalTier"
        name = "original"
        intervals: size = 1
        intervals [1]:
            xmin = 1.0
            xmax = 4.0
            text = "VV: se oli"
"""


def test_finka_segments_join_a_speaker_and_skip_overlaps() -> None:
    segs = FK.segments(GRID)
    assert [(s["speaker"], s["start"], s["end"]) for s in segs] == [("VV", 1.0, 7.0)]
    assert segs[0]["ref"] == "se oli, semmone iso nuotta. ja sillä pyyvettih."


def test_data_sources() -> None:
    assert FT.data_sources("skn+open+lamurre+parl+finka") == {"skn", "open", "lamurre", "parl", "finka"}
    assert FT.data_sources("skn+open+lamurre") == {"skn", "open", "lamurre"}
    with pytest.raises(argparse.ArgumentTypeError):
        FT.data_sources("skn+skn-all")
    with pytest.raises(argparse.ArgumentTypeError):
        FT.data_sources("skn+voxpopuli")
