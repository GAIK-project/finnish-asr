"""Border Karelia data (prepare_finka.py), its --data option and the target budget of prepare_targets.py."""

import json
from pathlib import Path

import finetune_whisper as FT
import prepare_finka as FK
import prepare_targets as PT


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

xmin = 0
xmax = 14
tiers? <exists>
size = 3
item []:
    item [1]:
        class = "IntervalTier"
        name = "MJ"
        xmin = 0
        xmax = 14
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
        xmin = 0
        xmax = 14
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
        xmin = 0
        xmax = 14
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


SHORT_GRID = """File type = "ooTextFile"
Object class = "TextGrid"

0
14
<exists>
2
"IntervalTier"
"VV"
0
14
3
0
1.0
""
1.0
4.0
"se oli ""iso"" nuotta [3] vuotta."
4.0
14
""
"TextTier"
"marks"
0
14
1
2.5
"x"
"""


def test_finka_reads_praats_short_format_like_the_long_one() -> None:
    assert FK.tiers(SHORT_GRID) == {"VV": [(1.0, 4.0, 'se oli "iso" nuotta [3] vuotta.')]}
    assert FK.tiers(GRID) == {
        "MJ": [(10.0, 12.0, "No mimmoin se nuotta ol?")],
        "VV": [
            (1.0, 4.0, "se oli, semmone ´iso nuotta."),
            (4.5, 7.0, "ja sillä pyyvettih."),
            (11.0, 14.0, "no se oli pitkä."),
        ],
    }


def test_finka_is_a_data_option_after_parl() -> None:
    assert "skn+open+lamurre+parl+finka" in FT.DATA_CHOICES
    assert "skn+open+lamurre+parl" in FT.DATA_CHOICES  # the main model's recipe stays as it was


def test_finka_loader_reads_the_targets(tmp_path, monkeypatch) -> None:
    d = tmp_path / "finka_fi"
    d.mkdir()
    row = {"id": "Salmi_01nA_SKNA_1_1a_0000", "ref": "se ol", "target": "Se oli."}
    (d / "train.targets.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(FT, "DATA", tmp_path)
    (item,) = FT.finka("train")
    assert item["source"] == "finka" and item["text"] == "Se oli."
    assert Path(item["path"]) == d / "train" / "Salmi_01nA_SKNA_1_1a_0000.wav"


def test_usd_estimate_uses_the_model_price(monkeypatch) -> None:
    monkeypatch.setattr(PT, "MODEL", "gpt-6.1-sol")
    assert PT.usd({"input_tokens": 1_000_000, "output_tokens": 100_000}) == 3.0
    monkeypatch.setattr(PT, "MODEL", "claude-opus-5-5")
    assert PT.usd({"input_tokens": 1_000_000, "output_tokens": 100_000}) == 6.0
