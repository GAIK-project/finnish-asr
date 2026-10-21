"""dialect_regions.py: region lookup and the prompt; the training collate masks the prompt."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from dialect_regions import NAMES, SKN, WRONG, lamurre_regions, prompt_text, region_of  # noqa: E402


def test_every_skn_municipality_has_a_known_region():
    assert len(SKN) == 50
    assert set(SKN.values()) == set(NAMES)


def test_wrong_is_a_derangement():
    assert sorted(WRONG) == sorted(WRONG.values()) == sorted(NAMES)
    assert all(WRONG[r] != r for r in WRONG)


def test_region_of_skn_and_lamurre_ids(tmp_path):
    tsv = tmp_path / "regions.tsv"
    tsv.write_text("hame_tamm1\tHÄM\tHämE\tTammela\nx_y1\tNONE\t\t\n", encoding="utf-8")
    lam = lamurre_regions(tsv)
    assert lam == {"hame_tamm1": "HÄM"}
    assert region_of("SKN10a_Mikkeli_0012.wav", lam) == "SAV"
    assert region_of("SKN15b_Rautu_0000", lam) == "KAA"
    assert region_of("hame_tamm1_0003.wav", lam) == "HÄM"
    assert region_of("1234.wav", lam) is None
    assert region_of("SKN01", lam) is None
    assert region_of("SKN01_0000.wav", lam) is None
    assert lamurre_regions(tmp_path / "missing.tsv") == {}


def test_prompt_text():
    assert prompt_text("SAV") == "Murre: savolaismurteet."
