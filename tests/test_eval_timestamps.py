"""Timestamp and long-form scoring."""

import json
import math
import shutil

import pytest

import eval_timestamps as et


def test_nearest_hits_measure_each_reference_start_to_the_closest_prediction() -> None:
    assert et.nearest_hits([1.0, 5.0], [0.8, 4.0, 5.1]) == pytest.approx([0.2, 0.1])
    assert et.nearest_hits([1.0], []) == [math.inf]


def test_boundary_stats_count_shares_and_ignore_missing_in_the_median() -> None:
    stats = et.boundary_stats([0.1, 0.4, 0.9, 3.0, math.inf])
    assert stats == {"sentence_starts": 5, "within_0_5s": 0.4, "within_1s": 0.6, "median_s": 0.65}
    assert et.boundary_stats([]) == {"sentence_starts": 0, "within_0_5s": None, "within_1s": None, "median_s": None}


def test_loops_flag_a_repeated_phrase_that_the_reference_does_not_repeat() -> None:
    phrase = "ja sitten me mentiin kotiin "
    assert et.loops(phrase * 4, "ja sitten me mentiin kotiin")
    assert not et.loops(phrase * 3, "ja sitten me mentiin kotiin")
    assert not et.loops(phrase * 4, phrase * 4)  # said four times: not a loop


def test_quiet_bounds_cut_long_audio_into_15_to_25_second_pieces() -> None:
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 0.1, et.SR * 70).astype("float32")
    audio[et.SR * 20 : et.SR * 20 + 1600] = 0  # a quiet 100 ms at 20 s
    bounds = et.quiet_bounds(len(audio), audio)
    assert bounds[0][0] == 0 and bounds[-1][1] == len(audio)
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:], strict=False))
    assert all((e - s) / et.SR <= 25 for s, e in bounds)
    assert abs(bounds[0][1] / et.SR - 20.05) < 0.06


ARMS = ["ft_whisper-large-v3-fi-skn-punct-ts", "hf_whisper-large-v3", "hub_whisper-large-v3-fi-dialects"]


def test_score_reproduces_the_committed_timestamp_results(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    for arm in ARMS:
        shutil.copy(et.OUT / f"{arm}.json", tmp_path / f"{arm}.json")
    committed = json.loads((et.OUT / "scores.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(et, "OUT", tmp_path)
    assert et.score() == 0
    assert json.loads((tmp_path / "scores.json").read_text(encoding="utf-8")) == committed
    meta = json.loads((tmp_path / "_run_metadata.json").read_text(encoding="utf-8"))
    assert meta["extra"]["arms"] == ARMS
    assert meta["model_or_deployment"][0] == "whisper-large-v3-fi-skn-punct-ts"  # a name, never a scratch path


def test_committed_timestamp_numbers_are_the_ones_the_docs_quote() -> None:
    s = json.loads((et.OUT / "scores.json").read_text(encoding="utf-8"))
    base, pub, ts = (s[a] for a in ("hf_whisper-large-v3", "hub_whisper-large-v3-fi-dialects", ARMS[0]))
    within_1s = [round(x["test_segments"]["boundaries"]["within_1s"] * 100) for x in (base, pub, ts)]
    assert within_1s == [61, 1, 58]
    assert ts["test_segments"]["boundaries"]["median_s"] == 0.04
    lf = [x["longform"] for x in (base, pub, ts)]
    assert [round(x["sequential"]["wer"], 1) for x in lf] == [37.5, 21.5, 25.2]
    assert [x["sequential"]["windows_with_loop"] for x in lf] == [0, 1, 3]
    assert [round(x["sequential_fallback"]["wer"], 1) for x in lf] == [37.5, 19.8, 18.1]
    assert [x["sequential_fallback"]["windows_with_loop"] for x in lf] == [0, 0, 0]
    assert [round(x["sequential_fallback"]["boundaries"]["within_1s"] * 100) for x in lf] == [63, 19, 49]
    assert [round(x["chunks"]["wer"], 1) for x in lf] == [39.4, 17.3, 17.1]


def test_raw_records_carry_the_fallback_settings_and_record_models_by_name() -> None:
    expected = {k: list(v) if isinstance(v, tuple) else v for k, v in et.KW_FALLBACK.items()}
    for arm in ARMS:
        d = json.loads((et.OUT / f"{arm}.json").read_text(encoding="utf-8"))
        assert d["arm"] == arm
        assert d["model"] in {"openai/whisper-large-v3", "GAIK-project/whisper-large-v3-fi-dialects", arm[3:]}
        assert d["settings"]["generate_fallback"] == expected
        assert all(set(w) >= {"sequential", "sequential_fallback", "chunks"} for w in d["longform"].values())
