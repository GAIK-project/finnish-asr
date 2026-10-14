"""The clip scorer (HY's normalisation) and the published clip numbers."""

import json

import jiwer
import pytest

import asr_scoring as qs
import fi_common as FI
import score_asr


def test_identical_text_scores_perfectly() -> None:
    text = "Siinä tehdään sitten semmoista, niin kuin sanotaanko high-end-implanttihoitoa."
    metrics, rows = qs.asr_metrics(text, text)
    assert metrics["WER"] == 0.0
    assert metrics["CER"] == 0.0
    assert metrics["CER_nospace"] == 0.0
    assert all(r["op"] == "equal" for r in rows)


def test_insertion_rows_point_at_the_next_reference_token() -> None:
    _, rows = qs.word_alignment("a b", "a b c")
    assert rows[-1] == {"op": "ins", "ref": None, "hyp": "c", "ref_idx": 2}


def test_newline_glued_tokens_are_reported_with_fixed_metrics() -> None:
    ref = "yksi kaksi kolme neljä"
    hyp = "yksi kaksi\nkolme neljä"
    metrics, _ = qs.asr_metrics(ref, hyp)
    assert metrics["WER"] == 50.0  # HY's split keeps "kaksi\nkolme" as one token
    merge = qs.whitespace_merge(ref, hyp)
    assert merge is not None
    assert merge["merged_tokens"] == ["kaksi\nkolme"]
    assert merge["WER"] == 0.0
    assert qs.whitespace_merge(ref, ref) is None


def test_hyphens_and_case_are_not_errors_but_compound_splits_are() -> None:
    metrics, _ = qs.asr_metrics("High-end implanttihoito.", "highend Implanttihoito")
    assert metrics["WER"] == 0.0
    metrics, _ = qs.asr_metrics("implanttihoito", "implantti hoito")
    assert metrics["WER"] == 200.0  # one substitution and one insertion against one reference word
    assert metrics["CER_nospace"] == 0.0


def _scores(name: str) -> dict[str, dict]:
    return {r["arm_id"]: r for r in json.loads((FI.RESULTS_DIR / name).read_text(encoding="utf-8"))["arms"]}


def test_every_clip_wer_nl_reproduces_from_the_frozen_reference_tokens() -> None:
    """Offline: WER_nl of every committed transcript, against the reference tokens in gt_alignment.json."""
    tokens = [t["token"] for t in json.loads(FI.GT_ALIGNMENT_PATH.read_text(encoding="utf-8"))["gt_tokens"]]
    reference = " ".join(tokens)
    scores = _scores("scores.json")
    arms = FI.load_arms()
    assert {a.arm_id for a in arms} == set(scores)
    norm = qs.ws_fixed_transform(False)
    for arm in arms:
        out = jiwer.process_words(reference, arm.text, norm, norm)
        wer = (out.substitutions + out.deletions + out.insertions) / (out.hits + out.substitutions + out.deletions)
        assert round(wer * 100, 2) == scores[arm.arm_id]["WER_nl"], arm.arm_id


@pytest.mark.skipif(
    not (FI.GT_FI_PATH.exists() and FI.GT_REVIEWED_PATH.exists()),
    reason="needs the references from gaik-toolkit: uv run python scripts/get_clip.py",
)
def test_scores_json_reproduces_with_the_downloaded_references() -> None:
    for reviewed, name in ((False, "scores.json"), (True, "scores.reviewed.json")):
        committed = json.loads((FI.RESULTS_DIR / name).read_text(encoding="utf-8"))
        fresh = score_asr.score(reviewed)
        assert fresh["arms"] == committed["arms"], name
        assert fresh["reference"] == committed["reference"]
    hy = _scores("scores.json")
    reviewed = _scores("scores.reviewed.json")
    for arm, wer, wer_reviewed in (
        ("hf_whisper-large-v3", 14.68, 13.73),
        ("vertex_gemini-3.5-flash", 14.18, 13.73),
        ("elevenlabs_scribe-v2", 18.16, 14.22),
        ("openai_gpt-4o-mini-transcribe", 13.93, 13.24),
    ):
        assert hy[arm]["metrics"]["WER"] == wer
        assert reviewed[arm]["metrics"]["WER"] == wer_reviewed
        assert reviewed[arm]["WER_against_hy"] == wer


def test_timestamp_variant_clip_numbers() -> None:
    hy, reviewed = _scores("scores.json"), _scores("scores.reviewed.json")
    for arm, wer, wer_reviewed in (
        ("ft_whisper-large-v3-fi-skn-punct-ts", 16.67, 16.18),
        ("ft_whisper-large-v3-fi-skn-punct-ts-chunks", 15.17, 14.46),
        ("ft_whisper-large-v3-fi-skn-punct-best", 62.69, 62.25),
        ("ft_whisper-large-v3-fi-skn-punct-best-chunks", 13.93, 13.97),
    ):
        assert hy[arm]["metrics"]["WER"] == wer
        assert reviewed[arm]["metrics"]["WER"] == wer_reviewed


def test_decoding_check_holds_the_six_decodings_of_the_model_card() -> None:
    doc = json.loads(score_asr.DECODING_PATH.read_text(encoding="utf-8"))
    assert doc["model"] == "GAIK-project/whisper-large-v3-fi-dialects"
    names = {"quiet_chunks", "quiet_chunks_norepeat", "chunk20", "chunk20_norepeat", "chunk30", "chunk30_norepeat"}
    assert set(doc["decodings"]) == names
    scores = json.loads(score_asr.DECODING_SCORES_PATH.read_text(encoding="utf-8"))["decodings"]
    card = {name: [round(scores[name]["WER"][ref], 1) for ref in ("hy", "reviewed")] for name in scores}
    assert card["quiet_chunks"] == [13.7, 13.5]
    assert card["chunk20"] == [14.7, 13.5]
    assert card["chunk30"] == [52.5, 51.5]
    assert card["chunk30_norepeat"] == [14.4, 14.2]


@pytest.mark.skipif(
    not (FI.GT_FI_PATH.exists() and FI.GT_REVIEWED_PATH.exists()),
    reason="needs the references from gaik-toolkit: uv run python scripts/get_clip.py",
)
def test_decoding_scores_reproduce_with_the_downloaded_references() -> None:
    committed = json.loads(score_asr.DECODING_SCORES_PATH.read_text(encoding="utf-8"))
    fresh = score_asr.score_decoding()
    assert fresh["decodings"] == committed["decodings"]
    assert fresh["references"] == committed["references"]
