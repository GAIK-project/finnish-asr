"""Shared paths, the committed clip transcripts and the frozen reference alignment."""

import hashlib
import json

import pytest

import fi_common as FI
import judge_asr


def test_load_arms_has_this_repos_and_qad1_arms_and_no_hy_files() -> None:
    arms = FI.load_arms()
    ids = [a.arm_id for a in arms]
    assert len(ids) == len(set(ids)) == 36
    assert not [i for i in ids if i.startswith("hy")]
    qad1 = {a.arm_id for a in arms if FI.QAD1_TAG in a.notes}
    assert qad1 == {p.stem for p in FI.QAD1_ASR_DIR.glob("*.json") if "__r" not in p.stem}
    assert len(qad1) == 11
    assert {"elevenlabs_scribe-v2", "ft_whisper-large-v3-fi-skn-punct-best-chunks", "qwen_qwen3-asr-0.6b"} <= set(ids)
    assert all(a.repeat == 1 for a in arms)


def test_every_transcript_points_at_the_downloaded_clip() -> None:
    files = [p for p in FI.ASR_DIR.rglob("*.json") if not p.name.startswith("_")]
    assert len(files) == 58  # 29 runs of this repository and 29 of QAD-1
    for p in files:
        arm = FI.load_arm(p)
        assert arm.input_ref == "data/ajokortti/Ajokortti.mp3", p.name
        assert arm.stage == "asr"
        assert not arm.arm_id.startswith("hy")


def test_reference_alignment_is_contiguous_sentences_over_402_tokens() -> None:
    g = json.loads(FI.GT_ALIGNMENT_PATH.read_text(encoding="utf-8"))
    tokens, sentences, audit = g["gt_tokens"], g["gt_sentences"], g["gt_audit"]
    assert len(tokens) == 402 and len(sentences) == 23
    assert sentences[0]["token_start"] == 0 and sentences[-1]["token_end"] == len(tokens)
    for prev, nxt in zip(sentences, sentences[1:], strict=False):
        assert prev["token_end"] == nxt["token_start"]
    assert len(audit["flags"]) == 42 and audit["threshold"] == 4 and len(audit["systems"]) == 7
    assert not [s for s in audit["systems"] if s.startswith("hy")]
    times = [t[k] for t in tokens for k in ("start", "end") if t.get(k) is not None]
    assert len(times) == 2 * len(tokens) and all(0 <= x <= 180 for x in times)


def test_committed_judge_records_match_the_transcripts_and_the_prompt() -> None:
    texts = {a.arm_id: a.text for a in FI.load_arms()}
    records = sorted(FI.JUDGE_DIR.glob("*/*.json"))
    assert len(records) == 56
    for p in records:
        r = json.loads(p.read_text(encoding="utf-8"))
        assert r["arm_id"] == p.stem and r["arm_id"] in texts
        assert r["text_sha256"] == hashlib.sha256(texts[r["arm_id"]].encode("utf-8")).hexdigest(), p
        assert r["prompt_sha256"] == judge_asr.PROMPT_SHA, p


def test_clip_files_are_pinned_to_gaik_toolkit() -> None:
    assert FI.TOOLKIT_RAW.startswith("https://raw.githubusercontent.com/GAIK-project/gaik-toolkit/main/")
    assert set(FI.CLIP_FILES) == {"Ajokortti.mp3", "reference.txt", "reference.reviewed.txt"}
    assert all(len(h) == 64 for h in FI.CLIP_FILES.values())
    assert FI.CLIP_DIR.relative_to(FI.REPO_ROOT).as_posix() == "data/ajokortti"


def test_require_clip_stops_with_a_hint(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(FI, "CLIP_DIR", tmp_path)
    with pytest.raises(SystemExit, match="get_clip.py"):
        FI.require_clip("reference.txt")


def test_rel_never_returns_an_absolute_path(tmp_path) -> None:
    assert FI.rel(FI.REPO_ROOT / "results" / "scores.json") == "results/scores.json"
    assert FI.rel(tmp_path / "x.json") == "x.json"
