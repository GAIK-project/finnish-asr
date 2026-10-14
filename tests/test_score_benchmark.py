"""The public test sets: committed scores reproduce from the committed per-utterance files."""

import json

import pytest

import score_benchmark as sb

RESULTS = sb.RESULTS_DIR / "benchmarks"


@pytest.mark.parametrize("bench", ["fleurs_fi", "voxpopuli_fi", "skn_fi"])
def test_committed_scores_reproduce_from_the_jsonl(bench: str) -> None:
    committed = json.loads((RESULTS / bench / "scores.json").read_text(encoding="utf-8"))["arms"]
    assert sb.score_rows(bench) == committed


def test_lamurre_has_numbers_only_and_falls_back_to_its_scores_json() -> None:
    d = RESULTS / "lamurre_fi"
    if list(d.glob("*.jsonl")):
        pytest.skip("local LA-murre files present (gitignored); the fallback is not used")
    committed = json.loads((d / "scores.json").read_text(encoding="utf-8"))["arms"]
    assert sb.score("lamurre_fi", write=False) == committed
    assert len(committed) == 13 and all(r["utterances"] == 583 for r in committed)
    for row in committed:  # numbers and identifiers, no text fields
        assert set(row) == {
            "arm_id", "label", "utterances", "WER", "CER", "SUB", "DEL", "INS",
            "ref_words", "empty_hyps", "seconds_per_utterance", "model", "gpu",
        }  # fmt: skip


def test_readme_benchmark_table_matches_the_scores() -> None:
    readme = (sb.PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    table = readme[readme.index(sb.START) + len(sb.START) : readme.index(sb.END)]
    skn = {r["arm_id"]: r for r in sb.score_rows("skn_fi")}
    assert f"| Whisper large-v3 | {skn['hf_whisper-large-v3']['WER']:.1f} |" in table
    published = skn["ft_whisper-large-v3-fi-skn-punct-best"]["WER"]
    assert f"(publishable)** †‡§ | {published:.1f} |" in table


def test_transform_lowercases_and_drops_punctuation() -> None:
    assert sb.transform(False)("Hei, Maailma!\nToinen  rivi.") == [["hei", "maailma", "toinen", "rivi"]]
