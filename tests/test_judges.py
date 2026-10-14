"""Judge maths and prompts: agreement statistics, score roll-ups, the strict schema, prompt hashes."""

import hashlib
import json

import pytest

import fi_common as FI
import judge_agreement as ja
import judge_asr
import judge_dialect as jd
import prepare_targets


def test_sign_test_is_two_sided_and_ignores_nothing_it_is_given() -> None:
    assert ja.sign_test(0, 0) == 1.0
    assert ja.sign_test(5, 5) == 1.0
    assert ja.sign_test(10, 0) == pytest.approx(2 / 2**10)
    assert ja.sign_test(9, 1) == pytest.approx(2 * 11 / 2**10)
    assert ja.sign_test(1, 9) == ja.sign_test(9, 1)


def test_ranks_average_ties_and_spearman_handles_them() -> None:
    assert ja.ranks([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]
    assert ja.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert ja.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert ja.spearman([1, 2, 2, 3], [1, 2, 3, 3]) == 0.833
    assert ja.spearman([1, 1, 1], [1, 2, 3]) is None  # no variation on one side


def _row(seg: str, words: int, **arms: tuple[int, int, int]) -> dict:
    return {
        "id": seg,
        "ref_words": words,
        "arms": {a: {"critical": c, "major": m, "minor": n} for a, (c, m, n) in arms.items()},
    }


def test_agreement_counts_pairs_where_both_judges_see_a_difference() -> None:
    a, b = ja.BASE, ja.PUBLISHED
    ref = {"s1": _row("s1", 10, **{a: (0, 1, 0), b: (0, 0, 1)}), "s2": _row("s2", 10, **{a: (0, 0, 0), b: (0, 0, 0)})}
    same = {"s1": _row("s1", 10, **{a: (1, 0, 0), b: (0, 0, 0)}), "s2": _row("s2", 10, **{a: (0, 0, 1), b: (0, 0, 0)})}
    out = ja.agreement(ref, same, [a, b])
    assert out["segments_both_judged"] == 2
    assert out["pairs_compared"] == 1  # s2: the reference judge sees no difference
    assert out["pairwise_same_preference"] == 1.0
    summary = ja.judge_summary(ref, [a, b])
    assert summary["points_per_100w"] == {a: 25.0, b: 5.0}
    assert summary["published_vs_base"]["fewer_points"] == 1 and summary["order_best_first"] == [b, a]


def test_committed_agreement_file_is_consistent() -> None:
    path = FI.RESULTS_DIR / "judgements_dialect" / "lamurre_fi" / "agreement-r2.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    claude = data["judges"]["claude-opus-5-5"]
    summary = json.loads((path.parent / "summary-r2.json").read_text(encoding="utf-8"))
    for arm, s in summary["arms"].items():
        assert claude["points_per_100w"][arm] == s["error_points_per_100w"]
    oss = data["judges"]["aitta-gpt-oss-120b"]
    assert oss["segments"] == 583 and oss["order_best_first"] == claude["order_best_first"]
    assert oss["against_reference"]["pairwise_same_preference"] == 0.807


def test_judge_asr_roll_up_scores_points_per_100_reference_words() -> None:
    ans = {
        "errors": [
            {"category": "wrong_name", "severity": "major", "reference_span": "a", "transcript_span": "b",
             "explanation": "x"},
            {"category": "omission", "severity": "minor", "reference_span": "c", "transcript_span": "",
             "explanation": "y"},
        ],
        "entities": [
            {"entity": "Straumann", "status": "correct", "transcript_form": "Straumann"},
            {"entity": "Futudent", "status": "variant", "transcript_form": "Futodent"},
            {"entity": "Peter", "status": "missing", "transcript_form": ""},
        ],
        "ref_doubts": [],
        "usability": 120,
        "summary": "ok",
    }  # fmt: skip
    s = judge_asr.roll_up(ans, ref_words=200)
    assert s["error_points_per_100w"] == 3.0
    assert s["error_counts"] == {"critical": 0, "major": 1, "minor": 1}
    assert (s["entities_ok"], s["entities_total"]) == (2, 3)
    assert s["usability"] == 100


def test_strict_schema_closes_every_object_and_inlines_refs() -> None:
    schema = judge_asr.JSON_SCHEMA
    text = json.dumps(schema)
    assert "$ref" not in text and "$defs" not in text and '"title"' not in text

    def objects(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                yield node
            for v in node.values():
                yield from objects(v)
        elif isinstance(node, list):
            for v in node:
                yield from objects(v)

    found = list(objects(schema))
    assert found and all(o["additionalProperties"] is False for o in found)
    assert all(set(o["required"]) == set(o["properties"]) for o in found)


def test_clip_judge_prompt_is_the_one_the_committed_records_used() -> None:
    meta = json.loads((FI.JUDGE_DIR / "_run_metadata.json").read_text(encoding="utf-8"))
    assert judge_asr.PROMPT_SHA == meta["extra"]["prompt_sha256"]


def test_punctuation_prompt_is_the_one_that_made_the_skn_targets() -> None:
    for split in ("train", "validation"):
        stats = json.loads((FI.RESULTS_DIR / "labels" / f"skn_{split}.targets.stats.json").read_text(encoding="utf-8"))
        assert hashlib.sha256(prepare_targets.PROMPTS["punct"].encode()).hexdigest() == stats["prompt_sha256"]


def test_punctuation_targets_keep_every_word() -> None:
    assert prepare_targets.accept("punct", "minä menin kotiin", "Minä menin kotiin.")
    assert not prepare_targets.accept("punct", "minä menin kotiin", "Minä menin kotiin tänään.")
    assert prepare_targets.accept("normalise", "a b c d e f g h i j", "A b c d e f g h.")
    assert not prepare_targets.accept("normalise", "a b c d e f g h i j", "A b c.")


def test_dialect_judge_is_a_new_public_prompt_version_with_the_published_schema() -> None:
    summary = json.loads(
        (FI.RESULTS_DIR / "judgements_dialect" / "lamurre_fi" / "summary-r2.json").read_text(encoding="utf-8")
    )
    assert summary["prompt_version"] == "dialect-judge-v1"
    assert jd.PROMPT_VERSION == "dialect-judge-v2"
    defaults = list(jd.ARMS)
    jd.configure(summary["arms_judged_together"], "r2", "foundry")
    try:
        assert jd.LABELS == "ABCD"
        rows = [
            {
                "ref_words": 20,
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "arms": {a: {"critical": 0, "major": i, "minor": 1} for i, a in enumerate(jd.ARMS)},
            }
        ]
        s = jd.summarise(rows)
        first = jd.ARMS[0]
    finally:
        jd.configure(defaults, "", "foundry")
    assert s["prompt_version"] == "dialect-judge-v2"
    assert s["arms"][first] == {
        "critical": 0,
        "major": 0,
        "minor": 1,
        "error_points_per_100w": 5.0,
        "segments_without_errors": 0,
        "segments_best_or_tied": 1,
    }
