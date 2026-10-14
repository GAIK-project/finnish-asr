"""Run metadata never holds secret values or local absolute paths."""

import json

import pytest

import provenance

ROOT = provenance.repo_root()


def test_repo_root_is_this_repository() -> None:
    assert (ROOT / "pyproject.toml").exists() and (ROOT / "scripts" / "provenance.py").exists()


def test_env_values_are_recorded_only_as_set_or_unset(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    secret = "sk-test-0123456789-not-a-real-key"
    monkeypatch.setenv("FIASR_TEST_SECRET", secret)
    monkeypatch.delenv("FIASR_TEST_MISSING", raising=False)
    meta = provenance.build_run_metadata(
        app_under_test="test",
        dataset_path=ROOT / "results" / "scores.json",
        runner_path=ROOT / "scripts" / "score_asr.py",
        cli_args=["--reviewed"],
        env_names=["FIASR_TEST_SECRET", "FIASR_TEST_MISSING"],
    )
    out = provenance.write_run_metadata(tmp_path, meta, name="_run_metadata_test.json")
    text = out.read_text(encoding="utf-8")
    assert secret not in text
    data = json.loads(text)
    assert data["env"] == {"FIASR_TEST_SECRET": "set", "FIASR_TEST_MISSING": "unset"}
    assert data["runner_path"] == "scripts/score_asr.py"
    assert data["dataset_path"] == "results/scores.json"
    assert data["repo"] == "GAIK-project/finnish-asr"
    assert data["cli_args"] == ["--reviewed"]


def test_paths_outside_the_repository_keep_only_their_name(tmp_path) -> None:
    meta = provenance.build_run_metadata(
        app_under_test="test", dataset_path=tmp_path / "corpus" / "test.jsonl", runner_path=ROOT / "scripts" / "x.py"
    )
    assert meta["dataset_path"] == "test.jsonl"
    assert str(tmp_path) not in json.dumps(meta)


def test_metadata_file_names_must_be_sidecars(tmp_path) -> None:
    with pytest.raises(ValueError):
        provenance.write_run_metadata(tmp_path, {}, name="scores.json")
    assert provenance.write_run_metadata(tmp_path, {"a": 1}).name == "_run_metadata.json"


def test_committed_run_metadata_records_no_values_and_no_local_paths() -> None:
    files = sorted((ROOT / "results").rglob("_*.json"))
    assert len(files) >= 10
    for p in files:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data.get("env"), dict):  # the older GPU sidecars use "env" for the venv name
            assert set(data["env"].values()) <= {"set", "unset"}, p
        text = json.dumps(data)
        for marker in (":\\\\", ":/Users", "/home/", "/scratch/", "/users/"):
            assert marker not in text, (p, marker)
