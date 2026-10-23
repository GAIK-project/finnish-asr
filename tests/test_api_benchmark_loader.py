"""Exercise the local loader with authored fixtures, without importing the API runner."""

import ast
import io
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "run_api_benchmark.py"
PARLIAMENT = ("parl_fi", "parl_terms_fi")
FLAT = ("fleurs_fi", "voxpopuli_fi", "lamurre_fi")


@pytest.fixture
def source_tree():
    # Executing only selected AST nodes avoids load_repo_env(), SDKs and main().
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def assignment(tree, name):
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )


@pytest.fixture
def benches(source_tree):
    return ast.literal_eval(assignment(source_tree, "BENCHES").value)


@pytest.fixture
def loader(source_tree, benches, tmp_path):
    function = next(
        node for node in source_tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_items"
    )
    scope = {
        "Path": Path,
        "json": json,
        "DATA": tmp_path / "data",
        "RESULTS_DIR": tmp_path / "results",
        "BENCHES": benches,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope["load_items"]


@pytest.fixture
def wav_bytes():
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 160)
    return out.getvalue()


def make_inputs(root, bench, wav_bytes, *, audio_count=3):
    references = root / "results" / "benchmarks" / bench / "hf_whisper-large-v3.jsonl"
    references.parent.mkdir(parents=True)
    # All text is authored for this test; no corpus or model output is used.
    rows = [{"id": f"item-{i}.wav", "ref": f"Testiviite {i}: ääni kuuluu.", "terms": ["Testisana"]} for i in range(3)]
    references.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    audio_root = root / "data" / bench
    if bench in PARLIAMENT:
        audio_root /= "test"
    audio_root.mkdir(parents=True)
    for row in rows[:audio_count]:
        (audio_root / row["id"]).write_bytes(wav_bytes)
    return rows


def test_benchmark_choices_preserve_existing_sets(benches):
    assert set(benches) == {"skn_fi", "dental10_fi", *FLAT, *PARLIAMENT}


@pytest.mark.parametrize("bench", PARLIAMENT)
def test_parliament_audio_path_and_provider_contract(loader, tmp_path, wav_bytes, bench):
    rows = make_inputs(tmp_path, bench, wav_bytes)
    assert loader(bench, None) == [
        {"id": row["id"], "ref": row["ref"], "wav": wav_bytes, "mime": "audio/wav"} for row in rows
    ]


@pytest.mark.parametrize("bench", FLAT)
def test_existing_flat_audio_layout(loader, tmp_path, wav_bytes, bench):
    rows = make_inputs(tmp_path, bench, wav_bytes)
    assert [item["id"] for item in loader(bench, None)] == [row["id"] for row in rows]


@pytest.mark.parametrize("bench", (*PARLIAMENT, *FLAT))
@pytest.mark.parametrize("limit", (0, 1, 2))
def test_limit_does_not_read_later_audio(loader, tmp_path, wav_bytes, bench, limit):
    rows = make_inputs(tmp_path, bench, wav_bytes, audio_count=limit)
    assert [item["id"] for item in loader(bench, limit)] == [row["id"] for row in rows[:limit]]
    with pytest.raises(SystemExit, match="missing"):
        loader(bench, None)


@pytest.mark.parametrize("bench", PARLIAMENT)
def test_missing_selected_audio_stops(loader, tmp_path, wav_bytes, bench):
    make_inputs(tmp_path, bench, wav_bytes, audio_count=0)
    with pytest.raises(SystemExit, match="missing"):
        loader(bench, 1)


def test_missing_references_stops(loader):
    with pytest.raises(SystemExit, match="needed for the ids and references"):
        loader("parl_fi", 1)


@pytest.mark.parametrize("bench", ("skn_fi", "dental10_fi", *FLAT, *PARLIAMENT))
def test_negative_limit_stops_before_reading_or_downloading(loader, bench):
    with pytest.raises(ValueError, match="limit must be non-negative"):
        loader(bench, -1)


def test_unknown_benchmark_stops_before_file_access(loader):
    with pytest.raises(ValueError, match="unknown benchmark"):
        loader("unknown", 1)


@pytest.mark.parametrize("configured", (False, True))
def test_data_root_default_and_override_without_process_env(source_tree, tmp_path, configured):
    data_root = tmp_path / "explicit-data"
    scope = {
        "Path": Path,
        "RESULTS_DIR": tmp_path / "results",
        "os": SimpleNamespace(environ={"FIASR_DATA": str(data_root)} if configured else {}),
    }
    exec(compile(ast.Module(body=[assignment(source_tree, "DATA")], type_ignores=[]), str(SOURCE), "exec"), scope)
    assert scope["DATA"] == (data_root if configured else tmp_path / "data")
