"""Synthetic archives keep parliament extraction ordered and archive opens bounded."""

from __future__ import annotations

import io
import json
import wave
import zipfile
from collections import Counter
from pathlib import Path

import pytest

import prepare_parl


@pytest.fixture
def training_archives(tmp_path, monkeypatch):
    raw = tmp_path / "parl_raw"
    raw.mkdir()
    monkeypatch.setattr(prepare_parl, "DATA", tmp_path)
    monkeypatch.setattr(prepare_parl, "RAW", raw)
    with zipfile.ZipFile(raw / "devtest.zip", "w") as archive:
        archive.writestr(prepare_parl.DEVTEST + "unseen_test.list", "2016-test/hidden_unseen.wav\n")
        archive.writestr(
            prepare_parl.DEVTEST + "dev-test-speakers.csv", "new_id,unused,old_id,file_name\n99,x,98,hidden\n"
        )
        archive.writestr(prepare_parl.DEVTEST + "2020-test/007-2020/unused.txt", "")
    audio = io.BytesIO()
    with wave.open(audio, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 64000)
    payload = audio.getvalue()
    rows_by_part = []
    part_names = []
    for part_no in (1, 2):
        path = raw / f"fi-parliament-asr-2015-2020-part{part_no}.zip"
        part_names.append(path.name)
        rows = []
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            # Physical member order deliberately differs from filename order.
            for index in (2, 0, 3, 1):
                uid = f"{part_no:05d}-{part_no:03d}-2019-{index * 400:05d}-{(index + 1) * 400:05d}"
                ref = f"synthetic training clip {part_no} {index}"
                archive.writestr(f"train/{uid}.wav", payload)
                archive.writestr(f"train/{uid}.trn", ref)
                rows.append({"id": f"{uid}.wav", "ref": ref, "speaker": f"{part_no:05d}", "seconds": 4.0})
            # These must be filtered before the seeded draw.
            for uid in ("00099-003-2019-00000-00400", "00003-007-2020-00000-00400"):
                archive.writestr(f"train/{uid}.wav", payload)
                archive.writestr(f"train/{uid}.trn", "excluded synthetic training clip")
        rows_by_part.append(rows)
    # Legacy seed-0 selection takes all four members in part 1 and the first two in part 2.
    return tmp_path, part_names, rows_by_part[0] + rows_by_part[1][:2], payload


def track_archive_opens(monkeypatch):
    real_zip_file = zipfile.ZipFile
    opened = []

    def tracked_zip_file(path, *args, **kwargs):
        archive = real_zip_file(path, *args, **kwargs)
        opened.append((Path(path).name, archive))
        return archive

    monkeypatch.setattr(prepare_parl.zipfile, "ZipFile", tracked_zip_file)
    return opened


def test_train_opens_each_archive_once_per_phase_and_preserves_outputs(training_archives, monkeypatch):
    data, part_names, expected_rows, payload = training_archives
    opened = track_archive_opens(monkeypatch)

    assert prepare_parl.train(24 / 3600) == 0

    rows = [json.loads(line) for line in (data / "parl_train/train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows == expected_rows
    output = data / "parl_train/train"
    assert sorted(path.name for path in output.iterdir()) == sorted(row["id"] for row in expected_rows)
    assert all((output / row["id"]).read_bytes() == payload for row in expected_rows)
    counts = Counter(name for name, _ in opened)
    assert {name: counts[name] for name in part_names} == dict.fromkeys(part_names, 3)
    assert counts["devtest.zip"] == 1
    assert all(archive.fp is None for _, archive in opened)


def test_train_closes_archives_when_extraction_fails(training_archives, monkeypatch):
    opened = track_archive_opens(monkeypatch)
    real_write_bytes = Path.write_bytes

    def fail_output_write(path, payload):
        if path.parent.name == "train":
            raise OSError("synthetic output failure")
        return real_write_bytes(path, payload)

    monkeypatch.setattr(Path, "write_bytes", fail_output_write)

    with pytest.raises(OSError, match="synthetic output failure"):
        prepare_parl.train(24 / 3600)

    assert opened
    assert all(archive.fp is None for _, archive in opened)


def test_recovery_uses_fresh_directory_and_refuses_overwrite(training_archives):
    data, _, expected_rows, payload = training_archives
    original = data / "parl_train" / "train"
    original.mkdir(parents=True)
    sentinel = original / "preserve.wav"
    sentinel.write_bytes(payload)
    assert prepare_parl.train(24 / 3600, "parl_train_recovery_test") == 0
    manifest = data / "parl_train_recovery_test" / "train.jsonl"
    assert [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()] == expected_rows
    assert sentinel.read_bytes() == payload
    with pytest.raises(FileExistsError):
        prepare_parl.train(24 / 3600, "parl_train_recovery_test")
    assert sentinel.read_bytes() == payload


@pytest.mark.parametrize("name", ["../escape", "nested/path", "/absolute", "", ".."])
def test_recovery_rejects_non_dataset_paths(training_archives, name):
    with pytest.raises(ValueError, match="single dataset directory"):
        prepare_parl.train(24 / 3600, name)
