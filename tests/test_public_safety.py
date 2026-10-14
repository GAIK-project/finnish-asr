"""What git would publish: no licence-restricted files, no clip copies, no secrets or local details.

Runs on `git ls-files --cached --others --exclude-standard`, i.e. every tracked file plus every new
file that .gitignore does not exclude, so it also catches a file before it is committed.
"""

import re
import subprocess
from fnmatch import fnmatch
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def publishable() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [line for line in out.splitlines() if line and (ROOT / line).exists()]


FORBIDDEN_PATHS = [
    "results/benchmarks/lamurre_fi/*.jsonl",  # LA-murre references paired with transcripts
    "results/judgements_dialect/*.jsonl",  # LA-murre judge records quote the references
    "results/judgements_dialect/*/*.jsonl",
    "results/labels/lamurre_*.jsonl",  # LA-murre texts and their LLM rewrites
    "results/judgements/*/hy2[56]_*.json",  # quote HY's non-public tool transcripts
    "results/transcriptions/hy2[56]_*",
    "results/transcriptions/*/hy2[56]_*",
    "results/finetune/*/adapter/*",  # weights, the internal LA-murre model among them
    "results/compare.html",  # embeds the clip
    "results/gpu_run/*.tar.gz",  # LUMI job logs
    "data/*",
    "models/*",
    ".env",
    "*.safetensors",
    "*.npy",
    "*.wav",
    "*.flac",
]
ALLOWED_AUDIO = {"docs/story/src/assets/heinola_0097.mp3", "docs/story/src/assets/palkane_0021.mp3"}  # SKN, CC BY


def test_no_licence_restricted_or_private_files() -> None:
    files = publishable()
    assert "README.md" in files and "scripts/provenance.py" in files
    bad = [f for f in files if any(fnmatch(f, pattern) for pattern in FORBIDDEN_PATHS)]
    assert bad == []
    audio = [f for f in files if f.lower().endswith((".mp3", ".m4a", ".ogg"))]
    assert set(audio) == ALLOWED_AUDIO


def test_lamurre_results_are_numbers_and_metadata_only() -> None:
    files = [
        f
        for f in publishable()
        if "/lamurre_fi/" in f or f.startswith(("results/labels/lamurre", "results/judgements_dialect/"))
    ]
    assert files
    for f in files:
        name = f.rsplit("/", 1)[-1]
        assert name.endswith(".json"), f
        assert (
            name == "scores.json"
            or name.endswith((".meta.json", ".stats.json"))
            or name.startswith(("summary", "agreement", "_run_metadata"))
        ), f


TEXT_SUFFIXES = {".py", ".md", ".json", ".jsonl", ".sh", ".sbatch", ".toml", ".cff", ".txt", ".html", ".css", ".svg"}
EMAIL = re.compile(r"(?<![$\w.])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
FORBIDDEN_TEXT = [
    re.compile(r"[A-Za-z]:[\\/]+Users[\\/]"),  # Windows home paths
    re.compile(r"/home/[a-z]"),
    re.compile(r"/scratch/project_\d"),  # a concrete LUMI project's scratch
    re.compile(r"project_46\d{5}"),  # a concrete LUMI project number
    re.compile(r"\bsk-(?:ant|proj)-[A-Za-z0-9]"),  # API keys
    re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]


def test_no_local_paths_keys_or_email_addresses_in_text_files() -> None:
    hits = []
    for f in publishable():
        if Path(f).suffix not in TEXT_SUFFIXES and Path(f).name not in {".gitignore", ".env.example", "LICENSE"}:
            continue
        text = (ROOT / f).read_text(encoding="utf-8", errors="replace")
        text = re.sub(r"base64,[A-Za-z0-9+/=]+", "", text)  # inlined images and audio of the story pages
        hits += [(f, p.pattern) for p in FORBIDDEN_TEXT if p.search(text)]
        hits += [(f, m) for m in EMAIL.findall(text)]
    assert hits == []


def test_env_example_holds_names_only() -> None:
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            name, _, value = line.partition("=")
            assert name.isupper() and value in {"", "global"}, line
