"""Every text file is committed with LF line endings, and every writer in scripts/ says so."""

import ast
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_gitattributes_normalise_text_to_lf() -> None:
    lines = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert "* text=auto eol=lf" in lines


def test_no_committed_file_has_crlf() -> None:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "--eol"], check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    crlf = [line.split("\t", 1)[1] for line in out.splitlines() if "crlf" in line.split("\t", 1)[0]]
    assert crlf == []


def writers_without_newline(path: Path) -> list[int]:
    """Lines of write_text(...) and text-mode open(..., "w" or "a") calls that do not pass newline=."""
    bad = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call) or any(k.arg == "newline" for k in node.keywords):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", None))
        modes = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        text_write = any(set(m) <= set("rwabxt+") and ("w" in m or "a" in m) and "b" not in m for m in modes)
        if name == "write_text" or (name == "open" and text_write):
            bad.append(node.lineno)
    return bad


def test_every_text_writer_writes_lf() -> None:
    found = {p.name: writers_without_newline(p) for p in sorted((ROOT / "scripts").glob("*.py"))}
    assert {name: lines for name, lines in found.items() if lines} == {}
