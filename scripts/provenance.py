"""Secret-free run metadata (``_run_metadata*.json``) for every runner that writes results.

Standard library only, so the LUMI environments, which have no python-dotenv, can import it too.
Environment variables are recorded only as "set" or "unset", never their values, and paths are
written relative to the repository root.

Adapted from the provenance helper of GAIK's internal evaluation repository. Result files made
before this repository existed carry ``gaik_evals_commit``: a commit of that private repository.
New files carry ``repo_commit`` of this one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = "GAIK-project/finnish-asr"


def repo_root(start: Path | None = None) -> Path:
    """The repository root: the first directory upwards with both pyproject.toml and scripts/."""
    current = (start or Path(__file__)).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "scripts").is_dir():
            return candidate
    return Path(__file__).resolve().parents[1]


def load_repo_env(start: Path | None = None) -> Path:
    """Load the repo-root .env (keys for the API runners) and return the root.

    python-dotenv is imported here, not at module level, so that the GPU environments can import
    this module without it.
    """
    from dotenv import load_dotenv

    root = repo_root(start)
    load_dotenv(root / ".env", override=True)
    return root


def _git_output(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _rel(root: Path, path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    try:
        return p.resolve().relative_to(root).as_posix()
    except ValueError:
        return p.name  # outside the repository: the name only, never a local absolute path


def _git_object_id(root: Path, path: Path | str | None) -> str | None:
    rel = _rel(root, path)
    if not rel:
        return None
    return _git_output(root, "rev-parse", f"HEAD:{rel}")


def build_run_metadata(
    *,
    app_under_test: str,
    dataset_path: Path | str | None,
    runner_path: Path | str,
    model_or_deployment: str | list[str] | None = None,
    cli_args: list[str] | None = None,
    env_names: list[str] | tuple[str, ...] = (),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A provenance block for one run: commit, runner, dataset, models, arguments, which keys were set."""
    root = repo_root(Path(runner_path))
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "repo": REPO,
        "repo_commit": _git_output(root, "rev-parse", "HEAD"),
        "repo_dirty": bool(_git_output(root, "status", "--porcelain")),
        "app_under_test": app_under_test,
        "runner_path": _rel(root, runner_path),
        "dataset_path": _rel(root, dataset_path),
        "dataset_git_object": _git_object_id(root, dataset_path),
        "model_or_deployment": model_or_deployment,
        "cli_args": list(sys.argv[1:] if cli_args is None else cli_args),
        "env": {name: "set" if os.environ.get(name) else "unset" for name in env_names},
    }
    if extra:
        metadata["extra"] = extra
    return metadata


def write_json(path: Path, data: Any) -> None:
    """Deterministic UTF-8 JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def write_run_metadata(results_dir: Path, metadata: dict[str, Any], name: str = "_run_metadata.json") -> Path:
    """Write the metadata sidecar into results_dir and return its path.

    ``name`` lets several runners share one folder without overwriting each other's sidecar.
    """
    if not name.startswith("_") or not name.endswith(".json"):
        raise ValueError(f"metadata file name must look like _*.json, got {name!r}")
    out = results_dir / name
    write_json(out, metadata)
    return out
