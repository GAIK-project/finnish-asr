"""Shared paths and result I/O for the Finnish ASR evaluations. Standard library only.

The webinar test case is the 180 s Ajokortti clip and the University of Helsinki's Finnish reference
transcript, both public in GAIK-project/gaik-toolkit (evaluation_layer/eval_methods/transcription_eval/
data). They are not copied into this repository: ``scripts/get_clip.py`` downloads them into
data/ajokortti/ (gitignored) and checks their SHA-256 against CLIP_FILES.

Every model run on the clip is one ``ArmResult`` JSON ("arm" = one model with one fixed configuration)
in results/transcriptions/. results/transcriptions/qad1/ holds the transcripts of GAIK's earlier runs
of the same clip (QAD-1: OpenAI, Gemini, the Haaga-Helia WhisperX server, Whisper on CSC Roihu), copied
from GAIK's internal evaluation repository in the same format, so they load next to the new ones.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT  # older name, kept for the scripts that import it

CLIP_DIR = REPO_ROOT / "data" / "ajokortti"  # gitignored; filled by get_clip.py
CASE_DIR = CLIP_DIR
AUDIO_PATH = CLIP_DIR / "Ajokortti.mp3"
GT_FI_PATH = CLIP_DIR / "reference.txt"
GT_REVIEWED_PATH = CLIP_DIR / "reference.reviewed.txt"
AUDIO_16K = CLIP_DIR / "Ajokortti.16k.npy"  # get_clip.py --prepare-audio

TOOLKIT_RAW = (
    "https://raw.githubusercontent.com/GAIK-project/gaik-toolkit/main/"
    "evaluation_layer/eval_methods/transcription_eval/data/"
)
# file name -> SHA-256 of the public copy (gaik-toolkit main, 54218f5d)
CLIP_FILES = {
    "Ajokortti.mp3": "a1ce6d671010305e894080d880ed960611b658db4077b023e9ebe96254d6ad28",
    "reference.txt": "30c4a98f17ed2cbaed00c2d0aace6184121be7ac25d35213bd2b189f2587f24d",
    "reference.reviewed.txt": "6b3f55a8fcfe2e8889ca59d974478da332918f55fcdba3a79835cc62c62a42a5",
}

RESULTS_DIR = REPO_ROOT / "results"
ASR_DIR = RESULTS_DIR / "transcriptions"
QAD1_ASR_DIR = ASR_DIR / "qad1"
SCORES_PATH = RESULTS_DIR / "scores.json"
JUDGE_DIR = RESULTS_DIR / "judgements"
GT_ALIGNMENT_PATH = RESULTS_DIR / "clip" / "gt_alignment.json"

QAD1_TAG = "[from QAD-1]"
APP_UNDER_TEST = "Finnish ASR model comparison (Ajokortti clip, HY reference via gaik-toolkit)"


@dataclass
class ArmResult:
    """One model's output for the clip, plus everything needed to reproduce it (QAD-1's format)."""

    arm_id: str
    stage: str  # always "asr" here
    provider: str
    model: str
    text: str
    params: dict[str, Any] = field(default_factory=dict)
    input_ref: str = ""  # repo-relative path of the exact input the model received
    source: str = "run"
    label: str = ""  # short human label for tables
    notes: str = ""
    words: list[dict[str, Any]] | None = None  # [{"word", "start", "end"}] when the model returns timing
    latency_s: float | None = None
    usage: dict[str, Any] | None = None
    created_at_utc: str = ""
    raw: dict[str, Any] | None = None  # provider response minus the text, for audit
    repeat: int = 1  # 1 = primary run; repeats of the same arm are saved as <arm_id>__r<n>

    @property
    def base_arm_id(self) -> str:
        return self.arm_id.split("__r", 1)[0]


def rel(path: Path) -> str:
    """Repo-relative POSIX path, for provenance fields; a path outside the repo gives its name only."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_arm(path: Path) -> ArmResult:
    return ArmResult(**json.loads(path.read_text(encoding="utf-8")))


def save_arm(result: ArmResult) -> Path:
    ASR_DIR.mkdir(parents=True, exist_ok=True)
    if not result.created_at_utc:
        result.created_at_utc = now_utc()
    out = ASR_DIR / f"{result.arm_id}.json"
    out.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return out


def load_arms(include_qad1: bool = True) -> list[ArmResult]:
    """Primary runs of this repository's arms, then QAD-1's (tagged in notes) that are not overridden."""
    arms = [load_arm(p) for p in sorted(ASR_DIR.glob("*.json")) if not p.name.startswith("_")]
    arms = [a for a in arms if a.repeat == 1]
    if include_qad1:
        own = {a.arm_id for a in arms}
        for p in sorted(QAD1_ASR_DIR.glob("*.json")):
            if p.name.startswith("_"):
                continue
            a = load_arm(p)
            if a.repeat == 1 and a.arm_id not in own:
                a.notes = (a.notes + " " if a.notes else "") + QAD1_TAG
                arms.append(a)
    return arms


def require_clip(*names: str) -> None:
    """Stop with a hint when the clip files are not downloaded yet."""
    paths = [CLIP_DIR / n for n in (names or ("Ajokortti.mp3", "reference.txt", "reference.reviewed.txt"))]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        hint = (
            "uv run --extra clip python scripts/get_clip.py --prepare-audio"
            if AUDIO_16K.name in missing
            else ("uv run python scripts/get_clip.py")
        )
        raise SystemExit(f"missing {', '.join(missing)} in data/ajokortti/: run {hint}")
