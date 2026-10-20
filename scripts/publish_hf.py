"""Publish the dialect model and its SKN dataset to the GAIK-project organisation on Hugging Face.

For maintainers: every command needs HF_TOKEN in the environment, a fine-grained token with write
access to GAIK-project (it is never written to disk). Readers do not need this script; the model and
dataset are public on the Hub. model, dataset and check run on the LUMI login node (it has the
internet and the files), in the hf environment:

    python scripts/publish_hf.py model      # weights, processor, card -> GAIK-project/whisper-large-v3-fi-dialects
    python scripts/publish_hf.py model --revision timestamps   # the timestamp variant -> its "timestamps" branch
    python scripts/publish_hf.py model --revision all-skn      # trained on all 50 SKN municipalities -> "all-skn"
    python scripts/publish_hf.py model --revision la-murre     # + 56 h of LA-murre interviews -> "la-murre"
    python scripts/publish_hf.py dataset    # SKN segments, parquet with audio -> GAIK-project/skn-finnish-dialects-asr
    python scripts/publish_hf.py check      # the model, downloaded back from the Hub, on a few SKN test segments
    uv run --extra hub python scripts/publish_hf.py cards              # the three cards and the chart (anywhere)
    uv run --extra hub python scripts/publish_hf.py dialect <eaf dir>  # fill text_dialect in the parquet shards

Repositories are created PRIVATE; making them public is a separate, manual decision. The cards come
from release/: MODEL_CARD.md is the README of the model's main branch, TIMESTAMPS_CARD.md the README of
its "timestamps" branch (the model trained with timestamp tokens, finetune_whisper.py --timestamps 0.5),
ALL_SKN_CARD.md of its "all-skn" branch (finetune_whisper.py --data skn-all+open), LAMURRE_CARD.md of
its "la-murre" branch (finetune_whisper.py --data skn+open+lamurre), and DATASET_CARD.md the dataset's.
"model --revision timestamps" first tags main as v1.0 (the model published on 13 Oct 2026), then
creates the branch and uploads the variant there. The "la-murre" branch holds merged weights only:
the LA-murre texts it was trained on (LLM rewrites of a CC BY-ND corpus) are never uploaded here.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("FIASR_DATA", ROOT / "data"))
MODELS = Path(os.environ.get("FIASR_MODELS", ROOT / "models"))
RELEASE = Path(__file__).resolve().parents[1] / "release"
ORG = "GAIK-project"
MODEL_REPO = f"{ORG}/whisper-large-v3-fi-dialects"
DATASET_REPO = f"{ORG}/skn-finnish-dialects-asr"
MODEL_DIR = MODELS / "whisper-large-v3-fi-skn-punct-best"
# revision -> (model directory, card); None is the main branch
REVISIONS = {
    None: (MODEL_DIR, "MODEL_CARD.md"),
    "timestamps": (MODELS / "whisper-large-v3-fi-skn-punct-ts", "TIMESTAMPS_CARD.md"),
    "all-skn": (MODELS / "whisper-large-v3-fi-skn-all", "ALL_SKN_CARD.md"),
    "la-murre": (MODELS / "whisper-large-v3-fi-skn-lamurre-best", "LAMURRE_CARD.md"),
}
MAIN_TAG = "v1.0"
SHARD_ROWS = 1500


def publish_model(revision: str | None = None) -> None:
    from huggingface_hub import HfApi

    model_dir, card = REVISIONS[revision]
    api = HfApi()
    api.create_repo(MODEL_REPO, private=True, exist_ok=True)
    if revision:
        api.create_tag(MODEL_REPO, tag=MAIN_TAG, tag_message="The published model of 13 Oct 2026", exist_ok=True)
        api.create_branch(MODEL_REPO, branch=revision, exist_ok=True)
    files = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "train_log.json",
    ]
    for name in files:
        api.upload_file(path_or_fileobj=str(model_dir / name), path_in_repo=name, repo_id=MODEL_REPO, revision=revision)
        print(f"uploaded {name}", flush=True)
    api.upload_file(
        path_or_fileobj=str(RELEASE / card), path_in_repo="README.md", repo_id=MODEL_REPO, revision=revision
    )
    print(f"done: https://huggingface.co/{MODEL_REPO} ({revision or 'main'}, private)")


def publish_cards() -> None:
    """The charts and MODEL_CARD.md to the model's main branch, each variant's card to its branch,
    DATASET_CARD.md to the dataset."""
    from huggingface_hub import HfApi

    api = HfApi()
    for chart in ("dialect-finetune.svg", "dialect-ranking.svg"):
        path = RELEASE.parent / "docs" / "img" / chart
        api.upload_file(path_or_fileobj=str(path), path_in_repo=chart, repo_id=MODEL_REPO)
    for revision, (_, card) in REVISIONS.items():
        api.upload_file(
            path_or_fileobj=str(RELEASE / card), path_in_repo="README.md", repo_id=MODEL_REPO, revision=revision
        )
        print(f"uploaded {card} -> {MODEL_REPO} ({revision or 'main'})", flush=True)
    api.upload_file(
        path_or_fileobj=str(RELEASE / "DATASET_CARD.md"),
        path_in_repo="README.md",
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print("cards uploaded")


def dialect_texts(eaf_dir: Path) -> dict[str, str]:
    """Segment id -> SKN dialect text, recomputed from SKN's *_full.eaf files with prepare_skn."""
    from prepare_skn import segments, sentences

    out = {}
    for f in sorted(eaf_dir.glob("*_full.eaf")):
        rec = f.name.removesuffix("_full.eaf")
        for i, g in enumerate(segments(sentences(f.read_bytes()))):
            out[f"{rec}_{i:04d}"] = (g["ref"], g["orig"] or "")
    return out


def publish_dialect(eaf_dir: Path) -> None:
    """Rewrite every shard with text_dialect filled, one commit per shard.

    The first upload had an empty text_dialect (prepare_skn matched tiers by time-slot id). The
    segments themselves are unchanged: each row's text_normalized must equal the recomputed one,
    or the shard is not uploaded. Shards are written with the same writer settings as
    publish_dataset. One shard at a time, each downloaded into its own folder that is deleted
    after the upload, so the run needs about 1.5 GB of free disk. Already-filled shards are skipped.
    """
    import shutil
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    texts = dialect_texts(eaf_dir)
    api = HfApi()
    shards = sorted(f for f in api.list_repo_files(DATASET_REPO, repo_type="dataset") if f.endswith(".parquet"))
    filled = empty = 0
    for name in shards:
        tmp = Path(tempfile.mkdtemp(prefix="skn_dialect_"))
        try:
            src = hf_hub_download(DATASET_REPO, name, repo_type="dataset", cache_dir=tmp / "cache")
            table = pq.read_table(src)
            ids, norm = table.column("id").to_pylist(), table.column("text_normalized").to_pylist()
            bad = [i for i, n in zip(ids, norm, strict=True) if i not in texts or texts[i][0] != n]
            if bad:
                raise SystemExit(f"{name}: {len(bad)} rows do not match the annotations, e.g. {bad[:3]}")
            new = [texts[i][1] for i in ids]
            filled += sum(1 for t in new if t)
            empty += sum(1 for t in new if not t)
            if table.column("text_dialect").to_pylist() == new:
                print(f"{name}: already filled", flush=True)
                continue
            col = table.schema.get_field_index("text_dialect")
            table = table.set_column(col, table.schema.field(col), pa.array(new, pa.string()))
            out = tmp / "out.parquet"
            pq.write_table(table, out, compression="zstd")
            del table
            shutil.rmtree(tmp / "cache")
            api.upload_file(
                path_or_fileobj=str(out),
                path_in_repo=name,
                repo_id=DATASET_REPO,
                repo_type="dataset",
                commit_message=f"Fill text_dialect from SKN's dialect sentence tier ({name})",
            )
            print(f"{name}: {len(new)} rows uploaded", flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"done: {filled} rows with dialect text, {empty} empty (no time-aligned dialect sentence in SKN)")


def rows(split: str) -> list[dict]:
    plain = [json.loads(x) for x in (DATA / "skn_fi" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    target_path = DATA / "skn_fi" / f"{split}.targets.jsonl"
    targets = {}
    if target_path.exists():
        for line in target_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            targets[r["id"]] = (r["target"], "llm-punctuation" if r.get("target_by") != "source" else "human")
    out = []
    for r in plain:
        text, source = targets.get(r["id"], (r["ref"], "human"))
        out.append({**r, "text": text, "text_source": source})
    return out


def publish_dataset() -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(DATASET_REPO, repo_type="dataset", private=True, exist_ok=True)
    schema = pa.schema(
        [
            ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
            ("id", pa.string()),
            ("municipality", pa.string()),
            ("skn_section", pa.int32()),
            ("seconds", pa.float32()),
            ("text", pa.string()),
            ("text_normalized", pa.string()),
            ("text_dialect", pa.string()),
            ("text_source", pa.string()),
        ]
    )
    for split in ("train", "validation", "test"):
        rs = rows(split)
        shards = max(1, (len(rs) + SHARD_ROWS - 1) // SHARD_ROWS)
        for k in range(shards):
            part = rs[k * SHARD_ROWS : (k + 1) * SHARD_ROWS]
            table = pa.Table.from_pylist(
                [
                    {
                        "audio": {"bytes": (DATA / "skn_fi" / split / r["id"]).read_bytes(), "path": r["id"]},
                        "id": r["id"].removesuffix(".wav"),
                        "municipality": r["municipality"],
                        "skn_section": r["number"],
                        "seconds": r["seconds"],
                        "text": r["text"],
                        "text_normalized": r["ref"],
                        "text_dialect": r.get("orig", ""),
                        "text_source": r["text_source"],
                    }
                    for r in part
                ],
                schema=schema,
            )
            buf = io.BytesIO()
            pq.write_table(table, buf, compression="zstd")
            name = f"data/{split}-{k:05d}-of-{shards:05d}.parquet"
            api.upload_file(
                path_or_fileobj=buf.getvalue(), path_in_repo=name, repo_id=DATASET_REPO, repo_type="dataset"
            )
            print(f"uploaded {name}: {len(part)} rows", flush=True)
    api.upload_file(
        path_or_fileobj=str(RELEASE / "DATASET_CARD.md"),
        path_in_repo="README.md",
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print(f"done: https://huggingface.co/datasets/{DATASET_REPO} (private)")


def check(files_only: bool = False) -> None:
    """Download the published model into an empty cache and transcribe 8 SKN test segments with it."""
    import tempfile

    import soundfile as sf
    from huggingface_hub import snapshot_download

    with tempfile.TemporaryDirectory() as tmp:
        path = snapshot_download(MODEL_REPO, cache_dir=tmp)
        names = sorted(p.name for p in Path(path).iterdir())
        print("files from the Hub:", names, flush=True)
        if files_only:
            return
        from transformers import pipeline

        asr = pipeline("automatic-speech-recognition", model=path)
        for r in rows("test")[:8]:
            audio, sr = sf.read(DATA / "skn_fi" / "test" / r["id"], dtype="float32")
            hyp = asr({"raw": audio, "sampling_rate": sr}, generate_kwargs={"language": "fi", "task": "transcribe"})
            print(json.dumps({"ref": r["ref"][:120], "hyp": hyp["text"][:120]}, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=["model", "dataset", "check", "cards", "dialect"])
    parser.add_argument("eaf_dir", nargs="?", type=Path, help="For dialect: the directory with SKN's *_full.eaf files.")
    parser.add_argument("--files-only", action="store_true", help="For check: only list the files on the Hub.")
    parser.add_argument(
        "--revision", choices=[r for r in REVISIONS if r], help="For model: upload this variant to its branch."
    )
    args = parser.parse_args()
    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN is not set")
        return 1
    if args.cmd == "model":
        publish_model(args.revision)
    elif args.cmd == "dataset":
        publish_dataset()
    elif args.cmd == "check":
        check(files_only=args.files_only)
    elif args.cmd == "cards":
        publish_cards()
    elif args.cmd == "dialect":
        if args.eaf_dir is None:
            parser.error("dialect needs the directory with SKN's *_full.eaf files")
        publish_dialect(args.eaf_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
