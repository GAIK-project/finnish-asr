"""Fine-tune Whisper on open Finnish speech with LoRA (GPU job on LUMI): a pilot of the training loop.

Training data, both freely licensed and already on the Hub:
    google/fleurs fi_fi train     read Wikipedia sentences, CC-BY-4.0
    facebook/voxpopuli fi train   European Parliament speeches, CC0 (rows with a punctuated transcript)
    SKN (--data skn or skn+open)  dialect interviews from 50 municipalities, CC BY 4.0, cut by
                                  prepare_skn.py; target = the normalised standard-Finnish text,
                                  or with --skn-text punct the same words with punctuation and
                                  capitals added by an LLM (prepare_targets.py punct)
    LA-murre (--data ...+lamurre) dialect interviews, CC BY-ND 4.0, cut by prepare_lamurre.py --train;
                                  target = its dialect transcript rewritten in standard Finnish by
                                  an LLM (prepare_targets.py normalise). This trains the internal
                                  model, which is never published (no-derivatives licence).
With --timestamps P a share P of the SKN items is trained with Whisper's timestamp tokens instead
of <|notimestamps|>: "<|0.00|> text<|1.84|><|1.84|> text<|5.60|>", one pair per sentence, the times
from SKN's annotation (prepare_timestamps.py -> results/labels/skn_parts.jsonl, which
`lumi.sh push-labels` copies to $FIASR_DATA/skn_fi/parts.jsonl). Whisper was trained
this way, and its long-form decoding needs the tokens; a model fine-tuned without them loses its
place in long audio. Which items get timestamps is drawn again every epoch; for validation it is
fixed (every other segment by a hash of its id). FLEURS and VoxPopuli have no times and stay
<|notimestamps|>.

--data skn-all+open adds the rest of SKN (prepare_skn.py --heldout: every segment of the 12 test and
validation municipalities, with heldout.targets.jsonl for --skn-text punct) to the training data, so the
model hears all 50 municipalities. Its validation loss then comes from FLEURS and VoxPopuli only, and
the SKN test no longer measures it.

Their validation splits give the loss that is logged after each epoch. The test splits and
the Ajokortti clip are never read here; run_benchmark.py and run_open_models.py score the
result on them.

After every epoch the validation loss is measured, and the epoch with the lowest loss is the one
that is merged and saved (4 epochs overfit: the loss was lowest after the second).

LoRA (rank 32 on the attention query and value projections) trains about 1 % of the weights,
so the base model's general ability is disturbed less than by full fine-tuning and the
adapter is a small file. At the end the adapter is merged into the base weights and the
whole model is saved under $FIASR_MODELS/<name>, where the two runners find it as arm
ft_<name>.

A plain PyTorch loop, not the transformers Trainer: the container's accelerate is older than
the Trainer in transformers 5 expects.

Inputs on LUMI ($FIASR_DATA): skn_fi/{train,validation}.jsonl and audio from prepare_skn.py, and for
--skn-text punct skn_fi/{train,validation}.targets.jsonl, the LLM-punctuated targets committed in
results/labels/ (`lumi.sh push-labels` copies them there; prepare_targets.py made them).

    python finetune_whisper.py --download                  # login node: the training data
    python finetune_whisper.py --name whisper-large-v3-fi-pilot [--base openai/whisper-large-v3]
        [--epochs 2] [--lr 1e-4] [--batch 8] [--limit N] [--data open|skn|skn+open|skn-all+open|skn+open+lamurre]
        [--skn-text plain|punct] [--timestamps 0.5]

Output: $FIASR_MODELS/<name>/ (merged model, processor, adapter/, train_log.json). `lumi.sh fetch-model
<name>` copies train_log.json into results/finetune/<name>/ (the adapter only with --adapter, gitignored).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import platform
import random
import sys
import tarfile
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SR = 16000
MAX_SECONDS = 30.0
ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("FIASR_DATA", ROOT / "data"))
MODELS = Path(os.environ.get("FIASR_MODELS", ROOT / "models"))
VOX_FILES = {"train": ["fi/train/0000.parquet", "fi/train/0001.parquet"], "validation": ["fi/validation/0000.parquet"]}


def download() -> None:
    from huggingface_hub import hf_hub_download

    out = DATA / "fleurs_fi"
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        tsv = hf_hub_download("google/fleurs", f"data/fi_fi/{split}.tsv", repo_type="dataset")
        (out / f"{split}.tsv").write_bytes(Path(tsv).read_bytes())
        if not (out / split).exists():
            with tarfile.open(
                hf_hub_download("google/fleurs", f"data/fi_fi/audio/{split}.tar.gz", repo_type="dataset")
            ) as tf:
                tf.extractall(out, filter="data")
    for files in VOX_FILES.values():
        for name in files:
            hf_hub_download("facebook/voxpopuli", name, repo_type="dataset", revision="refs/convert/parquet")
    print("done")


def fleurs(split: str) -> list[dict]:
    items = []
    with open(DATA / "fleurs_fi" / f"{split}.tsv", encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            path = DATA / "fleurs_fi" / split / row[1]
            if path.exists() and row[2].strip():
                items.append({"source": "fleurs", "path": str(path), "text": row[2].strip()})
    return items


def voxpopuli(split: str) -> list[dict]:
    """Rows with a punctuated transcript only: lower-cased text would teach Whisper to drop punctuation."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    items = []
    for name in VOX_FILES[split]:
        path = hf_hub_download("facebook/voxpopuli", name, repo_type="dataset", revision="refs/convert/parquet")
        for row in pq.read_table(path, columns=["audio", "raw_text"]).to_pylist():
            if (row["raw_text"] or "").strip():
                items.append({"source": "voxpopuli", "bytes": row["audio"]["bytes"], "text": row["raw_text"].strip()})
    return items


def skn(split: str, text: str = "plain") -> list[dict]:
    """Segments from prepare_skn.py. plain: the normalised text, no punctuation, mostly lower-case.
    punct: <split>.targets.jsonl from prepare_targets.py, the same words with punctuation and capitals."""
    name, field = (f"{split}.jsonl", "ref") if text == "plain" else (f"{split}.targets.jsonl", "target")
    rows = (DATA / "skn_fi" / name).read_text(encoding="utf-8").splitlines()
    d = DATA / "skn_fi" / split
    parts_path = DATA / "skn_fi" / "parts.jsonl"
    parts = {}
    if parts_path.exists():
        for line in parts_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            parts[r["id"]] = r["parts"]
    return [
        {
            "source": "skn",
            "id": r["id"],
            "path": str(d / r["id"]),
            "text": r[field],
            "parts": parts.get(r["id"].removesuffix(".wav")),
        }
        for r in map(json.loads, rows)
    ]


def timestamped(text: str, parts: list | None) -> str | None:
    """The text as Whisper timestamp segments, one per SKN sentence; None if the words do not line up.

    Parts count words with a letter or digit (prepare_timestamps.py); other tokens, such as a dash,
    stay with the word before them."""
    tokens = text.split()
    is_word = [any(c.isalnum() for c in t) for t in tokens]
    if not parts or sum(p[2] for p in parts) != sum(is_word):
        return None
    out, i, prev = [], 0, 0.0
    for k, (start, end, n) in enumerate(parts):
        piece, seen = [], 0
        while i < len(tokens) and (seen < n or (k == len(parts) - 1) or not is_word[i]):
            if is_word[i] and seen == n:
                break
            piece.append(tokens[i])
            seen += is_word[i]
            i += 1
        q0 = max(prev, round(min(start, 29.98) / 0.02) * 0.02)
        q1 = round(min(end, 30.0) / 0.02) * 0.02
        if q1 <= q0:
            q1 = q0 + 0.02
        out.append(f"<|{q0:.2f}|> {' '.join(piece)}<|{q1:.2f}|>")
        prev = q1
    return "".join(out) if prev <= 30.0 and i == len(tokens) else None


def lamurre(split: str) -> list[dict]:
    """LA-murre segments whose dialect text an LLM rewrote in standard Finnish (prepare_targets.py normalise)."""
    rows = (DATA / "lamurre_fi" / f"{split}.targets.jsonl").read_text(encoding="utf-8").splitlines()
    d = DATA / "lamurre_fi" / split
    return [{"source": "lamurre", "path": str(d / r["id"]), "text": r["target"]} for r in map(json.loads, rows)]


def read_audio(item: dict):
    import soundfile as sf

    audio, sr = sf.read(item["path"] if "path" in item else io.BytesIO(item["bytes"]), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        raise ValueError(f"{sr} Hz audio, expected {SR}")
    return audio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--name", default="whisper-large-v3-fi-pilot")
    parser.add_argument(
        "--data", choices=["open", "skn", "skn+open", "skn-all+open", "skn+open+lamurre"], default="open"
    )
    parser.add_argument("--skn-text", choices=["plain", "punct"], default="plain")
    parser.add_argument("--base", default="openai/whisper-large-v3")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--limit", type=int, help="Only N training items (smoke test).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timestamps", type=float, default=0.0, help="Share of SKN items trained with timestamps.")
    parser.add_argument("--env", help="Ignored: lumi/job.sbatch passes it to every runner.")
    args = parser.parse_args()
    if args.download:
        download()
        return 0

    import peft.import_utils
    import peft.tuners.lora.torchao as lora_torchao
    import torch
    from peft import LoraConfig, get_peft_model

    # LUMI's container ships a torchao that peft rejects with an ImportError. Nothing here is quantised.
    peft.import_utils.is_torchao_available = lambda: False
    lora_torchao.is_torchao_available = lambda: False
    from torch.utils.data import DataLoader, Dataset
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(args.base, language="fi", task="transcribe")
    tok, fe = processor.tokenizer, processor.feature_extractor
    sot, fi, transcribe, eot = (
        tok.convert_tokens_to_ids(t) for t in ("<|startoftranscript|>", "<|fi|>", "<|transcribe|>", "<|endoftext|>")
    )
    if tok.convert_tokens_to_ids("<|0.00|>") != tok.convert_tokens_to_ids("<|notimestamps|>") + 1:
        raise RuntimeError("unexpected timestamp token ids")

    class Speech(Dataset):
        def __init__(self, items: list[dict], train: bool = True):
            self.items, self.train = items, train

        def with_timestamps(self, it: dict) -> bool:
            if not args.timestamps or it.get("parts") is None:
                return False
            if not self.train:  # a fixed half of the validation segments
                return zlib.crc32(it["id"].encode()) % 2 == 0
            return random.Random(f"{it['id']}-{torch.initial_seed()}").random() < args.timestamps

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, i: int):
            it = self.items[i]
            audio = read_audio(it)[: int(MAX_SECONDS * SR)]
            feats = fe(audio, sampling_rate=SR, return_tensors="pt").input_features[0]
            if self.with_timestamps(it):
                stamped = timestamped(it["text"], it["parts"])
                if stamped:
                    ids = [sot, fi, transcribe] + tok(stamped, add_special_tokens=False).input_ids + [eot]
                    return feats, ids[:448]
            return feats, tok(it["text"]).input_ids[:448]

    def collate(batch):
        feats = torch.stack([b[0] for b in batch])
        width = max(len(b[1]) for b in batch)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        for i, (_, ids) in enumerate(batch):
            labels[i, : len(ids)] = torch.tensor(ids)
        return feats, labels[:, 1:]  # drop <|startoftranscript|>: the model prepends it when it shifts the labels

    train_items, val_items = [], []
    if args.data.startswith("skn-all"):  # all 50 municipalities; the SKN validation speakers are in training
        train_items += skn("train", args.skn_text) + skn("heldout", args.skn_text)
    elif "skn" in args.data:
        train_items += skn("train", args.skn_text)
        val_items += skn("validation", args.skn_text)
    if "lamurre" in args.data:
        train_items += lamurre("train")
        val_items += lamurre("validation")
    if "open" in args.data:
        train_items += fleurs("train") + voxpopuli("train")
        val_items += fleurs("dev") + voxpopuli("validation")
    random.shuffle(train_items)
    if args.limit:
        train_items, val_items = train_items[: args.limit], val_items[: max(16, args.limit // 10)]
    counts = {s: sum(1 for it in train_items if it["source"] == s) for s in ("skn", "lamurre", "fleurs", "voxpopuli")}
    if args.timestamps:
        usable = sum(1 for it in train_items if timestamped(it["text"], it.get("parts")))
        counts["skn_with_times"] = usable
        print(f"timestamps: {usable} SKN items have sentence times, {args.timestamps:.0%} drawn per epoch", flush=True)
    print(f"device={dev} train={len(train_items)} {counts} val={len(val_items)}", flush=True)

    model = WhisperForConditionalGeneration.from_pretrained(args.base)
    model.generation_config.language, model.generation_config.task = "fi", "transcribe"
    model.generation_config.forced_decoder_ids = None
    lora = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
    model = get_peft_model(model, lora).to(dev)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable {trainable / 1e6:.1f} M of {total / 1e6:.0f} M ({trainable / total:.2%})", flush=True)

    loader = DataLoader(
        Speech(train_items), batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=6, drop_last=True
    )
    val_loader = DataLoader(Speech(val_items, train=False), batch_size=args.batch, collate_fn=collate, num_workers=6)
    steps = len(loader) * args.epochs
    warmup = max(1, int(0.05 * steps))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (steps - s) / max(1, steps - warmup))
    )
    amp = {"device_type": "cuda", "dtype": torch.bfloat16, "enabled": dev == "cuda"}

    def val_loss() -> float:
        model.eval()
        losses, n = 0.0, 0
        with torch.no_grad():
            for feats, labels in val_loader:
                with torch.autocast(**amp):
                    out = model(input_features=feats.to(dev), labels=labels.to(dev))
                losses += out.loss.item() * len(feats)
                n += len(feats)
        model.train()
        return losses / n

    out_dir = MODELS / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log = {"args": vars(args), "train_items": counts, "val_items": len(val_items), "epochs": []}
    log["val_loss_before"] = round(val_loss(), 4)
    print(f"val loss before training: {log['val_loss_before']}", flush=True)
    step, t0 = 0, time.perf_counter()
    best_loss, best_state = float("inf"), {}
    model.train()
    for epoch in range(1, args.epochs + 1):
        running, seen = 0.0, 0
        for feats, labels in loader:
            with torch.autocast(**amp):
                loss = model(input_features=feats.to(dev), labels=labels.to(dev)).loss
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"loss is {loss.item()} at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            running += loss.item()
            seen += 1
            if step % 50 == 0:
                mins = (time.perf_counter() - t0) / 60
                print(f"epoch {epoch} step {step}/{steps} loss {running / seen:.4f} ({mins:.1f} min)", flush=True)
                running, seen = 0.0, 0
        vl = round(val_loss(), 4)
        log["epochs"].append(
            {"epoch": epoch, "step": step, "val_loss": vl, "minutes": round((time.perf_counter() - t0) / 60, 1)}
        )
        print(f"epoch {epoch} done: val loss {vl}", flush=True)
        if vl < best_loss:  # the saved model is the epoch with the lowest validation loss, not the last
            best_loss, log["best_epoch"] = vl, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
            model.save_pretrained(out_dir / "adapter")
        (out_dir / "train_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8", newline="\n")

    with torch.no_grad():
        for k, v in model.named_parameters():
            if k in best_state:
                v.copy_(best_state[k])
    print(f"merging epoch {log['best_epoch']} (val loss {best_loss})", flush=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    log["gpu"] = torch.cuda.get_device_name(0) if dev == "cuda" else None
    log["base"] = args.base
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8", newline="\n")
    from provenance import build_run_metadata, write_run_metadata  # standard library only

    write_run_metadata(
        out_dir,
        build_run_metadata(
            app_under_test="Finnish dialect ASR: LoRA fine-tune of Whisper",
            dataset_path=None,
            runner_path=Path(__file__),
            model_or_deployment=[args.base],
            extra={"name": args.name, "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
        ),
    )
    print(f"saved {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
