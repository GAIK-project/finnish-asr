"""Download this project's fine-tunes from Hugging Face into $FIASR_MODELS/<name>, where the runners look.

    python scripts/download_models.py [name ...]      (lumi.sh download models runs it on LUMI)

The names are the directory names run_open_models.py and run_benchmark.py turn into ft_<name> arms:

    whisper-large-v3-fi-skn-punct-best    GAIK-project/whisper-large-v3-fi-dialects, main (tag v1.0)
    whisper-large-v3-fi-skn-punct-ts      the same repository, revision timestamps
    whisper-large-v3-fi-skn-lamurre-best  GAIK-project/whisper-large-v3-fi-dialects-internal (private: HF_TOKEN
                                          with read access to the GAIK-project organisation)

Without names it fetches the two public ones. Weights only (config, tokenizer, safetensors). The internal
repository keeps only the LoRA adapter (adapter/); it is merged into openai/whisper-large-v3 here, as
finetune_whisper.py did when it saved the model.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MODELS = Path(os.environ.get("FIASR_MODELS", Path(__file__).resolve().parents[1] / "models"))
REPOS = {
    "whisper-large-v3-fi-skn-punct-best": ("GAIK-project/whisper-large-v3-fi-dialects", "main"),
    "whisper-large-v3-fi-skn-punct-ts": ("GAIK-project/whisper-large-v3-fi-dialects", "timestamps"),
    "whisper-large-v3-fi-skn-lamurre-best": ("GAIK-project/whisper-large-v3-fi-dialects-internal", "main"),
}


def merge(out: Path, base: str = "openai/whisper-large-v3") -> None:
    """Merge out/adapter into the base model and save the full model and processor in out/."""
    import peft.import_utils
    import peft.tuners.lora.torchao as lora_torchao

    peft.import_utils.is_torchao_available = lambda: False  # as in finetune_whisper.py (LUMI's container)
    lora_torchao.is_torchao_available = lambda: False
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model = WhisperForConditionalGeneration.from_pretrained(base)
    model = PeftModel.from_pretrained(model, out / "adapter").merge_and_unload()
    model.generation_config.language, model.generation_config.task = "fi", "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.save_pretrained(out)
    WhisperProcessor.from_pretrained(base, language="fi", task="transcribe").save_pretrained(out)
    print(f"merged {out / 'adapter'} into {base}", flush=True)


def main() -> int:
    from huggingface_hub import snapshot_download

    names = sys.argv[1:] or ["whisper-large-v3-fi-skn-punct-best", "whisper-large-v3-fi-skn-punct-ts"]
    for name in names:
        repo, rev = REPOS[name]
        out = MODELS / name
        snapshot_download(repo, revision=rev, local_dir=out, allow_patterns=["*.json", "*.safetensors", "*.txt"])
        if not (out / "config.json").exists() and (out / "adapter" / "adapter_config.json").exists():
            merge(out)
        print(f"done {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
