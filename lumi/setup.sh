#!/bin/bash
# Builds the three Python environments the GPU runners need, on the LUMI login node,
# inside CSC's PyTorch ROCm container. Each venv sees the container's torch (ROCm build)
# and pins it, so pip never replaces it with a CUDA wheel. This file is the only place the
# GPU dependencies are pinned (pyproject.toml covers the local, CPU-only scripts).
#   hf    transformers 5 (Whisper, Parakeet/Orukeet TDT, wav2vec2), peft for the LoRA fine-tune
#   qwen  qwen-asr (pins its own transformers)
#   nemo  nemo_toolkit[asr] (Canary, RASMUS)
# The container supplies torch 2.7.1+rocm6.2.4, torchaudio, numpy and pyarrow.
# Package specs reach the container through bash -c unquoted: pin with ==, never >= (a redirect).
# Usage (in the fiasr directory on scratch; lumi/lumi.sh setup does this):
#   LUMI_PROJECT=project_<number> bash lumi/setup.sh [hf|qwen|nemo ...]   (default: all three)
set -euo pipefail
LP=${LUMI_PROJECT:?set LUMI_PROJECT, e.g. project_<number>}
R=${R:-/scratch/$LP/$USER/fiasr}
SIF=${SIF:-/appl/local/containers/sif-images/lumi-pytorchv0-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif}
cd "$R"

build() {
  local name=$1; shift
  singularity exec -B "/scratch/$LP" "$SIF" bash -c "
    set -e
    \$WITH_CONDA
    cd \"$R\"   # /scratch is a symlink, so the container does not keep the working directory
    python -m venv --system-site-packages venv-$name
    source venv-$name/bin/activate
    python -c 'import torch; print(\"torch==\"+torch.__version__)' > venv-$name/constraints.txt
    pip install --quiet --upgrade pip
    pip install --quiet -c venv-$name/constraints.txt $*
    python -c 'import torch, sys; print(\"$name ok\", sys.version.split()[0], torch.__version__)'
  "
}

for env in "${@:-hf qwen nemo}"; do
  for e in $env; do
    case $e in
      hf)   build hf transformers==5.19.0 soundfile librosa huggingface_hub peft==0.21.2 ;;  # no KenLM: it does not build here
      qwen) build qwen qwen-asr soundfile ;;
      nemo) build nemo 'nemo_toolkit[asr]==3.0.0' soundfile ;;
      *) echo "unknown env $e"; exit 2 ;;
    esac
  done
done
