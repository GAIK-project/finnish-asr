#!/bin/bash
# Cuts a dialect corpus into segments on a CPU node: scripts/prepare_skn.py, or with
# --export=ALL,LUMI_PROJECT=...,FIASR_PREP='prepare_lamurre.py --train' the LA-murre training data.
# lumi/lumi.sh submits it from the fiasr directory on scratch:
#   sbatch --account=$LUMI_PROJECT --export=ALL,LUMI_PROJECT=$LUMI_PROJECT lumi/prep.sbatch
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=fiasr
#SBATCH --output=fiasr-prep-%j.out

LP=${LUMI_PROJECT:?LUMI_PROJECT is not set (lumi/lumi.sh exports it)}
R=/scratch/$LP/$USER/fiasr
SIF=/appl/local/containers/sif-images/lumi-pytorchv0-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif
export PYTHONUNBUFFERED=1 FIASR_DATA=$R/data OMP_NUM_THREADS=1
PREP=${FIASR_PREP:-prepare_skn.py}

cd "$R"
srun singularity exec -B "/scratch/$LP" "$SIF" bash -c "
  \$WITH_CONDA
  cd \"$R\"
  source venv-hf/bin/activate
  python scripts/$PREP --workers 12
"
