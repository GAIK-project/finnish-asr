#!/bin/bash
# Runs the open ASR arms of one environment on one LUMI GCD (half an MI250X).
# lumi/lumi.sh submits it from the fiasr directory on scratch with the project account:
#   sbatch --account=$LUMI_PROJECT --export=ALL,LUMI_PROJECT=$LUMI_PROJECT lumi/job.sbatch hf   (or qwen / nemo)
# Extra arguments pass through, e.g.  ... lumi/job.sbatch nemo --arm nemo_canary-1b-v2 --force
# FIASR_SCRIPT picks the runner (default run_open_models.py): --export=ALL,FIASR_SCRIPT=run_benchmark.py,...
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=00:45:00
#SBATCH --job-name=fiasr
#SBATCH --output=fiasr-%x-%j.out

ENV=${1:?usage: sbatch lumi/job.sbatch hf|qwen|nemo [args]}; shift
LP=${LUMI_PROJECT:?LUMI_PROJECT is not set (lumi/lumi.sh exports it)}
R=/scratch/$LP/$USER/fiasr
SIF=/appl/local/containers/sif-images/lumi-pytorchv0-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif
export HF_HOME=/scratch/$LP/$USER/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1   # no internet on compute nodes
export MIOPEN_USER_DB_PATH=/tmp/$USER-miopen-$SLURM_JOB_ID MIOPEN_CUSTOM_CACHE_DIR=/tmp/$USER-miopen-$SLURM_JOB_ID
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export FIASR_DATA=$R/data FIASR_MODELS=$R/models
SCRIPT=${FIASR_SCRIPT:-run_open_models.py}

cd "$R"
srun singularity exec -B "/scratch/$LP" "$SIF" bash -c "
  \$WITH_CONDA
  cd \"$R\"   # /scratch is a symlink, so the container does not keep the working directory
  source venv-$ENV/bin/activate
  [ $ENV = nemo ] && export PYTHONPATH=\$PWD/lumi/stubs   # hide aiter, megatron, tensorstore: see lumi/stubs
  rocm-smi --showproductname | grep -m1 'Card series' || true
  python scripts/$SCRIPT --env $ENV $*
"
