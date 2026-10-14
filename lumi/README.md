# Running the open models on LUMI

`lumi.sh` drives everything over SSH from your own machine: it copies files to LUMI's
scratch disk, builds the environments, submits Slurm jobs and copies results back.
Nothing runs on your machine and nothing needs to stay connected while a job runs. You need a
CSC user with an SSH key registered for LUMI and a LUMI project with GPU hours (see LUMI's
[first steps](https://docs.lumi-supercomputer.eu/firststeps/)). Everything lives in
`/scratch/<project>/$USER/fiasr`, weights in `/scratch/<project>/$USER/hf-cache` (about 30 GB added).

## The steps

```bash
export LUMI_USER=<your CSC user name>
export LUMI_PROJECT=project_<number>      # your LUMI project
L=lumi/lumi.sh                            # from the repo root
uv run --extra clip python scripts/get_clip.py --prepare-audio   # the clip at 16 kHz (push does it if missing)
bash $L push                 # scripts, job files and the 16 kHz clip -> scratch
bash $L setup                # three Python environments, about 15 min, once
bash $L download clip        # model weights, about 30 GB, once
bash $L download bench       # FLEURS Finnish test set and two more Whisper models
bash $L run clip hf          # also: qwen, nemo. One GPU job each, a few minutes
bash $L run bench hf         # FLEURS, 918 recordings; resumes if it hits the 3 h limit
bash $L status               # this repository's jobs and the last lines of their logs
bash $L fetch                # results/ back into the repo
bash $L download skn && bash $L prepare skn          # SKN dialect corpus, 24 GB, CPU job
bash $L fetch-labels         # its segment lists -> results/labels/ (for prepare_targets.py, prepare_timestamps.py)
bash $L push-labels          # SKN punctuated targets, sentence times and long-form windows -> $FIASR_DATA
bash $L download train       # FLEURS and VoxPopuli training data
bash $L run train hf --name <name> --data skn+open --skn-text punct --epochs 3 --lora-r 64
bash $L fetch-model <name>   # its train_log.json (add --adapter for the LoRA weights, gitignored)
bash $L run timestamps hf --arm hf_whisper-large-v3 --arm hub_whisper-large-v3-fi-dialects
bash $L prepare lamurre                              # LA-murre dialect test, 10 WAVs by HTTP range
bash $L download lamurre && bash $L prepare lamurre-train   # all LA-murre audio, 69 GB, CPU job
bash $L clean [--weights]    # remove the project from scratch when done
```

Arguments after the environment pass through to the runner:
`bash $L run clip nemo --arm nemo_canary-1b-v2 --force`, `bash $L run bench hf --limit 50`.

Three environments, because the libraries pin different `transformers` versions: `hf`
(transformers 5.19: Whisper, Parakeet, Orukeet, wav2vec2; peft for training), `qwen` (`qwen-asr`),
`nemo` (NeMo 3.0: Canary, RASMUS). Each venv sees the container's ROCm build of torch and pins it.
[`setup.sh`](setup.sh) is the only place these versions are pinned. Compute nodes have no internet,
which is why weights and data are fetched on the login node.

LA-murre (CC BY-ND 4.0) is downloaded from Kielipankki and processed on LUMI and locally only. Its
per-segment outputs (`results/benchmarks/lamurre_fi/*.jsonl`, the judge caches, `results/labels/lamurre_*`)
come back with `fetch` and are gitignored; do not commit them.

Cost: one GPU on `small-g` bills 0.5 GPU-hours per hour. The whole clip run is a few
GPU-minutes; the FLEURS run about half a GPU-hour; a three-epoch fine-tune about 3 hours.

## What went wrong on the way

- `/scratch` is a symlink: bind `/scratch/<project>` and `cd` inside the container.
- Package specs reach the container unquoted, so `>=` becomes a file redirect. Pin with `==`.
- `kenlm` does not build in the container (no C++ compiler), so wav2vec2 runs without its language model.
- NeMo 3.0 fails on three libraries the container ships: `aiter` (wants to compile at
  run time), `megatron` and `tensorstore` (built against NumPy 1). `lumi/stubs/` shadows
  them with modules that raise `ImportError`; the job puts it on `PYTHONPATH` for `nemo` only.
- `snapshot_download(allow_patterns=...)` fetched nothing for the RASMUS repo; its two files are fetched by name.
- A fresh `fiasr/` needs `download train` again before a fine-tune, or the job stops in two
  minutes on a missing `fleurs_fi/train.tsv` and its `afterok` jobs wait for ever.
- Every model directory in `fiasr/models/` is an arm `ft_<name>` (and `ft_<name>-chunks` for the
  clip). Whisper's long-form loop can make a fine-tune repeat a phrase: use the `-chunks` arm.
- A background download over ssh needs `ssh -f` and `< /dev/null`, or the session waits for it.
