#!/bin/bash
# One entry point for the LUMI side of this repository. Run from the repo root, in Git Bash or any shell
# with ssh and tar. First: export LUMI_USER=<CSC user name> LUMI_PROJECT=project_<number>
#
#   lumi/lumi.sh push                  copy scripts/, lumi/ and the 16 kHz clip to scratch
#   lumi/lumi.sh push-labels           copy the SKN training targets, sentence times and long-form windows, and
#                                      the parliament and Border Karelia targets (results/labels/{parl,finka}_*.targets.jsonl,
#                                      gitignored)
#   lumi/lumi.sh fetch-labels          copy the segment lists of prepare_skn.py, prepare_parliament.py and
#                                      prepare_finka.py back into results/labels/ (gitignored)
#   lumi/lumi.sh push-dental           copy the ten dental-webinar clips (data/dental10/*.16k.npy + .txt, gitignored)
#   lumi/lumi.sh fetch-audio <bench>   copy a test set's WAVs (fleurs_fi, voxpopuli_fi, lamurre_fi) into data/<bench>/
#                                      (gitignored), for run_api_benchmark.py
#   lumi/lumi.sh setup [hf|qwen|nemo]  build the Python environments (login node, ~15 min, once)
#   lumi/lumi.sh download [clip|bench] fetch weights (and for bench the FLEURS test set) on the login node
#   lumi/lumi.sh download models [name...]  this project's published fine-tunes from Hugging Face into models/;
#                                      HF_TOKEN from the local environment, if set, is passed on
#                                      (sent over ssh's stdin, never written to disk)
#   lumi/lumi.sh run clip  <env> [args]   the 180 s Ajokortti clip        (run_open_models.py, 45 min limit);
#                                         add --bench dental10_fi for the ten dental clips (3 h limit)
#   lumi/lumi.sh run bench <env> [args]   FLEURS Finnish test, 918 clips  (run_benchmark.py, 3 h limit);
#                                         add --bench voxpopuli_fi, skn_fi or lamurre_fi for the others
#   lumi/lumi.sh download train           the open training data (FLEURS and VoxPopuli Finnish, train)
#   lumi/lumi.sh download skn             Suomen kielen näytteitä (SKN) dialect corpus, 24 GB, CC BY 4.0
#   lumi/lumi.sh prepare skn              cut SKN into segments and splits (CPU job, prepare_skn.py)
#   lumi/lumi.sh prepare skn-heldout      every segment of the 12 test and validation municipalities (CPU job,
#                                         prepare_skn.py --heldout), for training on all of SKN
#   lumi/lumi.sh prepare lamurre          the LA-murre dialect test, 10 WAVs read by HTTP range (login node)
#   lumi/lumi.sh download lamurre         all LA-murre audio, 69 GB, CC BY-ND (background; a DONE file when ready)
#   lumi/lumi.sh prepare lamurre-train    cut LA-murre training segments (CPU job, prepare_lamurre.py --train)
#   lumi/lumi.sh download parl            Aalto Finnish Parliament ASR corpus, 2015-2020 part 1 and the dev/test
#                                         package, 57 GB, CLARIN PUB +ND (background; a DONE file when ready)
#   lumi/lumi.sh prepare parl [args]      cut its seeded training subsample (CPU job, prepare_parliament.py [--hours N])
#   lumi/lumi.sh download finka           Border Karelia dialect corpus, 25 GB, CC BY 4.0 (background; DONE file)
#   lumi/lumi.sh prepare finka            cut it into training segments (CPU job, prepare_finka.py)
#   lumi/lumi.sh run train hf [args]      LoRA fine-tune of Whisper (finetune_whisper.py, 16 h limit)
#   lumi/lumi.sh run timestamps hf --arm A  timestamps and long-form decoding on SKN (eval_timestamps.py, 3 h)
#                                      (--fallback: add the temperature-fallback long-form decode)
#   lumi/lumi.sh fetch-model <name> [--adapter]  copy a fine-tune's train_log.json (and with --adapter its
#                                      LoRA adapter, gitignored) into results/finetune/<name>/
#   lumi/lumi.sh status                this repository's jobs and the last lines of their logs
#   lumi/lumi.sh fetch                 copy results/ back into this repository
#   lumi/lumi.sh clean [--weights]     delete the fiasr directory on scratch; --weights also deletes the
#                                      model weights this repository downloaded from the shared hf-cache
# Every job is named fiasr; nothing here touches jobs with another name.
set -euo pipefail
U=${LUMI_USER:?set LUMI_USER to your CSC user name}
LP=${LUMI_PROJECT:?set LUMI_PROJECT to your LUMI project, e.g. project_<number>}
H=$U@lumi.csc.fi
S=/scratch/$LP/$U
R=$S/fiasr
SIF=/appl/local/containers/sif-images/lumi-pytorchv0-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif
MODELS="openai--whisper-large-v3 openai--whisper-large-v3-turbo Finnish-NLP--whisper-large-finnish-v3
  Finnish-NLP--wav2vec2-xlsr-1b-finnish-lm-v2 nvidia--parakeet-tdt-0.6b-v3 oruk--orukeet Qwen--Qwen3-ASR-1.7B
  Qwen--Qwen3-ASR-0.6B nvidia--canary-1b-v2 RASMUS--Finnish-ASR-Canary-v2"

in_container() {  # runs a command in the container on the login node, from $R
  ssh "$H" "cd $R && singularity exec -B /scratch/$LP $SIF bash -c '\$WITH_CONDA; cd $R; source venv-hf/bin/activate; export HF_HOME=$S/hf-cache FIASR_DATA=$R/data FIASR_MODELS=$R/models; $1'"
}

copy_to() {  # copy_to <local file> <path under $R>
  ssh "$H" "mkdir -p \$(dirname $R/$2) && cat > $R/$2" < "$1"
  echo "  $1 -> $R/$2"
}

cmd=${1:-help}; shift || true
case $cmd in
  push)
    [ -f data/ajokortti/Ajokortti.16k.npy ] || uv run --extra clip python scripts/get_clip.py --prepare-audio
    tar -cf - --exclude=__pycache__ scripts lumi data/ajokortti/Ajokortti.16k.npy | ssh "$H" "mkdir -p $R && tar -xf - -C $R"
    ssh "$H" "cd $R && find lumi scripts -type f \( -name '*.sh' -o -name '*.sbatch' -o -name '*.py' \) -exec sed -i 's/\r\$//' {} +"
    echo "pushed to $H:$R" ;;
  push-labels)
    # what finetune_whisper.py (--skn-text punct, --timestamps) and eval_timestamps.py read from $FIASR_DATA
    for split in train validation heldout; do
      if [ -f results/labels/skn_$split.targets.jsonl ]; then copy_to results/labels/skn_$split.targets.jsonl data/skn_fi/$split.targets.jsonl; fi
    done
    for c in parl finka; do for split in train validation; do
      if [ -f results/labels/${c}_$split.targets.jsonl ]; then copy_to results/labels/${c}_$split.targets.jsonl data/${c}_fi/$split.targets.jsonl; fi
    done; done
    copy_to results/labels/skn_parts.jsonl data/skn_fi/parts.jsonl
    copy_to results/labels/skn_longform.jsonl data/skn_longform.jsonl ;;
  push-dental)
    ls data/dental10/*.16k.npy >/dev/null || { echo "data/dental10/*.16k.npy missing"; exit 1; }
    tar -cf - data/dental10/*.16k.npy data/dental10/*.txt | ssh "$H" "mkdir -p $R && tar -xf - -C $R"
    echo "pushed $(ls data/dental10/*.16k.npy | wc -l) clips to $R/data/dental10" ;;
  fetch-audio)
    b=${1:?fleurs_fi, voxpopuli_fi or lamurre_fi}
    mkdir -p data/$b
    ssh "$H" "cd $R/data/$b/test && tar -cf - ." | tar -xf - -C data/$b
    echo "fetched $(ls data/$b | wc -l) files into data/$b" ;;
  fetch-labels)
    # the input of prepare_targets.py punct and prepare_timestamps.py; regenerable, so gitignored
    for split in train validation heldout; do
      ssh "$H" "test -f $R/data/skn_fi/$split.jsonl" || continue
      ssh "$H" "cat $R/data/skn_fi/$split.jsonl" > results/labels/skn_$split.jsonl
      echo "  $R/data/skn_fi/$split.jsonl -> results/labels/skn_$split.jsonl"
    done
    for c in parl finka; do for split in train validation; do
      ssh "$H" "test -f $R/data/${c}_fi/$split.jsonl" || continue
      ssh "$H" "cat $R/data/${c}_fi/$split.jsonl" > results/labels/${c}_$split.jsonl
      echo "  $R/data/${c}_fi/$split.jsonl -> results/labels/${c}_$split.jsonl"
    done; done ;;
  setup)
    ssh "$H" "cd $R && LUMI_PROJECT=$LP bash lumi/setup.sh $* 2>&1 | grep -E ' ok |rror' || true" ;;
  download)
    case ${1:-clip} in
      clip)  in_container "python scripts/run_open_models.py --download" | grep -E '^(downloading|done)' ;;
      bench) in_container "python scripts/run_benchmark.py --download" | grep -E '^(downloading|done|fleurs|voxpopuli)' ;;
      train) in_container "python scripts/finetune_whisper.py --download" | grep -E '^done' ;;
      models)
        shift  # the token, when set, goes over ssh's stdin into the job's environment only
        printf '%s\n' "${HF_TOKEN:-}" | ssh "$H" "read -r T; cd $R && HF_TOKEN=\$T singularity exec -B /scratch/$LP $SIF \
          bash -c '\$WITH_CONDA; source venv-hf/bin/activate; export HF_HOME=$S/hf-cache FIASR_MODELS=$R/models; \
          python scripts/download_models.py $*'" ;;
      lamurre) ssh -f "$H" "mkdir -p $R/data/lamurre_raw && cd $R/data/lamurre_raw && nohup sh -c 'for p in hame-kai \
               kare-karp kess-pohe pohki-sate satl-som uusi-varp eaf; do f=la-murre_wav_\$p.zip; [ \$p = eaf ] && f=la-murre_eaf.zip; \
               wget -q -c https://www.kielipankki.fi/download/la-murre/audio/\$f; done; echo done > DONE' > download.log 2>&1 < /dev/null &" ;;
      parl)  ssh -f "$H" "mkdir -p $R/data/parl_raw && cd $R/data/parl_raw && nohup sh -c 'for f in devtest.zip \
               fi-parliament-asr-2015-2020-part1.zip; do \
               wget -q -c https://www.kielipankki.fi/download/fi-parliament-asr/fi-parliament-asr-v2/\$f; done; \
               echo done > DONE' > download.log 2>&1 < /dev/null &" ;;
      finka) ssh -f "$H" "mkdir -p $R/data/finka_raw && cd $R/data/finka_raw && nohup sh -c 'wget -q -c \
               https://www.kielipankki.fi/download/finka/.zip/Raja-Karjalan_korpus_dl.zip; echo done > DONE' \
               > download.log 2>&1 < /dev/null &" ;;
      skn)   ssh "$H" "mkdir -p $R/data/skn_raw && cd $R/data/skn_raw && for p in 01-10 11-20 21-30 31-40 41-50; do \
               wget -q -c https://www.kielipankki.fi/download/SKN/skn_\$p.zip; done; ls -la" ;;
    esac ;;
  run)
    what=${1:?clip, bench, train or timestamps}; env=${2:?hf, qwen or nemo}; shift 2
    case $what in
      clip)  script=run_open_models.py; limit=00:45:00 ;;
      bench) script=run_benchmark.py;   limit=03:00:00 ;;
      train) script=finetune_whisper.py; limit=16:00:00 ;;
      timestamps) script=eval_timestamps.py; limit=03:00:00 ;;
      *) echo "clip, bench, train or timestamps"; exit 2 ;;
    esac
    case " $* " in *" --bench "*) [ $what = clip ] && limit=03:00:00 ;; esac   # ten 180 s clips per arm
    ssh "$H" "cd $R && sbatch --account=$LP --time=$limit --export=ALL,FIASR_SCRIPT=$script,LUMI_PROJECT=$LP lumi/job.sbatch $env $*" ;;
  status)
    ssh "$H" "squeue --me --name=fiasr -o '%.10i %.9T %.10M %.10l'; cd $R && for f in \$(ls -t fiasr-*.out 2>/dev/null | head -3); do echo \"== \$f\"; grep -E '^\[(run|FAIL| ok |skip)' \$f | cut -c1-300 | tail -8; done" ;;
  fetch)
    ssh "$H" "cd $R && tar -cf - results" | tar -xf - -C .
    echo "fetched into results/" ;;
  prepare)
    case ${1:-skn} in
      skn)     ssh "$H" "cd $R && sbatch --account=$LP --export=ALL,LUMI_PROJECT=$LP lumi/prep.sbatch" ;;
      skn-heldout) ssh "$H" "cd $R && sbatch --account=$LP --export=ALL,LUMI_PROJECT=$LP,FIASR_PREP='prepare_skn.py --heldout' lumi/prep.sbatch" ;;
      lamurre) in_container "python scripts/prepare_lamurre.py" ;;
      lamurre-train) ssh "$H" "cd $R && sbatch --account=$LP --export=ALL,LUMI_PROJECT=$LP,FIASR_PREP='prepare_lamurre.py --train' lumi/prep.sbatch" ;;
      finka) ssh "$H" "cd $R && sbatch --account=$LP --export=ALL,LUMI_PROJECT=$LP,FIASR_PREP=prepare_finka.py lumi/prep.sbatch" ;;
      parl) shift; ssh "$H" "cd $R && sbatch --account=$LP --export=ALL,LUMI_PROJECT=$LP,FIASR_PREP='prepare_parliament.py $*' lumi/prep.sbatch" ;;
    esac ;;
  fetch-model)
    name=${1:?model name}
    files="train_log.json"
    ssh "$H" "test -f $R/models/$name/_run_metadata.json" && files="$files _run_metadata.json"
    [ "${2:-}" = --adapter ] && files="$files adapter"   # weights: gitignored, never committed
    mkdir -p results/finetune/$name
    ssh "$H" "cd $R/models/$name && tar -cf - $files" | tar -xf - -C results/finetune/$name
    echo "fetched $files into results/finetune/$name" ;;
  clean)
    ssh "$H" "squeue --me --name=fiasr -h | grep -q . && { echo 'fiasr jobs still running'; exit 1; }; true"
    if [ "${1:-}" = --weights ]; then
      # huggingface_hub keeps file contents in a shared, reference-counted hub/blobs store, so deleting a
      # models--* directory frees nothing. `hf cache rm` drops the references and the contents no other
      # repo uses; other users' files in the shared cache stay. Needs the hf venv (setup).
      repos=""
      for m in $MODELS; do repos="$repos model/${m/--//}"; done
      ssh "$H" "cd $R/.. && singularity exec -B /scratch/$LP $SIF bash -c '\$WITH_CONDA; \
        source $R/venv-hf/bin/activate 2>/dev/null || true; export HF_HOME=$S/hf-cache; \
        hf cache rm -y $repos dataset/google/fleurs dataset/facebook/voxpopuli; hf cache ls | tail -1'"
      echo "removed this repository's weights from $S/hf-cache"
    fi
    ssh "$H" "rm -rf $R; echo removed $R" ;;
  *) sed -n '2,36p' "$0" ;;
esac
