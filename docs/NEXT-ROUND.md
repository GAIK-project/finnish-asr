# Round 6: Border Karelia dialects on top of the main model

Branch `round5-finka` (cut from `main` on 22 Oct 2026). Written before any model of this round was trained
or scored. The rule below decides; a number that comes out differently does not move it. The **Status**
section at the end says where the work stands and how to pick it up on another machine.

## Question

Does more dialect speech from a region the model has never heard (Border Karelia, Eastern Finnish dialects in
long contact with Karelian) improve dialect accuracy beyond the main model `parl`, without costing standard
Finnish?

## The rule, fixed before any run

The reference is the current main model `whisper-large-v3-fi-parl` (`main` since 21 Oct 2026), its figures
from `results/benchmarks/*/scores.json` (README pipeline decoding, greedy). A new model replaces it only if
it meets all four:

| Test | Must be | `parl`'s figure |
|---|---|---|
| SKN dialect test WER (600 segments) | ≤ 19.37 % | 19.37 % |
| LA-murre blind judge (Claude Opus 5.5, prompt `dialect-judge-v2`, all 583 segments) | fewer error points per 100 words than `parl` **in the same judge round** (`r6`, both arms judged together) | 11.73 in round 5 |
| FLEURS WER (918) | ≤ 8.12 % | 7.62 % + 0.5 |
| VoxPopuli WER (478) | ≤ 11.58 % | 11.08 % + 0.5 |

- Same decoding for both: `run_benchmark.py` through `lumi.sh run bench hf`, then `score_benchmark.py`.
- VoxPopuli moved 1.8 points between two seeds of the `la-murre` recipe (region-plan.md); a VoxPopuli
  failure within that noise is still a failure under this rule, and is reported as such.
- Parliament (`parl_fi`, `parl_terms_fi`) and LA-murre WER against the dialect text are reported but decide
  nothing.
- The judge round runs only for a model that passes the three WER conditions, with base Whisper large-v3 as
  an anchor arm; at most 10 USD.
- Publishing anything to Hugging Face needs the maintainer's go-ahead; this round does not publish.

## Data added

**Raja-Karjalan korpus (`finka`)**, Kotus via Kielipankki, [urn:nbn:fi:lb-2017121322](http://urn.fi/urn:nbn:fi:lb-2017121322),
**CC BY 4.0** (its readme: from 1 Jan 2018); the readme also says it contains personal data, so only
statistics of the per-segment texts are committed (`results/labels/finka_*.jsonl` is gitignored).

- 171 interviews, 118 h of audio (44.1 kHz), recorded in the 1960s–70s for Kotus's tape archive with evacuees
  from Ilomantsi, Korpiselkä, Impilahti, Suistamo, Salmi and Suojärvi. None is an SKN or LA-murre test
  locality. Download: one 25.4 GB zip (`lumi.sh download finka`, 4 min on LUMI).
- Every recording has a Praat TextGrid, one tier per speaker, in a coarse dialect spelling. 119 of the 171 are
  in Praat's short text format, 52 in the long one; [`prepare_finka.py`](../scripts/prepare_finka.py) reads
  both (the round-5 draft read only the long one and would have kept 21 h instead of 59 h).
- Segments: a speaker's utterances joined while the gap is under 1.5 s and the segment stays ≤ 25 s; dropped
  when another speaker overlaps, when shorter than 2 s or 3 words, or when it has an unclear-speech mark.
  80 264 utterances; kept 20 802 segments, of them **20 548 for training (58.1 h, 409 542 words;
  Korpiselkä 7 269, Impilahti 3 716, Ilomantsi 3 688, Suistamo 2 489, Suojärvi 1 713, Salmi 1 673)** and 200 for
  validation (0.63 h). Statistics: `results/labels/finka_segments.stats.json`.
- Validation: every recording of two informants (Ilomantsi 13, Suistamo 06), at most 100 segments each,
  never trained on.
- Targets: the dialect text rewritten in standard Finnish word for word by an LLM (`prepare_targets.py
  normalise`, the LA-murre prompt), GPT-6.1-sol in GAIK's Azure (`--provider azure`): on 200 SKN test
  segments it scored 12.1 % WER against SKN's human standard-Finnish text, Claude Opus 5.5 14.2 % (round 5
  check, 20 Oct 2026). LLM-written training targets, never test references.

## Run

| Name | `--data` | Other arguments |
|---|---|---|
| `whisper-large-v3-fi-r6-finka` | `skn+open+lamurre+parl+finka` | `--skn-text punct --epochs 2 --lora-r 64` (lr 1e-4, batch 8: the defaults) |

The same arguments as `parl` (its `_run_metadata.json` on LUMI) and the same parliament segments
(`FIASR_PARL_DIR=parl_train_recovery_20261020_2350`, the 37 740-segment set `parl` trained on). The validation
loss is not comparable with `parl`'s: the finka validation segments are added to it.

## Budget

| Item | Estimate | Spent |
|---|---|---|
| LLM, round total | at most 60 USD | 0.86 before this branch (round-5 model choice) |
| finka targets (train, `--max-usd 25`; validation, `--max-usd 1`) | about 12–16 USD with GPT-6.1-sol (0.14 USD for the 4 674 validation words) | **13.71**: train 13.57 (1 028 calls, 20 540 of 20 548 accepted, 0 failed), validation 0.14 (200 of 200) |
| Judge round `r6` (3 arms × 583 segments, Opus 5.5) | about 4–5 USD (round 5: 10.7 USD for 6 arms) | – |
| LUMI training, one GCD on `small-g` | `parl` took 6 h 36 min for 78 519 items; with about 20 000 more, 8–9 h: **4–4.5 GPU-hours** (0.5 per GCD-hour; the 16 h limit caps it at 8) | – |
| LUMI benchmarks, four sets | about 1 GPU-hour | – |

GAIK's LUMI allocation: 346 of 5 000 GPU-hours used on 22 Oct 2026.

## Status / how to resume

Last updated 22 Oct 2026, after the training job was submitted.

- [x] Branch from `main`; Border Karelia pieces ported from `dev` (`prepare_finka.py` with a reader for both Praat
      formats, `--data skn+open+lamurre+parl+finka`, `lumi.sh download|prepare finka`, finka labels in
      `fetch-labels` / `push-labels`, `prepare_targets.py --provider azure --max-usd`); `uv run pytest` passes.
- [x] `lumi.sh push`; `download finka` (25.4 GB); `prepare finka` (CPU job, 4 min); `fetch-labels`.
- [x] Targets with GPT-6.1-sol: train 20 540 of 20 548 accepted (8 dropped by the 30 % word-count rule), 13.57 USD;
      validation 200 of 200, 0.14 USD. `push-labels` done (`$FIASR_DATA/finka_fi/*.targets.jsonl` on LUMI).
- [x] **Training submitted** (job name `fiasr`, `small-g`, one GCD, 16 h limit): `whisper-large-v3-fi-r6-finka`,
      train 99 059 items (SKN 15 042, LA-murre 15 529, FLEURS 2 704, VoxPopuli 7 504, parliament 37 740,
      Border Karelia 20 540), validation 1 676. Expected about 8–9 h (`parl`: 6 h 36 min for 78 519), about
      4–4.5 GPU-hours.
- [x] **Benchmarks queued** behind it (`--dependency=afterok`, job name `fiasr`, 3 h limit each): SKN, FLEURS,
      VoxPopuli, LA-murre with `--arm ft_whisper-large-v3-fi-r6-finka`; about 1 GPU-hour.
- [ ] Fetch and score; apply the rule; judge round `r6` (only if the three WER conditions pass, at most 10 USD).

LLM spend this round: 0.86 (round 5 model choice) + 13.71 (finka targets) = **14.57 USD** of 60.

Resume (Git Bash, from the repo root, either machine):

```bash
git fetch && git switch round5-finka
set -a; . ~/.agents/env/haaga-helia/gaik/lumi.env; set +a     # LUMI_USER, LUMI_PROJECT
L=lumi/lumi.sh
bash $L status                       # the training job and the four queued benchmark jobs (all named fiasr)
# when the benchmarks have finished:
bash $L fetch                        # results/ back (benchmark outputs of LA-murre stay gitignored)
bash $L fetch-model whisper-large-v3-fi-r6-finka
uv run python scripts/score_benchmark.py
# if the training failed, resubmit it (labels are already on LUMI) and the benchmarks after it:
FIASR_PARL_DIR=parl_train_recovery_20261020_2350 bash $L run train hf --name whisper-large-v3-fi-r6-finka   --data skn+open+lamurre+parl+finka --skn-text punct --epochs 2 --lora-r 64
for b in skn_fi fleurs_fi voxpopuli_fi lamurre_fi; do bash $L run bench hf --bench $b --arm ft_whisper-large-v3-fi-r6-finka; done
```

The per-segment texts (`results/labels/finka_*.jsonl`) are gitignored; on another machine `fetch-labels` brings the
segment lists back, and the targets are on LUMI already (the local target cache stays on the machine that made it).
Every job of this repository is named `fiasr`; jobs with other names on the project are not ours.
