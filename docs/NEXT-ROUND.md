# Round 5: more dialect speech, and VoxPopuli back

Branch `dev`. This file is the plan and the log of the round; the **Status** section at the end says
where the work stands and how to pick it up on another machine.

## Goal and the rule, fixed before any run

A model that could replace `main` (tag `v1.0`) must meet all four, each against the numbers already
published:

| Test | Must be | Reference |
|---|---|---|
| SKN dialect test WER | ≤ 19.7 % | the `la-murre` revision |
| LA-murre blind judge (Claude Opus 5.5, prompt `dialect-judge-v2`, all 583 segments) | fewer points than the `la-murre` revision **in the same round** | 11.6 in round 4 |
| FLEURS WER | ≤ 8.1 % | main 7.6 % + 0.5 |
| VoxPopuli WER | ≤ 11.3 % | main 10.8 % + 0.5 |

If more than one run meets the rule, the candidate is the one with the lower validation loss on the
shared validation set (every run below uses the same validation segments), not the one with the best
test numbers. Publishing anything to Hugging Face needs the maintainer's go-ahead.

## Data added in this round

### Aalto Finnish Parliament ASR Corpus (for VoxPopuli)

- Kielipankki, http://urn.fi/urn:nbn:fi:lb-2022052002 (group http://urn.fi/urn:nbn:fi:lb-2021081105).
  Copyright holder: the Parliament of Finland.
- Licence (the corpus's `LICENSE.txt`, read 20 Oct 2026): audio **CLARIN PUB +BY +PRIV +ND +OTHER**,
  transcripts **CLARIN PUB +BY +PRIV**. Use, copying and training are allowed. +ND: "The user is not
  permitted to distribute derivative works, i.e. works containing copyrighted parts of the original."
  This is the condition Kielipankki assessed for LA-murre (CC BY-ND), where it found that the model's
  weights most likely contain no protected part of the corpus. Precedent: Finnish-NLP's
  `wav2vec2-xlsr-1b-finnish-lm-v2` is published under Apache 2.0 and trained mostly on this corpus.
  **Publishing a model trained on it is the maintainer's decision** (or a one-line question to
  Kielipankki, like the LA-murre one). Nothing of the audio or the per-segment texts is committed.
- What is used: the 2015–2020 set, package `fi-parliament-asr-2015-2020-part1.zip` (55 GB, 594 h, 299
  speakers, sessions from every year 2015–2020), and the hand-corrected 2016 dev set for validation.
  Transcripts are lower case, without punctuation, numbers as words, `<UNK>` for unaligned words.
- [`prepare_parliament.py`](../scripts/prepare_parliament.py): utterances of 2–25 s, ≥ 3 words, no
  `<UNK>`; a seeded sample of **60 h**, at most 300 utterances per speaker; 200 validation segments.
  Speakers who could be in VoxPopuli's test (Finnish MEPs and commissioners 2009–2020) are left out,
  so VoxPopuli does not score familiar voices.
- Targets: punctuation and capitals by an LLM (`prepare_targets.py punct`, the SKN prompt; words must
  stay identical), as for SKN.

### Border Karelia corpus (Raja-Karjalan korpus, `finka`): more dialect speech

- Kielipankki / Kotus, http://urn.fi/urn:nbn:fi:lb-2017121322. Licence **CC BY 4.0** (the corpus
  `readme.txt`: "Aineiston lisenssi on CC BY 4.0 (1.1.2018 alkaen)"); the readme also says the
  resource contains personal data. A model trained on it can be published under Apache 2.0 with
  attribution, as with SKN.
- 171 interviews, **118 h** of audio, every one with an utterance-level Praat TextGrid (one tier per
  speaker), from six parishes: Ilomantsi, Korpiselkä, Impilahti, Suistamo, Salmi, Suojärvi; recorded
  for Kotus's tape archive (SKNA) in the 1960s–70s. None of the parishes is an SKN or LA-murre test
  locality. Transcript: coarse dialect spelling with marks (´ palatalisation, ŋ, `*`, parentheses).
  Download: one zip of 25.4 GB.
- [`prepare_finka.py`](../scripts/prepare_finka.py), after `prepare_lamurre.py`: utterances joined while the gap is
  under 1.5 s and the segment stays ≤ 25 s, dropped when another speaker overlaps; marks cleaned;
  44.1 → 16 kHz. Validation: every recording of two informants (Ilomantsi 13, Suistamo 06), never
  trained on. Targets: dialect rewritten in standard Finnish by an LLM (`prepare_targets.py
  normalise`), as for LA-murre. The marks follow the corpus's `Litterointimerkit.pdf`.

### Other corpora checked (20 Oct 2026)

| Corpus | Speech | Transcripts | Licence | Model publishable under Apache 2.0? |
|---|---|---|---|---|
| Border Karelia (`finka`) | 118 h dialect interviews, 6 parishes | utterance TextGrids, dialect spelling | CC BY 4.0 (+ personal data) | yes, with attribution: **used** |
| Aalto Finnish Parliament ASR | 3 130 h plenary speech | machine-aligned, normalised | PUB +BY +PRIV +ND +OTHER (audio) | likely (ND as LA-murre): **used, publication to decide** |
| Follow-up Study of Dialects (`seuruu`) | 200 h, 10 towns, 1989–2007 | about half an hour per recording, TextGrid/EAF | RES +ID +PLAN +BY +PRIV +NORED +OTHER | no without the rights holder (Kotus); three towns are SKN test municipalities |
| Helsinki speech (`helpuhe`) | Helsinki interviews 1970s–2010s | partly aligned | RES +NC +PRIV +NORED +DEP | no (NC) |
| Finnish Broadcast Corpus (`FBC`) | 17 h Yle 2003, little transcribed | orthographic | RES +NC +ND +NORED | no |
| Finnish-American immigrants (`amersuom`) | narration tasks | none | RES +PRIV +NORED +DEP | no; no transcripts |
| Lahjoita puhetta | about 3 600 h colloquial | partly | RES, personal data | out of scope until Kielipankki answers (asked 20 Oct 2026) |
| Satakunta speech (`sapu`) | 37 h interviews 2007–2013, Rauma and Pori regions | semi-coarse dialect transcription | RES +NC +PRIV +NORED +DEP +OTHER | no (NC) |
| Aalto DSP conversations (`DSPCON`) | 7.4 h student conversations | verbatim, word-aligned | ACA +NC +PRIV +NORED +DEP +OTHER | no (NC) |
| Kotus's own dialect samples (murrekirja, Tävvel kielel, ...) | short MP3 excerpts | in books, not aligned | none stated (Tävvel kielel = the Border Karelia corpus, CC BY) | no licence on the MP3 pages |
| Outside Kielipankki (Hugging Face, OpenSLR, Zenodo, Common Voice) | – | – | – | nothing open with dialect transcripts; YODAS/Granary Finnish has only Whisper pseudo-labels |

Kielipankki's own terms say nothing about trained models; for ND, ACA and RES material only the rights
holder or Kielipankki can settle publication. Still open: Kielipankki resources available only through
LAT. Candidates to ask about for a later round: *Itäsuomalaisten kielellinen variaatio* (`kiveli-src`,
114 h of North Savo and North Karelia interviews, 76 of 89 transcribed, licence not stated), the
Ingrian dialect corpus (`ingrian-dialects`, about 100 h with ELAN transcripts, licence not set), the
Oulu tape archive (`ona`, over 7 000 h, RES, not yet released) and Tver Karelian 1957–71 (`tver-1980`,
28 h, CC BY, but Karelian rather than Finnish).

## LLM work and its budget (at most 60 USD in this round)

| Step | Size | Estimate |
|---|---|---|
| Model choice for targets: GPT-6.1-sol (Azure, about 2 / 10 USD per M tokens) against Claude Opus 5.5 (4 / 20), normalise prompt on 200 SKN test segments scored against SKN's human standard-Finnish text | 2 × 200 segments | **done: 0.84 USD** |
| Parliament punctuation | 60 h, about 370 000 words | 17 USD with GPT-6.1-sol (34 with Opus) |
| Border Karelia normalisation | about 60 h after filtering | 20–25 USD with GPT-6.1-sol |
| Blind LA-murre judge, round 5 (Claude Opus 5.5, as round 4) | 583 segments × 4–5 models | 8–10 USD |

**Model choice (decided 20 Oct 2026): GPT-6.1-sol.** The normalise prompt on 200 SKN test segments (20 per
test municipality, seed 0, the prompt's own example segment left out), dialect text in, scored with the
benchmark normalisation against SKN's human standard-Finnish text: GPT-6.1-sol **12.1 % WER** (0.19 USD),
Claude Opus 5.5 14.2 % (0.63 USD; the same figure as the earlier check on all 600). The dialect text as
it is: 68.3 %. One segment of 200 failed the word-count check for each model. `prepare_targets.py
--provider azure` writes the parliament and Border Karelia targets; `--max-usd` caps each run.

Spent so far: 0.86 USD (the comparison and a 20-segment trial).

## Runs (LUMI, one MI250X GCD each, LoRA r64, 3 epochs, lr 1e-4, batch 8, best epoch by validation loss)

| Run | `--data` | Purpose |
|---|---|---|
| `whisper-large-v3-fi-r5-parl` | `skn+open+lamurre+parl` | the `la-murre` recipe plus 60 h of parliament: does VoxPopuli come back without losing dialects? |
| `whisper-large-v3-fi-r5-finka` | `skn+open+lamurre+parl+finka` | plus Border Karelia: more dialect speech |

Tests for each: SKN, FLEURS, VoxPopuli, LA-murre (`run bench`), the clip chunked, and judge round
`r5` with base Whisper, main, `la-murre` and the new models.

## Status / how to resume

Last updated 20 Oct 2026.

- [x] Licences read; corpus survey (table above).
- [x] `prepare_parliament.py`, `finetune_whisper.py --data ...+parl`, `lumi.sh download parl` /
      `prepare parl`.
- [x] VoxPopuli speaker list in `prepare_parliament.py` (48 names, 38 of them speakers in the corpus).
- [x] `prepare_finka.py`, `finetune_whisper.py --data ...+finka`, `lumi.sh download finka` / `prepare finka`.
- [x] Model choice for the targets: GPT-6.1-sol; `prepare_targets.py --provider azure --max-usd`.
- [ ] LUMI login for this round (LUMI_USER / LUMI_PROJECT) — waiting for the maintainer.
- [ ] Downloads, cutting, targets, training, tests, judge, report.

Resume on another machine: `git switch dev`, read this file, then the commands in order:

```bash
export LUMI_USER=<CSC user name> LUMI_PROJECT=project_<number>
L=lumi/lumi.sh
bash $L push                      # scripts and this round's changes to scratch
bash $L download parl             # background; $R/data/parl_raw/DONE when ready
bash $L prepare parl              # CPU job: 60 h sample -> $FIASR_DATA/parl_fi
bash $L download finka            # background; $R/data/finka_raw/DONE when ready
bash $L prepare finka             # CPU job -> $FIASR_DATA/finka_fi
bash $L fetch-labels              # -> results/labels/{parl,finka}_{train,validation}.jsonl (gitignored)
T="uv run --extra judges python scripts/prepare_targets.py"
$T punct results/labels/parl_train.jsonl --provider azure --max-usd 22
$T punct results/labels/parl_validation.jsonl --provider azure --max-usd 1
$T normalise results/labels/finka_train.jsonl --provider azure --max-usd 25
$T normalise results/labels/finka_validation.jsonl --provider azure --max-usd 1
bash $L push-labels
bash $L run train hf --name whisper-large-v3-fi-r5-parl --data skn+open+lamurre+parl --skn-text punct --epochs 3 --lora-r 64
bash $L run train hf --name whisper-large-v3-fi-r5-finka --data skn+open+lamurre+parl+finka --skn-text punct --epochs 3 --lora-r 64
```

Jobs keep running on LUMI when the local machine disconnects; `bash $L status` shows them.
The per-segment texts of the new corpora are never committed (`results/labels/parl_*.jsonl` and
`finka_*.jsonl` are gitignored); after a machine change they come back with `fetch-labels`.
