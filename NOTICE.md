# Notices: data, assets and third-party material

The code in this repository is MIT-licensed ([`LICENSE`](LICENSE)). The data and assets below keep
the licences of their sources; when you reuse them, credit the source as given here.

## Speech corpora

**Suomen kielen näytteitä (SKN)**, Kotimaisten kielten keskus (Kotus), via Kielipankki – The Language
Bank of Finland, [urn:nbn:fi:lb-2020112937](http://urn.fi/urn:nbn:fi:lb-2020112937), CC BY 4.0.
In this repository: the SKN test references and model transcripts in `results/benchmarks/skn_fi/`,
the training and validation targets and sentence times in `results/labels/skn_*`, the long-form
windows and decodes in `results/labels/skn_longform.jsonl` and `results/timestamps/`, the examples in
`docs/DIALECT-EXAMPLES.md`, and two audio clips (`docs/story/src/assets/heinola_0097.mp3`,
`palkane_0021.mp3`, re-encoded from the published dataset
[GAIK-project/skn-finnish-dialects-asr](https://huggingface.co/datasets/GAIK-project/skn-finnish-dialects-asr)).
Changes: the interviews were cut into 2–25 s segments; in `results/labels/skn_{train,validation}.targets.jsonl`
the `target` field adds punctuation and capitals written by an LLM (Claude Opus 5.5) to SKN's
normalised transcript, with the words unchanged (`target_by` says which rows).

**FLEURS** (Google), `google/fleurs`, CC BY 4.0: references and model transcripts in
`results/benchmarks/fleurs_fi/`.

**VoxPopuli** (Meta), `facebook/voxpopuli`, CC0: references and model transcripts in
`results/benchmarks/voxpopuli_fi/`.

**Lauseopin arkiston murrekorpus (LA-murre)**, University of Turku and Kotus, via Kielipankki,
[urn:nbn:fi:lb-2020112935](http://urn.fi/urn:nbn:fi:lb-2020112935), CC BY-ND 4.0. Used as a test set and
to train an internal model. **No LA-murre text is in this repository**, because the references used
were modified (two annotation marks removed) and the licence does not allow sharing modified copies;
only numbers made from them are (`results/benchmarks/lamurre_fi/scores.json`, `*.meta.json`,
`results/judgements_dialect/lamurre_fi/*.json`, `results/labels/lamurre_*.stats.json`). The scripts
that process LA-murre write their outputs to gitignored paths.

## The Ajokortti webinar clip

The clip (`Ajokortti.mp3`) and its reference transcripts (`reference.txt`, by the University of
Helsinki team, and `reference.reviewed.txt`) are published in
[GAIK-project/gaik-toolkit](https://github.com/GAIK-project/gaik-toolkit/tree/main/evaluation_layer/eval_methods/transcription_eval/data)
and are not copied here; `scripts/get_clip.py` downloads them. This repository holds model transcripts
of the clip (`results/transcriptions/`), judge records that quote short spans of the reference and the
transcripts (`results/judgements/`), and the reference's tokens and sentences with word times
(`results/clip/gt_alignment.json`), for evaluation. Use of the clip itself follows the terms stated in
gaik-toolkit.

## Assets of the story pages

- `docs/story/src/assets/hero.webp`: an AI-generated illustration (GPT Image 2.5), no real place or
  person.
- `docs/story/src/assets/icons/*.svg`: [Lucide](https://lucide.dev/) icons (audio-lines, chart-column,
  cpu, database, file-text, graduation-cap, link, map-pin, pencil-line, scale, shield-check,
  sparkles), ISC licence, Copyright (c) 2026 Lucide Icons and Contributors; `database` and `link` are
  derived from Feather, MIT licence, Copyright (c) 2013-present Cole Bemis. The full licence text is in
  [`docs/story/src/assets/icons/LICENSE.lucide.txt`](docs/story/src/assets/icons/LICENSE.lucide.txt).

## Vendored code

`scripts/asr_scoring.py` is the ASR scorer of QAD-1 in GAIK's internal evaluation repository, which
implements the normalisation of the University of Helsinki team's `error_analysis.py`; it uses
[jiwer](https://github.com/jitsi/jiwer) (Apache 2.0) and [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz)
(MIT). `scripts/provenance.py` is adapted from the same repository's provenance helper.

## Models

The published model [GAIK-project/whisper-large-v3-fi-dialects](https://huggingface.co/GAIK-project/whisper-large-v3-fi-dialects)
is Apache 2.0, as its base model openai/whisper-large-v3. The other models evaluated here keep their
own licences; this repository holds only their transcripts.
