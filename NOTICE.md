# Notices: licences, data, assets and third-party material

| What | Licence |
|---|---|
| Code (`scripts/`, `tests/`, `lumi/`) | MIT, [`LICENSE`](LICENSE), copyright Haaga-Helia University of Applied Sciences (GAIK project) |
| Results, figures and documentation created by the GAIK project | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); credit "GAIK project, Haaga-Helia University of Applied Sciences" |
| Data from the corpora below, wherever it appears in this repository | the licence of its source, as listed below |
| Model weights of GAIK-project/whisper-large-v3-fi-dialects (main and the `timestamps`, `all-skn` and `la-murre` revisions) | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), as openai/whisper-large-v3 |
| The Ajokortti clip | not in this repository and not covered by its licences (see below) |

When you reuse data from this repository, credit the source as given here.

## Speech corpora

**Suomen kielen näytteitä (SKN)**, Kotimaisten kielten keskus (Kotus), via Kielipankki – The Language
Bank of Finland, [urn:nbn:fi:lb-2020112937](http://urn.fi/urn:nbn:fi:lb-2020112937),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
In this repository: the SKN test references and model transcripts in `results/benchmarks/skn_fi/`,
the training and validation targets and sentence times in `results/labels/skn_*`, the long-form
windows and decodes in `results/labels/skn_longform.jsonl` and `results/timestamps/`, the examples in
`docs/DIALECT-EXAMPLES.md`, and two audio clips (`docs/story/src/assets/heinola_0097.mp3`,
`palkane_0021.mp3`, re-encoded from the published dataset
[GAIK-project/skn-finnish-dialects-asr](https://huggingface.co/datasets/GAIK-project/skn-finnish-dialects-asr)).
Changes: the interviews were cut into 2–25 s segments; in `results/labels/skn_{train,validation}.targets.jsonl`
the `target` field adds punctuation and capitals written by an LLM (Claude Opus 5.5) to SKN's
normalised transcript, with the words unchanged (`target_by` says which rows).

**FLEURS** (Google), [`google/fleurs`](https://huggingface.co/datasets/google/fleurs),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): references and model transcripts in
`results/benchmarks/fleurs_fi/`. Cite: Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod,
Siddharth Dalmia, Jason Riesa, Clara Rivera, Ankur Bapna. "FLEURS: Few-shot Learning Evaluation of
Universal Representations of Speech." IEEE Spoken Language Technology Workshop (SLT) 2022,
[arXiv:2205.12446](https://arxiv.org/abs/2205.12446).

**VoxPopuli** (Meta), [`facebook/voxpopuli`](https://huggingface.co/datasets/facebook/voxpopuli),
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/): references and model transcripts in
`results/benchmarks/voxpopuli_fi/`. Cite: Changhan Wang et al. "VoxPopuli: A Large-Scale Multilingual
Speech Corpus for Representation Learning, Semi-Supervised Learning and Interpretation." ACL 2021,
[arXiv:2101.00390](https://arxiv.org/abs/2101.00390).

**Lauseopin arkiston murrekorpus (LA-murre)**, University of Turku and Kotus, via Kielipankki,
[urn:nbn:fi:lb-2020112935](http://urn.fi/urn:nbn:fi:lb-2020112935),
[CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/). Used as the test set of every model
here, and to train one model on recordings from other localities than the test, published as the
`la-murre` revision of GAIK-project/whisper-large-v3-fi-dialects: in October 2026 Kielipankki
(FIN-CLARIN), which distributes LA-murre, assessed that the weights most likely contain no
copyright-protected part of the corpus, so publishing them needs no separate permission. **No
LA-murre text is in this repository**, because the references used were modified (two annotation
marks removed), the training targets are LLM rewrites in standard Finnish, and the licence does not
allow sharing modified copies; only numbers made from them are (`results/benchmarks/lamurre_fi/scores.json`,
`*.meta.json`, `results/judgements_dialect/lamurre_fi/*.json`, `results/labels/lamurre_*.stats.json`).
The scripts that process LA-murre write their outputs to gitignored paths.

## The Ajokortti webinar clip

The clip (`Ajokortti.mp3`) is an excerpt of a public QAdental webinar; QAdental (Suomen
Kotilääkäripalvelu Oy) is a GAIK project partner. The audio is not stored here and is not covered by
this repository's licences: `scripts/get_clip.py` downloads it, with its reference transcripts
(`reference.txt`, by the University of Helsinki team, and `reference.reviewed.txt`), from
[GAIK-project/gaik-toolkit](https://github.com/GAIK-project/gaik-toolkit/tree/main/evaluation_layer/eval_methods/transcription_eval/data),
where they are public. This repository holds model transcripts of the clip (`results/transcriptions/`,
`results/clip/decoding_check.json`), judge records that quote short spans of the reference and the
transcripts (`results/judgements/`), and the reference's tokens and sentences with word times
(`results/clip/gt_alignment.json`), for evaluation.

## Assets of the story pages and the README

- `docs/story/src/assets/hero.webp` and `docs/img/banner.webp`: AI-generated illustrations (GPT
  Image 2.5; the banner made with the hero as a style reference), no real place or person.
- `docs/story/src/assets/icons/*.svg`: [Lucide](https://lucide.dev/) icons (audio-lines, chart-column,
  cpu, database, file-text, graduation-cap, link, map-pin, package, pencil-line, scale, shield-check,
  sparkles), ISC licence, Copyright (c) 2026 Lucide Icons and Contributors; `database` and `link` are
  derived from [Feather](https://feathericons.com/), MIT licence, Copyright (c) 2013-present Cole
  Bemis. The full licence text is in
  [`docs/story/src/assets/icons/LICENSE.lucide.txt`](docs/story/src/assets/icons/LICENSE.lucide.txt).
  The built pages (`docs/story/story.*.html`) carry the same credit in their footer, and the README's
  figures in `docs/img/` (drawn by `scripts/readme_figures.py`) use the same icons.
- `docs/listen/*.mp3`: three SKN test segments (Suomen kielen näytteitä, Kotus, via Kielipankki, CC BY 4.0),
  cut from the test split of GAIK-project/skn-finnish-dialects-asr by `scripts/build_listen.py`. The listening
  page plays the Ajokortti clip from gaik-toolkit; it is not copied here.
- `docs/img/logos/*.svg`: the logos of Google, Gemini, Meta, OpenAI, Hugging Face and Claude
  (Anthropic) from [svgl](https://svgl.app/); ElevenLabs and Qwen from the Iconify `logos` set
  (gilbarbara/logos, CC0 1.0 for the files); NVIDIA from the Iconify `simple-icons` set, in NVIDIA's
  green. They are their owners' trademarks, used in the README's figures only to name who made each
  model, data set or tool; no endorsement is implied, and they are not covered by this repository's
  licences.

## Vendored code

`scripts/asr_scoring.py` is the ASR scorer of QAD-1 in GAIK's internal evaluation repository, which
implements the normalisation of the University of Helsinki team's `error_analysis.py`; it uses
[jiwer](https://github.com/jitsi/jiwer) (Apache 2.0) and [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz)
(MIT). `scripts/provenance.py` is adapted from the same repository's provenance helper.

## Models

The published model [GAIK-project/whisper-large-v3-fi-dialects](https://huggingface.co/GAIK-project/whisper-large-v3-fi-dialects)
(tag `v1.0`) and its `timestamps`, `all-skn` and `la-murre` revisions are Apache 2.0, as their base
model openai/whisper-large-v3.
The other models evaluated here keep their own licences; this repository holds only their transcripts.

## Computing

We acknowledge CSC – IT Center for Science, Finland for awarding this project access to the LUMI
supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CSC (Finland) and the LUMI
consortium through CSC – IT Center for Science, Finland.

(Wording: the template for awards granted by LUMI consortium countries on
[lumi-supercomputer.eu/acknowledgement](https://lumi-supercomputer.eu/acknowledgement/), with the
granting organisation filled in for access through CSC, Finland's share of LUMI.)
