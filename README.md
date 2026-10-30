<p align="center">
  <img src="docs/img/banner.webp" alt="An old reel-to-reel tape recorder and a laptop showing a sound wave and a transcript, on a farmhouse kitchen table" width="100%">
</p>

# Finnish dialect speech-to-text

**We taught OpenAI's Whisper to understand Finnish dialects.** On dialect speech it gets about half as
many words wrong as before (40 % → 19 %), far fewer than any other model we tried, Gemini 3.5 Flash
included. Read and parliament speech stay about as good as before. The model and its SKN training data are free
to use. **New on 21 Oct 2026:** the main model is now trained on 72 h of Finnish parliament speech as well, which
lowered the dialect errors from 21.1 to 19.4 % and helps with names, acronyms and terms; the first model stays
available as `v1.0`.

🤗 [Model](https://huggingface.co/GAIK-project/whisper-large-v3-fi-dialects) ·
🗂️ [Training data](https://huggingface.co/datasets/GAIK-project/skn-finnish-dialects-asr) ·
🎬 [Video](https://gaik-project.github.io/finnish-asr/) ·
🔊 [Listen and compare](https://gaik-project.github.io/finnish-asr/listen.html) ·
📖 The story: [suomeksi](https://gaik-project.github.io/finnish-asr/story/story.fi.html) ·
[in English](https://gaik-project.github.io/finnish-asr/story/story.en.html) ·
🧰 [gaik-toolkit](https://github.com/GAIK-project/gaik-toolkit) ·
🌐 [gaik.ai](https://gaik.ai/)

Made by Akseli Leskinen ([laguagu](https://github.com/laguagu)), Haaga-Helia University of Applied
Sciences, in the [GAIK project](https://gaik.ai/), co-funded by the European Union (ERDF).

<img src="docs/img/leaderboard.svg" alt="Results at a glance: on dialect speech our model is first with 19.4 % of words wrong; on standard Finnish (FLEURS and VoxPopuli) ElevenLabs Scribe v2 is first with 5.4 % and our model is 8th with 8.8 %" width="100%">

- 🗣️ **Dialect speech: our model, by far.** 19.4 % of words wrong. The next best is Gemini 3.5 Flash,
  34.7 % when asked for standard Finnish (40.8 % when asked to write as spoken); base Whisper large-v3
  gets 40.4 % and Google Chirp 3 41.3 % (35.0 % without one segment where it repeated “se on” 1 262
  times).
- 🏢 **Standard Finnish: the commercial services.** On read and parliament speech (FLEURS and
  VoxPopuli together) ElevenLabs Scribe v2 gets 5.4 % of words wrong, gpt-4o-transcribe 6.6 % and our
  model 8.8 %, a little better than base Whisper.
- ⚖️ **Four AI judges agree.** Claude and three open models on CSC's Aitta service all find far fewer
  meaning errors in our models' dialect transcripts than in base Whisper's.
- 🔌 **The big API services do not fix dialects.** On the second dialect corpus a blind judge gives
  Gemini 3.5 Flash 27.2 and OpenAI's gpt-4o-transcribe 34.2 meaning-error points per 100 words, more
  than base Whisper (24.4); our model gets 11.7 (round 5 of the judge, below). On read and parliament speech the API services lead
  ([four kinds of speech](#6-how-it-compares)).

| Our model against the model it started from | Whisper large-v3 | Our model |
|---|--:|--:|
| 🗣️ Dialect speech: words wrong | 40.4 % | **19.4 %** |
| ⚖️ Another dialect corpus: meaning-error points per 100 words | 24.4 | **11.7** |
| 📖 Read Finnish (FLEURS): words wrong | 7.4 % | 7.6 % |
| 🏛️ Parliament speech (VoxPopuli): words wrong | 12.7 % | **11.1 %** |
| 🏷️ Parliament speech with names, acronyms and terms: words wrong | 14.5 % | **12.4 %** |

## 1. What it does

You give it Finnish speech, dialect or not. It writes standard Finnish, with punctuation.

<img src="docs/img/example.svg" alt="A Heinola test sentence: Whisper large-v3, Gemini 3.5 Flash and our model against a person's transcript" width="100%">

🔊 [Listen to this sentence and two more](https://gaik-project.github.io/finnish-asr/listen.html)

## 2. Where the data comes from

<img src="docs/img/data-sources.svg" alt="Whisper large-v3 by OpenAI as the starting model; SKN, FLEURS by Google and VoxPopuli by Meta for training and testing; LA-murre for testing and for the training of one variant" width="100%">

- **Whisper large-v3**, OpenAI: the model we started from. [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3), Apache 2.0.
- **SKN, Suomen kielen näytteitä**: dialect interviews from 50 municipalities, recorded in the
  1960s–70s, each sentence written down by a person. Kotus, via
  [Kielipankki](http://urn.fi/urn:nbn:fi:lb-2020112937), CC BY 4.0.
- **FLEURS**, Google: Wikipedia sentences read aloud. [google/fleurs](https://huggingface.co/datasets/google/fleurs), CC BY 4.0.
- **VoxPopuli**, Meta: European Parliament speeches. [facebook/voxpopuli](https://huggingface.co/datasets/facebook/voxpopuli), CC0.
- **LA-murre**: dialect interviews from 132 places. Its test set (10 places not in SKN) tests every
  model; only the `la-murre` version (below) also trained on the other places. University of Turku and
  Kotus, via [Kielipankki](http://urn.fi/urn:nbn:fi:lb-2020112935), CC BY-ND 4.0.

## 3. How we made it

<img src="docs/img/pipeline.svg" alt="Six steps: dialect speech, Opus adds punctuation, training on LUMI, word error test, blind LLM judge, published on Hugging Face" width="100%">

Training ran on [LUMI](https://www.lumi-supercomputer.eu/), the EuroHPC supercomputer hosted by CSC
in Kajaani. Every script is in [`scripts/`](scripts/); [Run it again](#run-it-again) lists the commands.

**Round by round:**

<img src="docs/img/journey.svg" alt="Training rounds: start 40.4 %, pilot on read speech, round 1 with dialect interviews 22.2 %, round 2 published 21.1 %, a version with LA-murre 19.7 % and a timestamps version 21.4 %" width="100%">

**Which version?**

| | Main model (since 21 Oct 2026) | `v1.0` (first model) | `la-murre` version |
|---|---|---|---|
| Training data | SKN + FLEURS + VoxPopuli + 56 h LA-murre + 72 h parliament | SKN + FLEURS + VoxPopuli | `v1.0`'s + 56 h LA-murre |
| Dialect test, words wrong | **19.4 %** | 21.1 % | 19.7 % |
| LA-murre, meaning-error points (judge round 5) | **11.7** | 14.0 | 12.1 |
| Read / parliament speech, words wrong | 7.6 % / 11.1 % | 7.6 % / **10.8 %** | 7.8 % / 11.4 % |
| Parliament with names and terms, words wrong ¹ | **12.4 %** | 13.7 % | 13.7 % |

Use the main model by default. For speech full of names and terms you know in advance, the
[`glossary` version](https://huggingface.co/GAIK-project/whisper-large-v3-fi-dialects/blob/glossary/README.md)
takes a word list as a prompt: on parliament speech it writes 93 % of the listed acronyms, English words and person
names right (76 % without the list) at 11.6 % WER (12.2 %). ¹ The main model was trained on other speakers and sessions of the same parliament
corpus. How it was made, the experiments around it (a region prompt, doubled standard Finnish, a glossary prompt)
and why: [`results/finetune/region-plan.md`](results/finetune/region-plan.md). The `la-murre` version was kept private
at first because of LA-murre's licence (CC BY-ND) and published on 20 Oct 2026 after Kielipankki's assessment
([below](#dialect-data-and-next-steps)).

## 4. What the language model did

The audio and the words are people's work. For the published model a language model, Claude Opus
5.5, helped in two places (for the `la-murre` version it also rewrote LA-murre's dialect spelling).
First, it added punctuation to the training text and nothing else:

<img src="docs/img/punctuation.svg" alt="Before: a transcript without punctuation. After: the same words with a question mark, a capital and a full stop" width="100%">

Second, it judged the test transcripts blind. A real example from the Ajokortti clip, three minutes of a
dental-education webinar ([what the clip is](#the-ajokortti-clip)):

<img src="docs/img/judge-example.svg" alt="A brand-name sentence from the webinar: Gemini 3.5 Flash gets all four brands right; our model writes Nobelbayer for Nobel Biocare, a major error" width="100%">

## 5. How we tested it

The test municipalities were kept out of training, so the test is speech the model had never heard:

<img src="docs/img/data-split.svg" alt="SKN's 50 municipalities: 38 for training, 2 for validation, 10 for the test; LA-murre as a second test" width="100%">

Two scores: word error rate counts every word that differs, the blind judge only errors that change the meaning:

<img src="docs/img/two-measures.svg" alt="Word error rate counts every differing word; the blind judge counts only meaning errors" width="100%">

## 6. How it compares

<img src="docs/img/dialect-ranking.svg" alt="Dialect test word error rate: our la-murre version 19.7 %, our published model 21.1 %; the best other model, Gemini 3.5 Flash asked for standard Finnish, 34.7 %; base Whisper 40.4 %" width="100%">

Across the open test sets, our model leads on dialect speech and parliament speech; on read
sentences gpt-4o-transcribe and Gemini are better:

<img src="docs/img/across-sets.svg" alt="Word error rate of Whisper large-v3, our model, gpt-4o-transcribe and Gemini 3.5 Flash on SKN dialects, FLEURS and VoxPopuli" width="100%">

Standard spoken Finnish with many names was measured on one three-minute webinar clip, **Ajokortti**,
the only non-dialect recording here whose reference a person has checked by ear. On it Gemini 3.5 Flash
makes the fewest meaning errors and plain Whisper large-v3 is the best open model; the best set-ups are
within about one point of WER of each other ([the clip, 26 set-ups](#the-ajokortti-clip)). Nine more webinar clips were
run too, but their references leave out fillers and were never checked, so read those numbers only as a
rough hint ([ten dental webinars](docs/REPORT.md#ten-dental-webinars-references-not-checked)).

So: use our model for dialect speech, and Gemini 3.5 Flash or plain Whisper large-v3 for standard
business Finnish with many names. 🔊 [Hear the differences](https://gaik-project.github.io/finnish-asr/listen.html)

## 7. Do other AI judges agree?

The fifth judging round (21 Oct 2026) put six transcripts of every segment of the second dialect corpus side by
side: base Whisper, two API services and our three published models. Claude Opus 5.5 and OpenAI's GPT-6.1-sol
(through GAIK's Azure) judged all 583 segments, gpt-oss-120b on CSC's Aitta a seeded sample of 200. Each judge is
stricter or milder, but all three rank the six the same way: our main model first, then the `la-murre` version and
the first model, and Gemini 3.5 Flash and gpt-4o-transcribe behind base Whisper:

<img src="docs/img/judges.svg" alt="Three blind judges on LA-murre: our main model first with every judge; gpt-4o-transcribe, Gemini 3.5 Flash and base Whisper far behind our fine-tuned models" width="100%">

| Meaning-error points per 100 words | Claude Opus 5.5 | GPT-6.1-sol | gpt-oss-120b (200) |
|---|--:|--:|--:|
| Ours, main model | **11.7** | **14.7** | **10.1** |
| Ours, `la-murre` | 12.1 | 14.9 | 10.5 |
| Ours, first model (`v1.0`) | 14.0 | 16.4 | 10.9 |
| Whisper large-v3 | 24.4 | 26.1 | 16.6 |
| Gemini 3.5 Flash (asked for standard Finnish) | 27.1 | 29.9 | 18.2 |
| OpenAI gpt-4o-transcribe | 34.0 | 36.2 | 23.1 |

GPT-6.1-sol prefers the same transcript as Claude in 93 % of pairs, gpt-oss-120b in 88 %
([`agreement-r5.json`](results/judgements_dialect/lamurre_fi/agreement-r5.json)). The main model's lead over the
`la-murre` version is small with every judge (0.2–0.4 points); read it as a tie on this corpus. The fourth round,
with the `all-skn` version, and the second round's open judges (Gemma 4 31B, Qwen3.6 27B, 92–93 % agreement) are
[in the report](docs/REPORT.md#second-opinions-open-model-judges-on-csc-aitta).

## 8. Why it got better

- 🎧 **It practised the exact task.** 118 hours of dialect interviews (SKN 62 h, LA-murre 56 h), each
  sentence paired with a standard-Finnish text, from places the test never uses.
- 🧭 **It learned dialects, not voices.** The 10 test municipalities were kept out, and it improved in
  all 10 of them.
- 🧱 **It kept what it knew.** Read and parliament speech stayed in the training mix, so they did not
  get worse (FLEURS 7.4 → 7.6 %, VoxPopuli 12.7 → 11.1 %).
- ✍️ **It writes sentences.** With punctuation added to the training text, its output reads like text.
- 🏛️ **Parliament speech taught it names and terms.** 72 h of plenary sessions lowered the dialect errors
  a little further (19.7 → 19.4 %) and the errors on names, acronyms and terms (13.7 → 12.4 %).
- 🏷️ **What is still weak: rare brand names.** The interviews have no brand names or dental terms, and the
  fine-tuned models spell them worse than base Whisper does; for those, use a general model or the
  `glossary` version with a word list.

## 9. Use it

<img src="docs/img/which-version.svg" alt="Use the main model by default; use the timestamps revision for sentence times or long audio" width="100%">

```python
import soundfile as sf
from transformers import pipeline  # pip install "transformers>=5" torch soundfile

asr = pipeline("automatic-speech-recognition", model="GAIK-project/whisper-large-v3-fi-dialects")
audio, sr = sf.read("clip.wav", dtype="float32")  # 16 kHz mono, up to 30 s
print(asr({"raw": audio, "sampling_rate": sr}, generate_kwargs={"language": "fi", "task": "transcribe"})["text"])
```

For longer audio, the [model card](https://huggingface.co/GAIK-project/whisper-large-v3-fi-dialects)
has code that cuts it into pieces at quiet moments. For audio that is mostly dialect, add
`revision="la-murre"` to the pipeline: 19.7 % of words wrong on the dialect test, a little weaker on
read and parliament speech ([its card](https://huggingface.co/GAIK-project/whisper-large-v3-fi-dialects/blob/la-murre/README.md)).

---

# Full details

Everything below is the full report in short: every model on every test set, how each was measured, and the
caveats. The longer write-up of each training round, judge round and decision is in
[`docs/REPORT.md`](docs/REPORT.md).

**Two kinds of test material.** Dialect speech is measured on held-out parts of two dialect corpora, SKN and
LA-murre, whose references are the corpora's human transcripts. Standard spoken Finnish with names is measured on
one webinar clip, Ajokortti, whose reference a person has checked at every disputed place; ten webinar clips
(Ajokortti and nine more) have references nobody has checked, so they are reported apart and only as a hint.
Parliament speech with names and terms is measured on held-out speakers of the Aalto parliament corpus.

## The models, round by round

| Version | Training data | SKN dialects WER % | LA-murre judge points | FLEURS WER % | VoxPopuli WER % | Status |
|---|---|--:|--:|--:|--:|---|
| Whisper large-v3 | (OpenAI's) | 40.4 | 24.4 | 7.4 | 12.7 | the starting point |
| Pilot | + FLEURS, VoxPopuli | – | – | 7.1 | 10.8 | not published |
| Round 1 | + SKN, 38 municipalities | 22.2 | 13.8 ² | 7.8 | 10.6 | not published |
| First model, `v1.0` | round 1 with punctuated text | 21.1 | 14.0 | 7.6 | 10.8 | tag `v1.0` (13 Oct 2026) |
| `timestamps` | `v1.0` + Whisper's timestamp tokens | 21.4 | 13.7 ³ | 7.4 | 11.2 | branch: sentence times, long audio |
| `all-skn` | `v1.0` on all 50 SKN municipalities | – ¹ | 13.1 ⁴ | 7.5 | 11.9 | branch: mostly-dialect audio |
| `la-murre` | `v1.0` + 56 h of LA-murre | 19.7 | 12.1 | 7.8 | 11.4 | branch (20 Oct 2026) |
| **Main model** (`parl`) | `la-murre` + 72 h of parliament speech | **19.4** | **11.7** | 7.6 | 11.1 | `main` since 21 Oct 2026 |
| `glossary` | the main model's recipe, trained to read a word list | 19.5 | – | 7.7 | 11.9 | branch (21 Oct 2026) |

LA-murre judge: meaning-error points per 100 words from Claude Opus 5.5, round 5 unless marked; the scale moves
between rounds, so compare within one round (² round 2, ³ round 3, ⁴ round 4). ¹ Trained on the SKN test
municipalities too. VoxPopuli moves by more than a point between two seeds of the same recipe.

**What we learned.**

- **Dialect data is what moves dialect accuracy.** Read and parliament speech alone (the pilot) improved only
  speech like it and got worse on the webinar clip; 62 h of SKN interviews nearly halved the errors, in all 10 test municipalities, and the gain carried
  over to LA-murre, a corpus the model had never heard.
- **Punctuated training text costs nothing.** Round 1 wrote almost no punctuation; with an LLM's punctuation in
  the training text (words unchanged) `v1.0` writes sentences at the same dialect accuracy.
- **More dialect speech and parliament speech help further.** 56 h of LA-murre took SKN to 19.7 %, 72 h of
  parliament speech to 19.4 % with better names, acronyms and terms; a dialect-region prompt and doubling the
  standard Finnish did not help ([`region-plan.md`](results/finetune/region-plan.md)).
- **A word list helps only a model trained to read one.** The `glossary` version writes 93 % of listed names and
  terms right (76 % without a list); a list given to an untrained model adds other errors.
- **Dialect training costs rare brand names.** On the webinar clip every fine-tune spells brand names and dental
  terms worse than base Whisper.
- **Decode in pieces.** Fed a whole 180 s clip, the fine-tunes loop; 15–25 s pieces cut at quiet points fix it,
  and only the `timestamps` version with its fallback decode handles long audio in one go.

## Public test sets: SKN and LA-murre dialects, FLEURS and VoxPopuli

Four open Finnish test sets:

- **SKN dialects**: 600 segments (2.5 h) of dialect interviews from 10 municipalities
  in five of SKN's six main dialect groups (no Ostrobothnian or Peräpohjola one), cut from Suomen
  kielen näytteitä (Kotus / Kielipankki, CC BY 4.0) by [`prepare_skn.py`](scripts/prepare_skn.py).
  The reference is SKN's normalised transcript: standard Finnish word forms, no punctuation. A model
  is right when it hears the dialect and writes standard Finnish. The test segments are the `test`
  split of the [published dataset](https://huggingface.co/datasets/GAIK-project/skn-finnish-dialects-asr).
- **LA-murre dialects**: 583 segments (2.3 h) from 10 localities that are not in SKN,
  cut from the Finnish Dialect Syntax Archive (University of Turku / Kotus, Kielipankki,
  CC BY-ND 4.0) by [`prepare_lamurre.py`](scripts/prepare_lamurre.py). Only the `la-murre`
  revision was trained on LA-murre, on recordings from other localities, never these 10. Its
  reference is a **dialect** transcript, so WER here charges a model for writing standard
  Finnish: every model lands at 52–73 %. Read this column with the
  [meaning judges](#7-do-other-ai-judges-agree), not on its own. Only the numbers are published
  ([`results/benchmarks/lamurre_fi/scores.json`](results/benchmarks/lamurre_fi/scores.json)): the
  licence does not allow sharing the modified references, the transcripts paired with them or the
  judge records, and `prepare_lamurre.py` rebuilds the test from Kielipankki.
- **FLEURS** (Google, CC BY 4.0): 918 Wikipedia sentences read aloud, about 10 s each.
- **VoxPopuli** (Meta, CC0): 478 European Parliament speech segments.

FLEURS and VoxPopuli are prepared or read speech, the sets model cards report; SKN is
spontaneous speech from mostly elderly speakers, recorded in the 1960s–70s.

<img src="docs/img/all-sets.svg" alt="Word error rate of every model on SKN dialects, FLEURS and VoxPopuli: ours best on dialects, ElevenLabs best on FLEURS, Finnish-NLP best on VoxPopuli" width="100%">

<!-- benchmark-table:start -->
| Model | SKN dialects WER % | LA-murre dialects WER % | FLEURS WER % | VoxPopuli WER % | Parliament WER % | FLEURS CER % | s / clip |
|---|--:|--:|--:|--:|--:|--:|--:|
| **Ours: pilot, FLEURS + VoxPopuli only** | – | – | 7.1 | 10.8 | – | 1.6 | 0.95 |
| **Ours: all-skn version** | – | 57.8 | 7.5 | 11.9 | 10.6 | 1.7 | 0.96 |
| **Ours: main model (parl)** | 19.4 | 57.4 | 7.6 | 11.1 | 8.5 | 1.8 | 0.98 |
| **Ours: la-murre version** | 19.7 | 57.4 | 7.8 | 11.4 | 9.2 | 1.8 | 0.96 |
| **Ours: first model (v1.0)** | 21.1 | 58.1 | 7.6 | 10.8 | 11.4 | 1.7 | 0.98 |
| **Ours: timestamps version** | 21.4 | 58.1 | 7.4 | 11.2 | 8.5 | 1.7 | 0.98 |
| **Ours: round 1** | 22.2 | 57.9 | 7.8 | 10.6 | – | 1.9 | 0.96 |
| Gemini 3.5 Flash (API, asked for standard Finnish) | 34.7 | 61.8 | – | – | – | – | – |
| Whisper large-v3 | 40.4 | 62.5 | 7.4 | 12.7 | 9.3 | 1.7 | 0.98 |
| Gemini 3.5 Flash (API, asked to write as spoken) | 40.8 | 60.2 | 5.1 | 11.5 | – | 1.9 | 2.24 |
| Google Chirp 3 (API) | 41.3 | – | 4.8 | 11.6 | – | 1.6 | 1.82 |
| Whisper large-v3 turbo | 42.5 | 62.7 | 8.4 | 13.5 | 9.7 | 2.2 | 0.19 |
| ElevenLabs Scribe v2 (API) | 44.9 | – | 3.3 | 9.1 | – | 1.7 | 3.09 |
| NVIDIA Canary 1B v2 | 45.6 | 65.6 | 8.7 | 11.4 | 10.2 | 2.1 | 0.30 |
| RASMUS Finnish-ASR-Canary-v2 + KenLM | 47.6 | 62.7 | 7.6 | 7.7 | 16.5 | 2.5 | 0.71 |
| NVIDIA Nemotron 3.5 Transcribe | 47.6 | 66.4 | 8.2 | 11.1 | 10.2 | 3.3 | 0.04 |
| Orukeet (Parakeet v3 fine-tune) | 48.5 | 64.0 | 11.6 | 15.9 | 12.7 | 2.8 | 0.04 |
| RASMUS Finnish-ASR-Canary-v2, greedy | 50.1 | 58.7 | 8.4 | 6.8 | 11.9 | 2.4 | 0.10 |
| NVIDIA Parakeet TDT 0.6B v3 | 52.5 | 66.7 | 13.0 | 17.5 | 14.6 | 3.1 | 0.18 |
| Finnish-NLP whisper-large-finnish-v3 | 56.6 | 51.6 | 7.1 | 6.6 | 9.2 | 1.7 | 0.89 |
| OpenAI gpt-4o-transcribe (API, via Azure) | 62.5 | 72.6 | 4.0 | 11.4 | – | 1.4 | 0.88 |
| Qwen3-ASR 1.7B | 66.7 | 72.9 | 25.1 | 27.5 | 25.5 | 5.8 | 0.34 |
| Finnish-NLP wav2vec2 XLS-R 1B (no LM) | 69.9 | 67.3 | 16.2 | 9.4 | 11.3 | 6.2 | 1.50 |
<!-- benchmark-table:end -->

A dash means no comparable measurement is retained here. `all-skn` has seen every SKN
municipality, so its SKN cell intentionally stays blank rather than being presented as
held-out accuracy. The timestamp variant's parliament result (962 segments, WER 8.5%)
was restored from an already completed and independently checked run; it was not rerun.

Orukeet and Parakeet were also evaluated on the complete SKN (600), LA-murre (583) and
parliament (962) test manifests. Coverage and references matched exactly, and a separate
WER/CER recount agreed for all six outputs. Empty transcripts remain errors in the scores.

Nine more open-model cells were measured on 25 Oct 2026 with the same runner
([`run_benchmark.py`](scripts/run_benchmark.py)), each model's decoding settings from its existing rows, and the same
scorer ([`score_benchmark.py`](scripts/score_benchmark.py)): Finnish-NLP wav2vec2 on SKN, LA-murre and parliament, and
Whisper large-v3 turbo, Finnish-NLP whisper-large-finnish-v3, Canary 1B v2, both RASMUS decodes and Qwen3-ASR 1.7B on
parliament. Coverage and references matched the test manifests exactly, and two separate WER/CER recounts agreed for
all nine outputs. **Finnish-NLP wav2vec2's parliament cell (11.3) may be optimistic:** its model card lists 228 h of
the Aalto Finnish Parliament ASR corpus in its fine-tuning data, and which parts of the corpus it held out was not
checked. RASMUS's training-data overlap with the parliament test was not checked either. On parliament RASMUS + KenLM
(16.5) is worse than greedy decoding (11.9); why was not investigated.

Nemotron 3.5 Transcribe (NVIDIA, released 28 Oct 2026, revision `c1766764`) was measured on 30 Oct 2026 on the same test
manifests with the same scorer, on one NVIDIA GH200 on CSC's Roihu, because the model runs on NVIDIA GPUs only. It used the
model card's vLLM 0.32 server and transcription settings and the card's own benchmark prompts, the verbatim one on four sets
and the FLEURS one on FLEURS; no test row was used to choose a prompt. Asked for standard Finnish with the prompt of the
Gemini row, it reaches 47.2 on SKN. Coverage and references matched exactly, no request failed, and two separate WER/CER
recounts agreed for all outputs. **Its s / clip (0.04) is not comparable with the other rows:** it is wall time per clip with
64 requests in flight on a GH200. **Its FLEURS and VoxPopuli cells may be optimistic:** the card lists both among its
training sources, and whether their test splits were excluded was not checked. The card makes no claim about dialects, and
its own Finnish FLEURS WER of 6.27 uses another normaliser. Settings are in the row's
[`meta.json`](results/benchmarks/skn_fi/vllm_nemotron-3.5-transcribe.meta.json).

Corpus WER after lower-casing and removing punctuation; seconds per clip are FLEURS GPU time on one AMD MI250X
(API services: request wall time). Bold rows are ours. Parliament: the Aalto corpus's held-out 2016 test speakers
(962 segments); our main model was trained on other speakers of the same corpus. Finnish-NLP's Whisper was trained
on FLEURS and RASMUS's Canary on FLEURS and VoxPopuli, so their numbers there are on familiar material. "As
spoken" and "standard Finnish" are two prompts; SKN's reference is in standard Finnish. Chirp 3 repeated "se on"
1 262 times in one SKN segment (35.0 % without it). The API services' training data is unknown, so for them the
test municipalities cannot be called held out. VoxPopuli keeps the parliament's digits and abbreviations.

## The Ajokortti clip

**Ajokortti** is a 180 s excerpt of a public dental-education webinar by QAdental (Suomen Kotilääkäripalvelu Oy, a
GAIK partner): two speakers, standard spoken Finnish with many names and brand names, **not dialect speech**. It
and the University of Helsinki (HY) team's reference transcript are published in gaik-toolkit's
[transcription_eval](https://github.com/GAIK-project/gaik-toolkit/tree/main/evaluation_layer/eval_methods/transcription_eval);
[`get_clip.py`](scripts/get_clip.py) downloads them. 26 set-ups were run on it, 12 of them open models, and scored
two ways: word error rate and two blind LLM judges that count only errors that change the meaning.

- 🏆 **Fewest meaning errors: Gemini 3.5 Flash**, first for both judges. Next come
  Google Chirp 3, ElevenLabs Scribe v2 and, as the best open models, plain **Whisper
  large-v3** and **large-v3 turbo**; the judges order those four differently.
- 📏 **WER ranks the models differently.** The Finnish-NLP Whisper fine-tune has the
  lowest WER (11.9 %) but more meaning errors than base Whisper large-v3 (14.7 %): it
  garbles brand names ("Nobel-Baker", "Kamloch-Konelloch"). ElevenLabs has a WER of
  18.2 % but only 5 minor errors for Claude, because it writes what was said ("mä", "semmonen")
  where the reference has standard forms. The reference also looks like a corrected
  WhisperX draft, which favours Whisper-family output on WER.
- 👂 **A listener's review of the reference confirms it.** After the 42 spans most systems
  disputed were checked by ear (14 corrected), the WhisperX lead disappears and the leading set
  sits within about one point, 13.2–14.2 % WER ([Reviewed reference](docs/REPORT.md#reviewed-reference)).
- 🖥️ **Local and good enough: Whisper large-v3 or large-v3 turbo.** Every other open
  model is clearly behind on this clip: Orukeet, RASMUS Finnish Canary, Canary 1B v2,
  Parakeet v3 and wav2vec2 at 22–32 % WER.
- ❌ **Qwen3-ASR is not usable for Finnish**: 58 % WER (1.7B) and 78 % (0.6B), although
  Finnish is on its language list.
- 🇫🇮 **The Finnish Canary fine-tune did not beat its base on names.** RASMUS
  Finnish-ASR-Canary-v2 scores 22–24 % WER against 24 % for NVIDIA's Canary 1B v2, and
  gets fewer of the 14 names right (7–11 against 10–13). Its model card reports 4.6 %
  on Common Voice: read speech, not a webinar.

<!-- readme-table:start -->
| Model | Type | WER % | CER % | Claude pts | Gemini pts | Names (Claude / Gemini) |
|---|---|--:|--:|--:|--:|--:|
| Gemini 3.5 Flash | API | 14.2 (14.2–17.9) | 5.0 | 0.8 | 1.2 | 14 / 14 of 14 |
| Google Chirp 3 | API | 19.6 | 6.6 | 1.2 | 3.7 | 14 / 11 of 14 |
| Whisper large-v3, chunked decoding | open | 15.7 | 6.7 | 1.7 | 5.5 | 14 / 14 of 14 |
| Whisper large-v3 | open | 14.7 | 6.0 | 2.7 | 5.2 | 14 / 13 of 14 |
| ElevenLabs Scribe v2 | API | 18.2 (17.2–18.4) | 4.9 | 1.2 | 7.0 | 14 / 12 of 14 |
| This project, la-murre version, chunked decoding | open | 15.9 | 4.7 | 3.0 | 6.2 | 14 / 12 of 14 |
| Whisper large-v3 turbo | open | 18.4 | 7.7 | 1.5 | 8.5 | 14 / 13 of 14 |
| This project's pilot: Whisper large-v3 + LoRA on FLEURS and VoxPopuli | open | 17.9 | 8.3 | 2.2 | 8.2 | 14 / 13 of 14 |
| OpenAI gpt-transcribe | API | 18.4 (18.4–19.1) | 8.5 | 4.2 | 8.5 | 12 / 12 of 14 |
| WhisperX + Finnish-NLP whisper-large-finnish-v3 (Haaga-Helia WhisperX server) | open | 11.9 | 3.5 | 4.5 | 8.7 | 13 / 11 of 14 |
| This project, round 1, chunked decoding | open | 14.9 | 5.3 | 3.7 | 9.7 | 14 / 13 of 14 |
| OpenAI gpt-4o-transcribe | API | 16.9 (16.2–18.4) | 6.7 | 6.0 | 7.5 | 11 / 10 of 14 |
| OpenAI whisper-1 | API | 23.1 | 11.2 | 3.0 | 10.4 | 14 / 14 of 14 |
| Finnish-NLP whisper-large-finnish-v3 (plain transformers) | open | 13.2 | 4.3 | 5.2 | 8.5 | 13 / 11 of 14 |
| This project, round 1: Whisper large-v3 + LoRA on SKN dialects, FLEURS and VoxPopuli | open | 17.9 | 5.6 | 4.5 | 9.4 | 13 / 12 of 14 |
| OpenAI gpt-4o-mini-transcribe | API | 13.9 (13.9–14.7) | 4.4 | 3.2 | 11.4 | 13 / 10 of 14 |
| Orukeet (Parakeet v3 fine-tune) | open | 26.4 | 10.2 | 6.2 | 10.7 | 13 / 11 of 14 |
| This project, first model (`v1.0`), chunked decoding | open | 13.9 | 5.2 | 6.7 | 10.9 | 11 / 10 of 14 |
| RASMUS Finnish-ASR-Canary-v2, greedy | open | 23.9 | 8.1 | 6.5 | 13.7 | 11 / 8 of 14 |
| RASMUS Finnish-ASR-Canary-v2 + KenLM | open | 22.1 | 8.5 | 8.5 | 15.4 | 11 / 7 of 14 |
| Gemini 3.1 Pro | API | 22.4 (15.4–22.4) | 9.3 | 7.0 | 18.4 | 12 / 10 of 14 |
| NVIDIA Canary 1B v2 | open | 24.1 | 10.3 | 9.0 | 17.9 | 13 / 10 of 14 |
| NVIDIA Parakeet TDT 0.6B v3 | open | 31.1 | 13.8 | 12.7 | 17.4 | 13 / 12 of 14 |
| Finnish-NLP wav2vec2 XLS-R 1B (no LM) | open | 31.6 | 9.1 | 6.2 | 26.1 | 13 / 10 of 14 |
| Qwen3-ASR 1.7B | open | 58.0 | 23.2 | 20.6 | 33.6 | 9 / 7 of 14 |
| Qwen3-ASR 0.6B | open | 77.9 | 33.9 | 28.1 | 41.0 | 7 / 4 of 14 |
<!-- readme-table:end -->

Sorted by the mean of the two judges. **pts** = meaning-error points per 100 reference words (critical 10, major
5, minor 1). **Names** = how many of the clip's 14 names and terms each judge found right. WER and CER use HY's
normalisation ([`asr_scoring.py`](scripts/asr_scoring.py)); a range is over three runs. Every number:
[`results/scores.json`](results/scores.json). File names in `results/` start with where the model ran (`openai_`,
`vertex_`, `elevenlabs_`, `google_` APIs; `hh_` the Haaga-Helia WhisperX server; `hf_`, `nemo_`, `qwen_` open
models; `ft_` our fine-tunes).

**The reference was checked where it mattered.** HY's reference looked like a corrected WhisperX draft; on 14 Oct
2026 Akseli Leskinen listened to the 42 spans where most systems hear something else and corrected 14 of them.
Against the checked text the WhisperX lead disappears and the best set-ups sit within about one point of each
other, 13.2–14.2 % WER, which one 408-word clip cannot separate
([table and details](docs/REPORT.md#reviewed-reference)). The clip has not been run with the current main model
or the `glossary` version yet.

## Timestamps and long-form audio

The published models were trained without Whisper's timestamp tokens, so their own timestamps are unusable: 1 % of
SKN sentence starts get a timestamp within 1 s (base Whisper 61 %). Their text is good, so decode them in 15–25 s
pieces and take the times from the cut points, as the model card's code does. Measured against SKN's own sentence
times and ten 4-minute interview windows ([`eval_timestamps.py`](scripts/eval_timestamps.py); full table in the
[report](docs/REPORT.md#timestamps-and-long-form-audio-full-measurements)):

| | Whisper large-v3 | `v1.0` | `timestamps` version |
|---|--:|--:|--:|
| SKN sentence starts within 1 s | 60.8 % | 0.8 % | 57.8 % |
| 4-minute interviews in one go, with fallback, WER % | 37.5 | 19.8 | **18.1** |
| The same in 15–25 s pieces, WER % | 39.4 | 17.3 | **17.1** |

### The timestamp variant

`v1.0`'s recipe with half of the SKN items carrying one timestamp pair per sentence. It keeps the dialect
accuracy (SKN 21.4 % against 21.1 %) but VoxPopuli got 0.4 points worse, so by the rule set before the run it did
not replace the main model; it is the `timestamps` revision ([card](release/TIMESTAMPS_CARD.md)). Use it for
sentence timestamps or long audio in one go, always with the fallback settings (without them it looped in 3 of 10
interviews):

```python
import soundfile as sf
from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition",
    model="GAIK-project/whisper-large-v3-fi-dialects",
    revision="timestamps",
    dtype="float16",
    device="cuda",
)
audio, sr = sf.read("interview.wav", dtype="float32")  # 16 kHz mono, any length
out = asr(
    {"raw": audio, "sampling_rate": sr},
    return_timestamps=True,
    generate_kwargs={
        "language": "fi",
        "task": "transcribe",
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "compression_ratio_threshold": 1.35,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_prev_tokens": False,
    },
)
for chunk in out["chunks"]:
    print(f"[{chunk['timestamp'][0]:7.2f} s] {chunk['text'].strip()}")
```

## Dialect data and next steps

| Dataset | What | Access |
|---|---|---|
| Suomen kielen näytteitä (SKN) | Dialect interviews from 50 localities, dialect and standard-Finnish transcripts | Kielipankki, CC BY 4.0: **used** for training and the dialect test |
| Lauseopin arkiston murrekorpus (LA-murre) | 142 dialect interviews from 132 localities, dialect transcripts only | Kielipankki, CC BY-ND 4.0: **used** as the second dialect test and for the `la-murre` and later versions |
| Aalto Finnish Parliament ASR Corpus | Plenary sessions 2008–2020 | Kielipankki, CLARIN PUB +BY +PRIV +ND: **used**, 72 h for the main model |
| Lahjoita puhetta (Donate Speech) | About 3 600 h of everyday Finnish from over 20 000 speakers | Kielipankki, CLARIN RES: needs a research plan |

On 20 Oct 2026 Kielipankki (FIN-CLARIN) assessed that weights trained on LA-murre most likely contain no
copyright-protected part of the corpus, so the `la-murre` and later versions could be published. Still open:

1. **The rewritten LA-murre texts.** They stay out of this repository and off Hugging Face; Kielipankki, which may
   distribute modified LA-murre material, has offered to take them in as a new version of the corpus.
2. **Today's speech.** Both dialect corpora are 1960s–70s interviews with elderly speakers; everyday conversation,
   slang and Finnish-English mixing are not tested.
3. **Names and terms in business speech.** The `glossary` version helps when the names are known in advance; a
   test on real customer speech with its own word list is still to do.
4. **Checked webinar references.** Only Ajokortti's reference has been checked by ear; the other nine webinar
   clips need a listener before their word error rates mean much.

The longer write-up of the 15–20 Oct steps (all of SKN, how well the LLM rewrites dialect, the API services on
the dialect test) is in the [report](docs/REPORT.md#dialect-data-and-next-steps-1520-oct-2026).

## What's in this folder

| Path | What it holds |
|---|---|
| [`scripts/`](scripts/) | Runners, scorer, judges, table and chart builders (see [Run it again](#run-it-again)). [`provenance.py`](scripts/provenance.py) writes each run's `_run_metadata*.json`; [`asr_scoring.py`](scripts/asr_scoring.py) is the clip scorer with HY's normalisation; [`get_clip.py`](scripts/get_clip.py) fetches the clip. |
| [`tests/`](tests/) | Offline tests (`uv run pytest`): no keys, no downloads, no GPU. |
| [`lumi/`](lumi/) | How the open models were run and trained on LUMI: environment setup, Slurm jobs, and stubs that hide three container libraries NeMo cannot use. |
| [`release/`](release/) | The Hugging Face cards, uploaded by [`publish_hf.py`](scripts/publish_hf.py): [`MODEL_CARD.md`](release/MODEL_CARD.md) (the main model), [`TIMESTAMPS_CARD.md`](release/TIMESTAMPS_CARD.md) (its `timestamps` revision), [`ALL_SKN_CARD.md`](release/ALL_SKN_CARD.md) (its `all-skn` revision) and [`DATASET_CARD.md`](release/DATASET_CARD.md). |
| [`docs/`](docs/) | [`REPORT.md`](docs/REPORT.md) (the long write-up of every round, judge round and decision), [`DIALECT-EXAMPLES.md`](docs/DIALECT-EXAMPLES.md) (20 SKN test segments side by side and WER per municipality), [`story/`](docs/story/) (the walk-through pages, one self-contained HTML file per language, built from `story/src/`) and [`img/`](docs/img/) (the dialect chart from `dialect_chart.py`, the front page's figures from [`readme_figures.py`](scripts/readme_figures.py), the banner, an AI illustration made with GPT Image 2.5, and the sources' logos in `logos/`, their owners' trademarks, from [svgl](https://svgl.app/)). [`video/`](docs/video/) holds the front page's 25-second video (21 Oct 2026; built as HTML with HyperFrames): two shots made with Runway Gen-4.5 through Replicate, the earlier video's ElevenLabs Music track, no people, no voice. The story pages are also served at [gaik-project.github.io/finnish-asr](https://gaik-project.github.io/finnish-asr/). |
| [`results/transcriptions/`](results/transcriptions/) | One JSON per model and run on the clip: transcript, settings, library versions, timing. [`qad1/`](results/transcriptions/qad1/) holds GAIK's earlier QAD-1 runs (OpenAI, Gemini, the gaik transcriber, the Haaga-Helia WhisperX server, Whisper on CSC Roihu). |
| [`results/judgements/`](results/judgements/) | Per judge and model: every meaning error quoted from the reference and the transcript, the name checks, reference doubts, scores. |
| [`results/scores.json`](results/scores.json), [`scores.reviewed.json`](results/scores.reviewed.json) | WER, CER and error counts for every clip transcript, against HY's and the reviewed reference. |
| [`results/clip/`](results/clip/) | [`gt_alignment.json`](results/clip/gt_alignment.json): the reference's sentences, word times and QAD-1's 42 reference doubts, frozen from QAD-1 for the listening page. [`decoding_check.json`](results/clip/decoding_check.json) and [`decoding_scores.json`](results/clip/decoding_scores.json): the published model from the Hub decoded six ways, the model card's settings table. |
| [`results/benchmarks/`](results/benchmarks/) | SKN, FLEURS and VoxPopuli: one line per recording (reference, transcript, seconds) and `scores.json`. LA-murre: `scores.json` and run metadata only. |
| [`results/judgements_dialect/`](results/judgements_dialect/) | The LA-murre meaning judges: `summary*.json` (rounds 1–3 and the Aitta judge), `agreement-r2.json` and run metadata (numbers only). |
| [`results/finetune/`](results/finetune/) | The training log of every fine-tune: per-epoch validation loss, data counts, the epoch kept. No weights. |
| [`results/labels/`](results/labels/) | SKN training targets with LLM punctuation, SKN sentence times per segment, the long-form windows; for LA-murre only the statistics. |
| [`results/timestamps/`](results/timestamps/) | Raw decodes with timestamps and long-form windows (base Whisper, the published model, the timestamp variant), and their scores. |
| [`results/gpu_run/`](results/gpu_run/) | GPU, host and library versions of the last clip job per environment (hf, nemo, qwen) on LUMI. |

Not here, on purpose: the clip and its references (download them from gaik-toolkit with
`get_clip.py`; `data/` is gitignored), any LA-murre text (per-segment benchmark files, judge records,
rewritten training texts: licence), HY's own tool transcripts (not public), model weights (the
published model is on Hugging Face) and LUMI job logs. The LA-murre outputs the public scripts can
regenerate are listed in [`.gitignore`](.gitignore).

## How it works

**Input.** Every model gets the same 180 s file with Finnish set as the language where
the model takes one, and no prompt or word list. Whisper and Qwen3-ASR take the whole
clip. Models limited to short audio (Canary, Parakeet, Orukeet, wav2vec2, Chirp 3)
get the same eight chunks, cut at the quietest 100 ms between 15 and 25 s.

**WER and CER.** The scorer of QAD-1 and the toolkit's transcription_eval, with HY's normalisation,
vendored in [`asr_scoring.py`](scripts/asr_scoring.py) and unchanged, so the numbers match HY's and
QAD-1's published ones. The public test sets use the usual model-card normalisation (lower-case,
punctuation removed, whitespace collapsed) and corpus-level WER ([`score_benchmark.py`](scripts/score_benchmark.py)).

**LLM judges on the clip.** Claude Opus 5.5 and Gemini 3.1 Pro each read the reference and one
transcript, without the model's name ([`judge_asr.py`](scripts/judge_asr.py)). The prompt tells them
to report only what would mislead a reader: a wrong word, name or number, missing content, invented
content. Fillers, spoken against written forms, compound spacing, punctuation and digits
against number words are explicitly not errors. Each judge also checks 14 names and
terms against their correct spelling (the reference itself misspells Dentsply Sirona)
and lists places where the reference looks wrong; those are reported, not scored.

**SKN data and training.** `prepare_skn.py` cuts SKN by municipality (2–25 s, seed 0);
`prepare_targets.py punct` adds LLM punctuation and keeps an answer only if its words are identical;
[`finetune_whisper.py`](scripts/finetune_whisper.py) trains a LoRA adapter on LUMI in a plain PyTorch
loop and keeps the epoch with the lowest validation loss.

**The LA-murre judge.** `judge_dialect.py` gives one judge the dialect reference and every model's
transcript of a segment at once, labelled with letters in an order shuffled per segment, and counts
meaning errors only. Rounds judge different sets of models (round 1: base Whisper, round 1's model,
Finnish-NLP; round 2: base Whisper and our three round-1 and round-2 models; round 3: base Whisper,
the published model and the timestamp variant); `--provider aitta` runs the same judgement on an open
model on CSC Aitta, and `judge_agreement.py` compares judges segment by segment. The
published scores were made with prompt `dialect-judge-v1`; this repository's copy is `v2`, identical
except that its three example word pairs come from SKN instead of LA-murre (the docstring explains).

**Timestamps.** `prepare_timestamps.py` turns SKN's sentence annotation into sentence times inside
every published segment and into the 10 long-form windows; `finetune_whisper.py --timestamps 0.5`
trains with them; `eval_timestamps.py` decodes on LUMI (`--fallback` adds the temperature-fallback
decode) and scores locally.

**Decoding settings.** `score_asr.py --decoding` scores the published model's six decodings of the
clip in `results/clip/decoding_check.json` against both references; the model card's table of
settings to avoid comes from it.

**Provenance.** Every runner that writes results writes a secret-free `_run_metadata*.json` beside
them: commit, runner, dataset path, models, arguments, and which keys were set (never their values).
Files made before this repository existed carry `gaik_evals_commit` and paths of GAIK's private
evaluation repository, where the work was first done; their content is unchanged.

**Where it ran.** Open models and training: one GPU on LUMI (AMD MI250X) in CSC's PyTorch
container. ElevenLabs, Chirp 3, OpenAI and Gemini: their APIs. LLM judges: Claude through Azure
Foundry, Gemini through Vertex AI, open models through CSC Aitta.

## Run it again

From the repo root, with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras                                  # Python deps for every local step
uv run pytest                                         # offline tests: no keys, no downloads, no GPU
uv run python scripts/get_clip.py --prepare-audio     # the clip and both references from gaik-toolkit (ffmpeg on PATH)

# the clip: API models, scores, judges, the README table
uv run python scripts/run_api_asr.py --repeats 3      # ElevenLabs, Chirp 3
uv run python scripts/score_asr.py                    # WER/CER, no keys; --reviewed for the reviewed reference
uv run python scripts/score_asr.py --decoding         # the model card's decoding-settings table
uv run python scripts/judge_asr.py                    # both judges (--claude-provider api for an Anthropic key)
uv run python scripts/make_readme_table.py            # the clip table above

# open models, public test sets and training on LUMI, one wrapper for every step (details: lumi/README.md)
export LUMI_USER=<your CSC user name> LUMI_PROJECT=project_<number>
L=lumi/lumi.sh
bash $L push && bash $L setup && bash $L download clip && bash $L download bench   # once
for e in hf qwen nemo; do bash $L run clip $e; bash $L run bench $e; done            # six jobs
bash $L status                                                                       # until the queue is empty
bash $L fetch
uv run --extra api-asr --extra judges --extra hub python scripts/run_api_benchmark.py \
  --arm vertex_gemini-3.5-flash --arm vertex_gemini-3.5-flash-std --arm google_chirp-3   # Gemini 3.5 Flash (two prompts), Chirp 3 on SKN
uv run python scripts/score_benchmark.py              # SKN/FLEURS/VoxPopuli table, no keys

# the dialect model: SKN, punctuated targets, training, its tests
bash $L download skn && bash $L prepare skn
bash $L fetch-labels                                  # prepare_skn.py's segment lists -> results/labels/ (gitignored)
uv run python scripts/prepare_targets.py punct results/labels/skn_train.jsonl
bash $L push-labels && bash $L download train
bash $L run train hf --name whisper-large-v3-fi-skn-punct-best --data skn+open --skn-text punct --epochs 3 --lora-r 64
bash $L run clip hf --arm ft_whisper-large-v3-fi-skn-punct-best-chunks
bash $L run bench hf --bench skn_fi && bash $L fetch
uv run python scripts/dialect_chart.py                # docs/img/dialect-finetune.svg, no keys
uv run python scripts/dialect_examples.py             # docs/DIALECT-EXAMPLES.md, no keys
uv run python scripts/build_story.py                  # docs/story/story.{en,fi}.html, no keys
uv run python scripts/readme_figures.py              # the front page's figures in docs/img/, no keys
uv run --extra hub python scripts/build_listen.py    # docs/listen.html and its SKN clips, no keys
uv run python scripts/build_compare.py                # results/compare.html (gitignored); fetches the SKN test parquet

# timestamps and long-form (prepare_timestamps.py needs SKN's *_full.eaf files)
uv run python scripts/prepare_timestamps.py <dir with SKN *_full.eaf>
bash $L push-labels && bash $L run timestamps hf --arm hf_whisper-large-v3 --arm hub_whisper-large-v3-fi-dialects

# the timestamp variant: training, its tests, and the timestamp evaluation with the fallback decode
bash $L run train hf --name whisper-large-v3-fi-skn-punct-ts --data skn+open --skn-text punct --epochs 3 --lora-r 64 --timestamps 0.5
bash $L run clip hf --arm ft_whisper-large-v3-fi-skn-punct-ts --arm ft_whisper-large-v3-fi-skn-punct-ts-chunks
for b in skn_fi fleurs_fi voxpopuli_fi; do bash $L run bench hf --bench $b --arm ft_whisper-large-v3-fi-skn-punct-ts; done
bash $L run timestamps hf --arm ft_whisper-large-v3-fi-skn-punct-ts
bash $L run timestamps hf --fallback --arm hf_whisper-large-v3 --arm hub_whisper-large-v3-fi-dialects --arm ft_whisper-large-v3-fi-skn-punct-ts
bash $L fetch && uv run python scripts/eval_timestamps.py score

# LA-murre (CC BY-ND): downloaded from Kielipankki; every per-segment output stays local (gitignored)
bash $L prepare lamurre && bash $L run bench hf --bench lamurre_fi && bash $L fetch
uv run python scripts/judge_dialect.py --round r2 --arms hf_whisper-large-v3,ft_whisper-large-v3-fi-skn,ft_whisper-large-v3-fi-skn-punct-best,ft_whisper-large-v3-fi-skn-lamurre-best
uv run python scripts/judge_dialect.py --round r2 --arms <the same> --provider aitta --judge-model openai/gpt-oss-120b
uv run python scripts/judge_agreement.py --round r2
uv run python scripts/judge_dialect.py --round r3 --arms hf_whisper-large-v3,ft_whisper-large-v3-fi-skn-punct-best,ft_whisper-large-v3-fi-skn-punct-ts

# the API services on every test set (FLEURS, VoxPopuli and LA-murre WAVs: bash $L fetch-audio <bench>)
for b in skn_fi fleurs_fi voxpopuli_fi lamurre_fi; do uv run --extra api-asr python scripts/run_api_benchmark.py   --bench $b --arm azure_gpt-4o-transcribe --arm elevenlabs_scribe-v2 --arm google_chirp-3; done

# all of SKN: every segment of the 12 held-out municipalities joins the training data
bash $L prepare skn-heldout && bash $L fetch-labels
uv run python scripts/prepare_targets.py punct results/labels/skn_heldout.jsonl && bash $L push-labels
bash $L run train hf --name whisper-large-v3-fi-skn-all --data skn-all+open --skn-text punct --epochs 3 --lora-r 64
for b in fleurs_fi voxpopuli_fi lamurre_fi; do bash $L run bench hf --bench $b --arm ft_whisper-large-v3-fi-skn-all; done
uv run python scripts/judge_dialect.py --round r4 --arms hf_whisper-large-v3,ft_whisper-large-v3-fi-skn-punct-best,ft_whisper-large-v3-fi-skn-all,ft_whisper-large-v3-fi-skn-lamurre-best,azure_gpt-4o-transcribe,vertex_gemini-3.5-flash-std
uv run python scripts/judge_dialect.py --round r4 --arms <the same> --provider aitta --judge-model openai/gpt-oss-120b   # Gemma 4 and Qwen3.6: --per-locality 20

# the ten dental-webinar clips (HY's private references in data/dental10/, gitignored)
bash $L push-dental && bash $L run clip hf --bench dental10_fi
uv run --extra judges python scripts/judge_dental.py   # blind judges; only summary.json is committed
```

Each step skips what it has already done; `--force` redoes it (`bash $L run clip hf --force`).
To add a model, add a line to `ARMS` in `run_open_models.py` and `run_benchmark.py`
and run `push`, `download` and `run` for its environment. `bash $L clean` removes the
project from LUMI's scratch when the work is over. Keys go in a `.env` at the repo root;
[`.env.example`](.env.example) lists their names per script. `publish_hf.py` needs a GAIK-project
write token and is for maintainers.

## Caveats

- **One 180-second clip.** It separates usable from unusable models and shows where WER
  misleads. It cannot rank models that are close, and it says nothing about dialects,
  noise or phone audio.
- **The ten dental webinars' references are unchecked.** They leave out fillers and a spot check found
  wrong and missing words; their numbers are a hint, not a ranking
  ([Ten dental webinars](docs/REPORT.md#ten-dental-webinars-references-not-checked)).
- **The judges are AI models and they are noisy**; read clip differences under about 2 points as
  ties ([How far the LLM judges can be trusted](docs/REPORT.md#how-far-the-llm-judges-can-be-trusted)).
- **The reference is reviewed only at its 42 disputed spans.** On 14 Oct 2026 a listener checked
  them and corrected 14 (gaik-toolkit's `reference.reviewed.txt`; see
  [Reviewed reference](docs/REPORT.md#reviewed-reference)); the rest is HY's unverified text. The judges flag
  0–3 reference doubts per transcript.
- **Each open model ran once**, with greedy or default decoding. The Finnish-NLP
  wav2vec2 model ran without its KenLM language model (it would not build in the
  container), which costs it accuracy. The RASMUS model ran with this project's energy
  chunking, not its own VAD or diarisation pipeline.
- **Speed was not measured on the clip**: the recorded times include model loading.
- **The SKN test shares SKN's conventions.** Test municipalities are never heard in
  training, but the recordings, interviewers' style and transcription conventions are
  SKN's own, which favours the model trained on SKN; the LA-murre test does not have
  that problem but is scored by model judges. Both corpora are 1960s–70s interviews
  with elderly speakers, not today's regional speech. The other models were not given SKN's
  conventions (for example, numbers written as words), which costs them some WER.
  Each fine-tune was trained once with one seed; read gaps of about 1 point per 100 words on
  LA-murre as ties unless the sign test says otherwise.
- **The LA-murre numbers cannot be checked from this repository alone.** No LA-murre text is here;
  rebuilding the test from Kielipankki (`prepare_lamurre.py`) and rerunning the models and the judge
  reproduces them. The public judge prompt differs from the one that made them in three example
  word pairs.
- **Two of the Aitta judges saw only a sample.** gpt-oss-120b judged all 583 segments; Gemma 4 31B
  and Qwen3.6 27B a seeded sample of 200 (198 and 197 answered), so their numbers carry more sampling noise.
- **The timestamp results rest on SKN's own sentence times and ten 4-minute windows**, and the
  timestamp variant, like every fine-tune here, was trained once with one seed.
- **The training texts are partly LLM-written.** Punctuation (SKN) and standard spellings
  (LA-murre, `la-murre` revision) come from Opus 5.5 with a word-level check, not from a person. The tests
  are not touched: their references are the corpora's human transcripts.
- **Older provenance points to a private repository.** Result files made before this repository
  existed record `gaik_evals_commit` and paths in GAIK's private evaluation repository, where the work
  was first done.

## Licences, attribution and credits

- **Code** (`scripts/`, `tests/`, `lumi/`): MIT, [`LICENSE`](LICENSE), copyright Haaga-Helia
  University of Applied Sciences (GAIK project).
- **Results, figures and documentation created by the GAIK project** (scores, judge records,
  training logs and model transcripts in `results/`, the chart, the story pages and these texts):
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit: GAIK project, Haaga-Helia
  University of Applied Sciences.
- **Data from other sources keep their licences:**
  - **Suomen kielen näytteitä (SKN)**, Kotimaisten kielten keskus (Kotus), via Kielipankki – The
    Language Bank of Finland, [urn:nbn:fi:lb-2020112937](http://urn.fi/urn:nbn:fi:lb-2020112937),
    [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: cut into 2–25 s segments;
    punctuation and capitals added to the training and validation targets by Claude Opus 5.5 (words
    unchanged).
  - **FLEURS**, Google, [google/fleurs](https://huggingface.co/datasets/google/fleurs),
    [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Conneau et al., "FLEURS: Few-shot
    Learning Evaluation of Universal Representations of Speech", IEEE SLT 2022
    ([arXiv:2205.12446](https://arxiv.org/abs/2205.12446)).
  - **VoxPopuli**, Meta, [facebook/voxpopuli](https://huggingface.co/datasets/facebook/voxpopuli),
    [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Wang et al., "VoxPopuli: A
    Large-Scale Multilingual Speech Corpus for Representation Learning, Semi-Supervised Learning and
    Interpretation", ACL 2021 ([arXiv:2101.00390](https://arxiv.org/abs/2101.00390)).
  - **Lauseopin arkiston murrekorpus (LA-murre)**, University of Turku and Kotus, via Kielipankki,
    [urn:nbn:fi:lb-2020112935](http://urn.fi/urn:nbn:fi:lb-2020112935),
    [CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/): the test set for every model
    here; only the `la-murre` revision was also trained on it (other localities). No LA-murre
    text is in this repository, only numbers made from it.
- **Model weights**: the published model and its `timestamps`, `all-skn` and `la-murre` revisions are
  [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), as the base model
  openai/whisper-large-v3; the published dataset is CC BY 4.0.
- **The Ajokortti clip** is an excerpt of a public QAdental webinar; QAdental (Suomen
  Kotilääkäripalvelu Oy) is a GAIK project partner. The audio is not stored here and is not covered
  by this repository's licences; [`get_clip.py`](scripts/get_clip.py) downloads it, with its
  reference transcripts (by the University of Helsinki team, and the reviewed one), from
  [gaik-toolkit](https://github.com/GAIK-project/gaik-toolkit/tree/main/evaluation_layer/eval_methods/transcription_eval),
  where they are public.
- **Story pages**: the illustration is AI-generated (GPT Image 2.5), the two audio clips are SKN
  segments (CC BY 4.0), and the icons are [Lucide](https://lucide.dev/) (ISC licence); the database
  and link icons are derived from [Feather](https://feathericons.com/) (MIT licence).

Details in [`NOTICE.md`](NOTICE.md).

Developed by Akseli Leskinen ([laguagu](https://github.com/laguagu)), Haaga-Helia University of
Applied Sciences, in the [GAIK project](https://gaik.ai/) (Generative AI-enhanced Knowledge
Management in Business), led by Haaga-Helia with the University of Helsinki and Tampere University,
co-funded by the European Union (ERDF). The Ajokortti reference transcript was made by the University
of Helsinki team.

**LUMI.** We acknowledge CSC – IT Center for Science, Finland for awarding this project access to
the LUMI supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CSC (Finland) and the LUMI
consortium through CSC – IT Center for Science, Finland.

**Roihu.** The Nemotron 3.5 Transcribe row was measured on CSC's Roihu supercomputer. We acknowledge CSC – IT Center
for Science, Finland, for computational resources.
