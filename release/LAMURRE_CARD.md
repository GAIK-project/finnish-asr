---
language:
- fi
license: apache-2.0
base_model: openai/whisper-large-v3
base_model_relation: finetune
pipeline_tag: automatic-speech-recognition
library_name: transformers
tags:
- whisper
- finnish
- dialects
- speech-recognition
- lora
datasets:
- GAIK-project/skn-finnish-dialects-asr
- google/fleurs
- facebook/voxpopuli
model-index:
- name: whisper-large-v3-fi-dialects (revision la-murre)
  results:
  - task:
      type: automatic-speech-recognition
    dataset:
      name: FLEURS Finnish
      type: google/fleurs
      config: fi_fi
      split: test
    metrics:
    - type: wer
      value: 7.8
  - task:
      type: automatic-speech-recognition
    dataset:
      name: VoxPopuli Finnish
      type: facebook/voxpopuli
      config: fi
      split: test
    metrics:
    - type: wer
      value: 11.4
---

# Whisper large-v3 for Finnish dialects, with 56 more hours of dialect interviews (revision `la-murre`)

This branch holds a **variant** of [GAIK-project/whisper-large-v3-fi-dialects](https://huggingface.co/GAIK-project/whisper-large-v3-fi-dialects):
the same LoRA fine-tune of [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3),
trained on the main model's data **plus 56 hours of dialect interviews from the Finnish Dialect
Syntax Archive (LA-murre)**, 124 localities. It hears dialect and writes standard Finnish with
punctuation.

It is this project's most accurate model on dialect speech: 19.7 % of words wrong on the SKN dialect
test, against the main model's 21.1 %. On read and parliament speech it is slightly weaker than the
main model, so the main branch (tag `v1.0`) stays the default. Use this variant when your audio is
mostly dialect.

Trained at Haaga-Helia University of Applied Sciences in the [GAIK project](https://gaik.ai/)
(Generative AI-enhanced Knowledge Management in Business), a research project led by Haaga-Helia
with the University of Helsinki and Tampere University, co-funded by the European Union (ERDF).

**[GAIK project](https://gaik.ai/)** · [Haaga-Helia University of Applied Sciences](https://www.haaga-helia.fi/en) · developed by Akseli Leskinen ([laguagu](https://github.com/laguagu))

📖 The story, with audio: [suomeksi](https://gaik-project.github.io/finnish-asr/story/story.fi.html) · [in English](https://gaik-project.github.io/finnish-asr/story/story.en.html) · 🔊 [listen and compare models](https://gaik-project.github.io/finnish-asr/listen.html) · 🗂️ [training data (SKN part)](https://huggingface.co/datasets/GAIK-project/skn-finnish-dialects-asr)

📊 [code, method and all results: GAIK-project/finnish-asr](https://github.com/GAIK-project/finnish-asr#dialect-data-and-next-steps) · 🧰 [gaik-toolkit](https://github.com/GAIK-project/gaik-toolkit) · 💻 [GAIK on GitHub](https://github.com/GAIK-project) · 🌐 [gaik.ai](https://gaik.ai/)

## Results

| Test | Whisper large-v3 | Main model (`v1.0`) | This variant |
|---|--:|--:|--:|
| SKN dialect interviews, 10 municipalities in no model's training, WER % (600 segments) | 40.4 | 21.1 | **19.7** |
| LA-murre dialect interviews, 10 localities held out of all training: meaning-error points per 100 words (blind LLM judge, 583 segments) ¹ | 25.0 | 14.2 | **11.6** |
| FLEURS Finnish, read sentences, WER % | 7.4 | **7.6** | 7.8 |
| VoxPopuli Finnish, parliament speech, WER % | 12.7 | **10.8** | 11.4 |
| A 180 s dental webinar in 15–25 s pieces, WER % | 15.7 | **13.9** | 15.9 |
| The same webinar: meaning-error points per 100 words (Claude / Gemini judge) | **1.7 / 5.5** | 6.7 / 10.9 | 3.0 / 6.2 |

WER after lower-casing and removing punctuation. ¹ The judge (Claude Opus 5.5) saw six transcripts of
each segment together, unlabelled and shuffled, in the fourth judging round, where the
[`all-skn` revision](https://huggingface.co/GAIK-project/whisper-large-v3-fi-dialects/blob/all-skn/README.md)
got 13.1. A second judge, GPT-6.1-sol, ranks the three the same way (14.9, 15.5 and 16.3 for this
variant, `all-skn` and the main model). LA-murre's test localities were held out of this variant's
training, but its other recordings were in it, so part of its lead on that test comes from knowing
the corpus. Compare points only within one judging round.

The variant does not replace the main model: by the rule used for the `all-skn` variant, FLEURS and
VoxPopuli may get at most 0.5 points worse, and VoxPopuli got 0.6 points worse.

## Use it

Exactly as the [main model](https://huggingface.co/GAIK-project/whisper-large-v3-fi-dialects), with
`revision="la-murre"`, and long audio cut into 15–25 s pieces at quiet points as its card shows:

```python
from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition",
    model="GAIK-project/whisper-large-v3-fi-dialects",
    revision="la-murre",
    dtype="float16",
    device="cuda",
)
print(asr("piece.wav", generate_kwargs={"language": "fi", "task": "transcribe"})["text"])
```

## How it was trained

The main model's recipe, with more dialect speech:

1. 🎙️ **Data.** The main model's training data: 15 042 SKN dialect segments
   ([Suomen kielen näytteitä](https://www.kielipankki.fi/corpora/skn/), 38 municipalities, 2–25 s),
   [FLEURS](https://huggingface.co/datasets/google/fleurs) `fi_fi` train (2 704) and
   [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) `fi` train (7 504). Added:
   15 529 segments (56.5 h) of [LA-murre](https://www.kielipankki.fi/corpora/la-murre/) interviews
   from 124 localities. Left out of training: LA-murre's 10 test localities, a second recording from
   a test locality, recordings from SKN's test municipalities, and two recordings kept for validation.
2. ✍️ **Training text.** SKN's targets are its human standard-Finnish transcripts with punctuation and
   capitals added by an LLM (Claude Opus 5.5, words unchanged). LA-murre has only dialect
   transcripts, so the same LLM rewrote them in standard Finnish, word for word, without
   rephrasing. Run on SKN's dialect transcripts, the same prompt was 14.2 % WER away from SKN's own
   human standard-Finnish text, so these targets are close to, not the same as, a person's work.
3. ⚙️ **Fine-tuning.** LoRA rank 64 on the attention `q_proj` and `v_proj`, learning rate 1e-4,
   batch 8, three epochs; validation on 1 476 items (SKN's two validation municipalities, the two
   LA-murre validation recordings, FLEURS and VoxPopuli). The epoch with the lowest validation loss
   is kept (epoch 2: 0.236). The adapter is merged into the weights published here.
4. 🖥️ **Compute.** 4 h 57 min on one AMD MI250X GPU on the [LUMI](https://www.lumi-supercomputer.eu/) supercomputer.

## Where it is weaker

- Read and parliament speech: 0.2 and 0.6 points more word errors than the main model on FLEURS and
  VoxPopuli.
- **Rare names and technical terms** in standard-Finnish audio, like the main model: for standard
  Finnish full of brand names, plain Whisper large-v3 or a commercial service is the safer choice.
- Both dialect corpora are 1960s–70s interviews with elderly speakers; today's regional speech was
  not tested. One training run and one seed.

## Credits and licences

- Developed by Akseli Leskinen ([laguagu](https://github.com/laguagu)), Haaga-Helia University of Applied Sciences, in the [GAIK project](https://gaik.ai/).
- Weights: Apache 2.0, as the base model [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3).
- **Suomen kielen näytteitä (SKN)**, Kotimaisten kielten keskus (Kotus), via Kielipankki – The
  Language Bank of Finland, CC BY 4.0, [urn:nbn:fi:lb-2020112937](http://urn.fi/urn:nbn:fi:lb-2020112937).
- **Lauseopin arkiston murrekorpus (LA-murre)**, the Finnish Dialect Syntax Archive, University of
  Turku and Kotus, via Kielipankki – The Language Bank of Finland, CC BY-ND 4.0,
  [urn:nbn:fi:lb-2020112935](http://urn.fi/urn:nbn:fi:lb-2020112935). The corpus's licence does not
  allow sharing modified copies, so this variant was first kept private. In October 2026
  Kielipankki (FIN-CLARIN), which distributes LA-murre, assessed that the model weights most likely
  contain no copyright-protected part of the corpus, so publishing them needs no separate
  permission: the model writes standard Finnish and cannot reproduce LA-murre's transcripts or its
  database content. The standard-Finnish rewrites it was trained on are not published.
- **FLEURS**, Google, CC BY 4.0. **VoxPopuli**, Meta, CC0.
- Computing on LUMI. We acknowledge CSC – IT Center for Science, Finland for awarding this project
  access to the LUMI supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CSC (Finland)
  and the LUMI consortium through CSC – IT Center for Science, Finland.
- Made in the [GAIK project](https://gaik.ai/), co-funded by the European Union (ERDF).
