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
- name: whisper-large-v3-fi-dialects (revision all-skn)
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
      value: 7.5
  - task:
      type: automatic-speech-recognition
    dataset:
      name: VoxPopuli Finnish
      type: facebook/voxpopuli
      config: fi
      split: test
    metrics:
    - type: wer
      value: 11.9
---

# Whisper large-v3 for Finnish dialects, trained on all of SKN (revision `all-skn`)

This branch holds a **variant** of [GAIK-project/whisper-large-v3-fi-dialects](https://huggingface.co/GAIK-project/whisper-large-v3-fi-dialects):
the same LoRA fine-tune of [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3),
trained on **all 50 SKN municipalities** (81 h of dialect speech) instead of 38 (62 h). It hears
dialect and writes standard Finnish with punctuation.

The main branch (tag `v1.0`) stays the recommended model. This variant makes fewer meaning errors on
dialect speech it never heard, and slightly more word errors on parliament speech. Use it when your
audio is mostly dialect.

Trained at Haaga-Helia University of Applied Sciences in the [GAIK project](https://gaik.ai/)
(Generative AI-enhanced Knowledge Management in Business), a research project led by Haaga-Helia
with the University of Helsinki and Tampere University, co-funded by the European Union (ERDF).

**[GAIK project](https://gaik.ai/)** · [Haaga-Helia University of Applied Sciences](https://www.haaga-helia.fi/en) · developed by Akseli Leskinen ([laguagu](https://github.com/laguagu))

📖 The story, with audio: [suomeksi](https://gaik-project.github.io/finnish-asr/story/story.fi.html) · [in English](https://gaik-project.github.io/finnish-asr/story/story.en.html) · 🔊 [listen and compare models](https://gaik-project.github.io/finnish-asr/listen.html) · 🗂️ [training data](https://huggingface.co/datasets/GAIK-project/skn-finnish-dialects-asr)

📊 [code, method and all results: GAIK-project/finnish-asr](https://github.com/GAIK-project/finnish-asr#dialect-data-and-next-steps) · 🧰 [gaik-toolkit](https://github.com/GAIK-project/gaik-toolkit) · 💻 [GAIK on GitHub](https://github.com/GAIK-project) · 🌐 [gaik.ai](https://gaik.ai/)

## Results

| Test | Whisper large-v3 | Main model (`v1.0`) | This variant |
|---|--:|--:|--:|
| LA-murre dialect interviews, 10 localities in no model's training: meaning-error points per 100 words (blind LLM judge, 583 segments) | 25.0 | 14.2 | **13.1** |
| FLEURS Finnish, read sentences, WER % | 7.4 | 7.6 | 7.5 |
| VoxPopuli Finnish, parliament speech, WER % | 12.7 | **10.8** | 11.9 |
| Ten 180 s dental webinars in 15–25 s pieces: meaning-error points per 100 words (Claude / Gemini judge) | 3.6 / 6.4 | 5.1 / 10.0 | 4.8 / 9.6 |

The SKN test cannot score this variant: its 10 municipalities are in its training data. On LA-murre
the judge (Claude Opus 5.5) saw six transcripts of each segment together, unlabelled and shuffled; this
variant got fewer points than the main model in 177 segments and more in 114 (sign test p = 0.0003).
Compare points only within one judging round; the main model's card quotes an earlier round.

The variant did not replace the main model: the rule set before the run allowed FLEURS and VoxPopuli
to get at most 0.5 points worse, and VoxPopuli got 1.1 points worse.

## Use it

Exactly as the [main model](https://huggingface.co/GAIK-project/whisper-large-v3-fi-dialects), with
`revision="all-skn"`, and long audio cut into 15–25 s pieces at quiet points as its card shows:

```python
from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition",
    model="GAIK-project/whisper-large-v3-fi-dialects",
    revision="all-skn",
    dtype="float16",
    device="cuda",
)
print(asr("piece.wav", generate_kwargs={"language": "fi", "task": "transcribe"})["text"])
```

## How it was trained

The main model's recipe, with more SKN:

1. 🎙️ **Data.** SKN dialect interviews ([Suomen kielen näytteitä](https://www.kielipankki.fi/corpora/skn/))
   cut into 2–25 s segments: the main model's 15 042 training segments plus every segment of the 12
   validation and test municipalities (4 601 segments, 19.5 h), so 19 643 segments from all 50
   municipalities, plus [FLEURS](https://huggingface.co/datasets/google/fleurs) `fi_fi` train (2 704)
   and [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) `fi` train (7 504). The targets
   are SKN's human standard-Finnish transcripts with punctuation and capitals added by an LLM (Claude
   Opus 5.5, text only, words unchanged).
2. ⚙️ **Fine-tuning.** LoRA rank 64 on the attention `q_proj` and `v_proj`, learning rate 1e-4,
   batch 8, three epochs; validation on FLEURS and VoxPopuli only (1 102 items), since no SKN
   municipality is left out. The epoch with the lowest validation loss is kept (epoch 2: 0.182). The
   adapter is merged into the weights published here.
3. 🖥️ **Compute.** 3 h 39 min on one AMD MI250X GPU on the [LUMI](https://www.lumi-supercomputer.eu/) supercomputer.

## Where it is weaker

- Parliament speech: 1.1 points more word errors than the main model on VoxPopuli.
- **Rare names and technical terms** in standard-Finnish audio, like the main model: for standard
  Finnish full of brand names, plain Whisper large-v3 or a commercial service is the safer choice.
- The dialect data are 1960s–70s interviews with elderly speakers; today's regional speech was not
  tested. One training run and one seed.

## Credits and licences

- Developed by Akseli Leskinen ([laguagu](https://github.com/laguagu)), Haaga-Helia University of Applied Sciences, in the [GAIK project](https://gaik.ai/).
- Weights: Apache 2.0, as the base model [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3).
- **Suomen kielen näytteitä (SKN)**, Kotimaisten kielten keskus (Kotus), via Kielipankki – The
  Language Bank of Finland, CC BY 4.0, [urn:nbn:fi:lb-2020112937](http://urn.fi/urn:nbn:fi:lb-2020112937).
- **FLEURS**, Google, CC BY 4.0. **VoxPopuli**, Meta, CC0.
- Used only to test this model, none of it in its training data: **Lauseopin arkiston murrekorpus
  (LA-murre)**, University of Turku and Kotus, via Kielipankki, CC BY-ND 4.0,
  [urn:nbn:fi:lb-2020112935](http://urn.fi/urn:nbn:fi:lb-2020112935).
- Computing on LUMI. We acknowledge CSC – IT Center for Science, Finland for awarding this project
  access to the LUMI supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CSC (Finland)
  and the LUMI consortium through CSC – IT Center for Science, Finland.
- Made in the [GAIK project](https://gaik.ai/), co-funded by the European Union (ERDF).
