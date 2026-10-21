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
- name: whisper-large-v3-fi-dialects (revision parl)
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
      value: 7.6
  - task:
      type: automatic-speech-recognition
    dataset:
      name: VoxPopuli Finnish
      type: facebook/voxpopuli
      config: fi
      split: test
    metrics:
    - type: wer
      value: 11.1
---

# Whisper large-v3, fine-tuned for Finnish dialects

Speech-to-text for **Finnish dialect speech**: it hears dialect and writes standard Finnish with punctuation. A LoRA
fine-tune of [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) on dialect interviews, read
sentences and 72 hours of Finnish parliament speech, so it also writes names, acronyms and terms more often right.

This is the main model since 21 Oct 2026: on the SKN dialect test 19.4 % of words wrong (the first published model,
tag `v1.0`, 21.1 %; Whisper large-v3 40.4 %), level on read speech, and better on names and terms. The model published on 13 Oct 2026 stays available as `revision="v1.0"`.

Trained at Haaga-Helia University of Applied Sciences in the [GAIK project](https://gaik.ai/)
(Generative AI-enhanced Knowledge Management in Business), a research project led by Haaga-Helia
with the University of Helsinki and Tampere University, co-funded by the European Union (ERDF).

**[GAIK project](https://gaik.ai/)** · [Haaga-Helia University of Applied Sciences](https://www.haaga-helia.fi/en) · developed by Akseli Leskinen ([laguagu](https://github.com/laguagu))

📖 The story, with audio: [suomeksi](https://gaik-project.github.io/finnish-asr/story/story.fi.html) · [in English](https://gaik-project.github.io/finnish-asr/story/story.en.html) · 🔊 [listen and compare models](https://gaik-project.github.io/finnish-asr/listen.html) · 🗂️ [training data (SKN part)](https://huggingface.co/datasets/GAIK-project/skn-finnish-dialects-asr)

📊 [code, method and all results: GAIK-project/finnish-asr](https://github.com/GAIK-project/finnish-asr) · 🧰 [gaik-toolkit](https://github.com/GAIK-project/gaik-toolkit) · 🌐 [gaik.ai](https://gaik.ai/)

![Dialect test word error rate of every model: this model 19.4 %, the first model 21.1 %, Gemini 3.5 Flash 34.7 %, Whisper large-v3 40.4 %](dialect-ranking.svg)

## Results

WER % after lower-casing and removing punctuation; the same decoding (transformers pipeline, greedy) for every model.

| Test | Whisper large-v3 | `v1.0` | `la-murre` | This model |
|---|--:|--:|--:|--:|
| SKN dialect interviews, 10 municipalities in no model's training (600 segments) | 40.4 | 21.1 | 19.7 | **19.4** |
| LA-murre dialect interviews, 10 held-out localities, against dialect spelling (583) | 62.5 | 58.1 | 57.4 | **57.4** |
| LA-murre dialect interviews, 10 held-out localities: meaning-error points per 100 words (blind LLM judge, round 5, 583 segments) | 24.4 | 14.0 | 12.1 | **11.7** |
| FLEURS Finnish, read sentences (918) | **7.4** | 7.6 | 7.8 | 7.6 |
| VoxPopuli Finnish, parliament speech (478) | 12.7 | **10.8** | 11.4 | 11.1 |
| Parliament, speakers held out of training (Aalto corpus 2016-test, 962) ¹ | 9.3 | 11.4 | 9.2 | **8.5** |
| Parliament segments with names, acronyms or terms (1 525) ¹ | 14.5 | 13.7 | 13.7 | **12.4** |
| Of those, acronyms, English words and person names written exactly | 66 % | 72 % | 71 % | **75 %** |

¹ This model was trained on other speakers and sessions of the same corpus, so these two tests are in its domain.
Paired over segments against `la-murre`: names-and-terms set −1.35 points (95 % interval −2.04 to −0.80), SKN −0.35
(−0.73 to +0.04); FLEURS and LA-murre level. VoxPopuli moves by more than a point between two seeds of the same
recipe, so it cannot separate these models. The names and terms were tagged by an LLM (Claude Opus 5.5) and the
tags checked on a sample; the headline counts the three classes that check found reliable.

## Use it

```python
from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition",
    model="GAIK-project/whisper-large-v3-fi-dialects",
    dtype="float16",
    device="cuda",
)
print(asr("piece.wav", generate_kwargs={"language": "fi", "task": "transcribe"})["text"])
```

Cut long audio into 15–25 s pieces at quiet points; the
[`v1.0` card](https://huggingface.co/GAIK-project/whisper-large-v3-fi-dialects/blob/v1.0/README.md) has code that does it.

## Variants

- **`main`: this model**; `revision="parl"` pins it.
- **`v1.0`: the first published model** (13 Oct 2026), SKN + FLEURS + VoxPopuli: 21.1 / 7.6 / 10.8 %. Its card has
  the long-form, timestamp and judge measurements.
- **`la-murre`**: `v1.0` + 56 h of LA-murre: 19.7 / 7.8 / 11.4 %.
- **`timestamps`**: `v1.0`'s recipe with Whisper's timestamp tokens, for sentence times and long audio in one go.
- **`all-skn`**: `v1.0`'s recipe on all 50 SKN municipalities.
- **`glossary`**: this model's recipe trained to take a word list (`Sanasto: …`) as a prompt; with the list it writes
  93 % of names and terms right (this model 75 %).

## How it was trained

1. 🎙️ **Data.** The `la-murre` variant's data: 15 042 SKN dialect segments, 15 529 LA-murre segments (56.5 h),
   FLEURS `fi_fi` train (2 704) and VoxPopuli `fi` train (7 504). Added: 37 740 segments (71.6 h) of the
   [Aalto Finnish Parliament ASR Corpus](https://www.kielipankki.fi/corpora/fi-parliament-asr/) 2015–2020 part 1,
   a seeded draw of 100 h leaving out the 2016-test speakers and the 2020 test sessions; segments with the corpus's
   `<UNK>` marker (an unknown word) were dropped.
2. ✍️ **Training text.** The parliament transcripts are lower case without punctuation. An LLM (Claude Haiku 5.5, and
   Sonnet 5.5 where Haiku changed a word) added punctuation and capitals with every word unchanged, checked by
   script; 362 segments neither could punctuate without changing a word were dropped. The other texts are as in
   the `la-murre` variant.
3. ⚙️ **Fine-tuning.** LoRA rank 64 on `q_proj` and `v_proj`, learning rate 1e-4, batch 8, two epochs (the best epoch
   was the second in every earlier run); validation loss 0.604 before, 0.232 after. Merged into the weights here.
4. 🖥️ **Compute.** 6 h 29 min on one AMD MI250X GPU on the [LUMI](https://www.lumi-supercomputer.eu/) supercomputer.

## Where it is weaker

- VoxPopuli: 0.3 points more word errors than `v1.0` (within the run-to-run noise of that set).
- Both dialect corpora are 1960s–70s interviews with elderly speakers; the parliament speech is formal. Everyday
  conversation, slang and Finnish-English mixing were not tested. One training run and one seed.

## Credits and licences

- Developed by Akseli Leskinen ([laguagu](https://github.com/laguagu)), Haaga-Helia University of Applied Sciences, in the [GAIK project](https://gaik.ai/).
- Weights: Apache 2.0, as the base model [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3).
- **Suomen kielen näytteitä (SKN)**, Kotus, via Kielipankki – The Language Bank of Finland, CC BY 4.0,
  [urn:nbn:fi:lb-2020112937](http://urn.fi/urn:nbn:fi:lb-2020112937).
- **Lauseopin arkiston murrekorpus (LA-murre)**, University of Turku and Kotus, via Kielipankki, CC BY-ND 4.0,
  [urn:nbn:fi:lb-2020112935](http://urn.fi/urn:nbn:fi:lb-2020112935). In October 2026 Kielipankki (FIN-CLARIN) assessed
  that model weights trained on it most likely contain no copyright-protected part of the corpus. Its
  standard-Finnish rewrites are not published.
- **Aalto Finnish Parliament ASR Corpus 2008–2020, version 2**, Aalto University speech recognition group, from the
  plenary sessions of the Parliament of Finland, via Kielipankki, CLARIN PUB +BY +PRIV +ND,
  [urn:nbn:fi:lb-2022052002](http://urn.fi/urn:nbn:fi:lb-2022052002). The model writes standard Finnish from the speech
  it is given and cannot reproduce the corpus. Neither the corpus nor its punctuated texts are published here, and
  nothing here is endorsed by the Parliament of Finland.
- **FLEURS**, Google, CC BY 4.0. **VoxPopuli**, Meta, CC0.
- Computing on LUMI. We acknowledge CSC – IT Center for Science, Finland for awarding this project
  access to the LUMI supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CSC (Finland)
  and the LUMI consortium through CSC – IT Center for Science, Finland.
- Made in the [GAIK project](https://gaik.ai/), co-funded by the European Union (ERDF).
