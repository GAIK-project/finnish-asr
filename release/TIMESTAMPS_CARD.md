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
- timestamps
datasets:
- GAIK-project/skn-finnish-dialects-asr
- google/fleurs
- facebook/voxpopuli
model-index:
- name: whisper-large-v3-fi-dialects (revision timestamps)
  results:
  - task:
      type: automatic-speech-recognition
    dataset:
      name: SKN dialect test (10 held-out municipalities)
      type: GAIK-project/skn-finnish-dialects-asr
      split: test
    metrics:
    - type: wer
      value: 21.4
    - type: cer
      value: 8.7
  - task:
      type: automatic-speech-recognition
    dataset:
      name: FLEURS Finnish
      type: google/fleurs
      config: fi_fi
      split: test
    metrics:
    - type: wer
      value: 7.4
  - task:
      type: automatic-speech-recognition
    dataset:
      name: VoxPopuli Finnish
      type: facebook/voxpopuli
      config: fi
      split: test
    metrics:
    - type: wer
      value: 11.2
---

# Whisper large-v3 for Finnish dialects, with timestamps (revision `timestamps`)

This branch holds a **variant** of [GAIK-project/whisper-large-v3-fi-dialects](https://huggingface.co/GAIK-project/whisper-large-v3-fi-dialects):
the same LoRA fine-tune of [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)
for Finnish dialect speech, trained so that it also writes Whisper's **timestamps**. It hears dialect
and writes standard Finnish with punctuation, and it can transcribe long recordings in one go with
sentence-level times.

The main branch (tag `v1.0`) stays the recommended model for plain transcription. Use this one when
you need **sentence timestamps**, or **long audio without cutting it yourself**, and then always with
the fallback settings below.

Trained at Haaga-Helia University of Applied Sciences in the [GAIK project](https://gaik.ai/)
(Generative AI-enhanced Knowledge Management in Business), a research project led by Haaga-Helia
with the University of Helsinki and Tampere University, co-funded by the European Union (ERDF).

🌐 [gaik.ai](https://gaik.ai/) · 📊 [code, method and results on GitHub: GAIK-project/finnish-asr](https://github.com/GAIK-project/finnish-asr#the-timestamp-variant) ·
🗂️ [training dataset](https://huggingface.co/datasets/GAIK-project/skn-finnish-dialects-asr)

## Results

| Test | Whisper large-v3 | Main model (`v1.0`) | This variant |
|---|--:|--:|--:|
| SKN dialect interviews, 10 municipalities not in training, WER % | 40.4 | **21.1** | 21.4 |
| FLEURS Finnish, read sentences, WER % | 7.4 | 7.6 | 7.4 |
| VoxPopuli Finnish, parliament speech, WER % | 12.7 | **10.8** | 11.2 |
| LA-murre dialect interviews, meaning-error points per 100 words (blind LLM judge, judged together) | 24.2 | 13.7 | 13.7 |
| SKN sentence starts with a timestamp within 1 s (600 test segments) | 60.8 % | 0.8 % | **57.8 %** |
| Ten 4-minute dialect interviews in one go, with the fallback settings, WER % | 37.5 | 19.8 | **18.1** |
| Sentence starts within 1 s in those interviews, with the fallback settings | 62.6 % | 18.9 % | 49.0 % |
| A 180 s Finnish dental webinar in one go with timestamps, WER % | 14.7 | 62.7 (repeats itself) | 16.7 |
| The same webinar in 15–25 s pieces, WER % | 15.7 | **13.9** | 15.2 |

WER after lower-casing and removing punctuation. Timestamps are compared with the sentence times of
SKN's own annotation; when the variant finds a sentence start, the median distance is 0.04 s. On the
webinar, against the reference a listener checked at its 42 disputed spans, the variant gives 16.2 %
in one go and 14.5 % in pieces (main model 14.0 % in pieces).

The variant did not replace the main model: VoxPopuli got 0.4 points worse and the webinar in pieces
0.5–1.3 points worse, which broke the rule agreed before the run. On the dialect tests the two are
level: a blind LLM judge on LA-murre gave the variant fewer meaning-error points than the main model
in 98 segments and more in 100.

## Use it

You need Python with **transformers 5 or newer** and preferably a GPU:

```bash
pip install "transformers>=5" torch soundfile
```

The audio must be **16 kHz mono**; it can be of any length. The fallback settings are OpenAI's recipe
against long-form repetition: when a 30 s window's text compresses too well (a loop) or its mean
log-probability is too low, that window is decoded again at the next temperature.

```python
import soundfile as sf
from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition",
    model="GAIK-project/whisper-large-v3-fi-dialects",
    revision="timestamps",
    dtype="float16",
    device="cuda",
)  # without a GPU: device="cpu" and no dtype (slow)

audio, sr = sf.read("interview.wav", dtype="float32")  # 16 kHz mono; resample first if it is not
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

Without the fallback settings, this variant looped in 3 of the 10 interviews (25.2 % WER). For plain
transcription without times, the main model with the quiet-point pieces of
[its card](https://huggingface.co/GAIK-project/whisper-large-v3-fi-dialects) is slightly better.

## Where it is weaker

- **Rare names and technical terms** in standard-Finnish audio, like the main model: for standard
  Finnish full of brand names, plain Whisper large-v3 is the safer choice.
- **Timestamps are sentence-level and approximate**: about half of the sentence starts in long
  interviews fall within 1 s; they are not word timings.
- The dialect data are 1960s–70s interviews with elderly speakers; today's regional speech was not
  tested. Output is standard Finnish, not a phonetic dialect transcript.
- One training run and one seed.

## How it was trained

The main model's recipe, with one change:

1. 🎙️ **Data.** SKN dialect interviews ([Suomen kielen näytteitä](https://www.kielipankki.fi/corpora/skn/))
   cut into 2–25 s segments and split by municipality (15 042 training segments, 62 h, from 38
   municipalities), plus [FLEURS](https://huggingface.co/datasets/google/fleurs) `fi_fi` train
   (2 704) and [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) `fi` train (7 504):
   25 250 segments. The targets are SKN's human standard-Finnish transcripts with punctuation and
   capitals added by an LLM (Claude Opus 5.5, text only, words unchanged).
2. ⏱️ **Timestamps.** In each epoch half of the SKN segments, drawn again every epoch, are trained
   with one timestamp pair per SKN sentence (`<|0.00|> text<|1.84|>`), the times taken from SKN's
   sentence annotation; 15 040 of the 15 042 training segments had usable times. The other items are
   trained without timestamps, as for the main model.
3. ⚙️ **Fine-tuning.** LoRA rank 64, alpha 128, dropout 0.05, on the attention `q_proj` and `v_proj`;
   AdamW, learning rate 1e-4 with 5 % warm-up and linear decay, batch 8, bf16. Three epochs, keeping
   the one with the lowest validation loss (epoch 2: 0.2158; not comparable with the main model's
   0.206, because half of the validation segments carry timestamp tokens). The adapter is merged
   into the weights published here.
4. 🖥️ **Compute.** 3 h 6 min of training with per-epoch validation on one AMD MI250X GPU on the
   [LUMI](https://www.lumi-supercomputer.eu/) supercomputer.

## How it was measured

- 📏 **Word error rate** on the SKN dialect test (600 segments from 10 municipalities not in
  training), FLEURS and VoxPopuli test, and a 180 s webinar clip.
- ⏱️ **Timestamps** against SKN's sentence times: on the 600 test segments decoded with
  `return_timestamps=True`, and on ten windows of about 4 minutes, one from each test municipality,
  decoded in one go with and without the fallback settings and in quiet-point pieces. A decode
  counts as a loop when one 5-word sequence occurs four times or more and more often than in the
  reference.
- ⚖️ **A blind LLM judge** (Claude Opus 5.5) on 577 segments of a second dialect corpus,
  [LA-murre](https://www.kielipankki.fi/corpora/la-murre/) (CC BY-ND 4.0, not used in training), with
  plain Whisper large-v3, the main model and this variant judged together, unlabelled and in shuffled
  order, counting only meaning errors.

Code, method and all results: [GAIK-project/finnish-asr](https://github.com/GAIK-project/finnish-asr#timestamps-and-long-form-audio).

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
