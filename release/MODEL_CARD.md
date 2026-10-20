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
- name: whisper-large-v3-fi-dialects
  results:
  - task:
      type: automatic-speech-recognition
    dataset:
      name: SKN dialect test (10 held-out municipalities)
      type: GAIK-project/skn-finnish-dialects-asr
      split: test
    metrics:
    - type: wer
      value: 21.1
    - type: cer
      value: 8.6
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
      value: 10.8
---

# Whisper large-v3, fine-tuned for Finnish dialects

Speech-to-text for **Finnish dialect speech**: it hears dialect and writes standard Finnish with
punctuation. A LoRA fine-tune of [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3),
trained at Haaga-Helia University of Applied Sciences in the [GAIK project](https://gaik.ai/)
(Generative AI-enhanced Knowledge Management in Business), a research project led by Haaga-Helia
with the University of Helsinki and Tampere University, co-funded by the European Union (ERDF).

**[GAIK project](https://gaik.ai/)** · [Haaga-Helia University of Applied Sciences](https://www.haaga-helia.fi/en) · developed by Akseli Leskinen ([laguagu](https://github.com/laguagu))

📖 The story, with audio: [suomeksi](https://gaik-project.github.io/finnish-asr/story/story.fi.html) · [in English](https://gaik-project.github.io/finnish-asr/story/story.en.html) · 🔊 [listen and compare models](https://gaik-project.github.io/finnish-asr/listen.html) · 🗂️ [training data](https://huggingface.co/datasets/GAIK-project/skn-finnish-dialects-asr) · ⏱️ [timestamp variant](https://huggingface.co/GAIK-project/whisper-large-v3-fi-dialects/blob/timestamps/README.md)

📊 [code, method and all results: GAIK-project/finnish-asr](https://github.com/GAIK-project/finnish-asr) · 🧰 [gaik-toolkit](https://github.com/GAIK-project/gaik-toolkit) · 💻 [GAIK on GitHub](https://github.com/GAIK-project) · 🌐 [gaik.ai](https://gaik.ai/)

| Test | Whisper large-v3 | This model | Best paid service |
|---|--:|--:|--:|
| 🗣️ Dialect interviews (SKN, 10 municipalities not in training), WER % | 40.4 | **21.1** | 34.7 (Gemini 3.5 Flash) |
| 🗣️ Another dialect corpus (LA-murre), meaning-error points per 100 words ¹ | 25.0 | **14.2** | 27.2 (Gemini 3.5 Flash) |
| 📖 Read sentences (FLEURS Finnish), WER % | 7.4 | 7.6 | **3.3** (ElevenLabs Scribe v2) |
| 🏛️ Parliament speech (VoxPopuli Finnish), WER % | 12.7 | 10.8 | **9.1** (ElevenLabs Scribe v2) |

![Dialect test word error rate of every model: this model 21.1 %, Gemini 3.5 Flash 34.7 %, Whisper large-v3 40.4 %, ElevenLabs 44.9 %, gpt-4o-transcribe 62.5 %](dialect-ranking.svg)

**Use it for dialect speech**, the way people talk outside the big cities and older speakers above all:
it makes about half the errors of Whisper large-v3 and fewer than every paid service tested (Gemini
3.5 Flash, Google Chirp 3, ElevenLabs Scribe v2, OpenAI gpt-4o-transcribe). On standard Finnish it stays at
Whisper's level, and the paid services are better.

WER after lower-casing and removing punctuation. ¹ LA-murre's reference is written in dialect, so a
blind LLM judge (Claude Opus 5.5) counted only errors that change the meaning (critical 10, major 5,
minor 1), all six models of a segment judged together; GPT-6.1-sol and gpt-oss-120b as judges give the
same order. Gemini was asked for standard Finnish on the dialect tests. 9.9 h of test audio in all;
every model and result: [GAIK-project/finnish-asr](https://github.com/GAIK-project/finnish-asr) ·
🔊 [listen and compare](https://gaik-project.github.io/finnish-asr/listen.html).

## Variants

- **`main`, tag `v1.0`: this model**, the one to use by default. `revision="v1.0"` pins it.
- **`timestamps`: the same recipe trained with Whisper's timestamp tokens** on half of the SKN
  segments. It writes usable sentence times (58 % of SKN sentence starts within 1 s; this model 1 %)
  and transcribes long audio in one go with OpenAI's temperature fallback (18.1 % WER on ten 4-minute
  dialect interviews, no loops; this model 19.8 %), at about the same dialect accuracy (SKN 21.4 %,
  same LA-murre judge score) but slightly worse on VoxPopuli (11.2 %) and on the webinar in pieces
  (15.2 %). Use it when you need sentence timestamps or long-form decoding without cutting the audio
  yourself, always with its fallback settings: load it with `revision="timestamps"`; its card is on
  [that branch](https://huggingface.co/GAIK-project/whisper-large-v3-fi-dialects/blob/timestamps/README.md).
- **`all-skn`: the same recipe trained on all 50 SKN municipalities** (81 h of dialect speech
  instead of 62). Fewer meaning errors on LA-murre (13.1 against this model's 14.2 in the same judging
  round), FLEURS level (7.5 %), VoxPopuli worse (11.9 %). Load it with `revision="all-skn"`; its
  card is on [that branch](https://huggingface.co/GAIK-project/whisper-large-v3-fi-dialects/blob/all-skn/README.md).
- **`la-murre`: the same recipe with 56 h more dialect interviews** from the Finnish Dialect Syntax
  Archive (LA-murre, 124 localities), their dialect spelling rewritten in standard Finnish by an LLM.
  The most accurate on dialects: SKN 19.7 % WER against this model's 21.1 %, and the fewest meaning
  errors on LA-murre's held-out localities (11.6 against 14.2 and `all-skn`'s 13.1 in the same judging
  round; it knows the corpus's other recordings). Slightly worse on FLEURS (7.8 %) and VoxPopuli
  (11.4 %). For audio that is mostly dialect: load it with `revision="la-murre"`; its card is on
  [that branch](https://huggingface.co/GAIK-project/whisper-large-v3-fi-dialects/blob/la-murre/README.md).

## Use it

You need Python with **transformers 5 or newer** (older versions cannot read this repository's
tokenizer files) and preferably a GPU:

```bash
pip install "transformers>=5" torch soundfile
```

The audio must be **16 kHz mono**. Whisper listens to at most 30 seconds at a time, so cut longer
audio into 15–25 s pieces at its quietest moments and transcribe each piece. The code below does
that and prints each piece with the time it starts at:

```python
import numpy as np
import soundfile as sf
from transformers import pipeline

SR = 16000
asr = pipeline(
    "automatic-speech-recognition", model="GAIK-project/whisper-large-v3-fi-dialects", dtype="float16", device="cuda"
)  # without a GPU: device="cpu" and no dtype (slow)


def quiet_chunks(audio, min_s=15, max_s=25, win_s=0.1):
    """Split at the quietest 100 ms between min_s and max_s into each piece."""
    win, start, pieces = int(win_s * SR), 0, []
    while len(audio) - start > max_s * SR:
        lo, hi = start + int(min_s * SR), start + int(max_s * SR)
        frames = (hi - lo) // win
        energy = (audio[lo : lo + frames * win].reshape(frames, win) ** 2).mean(axis=1)
        cut = lo + int(np.argmin(energy)) * win + win // 2
        pieces.append(audio[start:cut])
        start = cut
    return pieces + [audio[start:]]


audio, sr = sf.read("interview.wav", dtype="float32")  # 16 kHz mono; resample first if it is not
pieces = quiet_chunks(audio)  # a clip of 25 s or less stays one piece
out = asr(
    [{"raw": p, "sampling_rate": SR} for p in pieces],
    batch_size=8,
    generate_kwargs={"language": "fi", "task": "transcribe"},
)
start = 0.0
for piece, o in zip(pieces, out):
    print(f"[{start:6.1f} s] {o['text'].strip()}")
    start += len(piece) / SR
```

### Settings to avoid

Measured on a 180 s Finnish webinar (402 words), with the model downloaded from this repository,
against the University of Helsinki team's reference and against the version a listener checked at
its 42 disputed spans:

| Decoding | WER % | WER %, checked reference | |
|---|--:|--:|---|
| `quiet_chunks` above | **13.7** | **13.5** | recommended |
| pipeline `chunk_length_s=20` | 14.7 | 13.5 | works |
| pipeline `chunk_length_s=30` | 52.5 | 51.5 | **repeats one phrase over and over**: do not use |
| pipeline `chunk_length_s=30`, `no_repeat_ngram_size=6` | 14.4 | 14.2 | the guard stops the repeating |

Source: the six decodings in [`results/clip/decoding_check.json`](https://github.com/GAIK-project/finnish-asr/blob/main/results/clip/decoding_check.json),
scored by `score_asr.py --decoding` into [`decoding_scores.json`](https://github.com/GAIK-project/finnish-asr/blob/main/results/clip/decoding_scores.json).

Do not use `return_timestamps=True` with this model either. In that mode Whisper keeps its place in
long audio by writing time marks as it goes. This model was trained on text without those time marks,
so its times are wrong and on long audio it can lose its place and repeat itself; the `timestamps`
revision above was trained with them. Measured on SKN's dialect test
(600 segments, 791 sentence starts), only 0.8 % of the sentence starts had a time mark within 0.5 s
(plain Whisper large-v3: 53.5 %), and on ten 4-minute dialect recordings Whisper's own long-form mode
gave 21.5 % WER with one recording repeating a phrase, against 17.3 % for the pieces above
([details](https://github.com/GAIK-project/finnish-asr#timestamps-and-long-form-audio)). If you need
times, use the piece start times printed above, which are exact to the cut points, or the
`timestamps` revision.

The 13.9 % in the first table is the same model before upload, decoded one piece at a time like the
base model's row; this repository's copy with the code above (`batch_size=8`) gives 13.7 %, one word
of 402 apart.

## Where it is weaker

- **Rare names and technical terms** in standard-Finnish audio: on the webinar, two blind LLM
  judges found more name and term errors than for plain Whisper large-v3 ("Nobelbayer" for Nobel
  Biocare), although its WER was lower. For standard Finnish full of brand names, plain Whisper
  large-v3 is the safer choice.
- The dialect data are 1960s–70s interviews with elderly speakers; today's regional speech
  was not tested.
- Output is standard Finnish, not a phonetic dialect transcript.

## How it was trained

![Two panels, lower is better. SKN dialect test WER: Finnish-NLP 56.6, Whisper large-v3 40.4, round 1 22.2, this model 21.1, the la-murre revision 19.7. LA-murre meaning-error points per 100 words, blind judge: Whisper large-v3 25.9, round 1 13.8, this model 13.6, the la-murre revision 11.4.](dialect-finetune.svg)

"Ours, round 2 (published)" in the chart is this model; "Ours, round 2 + LA-murre" is the
`la-murre` revision.

1. 🎙️ **Dialect speech.** Interviews from [Suomen kielen näytteitä](https://www.kielipankki.fi/corpora/skn/)
   (SKN) were cut into 2–25 s segments and split **by municipality**: 38 for training (15 042
   segments, 62 h), 2 for validation, 10 for the test, so no test interviewee is heard in training. The
   segments are published as [GAIK-project/skn-finnish-dialects-asr](https://huggingface.co/datasets/GAIK-project/skn-finnish-dialects-asr).
2. 📚 **Standard Finnish beside it.** [FLEURS](https://huggingface.co/datasets/google/fleurs) `fi_fi`
   train (2 704 read sentences) and [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli)
   `fi` train (7 504 parliament segments) keep the model's standard Finnish in place. 25 250
   training segments in all.
3. ✍️ **Target text.** The target is SKN's human transcript in standard Finnish word forms, so the
   model hears dialect and writes standard Finnish. That text has no punctuation, so an LLM
   (Claude Opus 5.5, text only) added punctuation and capitals. An answer was kept only if its
   words were identical to the human transcript (15 024 of 15 042); the rest kept the plain text.
4. ⚙️ **Fine-tuning.** LoRA rank 64, alpha 128, dropout 0.05, on the attention `q_proj` and `v_proj`
   (about 2 % of the weights); AdamW, learning rate 1e-4 with 5 % warm-up and linear decay,
   batch 8, bf16. Three epochs, keeping the one with the lowest validation loss (epoch 2: 0.206,
   base model 0.538). The adapter is merged into the weights published here.
5. 🖥️ **Compute.** 3 h 8 min of training with per-epoch validation on one AMD MI250X GPU on the
   [LUMI](https://www.lumi-supercomputer.eu/) supercomputer.

## How it was measured

- 📏 **Word error rate** on test sets with a standard-Finnish reference: the SKN test (600
  segments from 10 municipalities not in training), FLEURS and VoxPopuli test, and a webinar clip.
- ⚖️ **A blind LLM judge** on a second dialect corpus, the Finnish Dialect Corpus of the Syntax Archive
  ([LA-murre](https://www.kielipankki.fi/corpora/la-murre/), CC BY-ND 4.0): 583 segments from 10
  localities outside SKN, not used for training this model. Its reference is in dialect spelling,
  so WER would charge a model for writing standard Finnish. Claude Opus 5.5 instead read the
  reference and every model's transcript of a segment, unlabelled and in shuffled order, and
  counted meaning errors: critical 10, major 5, minor 1 points, summed and divided by the
  reference words, per 100 words. Dialect against standard forms, punctuation and fillers are
  not errors.

One training run and one seed. The main judge is one model, Claude Opus 5.5. Three open models on
CSC's Aitta service, given the same blind task, agree: gpt-oss-120b on all 583 segments prefers the
same model as Claude in 81 % of the pairs where both see a difference, and Gemma 4 31B and Qwen3.6 27B,
on a sample, in 92–93 %. All four judges rank the `la-murre` revision first, this model and
round 1 about level, and plain Whisper large-v3 clearly last; their point scales differ, so compare
models within one judge. Code, method, all results and a comparison with 20 other set-ups:
[GAIK-project/finnish-asr](https://github.com/GAIK-project/finnish-asr); summaries in gaik-toolkit:
[finnish_dialect_asr](https://github.com/GAIK-project/gaik-toolkit/tree/main/evaluation_layer/eval_methods/finnish_dialect_asr) and
[transcription_eval](https://github.com/GAIK-project/gaik-toolkit/tree/main/evaluation_layer/eval_methods/transcription_eval#october-2026-current-models-and-an-llm-judge).

## Credits and licences

- Developed by Akseli Leskinen ([laguagu](https://github.com/laguagu)), Haaga-Helia University of Applied Sciences, in the [GAIK project](https://gaik.ai/).
- Weights: Apache 2.0, as the base model [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3).
- **Suomen kielen näytteitä (SKN)**, Kotimaisten kielten keskus (Kotus), via Kielipankki – The
  Language Bank of Finland, CC BY 4.0, [urn:nbn:fi:lb-2020112937](http://urn.fi/urn:nbn:fi:lb-2020112937).
  Download: [kielipankki.fi/download/SKN](https://www.kielipankki.fi/download/SKN/).
- **FLEURS**, Google, CC BY 4.0. **VoxPopuli**, Meta, CC0.
- Used only to test this model, none of it in its training data: **Lauseopin arkiston murrekorpus
  (LA-murre)**, University of Turku and Kotus, via Kielipankki, CC BY-ND 4.0,
  [urn:nbn:fi:lb-2020112935](http://urn.fi/urn:nbn:fi:lb-2020112935).
- Computing on LUMI. We acknowledge CSC – IT Center for Science, Finland for awarding this project access to
  the LUMI supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CSC (Finland) and the LUMI
  consortium through CSC – IT Center for Science, Finland.
- Made in the [GAIK project](https://gaik.ai/), co-funded by the European Union (ERDF).
