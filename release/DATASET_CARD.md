---
language:
- fi
license: cc-by-4.0
pretty_name: SKN Finnish dialects for speech recognition
task_categories:
- automatic-speech-recognition
tags:
- finnish
- dialects
- speech
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
  - split: validation
    path: data/validation-*.parquet
  - split: test
    path: data/test-*.parquet
dataset_info:
  features:
  - name: audio
    dtype:
      audio:
        sampling_rate: 16000
  - name: id
    dtype: string
  - name: municipality
    dtype: string
  - name: skn_section
    dtype: int32
  - name: seconds
    dtype: float32
  - name: text
    dtype: string
  - name: text_normalized
    dtype: string
  - name: text_dialect
    dtype: string
  - name: text_source
    dtype: string
---

# SKN Finnish dialects, cut for speech recognition

Finnish dialect interviews from **Suomen kielen näytteitä (SKN)**, cut into 2–25 s segments
with a standard-Finnish transcript, split **by municipality** for training and testing speech
recognition. Made at Haaga-Helia University of Applied Sciences in the [GAIK project](https://gaik.ai/)
(led by Haaga-Helia with the University of Helsinki and Tampere University, co-funded by the
European Union, ERDF) to train [GAIK-project/whisper-large-v3-fi-dialects](https://huggingface.co/GAIK-project/whisper-large-v3-fi-dialects).

| Split | Segments | Hours | Municipalities |
|---|--:|--:|---|
| train | 15 042 | 61.9 | 38 |
| validation | 200 | 0.9 | Pälkäne, Sotkamo |
| test | 600 | 2.5 | Alastaro, Artjärvi, Eurajoki, Heinola, Hinnerjoki, Ikaalinen, Jaala, Mikkeli, Rautu, Vihti |

The test municipalities are SKN numbers 5, 10 … 50: every fifth booklet of the series, in publication
order. By Kielipankki's dialect classification they cover the south-western, south-western
transitional, Häme, Savo and south-eastern dialects; no Ostrobothnian or Peräpohjola municipality is in
the test (all nine are in train). No test interviewee is in training. Interviewer speech is kept, and
SKN's fieldworkers recorded several municipalities, so a few test segments may contain an
interviewer's voice that is also heard in training.

## Fields

| Field | What |
|---|---|
| `audio` | 16 kHz mono, resampled from SKN's 44.1 kHz recordings |
| `text` | the training target: train and validation have punctuation and capitals (see below); test has `text_normalized` |
| `text_normalized` | SKN's human transcript in standard Finnish word forms, no punctuation |
| `text_dialect` | SKN's human transcript in dialect spelling, in SKN's own notation (stress and annotation marks kept); empty for 43 segments where SKN has no dialect sentence at the same times |
| `text_source` | `human` or `llm-punctuation` |
| `municipality`, `skn_section` | the place, spelled as in SKN's file names without ä and ö (`Palkane` = Pälkäne, `Artjarvi` = Artjärvi), and the SKN number (1–50) |

## How it was made

1. SKN's sentence annotations of all speakers, in time order, are joined while the gap is under
   1.5 s and the segment stays under 25 s; segments under 2 s or with fewer than 3 words are dropped.
2. Test and validation keep a reproducible sample (seed 0) of 60 and 100 segments per municipality.
3. For train and validation, Claude Opus 5.5 added punctuation and capitals to the standard-Finnish
   text, 20 segments per call, text only. An answer was kept only if its words were exactly the
   human transcript's; 15 024 of 15 042 train segments passed, the others have
   `text_source = human`. **Test transcripts are untouched human text.**

Scripts: [`prepare_skn.py`](https://github.com/GAIK-project/finnish-asr/blob/main/scripts/prepare_skn.py) and
[`prepare_targets.py`](https://github.com/GAIK-project/finnish-asr/blob/main/scripts/prepare_targets.py) in
[GAIK-project/finnish-asr](https://github.com/GAIK-project/finnish-asr), which also holds the method and all results (summary in
[gaik-toolkit · finnish_dialect_asr](https://github.com/GAIK-project/gaik-toolkit/tree/main/evaluation_layer/eval_methods/finnish_dialect_asr)). Its
[`results/labels/skn_parts.jsonl`](https://github.com/GAIK-project/finnish-asr/blob/main/results/labels/skn_parts.jsonl) gives SKN's sentence
times inside every segment, by `id`.

## Caveats

- `text_dialect` was empty in every row of the first upload (13 Oct 2026); it was filled on 14 Oct 2026. The audio, the
  other fields and the splits did not change.
- 1960s–70s interviews with mostly elderly speakers, recorded in varying conditions.
- Interviewer speech is included where it falls inside a segment, as in SKN's annotation.
- Punctuation in train and validation is model-written, checked only for unchanged words.
- In a few dozen segments SKN's normalised transcript itself keeps some punctuation or dialect notation;
  `text_normalized` (and so `text`) leaves it as SKN has it.

## Source and licence

Developed by Akseli Leskinen ([laguagu](https://github.com/laguagu)), Haaga-Helia University of Applied Sciences, in the [GAIK project](https://gaik.ai/).

**Suomen kielen näytteitä**, Kotimaisten kielten keskus (Kotus), via Kielipankki – The Language
Bank of Finland, [urn:nbn:fi:lb-2020112937](http://urn.fi/urn:nbn:fi:lb-2020112937), CC BY 4.0.
Original corpus: [kielipankki.fi/corpora/skn](https://www.kielipankki.fi/corpora/skn/),
download [kielipankki.fi/download/SKN](https://www.kielipankki.fi/download/SKN/). This derived
dataset is also CC BY 4.0; credit Kotus and Kielipankki as the source.
