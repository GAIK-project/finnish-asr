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
- contextual-biasing
- lora
datasets:
- GAIK-project/skn-finnish-dialects-asr
- google/fleurs
- facebook/voxpopuli
---

# Whisper large-v3 for Finnish, with a glossary prompt (revision `glossary`)

This branch holds a variant of [GAIK-project/whisper-large-v3-fi-dialects](https://huggingface.co/GAIK-project/whisper-large-v3-fi-dialects)
that was **trained to use a word list**. Give it the names and terms the speech may contain, and it writes them
right far more often; give it no list and it works like the main model.

Trained at Haaga-Helia University of Applied Sciences in the [GAIK project](https://gaik.ai/)
(Generative AI-enhanced Knowledge Management in Business), a research project led by Haaga-Helia
with the University of Helsinki and Tampere University, co-funded by the European Union (ERDF).

**[GAIK project](https://gaik.ai/)** · [Haaga-Helia University of Applied Sciences](https://www.haaga-helia.fi/en) · developed by Akseli Leskinen ([laguagu](https://github.com/laguagu))

📊 [code, method and all results: GAIK-project/finnish-asr](https://github.com/GAIK-project/finnish-asr) · 🌐 [gaik.ai](https://gaik.ai/)

## Results

1 525 held-out segments of Finnish parliament speech that contain names, acronyms or terms (Aalto corpus, speakers
and sessions not in training). The list holds the segment's own words plus five words from other segments, so the
model must pick, not copy.

| | WER % | Acronyms, English words and person names written exactly |
|---|--:|--:|
| Whisper large-v3, no list | 14.5 | 66 % |
| Whisper large-v3 with the list (not trained for it) | 15.6 | 79 % |
| Main model, no list | 12.4 | 75 % |
| Main model with the list (not trained for it) | 13.0 | 84 % |
| This model, no list | 12.2 | 76 % |
| **This model with the list** | **11.6** | **93 %** |

A list helps an untrained model's names but adds other errors; trained for it, the model gains on both. Without a
list this model matches the main model: SKN dialect test 19.5 % WER (main 19.4), FLEURS 7.7 (7.6), LA-murre 57.5
(57.4), VoxPopuli 11.9 (11.1; this set moves by more than a point between two seeds of one recipe).

The list in this test always held every true word, the best case for a customer vocabulary; a list that misses words
helps less. The words were tagged by an LLM (Claude Opus 5.5) and the tags checked on a sample.

## Use it

Write the list as Whisper's previous-text prompt, in the form the model was trained on: `Sanasto: ` and the words in
lower case, separated by commas, ending with a full stop.

```python
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

repo, rev = "GAIK-project/whisper-large-v3-fi-dialects", "glossary"
proc = WhisperProcessor.from_pretrained(repo, revision=rev)
model = WhisperForConditionalGeneration.from_pretrained(repo, revision=rev, dtype=torch.float16).to("cuda")

feats = proc.feature_extractor(audio_16k, sampling_rate=16000, return_tensors="pt").input_features
prompt = proc.tokenizer.get_prompt_ids("Sanasto: orpo, kela, sote, finnair.", return_tensors="pt").to("cuda")
ids = model.generate(feats.to("cuda", torch.float16), language="fi", task="transcribe",
                     prompt_ids=prompt, max_new_tokens=444 - len(prompt))
print(proc.batch_decode(ids, skip_special_tokens=True)[0])
```

Keep the list short (tens of words, not hundreds): the prompt and the transcript share Whisper's 448 positions.
Cut long audio into 15–25 s pieces at quiet points.

## How it was trained

The main model's data and recipe (SKN, LA-murre, FLEURS, VoxPopuli and 72 h of parliament speech; LoRA rank 64,
two epochs, 6 h 35 min on one AMD MI250X on LUMI), except that half of the items, drawn again every epoch, were given
a list: the item's rare words (seen at most three times in the training texts, up to five) and 0–10 rare words from
other items, shuffled. No loss falls on the list. The idea follows B-Whisper (arXiv 2502.11572).

## Credits and licences

As the [main model](https://huggingface.co/GAIK-project/whisper-large-v3-fi-dialects): weights Apache 2.0; SKN (Kotus,
CC BY 4.0, [urn:nbn:fi:lb-2020112937](http://urn.fi/urn:nbn:fi:lb-2020112937)); LA-murre (University of Turku and
Kotus, CC BY-ND 4.0, [urn:nbn:fi:lb-2020112935](http://urn.fi/urn:nbn:fi:lb-2020112935)); Aalto Finnish Parliament
ASR Corpus v2 (CLARIN PUB +BY +PRIV +ND, [urn:nbn:fi:lb-2022052002](http://urn.fi/urn:nbn:fi:lb-2022052002)), all via
Kielipankki; FLEURS (Google, CC BY 4.0), VoxPopuli (Meta, CC0). Computing on LUMI: we acknowledge CSC – IT Center for
Science, Finland for awarding this project access to the LUMI supercomputer, owned by the EuroHPC Joint Undertaking.
Made in the [GAIK project](https://gaik.ai/), co-funded by the European Union (ERDF).
