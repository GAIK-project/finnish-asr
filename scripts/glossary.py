"""A glossary prompt: words the speaker may say, given to Whisper as previous-text context.

The idea follows B-Whisper (arXiv 2502.11572): prompting a plain Whisper with a word list helps the listed
words but hurts the rest, while a Whisper fine-tuned on prompts that hold the utterance's rare words mixed
with distractors learns to use the list when it helps and ignore it when it does not.

Training (finetune_whisper.py --glossary-prompt P): for a share P of the items the prompt lists that item's
rare words (lower-cased tokens of 4+ letters seen at most RARE_MAX times in the training texts, up to 5)
and 0 to 10 rare words drawn from other items, shuffled. Evaluation (run_benchmark.py --glossary tags):
the item's tagged terms (parl_terms_fi) plus 5 distractors drawn from other items' terms. The words are
lower-case in both, as a customer's list would be normalised. Text: "Sanasto: a, b, c." Standard library only.
"""

from __future__ import annotations

import random
import re
from collections import Counter

RARE_MAX = 3
MAX_TRUE, MAX_DISTRACT = 5, 10


def words(text: str) -> list[str]:
    return [w for w in re.sub(r"[^\w]+", " ", text.lower()).split() if w.isalpha()]


def rare_vocab(texts: list[str]) -> set[str]:
    counts = Counter(w for t in texts for w in words(t))
    return {w for w, c in counts.items() if c <= RARE_MAX and len(w) >= 4}


def glossary_text(entries: list[str]) -> str:
    return "Sanasto: " + ", ".join(entries) + "."


def train_glossary(text: str, rare: set[str], pool: list[str], rng: random.Random) -> list[str]:
    own = list(dict.fromkeys(w for w in words(text) if w in rare))
    rng.shuffle(own)
    entries = own[:MAX_TRUE] + rng.sample(pool, k=min(len(pool), rng.randint(0, MAX_DISTRACT)))
    entries = list(dict.fromkeys(entries))
    rng.shuffle(entries)
    return entries


def eval_glossary(terms: list[str], pool: list[str], rng: random.Random, distractors: int = 5) -> list[str]:
    own = list(dict.fromkeys(terms))
    others = [w for w in pool if w not in own]
    entries = own + rng.sample(others, k=min(len(others), distractors))
    rng.shuffle(entries)
    return entries
