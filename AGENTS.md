# Working in this repository

This is the public home of the GAIK project's Finnish speech-to-text work. Everything committed here
is public, so the publication rules below come first.

## Never commit

- **LA-murre text** (Lauseopin arkiston murrekorpus, CC BY-ND 4.0): no references, no model
  transcripts paired with them, no judge records, no LLM rewrites, not even short quotes in docs,
  docstrings or prompts. That covers `results/benchmarks/lamurre_fi/*.jsonl`,
  `results/judgements_dialect/**/*.jsonl` and `results/labels/lamurre_*.jsonl` (all gitignored).
  Numbers made from them (`scores.json`, `summary*.json`, `agreement-*.json`, `*.stats.json`,
  `_run_metadata*.json`) are fine.
- **Model weights and adapters** (`results/finetune/*/adapter/` and `models/` are gitignored). The
  model trained on LA-murre is public as the `la-murre` revision of the Hugging Face model (Kielipankki
  found in October 2026 that the weights hold no protected part of the corpus); its training texts
  are LA-murre text and stay out (copies only on LUMI and in `data/`; the private HF repository that held them was deleted on 21 Oct 2026).
- **HY material beyond what gaik-toolkit already publishes.** The Ajokortti clip and its references
  are fetched from gaik-toolkit with `scripts/get_clip.py` into `data/` (gitignored); do not copy the
  mp3, and do not add HY's own tool transcripts (`hy25_*`, `hy26_*`) or judge records that quote them.
- **Secrets and local details**: no `.env`, keys or tokens, no absolute local paths, no LUMI user
  names, project numbers or scratch paths (use `$LUMI_USER`, `$LUMI_PROJECT`), no e-mail addresses,
  no material from other GAIK projects.
- **Pages that embed the clip**: `results/compare.html` is built locally and gitignored.

`uv run pytest` includes `tests/test_public_safety.py`, which checks `git ls-files` for these.

## Conventions

- Work from the repo root and use `uv run ...`; offline tests: `uv run pytest`; lint:
  `uv run ruff check` and `uv run ruff format --check`.
- Every runner that writes results writes a secret-free `_run_metadata*.json` with
  `scripts/provenance.py` (`write_run_metadata(dir, meta, name=...)`); env values are recorded only as
  set/unset.
- Ground truth comes from human annotation or verified sources only. Model or LLM output is never
  used as a reference; LLM-written training targets are marked as such.
- README layout: TL;DR, results sections, What's in this folder, How it works, Run it again, Caveats.
  `score_benchmark.py` and `make_readme_table.py` write between the README's table markers.
- GPU dependencies are pinned only in `lumi/setup.sh`; `pyproject.toml` covers the local scripts.
- Text files are LF (`.gitattributes`); every writer in `scripts/` passes `newline="\n"`
  (`tests/test_line_endings.py` checks both).
