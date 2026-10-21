# Region prompt and standard Finnish: plan of the fifth training round

Written on 20 Oct 2026, before any of these models was trained or scored. The rules below decide; a
number that comes out differently does not move them.

## Questions

1. **Does telling the model the dialect region help?** One model trained with Whisper's previous-text
   prompt `Murre: savolaismurteet.` on half of its dialect segments, then decoded with the true region,
   with no prompt, and with a wrong, far region (the control: a prompt that only says "this is
   dialect" helps the true and the wrong region alike).
2. **Can the dialect gain keep standard Finnish?** The `la-murre` version wins on dialect (SKN 19.7 %)
   but loses on FLEURS (7.8 against 7.6 %) and VoxPopuli (11.4 against 10.8 %). Here FLEURS and
   VoxPopuli go into every epoch twice.

## Regions

Six, LA-murre's own top level: LOU, LVÄ, HÄM, POH (with Peräpohjola), SAV, KAA
([`dialect_regions.py`](../../scripts/dialect_regions.py)). SKN's from each recording's IMDI
metadata, LA-murre's from the `dialect_region` of its public VRT files. Test coverage: SKN test LOU 120,
LVÄ 120, HÄM 240, SAV 60, KAA 60 segments (no POH municipality is in SKN's test); LA-murre test one
or more localities in every region.

## Runs

The recipe of the `la-murre` version for all (SKN + FLEURS + VoxPopuli + LA-murre, punctuated SKN
targets, LoRA r 64, 3 epochs, lr 1e-4, batch 8, the epoch with the lowest validation loss kept), one
GCD each on `small-g`:

| Name | Change | Why |
|---|---|---|
| `skn-lamurre-best` (exists) | – | control, seed 0 |
| `lamurre-s1` | seed 1 | how far two seeds of the same recipe land apart |
| `region` | `--region-prompt 0.5` | question 1 |
| `lamurre-open2` | `--open-repeat 2` | question 2 |
| `region-open2` | both | whether they add up |

Scored on SKN (600), FLEURS (918), VoxPopuli (478) and LA-murre (WER against dialect text, a
secondary signal; the blind LLM judge only for a model that passes). The two prompt models and the
control are decoded on SKN and LA-murre three times (`--region none|correct|wrong`); everything else
with the usual pipeline. [`region_compare.py`](../../scripts/region_compare.py) gives the paired
bootstrap intervals.

## Rules

- **Seed gap** = |WER(`lamurre-s1`) − WER(`skn-lamurre-best`)| on SKN, both without a prompt.
- **The region prompt works** if on SKN `region@correct` is lower than `region@none` by more than the
  seed gap with a bootstrap interval below zero, and lower than `region@wrong` by more than the seed gap.
  If `region@wrong` is about as good as `region@correct`, the prompt says only "dialect", not where.
- **A new main model** (decoded without a prompt) needs: SKN WER below the main model's 21.1 %, and
  FLEURS and VoxPopuli at most 0.5 points worse than the main model (8.1 and 11.3 %), the rule of the
  earlier rounds. Among those, the lowest SKN WER; a tie within the seed gap goes to the simpler recipe.
- Per-region numbers are reported, with their segment counts, but decide nothing on their own: 60
  segments from one municipality are too few.
- Region-specific adapters (a Savo-only model) come only after question 1: if a prompt does not help,
  a separate model per region is unlikely to, with 3 to 16 training municipalities per region.

## Amendment, 20 Oct 2026, before any new model was scored

The control's three decodes came first (results/finetune/region/). Two things they showed:

- Without prompt training a region prompt does nothing useful: on SKN `none` 20.68, `correct` 20.94,
  `wrong` 20.93 % (correct − none +0.26, interval +0.01 to +0.51); on LA-murre 57.77 / 57.61 / 57.67.
- The direct `generate()` path and the README's pipeline give the same model different numbers: 20.68
  against 19.7 % on SKN, both greedy; the outputs differ by a word here and there in 252 of 600
  segments. So: prompt comparisons use `generate()` only (`--region`), and the main-model rule and the
  README use the pipeline only. Every new model is decoded both ways.

## Cost

About 5–6 h per training on one GCD (0.5 GPU-hours an hour on `small-g`), the scoring a few hours:
under 20 GPU-hours of the GAIK project's 5 000.

## Result, question 1 (20 Oct 2026, 20:50)

The region prompt does not help. On SKN (600 segments, `generate()` decoding) the region model gives 20.94 %
without a prompt, 20.92 % with the true region and 20.91 % with a far wrong one: true − none −0.02 (interval −0.18 to
+0.14), wrong − none −0.04. The seed gap of the recipe is 0.04 (control 20.68, seed 1 20.64), so the rule's first
condition fails. On LA-murre the same: none 57.65, true 57.80, wrong 57.83. Training with prompts also cost the model
0.26 points when decoded without one (control − region −0.26, interval −0.51 to −0.01). Six coarse regions carry
nothing the audio does not already give; a separate model per region is not worth training on this data
(question 1's rule). Question 2 waits for the open2 runs.

## Result, question 2 and the round (21 Oct 2026)

Pipeline WER %, all sets complete (parl_fi: parliament 2016-test/unseen, 962 segments; parl_terms_fi: 1 525 held-out
parliament segments with tagged names, acronyms and terms; both from the Aalto corpus, outputs gitignored):

| Model | SKN | FLEURS | VoxPopuli | parl_fi | parl_terms_fi | LA-murre |
|---|--:|--:|--:|--:|--:|--:|
| Whisper large-v3 | 40.38 | 7.43 | 12.71 | 9.29 | 14.53 | 62.52 |
| main (v1.0) | 21.08 | 7.61 | 10.83 | 11.38 | 13.65 | 58.10 |
| la-murre (seed 0) | 19.72 | 7.81 | 11.45 | 9.23 | 13.70 | 57.41 |
| la-murre recipe, seed 1 | 19.69 | 7.86 | 13.25 | 8.85 | 14.07 | 57.75 |
| region prompt | 19.76 | 7.78 | 12.08 | 9.52 | 13.48 | 58.17 |
| standard Finnish ×2 | 20.74 | 7.92 | 11.17 | 9.28 | 13.84 | 58.07 |
| region + standard ×2 | 19.34 | 8.40 | 10.93 | 9.91 | 14.02 | 59.63 |

- **Seeds:** the same recipe lands 0.03 apart on SKN but 1.8 apart on VoxPopuli (11.45 against 13.25); the gap is
  insertions (2.84 against 4.42 %), Whisper's repeated phrases on a few long clips. VoxPopuli and parl_fi cannot
  separate these models; SKN can.
- **Question 2:** doubling FLEURS and VoxPopuli did not hold standard Finnish better. With the la-murre data it cost
  1.0 point on SKN and moved FLEURS the wrong way (7.92); with the region prompt it gave the best SKN (19.34) but the
  worst FLEURS (8.40) and LA-murre (59.63).
- **The rule for a new main model** (SKN under 21.1, FLEURS at most 8.1, VoxPopuli at most 11.3) passes only
  "standard Finnish ×2" (20.74 / 7.92 / 11.17), which is worse on SKN than the published la-murre version and within
  the VoxPopuli seed noise of it. No new model is proposed; the la-murre version stays the best dialect model.
- **Names and terms:** every fine-tune writes 71–72 % of the tagged acronyms, English words and person names
  exactly, against 66 % for base Whisper; the gain is acronyms (59 → 80 %). Person names stay at 65–67 % in every
  model: the next round's target.
- **Synthetic speech:** Whisper large-v3 makes 14.5 % WER on Chatterbox Multilingual's Finnish (40 FLEURS dev
  sentences, four cloned voices) against 7.4 % on real FLEURS: intelligible, but twice the errors of real speech.
  Usable at most as a small share beside real speech, for words that real data lacks.
