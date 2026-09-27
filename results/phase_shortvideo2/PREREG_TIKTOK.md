# Registration: FREEDOM on a second short-video dataset (TikTok)

Recorded 2026-09-24 (UTC-7), before any converged TikTok run. Only a 30-epoch and a 200-epoch
unconverged smoke run (seed 2024; results/phase_shortvideo2/_scratch) exist; they are not used.

## Why TikTok
The short-video dataset of the prior single-run audit (DY from NineRec) could not be downloaded
(HTTP 403, 2026-09-23). TikTok (MMSSL/DiffMM preprocessed release: 9,308 users, 6,710 items,
68,722 interactions, source split kept) is the only reachable short-video dataset with both visual
and text features that multimodal-recommendation papers use. Its "image" is a 128-d video key-frame
feature from an undocumented encoder (no raw content), text is 768-d Sentence-BERT; validation and
test hold one item per user. Features are strongly alike (mean pairwise cosine 0.856 vs 0.508
MicroLens, 0.224 Baby).

## Design (identical to the MicroLens/Amazon study)
FREEDOM, published config unchanged (no per-dataset tuning), conditions full / no_image / no_text,
seeds 2024-2031, patience 100, cap 3000 (scripts/exp_tiktok_holdout.py). Exact image and text
deletion on every converged full checkpoint (same instrument as results/phase_convergence/
converged_knockout.json). Behavioral alignment (same statistic as Table 3) on CPU. One converged
LightGCN seed as the behavior-only reference. Rule: the two-floor rule plus paired t, negative-seed
count and 95% CI, exactly as for the other datasets.

## Predictions
- T1: retraining without text costs less than 10% of Recall@20 (Amazon: 22-35%).
- T2: retraining without images is not SIGNIFICANT (BELOW or MARGINAL).
- T3: text does not dominate: the paired difference (no_text - no_image) does not favor image
      removal by more than both floors, i.e. text's cost does not significantly exceed image's.
- T4 (tests the paper's alignment claim): if T3 holds, TikTok's text-behavior alignment does not
      exceed its image-behavior alignment by more than on MicroLens.
All outcomes are reported whether or not they match.
