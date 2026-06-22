# MechInterp — Causal Modality Analysis & Late Fusion for Multimodal Recommenders

Code for **measuring whether multimodal recommenders actually use the image modality**, and a
constructive **late-fusion recommender (MV-MoE)** motivated by that measurement.

The core question: a multimodal recommender is *given* product photos and text — but does it *use*
the image when ranking? This toolkit answers it causally, against a per-dataset training-noise floor,
and then turns the diagnosis into a model.

> **Anonymous release.** This repository is identity-free by design. Absolute paths are placeholders
> rooted at `/workspace` (`/workspace/MechInterp` for this repo, `/workspace/Recsys` for the external
> data directory); adjust them, or symlink, to your layout. Heavy artifacts (datasets, checkpoints,
> activations, results) are **not** included — see *Data* below.

## What's here

| Capability | Where |
|---|---|
| **Exact stream knockout** — delete a modality's additive term from an additive-fusion recommender and measure ΔRecall/ΔNDCG vs. a noise floor | `scripts/phase1_knockout.py`, `scripts/phase_gaps3_knockouts_alignment.py` |
| **Per-dataset noise floor (MDE)** — multi-seed retraining → minimum detectable effect | `scripts/phase0_noisefloor.py`, `scripts/exp_mde_perdataset.py`, `scripts/exp_mde_analysis.py` |
| **Image-weight sweep** — frozen re-mix + retrain to separate "ignored by construction" from "intrinsically weak" | `scripts/phase2_weight_sweep.py`, `scripts/phase2_retrain.py` |
| **Geometry diagnostics** — effective rank, anisotropy, modality gap, alignment/uniformity | `src/interp/geometry.py`, `scripts/phase3_geometry.py` |
| **Behavioral-alignment metric** — does a stream's similarity predict co-purchase? (bootstrap CIs + permutation test) | `scripts/exp_alignment_stats.py` |
| **Cross-architecture screen** — input-mean knockout across many published recommenders | `scripts/phasex_crossarch_knockout.py` |
| **CLIP confound control** — re-encode image/text with CLIP ViT-L/14 and re-test | `scripts/trackA_extract_clip.py`, `scripts/trackA_clip_freedom.py` |
| **Sparse autoencoders** — TopK/L1 SAE + AuxK dead-feature revival, modality-origin attribution, causal steering | `src/sae/`, `scripts/phase4_sae.py`, `scripts/phase5_steering.py`, `src/interp/steering.py` |
| **Per-hop / universality analysis** | `scripts/phase67_universality_paths.py` |
| **MV-MoE** — validation-tuned, score-level late fusion of complementary views (bagged base model, user-kNN, behavior-aligned image/text item–item) | `scripts/ensemble_eval.py`, `scripts/component_ablation.py`, `scripts/finalize.py`, `scripts/bagged_baseline.py`, `scripts/mvmoe_seeds.py`, `scripts/userknn_scores.py`, `scripts/build_ba_i2i.py` |
| **Cross-domain (short-video) generalization** | `scripts/run_microlens_all.sh`, `scripts/finalize_microlens.py`, `scripts/gen_microlens_views.py` |

## Layout
```
src/
  sae/         TopK/L1 SAE (sae.py) + trainer (trainer.py): AuxK revival, decoder-norm constraint
  utils/       activation capture & caching (activations.py), deterministic seeding (seed.py)
  models/      FrozenRecommender contract (base.py), GUME+BAI wrapper (gume_bai.py),
               data/checkpoint bridge (recsys_bridge.py), SASRec (sasrec.py)
  interp/      geometry.py, feature_metrics.py, ranking_effects.py (ΔRecall/ΔNDCG/RBO), steering.py
scripts/       experiment drivers (phase*/track*/exp_*), the MV-MoE pipeline, encoders, analysis utils
configs/       default.yaml — per-experiment template
```

## Setup
```bash
bash scripts/setup_env.sh        # conda create -n mechinterp python=3.11 + pip install -r requirements.txt
conda activate mechinterp
python scripts/smoke_test.py     # sanity-check the SAE backbone on GPU
```
See `environment.yml` / `requirements.txt`. A CUDA GPU is assumed for full-ranking scoring and retrains.

## Data
The experiment drivers read recommender data and precomputed item features in **MMRec layout** from an
external directory (placeholder `/workspace/Recsys/data/<dataset>/`), e.g. `<dataset>.inter` interaction
splits and `image_feat.npy` / `text_feat.npy`. Datasets used: the public Amazon review benchmarks
(Baby, Sports, Clothing, Electronics) and the public MicroLens short-video benchmark. These are **not**
redistributed here; point the scripts at your own copy and adjust the placeholder paths.

## Notes
- Scripts are intentionally self-contained drivers; several import shared evaluators (e.g.
  `ensemble_eval.py`) and so expect to be run from `scripts/` (or with `scripts/` on `PYTHONPATH`).
- All metrics are full-ranking Recall@K / NDCG@K / MAP@K with training-history masking; every reported
  effect is judged against the per-dataset MDE noise floor.
- Conda interpreter paths have been normalized to plain `python`; activate the env first.
