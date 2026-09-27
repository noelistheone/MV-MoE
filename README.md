# Does the Recommender Use the Picture? — code and results

Code and result artifacts for measuring whether multimodal recommenders use item images, judged
against the accuracy change that retraining with another random seed produces on its own.

> **Anonymous release for double-blind review.** Nothing here identifies the authors (the repository
> name is a historical artifact). Absolute paths are placeholders rooted at `/workspace`:
> `/workspace/MechInterp` is this repository and `/workspace/Recsys` is the training framework plus
> data (see *Setup*). Datasets and model checkpoints are not redistributed; every **result artifact**
> the paper's numbers come from **is** included, under `results/`.

## Where the paper's numbers come from

Every number in the paper is computed from the JSON files under `results/`, which are shipped. The
drivers below regenerate them from data and checkpoints (GPU); the summary scripts recompute the
reported statistics from the shipped JSON alone (CPU, no data needed):

```bash
python scripts/exp_c3conv_score.py --verdicts    # cross-architecture verdicts -> results/phase_c3conv/verdicts.json
python scripts/noise_verdicts.py                 # retraining controls -> results/phase_noise/noise_verdicts.json
python scripts/tiktok_verdicts.py                # TikTok -> results/phase_shortvideo2/tiktok_verdicts.json
python scripts/holdout_verdicts.py               # FREEDOM retraining -> results/phase_holdout/final_verdicts*.json
python scripts/derive_sac_numbers.py             # derived numbers -> results/phase_paper/derived_for_paper.json
python scripts/final_impact_summary.py           # counts in Sections 5-6 -> results/phase_paper/final_impact_summary.json
python scripts/make_fig_retraining.py            # Figure 1
python scripts/gen_facts.py                      # screen-era facts -> results/FACTS.md
```

`results/FACTS.md` covers the default-protocol screen and the single-checkpoint analyses. Its DAMRS
screen and averaging rows come from the first DAMRS port, whose user-item graph was empty (see
*DAMRS port repair* below); the paper uses the repaired cells in
`results/phase_c3conv_g11/screen_g11.json` instead.

## The two instruments

- **Deletion** (scoring time): for a trained model whose score is a sum of behavioral, image and text
  terms, remove the image term and re-rank. For FREEDOM the decomposition is exact (reconstruction
  error ≤2×10⁻⁶); other architectures are deleted exactly, deleted and renormalized, or bracketed
  between two conventions (`scripts/exact_ko/`, one module per architecture).
- **Retraining** (training time): train the model without the modality — no similarity graph and no
  auxiliary loss for it — on matched seeds (`scripts/exp_modality_holdout.py`).

## The adjudication rule

For a delta `d` measured over paired seeds:

- `F_level  = 2 · sd(Recall@20 of the full model across seeds)`
- `F_paired = 2 · sd(per-seed paired deltas)`
- **significant** iff `|d| > max(F_level, F_paired)`; **below floor** iff below both; **marginal**
  otherwise. The floors are magnitude thresholds, not tests, so every eight-seed verdict also carries
  a paired t-test, the count of negative seeds and a 95% confidence interval.

Implemented in `scripts/holdout_verdicts.py` (retraining), `scripts/exp_converged_knockout.py`
(deletion) and `scripts/phase1_knockout.py` (floors; a floor from fewer than five seeds with an effect
between 0.5× and 2× of it is flagged *borderline*).

## Paper → code → artifacts

| Paper | Driver(s) | Artifact(s) |
|---|---|---|
| Figure 1 — retraining FREEDOM without a modality, per seed | `scripts/make_fig_retraining.py` | `results/phase_paper/fig_retraining.json` |
| Table 1 — FREEDOM, deletion and retraining, eight converged seeds, five datasets | `scripts/exp_converged_knockout.py`, `scripts/exp_modality_holdout.py` + `scripts/holdout_verdicts.py`; TikTok: `scripts/exp_tiktok_holdout.py`, `scripts/tiktok_converged_knockout.py`, `scripts/tiktok_verdicts.py` | `results/phase_convergence/converged_knockout.json`, `results/phase_holdout/final_verdicts_p100.json`, `results/phase_shortvideo2/tiktok_verdicts.json` |
| Table 2 — datasets and floors | `scripts/dataset_stats.py`, `scripts/convert_tiktok_mmrec.py`; feature provenance: `scripts/microlens_clip_verify.py`, `scripts/microlens_bgem3_verify.py` | `results/phase0/dataset_stats.json`, `results/phase_shortvideo2/` |
| Table 3 — retraining controls (keep-weight, noise replacement) | `scripts/exp_noise_holdout.py`, `scripts/noise_verdicts.py`, `scripts/derive_sac_numbers.py` | `results/phase_noise/noise_verdicts.json`, `results/phase_paper/derived_for_paper.json` |
| Section 4.3 — CLIP encoder runs | `scripts/trackA_extract_clip.py`, `scripts/trackA_clip_freedom.py` | `results/trackA_clip/clip_freedom.json` |
| Table 4 — behavioral alignment | `scripts/exp_alignment_stats.py`, `scripts/tiktok_alignment.py` | `results/phase_align/`, `results/phase_micro/alignment_microlens.json`, `results/phase_shortvideo2/alignment_tiktok.json` |
| Table 5 (Baby, Sports) and the four-dataset results of Section 5 — converged deletion across ten architectures (eight seeds on Baby, Sports and Clothing; five on MicroLens for eight of them) | `scripts/exp_patience_control.py` (training), `scripts/exp_damrs_g11.py` (repaired DAMRS), `scripts/exp_c3conv_score.py` (deletion, guards, verdicts, multiplicity), `scripts/c3conv_smore_bracket.py`, `scripts/final_impact_summary.py` | `results/phase_c3conv/verdicts.json`, `results/phase_c3conv/scored_*.json`, `results/phase_c3conv/smore_bracket.json`, `results/phase_c3conv_g11/runs_*.json`, `results/phase_paper/final_impact_summary.json` |
| Section 5 — preliminary default-protocol screen | `scripts/exp_exact_crossarch.py`, `scripts/exact_ko/*.py`; repaired DAMRS cells: `scripts/exp_damrs_g11.py --protocol default`, `scripts/exp_damrs_g11_screen.py` | `results/phase_exact/exact_crossarch.json`, `results/phase_c3conv_g11/screen_g11.json` |
| Table 6 — test-time averaging vs deletion | `scripts/phasex_crossarch_knockout.py`, `scripts/exp_damrs_g11_screen.py`, `scripts/final_impact_summary.py` | `results/phase_exact/screen_vs_exact.json`, `results/phase_c3conv_g11/screen_g11.json` |
| Section 6.2 — training length, single runs, MENTOR's early peak | `scripts/exp_mde_perdataset.py`, `scripts/analysis_single_seed.py`, `scripts/c3conv_mentor_clothing_sensitivity.py` | `results/phase_mde/`, `results/phase_novelty/single_seed_instability.json`, `results/phase_c3conv/sensitivity_mentor_clothing.json` |
| Section 7 — registrations and their timing | — | `results/phase_micro/PREREG.md`, `results/phase_holdout/PREREG_HOLDOUT.md` (both redacted for double-blind review), `results/phase_c3conv/PREREG_C3CONV*.md`, `results/phase_noise/PREREG_NOISE.md`, `results/phase_shortvideo2/PREREG_TIKTOK*.md` (verbatim; sha256 in `results/prereg_hashes.txt`) |

## DAMRS port repair

The first DAMRS port filled its user-item adjacency with a dictionary update on a sparse matrix, as
the official code does. In recent versions of the numerical library that update leaves the matrix
empty, so those models trained without user-item propagation. `recsys/src/models/damrs.py` builds
the adjacency by coordinate assignment and asserts its size. All DAMRS runs the paper reports were
trained with this repaired file (`results/phase_c3conv_g11/`), under a registered amendment
(`results/phase_c3conv/PREREG_C3CONV_AMEND2.md` and its erratum). Each run records the sha256 of the
model file at its start and end; the registered value is `d001139f…`. The shipped copy differs from
that file only in scrubbed comments and paths, so the repaired-DAMRS scripts accept both hashes. The
rows scored under the name `damrs` in `results/phase_c3conv/scored_*.json` come from the first port
and are void: in `results/phase_c3conv/verdicts.json` they are never judged and outside every family
(`verdicts_preAMEND2.json` keeps the verdicts computed before the repair, for the record).
`results/phase_c3conv/NOTES_DAMRS_GRAPH.md` records our first, wrong diagnosis of the resulting
guard refusal and its correction.

## Layout
```
scripts/   experiment drivers; exact_ko/ holds one deletion module per architecture
src/       checkpoint bridge (models/), ranking effects incl. RBO (interp/), seeding and capture (utils/),
           and a sparse autoencoder that only the smoke test exercises (sae/)
recsys/    the training framework subset the drivers import: shared trainer, data, evaluator and the
           thirteen measured models (see NOTICE)
results/   every JSON artifact the paper's numbers come from, plus the generated FACTS.md
configs/   default.yaml
```

## Setup
```bash
conda env create -f environment.yml    # or: pip install -r requirements.txt  (Python 3.11)
conda activate mechinterp
mkdir -p /workspace && ln -s "$PWD" /workspace/MechInterp && ln -s "$PWD/recsys" /workspace/Recsys
# place the datasets under /workspace/Recsys/data/<dataset>/ (MMRec layout: <dataset>.inter,
# image_feat.npy, text_feat.npy)
```
Training and full-ranking evaluation assume a CUDA GPU. Regenerating the numbers from `results/`
needs neither.

## Data
Public benchmarks, not redistributed: the Amazon review datasets (Baby, Sports & Outdoors, Clothing,
Electronics) in the MMRec preprocessed layout, the MicroLens micro-video benchmark in its
MMRec-format release with pre-extracted features, and the TikTok release used by MMSSL
(`scripts/convert_tiktok_mmrec.py` converts it to the MMRec layout; `scripts/recsys_extra_datasets.py`
registers it with the training framework).

## Reproducibility notes
- Early stopping and model selection use validation Recall@20; test metrics are reported only for the
  selected checkpoint.
- Every converged checkpoint's deletion baseline is checked against the Recall@20 its training run
  logged; a mismatch (e.g. a checkpoint read while still being written) is refused, not scored.
- Converged runs stop after 100 epochs without a gain in validation Recall@20 (cap 3,000) and use
  seeds 2024-2031; on MicroLens the ten further architectures use 2024-2028 (MENTOR and MGCN only 2024).
  The default protocol stops after 20.
