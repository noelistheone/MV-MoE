# Does the Recommender Use the Picture? — code and results

Code and result artifacts for measuring whether multimodal recommenders use item images, judged
against the accuracy change that retraining with another random seed produces on its own.

> **Anonymous release for double-blind review.** Nothing here identifies the authors (the repository
> name is a historical artifact). Absolute paths are placeholders rooted at `/workspace`:
> `/workspace/MechInterp` is this repository and `/workspace/Recsys` is the training framework plus
> data (see *Setup*). Datasets and model checkpoints are not redistributed; every **result artifact**
> the paper's numbers come from **is** included, under `results/`.

## Regenerate every certified number

```bash
python scripts/gen_facts.py        # reads results/**/*.json, writes results/FACTS.md
```

`results/FACTS.md` is the certified number source for the paper: every quantitative statement in the
paper is either in this file or computed directly from a named artifact. The generator reads only the
shipped JSON files, so it runs without GPUs, data or checkpoints. It rounds half-up on exact decimals,
so its output does not depend on the Python version.

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
| Table 1 — FREEDOM, deletion and retraining on eight converged seeds per dataset | `scripts/exp_converged_knockout.py`; `scripts/exp_modality_holdout.py` + `scripts/holdout_verdicts.py` | `results/phase_convergence/converged_knockout.json`; `results/phase_holdout/final_verdicts_p100.json` |
| Table 2 — encoder control | `scripts/trackA_extract_clip.py`, `scripts/trackA_clip_freedom.py` | `results/trackA_clip/clip_freedom.json` |
| Table 3 — behavioral alignment | `scripts/exp_alignment_stats.py` | `results/phase_align/`, `results/phase_micro/alignment_microlens.json` |
| Table 4 — deletion across eleven architectures | `scripts/exp_exact_crossarch.py`, `scripts/exact_ko/*.py`, `scripts/rejudge_exact.py` | `results/phase_exact/exact_crossarch.json` |
| Table 5 — test-time averaging vs deletion | `scripts/phasex_crossarch_knockout.py` | `results/phase_exact/screen_vs_exact.json` |
| Dataset statistics, feature storage | `scripts/dataset_stats.py` | `results/phase0/dataset_stats.json` |
| Registered rule and predictions (Threats to Validity) | — | `results/phase_micro/PREREG.md`, `results/phase_holdout/PREREG_HOLDOUT.md` (redacted for double-blind review) |

Every number in the paper, with its artifact, is in `results/FACTS.md` (regenerate with `scripts/gen_facts.py`).

## Experiments → code → artifacts (all)

| Experiment | Driver(s) | Artifact(s) |
|---|---|---|
| FREEDOM deletion, eight converged seeds per dataset | `scripts/exp_converged_knockout.py` | `results/phase_convergence/converged_knockout.json` |
| FREEDOM retraining without a modality, eight paired seeds | `scripts/exp_modality_holdout.py`, `scripts/holdout_verdicts.py` | `results/phase_holdout/final_verdicts_p100.json` |
| Single-checkpoint deletion, Electronics and LGMRec | `scripts/phase1_knockout.py` | `results/phase1/knockout_modality.json` |
| Seed-noise floors | `scripts/phase0_noisefloor.py`, `scripts/exp_mde_perdataset.py`, `scripts/phase_micro_paired_floor.py` | `results/phase_mde/`, `results/phase_micro/` |
| Encoder control (CNN/CLIP image × BERT/CLIP text) | `scripts/trackA_extract_clip.py`, `scripts/trackA_clip_freedom.py` | `results/trackA_clip/clip_freedom.json` |
| Behavioral alignment (bootstrap CIs, permutation test) | `scripts/exp_alignment_stats.py` | `results/phase_align/`, `results/phase_micro/alignment_microlens.json` |
| Image-weight re-mix on trained models | `scripts/phase2_weight_sweep.py` | `results/phase2/` |
| Deletion across eleven architectures, per-cell floors | `scripts/exp_exact_crossarch.py`, `scripts/exact_ko/*.py`, `scripts/rejudge_exact.py` | `results/phase_exact/exact_crossarch.json` |
| Test-time feature averaging on the same checkpoints | `scripts/phasex_crossarch_knockout.py` | `results/phasex_crossarch/crossarch_knockout.json`, `results/phase_exact/screen_vs_exact.json` |
| Baselines re-trained to convergence (reproduction check) | `scripts/exp_patience_control.py` | `results/phase_convergence/table_rerun.json` |

## Layout
```
scripts/   experiment drivers; exact_ko/ holds one deletion module per architecture
src/       checkpoint bridge (models/), ranking effects incl. RBO (interp/), seeding and capture (utils/)
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
Electronics) in the MMRec preprocessed layout, and the MicroLens micro-video benchmark in its
MMRec-format release with pre-extracted features.

## Reproducibility notes
- Early stopping and model selection use validation Recall@20; test metrics are reported only for the
  selected checkpoint.
- Every converged checkpoint's deletion baseline is checked against the Recall@20 its training run
  logged; a mismatch (e.g. a checkpoint read while still being written) is refused, not scored.
