# Registration: does the instrument explain the size gap to prior single-run audits?

Recorded 2026-09-24 (UTC-7), before any full-length run of these conditions. Only 2-3 epoch smoke
runs exist (results/phase_noise/_smoke), not used for any estimate.

## Question
Our FREEDOM retraining removes a modality (v_feat/t_feat = None; the surviving similarity graph gets
weight 1). Ye et al. instead replace it with Gaussian noise (their code: N(1, 0.1), 4096-d image,
384-d text, fixed per item), which makes a hub-shaped random kNN graph; Pomo et al.'s image-only
FREEDOM keeps the image graph at its mixing weight 0.1. We test whether these instrument choices move
the retraining cost under our converged eight-seed protocol.

## Design
scripts/exp_noise_holdout.py; FREEDOM, published config, patience 100, cap 3000, seeds 2024-2031,
paired by seed with the existing full / no_image / no_text runs
(results/phase_holdout/freedom_p100*_runs.json). Our graph builder is kept for every condition
(not Ye's directed self-including builder), so this tests noise-vs-removal under our FREEDOM, not a
replication of Ye's pipeline. Noise seed per run: SeedSequence([run_seed, 42]).
- Baby: noise_image, noise_text, no_image_keepw, no_text_keepw
- Sports: noise_image, noise_text
- Clothing: no_text_keepw

## Predictions (directional, paired over eight seeds)
- N1: noise_image costs more than no_image (mean paired difference < 0) on Baby and Sports.
- N2: noise_text costs more than no_text on Baby and Sports.
- N3: no_text_keepw costs less than no_text on Baby and Clothing (part of our text-retraining cost
      comes from giving the image graph weight 1).
Each contrast is judged with the two-floor rule and a paired t-test; all are reported.
