# Disclosure attached to PREREG_TIKTOK.md (written 2026-09-24, after the TikTok runs finished)

Found by the independent integrity audit (workflow verify-tiktok-final): the registration was NOT
blind for T1-T3. Before PREREG_TIKTOK.md was written (01:08 PDT), a 30-epoch single-seed smoke run
(results/phase_shortvideo2/smoke30_freedom_tiktok.json, 23:34 PDT the previous day, seed 2024,
TEST metrics) had shown Recall@20 full 0.0615, no_image 0.0594 (-3.4%), no_text 0.0728 (+18.3%),
and the author of the registration had read that report. The registration says the smoke runs are
"not used" but does not report their outcome. T1-T3 must therefore be described as predictions
made after an unconverged single-seed pilot, not as blind predictions. T4 (alignment) had no pilot.

Two further points the audit asked to state explicitly:
- T1 ("costs less than 10%") holds only because retraining without text RAISES Recall@20 by 10.2%
  (8/8 seeds, p = 8e-4); the two-floor rule calls it BELOW-FLOOR at 0.99x the paired floor, so the
  primary rule and the paired t-test disagree on this cell.
- T4's operationalization (raw features, absolute gap, uniform null) was fixed in code after six
  converged runs had finished; all four variants give the same outcome (T4 fails).
