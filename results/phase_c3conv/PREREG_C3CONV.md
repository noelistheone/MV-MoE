# Registration: converged multi-seed re-test of the cross-architecture deletion screen (C3)

Recorded 2026-09-23 (UTC+8), before any new seed of this design was trained. Only the seed-2024
converged checkpoint of each (model, dataset) exists at recording time (results/phase_convergence/table_rerun.json).

## Design
- Models: the ten non-FREEDOM architectures whose scores image reaches: VBPR, DAMRS, MMGCN, LATTICE,
  MGCN, MENTOR, GUME, SMORE, COHESION, LGMRec (BM3 excluded: content off the scoring path;
  FREEDOM already has eight converged seeds per dataset).
- Training: the converged protocol of the FREEDOM study (patience 100, cap 3000 epochs), published
  configs unchanged, seeds 2024-2031 (eight; 2024 already trained). Order of datasets by cost:
  Baby, then Sports, then Clothing, then MicroLens as budget allows. A dataset enters the analysis
  only when every model in it has >= 5 seeds; the seed count used per cell is reported.
- Instrument: the same per-architecture deletion modules (scripts/exact_ko/), default variant, on
  every seed's checkpoint; every checkpoint's deletion baseline must match its trainer-logged
  Recall@20 (staleness guard), else the row is refused.

## Rule (unchanged from the paper)
- F_level = 2 sd(Recall@20 of the full model across seeds); F_paired = 2 sd(per-seed deletion deltas).
- SIGNIFICANT iff |mean delta| > max(F_level, F_paired); BELOW iff below both; MARGINAL otherwise.
- Every cell also gets a two-sided paired t-test on the per-seed deletion deltas (H0: mean 0),
  the count of negative seeds and a 95% CI. Multiplicity across all judged cells:
  Benjamini-Hochberg at q = 0.05 and Holm at alpha = 0.05, on the paired-t p-values.

## Predictions (from the default-patience screen, stated before the runs)
- P1: LGMRec/Baby image deletion is SIGNIFICANT and survives BH.
- P2: FREEDOM-like frozen-graph image paths (LATTICE Baby) and GUME, MENTOR, MGCN are not SIGNIFICANT.
- P3: fewer cells are SIGNIFICANT under the two-floor rule than cleared the level floor in the screen
  (13 of 32), because the rule also requires clearing the paired floor.
Whatever the outcome, all judged cells are reported; none is dropped after seeing results.
