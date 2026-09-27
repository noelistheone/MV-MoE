# Amendment 1 to PREREG_C3CONV.md

Recorded 2026-09-24 (UTC-7), after the scorer was built and audited and BEFORE any multi-seed verdict
was computed (at recording time no dataset had >= 5 seeds for any cell, so no registered verdict,
floor or p-value could have been seen). Only seed-2024/2025/2026 single-checkpoint deltas had been
printed during scorer testing. Each item states the problem found, the decision, and why.

A1. Multiplicity family. The registration says "across all judged cells". The first scorer build
    corrected image and text cells in two separate families. DECISION: the PRIMARY family is all
    judged image AND text cells together (literal reading of the registration). The image-only family
    (the convention of the paper's default-patience screen) is reported as SECONDARY.
A2. Undefined t statistic. If all per-seed deltas of a judged cell are exactly equal, the t statistic
    is undefined. DECISION: p = 1.0 if they are all exactly 0 (no effect on any seed), p = 0.0
    otherwise; the cell stays in the family (never dropped), so m is not reduced.
A3. Baseline guard. An exact match of the recomputed baseline to the trainer-logged Recall@20 is
    impossible for three architectures: LGMRec and COHESION score stochastically at inference, and
    LATTICE rebuilds an item graph that the checkpoint does not store. DECISION: exact match for the
    other seven; for LGMRec and COHESION the logged value must lie within 4 sd of 10 re-draws of the
    model's own baseline; for LATTICE, checkpoint identity (stored epoch = recorded best epoch, config
    snapshot = recorded seed/patience/cap) with the discrepancy recorded. Each row carries its guard
    mode; a sensitivity analysis dropping the non-exact rows is reported alongside.
A4. Level floor source. DECISION: F_level (primary) = 2 sd of the trainer-logged test Recall@20 of the
    full model across seeds, the convention of the existing floor code; the version on recomputed
    baselines is reported as secondary. The two coincide for exact-guard rows.
A5. Reconstruction tolerance. MMGCN fails the absolute 1e-5 tolerance on Sports/Clothing/MicroLens
    at 1.1e-5 to 3.8e-5 with score magnitudes 45-86, i.e. relative error 2.5e-7 to 4.5e-7 (a few
    float32 ulps). DECISION: a row passes if its absolute error is <= 1e-5 OR its error relative to
    the largest absolute score of that check is <= 1e-6 (about 8 float32 ulps), uniformly for all
    architectures. Rows failing both are refused.
A6. MicroLens scope. Converged single runs cost 10.0 h (MENTOR) and 8.4 h (MGCN) on MicroLens.
    DECISION: MicroLens covers the other eight models (VBPR, DAMRS, MMGCN, LATTICE, GUME, SMORE,
    COHESION, LGMRec) with seeds 2024-2028 (five); it is admitted when all eight have >= 5 seeds.
    MENTOR and MGCN on MicroLens are reported as not run. Baby, Sports and Clothing keep all ten
    models and seeds 2024-2031, admitted as registered (all ten >= 5 seeds).
A7. Device. Where GPU memory forces CPU scoring (COHESION/MicroLens), one checkpoint is scored on
    both devices and the deltas compared; the device is recorded per row.
