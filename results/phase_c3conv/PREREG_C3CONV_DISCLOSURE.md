# Disclosure attached to PREREG_C3CONV.md (written 2026-09-24)

Found by the independent verification of the Baby results:
1. Timing. PREREG_C3CONV.md was last modified 2026-09-23 23:14:50 -0700 (sha256 e80a2e83...,
   printed to the session log at creation but not entered in results/_queue/prereg_hashes.txt).
   The first new training runs (mgcn and cohesion, Baby, seed 2025) had started about five minutes
   earlier, around 23:09 -0700; none had finished, so no new result existed when it was written.
   Its header "2026-09-23 (UTC+8)" is wrong: the local time was UTC-7 (2026-09-24 06:14 UTC).
2. SMORE. Its registered default arms alias the HIGH end of the module's bracket (own branch +
   shared bilinear joint branch), so SMORE's "image" and "text" cells both measure the joint branch
   and identify no per-modality effect (as in the screen, footnote j). The LOW end and the joint arm
   are now recorded for every SMORE checkpoint (results/phase_c3conv/smore_bracket.json, reporting
   only); on Baby the low image end is within +/-0.0006 of zero on all 8 seeds.
3. AMEND1 A3 sensitivity analysis (exact-guard models only) was added to verdicts.json as the
   family "sensitivity_A3_exact_guard_only" after the Baby verification pointed out it was missing.
4. Interpretation flags (deletion raises accuracy; joint term; branch not content; non-exact
   guard) were added to each cell. They change no verdict.
