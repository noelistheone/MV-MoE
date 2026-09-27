# Erratum to PREREG_C3CONV_AMEND2.md

Recorded 2026-09-26 (UTC-7) after an independent read-only audit of AMEND2 and before any
repaired-port DAMRS cell could be judged. At recording time the repaired-port converged DAMRS runs
had finished for Baby seeds 2024-2030 only, and two of them had been scored by the scoring loop
(Baby s2024 and s2025; image-deletion deltas +0.002270 and +0.002290, text -0.001087 and -0.000042,
printed in results/phase_c3conv/_score_loop3.log). No repaired-port cell had 5 scored seeds, and no
repaired-port default-protocol (screen) run had started. None of the items below changes a decision
about the C3 cells; E3 and E4 change code before its first use.

E1. B1 states the wrong mechanism. Wrong: "dok_matrix no longer subclasses dict in scipy >= 1.14".
    Correct, verified in the installed scipy 1.17.1: dok_matrix still subclasses dict but keeps its
    entries in a private `_dict`, so `dict.update(A, data)` writes into the unused dict base and
    A.nnz stays 0 (two entries written: nnz 0, dict length 2, dense sum 0). The conclusion of B1 (a
    silent no-op that left the user-item graph empty) stands. The version threshold ">= 1.14" is
    not verified here and is withdrawn. The same wrong sentence sits in the upstream model file's
    comment, which is read-only for this workspace.
E2. B6 is wrong about the configuration of the original Baby screen checkpoint. Wrong: "the original
    screen checkpoint was a seed-2024 run at the same configuration". The Baby screen checkpoint
    (damrs_baby_20260616_100604.pt) was trained with train_batch_size 4096; the model yaml, the
    original floor runs and all repaired-port runs use 2048, as did the Sports, Clothing and
    MicroLens screen checkpoints. The repaired re-screen uses the yaml configuration (2048) on every
    dataset, i.e. the configuration of the floor runs it is judged against.
E3. B6 says the instruments are re-applied "unchanged". The re-screen driver
    (scripts/exp_damrs_g11_screen.py) as first written admitted a cell under the AMEND1 A5 relative
    reconstruction tolerance and did not suppress a cell whose guard failed. It was corrected before
    its first run: the screen's own rule applies (absolute reconstruction error <= 1e-5, else refused,
    as in exp_exact_crossarch.run_exact); the A5 result is recorded for information only; a cell
    failing the baseline or reconstruction guard is marked REFUSED and carries no comparison fields.
    The field "exact" now has the meaning it has in results/phase_exact/screen_vs_exact.json;
    "materially_misreported" is not produced because no saved script defines it; run_exact's
    verdict_reliability and attribution are recorded. The driver runs inside the scoring loop, so
    the loop adds at most one GPU process next to the five queue workers.
E4. Scorer (scripts/exp_c3conv_score.py), changed before any repaired cell was judged: (a) a record
    named damrs_g11 in the original run files is ignored, so only results/phase_c3conv_g11/ can
    supply that key; (b) repaired-port duplicates with different Recall@20 are excluded, as for every
    other model (previously the first record was kept).
E5. The runtime graph check of B3(b) is saved as
    results/phase_c3conv_g11/probe_graphs_all_models_baby.json (sha256 631835d7...; script
    scripts/probe_graphs_all_models.py).
E6. The stopped original-queue job (MicroLens DAMRS seed 2025) left the partial checkpoint
    results/phase0/_scratch/ckpts/pat100e3000_damrs_microlens_s2025.pt (last written 00:50:19). No
    run record points to it, so no scorer reads it. It is kept and not used.
E7. B2's account of the superseded notes is incomplete: the `random.sample` neighbour path those notes
    blamed is never executed (the helper `_random()` in the model file is defined but not called),
    so the kNN graphs do not depend on Python's RNG at all. The corrected notes say so.
