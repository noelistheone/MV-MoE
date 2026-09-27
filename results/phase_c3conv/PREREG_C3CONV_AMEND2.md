# Amendment 2 to PREREG_C3CONV.md: DAMRS port defect

Recorded 2026-09-26 (UTC-7), BEFORE any DAMRS run with the repaired port was trained under this
amendment. At recording time exactly one repaired-port DAMRS run had finished and been scored
(Clothing seed 2031, pooled by mistake with six defective-port seeds into one Clothing cell), and one
was in training (MicroLens seed 2025, original queue). No repaired-port cell had >= 5 seeds, so no
repaired-port verdict, floor or p-value could have been seen.

B1. Defect. The DAMRS port filled its user-item adjacency with `dict.update(A, data)` on a scipy
    `dok_matrix`. dok_matrix no longer subclasses dict in scipy >= 1.14 (this environment: 1.17.1),
    so the call was a silent no-op and the user-item (LightGCN) propagation graph was EMPTY; the
    item-item image, text and session graphs were unaffected. The model file (read-only for this
    workspace, maintained outside it) was repaired upstream at 2026-09-25 20:40:12 -0700; the
    repaired file has sha256 d001139f3cbafff1eac8e3a1c82019cbc96deb4fec7db56ae01bc51731a4f5ed. It
    builds A by COO assignment and asserts nnz(A) = nnz(norm_adj) = 2 nnz(train). Every DAMRS
    process started before that time ran the defective port:
      - C3 re-test: Baby s2024-2031, Sports s2024-2031, Clothing s2024-2030, MicroLens s2024;
      - default-protocol screen: the four seed-2024 screen checkpoints (Baby, Sports, Clothing,
        MicroLens) and the floor runs (Baby, Sports, Clothing, seeds 2024-2026);
      - test-time averaging comparison: the three DAMRS cells (the same screen checkpoints).
    The only repaired-port run from the original queue that finished is Clothing s2031 (its process
    started after 20:40:12; queue log).
B2. Earlier notes were wrong. NOTES_DAMRS_GRAPH.md attributed the refused Clothing s2030 row to
    rebuilt random kNN neighbours. Actual cause: s2030 was trained on the defective port and scored
    after the repair, so the scorer rebuilt a non-empty user-item graph under weights trained without
    one, and the exact baseline guard (G2) refused the row, as designed. The CPU probe reproduced
    neither s2029 nor s2030 for the same reason. The notes are corrected in place.
B3. Other architectures are not affected. (a) No other file of the imported model source tree was
    modified after 2026-09-20 (mtime listing). (b) All 13 architectures used in this project
    (FREEDOM, LGMRec, LightGCN, VBPR, MMGCN, LATTICE, BM3, MGCN, MENTOR, DAMRS, SMORE, COHESION,
    GUME) were built on Baby with the current code; every sparse graph each registers has nnz > 0.
    (c) Code reading: only DAMRS fills an adjacency by dict.update; COHESION's equivalent was
    repaired before this campaign started; SMORE and GUME fill theirs by lil assignment; the rest use
    the framework's COO builder.
B4. DECISION (primary analysis). DAMRS cells are computed ONLY from repaired-port runs trained by
    scripts/exp_damrs_g11.py, which calls the same trainer (train_one) with the same configuration
    and protocol (patience 100, cap 3000). Each record carries the model file's sha256 at start and
    at end; a run is admitted only if both equal the hash in B1. Seeds: 2024-2031 on Baby, Sports and
    Clothing; 2024-2028 on MicroLens (A6). Clothing s2031 is re-trained under the wrapper as well, for
    uniform provenance; its original record serves only as a reproducibility check. The MicroLens
    seed-2025 job the original queue started on the repaired port is stopped and re-queued under the
    wrapper, and seeds 2026-2028 are moved to the wrapper. Output: results/phase_c3conv_g11/
    runs_<ds>.json; checkpoints prefixed g11pat100e3000_ (no original checkpoint is overwritten).
    Instrument, guards, floors, verdict rule and families are unchanged; in every family the repaired
    DAMRS cells take the place of the defective ones.
B5. Defective-port results are never pooled with repaired-port seeds and never enter a family. They
    stay unmodified in their files and may be reported only as a labelled disclosure item ("DAMRS
    with an empty user-item graph").
B6. Screen and averaging. The DAMRS screen cells and floors are recomputed from repaired-port runs at
    the default protocol (patience 20, cap 1000; seeds 2024-2026; prefix g11nf_; output
    results/phase_c3conv_g11/screen_runs_<ds>.json). The screen checkpoint is the seed-2024 run, as
    the original screen checkpoint was a seed-2024 run at the same configuration. The exact deletion
    and test-time averaging instruments are re-applied unchanged to the repaired seed-2024 checkpoints
    on Baby, Sports and Clothing. MicroLens had no screen floor and is not re-screened.
B7. Void cells. The current Clothing DAMRS verdict (six defective seeds plus one repaired seed) and
    any MicroLens DAMRS verdict that uses the defective seed 2024 are void.
