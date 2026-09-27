# CORRECTION (2026-09-26): the diagnosis below is wrong

The refused Clothing s2030 row was not caused by rebuilt random kNN neighbours or by the device.
The DAMRS model file was repaired upstream at 2026-09-25 20:40:12 -0700 (sha256 d001139f...): its
user-item adjacency had been filled with `dict.update` on a scipy dok_matrix. In the installed scipy
(1.17.1) dok_matrix still subclasses dict but keeps its entries in a private `_dict`, so the call
writes into the unused dict base and nnz stays 0: a silent no-op. Every DAMRS run started before that
time therefore trained with an EMPTY user-item graph.
Seed 2030 was trained on that defective port and scored after the repair, so the scorer rebuilt a
non-empty user-item graph under weights trained without one; the exact baseline guard refused it,
as designed. Seed 2029 matched exactly because it was scored before the repair (same code for
training and scoring). The CPU probe reproduced neither seed because it ran after the repair.
The superseded text is wrong on a second count too: the `random.sample` neighbour path it blames is
never executed (the helper `_random()` is defined but not called), so the kNN graphs do not depend on
Python's RNG. Withdrawn: "DAMRS must never be scored on CPU". Consequences and the re-training decision:
PREREG_C3CONV_AMEND2.md (items B1-B7) and PREREG_C3CONV_AMEND2_ERRATUM.md. The original text is kept below for the record.

---

# (Superseded) DAMRS: the rebuilt item graphs are not always the graphs the model was trained with

Found while investigating the one refused DAMRS row (Clothing, seed 2030; recomputed baseline Recall@20
0.091475 vs trainer-logged 0.087171).

DAMRS builds its image and text kNN graphs at construction time and stores them as non-persistent
buffers, so every reload rebuilds them. Items with at most k nonzero similarities get neighbours from
Python's `random.sample`, and `torch.topk` breaks ties, so the rebuilt graphs can differ between
processes and devices. A read-only probe (CPU rebuild, validation and test Recall@20):

| seed | valid logged at best epoch | valid rebuilt (CPU) | test logged | test rebuilt (CPU) | GPU scorer baseline |
|---|---|---|---|---|---|
| 2030 | 0.087579 | 0.092173 | 0.087171 | 0.091475 | 0.091475 (refused) |
| 2029 | 0.086820 | 0.091443 | 0.084947 | 0.090471 | 0.084947 (exact match, accepted) |

So a CPU rebuild reproduces neither run, while the GPU scorer reproduced seed 2029 exactly and not
seed 2030. The registered exact baseline guard is what separates the two: every accepted DAMRS row
(Baby 8/8, Sports 8/8, Clothing 7/8) reproduced its trainer-logged Recall@20 exactly on GPU, i.e. was
scored with the graphs it was trained with; the one row that did not was refused. Consequences:
DAMRS must never be scored on CPU, and a DAMRS row may be accepted only through the exact guard.
Probe output: probe_damrs_clothing.json.
