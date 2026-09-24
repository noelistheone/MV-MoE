> **Redacted copy for double-blind review.** Redactions, each marked in place: local file
> paths; verbatim quotations of an earlier version of this work and of its reviews; a
> pointer to other work by the authors; the name of a method outside this submission.
> Everything else, including the date line, the rules, the predictions and every addendum,
> is verbatim.

# Pre-registration — train-time modality holdout (the ROAR control)

*Written 2026-09-04. Registers the analysis BEFORE the remaining seeds and datasets run.
Baby's 3 seeds per condition were already complete when this was written; that is disclosed
in §5 and is why the added statistic is registered here rather than claimed post hoc.*

## 1. Why

[Redacted for double-blind review: quotations from reviews of an earlier version of this work,
objecting that removing the image pathway after training cannot show what image contributed
while the model was trained.] The objection is right.

The paper's existing λ-sweep does **not** answer this. Setting `mm_image_weight = 0` removes
image from the item–item graph but **leaves the image auxiliary BPR loss running**
(`freedom.py:179-183`), so image gradients still shape the embeddings. λ=0 is a graph
ablation, not a training ablation.

## 2. Design

Three conditions, retrained from scratch, seeds paired:

| condition | image in graph | image in aux loss |
|---|---|---|
| `full` (λ=0.1) | yes | yes |
| `λ=0` (already in `results/phase2/weight_sweep_retrain.json`, same seeds) | no | **yes** |
| `no_image` (this study, `v_feat=None`) | no | no |

`full − no_image` = the **total** train-time image contribution.
`λ=0 − no_image` = the **auxiliary-loss-only** contribution.
`full − λ=0` = the **graph-only** contribution.
`no_text` (`t_feat=None`) is the symmetric positive control: if withholding text also costs
nothing, the cell has no power and is inadmissible.

Assertion enforced in code: under `no_image` the constructed model has no `v_feat` and no
`image_trs`; symmetrically for `no_text`.

## 3. Statistics — and why a second one is registered

The paper's standing rule is `MDE = 2·sd` against a per-(model,dataset) floor. That rule was
designed for an **inference-time** knockout, which is deterministic *within* a checkpoint, so
seed-to-seed variation of the delta is the whole uncertainty and a 2·sd band is the right
detection threshold.

A retrain-vs-retrain comparison is a different design: both arms are independent training
runs, and the quantity of interest is the **mean** paired difference, not whether a single
run would be detectable. For a mean, the standard test is the paired t-test, and unlike
`2·sd` its power **does** grow with seeds. Both are therefore pre-registered, with the
2·sd rule kept as the conservative primary so the paper's discipline is unbroken:

- **Primary (conservative):** the two-floor rule of `results/phase_micro/PREREG.md` §3 —
  SIGNIFICANT iff `|mean d| > max(F_paired, F_level)`; NULL iff below both; else **MARGINAL**,
  reported as MARGINAL and never rounded.
- **Secondary (registered here):** paired t-test on `d(s) = R@20_cond(s) − R@20_full(s)`,
  reported with t, df and p, and with the seed count that produced it.
- A disagreement between them is **reported as a disagreement**, not resolved by picking one.

## 4. Seeds

**8 seeds per condition** on Baby/Sports/Clothing/MicroLens (Elec deferred on cost). Eight
because the secondary statistic's power scales with n while the primary's band does not, and
because the `no_image` arm is empirically far noisier than `full` (see §5) — so the seed
count must be set by the noisier arm.

## 5. Full disclosure of what was already seen

Baby, 3 seeds per condition, was complete before this file was written:

| condition | per-seed R@20 (2024/2025/2026) | mean |
|---|---|---|
| full | 0.09245 / 0.09295 / 0.09271 | 0.09270 |
| no_image | 0.09094 / 0.08717 / 0.08803 | 0.08871 |
| no_text | 0.06719 / 0.07063 / 0.06919 | 0.06900 |

`no_image` paired Δ = **−0.00399**, F_paired 0.00443, F_level 0.00049 → **MARGINAL** under the
primary rule (0.90× F_paired but 8.07× F_level). Paired t = −3.12, df=2, **p ≈ 0.09** — also
not significant at n=3. `no_text` paired Δ = −0.02370 → SIGNIFICANT under both (8.0× F_paired,
t = −27.8), so the cell is admissible.

So the secondary statistic is being registered while the primary already reads MARGINAL and
the secondary already reads non-significant. It is registered because n=3 cannot decide
either way, not because it gives a better answer — at n=3 it gives the *same* answer.

**Registered directional prediction, before the extra seeds run:** if the `no_image` paired
sd stays near 0.0022, then at 8 seeds the primary rule will still read MARGINAL (the 2·sd
band does not shrink) while the secondary will reach t ≈ −5.1, p < 0.002. That disagreement
is the expected outcome and will be reported as such — the honest reading being *"withholding
image from training costs about 4% of Recall@20 on Baby on average, but the effect is smaller
than the run-to-run spread it induces, so no single training run is reliably distinguishable."*

## 6. A finding already visible in the disclosed data, registered as a hypothesis

`sd(no_image runs) / sd(full runs) = 7.9` on Baby (0.00197 vs 0.00025). **H-var: withholding
image destabilises training rather than only shifting its mean.** This is registered as a
hypothesis to be tested on the remaining datasets, not claimed from Baby alone. If it
replicates, it is the substantive result of this experiment: the image pathway acts partly as
a regulariser, which is invisible to any inference-time knockout and would explain why the
inference-time and train-time measurements differ by an order of magnitude in point estimate.

## 7. What this experiment can and cannot establish

- It CAN bound the total contribution of a modality over the whole of training, closing the
  objection that inference-time knockout misses training-time influence.
- It CANNOT attribute that contribution to a pathway: the retrained model differs from the
  original in every weight, so `full − no_image` mixes the modality's direct contribution
  with all downstream re-optimisation. The inference-time knockout remains the only
  *pathway-attributable* measurement. The two are complementary bounds and the paper must say
  so, rather than presenting the larger number as the "real" one.
- On Baby the point estimates differ by ~13× (inference-time −0.00029 vs train-time −0.00399).
  That ratio is itself the reportable quantity: it QUANTIFIES the paper's existing
  "lower bound" caveat instead of merely asserting it.
