> **Redacted copy for double-blind review.** Redactions, each marked in place: local file
> paths; verbatim quotations of an earlier version of this work and of its reviews; a
> pointer to other work by the authors; the name of a method outside this submission.
> Everything else, including the date line, the rules, the predictions and every addendum,
> is verbatim.

# Pre-registration — The exact image knockout on short-video (MicroLens)

*Written 2026-09-04, BEFORE any run in this study. Workspace `/workspace/MechInterp`.
All artifacts land in `results/phase_micro/`. `/workspace/Recsys` and every other
neighbouring workspace stay READ-ONLY.*

## 0. Why this study exists

[Redacted for double-blind review: a verbatim quotation of an earlier version of this work,
which named short-video data as its next test.]

MicroLens (short-video) currently appears in the paper only through [redacted: a method outside this submission] — a
constructive, correlational result. The causal instrument (exact structural knockout
against a measured noise floor) has never been pointed at it. This study points it there.

## 1. DISCLOSURE — this is a REPLICATION-grade pre-registration, not a blind one

An audit agent ran an unpersisted, in-memory dry run of the primary measurement on
2026-09-04 before this file was written. Its point estimates are recorded here IN FULL so
that no number below can be presented later as a blind prediction:

| quantity (dry run, unpersisted) | value |
|---|---|
| FREEDOM/MicroLens base R@20 | 0.09785 |
| image knockout dR@20 | -0.000848 |
| image knockout dN@20 | -0.000434 |
| text knockout dR@20 | -0.006861 |
| both knockout dR@20 | -0.007793 |
| reconstruction max err | 9.5e-7 |
| \|\|cf\|\| / \|\|h_img\|\| / \|\|h_txt\|\| | 7.810 / 0.451 / 4.527 |
| raw alignment gap: image / text / video | 0.1462 / 0.0518 / 0.0041 |
| propagated gap: h_img / h_txt / cf | 0.3170 / 0.2157 / 0.6323 |

**What is genuinely undetermined:** the verdict. No FREEDOM/MicroLens noise floor exists
anywhere on disk, so whether -0.000848 is significant is unmeasured. The nearest measured
MicroLens floor (GUME, 5 seeds, 2sd = 0.000725) puts the effect at 1.17x — a knife edge.
The floor, and therefore the verdict, is what this study measures.

Everything below is fixed before the floor exists.

## 2. Hypotheses

**H1 (primary, causal).** FREEDOM's exact structural image knockout on MicroLens is within
its own MicroLens noise floor, while the text knockout clears it.

**H2 (mechanism, directional).** The propagated behavioral-alignment ordering INVERTS on
MicroLens relative to all four Amazon datasets: `h_img > h_txt` with non-overlapping 95%
bootstrap CIs, while the knockout ordering `|d_img| << |d_txt|` is preserved.

If H1 and H2 both hold, the paper's C4 mechanism ("image is ignored because
`h_img << h_txt << CF` — image is the least co-purchase-predictive stream") **does not
transfer** and must be restated as an allocation account: use tracks the architecture's
fixed coefficient (`mm_image_weight = 0.1`, giving ||h_img|| = 0.451 vs ||h_txt|| = 4.527),
not behavioral informativeness. That is a correction to an earlier claim, and it is the
main scientific product of this study.

**H3 (retrain, escapes the constant).** The inference-time knockout cannot exceed what
lambda=0.1 allocates. Retraining FREEDOM at lambda in {0.0, 0.2, 1.0} tests whether the model
*could* learn to use image where the image kernel is the better co-consumption kernel.
Pre-registered reading: on Amazon, image-only retraining collapses on 4/4 (Baby 0.0691 vs
text-only 0.0914; Clothing 0.0603 vs 0.0916, i.e. ~24%). "Image-only FREEDOM does NOT
collapse relative to text-only on MicroLens" is a reachable falsifier of the generalized
architecture account.

## 3. Primary statistic and the two floors

Primary statistic: the per-seed paired delta, averaged over seeds,

    d_img(s) = R@20( cf + h_txt )  -  R@20( cf + h_img + h_txt )     [seed s's own checkpoint]

A structural knockout is deterministic *within* a checkpoint, so the sd of the per-seed
delta is the correct floor; seeds stabilize sigma-hat, they do not shrink MDE = 2*sd.

TWO floors, both fixed now, and the decision requires them to AGREE:

- `F_paired = 2 * sd_s[ d_img(s) ]`   — tight, the paired floor
- `F_level  = 2 * sd_s[ R@20(s) ]`    — the convention used throughout (Exp A1)

**Verdict rule.**
- SIGNIFICANT iff `|mean_s d_img| > max(F_paired, F_level)`
- NULL iff `|mean_s d_img| <` both
- MARGINAL otherwise — **reported as MARGINAL, never rounded to either.**

This two-floor rule exists to pre-empt floor-shopping, the exact failure Exp A1 was built
to fix. No floor may be borrowed from another dataset or another model.

**Admissibility gate.** The cell yields any verdict only if the text knockout clears both
floors, `|mean_s d_txt| > max(F_paired_txt, F_level)`. If image AND text are both null, the
model never learned to use content on MicroLens and the cell is uninterpretable — it is
reported as inadmissible, not as evidence for H1. (The dry run says text is 8.1x image, so
this is expected to pass; it stays pre-registered so it cannot be dropped later.)

Note the gate's known weakness, stated up front: text carries 0.9 of `mm_adj` by
construction, so a passing text control is partly mechanical. The informative quantity is
the RATIO |d_txt| / |d_img|, not the gate's pass/fail.

## 4. Seeds

8 FREEDOM seeds (2024-2031), 3 LGMRec seeds (2024-2026). Eight rather than three because
the effect sits within ~1.2x of the nearest measured floor: a 3-seed sigma (2 dof, ~52%
relative standard error) would let the verdict flip on sampling noise in the floor itself.

## 5. Secondary readouts (fixed now)

- Top-20 overlap and RBO, reported **with p = 0.9 and the 0.878 identical-list ceiling**
  stated (the paper currently omits both — a defect fixed here).
- A per-user paired continuous rank statistic: mean delta log2(rank) of the held-out item
  at K = 1000. MicroLens R@20 per user is near-binary, so the continuous statistic has
  orders of magnitude more effective n and is the sensitivity check on H1's null.
- Frozen image-weight sweep over w in {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}: the cheapest
  defense against "your null is an artifact of mm_image_weight = 0.1".

## 6. Declared negative side-cell (no claim will be built on it)

MicroLens ships `video_feat.npy` (17228, 768). We will measure and report its degeneracy
(effective rank, mean all-pairs cosine) as evidence that the *released* video features
cannot support a third-modality claim. This is a disclosure, not a result.

## 7. Scope limits fixed in advance

1. **No encoder-confound control is possible.** MicroLens ships no raw frames locally, only
   precomputed features whose provenance is a single URL. The CLIP/SigLIP re-encode that
   anchors the paper's C2 on Amazon cannot be replicated. We will state this, not work
   around it.
2. **FREEDOM is the worst of the trained baselines on MicroLens** (0.09785, 11.1% BELOW
   pure-CF LightGCN 0.11002 — the only one of five datasets where that inversion holds).
   This must be disclosed in the same paragraph as the result, answered with the frozen
   sweep and the H3 retrain, and not argued away.
3. **The raw co-consumption flip is a PREMISE, not a result of this study.** That image
   out-predicts text for co-consumption on MicroLens is already known from
   earlier work [redacted for double-blind review]. It is cited as
   the motivation for the test. What is new here is the *propagated-stream* inversion
   together with the *causal* null.
4. **Prior-art corrections to apply before writing** (verified by the audit round):
   arXiv:2608.05655 already runs MicroLens-100K with 5 seeds, paired t-tests and BH-FDR
   and reports a significant aggregate content gain — our image-null/text-significant
   prediction must be stated as CONSISTENT with it (text carries the gain), and this
   reconciliation is registered here, before the run. arXiv:2512.21863 (ICMR'26) reports
   raw video (0.0975) beating ID-only (0.0909) on MicroLens while cover-image (0.0862)
   loses to it — so the claim under test concerns the COVER-FRAME interface specifically,
   never "content is redundant on short-video". arXiv:2508.05377 already published that
   images tend to be more advantageous in short-video, so the raw inversion is not claimed
   as novel.

## 8. What the null looks like, and why it is the better branch

Null = image knockout inside both floors while text clears both, in a domain where the raw
image kernel is ~3x better aligned with co-consumption than text AND the propagated
ordering inverts. That dissociates modality *informativeness* from modality *use* in the
one domain where the "weak features" reading was strongest, converts the paper's single
largest stated limitation into a positive result, and forces C4 from "image is behaviorally
weak" to "use tracks the architecture's fixed allocation".

The one outcome that is NOT publishable is image-null AND text-null (inadmissible cell).

## 9. Audit discipline

After Step 1, Step 3 and Step 4, an independent READ-ONLY agent (agentType `Explore`,
report-only) must re-derive every reported number from raw JSON, verify loaded checkpoint
names against `ckpt_pins.json`, confirm nothing outside `/workspace/MechInterp` was
written, and check that no cell whose verdict is MARGINAL is described as significant.

---

# ADDENDUM 1 — 2026-09-04, after the Step-1 adversarial audit (append-only)

The independent read-only audit of Step 1 found six things this pre-registration got wrong
or left ambiguous. Corrections, in the audit's own order of severity. Nothing above is
edited; this section overrides it where they conflict.

## A1.1 The disclosure in §1 was incomplete in the way that matters most

§1 disclosed a dry run of **FREEDOM's** knockout and the alignment streams. It said nothing
about LGMRec — and LGMRec became the headline. Stated plainly, on the record:

> **The LGMRec/MicroLens knockout had NOT been run when §1 was written.** No LGMRec number
> appears in the §1 table because none existed. The image-dominant result
> (d_img = −0.006510 vs d_txt = −0.003059) was produced for the first time by the persisted
> Step-1 run. The FREEDOM branch is replication-grade; the LGMRec branch is blind.

## A1.2 The dry run's configuration is unknown, and its numbers do not exactly reproduce

The persisted run differs from the §1 table: `recon_max_err` 1.43e-6 (disclosed 9.5e-7) and
every alignment gap off by ~1% (raw video moved the other way: 0.0036 vs 0.0041 disclosed).
The persisted run pins `n_pairs=20000, pair_seed=0, boot_seed=1234, perm_seed=5678`. **The
dry run's configuration was not recorded and cannot be recovered.** The §1 numbers should
therefore be read as approximate prior knowledge, not as a checkable baseline. Only the
persisted artifacts are evidence.

## A1.3 H2 is not a prediction and must not be written as one

H2's direction (`h_img > h_txt`) was already visible in the dry run. What is registered is
**the decision rule** (non-overlapping 95% bootstrap CIs on the pre-specified Wang & Isola
gap) — not the direction. Any write-up saying "we predicted the inversion" is false.

## A1.4 The two-floor rule was unimplementable as coded — now fixed

`exp_mde_perdataset.py` computes only `F_level = 2·sd_s[R@20]`. `F_paired = 2·sd_s[d_img(s)]`
existed in no script, so the 12-GPU-hour floor run in flight would have delivered half the
pre-registered rule and invited exactly the single-floor shopping §3 bans. Step 3 must run a
per-seed knockout over the 8 FREEDOM seed checkpoints (now recorded by `train_one` via
`ckpt_path`) before any verdict is reported.

## A1.5 A zero-floor bug adjudicated an unjudged quantity

`phase2_weight_sweep.py` compared against `(mde or 0)`, so a missing floor silently became a
floor of **zero** and `weightsweep_microlens.json` shipped
`"image_helps_when_upweighted": true` for a +0.0004 effect with `"MDE_R@20": null`. Fixed to
return `None` when no floor exists. The affected artifact is regenerated below.

## A1.6 Registered corrections to the headline's framing

- **"C4 breaks" was an overclaim.** What fails to transfer to MicroLens is C4's *premise*
  (the ordering `h_img < h_txt`); C4's *conclusion* survives, including `h_img ≪ CF`
  (0.3195 vs 0.6342). The defensible statement is: **the mechanism's premise fails to
  transfer while its conclusion holds** — which is the more interesting claim anyway.
- **The alignment inversion does not evidence the allocation account.** The gap is computed
  on L2-normalized streams, so it is *rigorously invariant* to λ (`normalize(λx) = normalize(x)`
  for λ>0). That invariance is good — it means the inversion is not a λ artifact, and the
  Amazon comparison is apples-to-apples — but it also means the metric is blind by
  construction to the 0.4507-vs-4.5269 norm gap the allocation account rests on. The two
  measurements are logically orthogonal and must be reported as such.
- **The paper already claims the routing mechanism** (`main.tex:247, 375`: image is
  exploitable "only by routing it onto the inference path, as LGMRec does"). MicroLens is a
  **new-domain confirmation** of that claim plus the **first cell in which image dominates
  text at all**, not a correction to it.
- **"Text dominates in 8/8 Amazon cells"** is a magnitude claim stated as a contribution
  claim; freedom/baby text-KO is **+0.002533** (removing text *helps* R@20). Correct wording:
  **|d_txt| exceeds |d_img| in 8/8**.
- **The λ share was mislabelled.** The residual column (0.572–0.615 Amazon, 0.896 MicroLens)
  is the factor *not* explained by λ, not the percentage explained. Correct log-shares:
  λ accounts for **~80% of the log-suppression on Amazon and ~95% on MicroLens**.

## A1.7 Write-scope breach, disclosed rather than tidied away

Two Python bytecode caches were written outside the workspace today:
two bytecode caches in the read-only training-framework directory [file names redacted] (15:12), by
an audit agent importing those modules while checking the co-consumption premise. No source
or data was altered, and the timestamps precede every Step-1 artifact. They are **left in
place rather than deleted** — deleting is another write, and removing the evidence of a
breach is worse than the breach. All subsequent commands run under
`PYTHONDONTWRITEBYTECODE=1`.

## A1.8 New controls registered and run in response to the audit (Step 1b)

Three attacks were raised; all three are now measured, and the results are in
`controls_microlens.json` and `alignment_degreematched.json`:

1. **Allocation-vs-routing confound.** FREEDOM allocates image:text = 1:10 (λ=0.1); LGMRec
   allocates 1:1 by construction. Registered control: sweep the allocation of BOTH at
   inference and trace |d_img|/|d_txt|. Kill condition for the routing account: FREEDOM
   becomes image-dominant at some allocation, or LGMRec loses image-dominance at 1:9 by the
   same factor FREEDOM has.
2. **Gumbel-draw spread.** LGMRec's hypergraph resamples every forward (`F.gumbel_softmax`
   is a functional and is NOT disabled by `model.eval()`), so the flip rested on one draw.
   Registered control: 12 draws; report sd[|d_img|−|d_txt|]. Kill condition: the sd is
   comparable to the 0.00345 gap.
3. **Popularity confound in the alignment null.** Co-consumption pairs are sampled one per
   user from that user's history, so items enter them proportional to degree, while the null
   drew items uniformly. Registered control: re-draw the null from the empirical item-degree
   distribution, on MicroLens **and all four Amazon datasets** (this also re-tests the
   paper's cross-architecture table). Kill condition: any ordering changes.

## A1.9 PREREG §5 bullet 2 is now implemented

The per-user paired continuous rank statistic (mean Δlog2(rank) at K=1000) had no
implementation. It is now in `phase_micro_controls.py` and run. Note for the write-up: with
103,989 positives over 98,129 MicroLens users (~1.06 per user), per-positive and per-user
clustering are nearly equivalent here, so the naive SE is not materially inflated — unlike
on the Amazon datasets, where it would be.

---

# ADDENDUM 2 — 2026-09-04, after the Step-1b audit (append-only)

The second independent audit broke two claims outright and mis-specified a third. All are
corrected; the corrections cost the headline its strongest-sounding number and replaced it
with a smaller, true one.

## A2.1 The "7.1× more image reliance" claim is WITHDRAWN

At the matched ~1:9 allocation the ratio-of-ratios decomposes as

    |d_img| LGMRec/FREEDOM = 1.681×   (image effect,  26.5% of the log product)
    |d_txt| FREEDOM/LGMRec = 4.218×   (text  effect,  73.5% of the log product)

So it is **1.7× more image reliance and 4.2× less text reliance**, not "7.1× more image
reliance". Worse, at that point LGMRec relies on content **2.53× LESS in total**
(|d_img|+|d_txt| = 0.00305 vs FREEDOM's 0.00771) and is **5.24% off its own R@20**
(0.10529 → 0.09977), while FREEDOM sits **at its trained λ=0.1**. The comparison is
trained-vs-perturbed and the perturbed side supplies 73.5% of the effect — the confound and
the effect are the same thing. Withdrawn.

## A2.2 "MicroLens image is relatively far less bad" is WITHDRAWN

It compared MicroLens's **frozen** sweep against Amazon's **retrained** runs. Matched
frozen-vs-frozen (`frozen_imageonly_penalty.json`): baby 3.76%, elec 6.61%, **microlens
9.35%**, sports 12.13%, clothing 12.21%. MicroLens is **mid-range**, not an outlier. On
Amazon, retraining widens this gap 2.0–6.5×, so no claim about MicroLens image being
relatively less bad can be made before the Step-4 retrain.

## A2.3 "LGMRec allocates 1:1 by construction" is only half true

True of the MGE branch (`F.normalize(v) + F.normalize(t)`, unit norms to 7 digits). The GHE
branch has `‖av‖ = 832.8` vs `‖at‖ = 530.4` — an **effective 1.570:1 image tilt** — and
because it is `α·F.normalize(w_i·av + w_t·at)`, weighting there changes **direction only,
never magnitude**. The `w_i + w_t = 2` constraint also does not hold content mass fixed:
total content norm is 1.905 at balance and 2.100 at both endpoints (±10.2%). Every
allocation-matching claim is qualified accordingly.

## A2.4 The degree-matched null was mis-specified; corrected and re-run

`copurchase_pairs` draws ONE pair per eligible user uniformly within that user's history, so
the induced item marginal is

    p*(i) ∝ Σ_{u : i ∈ H_u, |H_u| ≥ 2} 1/|H_u|      NOT   deg(i)

The two differ (TV 0.057–0.078); raw degree over-tilts MicroLens's top-1% mass by 1.29×.
`exp_alignment_stats.copurchase_item_marginal()` now computes p* exactly, and the null was
re-run **on the paper's canonical pinned checkpoints**. Every ordering holds.

## A2.5 Artifacts were split across two checkpoint generations

`ckpt_pins.json` covered only MicroLens, so the Amazon cells resolved through
`latest_ckpt()` — which today returns **July 2026 re-runs**, while the paper's
`results/phase1` table was computed on **May/June** checkpoints. One document was describing
two different models. All 12 Amazon cells are now pinned to the paper's canonical
checkpoints and the affected artifact regenerated.

## A2.6 The rank statistic now has the null it needed — and passes

With n ≈ 10⁵ positives, *any* non-orthogonal additive stream gives a large t, so t = 30.2 on
h_img established only `h_img ≠ 0`. Two matched-size nulls, both preserving h_img's exact
per-item norms:

| stream | mean Δlog₂(rank) | t |
|---|---|---|
| **real h_img** | **+0.01188** | **+30.2** |
| h_img with rows permuted across items | −0.00137 | −4.3 |
| isotropic Gaussian at matched norms | −0.00071 | −2.8 |

Both nulls are **negative** (removing them slightly helps) and 8.7–16.8× smaller. The effect
is content-specific, not a size artifact. Also corrected: the sign convention was written
backwards in the artifact strings (positive = the stream was helping), and the censoring
disclosure quoted the OR count (49,523) where only the **AND** count (48,572 = 46.7%)
contributes exactly zero.

## A2.7 The MDE seed guard is a reliability policy, not a refusal

A first attempt refused any floor with n < 5 — which would have invalidated the paper's own
LGMRec floors used in the paper (n = 3, effects 4.4–12.8× MDE, entirely safe). MDE = 2·sd does not
shrink with n; small n means an *imprecise* floor (sd rel-SE ≈ 71%/50%/35%/27% at n =
2/3/5/8), which is dangerous only near the boundary. `verdict_reliability()` now marks a
verdict UNRELIABLE only when the effect is borderline (|d|/MDE ∈ [0.5, 2]) **and** the floor
is thin (n < 5). This preserves every reported verdict and blocks exactly the failure the
in-flight 8-seed run was registered to avoid.

## A2.8 The audit's alternative single-factor reading is PARTLY CORRECT, and is adopted

The audit proposed that the flip needs no routing account at all — LGMRec's *text* stream may
simply be weak on MicroLens. Tested directly (`lgmrec_crossdomain.json`, knockout as % of
each model's own base R@20, identical decomposition on all five):

| dataset | image % of base | text % of base | ratio |
|---|---|---|---|
| baby | 8.65% | 11.95% | 0.724 |
| sports | 4.52% | 12.55% | 0.360 |
| clothing | 2.97% | 13.75% | 0.216 |
| elec | 0.10% | 7.91% | 0.013 |
| **microlens** | **6.18%** | **2.91%** | **2.128** |

MicroLens text (2.91%) is **below the entire Amazon range** (7.91–13.75%), while MicroLens
image (6.18%) is **inside** it (0.10–8.65%) and below Baby's 8.65%. **The flip is driven more
by text collapsing than by image rising** — 0.25× the Amazon mean for text versus 1.52× for
image. The registered framing changes accordingly: this is not "short-video makes image
matter"; it is "on short-video the text stream stops carrying, and the on-path model's
ordering follows". (Consistent with the content side: MicroLens raw text has the weakest
alignment gap of all five datasets, 0.0417, and MicroLens raw image the strongest, 0.1156.)

## A2.9 A tracking claim was tested and NOT established — recorded so it is not retried

Tempting follow-on: "on-path architectures track content informativeness; frozen-graph ones
do not." Tested as the log-log correlation across the 5 datasets between each architecture's
causal image/text ratio and the raw-content alignment image/text ratio
(`tracking_correlation.json`): **LGMRec r = +0.73 (Spearman +0.70), FREEDOM r = +0.48
(Spearman +0.50)**. With n = 5 neither is significant and they do not separate. **Not
claimed.**

## A2.10 What the headline is now

Withdrawn: the 7.1× routing factor; "image relatively less bad"; and the
"between-architecture dissociation" as the primary frame (FREEDOM's exact additive deletion
and LGMRec's delete-then-renormalise are different operations, so the ten-cell grid is not
homogeneous).

What is supported, and it is still the paper's own next test:

1. **FREEDOM does not move.** Across a domain shift that measurably changes which content
   modality is informative (raw alignment ratio 0.22–0.76 on Amazon → 2.77 on MicroLens),
   FREEDOM's causal image/text ratio stays at 0.124, inside its Amazon range — and **no
   inference-time allocation makes it image-dominant** (max 0.989 at λ=0.9, where the image
   delta is *positive*, i.e. image is actively harmful).
2. **λ accounts for ~80% of the log norm-suppression on Amazon and ~95% on MicroLens.**
3. **The mechanism's premise does not transfer** (`h_img > h_txt` on MicroLens, robust to the
   corrected null and canonical checkpoints) **while its conclusion holds** (`h_img ≪ CF`).
4. **"Ignored" is a top-20 resolution statement, not a zero**: at K = 1000 the image stream
   carries a real, content-specific effect (+0.83% mean rank degradation, t = 30.2, against
   matched-size nulls at −0.10%/−0.05%).
5. **Within LGMRec** (identical decomposition, 5 domains) MicroLens is the only
   image-dominant cell, robust over 12 gumbel draws — **driven primarily by text collapse.**
