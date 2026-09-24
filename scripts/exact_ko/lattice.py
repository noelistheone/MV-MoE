"""Exact structural knockout for LATTICE (Zhang et al., MM 2021).

SOURCE OF TRUTH: /workspace/Recsys/src/models/lattice.py  (READ-ONLY, read line by line)

==============================================================================
1. What the architecture actually computes at inference
==============================================================================
`full_sort_predict` -> `_propagate()`, verbatim:

    A_item = sparse_row_topk( coalesce(                                      # <- content
                 w_v(1-lam) * K(image_trs(v_feat))  +  w_v*lam * K(v_feat)
               + w_t(1-lam) * K(text_trs(t_feat))   +  w_t*lam * K(t_feat) ), k)
    h      = A_item @ item_id_embedding.weight                    (n_layers = 1)
    ego    = [user_embedding ; item_id_embedding]
    out    = mean_{l=0..n_ui_layers} (norm_adj^l @ ego)           (LightGCN, pure ID/CF)
    u, i_cf = split(out)
    i      = i_cf + F.normalize(h, p=2, dim=-1)                   # <-- THE content term
    score  = u[users] @ i.T

with `w = softmax(modal_weight)` and `K(.) = build_knn_graph(., knn_k)` = a binary,
symmetrised, D^-1/2-normalised top-k cosine graph.

==============================================================================
2. Six structural facts that decide what a knockout is here
==============================================================================
F1  THE CONTENT PATHWAY IS EXACTLY ONE ADDITIVE TERM: `F.normalize(h)`.  No content
    vector is ever added to, concatenated with, or gated into the scored embedding; the
    features only ever choose the EDGES of an item-item graph that propagates the ID
    table.  Deleting `F.normalize(h)` outright leaves `u @ i_cf.T` -- a pure LightGCN.
    That is `both_knockout`, and it is STRICTLY EXACT (no survivor, nothing to
    renormalise, no shared parameter touched).

F2  PER-MODALITY DELETION IS NOT EXACT, because of the row-wise `F.normalize(h, dim=-1)`.
    In the baseline EVERY item's content term has norm exactly 1.000; after deleting one
    modality's two graph terms the survivor is renormalised back to norm exactly 1.000.
    So the model's own single-modality counterfactual changes only the DIRECTION of the
    content term and removes ZERO magnitude.  Per CONTRACT rule 3 this is
    `EXACTNESS = "DELETE_PLUS_RENORM"` and both ends are emitted:
        `<m>_knockout`         h_surv / ||h_surv||   (renormalised; the model's own
                                                      counterfactual, == LATTICE built
                                                      with that feature absent)
        `<m>_knockout_nonorm`  h_surv / ||h_full||   (baseline denominator kept, i.e. only
                                                      the mass actually removed is gone)

F3  THE TOP-K IS A SECOND SHARED SELECTOR, on top of the normaliser.  `sparse_row_topk`
    keeps 10 edges per row out of the 4-way weighted sum, so deleting a modality lets the
    SURVIVING modality re-fill the freed slots.  `<m>_knockout` (F2) allows that re-fill --
    it is what the architecture itself would compute.  `<m>_knockout_fixsupport` freezes
    the baseline's selected 70,500 edges and only subtracts the deleted modality's
    contribution to them, so no re-selection happens.  Items whose entire selected support
    came from the deleted modality end with an all-zero row, `normalize(0) = 0`, and lose
    the content term outright -- the only channel through which a per-modality knockout can
    remove magnitude rather than merely rotate.

F4  THE SOFTMAX FUSION WEIGHT IS RANK- AND VALUE-NEUTRAL UNDER DELETION.  With one modality
    gone the surviving two parts are both scaled by the same positive `w_m`; a global
    positive scale does not change `sparse_row_topk`'s selection and is annihilated by
    `F.normalize(h)`.  So the "renormalise the softmax over survivors?" question -- which
    the source resolves one way (`w_idx = 0` when `v_feat is None`) and a naive reader the
    other (keep `w_idx = 1`) -- is VACUOUS.  Certificate:
    `attribution()["softmax_reindex_max_delta_normalized_h"]`.

F5  `image_trs` / `text_trs` ARE FROZEN AT INITIALISATION.  Their only use site is
    `img_proj = self.image_trs(self.v_feat).detach()` (lattice.py:96, and :104 for text) --
    the output is DETACHED, nothing else reads them, and `weight_decay = 0.0`, so they
    receive exactly zero gradient for the whole run.  Measured: the checkpoint's
    `image_trs.*` / `text_trs.*` are BITWISE IDENTICAL to a fresh seed-2024 init
    (max|ckpt - fresh init| = 0.000e+00 for all four tensors).  LATTICE's advertised
    "learned item-item graph" is therefore a graph built from a FROZEN RANDOM PROJECTION of
    the raw feature.  It is still on the INFERENCE path (it is rebuilt from live features
    inside `_rebuild_item_adj`), so it is ablated together with its modality -- but it is
    not learned.  `modal_weight` DOES move (2.36e-3 from init) because it keeps gradient
    through the first batch of each epoch.

F6  CONTENT IS RE-READ AT INFERENCE.  `_item_adj` is a plain attribute (not a buffer), so
    it is None after `__init__` and `_build_item_graph` is True; the first
    `full_sort_predict` after a checkpoint reload REBUILDS the fused graph from the live
    `v_feat`/`t_feat`.  LATTICE is therefore graph-mediated like FREEDOM (C1) but with a
    LIVE rather than frozen content interface.

==============================================================================
3. What the input-mean screen actually measured on LATTICE
==============================================================================
The screen (`scripts/phasex_crossarch_knockout.py`) re-instantiates the model with
`v_feat := per-dim mean`.  Every item then has the SAME image vector, so every off-diagonal
cosine in `K(v_mean)` is the same number and `torch.topk` breaks the total tie by index:
each item is wired to the ten lowest-numbered items that are not itself.  After
symmetrisation those ten items become degree-7050 hubs and D^-1/2 drives their edge weights
to ~1/sqrt(10*7050).  The screen therefore does NOT delete the image graph; it REPLACES it
with a degenerate low-id hub graph whose weights are small enough to mostly lose the fused
top-k -- an uncontrolled mixture of "delete" and "inject garbage".  `*_screen_meanfeat`
reproduces it inside this module so the published number and the exact arms are measured
under ONE evaluation protocol.  (Unlike DAMRS, the screen is not bit-identical-broken here:
it does not make image_knockout == both_knockout.)

==============================================================================
4. Content-interface taxonomy
==============================================================================
Legend as fixed in scripts/exact_ko/mmgcn.py (the repo has no other definition):
    C1  content-derived item-item graph propagating the ID table; no content vector is an
        argument of the scoring function                                    <-- LATTICE
    C2  live content feature projected and ADDED to the scored embedding (VBPR, MENTOR)
    C3  content as the layer-0 node feature of a per-modality GNN tower (MMGCN)
    C4  content only in training-time objectives; off the inference path
    C5  ID-only: content never reaches the scoring function
LATTICE is C1 by interface, with the FREEDOM/DAMRS "frozen" qualifier removed (F6).

==============================================================================
5. Measured on baby (2026-09-04), ckpt lattice_baby_20260523_002958.pt
==============================================================================
Certificates
    recon_error                       2.9e-06  (model's own self-floor 2.4-3.8e-06; 1.4 ULP
                                                at max|score| = 17.32; cusparse mm is not
                                                bit-reproducible, so this IS the floor)
    top-20 agreement with full_sort_predict   1.000000 over 4096 users
    baseline R@20 - logged R@20      -1.4e-17   (0.08552953332202495 vs ...497)
    baseline - PUBLISHED screen baseline       0.0 on R@10/R@20/N@10/N@20 (bitwise)
    *_screen_meanfeat vs input-mean screen dR@20  abs diff 0.0 on all three arms
    image_knockout  vs LATTICE instantiated with v_feat=None:  max|score| 1.9e-06,
                    top-20 set agreement 1.000000 over 4096 users
    text_knockout   vs LATTICE instantiated with t_feat=None:  max|score| 2.9e-06,
                    top-20 set agreement 1.000000
    max |u_arm - u_baseline| over all 12 arms  0.0 (surviving pathway bitwise untouched)
    softmax_reindex_max_delta (F4)    1.5e-07 image / 0.0 text

dR@20 / dN@20 (baseline R@20 0.085530, N@20 0.037325; screen noise hint 0.0026)
    image_knockout              +0.000523 / +0.000124     <- headline, model's own cf
    image_knockout_nonorm       -0.001528 / -0.001215
    image_knockout_fixsupport   -0.001221 / -0.000589
    image_screen_meanfeat       +0.000498 / +0.000106
    text_knockout               -0.002915 / -0.001269     <- headline
    text_knockout_nonorm        -0.003444 / -0.001397
    text_knockout_fixsupport    -0.002318 / -0.001354
    text_screen_meanfeat        -0.002967 / -0.001273
    both_knockout (EXACT)       -0.000365 / -0.000712     <- whole content pathway
    both_screen_meanfeat        -0.000416 / -0.000765
    control_content_permuted    -0.006579 / -0.002605     <- content scrambled, not removed

WHAT THE NUMBERS SAY (read F2 first)
    Deleting the ENTIRE content pathway costs -0.000365 R@20 -- 14% of the screen's own
    noise hint. Deleting TEXT ALONE costs -0.002915, i.e. EIGHT TIMES MORE than deleting
    all content. That ordering is impossible for a quantity that measures "how much
    information this modality supplies", and it is the direct consequence of F2: a
    single-modality arm cannot remove the content term, it can only ROTATE it, and the
    model is more damaged by a wrong unit-norm content term than by none at all. The
    rotation scale confirms it end to end:
        cos(content_full, content_arm)   1.000 -> 0.0        (baseline)
                                         0.685 -> +0.000523  (image_knockout)
                                         0.494 -> -0.002915  (text_knockout)
                                         0.008 -> -0.006579  (control_content_permuted)
                                         term deleted -> -0.000365 (both_knockout)
    dR@20 is monotone in the rotation angle and is NOT explained by information loss.
    So on LATTICE the only knockout number that supports a causal claim about content is
    `both_knockout`, and it says the content pathway is worth ~0 at test time; the
    per-modality numbers are rotation artifacts and must be reported as the F2 bracket.

    The input-mean screen is NOT structurally broken here (unlike DAMRS). It reproduces to
    abs diff 0.0, and the reason is measurable: only 204 of the 70,500 fused edges come
    from the injected degenerate hub graph, so the screen's fused graph has Jaccard 0.9997
    with exact deletion. That is an accident of D^-1/2 shrinking the hub weights below the
    fused top-k threshold, not a property of the protocol -- and it still inherits F2,
    so the screen's per-modality numbers are rotation artifacts for the same reason.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F

ROOT = Path("/workspace/MechInterp")
RECSYS = Path("/workspace/Recsys")
for _p in (str(RECSYS), str(ROOT / "src" / "models"), str(ROOT / "src" / "interp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.data.graph_utils import build_knn_graph, sparse_row_topk   # noqa: E402

CLASS = "C1"
CLASS_DESC = ("C1 = content chooses the EDGES of an item-item graph that propagates the ID "
              "embedding table; no content vector is ever an argument of the scoring "
              "function. Unlike FREEDOM/DAMRS the graph is rebuilt from LIVE features at "
              "inference (F6), and its 'learned' half is a frozen random projection (F5).")
EXACTNESS = "DELETE_PLUS_RENORM"   # per-modality arms: row-wise F.normalize(h) (F2).
                                   # `both_knockout` is strictly EXACT (F1).

CKPT_PIN = {
    # results/phase_micro/ckpt_pins.json has no lattice entry; these are the checkpoints the
    # published cross-architecture screen used (latest_ckpt == the only run per dataset).
    # This module does NOT write to ckpt_pins.json -- reported for the driver to add.
    "baby":     "/workspace/Recsys/ckpts/lattice_baby_20260523_002958.pt",
    "sports":   "/workspace/Recsys/ckpts/lattice_sports_20260523_020020.pt",
    "clothing": "/workspace/Recsys/ckpts/lattice_clothing_20260523_051600.pt",
    "elec":     "/workspace/Recsys/ckpts/lattice_elec_20260531_050020.pt",
}

ARMS = {
    "baseline": "model as trained, re-derived from parts (asserted == full_sort_predict)",
    "image_knockout": "HEADLINE. The two image graph terms are deleted from the fused sum; "
                      "the graph is re-top-k'd and h renormalised. Exactly what LATTICE "
                      "computes when v_feat is absent. Renormalised end of the F2 bracket.",
    "image_knockout_nonorm": "same surviving graph, baseline denominator ||h_full|| kept -- "
                             "only the mass actually removed is gone (F2 mass end).",
    "image_knockout_fixsupport": "baseline's 70,500 selected edges frozen; only the image "
                                 "contribution to them is subtracted, so the text graph "
                                 "cannot re-fill the freed slots (F3).",
    "image_screen_meanfeat": "reproduction of the PUBLISHED coarse screen (v_feat := per-dim "
                             "mean) inside this module's evaluation protocol.",
    "text_knockout": "as image_knockout, text terms deleted",
    "text_knockout_nonorm": "as image_knockout_nonorm, text terms deleted",
    "text_knockout_fixsupport": "as image_knockout_fixsupport, text terms deleted",
    "text_screen_meanfeat": "input-mean screen, text",
    "both_knockout": "EXACT. The whole content term F.normalize(h) is deleted -> pure "
                     "LightGCN u @ i_cf.T. No survivor, so *_nonorm coincides with it and "
                     "is not emitted as a duplicate arm.",
    "both_screen_meanfeat": "input-mean screen, both modalities",
    "control_content_permuted": "CONTROL, not a knockout. The content term is kept at norm "
                                "1.000 but its rows are permuted (fixed seed 0), destroying "
                                "all content while preserving the additive structure the "
                                "renormalised arms preserve. Calibrates whether a "
                                "single-modality dR@20 is a content effect or just the "
                                "size of ANY rotation of a unit-norm content term.",
}
BRACKET = {
    # (mass-removed end, renormalised end) -- the true single-modality effect lies between
    "image_renorm": ("image_knockout_nonorm", "image_knockout"),
    "text_renorm": ("text_knockout_nonorm", "text_knockout"),
    # (no re-selection, model's own re-selection) -- the top-k selector's contribution
    "image_reselect": ("image_knockout_fixsupport", "image_knockout"),
    "text_reselect": ("text_knockout_fixsupport", "text_knockout"),
}
EXACT_ARMS = ("baseline", "both_knockout")


# --------------------------------------------------------------------------- graph algebra
def _feat(model, m: str) -> torch.Tensor:
    return model.v_feat if m == "v" else model.t_feat


def _trs(model, m: str):
    return model.image_trs if m == "v" else model.text_trs


def _orig(model, m: str) -> torch.Tensor:
    return model._orig_image_adj if m == "v" else model._orig_text_adj


@torch.no_grad()
def _parts(model, mods: Iterable[str] = ("v", "t"),
           feat_override: Optional[Dict[str, torch.Tensor]] = None,
           native_widx: bool = False):
    """The list of (sparse_graph, scalar_weight) that `_rebuild_item_adj` concatenates.

    Order is preserved exactly (learned_v, orig_v, learned_t, orig_t) so that `coalesce`
    sums the values in the same order as the model does.

    native_widx=True reproduces the source's own re-indexing when a modality is absent
    (`w_idx = 1 if self.v_feat is not None else 0`); False keeps each modality's original
    softmax slot. F4 certifies the two are equivalent.
    """
    feat_override = feat_override or {}
    weight = F.softmax(model.modal_weight, dim=0)
    dev = model.modal_weight.device
    lam = model.lambda_coeff
    present = [m for m in ("v", "t") if _feat(model, m) is not None and m in mods]
    parts = []
    for m in present:
        w_idx = present.index(m) if native_widx else {"v": 0, "t": 1}[m]
        w = weight[w_idx]
        f = feat_override.get(m)
        if f is None:
            learned = build_knn_graph(_trs(model, m)(_feat(model, m)).detach(), model.knn_k)
            orig = _orig(model, m)
        else:                       # screen: BOTH graphs are rebuilt from the fake feature
            learned = build_knn_graph(_trs(model, m)(f).detach(), model.knn_k)
            orig = build_knn_graph(f, model.knn_k)
        parts.append((learned.to(dev), w * (1 - lam)))
        parts.append((orig.to(dev), w * lam))
    return parts


@torch.no_grad()
def _coalesced_sum(parts):
    idx = torch.cat([sp.indices() for sp, _ in parts], dim=1)
    val = torch.cat([sp.values() * w for sp, w in parts])
    return torch.sparse_coo_tensor(idx, val, parts[0][0].shape).coalesce()


@torch.no_grad()
def _fuse(model, parts) -> torch.Tensor:
    """Byte-faithful replay of `_rebuild_item_adj`."""
    return sparse_row_topk(_coalesced_sum(parts), model.knn_k)


@torch.no_grad()
def _restrict_to_support(sp_sum: torch.Tensor, support_idx: torch.Tensor,
                         n: int) -> torch.Tensor:
    """Keep only entries of `sp_sum` that lie on `support_idx`, with the support's zeros.

    Used by the fixsupport arm (F3): the baseline's selected edge set is frozen and the
    surviving modality's value is read off it, so no re-selection can occur. Entries of the
    support with no surviving contribution stay at exactly 0.
    """
    zero = torch.zeros(support_idx.shape[1], dtype=sp_sum.values().dtype,
                       device=sp_sum.values().device)
    comb = torch.sparse_coo_tensor(
        torch.cat([sp_sum.indices(), support_idx], dim=1),
        torch.cat([sp_sum.values(), zero]), sp_sum.shape).coalesce()
    lin_c = comb.indices()[0] * n + comb.indices()[1]
    lin_s, _ = torch.sort(support_idx[0] * n + support_idx[1])
    pos = torch.searchsorted(lin_s, lin_c).clamp_max(lin_s.numel() - 1)
    keep = lin_s[pos] == lin_c
    return torch.sparse_coo_tensor(comb.indices()[:, keep], comb.values()[keep],
                                   sp_sum.shape).coalesce()


@torch.no_grad()
def _cf(model) -> Tuple[torch.Tensor, torch.Tensor]:
    """LightGCN user-item propagation -- the surviving pathway. Never modified."""
    ego = torch.cat([model.user_embedding.weight, model.item_id_embedding.weight], dim=0)
    all_embs = [ego]
    for _ in range(model.n_ui_layers):
        ego = torch.sparse.mm(model.norm_adj, ego)
        all_embs.append(ego)
    out = torch.stack(all_embs, dim=1).mean(dim=1)
    return torch.split(out, [model.n_users, model.n_items], dim=0)


@torch.no_grad()
def _h(model, item_adj: torch.Tensor) -> torch.Tensor:
    h = model.item_id_embedding.weight
    for _ in range(model.n_layers):
        h = torch.sparse.mm(item_adj, h)
    return h


@torch.no_grad()
def _assemble(model, h: Optional[torch.Tensor], denom: Optional[torch.Tensor] = None,
              cf: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
    """i = i_cf + content_term. h=None deletes the content term outright (F1).

    `cf` is the SHARED surviving pathway, computed once per variants() call and reused by
    every arm. Reusing it (rather than recomputing per arm) is what makes "the surviving
    pathway is bitwise identical in every arm" a real certificate: `torch.sparse.mm` on
    CUDA is not bit-reproducible, so recomputing it would leave a ~1e-7 pseudo-difference
    that is a hardware artifact, not an intervention.
    """
    u, i_cf = _cf(model) if cf is None else cf
    if h is None:
        return u, i_cf
    term = F.normalize(h, p=2, dim=-1) if denom is None else h / denom
    return u, i_cf + term


# --------------------------------------------------------------------------- contract API
@torch.no_grad()
def variants(model, dataset=None, device=None) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    n = model.n_items
    full_parts = _parts(model)
    A_full = _fuse(model, full_parts)
    h_full = _h(model, A_full)
    denom_full = h_full.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    cf = _cf(model)          # the surviving pathway: computed ONCE, shared by every arm

    out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {
        "baseline": _assemble(model, h_full, cf=cf)}

    # per-dim input-mean features, exactly as the input-mean screen builds them
    means = {m: _feat(model, m).mean(0, keepdim=True).expand_as(_feat(model, m)).contiguous()
             for m in ("v", "t") if _feat(model, m) is not None}

    for mod, m in (("image", "v"), ("text", "t")):
        if _feat(model, m) is None:
            continue
        surv = tuple(x for x in ("v", "t") if x != m)
        # --- headline: the model's own counterfactual (delete the terms, re-top-k, renorm)
        A_ko = _fuse(model, _parts(model, mods=surv))
        h_ko = _h(model, A_ko)
        out[f"{mod}_knockout"] = _assemble(model, h_ko, cf=cf)
        out[f"{mod}_knockout_nonorm"] = _assemble(model, h_ko, denom=denom_full, cf=cf)
        # --- no re-selection end (F3)
        A_fix = _restrict_to_support(_coalesced_sum(_parts(model, mods=surv)),
                                     A_full.indices(), n)
        out[f"{mod}_knockout_fixsupport"] = _assemble(model, _h(model, A_fix), cf=cf)
        # --- input-mean screen
        out[f"{mod}_screen_meanfeat"] = _assemble(
            model, _h(model, _fuse(model, _parts(model, feat_override={m: means[m]}))), cf=cf)
        del A_ko, h_ko, A_fix

    # --- EXACT: the whole content term is deleted (F1)
    out["both_knockout"] = _assemble(model, None, cf=cf)
    out["both_screen_meanfeat"] = _assemble(
        model, _h(model, _fuse(model, _parts(model, feat_override=means))), cf=cf)
    # CONTROL (see ARMS): same additive structure, same unit norm, zero content.
    g = torch.Generator(device="cpu").manual_seed(0)
    perm = torch.randperm(n, generator=g).to(h_full.device)
    out["control_content_permuted"] = _assemble(model, h_full[perm], cf=cf)
    return out


@torch.no_grad()
def attribution(model, dataset=None, device=None) -> dict:
    n = model.n_items
    lam = model.lambda_coeff
    w = F.softmax(model.modal_weight, dim=0)
    full_parts = _parts(model)
    A_full = _fuse(model, full_parts)
    h_full = _h(model, A_full)
    u, i_cf = _cf(model)
    content = F.normalize(h_full, p=2, dim=-1)
    i_full = i_cf + content

    a: dict = {
        "interface": "content -> item-item graph edges -> propagates ID table (C1, live)",
        "fusion_coefficients": {
            "modal_weight_raw": [float(x) for x in model.modal_weight.detach().cpu()],
            "softmax(modal_weight)": {"image": float(w[0]), "text": float(w[1])},
            "lambda_coeff (orig-graph share)": float(lam),
            "effective_part_weights": {
                "image_learned": float(w[0] * (1 - lam)), "image_orig": float(w[0] * lam),
                "text_learned": float(w[1] * (1 - lam)), "text_orig": float(w[1] * lam)},
            "knn_k": int(model.knn_k), "n_layers(item graph)": int(model.n_layers),
            "n_ui_layers(LightGCN)": int(model.n_ui_layers)},
        # --- additive streams of the scored item vector
        "streams": {
            "||i_cf|| (LightGCN, surviving pathway)": float(i_cf.norm(dim=-1).mean()),
            "||content term|| = ||normalize(h)||": float(content.norm(dim=-1).mean()),
            "||content term|| min": float(content.norm(dim=-1).min()),
            "||i_total||": float(i_full.norm(dim=-1).mean()),
            "||u||": float(u.norm(dim=-1).mean()),
            "content_share_of_item_norm": float(content.norm(dim=-1).mean()
                                                / i_full.norm(dim=-1).mean()),
            "cos(i_cf, content term)": float(F.cosine_similarity(i_cf, content).mean()),
            "||h|| before normalize (mean/min)": [float(h_full.norm(dim=-1).mean()),
                                                 float(h_full.norm(dim=-1).min())],
        },
        # The renormalised arms can only ROTATE the content term (F2). This is the rotation
        # scale: pair each arm's cos(content_full, content_arm) with its dR@20 to see whether
        # the measured effect tracks rotation angle rather than information loss.
        "rotation_scale": {
            "baseline": 1.0,
            "control_content_permuted": float(F.cosine_similarity(
                content, content[torch.randperm(
                    n, generator=torch.Generator(device="cpu").manual_seed(0)
                ).to(content.device)]).mean()),
            "both_knockout (term deleted, no direction)": None,
            "note": "per-modality values live in "
                    "per_modality[*]['cos(content_full, content_ko)']"},
        # --- F5: the 'learned' projections never trained
        "F5_trs_frozen_at_init": _trs_certificate(model, dataset),
    }

    # --- F2/F3 per-modality structure
    per_mod = {}
    for mod, m in (("image", "v"), ("text", "t")):
        if _feat(model, m) is None:
            continue
        surv = tuple(x for x in ("v", "t") if x != m)
        sum_surv = _coalesced_sum(_parts(model, mods=surv))
        A_ko = sparse_row_topk(sum_surv, model.knn_k)
        h_ko = _h(model, A_ko)
        A_fix = _restrict_to_support(sum_surv, A_full.indices(), n)
        h_fix = _h(model, A_fix)
        learned = build_knn_graph(_trs(model, m)(_feat(model, m)).detach(), model.knn_k)
        e_learned = _edgeset(learned)
        e_orig = _edgeset(_orig(model, m))
        # F4 certificate: does the softmax slot re-indexing matter?
        h_alt = _h(model, _fuse(model, _parts(model, mods=surv, native_widx=True)))
        per_mod[mod] = {
            "||h|| after deleting this modality (mean)": float(h_ko.norm(dim=-1).mean()),
            "||h_ko||/||h_full|| (mean ratio = mass kept)":
                float((h_ko.norm(dim=-1) / h_full.norm(dim=-1).clamp_min(1e-12)).mean()),
            "cos(content_full, content_ko)":
                float(F.cosine_similarity(content, F.normalize(h_ko, dim=-1)).mean()),
            "items whose content term is exactly 0 after KO (renorm arm)":
                int((h_ko.norm(dim=-1) == 0).sum()),
            "items whose content term is exactly 0 after KO (fixsupport arm)":
                int((h_fix.norm(dim=-1) == 0).sum()),
            "fixsupport: surviving edges of the frozen support":
                int((A_fix.values() != 0).sum()),
            "fixsupport: frozen support size": int(A_full._nnz()),
            "learned-vs-orig graph Jaccard (F5: random projection)":
                len(e_learned & e_orig) / len(e_learned | e_orig),
            "softmax_reindex_max_delta_normalized_h":
                float((F.normalize(h_ko, dim=-1) - F.normalize(h_alt, dim=-1)).abs().max()),
        }
        del sum_surv, A_ko, h_ko, A_fix, h_fix, h_alt
    a["per_modality"] = per_mod

    # --- provenance of the 70,500 fused edges actually selected
    a["fused_edge_provenance"] = _provenance(model, A_full)
    # --- what the input-mean screen degenerates into
    a["screen_degeneracy"] = _screen_certificate(model)
    a["both_knockout_is_exact"] = True
    a["both_knockout_note"] = ("deleting F.normalize(h) leaves u @ i_cf.T (pure LightGCN); "
                               "nothing survives to renormalise, so the DELETE_PLUS_RENORM "
                               "ambiguity does not apply to this arm")
    return a


def _edgeset(sp: torch.Tensor) -> set:
    i = sp.indices().cpu()
    return set(zip(i[0].tolist(), i[1].tolist()))


@torch.no_grad()
def _provenance(model, A_full: torch.Tensor) -> dict:
    """For each SELECTED edge, which modality contributed a nonzero value to it."""
    n = model.n_items
    out = {"selected_edges": int(A_full._nnz())}
    masks = {}
    for mod, m in (("image", "v"), ("text", "t")):
        if _feat(model, m) is None:
            continue
        sub = _restrict_to_support(_coalesced_sum(_parts(model, mods=(m,))),
                                   A_full.indices(), n).coalesce()
        masks[mod] = (sub.values() != 0)
        out[f"edges with a {mod} contribution"] = int(masks[mod].sum())
        out[f"{mod} share of selected edge mass"] = float(
            sub.values().sum() / A_full.values().sum())
    if len(masks) == 2:
        mi, mt = masks["image"], masks["text"]
        out["edges from image only"] = int((mi & ~mt).sum())
        out["edges from text only"] = int((mt & ~mi).sum())
        out["edges from both"] = int((mi & mt).sum())
    return out


@torch.no_grad()
def _screen_certificate(model) -> dict:
    """Show that the input-mean screen replaces, rather than deletes, the modality graph.

    Also measures HOW MUCH of the injected graph survives the fused top-k, which is what
    decides whether the screen behaves like a deletion (LATTICE) or like a joint
    both-modality wipe (DAMRS).
    """
    cert = {"protocol": "v_feat := per-dim mean over items (test-time mean replacement)"}
    for mod, m in (("image", "v"), ("text", "t")):
        f = _feat(model, m)
        if f is None:
            continue
        surv = tuple(x for x in ("v", "t") if x != m)
        fm = f.mean(0, keepdim=True).expand_as(f).contiguous()
        g_mean = build_knn_graph(fm, model.knn_k)
        g_real = _orig(model, m)
        e_mean, e_real = _edgeset(g_mean), _edgeset(g_real)
        cols = g_mean.indices()[1]
        # what the screen's FUSED graph looks like next to the exact-deletion fused graph
        A_scr = _fuse(model, _parts(model, feat_override={m: fm}))
        A_del = _fuse(model, _parts(model, mods=surv))
        e_scr, e_del = _edgeset(A_scr), _edgeset(A_del)
        inject = len(e_scr & e_mean)
        cert[mod] = {
            "fused-graph Jaccard(screen, exact deletion)": len(e_scr & e_del) / len(e_scr | e_del),
            "screen edges that come from the injected degenerate graph": inject,
            "screen fused edges total": len(e_scr),
            "all item feature rows identical after substitution":
                bool((fm - fm[0:1]).abs().max().item() == 0.0),
            "edges of the degenerate graph": len(e_mean),
            "Jaccard(degenerate graph, real graph)": len(e_mean & e_real) / len(e_mean | e_real),
            "distinct neighbour ids in the degenerate graph": int(cols.unique().numel()),
            "max neighbour id used": int(cols.max()),
            "verdict": ("tie-broken top-k over a constant similarity row -> a hub graph on "
                        "the lowest item ids: an INJECTED graph, not a deleted one; whether "
                        "that behaves like a deletion depends on how much of it survives "
                        "the fused top-k (see the two fields above)"),
        }
        del fm, g_mean, A_scr, A_del
    return cert


@torch.no_grad()
def _trs_certificate(model, dataset=None) -> dict:
    """F5: are image_trs/text_trs still at their seed-2024 initialisation?"""
    cert = {"only_use_site": "lattice.py:96 / :104, output .detach()-ed -> zero gradient",
            "weight_decay": 0.0}
    try:
        from recsys_bridge import load_frozen                       # noqa: E402
        ds = dataset if isinstance(dataset, str) else getattr(model, "_dataset_name", None)
        if ds is None:
            cert["fresh_init_comparison"] = "skipped (dataset name unknown)"
            return cert
        _, _, m0, _ = load_frozen("lattice", ds, "cpu", load_ckpt=False)
        got = dict(model.named_parameters())
        ref = dict(m0.named_parameters())
        cert["max|ckpt - fresh seed-2024 init|"] = {
            k: float((got[k].detach().cpu() - ref[k].detach().cpu()).abs().max())
            for k in ("image_trs.weight", "image_trs.bias",
                      "text_trs.weight", "text_trs.bias", "modal_weight")
            if k in got and k in ref}
        del m0
    except Exception as exc:                                        # pragma: no cover
        cert["fresh_init_comparison"] = f"failed: {type(exc).__name__}: {exc}"
    return cert


@torch.no_grad()
def _recon_pair(model, u, i, chunk: int = 2048) -> float:
    dev = u.device
    worst = 0.0
    for s in range(0, model.n_users, chunk):
        users = torch.arange(s, min(s + chunk, model.n_users), device=dev)
        ref = model.full_sort_predict({"user": users})
        worst = max(worst, float((u[users] @ i.t() - ref).abs().max()))
        del ref
    return worst


@torch.no_grad()
def recon_error(model, dataset=None, device=None, chunk: int = 2048) -> float:
    """max |baseline_scores - model.full_sort_predict| over ALL users, in chunks."""
    u, i = variants(model, dataset, device)["baseline"]
    return _recon_pair(model, u, i, chunk)


@torch.no_grad()
def recon_error_detail(model, dataset=None, device=None, chunk: int = 2048,
                       repeats: int = 5) -> dict:
    """recon_error plus the model's OWN self-consistency floor.

    `torch.sparse.mm` on CUDA is not bit-reproducible, so the model already disagrees with
    itself across two calls; any decomposition error below that floor is unmeasurable."""
    u, i = variants(model, dataset, device)["baseline"]
    su, si = model._propagate()
    su, si = su.detach(), si.detach()
    err = _recon_pair(model, u, i, chunk)
    floors = [_recon_pair(model, su, si, chunk) for _ in range(repeats)]
    scale = float((su[:chunk] @ si.t()).abs().max())
    users = torch.arange(0, min(4096, model.n_users), device=u.device)
    mine_k = torch.topk(u[users] @ i.t(), 20, dim=-1).indices
    ref_k = torch.topk(model.full_sort_predict({"user": users}), 20, dim=-1).indices
    same = [set(a.tolist()) == set(b.tolist()) for a, b in zip(mine_k.cpu(), ref_k.cpu())]
    return {"recon_error": err, "self_floor_max": max(floors), "self_floor_runs": floors,
            "max_abs_score": scale, "rel_recon_error": err / scale,
            "float32_ulp_at_max_score": float(torch.finfo(torch.float32).eps * scale),
            "top20_set_agreement_frac": float(sum(same)) / len(same),
            "top20_users_checked": len(same)}


# --------------------------------------------------------------------------- standalone run
def _main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="baby")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    from recsys_bridge import load_frozen                             # noqa: E402
    from ranking_effects import evaluate_item_matrix, ranking_change  # noqa: E402

    pins = json.loads((ROOT / "results" / "phase_micro" / "ckpt_pins.json").read_text())
    pin = args.ckpt or pins.get(f"lattice/{args.dataset}", {}).get("path") \
        or CKPT_PIN.get(args.dataset)
    cfg, ds, model, test_loader = load_frozen("lattice", args.dataset, device, ckpt_path=pin)
    model._dataset_name = args.dataset
    print(f"ckpt = {model._ckpt_name}  (pinned={model._ckpt_pinned})", flush=True)
    print(f"missing keys = {model._missing_keys}", flush=True)

    det = recon_error_detail(model, args.dataset, device)
    print("recon: " + json.dumps(det), flush=True)

    attr = attribution(model, args.dataset, device)
    print(json.dumps(attr, indent=2), flush=True)

    vs = variants(model, args.dataset, device)
    res, base_topk = {}, None
    for name, (u, i) in vs.items():
        m, topk, _ = evaluate_item_matrix(u, i, test_loader, device)
        m = {k: float(v) for k, v in m.items()}
        if name == "baseline":
            base_topk = topk
            res[name] = {"metrics": m}
        else:
            res[name] = {"metrics": m,
                         "delta": {k: m[k] - res["baseline"]["metrics"][k] for k in m},
                         "ranking_change": ranking_change(base_topk, topk)}
        d = "" if name == "baseline" else \
            f"  dR@20={m['Recall@20'] - res['baseline']['metrics']['Recall@20']:+.6f}" \
            f"  dN@20={m['NDCG@20'] - res['baseline']['metrics']['NDCG@20']:+.6f}"
        print(f"  {name:30s} R@20={m['Recall@20']:.6f}  N@20={m['NDCG@20']:.6f}{d}", flush=True)

    logged = json.loads((RECSYS / "logs" / model._ckpt_name.replace(".pt", "") /
                         "result.json").read_text())["test_result"]
    base_m = res["baseline"]["metrics"]

    # cross-check: do the *_screen_meanfeat arms reproduce the PUBLISHED coarse screen?
    screen_check = {}
    sp = ROOT / "results" / "phasex_crossarch" / "crossarch_knockout.json"
    if sp.is_file():
        pubs = [e for e in json.loads(sp.read_text())
                if e["model"] == "lattice" and e["dataset"] == args.dataset]
        if pubs:
            pub = pubs[0]
            screen_check["baseline_vs_published_baseline"] = {
                k: base_m[k] - float(pub["baseline"][k]) for k in base_m if k in pub["baseline"]}
            for arm, key in (("image_screen_meanfeat", "image_knockout"),
                             ("text_screen_meanfeat", "text_knockout"),
                             ("both_screen_meanfeat", "both_knockout")):
                if arm in res and key in pub:
                    screen_check[arm] = {
                        "mine_dR@20": res[arm]["delta"]["Recall@20"],
                        "published_dR@20": float(pub[key]["Recall@20"]),
                        "abs_diff": abs(res[arm]["delta"]["Recall@20"]
                                        - float(pub[key]["Recall@20"]))}
            print("published-screen check: " + json.dumps(screen_check), flush=True)
    out = {"model": "lattice", "dataset": args.dataset, "CLASS": CLASS,
           "CLASS_DESC": CLASS_DESC, "EXACTNESS": EXACTNESS, "exact_arms": EXACT_ARMS,
           "ckpt": model._ckpt_path, "recon": det,
           "logged_test": {k: float(v) for k, v in logged.items()},
           "baseline_minus_logged": {k: base_m[k] - float(logged[k]) for k in base_m
                                     if k in logged},
           "attribution": attr, "arms": res, "arm_docs": ARMS, "bracket": BRACKET,
           "published_screen_check": screen_check}
    path = Path(args.out) if args.out else (ROOT / "results" / "_scratch" /
                                            f"exact_ko_lattice_{args.dataset}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
