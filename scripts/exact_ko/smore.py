"""SMORE (WSDM'25) — content-knockout module.  EXACTNESS = BRACKET_ONLY.

Source read line by line: /workspace/Recsys/src/models/smore.py
Contract: scripts/exact_ko/CONTRACT.md

==============================================================================
1. What the scoring function is
==============================================================================
`full_sort_predict` -> `forward(smore_norm_adj)` -> plain inner product of

    all_embeds = content_embeds + side_embeds ,  side_embeds = (A_v + A_t + A_f)/3

with, writing E = item_id_embedding.weight and n_layers = 1:

    content_embeds = mean_{l=0..L} (A_ui^l [U ; E])            # LightGCN, PURE CF
    conv_v = irfft( rfft(W_v e_v) * w_v )                      # per-modality spectral filter
    conv_t = irfft( rfft(W_t e_t) * w_t )
    conv_f = irfft( rfft(W_t e_t) * rfft(W_v e_v) * w_f )      # <-- BILINEAR in (v,t)
    X_m    = A_m @ ( E * sigmoid(G_m conv_m + g_m) )           # m in {v,t,f}; A_f = max(A_v,A_t)
    Z_m    = [ R @ X_m ; X_m ]
    A_v    = p_v * softmax(Q_v Z_f) * Z_v                      # attention driven by the JOINT
    A_t    = p_t * softmax(Q_t Z_f) * Z_t
    A_f    = p_f * Z_f
    p_m    = sigmoid(P_m content_embeds)                       # "modality preference", CF-driven

`self.dropout` (p=0.1) wraps p_v/p_t/p_f; it is an `nn.Dropout` module, so `model.eval()`
really does turn it off (contract rule 5 — there is no functional RNG anywhere in SMORE's
inference path; the module asserts `model.training is False`).

==============================================================================
2. Why there is NO exact per-modality knockout (the survey's ENTANGLED_NO_EXACT)
==============================================================================
Three separate entanglements, in decreasing order of size:

E1  **Bilinear joint branch.**  conv_f = irfft(F_t ⊙ F_v ⊙ w_f) is a circular CONVOLUTION
    of the two modalities.  A bilinear map has no main effects: setting either factor to
    zero sends the whole term to zero.  Measured on baby (attribution):
        ||conv_f(v,t)|| = 297.05 ,  ||conv_f(v,0)|| = 0.0 ,  ||conv_f(0,t)|| = 0.0
    i.e. 100% of the joint's spectral mass is interaction, 0% is per-modality main effect.
    So no additive split  A_f = A_f^image + A_f^text  exists at ANY coefficient.  This is
    the reason the module is BRACKET_ONLY and not EXACT.

E2  **Max-pooled fusion graph.**  A_f = max(A_v, A_t) elementwise over the sparse union.
    On baby: nnz(A_v) = nnz(A_t) = 70,500, nnz(A_f) = 126,596 = |union| exactly; only
    20.4% of the image edges are also text edges, so 44.3% of the fusion graph's edges are
    image-exclusive.  max() is not additive either.

E3  **Attention leak.**  softmax(Q_v Z_f) and softmax(Q_t Z_f) are computed from the JOINT
    branch, so image influences the *text* branch's per-dimension attention.  Even the `hi`
    end of the image bracket therefore leaves a residual image path.  Its size is measured
    (`att_leak_*` in attribution) and an extra arm `*_knockout_hi_attfree` closes the gate
    half of it.

==============================================================================
3. The bracket, and what IS exact
==============================================================================
side_embeds is a fixed 1/3-weighted sum of three terms, so deleting a term is an EXACT
deletion of an additive score contribution (no renormalisation, no input corruption):

    both_knockout          = drop A_v, A_t, A_f  ->  all_embeds = content_embeds
                             EXACT and complete: this removes every content interface at
                             once (gates, spectral filters, AND the three frozen kNN
                             graphs), leaving the pure LightGCN CF propagation.
    image_knockout_lo      = drop A_v only        (image's EXCLUSIVE branch)   -> LOWER bound
    image_knockout_hi      = drop A_v and A_f     (everything image touches)   -> UPPER bound,
                             but it also destroys text's share of the joint.
    joint_knockout         = drop A_f only        (the un-attributable middle term)
    text_knockout_lo/hi    = mirror image.

`image_knockout` / `text_knockout` (the bare names the driver expects) are ALIASES OF THE
`hi` END.  That is the conservative choice for a paper whose claim is "the effect is
small": if even the upper bracket end is small, the claim holds.  Never quote a bare
`image_knockout` number for SMORE without its `_lo` partner — see BRACKET.

Predicted (and verified) identity:  d(image_lo) + d(text_lo)  falls far short of d(both),
and the gap is d(joint) — because the joint branch is where the content-side mass lives.

Second, finer family (the *live* content interface on its own).  Content enters each
branch ONLY through the gate pre-activation `G_m conv_m + g_m`; deleting the `G_m conv_m`
term is again an exact term deletion, leaving the learned constant gate sigmoid(g_m) and
keeping the branch, its ID mass and its graph:

    image_gatezero / text_gatezero / joint_gatezero / all_gatezero

`all_gatezero` is the sharpest available statement of "the trained content vectors are
off", with only the frozen kNN graph interface (C1) left alive.  The frozen graph cannot
be deleted separately: A_m enters as X_m = A_m @ (...) with no residual, so A_m -> 0 IS
branch deletion, and A_m -> I would be a substitution, not a deletion.  Hence the graph
interface is bracketed by (gatezero, branch-off), not isolated.

==============================================================================
4. What the input-mean screen actually measured on SMORE  (a real defect)
==============================================================================
`image_embedding = nn.Embedding.from_pretrained(v_feat.clone(), freeze=False)` is a
PARAMETER, and `forward` reads `self.image_embedding.weight`, never `self.v_feat`.
The screen (phasex_crossarch_knockout.build_eval) constructs the model with per-dim mean
features and THEN calls `load_state_dict(..., strict=False)`, which overwrites the mean
initialisation with the trained table.  So on SMORE the substitution has zero effect on
the live content path; the only thing it changes is the set of non-persistent kNN-graph
buffers, and with constant features every cosine is 1, so `build_knn_normalized_graph`
degenerates to "every item points at the same 10 items".
=> the published SMORE screen row is a DEGENERATE-GRAPH corruption, not a content
   knockout.  `*_screen_meanfeat` reproduces it inside this module (certified against the
   published numbers) so the reader can see the two protocols side by side, and
   `attribution` carries the bitwise certificate that the on-path table is the checkpoint's.

==============================================================================
5. Content-interface taxonomy
==============================================================================
Legend as fixed in scripts/exact_ko/mmgcn.py:
    C1 frozen content-derived item-item graph | C2 live content ADDED to the scored embed
    C3 content as layer-0 node feature of a per-modality tower | C4 training-only | C5 ID-only
SMORE is C1 + a variant of C2 in which the projected content MULTIPLIES the ID embedding
(a sigmoid gate) instead of being added -- written "C2g" here -- plus the bilinear joint
branch that belongs to no single modality.  CLASS = "C1+C2g".

Contract rule 6 (freeze=False tables) applies and is measured:
    cos(image_embedding.weight, raw v_feat) = 0.999998  -> still literally the encoder's
        image features (the table barely moved; 4096-d, lr 1e-3, 41 epochs)
    cos(text_embedding.weight,  raw t_feat) = 0.771665  -> the text table is now largely a
        free learned table; a "text knockout" here is only ~partly an encoder knockout.

==============================================================================
6. MEASURED — smore/baby, ckpt smore_baby_20260615_214121.pt, torch 2.12.0+cu126
==============================================================================
recon_error = 2.9e-06 .. 3.8e-06 over repeated runs (tol 1e-5).  The residual is
    torch.sparse.mm's own run-to-run nondeterminism, NOT a modelling gap: calling
    model.forward() twice already differs by 2.4e-07 in the embeddings, and only the
    recon number (a max over ~29M score entries) is sensitive to it -- three full runs of
    this module produced BITWISE identical Recall/NDCG on all 18 arms.
baseline R@20 = 0.09948956745948262 vs logged test 0.0994895674594826  (delta 1.4e-17).

    arm                        dR@20      dN@20     top20-overlap
    image_knockout_lo        -0.000180  -0.000045      0.981
    text_knockout_lo         +0.000257  +0.000127      0.976   <- deleting text HELPS
    joint_knockout           -0.004569  -0.002182      0.775
    image_knockout_hi        -0.005077  -0.002269      0.771
    text_knockout_hi         -0.004789  -0.002354      0.768
    image_kn_hi_attfree      -0.004923  -0.002258      0.770
    text_kn_hi_attfree       -0.004925  -0.002411      0.768
    both_knockout (EXACT)    -0.005028  -0.002388      0.764
    image_gatezero           -0.000214  -0.000085      0.990
    text_gatezero            +0.000206  +0.000109      0.989
    joint_gatezero           -0.001492  -0.000802      0.888
    all_gatezero             -0.001706  -0.000841      0.883
    image_screen_meanfeat    -0.002516  (input-mean screen: -0.002516, diff 0.0e+00)
    text_screen_meanfeat     -0.004871  (input-mean screen: -0.004871, diff 0.0e+00)
    both_screen_meanfeat     -0.006314  (input-mean screen: -0.006314, diff 4.9e-07)

    image bracket [lo, hi] = [-0.00018, -0.00492]  (27x)
    text  bracket [lo, hi] = [+0.00026, -0.00492]  (sign change)

Predicted identity, VERIFIED:
    d(image_lo) + d(text_lo) = +7.71e-05      d(both) = -5.03e-03
    gap = -5.10e-03,  d(joint) = -4.57e-03,  residual = -5.4e-04
    i.e. essentially the ENTIRE content-side effect on baby lives in the joint branch,
    which belongs to neither modality.  Per-modality attribution is not identifiable.

Three things the source shows that the survey prior did not:
  (i)  the sigmoid gate on the IMAGE branch is fully saturated -- gate_v mean 1.0000,
       std 5.1e-07, min 0.99978, 100% of entries > 0.999, driven by ||conv_v|| = 540.
       The image branch is therefore, numerically, the ID embedding propagated over the
       image kNN graph: content has been squeezed out of the live path and survives only
       as graph STRUCTURE.  (gate_t std 0.048, gate_f std 0.041 -- not saturated.)
  (ii) `both_knockout` IS exact and complete (all_embeds = content_embeds), so SMORE does
       admit an exact JOINT content knockout; only the per-modality SPLIT is unidentifiable.
  (iii) the input-mean screen on SMORE never touched the content path at all
       (section 4) -- reproduced here to 0.0 / 4.9e-07 by changing ONLY the kNN graphs.

Noise: there is no retrain-seed MDE for smore in results/phase_mde (only lattice, mmgcn,
vbpr). Against the screen's NOISE_HINT of 0.0026, the joint/both effects are ~2x the hint
and the per-modality lo effects are ~15x BELOW it. All arms here are paired against one
frozen checkpoint and are exactly reproducible, so the relevant floor is the retraining
floor, not an eval floor.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

ROOT = Path("/workspace/MechInterp")
RECSYS = Path("/workspace/Recsys")
for _p in (str(RECSYS), str(ROOT / "src" / "models"), str(ROOT / "src" / "interp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ----------------------------------------------------------------- contract API
CLASS = "C1+C2g"
CLASS_PARTS = {
    "graph_path": "C1  (image_original_adj / text_original_adj / fusion_adj, frozen at __init__)",
    "gate_path": "C2g (projected content MULTIPLIES the ID embedding through a sigmoid gate)",
    "joint_path": "entangled (bilinear spectral product of both modalities; belongs to neither)",
}
EXACTNESS = "BRACKET_ONLY"

# SMORE has NO smore entry in results/phase_micro/ckpt_pins.json (reported to the driver
# author). Only one smore run exists per Amazon dataset, so latest_ckpt() agrees with the
# pin below; pinned anyway so the artifact names the checkpoint it describes.
CKPT = {
    "baby":     "/workspace/Recsys/ckpts/smore_baby_20260615_214121.pt",
    "sports":   "/workspace/Recsys/ckpts/smore_sports_20260615_214542.pt",
    "clothing": "/workspace/Recsys/ckpts/smore_clothing_20260615_215358.pt",
    "microlens": "/workspace/Recsys/ckpts/smore_microlens_20260621_132957.pt",
}
LOGGED_RUN = {
    "baby":     "smore_baby_20260615_214121",
    "sports":   "smore_sports_20260615_214542",
    "clothing": "smore_clothing_20260615_215358",
    "microlens": "smore_microlens_20260621_132957",
}

# mode per branch: "full" | "gz" (gate content term deleted) | "off" (additive term deleted)
# att: which Z_f feeds softmax(Q_. Z_f).  None = follow the joint branch's own mode
#      ("full" when the joint is full or off -- deleting the additive A_f term does NOT
#       change what the model computed for the attention; "gz" when the joint is gz).
_ARMS: Dict[str, Tuple[str, str, str, str | None]] = {
    #                              v       t       f      att
    "baseline":                  ("full", "full", "full", None),
    # ---- branch-level bracket (exact deletions of additive terms) ----
    "image_knockout_lo":         ("off",  "full", "full", None),
    "text_knockout_lo":          ("full", "off",  "full", None),
    "joint_knockout":            ("full", "full", "off",  None),
    "image_knockout_hi":         ("off",  "full", "off",  None),
    "text_knockout_hi":          ("full", "off",  "off",  None),
    "image_knockout":            ("off",  "full", "off",  None),   # alias of _hi (see docstring)
    "text_knockout":             ("full", "off",  "off",  None),   # alias of _hi
    "both_knockout":             ("off",  "off",  "off",  None),   # EXACT, complete
    # ---- E3: also remove the joint's GATE content from the surviving branch's attention
    "image_knockout_hi_attfree": ("off",  "full", "off",  "gz"),
    "text_knockout_hi_attfree":  ("full", "off",  "off",  "gz"),
    # ---- live-content-interface family (branch, ID mass and frozen graph all kept) ----
    "image_gatezero":            ("gz",   "full", "full", None),
    "text_gatezero":             ("full", "gz",   "full", None),
    "joint_gatezero":            ("full", "full", "gz",   None),
    "all_gatezero":              ("gz",   "gz",   "gz",   None),
    # ---- reproduction of the input-mean screen (graph-only corruption, see S4) ----
    "image_screen_meanfeat":     ("scr",  "full", "scr",  None),
    "text_screen_meanfeat":      ("full", "scr",  "scr",  None),
    "both_screen_meanfeat":      ("scr",  "scr",  "scr",  None),
}
ARM_EXACTNESS = {
    "baseline": "EXACT",
    "both_knockout": "EXACT",
    "joint_knockout": "EXACT",
    "image_knockout_lo": "EXACT", "text_knockout_lo": "EXACT",
    "image_knockout_hi": "EXACT", "text_knockout_hi": "EXACT",
    "image_knockout": "BRACKET_ONLY", "text_knockout": "BRACKET_ONLY",
    "image_knockout_hi_attfree": "EXACT", "text_knockout_hi_attfree": "EXACT",
    "image_gatezero": "EXACT", "text_gatezero": "EXACT",
    "joint_gatezero": "EXACT", "all_gatezero": "EXACT",
    "image_screen_meanfeat": "SCREEN_REPRO", "text_screen_meanfeat": "SCREEN_REPRO",
    "both_screen_meanfeat": "SCREEN_REPRO",
}
# every *_knockout arm above is an exact deletion of the additive terms it names; the
# BRACKET is about ATTRIBUTION -- which modality the deleted mass belongs to.
BRACKET = {
    "image": ("image_knockout_lo", "image_knockout_hi_attfree"),
    "text":  ("text_knockout_lo",  "text_knockout_hi_attfree"),
}
UNDEFINED_ARMS = {
    "joint_image_share": "conv_f = irfft(F_t ⊙ F_v ⊙ w_f) is BILINEAR: zeroing either "
                         "modality zeroes the whole joint, so its 'image part' and 'text "
                         "part' do not exist (measured: main effects are exactly 0.0).",
    "graph_only_knockout": "A_m enters as X_m = A_m @ (E*gate) with no residual term, so "
                           "deleting the frozen kNN graph IS deleting the branch; isolating "
                           "the C1 interface would require a substitution, not a deletion.",
}

_CACHE_ATTR = "_exact_ko_smore_streams"


# --------------------------------------------------------------- internals
@torch.no_grad()
def _branch(model, conv: torch.Tensor, gate, adj, gz: bool) -> torch.Tensor:
    """One modality branch: Z_m = [R @ X_m ; X_m],  X_m = A_m @ (E * sigmoid(G conv + g)).

    gz=True deletes the `G conv` term inside the gate's pre-activation -- the content's
    ONLY entry point into this branch's features -- leaving the learned constant gate
    sigmoid(g).  That is a term deletion, not an input substitution."""
    E = model.item_id_embedding.weight
    g = gate(torch.zeros_like(conv)) if gz else gate(conv)
    x = E * g
    for _ in range(model.n_layers):
        x = torch.sparse.mm(adj, x)
    return torch.cat([torch.sparse.mm(model.R, x), x], dim=0)


@torch.no_grad()
def _knn(model, feat_cpu: torch.Tensor, k: int):
    """One frozen kNN graph through SMORE's OWN constructors, on CPU (that is where
    __init__ built them; topk tie-breaking is device/precision sensitive)."""
    smore_mod = sys.modules[type(model).__module__]
    return smore_mod.build_knn_normalized_graph(smore_mod.build_sim(feat_cpu), k, "sym")


@torch.no_grad()
def _streams(model, dataset, device) -> dict:
    cached = getattr(model, _CACHE_ATTR, None)
    if cached is not None:
        return cached

    # ---- structural preconditions -------------------------------------------------
    assert model.v_feat is not None and model.t_feat is not None, "smore needs both modalities"
    assert model.training is False, "model must be in eval(): self.dropout(p=%.2f) wraps the " \
                                    "modality-preference gates" % model.dropout.p
    assert int(model.n_layers) == 1, ("the per-branch decomposition below replays n_layers "
                                      f"item-graph hops; got {model.n_layers}")

    nU, nI = int(model.n_users), int(model.n_items)
    image_feats = model.image_trs(model.image_embedding.weight)
    text_feats = model.text_trs(model.text_embedding.weight)
    conv_v, conv_t, conv_f = model.spectrum_convolution(image_feats, text_feats)

    # E1 certificate: the joint is bilinear -> it has NO per-modality main effect.
    zi = torch.zeros_like(image_feats)
    zt = torch.zeros_like(text_feats)
    joint_main_v = float(model.spectrum_convolution(image_feats, zt)[2].norm(dim=-1).mean())
    joint_main_t = float(model.spectrum_convolution(zi, text_feats)[2].norm(dim=-1).mean())
    del zi, zt

    # ---- CF trunk (survives every arm; contract rule 3) ----------------------------
    ego = torch.cat([model.user_embedding.weight, model.item_id_embedding.weight], dim=0)
    embs = [ego]
    for _ in range(model.n_ui_layers):
        ego = torch.sparse.mm(model.smore_norm_adj, ego)
        embs.append(ego)
    content = torch.stack(embs, dim=1).mean(dim=1)
    del embs, ego

    # ---- graph rebuild + bitwise check against the model's own frozen buffers ------
    v_cpu = model.v_feat.detach().cpu()
    t_cpu = model.t_feat.detach().cpu()
    ia_c = _knn(model, v_cpu, model.image_knn_k)
    ta_c = _knn(model, t_cpu, model.text_knn_k)
    graph_err = 0.0
    for got, ref in ((ia_c, model.image_original_adj), (ta_c, model.text_original_adj),
                     (model._max_pool_fusion(ia_c, ta_c), model.fusion_adj)):
        g, r = got.coalesce(), ref.coalesce().cpu()
        assert torch.equal(g.indices(), r.indices()), "kNN graph sparsity pattern not reproduced"
        graph_err = max(graph_err, float((g.values() - r.values()).abs().max()))
    assert graph_err == 0.0, f"kNN graph rebuild is not bitwise identical ({graph_err:.3e})"

    # ---- the input-mean screen's degenerate graphs (S4) ------------------------------
    # per-dim mean features exactly as phasex_crossarch_knockout.run_model computes them
    v_scr = v_cpu.mean(0, keepdim=True).expand_as(v_cpu).contiguous()
    t_scr = t_cpu.mean(0, keepdim=True).expand_as(t_cpu).contiguous()
    ia_sc = _knn(model, v_scr, model.image_knn_k)
    ta_sc = _knn(model, t_scr, model.text_knn_k)
    ia_s, ta_s = ia_sc.to(device), ta_sc.to(device)
    fa_s = model._max_pool_fusion(ia_sc, ta_sc).to(device)        # both screened
    fa_vs = model._max_pool_fusion(ia_sc, ta_c).to(device)        # image screened only
    fa_ts = model._max_pool_fusion(ia_c, ta_sc).to(device)        # text screened only
    n_star_img = int(ia_sc.coalesce().indices()[1].unique().numel())
    del v_cpu, t_cpu, v_scr, t_scr, ia_c, ta_c, ia_sc, ta_sc

    # ---- the three branches, in every mode -----------------------------------------
    s: dict = {"nU": nU, "nI": nI, "device": device, "content": content}
    for key, conv, gate, adj in (("v", conv_v, model.gate_v, model.image_original_adj),
                                 ("t", conv_t, model.gate_t, model.text_original_adj),
                                 ("f", conv_f, model.gate_f, model.fusion_adj)):
        s[f"{key}_full"] = _branch(model, conv, gate, adj, gz=False)
        s[f"{key}_gz"] = _branch(model, conv, gate, adj, gz=True)
        s[f"conv_{key}"] = conv
        s[f"gate_{key}"] = gate(conv)
        s[f"gate0_{key}"] = gate(torch.zeros_like(conv))[:1].clone()   # constant row
    # screen arms: same weights, degenerate graphs (that is ALL the screen changes)
    s["v_scr"] = _branch(model, conv_v, model.gate_v, ia_s, gz=False)
    s["t_scr"] = _branch(model, conv_t, model.gate_t, ta_s, gz=False)
    s["f_scr_v"] = _branch(model, conv_f, model.gate_f, fa_vs, gz=False)   # image screened
    s["f_scr_t"] = _branch(model, conv_f, model.gate_f, fa_ts, gz=False)   # text screened
    s["f_scr_b"] = _branch(model, conv_f, model.gate_f, fa_s, gz=False)    # both screened

    # ---- CF-driven preference gates and joint-driven attention ---------------------
    s["p_v"] = model.dropout(model.gate_image_prefer(content))
    s["p_t"] = model.dropout(model.gate_text_prefer(content))
    s["p_f"] = model.dropout(model.gate_fusion_prefer(content))
    for tag, zf in (("full", s["f_full"]), ("gz", s["f_gz"]),
                    ("scr_v", s["f_scr_v"]), ("scr_t", s["f_scr_t"]), ("scr_b", s["f_scr_b"])):
        s[f"att_v_{tag}"] = model.softmax(model.query_v(zf))
        s[f"att_t_{tag}"] = model.softmax(model.query_t(zf))

    s["joint_main_v"] = joint_main_v
    s["joint_main_t"] = joint_main_t
    s["graph_rebuild_err"] = graph_err
    s["screen_n_star_targets"] = n_star_img
    setattr(model, _CACHE_ATTR, s)
    return s


def _att_tag(mf: str, att: str | None) -> str:
    if att is not None:
        return att
    if mf == "gz":
        return "gz"
    if mf == "scr":
        return "scr_b"      # overwritten by the caller for the single-modality screens
    return "full"


@torch.no_grad()
def _terms(s, mv: str, mt: str, mf: str, att: str | None, screen_tag: str = "scr_b"):
    tag = _att_tag(mf, att)
    if tag == "scr_b":
        tag = screen_tag
    att_v, att_t = s[f"att_v_{tag}"], s[f"att_t_{tag}"]
    out = {}
    if mv != "off":
        zv = s["v_scr"] if mv == "scr" else s[f"v_{mv}"]
        out["A_v"] = s["p_v"] * (att_v * zv)
    if mt != "off":
        zt = s["t_scr"] if mt == "scr" else s[f"t_{mt}"]
        out["A_t"] = s["p_t"] * (att_t * zt)
    if mf != "off":
        zf = s[f"f_{screen_tag}"] if mf == "scr" else s[f"f_{mf}"]
        out["A_f"] = s["p_f"] * zf
    return out


@torch.no_grad()
def _assemble(s, mv: str, mt: str, mf: str, att: str | None = None, screen_tag: str = "scr_b"):
    """all = content_embeds + (A_v + A_t + A_f)/3, with the named terms DELETED.

    The 1/3 is a fixed constant, not a data-dependent normaliser, so dropping a term is an
    exact deletion of that additive score contribution (no DELETE_PLUS_RENORM needed)."""
    terms = _terms(s, mv, mt, mf, att, screen_tag)
    allv = s["content"].clone()
    for v in terms.values():
        allv += v / 3.0
    nU = s["nU"]
    return allv[:nU].contiguous(), allv[nU:].contiguous()


_SCREEN_TAG = {"image_screen_meanfeat": "scr_v", "text_screen_meanfeat": "scr_t",
               "both_screen_meanfeat": "scr_b"}


# --------------------------------------------------------------- contract API
@torch.no_grad()
def variants(model, dataset, device) -> dict:
    s = _streams(model, dataset, device)
    return {name: _assemble(s, *spec, screen_tag=_SCREEN_TAG.get(name, "scr_b"))
            for name, spec in _ARMS.items()}


@torch.no_grad()
def attribution(model, dataset, device) -> dict:
    s = _streams(model, dataset, device)
    nU = s["nU"]
    t = _terms(s, "full", "full", "full", None)
    A_v, A_t, A_f = t["A_v"], t["A_t"], t["A_f"]
    side = (A_v + A_t + A_f) / 3.0
    allv = s["content"] + side

    def mi(x):   # mean per-ITEM L2 norm
        return float(x[nU:].norm(dim=-1).mean())

    def mu(x):   # mean per-USER L2 norm
        return float(x[:nU].norm(dim=-1).mean())

    nv, nt, nf = mi(A_v) / 3, mi(A_t) / 3, mi(A_f) / 3
    n_side, n_all = mi(side), mi(allv)
    hi_v = mi((A_v + A_f) / 3.0)
    hi_t = mi((A_t + A_f) / 3.0)

    out = {
        # ---- additive score streams (item side), already including the 1/3 --------
        "||content_i|| (CF trunk, survives everything)": mi(s["content"]),
        "||A_v/3||_i (image branch)": nv,
        "||A_t/3||_i (text branch)": nt,
        "||A_f/3||_i (joint branch)": nf,
        "||side||_i": n_side, "||all||_i": n_all,
        "||content_u||": mu(s["content"]), "||A_v/3||_u": mu(A_v) / 3,
        "||A_t/3||_u": mu(A_t) / 3, "||A_f/3||_u": mu(A_f) / 3,
        "joint_share_of_side_norm": nf / n_side,
        "joint_over_image_branch": nf / nv,
        "bracket_ratio_image (||A_v+A_f||/||A_v||)": hi_v / nv,
        "bracket_ratio_text (||A_t+A_f||/||A_t||)": hi_t / nt,
        "||image_bracket_lo||_i": nv, "||image_bracket_hi||_i": hi_v,
        # ---- E1: the joint has NO per-modality main effect ------------------------
        "||conv_f(v,t)||": float(s["conv_f"].norm(dim=-1).mean()),
        "||conv_f(v,0)|| (image main effect)": s["joint_main_v"],
        "||conv_f(0,t)|| (text main effect)": s["joint_main_t"],
        "joint_interaction_mass_fraction": 1.0,
        # ---- E2: the fusion graph is a max-pool, not a sum -------------------------
        "nnz(A_v)": int(model.image_original_adj._nnz()),
        "nnz(A_t)": int(model.text_original_adj._nnz()),
        "nnz(A_f)=|union|": int(model.fusion_adj._nnz()),
        "fusion_edges_image_exclusive_frac": 1.0 - int(model.text_original_adj._nnz()) / int(model.fusion_adj._nnz()),
        # ---- E3: how much image leaks into the TEXT branch's attention -------------
        "att_leak_t_max_abs (full vs joint-gatezero)": float((s["att_t_full"] - s["att_t_gz"]).abs().max()),
        "att_leak_t_mean_cos": float(F.cosine_similarity(s["att_t_full"], s["att_t_gz"], dim=-1).mean()),
        "att_leak_v_mean_cos": float(F.cosine_similarity(s["att_v_full"], s["att_v_gz"], dim=-1).mean()),
        # ---- the live content interface: the gates ---------------------------------
        # sigmoid gates are near-SATURATED: content changes the gate's gain much more than
        # its direction, and for image hardly at all.
        "gate_v_mean": float(s["gate_v"].mean()), "gate_v_std": float(s["gate_v"].std()),
        "gate_v_min": float(s["gate_v"].min()),
        "gate_v_frac_gt_0.999": float((s["gate_v"] > 0.999).float().mean()),
        "gate_t_mean": float(s["gate_t"].mean()), "gate_t_std": float(s["gate_t"].std()),
        "gate_t_min": float(s["gate_t"].min()),
        "gate_f_mean": float(s["gate_f"].mean()), "gate_f_std": float(s["gate_f"].std()),
        "gate_f_min": float(s["gate_f"].min()),
        "||conv_v||": float(s["conv_v"].norm(dim=-1).mean()),
        "||conv_t||": float(s["conv_t"].norm(dim=-1).mean()),
        "const_gate_v_mean sigmoid(g_v)": float(s["gate0_v"].mean()),
        "const_gate_t_mean sigmoid(g_t)": float(s["gate0_t"].mean()),
        "const_gate_f_mean sigmoid(g_f)": float(s["gate0_f"].mean()),
        "cos(gate_v, sigmoid(g_v))": float(F.cosine_similarity(
            s["gate_v"], s["gate0_v"].expand_as(s["gate_v"]), dim=-1).mean()),
        "cos(gate_t, sigmoid(g_t))": float(F.cosine_similarity(
            s["gate_t"], s["gate0_t"].expand_as(s["gate_t"]), dim=-1).mean()),
        "cos(gate_f, sigmoid(g_f))": float(F.cosine_similarity(
            s["gate_f"], s["gate0_f"].expand_as(s["gate_f"]), dim=-1).mean()),
        # ---- contract rule 6: freeze=False content tables --------------------------
        "drift_cos(image_embedding, raw v_feat)": float(F.cosine_similarity(
            model.image_embedding.weight, model.v_feat, dim=-1).mean()),
        "drift_cos(text_embedding, raw t_feat)": float(F.cosine_similarity(
            model.text_embedding.weight, model.t_feat, dim=-1).mean()),
        "||image_embedding||": float(model.image_embedding.weight.norm(dim=-1).mean()),
        "||raw v_feat||": float(model.v_feat.norm(dim=-1).mean()),
        "||text_embedding||": float(model.text_embedding.weight.norm(dim=-1).mean()),
        "||raw t_feat||": float(model.t_feat.norm(dim=-1).mean()),
        # ---- fusion / architecture coefficients ------------------------------------
        "side_mixture_weight (each branch)": 1.0 / 3.0,
        "n_layers(item graph)": int(model.n_layers), "n_ui_layers": int(model.n_ui_layers),
        "image_knn_k": int(model.image_knn_k), "text_knn_k": int(model.text_knn_k),
        "dropout_p (eval: identity)": float(model.dropout.p),
        # ---- exactness diagnostics ---------------------------------------------------
        "knn_graph_rebuild_max_err": s["graph_rebuild_err"],
        "screen_n_distinct_targets_in_degenerate_graph": s["screen_n_star_targets"],
        "training_only_not_ablated": ["cl_loss / InfoNCE", "bpr_loss", "reg_weight"],
    }
    del A_v, A_t, A_f, side, allv, t
    return out


@torch.no_grad()
def screen_certificate(model, device) -> dict:
    """Proof for S4: the coarse screen cannot touch SMORE's live content path.

    `image_embedding.weight` is a checkpoint PARAMETER, so building the model with
    per-dim-mean features and then load_state_dict() restores the trained table
    bitwise. Only the non-persistent kNN-graph buffers see the substituted features."""
    state = torch.load(model._ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
    out = {"ckpt_has_image_embedding.weight": "image_embedding.weight" in state,
           "ckpt_has_text_embedding.weight": "text_embedding.weight" in state,
           "v_feat_is_persistent_buffer": "v_feat" in state,
           "forward_reads_v_feat": False}
    for k in ("image_embedding.weight", "text_embedding.weight"):
        if k in state:
            out[f"onpath_table_is_bitwise_ckpt[{k}]"] = bool(torch.equal(
                dict(model.named_parameters())[k].detach().cpu(), state[k].cpu()))
    del state
    return out


@torch.no_grad()
def recon_error(model, dataset, device, n_users: int = 4096, chunk: int = 1024) -> float:
    """max |baseline score - model.full_sort_predict| over the first n_users users."""
    s = _streams(model, dataset, device)
    u, i = _assemble(s, *_ARMS["baseline"])
    worst = 0.0
    n = min(n_users, s["nU"])
    for start in range(0, n, chunk):
        idx = torch.arange(start, min(start + chunk, n), device=device)
        ref = model.full_sort_predict({"user": idx})
        worst = max(worst, float((ref - (u[idx] @ i.t())).abs().max()))
        del ref
    return worst


# --------------------------------------------------------------- standalone run
def main() -> int:
    import argparse, json, time
    import numpy as np
    from recsys_bridge import load_frozen                            # noqa: E402
    from ranking_effects import evaluate_item_matrix, ranking_change  # noqa: E402

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="baby")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    cfg, ds, model, loader = load_frozen("smore", args.dataset, device,
                                         ckpt_path=CKPT.get(args.dataset))
    logged = json.loads((RECSYS / "logs" / LOGGED_RUN[args.dataset] / "result.json").read_text())
    logged_r20 = float(logged["test_result"]["Recall@20"])

    t0 = time.time()
    err = recon_error(model, ds, device)
    attr = attribution(model, ds, device)
    cert = screen_certificate(model, device)
    arms = variants(model, ds, device)

    print(f"\nsmore/{args.dataset}  ckpt={Path(model._ckpt_path).name}  torch={torch.__version__}")
    print(f"CLASS={CLASS}  EXACTNESS={EXACTNESS}")
    print(f"recon_error = {err:.3e}   (tolerance 1e-5)")

    res, base_topk, base_m = {}, None, None
    for name, (u, i) in arms.items():
        m, topk, _ = evaluate_item_matrix(u, i, loader, device)
        m = {k: float(v) for k, v in m.items()}
        if name == "baseline":
            base_m, base_topk = m, topk
            print(f"  {name:<28s} R@20={m['Recall@20']:.8f}  N@20={m['NDCG@20']:.8f}   "
                  f"(logged {logged_r20:.8f}, delta {m['Recall@20'] - logged_r20:+.2e})")
            res[name] = {"metrics": m, "logged_delta": m["Recall@20"] - logged_r20}
            continue
        rc = ranking_change(base_topk, topk, k=20)
        dr = m["Recall@20"] - base_m["Recall@20"]
        dn = m["NDCG@20"] - base_m["NDCG@20"]
        print(f"  {name:<28s} R@20={m['Recall@20']:.6f}  dR@20={dr:+.6f}  dN@20={dn:+.6f}  "
              f"top20-ovl={rc['overlap@20']:.4f}  [{ARM_EXACTNESS[name]}]")
        res[name] = {"metrics": m, "dR@20": dr, "dN@20": dn, "ranking_change": rc,
                     "exactness": ARM_EXACTNESS[name]}

    # ---- the predicted bracket identity -------------------------------------------
    d = {k: res[k]["dR@20"] for k in res if "dR@20" in res[k]}
    ident = {
        "d(image_lo)+d(text_lo)": d["image_knockout_lo"] + d["text_knockout_lo"],
        "d(both)": d["both_knockout"],
        "gap = d(both)-[d(image_lo)+d(text_lo)]":
            d["both_knockout"] - (d["image_knockout_lo"] + d["text_knockout_lo"]),
        "d(joint)": d["joint_knockout"],
        "residual (gap - d(joint))":
            d["both_knockout"] - (d["image_knockout_lo"] + d["text_knockout_lo"]) - d["joint_knockout"],
        "image_bracket": [d["image_knockout_lo"], d["image_knockout_hi_attfree"]],
        "text_bracket": [d["text_knockout_lo"], d["text_knockout_hi_attfree"]],
    }
    print("\nbracket identity check:")
    for k, v in ident.items():
        print(f"  {k:<45s} {v}")

    published = {"baby": {"image": -0.0025160097466603165, "text": -0.004871371022052456,
                          "both": -0.006314}}.get(args.dataset)
    if published:
        print("\ninput-mean screen (crossarch_knockout.json) vs this module's repro:")
        for k, pk in (("image_screen_meanfeat", "image"), ("text_screen_meanfeat", "text"),
                      ("both_screen_meanfeat", "both")):
            print(f"  {k:<28s} repro dR@20={d[k]:+.6f}   published {published[pk]:+.6f}   "
                  f"diff {d[k]-published[pk]:+.2e}")

    out = {"model": "smore", "dataset": args.dataset, "class": CLASS, "class_parts": CLASS_PARTS,
           "exactness": EXACTNESS, "arm_exactness": ARM_EXACTNESS, "bracket": BRACKET,
           "undefined_arms": UNDEFINED_ARMS,
           "checkpoint": model._ckpt_path, "ckpt_pinned": model._ckpt_pinned,
           "logged_test_R@20": logged_r20, "recon_error": err,
           "attribution": attr, "screen_certificate": cert, "arms": res,
           "bracket_identity": ident,
           "env": {"numpy": np.__version__, "torch": torch.__version__},
           "seconds": time.time() - t0}
    path = Path(args.out) if args.out else (ROOT / "results" / "_scratch" / "exact_ko" /
                                            f"smore_{args.dataset}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print("\nattribution:\n" + json.dumps(attr, indent=1, default=str))
    print("screen_certificate: " + json.dumps(cert))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
