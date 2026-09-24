"""Exact-as-possible structural knockout for GUME (CIKM 2024).

Source of truth: /workspace/Recsys/src/models/gume.py (READ-ONLY).

======================================================================================
1. WHAT THE SCORING FUNCTION ACTUALLY IS
======================================================================================
`full_sort_predict` calls `forward(self.gume_norm_adj)` and dots the two halves of
`all_embeds`.  Written out (verbatim variable names on the right):

    gate_v = image_space_trans(V_table) = sigmoid(W_v2 (W_v1 V_table + b1) + b2)   # in (0,1)^d
    gate_t = text_space_trans(T_table)  = sigmoid(W_t2 (W_t1 T_table + b1) + b2)   # in (0,1)^d
    v_item = E_id * gate_v          (elementwise)          # image_item_embeds
    t_item = E_id * gate_t                                 # text_item_embeds

    ext_id = conv_ui(gume_norm_adj, E_user, E_id)          # 3-layer LightGCN, mean of 4 layers
    X      = [ R @ (A_img^2 v_item) ; A_img^2 v_item ]     # explicit_image_embeds
    Y      = [ R @ (A_txt^2 t_item) ; A_txt^2 t_item ]     # explicit_text_embeds
    (the exponent is n_layers, =2 on baby, =1 on sports/clothing)

    a_v, a_t = softmax([separate_coarse(X), separate_coarse(Y)])      # per-ROW scalars, a_v+a_t=1
    coarse   = a_v*X + a_t*Y
    g_v, g_t = image_behavior(ext_id), text_behavior(ext_id)          # per-row VECTORS in (0,1)^d
    integ    = ( g_v*(X-coarse) + g_t*(Y-coarse) + coarse ) / 3
    all      = ext_id + integ

The attribute-separation block collapses ALGEBRAICALLY.  Using a_t = 1-a_v:

    g_v*(X - coarse) = g_v*(a_t X - a_t Y) =  a_t g_v * (X-Y)
    g_t*(Y - coarse) = g_t*(a_v Y - a_v X) = -a_v g_t * (X-Y)
=>  integ = [ c * X + (1-c) * Y ] / 3 ,   c := a_t*g_v - a_v*g_t + a_v          (*)

and because g_v,g_t in (0,1) and a_v,a_t in (0,1) with a_v+a_t=1, c lies in [0,1]
elementwise: GUME's whole fusion head is a **per-element convex blend of the two
explicit-modality streams, scaled by a hard-coded 1/3**, added to the LightGCN term.
Identity (*) is verified numerically in `attribution` (max abs residual 6e-8 on baby).

======================================================================================
2. WHY THIS IS *NOT* AN EXACT PER-MODALITY KNOCKOUT  (two separate reasons)
======================================================================================
(R1) SHARED NORMALISER.  The blend is convex.  Deleting c*X leaves (1-c)*Y, but the
     model's own one-modality counterfactual would renormalise the mixture (a softmax
     over a single logit is 1, so coarse=Y and fine_text=0, i.e. integ = Y/3).
     Contract rule 3 => EXACTNESS = "DELETE_PLUS_RENORM", and we emit BOTH ends:
         image_knockout          integ = Y/3                (renormalised)
         image_knockout_nonorm   integ = ((1-c)*Y)/3        (only the removed mass gone)
     On baby c is tiny (item mean 0.098), so the two ends differ by ~10% of the text
     stream; both are reported, the truth is bracketed by them.

(R2) AN IRREDUCIBLY JOINT TERM (the hard part).  `gume_norm_adj` is NOT the plain
     bipartite graph.  `_find_inter_add_edge` puts an item-item block into it whose
     edges are the per-item INTERSECTION of the image-kNN and the text-kNN neighbour
     sets (7,407 directed edges on baby, on top of 2*118,551 UI edges).  That block
       (a) adds item->item propagation inside `conv_ui`, and
       (b) changes every item's degree, hence rescales the WHOLE normalised graph and
           the user-item block R that both explicit streams are pushed through
           (measured on baby: mean relative change of R = 4.1%, max |dR| = 0.265).
     So content also enters the dominant `ext_id` term -- and it enters as an
     INTERSECTION, which is a function of both modalities jointly.  Deleting "image"
     from an intersection is not defined: with the image neighbourhoods gone the
     intersection is empty, which is the *same* object you get by deleting text.
     Per-modality reliance is therefore NOT IDENTIFIED, and we say so with a bracket:

         LOW  end  image_knockout / image_knockout_nonorm : delete the image branch only
         HIGH end  image_knockout_hi : delete the image branch AND the joint II block
         (the joint block's own price is measured alone by `graph_knockout`)

     `both_knockout` (both branches off, II block still in) and `both_knockout_hi`
     (= id_only_floor, all content off) bracket the joint effect the same way.

======================================================================================
3. C6 (freeze=False) AND WHAT THE PUBLISHED SCREEN ACTUALLY DID
======================================================================================
`image_embedding` / `text_embedding` are `nn.Embedding.from_pretrained(feat, freeze=False)`
-> they are LEARNABLE and they ARE in the checkpoint (verified: state_dict contains
`image_embedding.weight` / `text_embedding.weight`; recsys_bridge asserts no learnable
key is missing).  Consequence for the coarse screen in
results/phasex_crossarch/crossarch_knockout.json, which substitutes the per-dimension
mean for v_feat/t_feat *at construction time* and then loads the checkpoint:

    the substituted feature is OVERWRITTEN by `load_state_dict`.  The screen therefore
    NEVER touches the on-path content table.  All it perturbs is the two frozen kNN
    graphs (built in __init__ from the raw feature) and, through them, the joint
    intersection block.

`screen_blindspot_certificate` proves both halves of that statement, and additionally
shows the collateral damage: under the mean feature every cosine is identical, so the
image kNN graph degenerates to "the same 10 lowest-id items for every item" (verified:
1 unique neighbour row for all 7,050 items) and the image/text intersection collapses
from 7,407 edges to 12 -- i.e. the screen's "image" arm silently also removed ~99.8%
of the JOINT term.  Its measured image dR@20 (-9.7e-05) is thus not an image
measurement at all.

Drift cosines (contract rule 6), baby: image cos(trained,raw) mean 0.99990 (min 0.9935),
text mean 0.97131 (min 0.8379), ||trained||/||raw|| = 1.0002 / 1.1533.  The tables have
barely moved off the encoder features, so "content table" here really is encoder content.

======================================================================================
4. A DEGENERACY WORTH REPORTING: THE TEXT GATE IS SATURATED
======================================================================================
`text_space_trans` ends in a Sigmoid, and on baby it is pinned at the top:
pre-activation min 6.98 / mean 15.2 => gate_t in [0.999074, 1.0], mean 0.999999, and
19.97% of its 7050x64 entries are EXACTLY 1.0 in float32.  So t_item = E_id * gate_t
is the ID table modulated by at most 0.09%: at inference the trained TEXT TABLE is
numerically inert and GUME's text pathway is, to 1e-3, the pure C1 interface
"propagate the ID embedding over a frozen text-kNN graph" (exactly FREEDOM's).
The image gate is genuinely live by contrast (mean 0.2045, std 0.3839, pre-activation
range [-80.9, 67.9]).  The `*_table_knockout` arms measure this directly: they replace
the gate by its multiplicative identity 1 (the neutral element of a multiplicative
interface -- the multiplicative analogue of deleting an additive term), which removes
the trained table from the score while keeping the raw-feature kNN graph.

======================================================================================
5. TRAINING-ONLY BRANCHES (contract rule 4)
======================================================================================
`extended_image_user` / `extended_text_user` (2 x n_users x 64 parameters) feed
`extended_image_embeds` / `extended_text_embeds` -> `extended_it_embeds`, which
`forward` returns ONLY under train=True (used by the um_loss InfoNCE).  They are
absent from the inference path and are left untouched in every arm;
`_training_only_certificate` proves it by autograd-unreachability (a bitwise probe is
useless here, see 6).  `conv_ui` is also called for them during training only.

======================================================================================
6. NON-DETERMINISM (measured, not assumed)
======================================================================================
`torch.sparse.mm` on this GPU is NOT bitwise reproducible (two identical calls differ
by up to 7.2e-07; `full_sort_predict` called twice differs by up to 1.9e-06 on a score
scale of ~21.7, and `torch.use_deterministic_algorithms(True)` does not fix it for
cuSPARSE SpMM).  Therefore:
  * `recon_error` cannot be 0 by construction; measured 2.86e-06 over all 19,445 users,
    i.e. 1.3e-07 relative -- inside the contract's 1e-5 driver tolerance but NOT
    bitwise, and the residual is the GPU's, not the decomposition's (the decomposition
    residual against `forward` is 2.4e-07, the same size as forward-vs-forward);
  * all arms are built from ONE shared forward pass, so arm-vs-arm differences are
    exactly paired and carry no SpMM noise;
  * certificates use autograd reachability, not bitwise equality.

======================================================================================
6b. WHY THE *_hi ARMS REBUILD R AS WELL AS adj
======================================================================================
`GUME._get_adj_mat` returns `norm_adj` and `R` from ONE normalisation: R is literally
the [:n_users, n_users:] block of norm_adj (asserted bitwise in
`attribution["R_is_the_user_item_block_of_norm_adj"]`).  The joint item-item block
changes item degrees, hence rescales R too.  So the *_hi counterfactual rebuilds BOTH
through GUME's own code path; `model.forward(plain_adj)` alone would keep the trained,
content-normalised R and be an inconsistent hybrid.  `screen_blindspot_certificate`
also proves the rebuild path is faithful: feeding `_get_adj_mat` the TRUE intersection
reproduces `gume_norm_adj` bitwise (same sparsity pattern, max value diff 0.0), so the
*_hi arms differ from `baseline` by the II block and nothing else.

======================================================================================
MEASURED ON BABY (ckpt gume_baby_20260615_235120, the one the input-mean screen used)
======================================================================================
recon_error 2.9e-06 (== the model's own self-vs-self SpMM noise, 1.9e-06--2.9e-06);
baseline R@20 0.10236603315543916 vs logged 0.10236603315543918 (diff -2.8e-17);
N@20 differs by -1.5e-07 (one tie-break flip caused by the SpMM noise above).

    arm                       dR@20        dN@20
    image_knockout          +0.00009857  +0.00017009      LOW  end   |
    image_knockout_nonorm   +0.00002449  +0.00010957      LOW  end   | image bracket
    image_knockout_hi       +0.00016885  +0.00037437      HIGH end   |
    text_knockout           -0.00136780  -0.00109271      LOW  end   |
    text_knockout_nonorm    -0.00122858  -0.00106948      LOW  end   | text bracket
    text_knockout_hi        -0.00143288  -0.00109465      HIGH end   |
    both_knockout           -0.00101002  -0.00093942
    both_knockout_hi        -0.00109695  -0.00098862      true content-free floor
    graph_knockout          -0.00000257  +0.00021035      joint II block alone
    image_table_knockout    -0.00014571  +0.00003272      trained image table alone
    text_table_knockout     +0.00000000  +0.00000015      trained text table alone

Readings:
  * DELETING IMAGE IS NON-NEGATIVE on every arm (+2.4e-05 .. +1.7e-04 R@20): GUME/baby
    does not use image, and the whole bracket is 10--70x below the 3-seed training MDE
    for this model (2*sd over the logged seeds 2024/2025/2026 = 1.74e-03).  The published
    screen's -9.7e-05 has the OPPOSITE SIGN to the exact measurement.
  * The screen also inflates the text and joint effects: screen text -1.81e-03 vs exact
    bracket [-1.23e-03, -1.43e-03]; screen both -3.16e-03 vs exact [-1.01e-03, -1.10e-03],
    i.e. ~3x too large -- unsurprising, since its "both" arm is a garbage-graph condition
    (see 3), not a deletion.
  * The renormalisation bracket (knockout vs knockout_nonorm) is 7.4e-05 wide on image and
    1.4e-04 on text; the JOINT bracket (lo vs hi) is 7.0e-05 / 6.5e-05 wide.  Both are
    small here, so the qualitative conclusion is bracket-independent.
  * The joint II block, alone, is worth -2.6e-06 R@20 (`graph_knockout`) -- the graph
    enhancement that motivates GUME's title buys ~nothing at test time on baby.
  * `text_table_knockout` is 0.00000000 R@20 (N@20 +1.5e-07): the TRAINED TEXT TABLE is
    inert (saturated gate, docstring 4).  Every bit of GUME's text effect on baby comes
    from the FROZEN raw-feature text kNN graph, i.e. the C1 interface, not from the
    learnable C2 table.  Checked on all three baby seeds: gate_t mean 0.999999 / 0.999999
    / 0.999857, fraction of entries below 0.99 = 0.000 / 0.000 / 0.00014.

CROSS-SEED STABILITY OF THE FUSION COEFFICIENT (all three baby checkpoints):
    c (item mean)  0.098 / 0.097 / 0.100      <- the quantity that matters, very stable
    a_v (softmax)  0.810 / 0.149 / 0.139      <- NOT interpretable on its own
The image share of the fused content term by norm is 3.95% (||cX/3|| 0.0326 vs
||(1-c)Y/3|| 0.7911), and the content term itself is only 0.80 against ||ext_id|| 2.36.

======================================================================================
7. CONFIG HAZARD FOR OTHER DATASETS (affects the input-mean screen, not baby)
======================================================================================
configs/model/gume.yaml has the BABY block active (n_layers=2) and the sports/clothing
blocks commented out, but the trained runs used n_layers=1 there
(logs/gume_sports_.../config.yaml).  `load_frozen` reads the yaml, so sports/clothing
load with the WRONG item-item propagation depth -- which is why the input-mean screen's
gume baselines on those two datasets (0.11920 / 0.09944) match NO logged run
(0.11898 / 0.09926).  `_main` repairs this by setting `model.n_layers` from
`DATASET_N_LAYERS` after loading (n_layers is read only inside `conv_ii`, never at
construction), and records it in the output JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

ROOT = Path("/workspace/MechInterp")
RECSYS = Path("/workspace/Recsys")
for _p in (str(RECSYS), str(ROOT / "src" / "models"), str(ROOT / "src" / "interp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ------------------------------------------------------------------ contract fields
CLASS = "C1+C2"          # NOT a single class -- see section 1/2 of the docstring.
CLASS_PARTS = {
    "knn_graph_path": "C1  (frozen image/text item-item kNN graphs, built in __init__ "
                      "from the RAW features; identical interface to FREEDOM/DAMRS)",
    "gate_path": "C2  (live, learnable content tables entering the score through a "
                 "MULTIPLICATIVE sigmoid gate on the ID embedding, not an additive term)",
    "ui_graph_intersection": "C1-JOINT (irreducible): the item-item block injected into "
                             "gume_norm_adj is the per-item INTERSECTION of the two kNN "
                             "neighbourhoods, so it is a function of both modalities and "
                             "cannot be attributed to either one",
}
CLASS_DESC = ("content reaches the score through three interfaces at once: two frozen "
              "per-modality kNN graphs (C1), two learnable content tables acting as "
              "multiplicative gates on the ID embedding (C2), and a JOINT image-AND-text "
              "kNN-intersection block spliced into the user-item graph that carries the "
              "dominant ext_id term (not identifiable per modality).")

EXACTNESS = "DELETE_PLUS_RENORM"
# Extra, non-contract metadata so a driver that only reads EXACTNESS is not misled:
# the renormalisation bracket is NOT the only source of non-identification here.
EXACTNESS_QUALIFIER = "PLUS_UNIDENTIFIED_JOINT_TERM"
IDENTIFIED = False
HEADLINE_BRACKET = {
    "image": {"low": ["image_knockout", "image_knockout_nonorm"], "high": "image_knockout_hi"},
    "text": {"low": ["text_knockout", "text_knockout_nonorm"], "high": "text_knockout_hi"},
    "both": {"low": "both_knockout", "high": "both_knockout_hi"},
    "why": "low = the modality's own branch deleted (graph enhancement kept); "
           "high = its branch AND the joint image-and-text kNN-intersection block that "
           "content splices into the user-item graph. The truth is inside.",
}

ARMS = {
    "baseline": "model as trained, re-derived term by term (recon_error asserts it equals "
                "full_sort_predict)",
    "image_knockout": "LOW end. Image branch deleted; convex blend renormalised onto text "
                      "(integ = Y/3), i.e. GUME's own single-modality counterfactual. "
                      "gume_norm_adj (with its joint II block) kept.",
    "image_knockout_nonorm": "same deletion, blend NOT renormalised (integ = (1-c)*Y/3): "
                             "only the mass actually removed is gone.",
    "image_knockout_hi": "HIGH end. Image branch deleted AND the joint kNN-intersection "
                         "block removed from the user-item graph (adj and R rebuilt with "
                         "GUME's own _get_adj_mat on an empty item-item block).",
    "text_knockout": "LOW end, text branch deleted, blend renormalised onto image (integ = X/3).",
    "text_knockout_nonorm": "same deletion, blend not renormalised (integ = c*X/3).",
    "text_knockout_hi": "HIGH end: text branch + joint II block deleted.",
    "both_knockout": "both explicit branches deleted (all = ext_id); the content-derived "
                     "II block is STILL in the graph, so this is not a content-free floor.",
    "both_knockout_hi": "true content-free floor: both branches deleted and the II block "
                        "removed -> plain LightGCN over the bipartite graph with GUME's "
                        "trained ID/user tables (alias: id_only_floor).",
    "graph_knockout": "DIAGNOSTIC: only the joint II block removed, both branches intact. "
                      "Measures the joint term's own price.",
    "image_table_knockout": "DIAGNOSTIC (contract rule 6, C6): the trained image TABLE is "
                            "removed from the score by setting its multiplicative gate to "
                            "the identity 1; the raw-feature image kNN graph is kept.",
    "text_table_knockout": "DIAGNOSTIC (C6): same for the text table. Expected ~0 because "
                           "the text sigmoid gate is saturated at 1 (see docstring 4).",
}

# ------------------------------------------------------------------ pins
# The input-mean screen resolved checkpoints with latest_ckpt() at a time when only the
# June runs existed: its gume/baby baseline R@20 (0.10236603315543916) is the June run
# 20260615_235120 (logged 0.10236603315543918).  latest_ckpt() TODAY returns the July
# re-run 20260712_015306 (seed 2026) instead, so we PIN.
# NOTE FOR THE DRIVER: results/phase_micro/ckpt_pins.json has no gume entry; these pins
# belong there (this module does not write to that file).
CKPT_PIN = {
    "baby": str(RECSYS / "ckpts" / "gume_baby_20260615_235120.pt"),
    "sports": str(RECSYS / "ckpts" / "gume_sports_20260616_015525.pt"),
    "clothing": str(RECSYS / "ckpts" / "gume_clothing_20260616_020517.pt"),
    "microlens": str(RECSYS / "ckpts" / "gume_microlens_20260620_190737.pt"),
}
LOGGED_RUN = {
    "baby": str(RECSYS / "logs" / "gume_baby_20260615_235120" / "result.json"),
    "sports": str(RECSYS / "logs" / "gume_sports_20260616_015525" / "result.json"),
    "clothing": str(RECSYS / "logs" / "gume_clothing_20260616_020517" / "result.json"),
    "microlens": str(RECSYS / "logs" / "gume_microlens_20260620_190737" / "result.json"),
}
# n_layers (conv_ii depth) ACTUALLY used by each trained run; configs/model/gume.yaml
# only carries the baby block, so load_frozen gets sports/clothing wrong.  Read only
# inside conv_ii, so it can be corrected after loading.
DATASET_N_LAYERS = {"baby": 2, "sports": 1, "clothing": 1, "microlens": 2, "elec": 1}

FUSION_DIV = 3.0     # hard-coded in GUME.forward: integration_embeds = (...)/3

_PLAIN_GRAPH_CACHE: dict = {}


# ------------------------------------------------------------------ graph helpers
def _plain_graph(model):
    """(adj, R) for the SAME graph GUME builds, but with an EMPTY item-item block.

    Built with GUME's own `_get_adj_mat` / `_sp2t` so the normalisation code path is
    byte-for-byte the model's; only the item-item block it is fed differs.  Cached per
    model instance (id()), rebuilt cost ~7 s on baby.
    """
    key = id(model)
    hit = _PLAIN_GRAPH_CACHE.get(key)
    if hit is not None:
        return hit
    dev = model.item_id_embedding.weight.device
    empty = sp.coo_matrix((model.n_items, model.n_items), dtype=int).tolil()
    norm_adj, R = model._get_adj_mat(empty)
    out = (model._sp2t(norm_adj).to(dev), model._sp2t(R).to(dev))
    _PLAIN_GRAPH_CACHE[key] = out
    return out


# ------------------------------------------------------------------ decomposition
@torch.no_grad()
def _parts(model, adj=None, R=None, image_gate: str = "model", text_gate: str = "model") -> dict:
    """Re-derive GUME.forward term by term.

    adj / R default to the model's own `gume_norm_adj` / `R` buffers (never rebuilt --
    they were constructed on CPU in __init__ and moved by .to(device), so rebuilding
    them would risk a different topk tie-break).  Pass the `_plain_graph` pair to get
    the counterfactual in which the joint kNN-intersection block is absent.

    image_gate / text_gate: "model" = the trained content table's sigmoid gate;
    "ones"  = the multiplicative identity, i.e. that content TABLE removed from the
    score while its frozen kNN graph stays (C6 arm).
    """
    m = model
    adj = m.gume_norm_adj if adj is None else adj
    R = m.R if R is None else R
    E = m.item_id_embedding.weight

    gv = torch.ones_like(E) if image_gate == "ones" else m.image_space_trans(m.image_embedding.weight)
    gt = torch.ones_like(E) if text_gate == "ones" else m.text_space_trans(m.text_embedding.weight)
    v_item, t_item = E * gv, E * gt

    ext_id = m.conv_ui(adj, m.user_embedding.weight, E)

    xi = m.conv_ii(m.image_original_adj, v_item)
    X = torch.cat([torch.sparse.mm(R, xi), xi], dim=0)
    ti = m.conv_ii(m.text_original_adj, t_item)
    Y = torch.cat([torch.sparse.mm(R, ti), ti], dim=0)

    w = m.softmax(torch.cat([m.separate_coarse(X), m.separate_coarse(Y)], dim=-1))
    a_v, a_t = torch.split(w, 1, dim=-1)
    g_v, g_t = m.image_behavior(ext_id), m.text_behavior(ext_id)

    # GUME's own expression ...
    coarse = a_v * X + a_t * Y
    integ_ref = (g_v * (X - coarse) + g_t * (Y - coarse) + coarse) / FUSION_DIV
    # ... and the collapsed convex form (identity (*) in the docstring)
    c = a_t * g_v - a_v * g_t + a_v
    integ = (c * X + (1.0 - c) * Y) / FUSION_DIV
    ident_res = float((integ - integ_ref).abs().max())

    return {"ext_id": ext_id, "X": X, "Y": Y, "c": c, "a_v": a_v, "g_v": g_v, "g_t": g_t,
            "v_gate": gv, "t_gate": gt, "integ": integ_ref, "fused": ext_id + integ_ref,
            "identity_residual": ident_res}


def _split(model, u_and_i: torch.Tensor):
    return torch.split(u_and_i, [model.n_users, model.n_items], dim=0)


# ------------------------------------------------------------------ contract API
@torch.no_grad()
def variants(model, dataset=None, device=None) -> dict:
    """name -> (user_emb [n_users,d], item_emb [n_items,d]).

    All arms that share a graph are built from ONE forward pass of `_parts`, so their
    differences are exactly paired (no cuSPARSE noise between arms).
    """
    p = _parts(model)                                   # GUME's own graph
    q = _parts(model, *_plain_graph(model))             # joint II block removed
    pv = _parts(model, image_gate="ones")               # image table off the score
    pt = _parts(model, text_gate="ones")                # text  table off the score

    d = FUSION_DIV
    out = {
        "baseline": _split(model, p["fused"]),
        # --- image ---------------------------------------------------------------
        "image_knockout":        _split(model, p["ext_id"] + p["Y"] / d),
        "image_knockout_nonorm": _split(model, p["ext_id"] + (1.0 - p["c"]) * p["Y"] / d),
        "image_knockout_hi":     _split(model, q["ext_id"] + q["Y"] / d),
        # --- text ----------------------------------------------------------------
        "text_knockout":         _split(model, p["ext_id"] + p["X"] / d),
        "text_knockout_nonorm":  _split(model, p["ext_id"] + p["c"] * p["X"] / d),
        "text_knockout_hi":      _split(model, q["ext_id"] + q["X"] / d),
        # --- both ----------------------------------------------------------------
        "both_knockout":         _split(model, p["ext_id"]),
        "both_knockout_hi":      _split(model, q["ext_id"]),
        # --- diagnostics ---------------------------------------------------------
        "graph_knockout":        _split(model, q["fused"]),
        "image_table_knockout":  _split(model, pv["fused"]),
        "text_table_knockout":   _split(model, pt["fused"]),
    }
    return out


@torch.no_grad()
def recon_error(model, dataset=None, device=None, chunk: int = 2048) -> float:
    """max |baseline_scores - GUME.full_sort_predict| over ALL users.

    NOTE the floor: torch.sparse.mm is not bitwise reproducible on CUDA here, so
    full_sort_predict vs itself already differs by ~1.9e-06 on this score scale.
    """
    p = _parts(model)
    u_all, i_all = _split(model, p["fused"])
    dev = i_all.device
    err = 0.0
    for s in range(0, model.n_users, chunk):
        idx = torch.arange(s, min(s + chunk, model.n_users), device=dev)
        ref = model.full_sort_predict({"user": idx})
        mine = u_all[idx] @ i_all.t()
        err = max(err, float((ref - mine).abs().max()))
        del ref, mine
    return err


@torch.no_grad()
def attribution(model, dataset=None, device=None, heavy: bool = True) -> dict:
    p = _parts(model)
    nU = model.n_users
    d = FUSION_DIV

    def mn(x):
        return float(x.norm(dim=-1).mean())

    def st(x):
        x = x.float()
        return {"mean": float(x.mean()), "std": float(x.std()),
                "min": float(x.min()), "max": float(x.max())}

    X, Y, c = p["X"], p["Y"], p["c"]
    img_term = c * X / d
    txt_term = (1.0 - c) * Y / d

    out = {
        "n_layers_conv_ii": int(model.n_layers),
        "n_ui_layers": int(model.n_ui_layers),
        "knn_k": int(model.knn_k),
        "fusion_divisor": d,
        "fusion_divisor_source": "hard-coded /3 in GUME.forward; not learned",

        # ---- additive-stream norms (items only; the ranking is over items) -------
        "||ext_id||_item": mn(p["ext_id"][nU:]),
        "||integration||_item": mn(p["integ"][nU:]),
        "||X_explicit_image||_item": mn(X[nU:]),
        "||Y_explicit_text||_item": mn(Y[nU:]),
        "||c*X/3||_item": mn(img_term[nU:]),
        "||(1-c)*Y/3||_item": mn(txt_term[nU:]),
        "image_share_of_integration_by_norm":
            mn(img_term[nU:]) / (mn(img_term[nU:]) + mn(txt_term[nU:])),
        "||fused||_item": mn(p["fused"][nU:]),

        # ---- fusion coefficients (all LEARNED / input-dependent) ----------------
        "convex_coeff_c_items": st(c[nU:]),
        "convex_coeff_c_users": st(c[:nU]),
        "convex_coeff_c_all": st(c),
        "convex_coeff_definition": "c = a_t*g_v - a_v*g_t + a_v ; integ = (c*X + (1-c)*Y)/3",
        "convex_identity_max_residual": p["identity_residual"],
        "softmax_a_v_items": st(p["a_v"][nU:]),
        "behavior_gate_g_v": st(p["g_v"]),
        "behavior_gate_g_t": st(p["g_t"]),

        # ---- the multiplicative content gates (C2 interface) --------------------
        "image_space_gate": st(p["v_gate"]),
        "text_space_gate": st(p["t_gate"]),
        "image_gate_frac_exactly_one": float((p["v_gate"] == 1.0).float().mean()),
        "text_gate_frac_exactly_one": float((p["t_gate"] == 1.0).float().mean()),
        "text_gate_is_saturated": bool(float(p["t_gate"].min()) > 0.99),
        "gate_note": "gate multiplies the ID embedding elementwise; a saturated gate (==1) "
                     "means the trained content TABLE is inert at inference and only the "
                     "raw-feature kNN graph still carries that modality.",

        # ---- graphs -------------------------------------------------------------
        "graph_nnz": {
            "image_original_adj": int(model.image_original_adj._nnz()),
            "text_original_adj": int(model.text_original_adj._nnz()),
            "gume_norm_adj": int(model.gume_norm_adj._nnz()),
            "R": int(model.R._nnz()),
        },
    }

    # joint-block size and the rescaling it causes (needs the plain graph)
    adj_p, R_p = _plain_graph(model)
    dR = (R_p.to_dense() - model.R.to_dense()).abs()
    out["joint_intersection_block"] = {
        "edges": int(model.gume_norm_adj._nnz() - adj_p._nnz()),
        "edges_per_item": float((model.gume_norm_adj._nnz() - adj_p._nnz()) / model.n_items),
        "R_max_abs_change": float(dR.max()),
        "R_mean_rel_change": float(dR.sum() / model.R.to_dense().abs().sum()),
        "note": "per-item INTERSECTION of the image-kNN and text-kNN neighbourhoods; a "
                "function of BOTH modalities, so not attributable to either.",
    }
    del dR

    # ---- C6: drift of the learnable content tables off the raw encoder features --
    out["C6_freeze_false_tables"] = {}
    for name, raw_attr in (("image", "v_feat"), ("text", "t_feat")):
        tab = getattr(model, f"{name}_embedding").weight
        raw = getattr(model, raw_attr).to(tab.device)
        cos = torch.nn.functional.cosine_similarity(tab, raw, dim=-1)
        out["C6_freeze_false_tables"][name] = {
            "learnable": bool(tab.requires_grad),
            "drift_cos_mean": float(cos.mean()), "drift_cos_min": float(cos.min()),
            "drift_cos_max": float(cos.max()),
            "frac_rows_cos_below_0.99": float((cos < 0.99).float().mean()),
            "norm_ratio_trained_over_raw": float(tab.norm(dim=-1).mean() / raw.norm(dim=-1).mean()),
            "graph_built_from": "RAW feature (v_feat/t_feat) in __init__, NOT this table",
        }

    out["R_is_the_user_item_block_of_norm_adj"] = _R_block_check(model)
    out["renorm_counterfactual_check"] = _renorm_counterfactual_check(model, p)
    out["training_only_certificate"] = _training_only_certificate(model)
    if heavy:
        out["screen_blindspot_certificate"] = screen_blindspot_certificate(model)
    return out


@torch.no_grad()
def _R_block_check(model) -> dict:
    """`R` is the [:n_users, n_users:] block of the SAME normalised matrix as
    `gume_norm_adj` (GUME._get_adj_mat returns both from one normalisation).  So the
    joint item-item block rescales R as well, and the *_hi arms MUST rebuild both --
    keeping the trained R with a plain adj would be an inconsistent hybrid.  Verified
    here bitwise."""
    a = model.gume_norm_adj.coalesce()
    idx, val = a.indices(), a.values()
    nU = model.n_users
    m = (idx[0] < nU) & (idx[1] >= nU)
    sub = torch.sparse_coo_tensor(torch.stack([idx[0][m], idx[1][m] - nU]), val[m],
                                  (nU, model.n_items)).coalesce()
    r = model.R.coalesce()
    same = bool(torch.equal(sub.indices(), r.indices()))
    return {"same_sparsity_pattern": same,
            "max_abs_value_diff": (float((sub.values() - r.values()).abs().max()) if same
                                   else float("nan"))}


@torch.no_grad()
def _renorm_counterfactual_check(model, p=None) -> dict:
    """Show that the renormalised arm IS the model's own single-modality expression.

    Feed GUME's OWN attribute-separation block (its softmax / separate_coarse /
    image_behavior / text_behavior, not our algebra) with X := Y.  Then
    separate_coarse(X) == separate_coarse(Y) => a_v = a_t = 1/2, coarse = Y, both fine
    terms are exactly 0 and integ = Y/3 -- which is precisely `image_knockout`.  The
    same substitution with Y := X gives `text_knockout`.  So "delete the branch and
    renormalise" is not our invention: it is what GUME computes when one branch stops
    carrying its own signal (equivalently, a softmax over a single surviving logit).
    """
    m = model
    p = _parts(m) if p is None else p
    ext_id, X, Y = p["ext_id"], p["X"], p["Y"]
    out = {}
    for name, (A, B) in (("image_knockout", (Y, Y)), ("text_knockout", (X, X))):
        w = m.softmax(torch.cat([m.separate_coarse(A), m.separate_coarse(B)], dim=-1))
        a_v, a_t = torch.split(w, 1, dim=-1)
        coarse = a_v * A + a_t * B
        integ = (m.image_behavior(ext_id) * (A - coarse)
                 + m.text_behavior(ext_id) * (B - coarse) + coarse) / FUSION_DIV
        arm = ext_id + (Y if name == "image_knockout" else X) / FUSION_DIV
        out[name] = {"max_abs_diff_vs_model_own_expression": float((ext_id + integ - arm).abs().max())}
    out["_claim"] = ("each renormalised arm equals GUME's own fusion block evaluated with "
                     "the deleted stream replaced by the surviving one (a_v=a_t=1/2, both "
                     "fine-grained terms exactly 0) -- diffs must be ~1e-7 float noise")
    return out


# ------------------------------------------------------------------ certificates
def _training_only_certificate(model) -> dict:
    """`extended_image_user` / `extended_text_user` must be unreachable from a score.

    Autograd reachability, not bitwise equality: cuSPARSE SpMM is non-deterministic on
    this GPU (see docstring 6), so a bitwise before/after probe would report a false
    'reachable' for every parameter.
    """
    dev = model.item_id_embedding.weight.device
    users = torch.arange(0, min(256, model.n_users), device=dev)
    names = ["extended_image_user", "extended_text_user",
             "image_embedding", "text_embedding", "item_id_embedding", "user_embedding"]
    params = [getattr(model, n).weight for n in names]
    with torch.enable_grad():
        scores = model.full_sort_predict({"user": users})
        grads = torch.autograd.grad(scores.sum(), params, allow_unused=True, retain_graph=False)
    cert = {}
    for n, g in zip(names, grads):
        cert[n] = {"on_inference_path": g is not None,
                   "grad_absmax": (float(g.abs().max()) if g is not None else 0.0)}
    cert["_claim"] = ("extended_image_user / extended_text_user are training-only "
                      "(forward returns extended_it_embeds only when train=True) and must "
                      "show on_inference_path == False")
    return cert


@torch.no_grad()
def screen_blindspot_certificate(model, max_items: int = 12000) -> dict:
    """Prove what the published per-dimension-mean screen did and did not do to GUME.

    (1) The on-path content TABLES are learnable and come from the checkpoint, so the
        screen's substituted feature is overwritten -> the tables are identical in all
        of its arms.  Certified by (a) requires_grad, (b) the table differing from the
        raw feature it was initialised from, (c) the key being present in the loaded
        checkpoint (recsys_bridge refuses to load if a learnable key is missing).
    (2) What the screen DOES change: the frozen kNN graphs.  Under a constant feature
        every cosine is equal, so topk returns the same lowest-id neighbours for every
        item and the image/text INTERSECTION (the joint term) collapses too.  Rebuilt
        here with GUME's own build_sim/build_knn_normalized_graph/_find_inter_add_edge
        on CPU.
    """
    cert = {"tables_untouched_by_screen": {}, "graph_damage": None}
    for name, raw_attr in (("image", "v_feat"), ("text", "t_feat")):
        tab = getattr(model, f"{name}_embedding").weight
        raw = getattr(model, raw_attr).to(tab.device)
        cert["tables_untouched_by_screen"][name] = {
            "table_is_learnable_parameter": bool(tab.requires_grad),
            "table_equals_raw_feature": bool(torch.equal(tab, raw)),
            "table_loaded_from_checkpoint": bool(
                f"{name}_embedding.weight" not in getattr(model, "_missing_keys", [])),
            "conclusion": "screen substitutes the raw feature at __init__, load_state_dict "
                          "then overwrites this table -> screen never removes it",
        }

    if model.n_items > max_items:
        cert["graph_damage"] = {"skipped": f"n_items={model.n_items} > {max_items}"}
        return cert

    gmod = sys.modules.get("_recsys_model_gume")
    if gmod is None:
        cert["graph_damage"] = {"skipped": "GUME source module not in sys.modules"}
        return cert

    v = model.v_feat.detach().cpu()
    t = model.t_feat.detach().cpu()
    v_mean = v.mean(0, keepdim=True).expand_as(v).contiguous()
    _, knn_true_v = gmod.build_knn_normalized_graph(gmod.build_sim(v), model.knn_k, 'sym')
    _, knn_true_t = gmod.build_knn_normalized_graph(gmod.build_sim(t), model.knn_k, 'sym')
    _, knn_mean_v = gmod.build_knn_normalized_graph(gmod.build_sim(v_mean), model.knn_k, 'sym')

    uniq_rows = torch.unique(knn_mean_v, dim=0).shape[0]
    ii_true = model._find_inter_add_edge(knn_true_v, knn_true_t)
    inter_true = ii_true.nnz
    inter_screen = model._find_inter_add_edge(knn_mean_v, knn_true_t).nnz

    # Faithfulness of the *_hi graph counterfactual: rebuilding gume_norm_adj through the
    # SAME _get_adj_mat path but with the TRUE intersection must reproduce the model's own
    # buffer exactly. Then the only difference in the _hi arms is the II block itself.
    reb_adj, reb_R = model._get_adj_mat(ii_true.tolil())
    reb_t = model._sp2t(reb_adj)
    ref = model.gume_norm_adj.cpu().coalesce()
    same_idx = bool(torch.equal(reb_t.indices(), ref.indices()))
    vmax = float((reb_t.values() - ref.values()).abs().max()) if same_idx else float("nan")
    cert["graph_rebuild_faithful"] = {
        "same_sparsity_pattern": same_idx,
        "max_abs_value_diff": vmax,
        "note": "rebuild with the true kNN-intersection reproduces gume_norm_adj bitwise, "
                "so the *_hi arms differ from baseline ONLY by the joint II block",
    }
    del reb_adj, reb_R, reb_t, ref
    cert["graph_damage"] = {
        "image_knn_unique_neighbour_rows_under_mean": int(uniq_rows),
        "image_knn_unique_neighbour_rows_true": int(torch.unique(knn_true_v, dim=0).shape[0]),
        "first_row_under_mean": knn_mean_v[0].tolist(),
        "intersection_edges_true": int(inter_true),
        "intersection_edges_under_image_mean": int(inter_screen),
        "joint_block_destroyed_frac": float(1.0 - inter_screen / max(inter_true, 1)),
        "conclusion": "the screen's 'image' arm degenerates the image kNN graph to one "
                      "constant neighbour row AND wipes out most of the JOINT "
                      "intersection block, while leaving the trained image table intact: "
                      "it is neither an image deletion nor a per-modality measurement.",
    }
    del v, t, v_mean
    return cert


# ------------------------------------------------------------------ self-test runner
def _main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="baby")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--out", default=str(ROOT / "results" / "_scratch" / "exact_ko"))
    args = ap.parse_args(argv)

    from recsys_bridge import load_frozen                              # noqa: E402
    from ranking_effects import evaluate_item_matrix, ranking_change   # noqa: E402

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    ck = CKPT_PIN.get(args.dataset)
    cfg, ds, model, test_loader = load_frozen("gume", args.dataset, device, ckpt_path=ck)
    print(f"loaded ckpt={model._ckpt_name} pinned={model._ckpt_pinned}", flush=True)
    print(f"missing_keys={model._missing_keys}\nunexpected_keys={model._unexpected_keys}", flush=True)

    # config repair (see docstring 7)
    want = DATASET_N_LAYERS.get(args.dataset)
    n_layers_from_yaml = int(model.n_layers)
    if want is not None and want != n_layers_from_yaml:
        print(f"!! n_layers from configs/model/gume.yaml = {n_layers_from_yaml}, but the "
              f"trained run used {want}; correcting (conv_ii depth only).", flush=True)
        model.n_layers = want

    err = recon_error(model, args.dataset, device)
    print(f"recon_error = {err:.3e}  (tol {args.tol:.0e})", flush=True)
    assert err < args.tol, f"reconstruction error {err} exceeds tolerance {args.tol}"

    # measured non-determinism floor of the model's own scoring path
    uu = torch.arange(0, min(2048, model.n_users), device=device)
    with torch.no_grad():
        s1 = model.full_sort_predict({"user": uu})
        s2 = model.full_sort_predict({"user": uu})
        nd = float((s1 - s2).abs().max())
        scale = float(s1.abs().max())
    del s1, s2
    print(f"self-vs-self score noise (cuSPARSE) = {nd:.3e} on score scale {scale:.2f}", flush=True)

    logged = None
    lp = LOGGED_RUN.get(args.dataset)
    if lp and Path(lp).is_file():
        logged = json.loads(Path(lp).read_text())["test_result"]

    vs = variants(model, args.dataset, device)
    res, base_topk = {}, None
    for name, (u, M) in vs.items():
        met, tk, _ = evaluate_item_matrix(u, M, test_loader, device)
        res[name] = {k: float(v) for k, v in met.items()}
        if name == "baseline":
            base_topk = tk
        else:
            res[name]["_ranking_change"] = ranking_change(base_topk, tk, k=20)
        dR = res[name]["Recall@20"] - res["baseline"]["Recall@20"]
        dN = res[name]["NDCG@20"] - res["baseline"]["NDCG@20"]
        print(f"  {name:24s} R@20={res[name]['Recall@20']:.8f} dR@20={dR:+.8f} dN@20={dN:+.8f}",
              flush=True)

    attr = attribution(model, args.dataset, device, heavy=True)
    out = {
        "model": "gume", "dataset": args.dataset,
        "CLASS": CLASS, "CLASS_PARTS": CLASS_PARTS, "CLASS_DESC": CLASS_DESC,
        "EXACTNESS": EXACTNESS, "EXACTNESS_QUALIFIER": EXACTNESS_QUALIFIER,
        "IDENTIFIED": IDENTIFIED, "HEADLINE_BRACKET": HEADLINE_BRACKET, "ARMS": ARMS,
        "ckpt": model._ckpt_path, "ckpt_pinned": model._ckpt_pinned,
        "n_layers_used": int(model.n_layers), "n_layers_from_yaml": n_layers_from_yaml,
        "recon_error": err, "score_selfnoise_max": nd, "score_scale": scale,
        "logged_test_result": logged,
        "baseline_minus_logged_R@20": (res["baseline"]["Recall@20"] - logged["Recall@20"]) if logged else None,
        "baseline_minus_logged_N@20": (res["baseline"]["NDCG@20"] - logged["NDCG@20"]) if logged else None,
        "arms": res, "attribution": attr,
    }
    op = Path(args.out)
    op.mkdir(parents=True, exist_ok=True)
    p = op / f"gume_{args.dataset}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}", flush=True)
    print(json.dumps(attr, indent=2)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
