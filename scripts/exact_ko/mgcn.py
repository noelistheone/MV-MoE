"""Exact structural knockout for MGCN (Yu et al., ACM MM 2023).

SOURCE OF TRUTH: /workspace/Recsys/src/models/mgcn.py  (READ-ONLY; verified line by line)
Reference calibration: scripts/phase1_knockout.py (FREEDOM/LGMRec).

==============================================================================
1. What MGCN actually computes at inference
==============================================================================
`full_sort_predict` -> `_propagate()`, verbatim:

    image_feats = image_trs(image_embedding.weight)          # 4096 -> 64  (TRAINED table!)
    text_feats  = text_trs(text_embedding.weight)            #  384 -> 64  (TRAINED table!)
    Iv = item_id_embedding.weight * gate_v(image_feats)      # MULTIPLICATIVE sigmoid gate
    Tv = item_id_embedding.weight * gate_t(text_feats)
    content_embeds = mean_l LightGCN^l(norm_adj)             # pure ID/CF, users+items
    Iv = image_adj^n_layers @ Iv ;  I = [R @ Iv ; Iv]        # frozen content kNN graph
    Tv = text_adj^n_layers  @ Tv ;  T = [R @ Tv ; Tv]
    a  = softmax([q(I), q(T)])[:,0]                          # ONE shared scalar per node
    common   = a*I + (1-a)*T
    sep_img  = p_v * (I - common) ;  sep_txt = p_t * (T - common)
        with p_v = gate_image_prefer(content_embeds), p_t = gate_text_prefer(content_embeds)
    side_embeds = (sep_img + sep_txt + common) / 3
    all_embeds  = content_embeds + side_embeds ;  score = all_u @ all_i^T

==============================================================================
2. The exact bilinear decomposition (measured, not assumed)
==============================================================================
Collecting terms (p_v, p_t elementwise in (0,1)^d; a a per-node scalar):

    3 * side = p_v*I + p_t*T + (1 - p_v - p_t) * (a*I + (1-a)*T)
             = [p_v + (1-p_v-p_t)*a] * I  +  [p_t + (1-p_v-p_t)*(1-a)] * T

    side = A (*) I + B (*) T,
      A = (p_v*(1-a) + a*(1-p_t)) / 3 > 0
      B = (p_t*a   + (1-a)*(1-p_v)) / 3 > 0
      A + B == 1/3   EXACTLY, elementwise, for every node and every dimension.

Verified on baby: max|side - (A*I + B*T)| = 7.5e-09, max|A+B-1/3| = 3.0e-08.
So MGCN splits a FIXED side-information budget of 1/3 between the two views. The whole
knockout question reduces to: when one view is deleted, does its share of the budget go
to the survivor, or is it destroyed?

==============================================================================
3. Why this is DELETE_PLUS_RENORM, and how tight the bracket is
==============================================================================
The only cross-modal coupling is the 2-way softmax `a`. Deleting a view leaves it with a
single logit, so the softmax renormalises to 1 -- that is a SHARED NORMALISER in the sense
of CONTRACT rule 3. Both ends are emitted:

  `<m>_knockout`        RENORMALISED = the model's own single-modality counterfactual.
                        With image gone: common = T, sep_txt = T - common = 0, sep_img
                        absent  ->  side = T/3, i.e. B' = 1/3. The budget A+B=1/3 is
                        conserved and handed to the survivor.
  `<m>_knockout_nonorm` MASS REMOVED. Drop the A(*)I term with A, B, a frozen at their
                        baseline values -> side = B(*)T. Exactly the image mass and
                        nothing else is gone; total side mass drops.

MEASURED bracket on baby (mean over nodes/dims):
    A = 0.173889, B = 0.159445 (A+B = 1/3);  a = 0.500042 +/- 0.001647 (min .4770 max .5207)
    surviving-coefficient ratio  image arm: (1/3)/A = 1.917x ;  text arm: (1/3)/B = 2.091x
i.e. the softmax itself is within 0.01% of an even split, but the RENORMALISATION still
doubles the survivor's weight -- the bracket is a factor ~2 in coefficient space, NOT the
~1.5x the source survey estimated. In dR@20 the two ends are much closer (see 6).

`both_knockout` has NO ambiguity: with both views gone side = 0 and all_embeds =
content_embeds (pure LightGCN). `both_knockout_nonorm` is bitwise identical to it, and
this module asserts that -- so the both-arm is EXACT.

==============================================================================
4. TWO content interfaces, and the C6 hazard (this is the headline finding)
==============================================================================
Content reaches the score through two structurally different doors that read two
DIFFERENT tensors:

  (i)  GRAPH door (class C1). `image_adj = build_knn_graph(self.v_feat, knn_k)` is built in
       __init__ from the RAW feature buffer and registered `persistent=False`. It is a
       frozen operator; the raw feature is never read again at inference.
  (ii) GATE door (class C2, multiplicative). `image_embedding =
       nn.Embedding.from_pretrained(v_feat.clone(), freeze=False)` is a TRAINABLE PARAMETER
       initialised at the raw feature. It is what `image_trs` reads. It IS in the
       checkpoint (`image_embedding.weight`, verified).

CONSEQUENCE FOR THE PUBLISHED SCREEN (CONTRACT rule 6, taxonomy C6). The screen
(scripts/phasex_crossarch_knockout.py) re-instantiates the model with v -> per-dim mean and
then calls `load_state_dict(..., strict=False)`. Because `image_embedding.weight` is a
learnable parameter present in the checkpoint, THE TRAINED TABLE IS RESTORED OVER THE
MEAN-INITIALISED ONE: the gate door never sees the perturbation. What the screen actually
perturbs is only the graph door, whose buffer is non-persistent and therefore rebuilt from
the mean feature into a DEGENERATE graph (all pairwise cosines equal -> top-k is decided by
index tie-breaking, not by content).

This module certifies that claim two independent ways:
  * `attribution()["c6"]` -- the table is a parameter, it is in the checkpoint, and its
    drift cosine against the raw feature is reported.
  * `<m>_screen_meanfeat` -- an arm that changes ONLY `image_adj`/`text_adj` to the
    mean-feature graph and touches nothing else. On baby it reproduces the published
    screen deltas BITWISE:
        image  -0.00056998371475101839  (published -0.00056998371475101839, |diff| 0.0)
        text   -0.00060212565355276249  (published -0.00060212565355276249, |diff| 0.0)
        both   -0.00097497214365305229  (published -0.00097497214365305229, |diff| 0.0)
    A graph-only edit reproducing the screen exactly IS the proof that the screen's
    feature-path edit was a no-op.
  * `_main()` additionally re-instantiates MGCN with the mean feature, loads the same
    checkpoint, and shows `image_embedding.weight` and the on-path tensor
    `image_trs(image_embedding.weight)` come out BITWISE IDENTICAL to the unperturbed
    model, while `image_adj` does not.

So the published mgcn screen entry is not "image knocked out". It is "the image kNN graph
replaced by an index-order graph, with the trained content table left fully intact". It is
a graph-degeneracy probe, not a content measurement -- structurally uninformative about the
gate door and confounded on the graph door.

==============================================================================
5. Extra arms that separate the two doors (both EXACT, no shared normaliser touched)
==============================================================================
`<m>_gate_knockout`   Delete the content term inside the gate's first affine map:
                      image_trs(E) = W E + b  ->  b. The view still exists, the softmax
                      still has two inputs, everything downstream recomputes naturally, so
                      this is an EXACT deletion (no renormalisation ambiguity). It removes
                      100% of the content that flows through the gate door.
`<m>_graph_knockout`  Replace the frozen content kNN graph by the identity (no propagation).
                      There is no content-free kNN graph to substitute, so this deletes the
                      content-derived SMOOTHING together with the smoothing itself: read it
                      as an upper bound on the graph door, not a pure content measurement.
`<m>_graph_random`    Control for the arm above: a kNN graph with identical k, symmetrisation
                      and Laplacian normalisation, built from N(0,1) features (seed
                      GRAPH_SEED). Same structural capacity, zero content. The interval
                      [graph_random, baseline] is the content-attributable part of the graph
                      door; [graph_knockout, graph_random] is the price of smoothing itself.

==============================================================================
6. The gate door is SATURATED -- MGCN is, in effect, a C1 architecture
==============================================================================
Measured on baby: the gate pre-activation is dominated by content (||W f|| = 67.3 vs
||b|| = 0.13 for image), but the sigmoid is saturated:

                                     image      text
    mean gate value                  0.9979     0.9981
    frac. of coordinates > 0.99      0.9887     0.9871
    across-item sd of the gate       0.0264     0.0292
    ||E*gate - E|| / ||E||           0.0085     0.0019     (pre-graph gate distortion)

So `gate_v(image_trs(E_img))` is a near-constant all-ones multiplier: it rescales the ID
embedding by <1% and transmits almost no item-specific content. The C2 door is behaviourally
dead (`image_gate_knockout` dR@20 = -0.000103, `text_gate_knockout` -0.000077) while the C1
door is 10-20x larger (`image_graph_knockout` -0.000977, `text_graph_knockout` -0.001504).
On baby, MGCN's "gated multi-view" fusion therefore degenerates at convergence to FREEDOM's
interface: a frozen content kNN graph smoothing the ID table.

Scope: the IMAGE gate is saturated on sports too (mean 0.992, 91.0% of coordinates > 0.99,
distortion 2.4%), but the TEXT gate on sports is NOT (mean 0.855, 0.03% > 0.99, distortion
16.5%). So "the gate door is dead" is a per-(dataset, modality) measurement this module
makes, not an architectural law. It is dead for image on both datasets checked.

==============================================================================
7. Measured on baby (ckpt mgcn_baby_20260523_001339, logged test R@20 0.0753058094444055)
==============================================================================
    baseline reproduces the logged test R@20 AND NDCG@20 to EXACTLY 0.0 (bitwise)
    recon_error vs full_sort_predict = 4.768e-07, which is EXACTLY the model's own
    self-consistency floor (same value on 5 repeats) and exactly 1 float32 ulp at
    max|score| = 4.1437.  rel = 1.15e-07; top-20 sets agree on 4096/4096 users.
    So the decomposition error is 0 to the limit of what CUDA sparse-mm can express.

    arm                       dR@20        dN@20
    image_knockout          +0.000008   +0.000147     <- renormalised (headline)
    image_knockout_nonorm   -0.000377   -0.000086     <- mass removed
    image_gate_knockout     -0.000103   -0.000003     <- EXACT, C2 door
    image_graph_knockout    -0.000977   -0.000164     <- C1 door, upper bound
    image_graph_random      -0.000626   -0.000094     <- content-free graph control
    image_screen_meanfeat   -0.000570   -0.000124     <- the published number, reproduced
    text_knockout           -0.000576   -0.000310
    text_knockout_nonorm    -0.000791   -0.000328
    text_gate_knockout      -0.000077   -0.000056     <- EXACT, C2 door
    text_graph_knockout     -0.001504   -0.000471
    text_graph_random       -0.000774   -0.000325
    text_screen_meanfeat    -0.000602   -0.000308     <- the published number, reproduced
    both_knockout           -0.001095   -0.000423     <- EXACT (pure LightGCN)
    both_gate_knockout      -0.000180   -0.000084
    both_graph_knockout     -0.001701   -0.000526
    both_screen_meanfeat    -0.000975   -0.000383     <- the published number, reproduced

    Every arm is inside the paper's noise hint (0.0026 R@20) and inside |both_knockout|
    = 0.0011, the whole side-information branch. The image bracket [-0.000377, +0.000008]
    STRADDLES ZERO: on MGCN/baby the image door is not measurably used, and that verdict
    no longer rests on the broken screen. Text is used, but only through the graph door.

    Note the *_graph_random control: replacing the image graph by a CONTENT-FREE kNN graph
    of the same k already costs -0.000626 of the -0.000977 that removing the operator
    outright costs. Only ~36% of the image graph door's effect is attributable to content;
    the rest is the smoothing operator itself. For text the split is -0.000774 of -0.001504,
    i.e. ~49% attributable to content.

==============================================================================
8. Also run on sports (ckpt mgcn_sports_20260523_025748) as a generality check
==============================================================================
    baseline - logged test R@20 = -1.4e-17 (1 double ulp), NDCG@20 = 0.0
    recon_error 2.4e-07 (self-floor 1.2e-07, i.e. 2 ulp at max|score| 1.231); top-20 sets
    agree on 4096/4096 users.  a = 0.5000018 +/- 0.0002744; A = 0.17973, B = 0.15361.
    The three *_screen_meanfeat arms again reproduce the published sports screen deltas
    BITWISE (image -4.213719871902499e-05, text exactly 0.0, both -3.745528775024598e-05).
    Every knockout arm is <= 5.5e-04 in |dR@20| -- an order of magnitude below the noise
    hint -- and `image_graph_knockout` is POSITIVE (+0.000552): on sports, removing MGCN's
    image kNN graph improves Recall@20.

==============================================================================
9. Content-interface taxonomy
==============================================================================
Legend as fixed in scripts/exact_ko/mmgcn.py:
    C1  frozen content-derived item-item graph; content off-path at inference (FREEDOM)
    C2  live content feature projected into the scored embedding (VBPR, LGMRec)
    C3  content as the layer-0 node feature of a per-modality GNN tower
    C4  content only in training-time objectives
    C5  ID-only: content never reaches the scoring function
    C6  (hazard flag, CONTRACT rule 6) the on-path content tensor is a freeze=False table
MGCN = "C1+C2", like COHESION, but with two twists: the C2 half is a MULTIPLICATIVE sigmoid
gate rather than an additive term, and the two doors read DIFFERENT tensors -- C1 reads the
raw frozen buffer, C2 reads the trained C6 table. That split is exactly what breaks the
input-mean screen.
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

from src.data.graph_utils import build_knn_graph              # noqa: E402

# ---------------------------------------------------------------- contract fields
CLASS = "C1+C2"
CLASS_PARTS = {"graph_door": "C1 (frozen kNN graph from the RAW v_feat/t_feat buffer)",
               "gate_door": "C2-multiplicative (sigmoid gate on a trained C6 table)"}
CLASS_DESC = ("Two content doors reading two different tensors: a frozen content kNN graph "
              "built at __init__ from the raw feature buffer (C1), and a multiplicative "
              "sigmoid gate on item_id_embedding driven by a freeze=False trainable content "
              "table restored from the checkpoint (C2 + C6 hazard).")
EXACTNESS = "DELETE_PLUS_RENORM"   # view-level arms (2-way softmax renormalises).
                                   # both_knockout and every *_gate_knockout arm are EXACT.

GRAPH_SEED = 12345                 # for the content-free kNN control graph

# Only one mgcn/baby run exists on disk, so latest_ckpt() already resolves to it; pinned
# anyway so this module measures the same model the input-mean screen reported.
# NOTE FOR THE DRIVER: results/phase_micro/ckpt_pins.json has no mgcn entry; these pins
# should be added there (this module does not write to that file).
CKPT_PIN = {
    "baby":      str(RECSYS / "ckpts" / "mgcn_baby_20260523_001339.pt"),
    "sports":    str(RECSYS / "ckpts" / "mgcn_sports_20260523_025748.pt"),
    "clothing":  str(RECSYS / "ckpts" / "mgcn_clothing_20260523_060330.pt"),
    "microlens": str(RECSYS / "ckpts" / "mgcn_microlens_20260621_014603.pt"),
    # elec has TWO runs on disk; the input-mean screen's latest_ckpt() took 20260602_124923.
    "elec":      str(RECSYS / "ckpts" / "mgcn_elec_20260602_124923.pt"),
}

ARMS = {
    "baseline": "model as trained, re-derived from parts (asserted == full_sort_predict)",
    "image_knockout": "HEADLINE / renormalised: the image VIEW is deleted; the 2-way softmax "
                      "collapses to the text logit, so side = T/3 and the 1/3 side budget is "
                      "handed to text. The model's own single-modality counterfactual.",
    "image_knockout_nonorm": "MASS REMOVED: drop the A(*)I term with A, B, a frozen at "
                             "baseline -> side = B(*)T. Exactly the image mass is gone.",
    "image_gate_knockout": "EXACT: delete the content term inside image_trs (W E + b -> b), "
                           "so the gate becomes a constant vector. The view, its graph and "
                           "the softmax are untouched.",
    "image_graph_knockout": "the frozen image kNN graph is replaced by the identity (no "
                            "propagation). Upper bound on the graph door: removes the "
                            "smoothing itself as well as its content.",
    "image_graph_random": "control for the arm above: same k, symmetrisation and "
                          "normalisation, graph built from N(0,1) features (GRAPH_SEED).",
    "image_screen_meanfeat": "bitwise reproduction of the PUBLISHED coarse screen: only "
                             "image_adj is rebuilt from the per-dim mean feature; the trained "
                             "image_embedding table stays on the path (C6).",
    "text_knockout": "as image_knockout, text view",
    "text_knockout_nonorm": "as image_knockout_nonorm, text view",
    "text_gate_knockout": "as image_gate_knockout, text view",
    "text_graph_knockout": "as image_graph_knockout, text view",
    "text_graph_random": "as image_graph_random, text view",
    "text_screen_meanfeat": "input-mean screen, text",
    "both_knockout": "EXACT: both views deleted -> side = 0 -> all_embeds = content_embeds "
                     "(pure LightGCN). No renormalisation ambiguity exists here.",
    "both_knockout_nonorm": "the nonorm reading of the same arm; asserted BITWISE IDENTICAL "
                            "to both_knockout (certificate that the both-arm is exact).",
    "both_gate_knockout": "content deleted in BOTH gates; both graphs kept",
    "both_graph_knockout": "both content kNN graphs replaced by the identity",
    "both_screen_meanfeat": "input-mean screen, both",
}
BRACKET = {
    "image": ("image_knockout_nonorm", "image_knockout"),
    "text": ("text_knockout_nonorm", "text_knockout"),
    "image_graph_door": ("image_graph_knockout", "image_graph_random"),
    "text_graph_door": ("text_graph_knockout", "text_graph_random"),
}
EXACT_ARMS = ["both_knockout", "image_gate_knockout", "text_gate_knockout",
              "both_gate_knockout"]


# ---------------------------------------------------------------- forward replay
def _mean_feat(feat: torch.Tensor) -> torch.Tensor:
    """The input-mean screen's perturbation, verbatim (phasex_crossarch_knockout.run_model)."""
    return feat.mean(0, keepdim=True).expand_as(feat).contiguous()


@torch.no_grad()
def _graphs(model) -> dict:
    """Every item-item operator this module needs, all built on CPU then moved.

    build_knn_graph is CPU/GPU-float sensitive (cosine top-k tie-breaking), and the model
    built image_adj/text_adj on CPU inside __init__ (v_feat was still on CPU then), so any
    graph we compare against them MUST be built on CPU too.
    """
    dev = model.item_id_embedding.weight.device
    out = {"v": {"real": model.image_adj, "identity": None},
           "t": {"real": model.text_adj, "identity": None}}
    for key, feat in (("v", model.v_feat), ("t", model.t_feat)):
        f = feat.detach().cpu()
        out[key]["screen"] = build_knn_graph(_mean_feat(f), model.knn_k).to(dev)
        g = torch.Generator().manual_seed(GRAPH_SEED)
        rnd = torch.randn(f.shape, generator=g, dtype=torch.float32)
        out[key]["random"] = build_knn_graph(rnd, model.knn_k).to(dev)
    return out


@torch.no_grad()
def _parts(model, graph: Dict[str, str] | None = None, gate_del: Tuple[str, ...] = (),
           graphs: dict | None = None) -> dict:
    """Byte-faithful replay of MGCN._propagate with switchable content doors.

    graph[key] in {'real','identity','screen','random'}   (the C1 door)
    key in gate_del -> the content term inside <m>_trs is deleted (the C2 door)
    """
    m = model
    graph = graph or {}
    if graphs is None and any(v != "real" for v in graph.values()):
        graphs = _graphs(m)

    pre_v = m.image_trs(m.image_embedding.weight)
    if "v" in gate_del:
        pre_v = m.image_trs.bias.unsqueeze(0).expand_as(pre_v)
    pre_t = m.text_trs(m.text_embedding.weight)
    if "t" in gate_del:
        pre_t = m.text_trs.bias.unsqueeze(0).expand_as(pre_t)

    Iv = m.item_id_embedding.weight * m.gate_v(pre_v)
    Tv = m.item_id_embedding.weight * m.gate_t(pre_t)

    ego = torch.cat([m.user_embedding.weight, m.item_id_embedding.weight], dim=0)
    stack = [ego]
    for _ in range(m.n_ui_layers):
        ego = torch.sparse.mm(m.norm_adj, ego)
        stack.append(ego)
    content = torch.stack(stack, dim=1).mean(dim=1)

    for key, blk in (("v", "Iv"), ("t", "Tv")):
        mode = graph.get(key, "real")
        x = Iv if key == "v" else Tv
        if mode != "identity":
            adj = m.image_adj if (key == "v" and mode == "real") else \
                  m.text_adj if (key == "t" and mode == "real") else graphs[key][mode]
            for _ in range(m.n_layers):
                x = torch.sparse.mm(adj, x)
        if key == "v":
            Iv = x
        else:
            Tv = x

    I = torch.cat([torch.sparse.mm(m.R, Iv), Iv], dim=0)
    T = torch.cat([torch.sparse.mm(m.R, Tv), Tv], dim=0)
    att = torch.cat([m.query_common(I), m.query_common(T)], dim=-1)
    a = F.softmax(att, dim=-1)[:, 0:1]
    pv = m.gate_image_prefer(content)
    pt = m.gate_text_prefer(content)
    A = (pv + (1.0 - pv - pt) * a) / 3.0
    B = (pt + (1.0 - pv - pt) * (1.0 - a)) / 3.0
    return {"I": I, "T": T, "a": a, "pv": pv, "pt": pt, "A": A, "B": B, "content": content}


def _side(p: dict, mode: str = "full") -> torch.Tensor:
    """side_embeds under one of the deletion readings (see docstring section 3)."""
    if mode == "full":
        common = p["a"] * p["I"] + (1.0 - p["a"]) * p["T"]
        return (p["pv"] * (p["I"] - common) + p["pt"] * (p["T"] - common) + common) / 3.0
    if mode == "drop_v_renorm":      # softmax collapses to the text logit -> side = T/3
        return p["T"] / 3.0
    if mode == "drop_t_renorm":
        return p["I"] / 3.0
    if mode == "drop_v_nonorm":      # remove exactly the image mass A(*)I
        return p["B"] * p["T"]
    if mode == "drop_t_nonorm":
        return p["A"] * p["I"]
    if mode == "drop_both":
        return torch.zeros_like(p["content"])
    raise ValueError(f"unknown side mode {mode!r}")


def _split(model, allv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return allv[:model.n_users], allv[model.n_users:]


@torch.no_grad()
def _emb(model, graph=None, gate_del=(), side_mode="full", graphs=None):
    p = _parts(model, graph=graph, gate_del=gate_del, graphs=graphs)
    out = _split(model, p["content"] + _side(p, side_mode))
    del p
    return out


# ---------------------------------------------------------------- contract API
@torch.no_grad()
def variants(model, dataset=None, device=None) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    g = _graphs(model)
    # Every arm that only changes the SIDE READING (baseline / view knockouts) is derived
    # from ONE `_parts` call. torch.sparse.mm on CUDA is not bit-reproducible, so deriving
    # them from separate forward passes would inject a ~1 ulp difference into arms that are
    # supposed to be exactly comparable (and would make the both_knockout identity below
    # fail for a reason that has nothing to do with the architecture).
    p = _parts(model, graphs=g)
    c = p["content"]
    out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {
        "baseline": _split(model, c + _side(p, "full")),
        "image_knockout": _split(model, c + _side(p, "drop_v_renorm")),
        "image_knockout_nonorm": _split(model, c + _side(p, "drop_v_nonorm")),
        "text_knockout": _split(model, c + _side(p, "drop_t_renorm")),
        "text_knockout_nonorm": _split(model, c + _side(p, "drop_t_nonorm")),
        "both_knockout": _split(model, c + _side(p, "drop_both")),
        "both_knockout_nonorm": _split(model, c + _side(p, "drop_both")),
    }
    # CONTRACT rule 3 / docstring section 3: with both views gone side = 0 under BOTH
    # readings, so the renormalisation ambiguity vanishes and the both-arm is EXACT.
    assert torch.equal(out["both_knockout"][1], out["both_knockout_nonorm"][1]), \
        "both_knockout and its nonorm reading must be bitwise identical"
    assert float(_side(p, "drop_both").abs().max()) == 0.0
    del p, c
    # Arms that change a content DOOR need their own forward pass.
    for mod, key in (("image", "v"), ("text", "t")):
        out[f"{mod}_gate_knockout"] = _emb(model, gate_del=(key,), graphs=g)
        out[f"{mod}_graph_knockout"] = _emb(model, graph={key: "identity"}, graphs=g)
        out[f"{mod}_graph_random"] = _emb(model, graph={key: "random"}, graphs=g)
        out[f"{mod}_screen_meanfeat"] = _emb(model, graph={key: "screen"}, graphs=g)
    out["both_gate_knockout"] = _emb(model, gate_del=("v", "t"), graphs=g)
    out["both_graph_knockout"] = _emb(model, graph={"v": "identity", "t": "identity"}, graphs=g)
    out["both_screen_meanfeat"] = _emb(model, graph={"v": "screen", "t": "screen"}, graphs=g)
    del g
    return out


@torch.no_grad()
def attribution(model, dataset=None, device=None) -> dict:
    nU = model.n_users
    g = _graphs(model)
    p = _parts(model, graphs=g)
    side = _side(p, "full")

    def mni(x):
        return float(x[nU:].norm(dim=-1).mean())

    def mnu(x):
        return float(x[:nU].norm(dim=-1).mean())

    A, B, I, T = p["A"], p["B"], p["I"], p["T"]
    a = {
        # ---- additive stream norms (item side unless noted) --------------------
        "||content_i|| (LightGCN/ID)": mni(p["content"]),
        "||content_u||": mnu(p["content"]),
        "||side_i||": mni(side), "||side_u||": mnu(side),
        "||image_view_i|| (I)": mni(I), "||text_view_i|| (T)": mni(T),
        "||A*I||_i (image stream in side)": mni(A * I),
        "||B*T||_i (text stream in side)": mni(B * T),
        "side_share_of_all_i": mni(side) / (mni(p["content"] + side) + 1e-12),
        # ---- fusion coefficients ----------------------------------------------
        "softmax_a_mean": float(p["a"].mean()), "softmax_a_sd": float(p["a"].std()),
        "softmax_a_min": float(p["a"].min()), "softmax_a_max": float(p["a"].max()),
        "A_mean": float(A.mean()), "B_mean": float(B.mean()),
        "gate_image_prefer_mean": float(p["pv"].mean()),
        "gate_text_prefer_mean": float(p["pt"].mean()),
        "A_plus_B_minus_one_third_max_abs": float((A + B - 1.0 / 3.0).abs().max()),
        "side_bilinear_decomposition_max_err": float((side - (A * I + B * T)).abs().max()),
        # ---- how wide is the DELETE_PLUS_RENORM bracket, in coefficient space --
        "renorm_gain_image": float((1.0 / 3.0) / A.mean()),
        "renorm_gain_text": float((1.0 / 3.0) / B.mean()),
        "n_layers": int(model.n_layers), "n_ui_layers": int(model.n_ui_layers),
        "knn_k": int(model.knn_k),
    }

    # ---- the two doors, per modality ------------------------------------------
    doors = {}
    for mod, key, trs, gate, feat, tbl, adj in (
            ("image", "v", model.image_trs, model.gate_v, model.v_feat,
             model.image_embedding.weight, model.image_adj),
            ("text", "t", model.text_trs, model.gate_t, model.t_feat,
             model.text_embedding.weight, model.text_adj)):
        pre_full = trs(tbl)
        pre_del = trs.bias.unsqueeze(0).expand_as(pre_full)
        gf, gd = gate(pre_full), gate(pre_del)
        wf = pre_full - trs.bias.unsqueeze(0)
        sc = adj.coalesce()
        sr = g[key]["screen"].coalesce()
        rr = g[key]["random"].coalesce()
        def edge_set(x):
            ii = x.indices()
            return set(map(tuple, ii.t().cpu().tolist()))
        e_real, e_screen, e_rand = edge_set(sc), edge_set(sr), edge_set(rr)
        doors[mod] = {
            # --- gate door (C2) ---
            "||W f||_mean (content part of the gate pre-activation)": float(wf.norm(dim=-1).mean()),
            "||b_trs||": float(trs.bias.norm()),
            "content_share_of_gate_preactivation":
                float(wf.norm(dim=-1).mean() / pre_full.norm(dim=-1).mean()),
            "||gate(full)||_mean": float(gf.norm(dim=-1).mean()),
            "||gate(content deleted)||_mean": float(gd.norm(dim=-1).mean()),
            "cos(gate_full, gate_content_deleted)_mean":
                float(F.cosine_similarity(gf, gd, dim=-1).mean()),
            "gate_rows_identical_after_deletion_max_delta": float((gd - gd[0:1]).abs().max()),
            # How much item-specific information does the multiplicative gate actually
            # transmit?  gate() in (0,1)^64, so ||gate||=8 means every coordinate is
            # saturated to 1 and the gate is a constant multiplier carrying NO content.
            "gate_mean_value": float(gf.mean()),
            "gate_across_item_sd_mean": float(gf.std(dim=0).mean()),
            "gate_frac_coords_saturated_gt0.99": float((gf > 0.99).float().mean()),
            "gate_frac_coords_saturated_lt0.01": float((gf < 0.01).float().mean()),
            "||I - item_id_embedding||/||item_id_embedding|| (pre-graph, gate distortion)":
                float(((model.item_id_embedding.weight * gf - model.item_id_embedding.weight)
                       .norm(dim=-1) / model.item_id_embedding.weight.norm(dim=-1)).mean()),
            # --- graph door (C1) ---
            "graph_edges_real": int(sc.indices().shape[1]),
            "graph_edges_screen_meanfeat": int(sr.indices().shape[1]),
            "graph_edges_random": int(rr.indices().shape[1]),
            "graph_edge_jaccard(real, screen)":
                len(e_real & e_screen) / max(1, len(e_real | e_screen)),
            "graph_edge_jaccard(real, random)":
                len(e_real & e_rand) / max(1, len(e_real | e_rand)),
            "graph_built_from": "RAW frozen buffer (v_feat/t_feat), NOT the trained table",
        }
    a["doors"] = doors

    # ---- CONTRACT rule 6: the C6 hazard, certified ----------------------------
    param_ids = {id(q) for q in model.parameters()}
    ck_keys = set()
    ckpt = getattr(model, "_ckpt_path", None)
    if ckpt is not None:
        try:
            ck_keys = set(torch.load(ckpt, map_location="cpu",
                                     weights_only=False)["model_state_dict"].keys())
        except Exception:                                            # noqa: BLE001
            ck_keys = set()
    a["c6"] = {
        "image_embedding_is_trainable_parameter":
            id(model.image_embedding.weight) in param_ids and model.image_embedding.weight.requires_grad,
        "text_embedding_is_trainable_parameter":
            id(model.text_embedding.weight) in param_ids and model.text_embedding.weight.requires_grad,
        "image_embedding.weight_in_checkpoint": "image_embedding.weight" in ck_keys,
        "text_embedding.weight_in_checkpoint": "text_embedding.weight" in ck_keys,
        "drift_cos(image table, raw v_feat)":
            float(F.cosine_similarity(model.image_embedding.weight, model.v_feat, dim=-1).mean()),
        "drift_cos(text table, raw t_feat)":
            float(F.cosine_similarity(model.text_embedding.weight, model.t_feat, dim=-1).mean()),
        "drift_max_abs(image table - raw v_feat)":
            float((model.image_embedding.weight - model.v_feat).abs().max()),
        "drift_max_abs(text table - raw t_feat)":
            float((model.text_embedding.weight - model.t_feat).abs().max()),
        "v_feat_is_persistent_buffer": False,   # register_buffer(..., persistent=False)
        "image_adj_is_persistent_buffer": False,
        "verdict": ("the screen's input-mean substitution is OVERWRITTEN on the gate door by "
                    "load_state_dict (parameter present in the checkpoint) and only reaches "
                    "the non-persistent graph door -- see the *_screen_meanfeat arms, which "
                    "reproduce the published deltas with a GRAPH-ONLY edit"),
    }
    a["training_only_not_ablated"] = ["InfoNCE cl_loss (calculate_loss only)", "EmbLoss reg"]
    a["stochastic_ops_at_inference"] = "none (no dropout, no gumbel; _propagate is deterministic)"
    del p, side, g
    return a


# ---------------------------------------------------------------- reconstruction
@torch.no_grad()
def _recon_pair(model, u, i, chunk: int = 4096) -> float:
    dev = u.device
    worst = 0.0
    for s in range(0, model.n_users, chunk):
        users = torch.arange(s, min(s + chunk, model.n_users), device=dev)
        ref = model.full_sort_predict({"user": users})
        worst = max(worst, float((u[users] @ i.t() - ref).abs().max()))
        del ref
    return worst


@torch.no_grad()
def recon_error(model, dataset=None, device=None, chunk: int = 4096) -> float:
    """max |baseline_scores - model.full_sort_predict| over ALL users, in chunks."""
    u, i = _emb(model)
    err = _recon_pair(model, u, i, chunk)
    del u, i
    return err


@torch.no_grad()
def recon_error_detail(model, dataset=None, device=None, chunk: int = 4096,
                       repeats: int = 5) -> dict:
    """recon_error plus the model's OWN self-consistency floor.

    torch.sparse.mm on CUDA is not bit-reproducible, so scoring the model against itself
    already disagrees at some epsilon. Any decomposition error below that floor is
    unmeasurable, so both numbers are reported together."""
    u, i = _emb(model)
    mu, mi, _, _ = model._propagate()
    err = _recon_pair(model, u, i, chunk)
    floors = [_recon_pair(model, mu, mi, chunk) for _ in range(repeats)]
    scale = float((u[:chunk] @ i.t()).abs().max())
    users = torch.arange(0, min(4096, model.n_users), device=u.device)
    mine_k = torch.topk(u[users] @ i.t(), 20, dim=-1).indices
    ref_k = torch.topk(model.full_sort_predict({"user": users}), 20, dim=-1).indices
    same = [set(x.tolist()) == set(y.tolist()) for x, y in zip(mine_k.cpu(), ref_k.cpu())]
    out = {"recon_error": err, "self_floor_max": max(floors), "self_floor_runs": floors,
           "rel_recon_error": err / max(scale, 1e-12), "max_abs_score": scale,
           "top20_set_agreement_frac": float(sum(same)) / len(same),
           "top20_users_checked": len(same)}
    del u, i, mu, mi, mine_k, ref_k
    return out


# ---------------------------------------------------------------- C6 certificate
@torch.no_grad()
def screen_certificate(model, cfg, dataset, device) -> dict:
    """Re-instantiate MGCN the way the input-mean screen does (v -> per-dim mean, then
    load_state_dict) and show what the perturbation actually reaches.

    Expected: image_embedding.weight and the on-path tensor image_trs(image_embedding.weight)
    come back BITWISE IDENTICAL to the unperturbed model, while image_adj does not."""
    import numpy as np
    from recsys_bridge import load_recsys_model_class
    from src.data.graph_utils import build_norm_adj

    v = torch.from_numpy(dataset.v_feat[:].copy())
    t = torch.from_numpy(dataset.t_feat[:].copy())
    norm_adj = build_norm_adj(dataset.train_matrix, dataset.n_users, dataset.n_items)
    ModelCls = load_recsys_model_class("mgcn")
    pert = ModelCls(config=cfg, n_users=dataset.n_users, n_items=dataset.n_items,
                    norm_adj=norm_adj, v_feat=_mean_feat(v), t_feat=t).to(device)
    state = torch.load(model._ckpt_path, map_location=device,
                       weights_only=False)["model_state_dict"]
    pert.load_state_dict(state, strict=False)
    pert.eval()

    on_path_ref = model.image_trs(model.image_embedding.weight)
    on_path_pert = pert.image_trs(pert.image_embedding.weight)
    real_e = set(map(tuple, model.image_adj.coalesce().indices().t().cpu().tolist()))
    pert_e = set(map(tuple, pert.image_adj.coalesce().indices().t().cpu().tolist()))
    out = {
        "table_bitwise_identical_after_load":
            bool(torch.equal(pert.image_embedding.weight, model.image_embedding.weight)),
        "on_path_image_feats_bitwise_identical": bool(torch.equal(on_path_pert, on_path_ref)),
        "on_path_image_feats_max_abs_delta": float((on_path_pert - on_path_ref).abs().max()),
        "constructed_v_feat_max_abs_delta_vs_real": float((pert.v_feat - model.v_feat).abs().max()),
        "image_adj_edge_jaccard_vs_real": len(real_e & pert_e) / max(1, len(real_e | pert_e)),
        "image_adj_changed": bool(real_e != pert_e),
        "interpretation": ("the screen's perturbation is a NO-OP on the gate door "
                           "(feature path) and a total rewrite of the graph door"),
    }
    del pert, state, on_path_ref, on_path_pert
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- standalone run
def _main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="baby")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-screen-cert", action="store_true")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    from recsys_bridge import load_frozen                             # noqa: E402
    from ranking_effects import evaluate_item_matrix, ranking_change  # noqa: E402

    pins_path = ROOT / "results" / "phase_micro" / "ckpt_pins.json"
    pin = args.ckpt
    if pin is None:
        pins = json.loads(pins_path.read_text()) if pins_path.is_file() else {}
        pin = pins.get(f"mgcn/{args.dataset}", {}).get("path") or CKPT_PIN.get(args.dataset)
    cfg, ds, model, test_loader = load_frozen("mgcn", args.dataset, device, ckpt_path=pin)
    print(f"ckpt = {model._ckpt_name}  (pinned={model._ckpt_pinned})", flush=True)
    print(f"missing keys = {model._missing_keys}", flush=True)

    det = recon_error_detail(model, ds, device)
    print("recon: " + json.dumps(det), flush=True)

    attr = attribution(model, ds, device)
    print(json.dumps(attr, indent=2), flush=True)

    cert = None
    if not args.no_screen_cert:
        cert = screen_certificate(model, cfg, ds, device)
        print("screen certificate: " + json.dumps(cert, indent=2), flush=True)

    vs = variants(model, ds, device)
    res, base_topk, base_m = {}, None, None
    for name, (u, i) in vs.items():
        m, topk, _ = evaluate_item_matrix(u, i, test_loader, device)
        m = {k: float(v) for k, v in m.items()}
        if name == "baseline":
            base_topk, base_m = topk, m
            res[name] = {"metrics": m}
        else:
            res[name] = {"metrics": m, "delta": {k: m[k] - base_m[k] for k in m},
                         "ranking_change": ranking_change(base_topk, topk)}
        d = "" if name == "baseline" else \
            f"  dR@20={m['Recall@20'] - base_m['Recall@20']:+.6f}  dN@20={m['NDCG@20'] - base_m['NDCG@20']:+.6f}"
        print(f"  {name:26s} R@20={m['Recall@20']:.6f}  N@20={m['NDCG@20']:.6f}{d}", flush=True)

    logged = json.loads((RECSYS / "logs" / model._ckpt_name.replace(".pt", "") /
                         "result.json").read_text())["test_result"]
    # Cross-check the *_screen_meanfeat arms against the input-mean screen.
    screen_cmp = {}
    pub_path = ROOT / "results" / "phasex_crossarch" / "crossarch_knockout.json"
    if pub_path.is_file():
        pub = {f"{r.get('model')}/{r.get('dataset')}": r for r in json.loads(pub_path.read_text())}
        row = pub.get(f"mgcn/{args.dataset}")
        if row and "error" not in row:
            for mod in ("image", "text", "both"):
                arm = f"{mod}_screen_meanfeat"
                if arm in res and f"{mod}_knockout" in row:
                    mine = res[arm]["delta"]["Recall@20"]
                    theirs = float(row[f"{mod}_knockout"]["Recall@20"])
                    screen_cmp[mod] = {"mine_dR@20": mine, "published_dR@20": theirs,
                                       "abs_diff": abs(mine - theirs)}
            screen_cmp["baseline_abs_diff"] = abs(base_m["Recall@20"]
                                                  - float(row["baseline"]["Recall@20"]))

    out = {"model": "mgcn", "dataset": args.dataset, "CLASS": CLASS,
           "CLASS_PARTS": CLASS_PARTS, "EXACTNESS": EXACTNESS,
           "exact_arms": EXACT_ARMS, "ckpt": model._ckpt_path, "recon": det,
           "logged_test": {k: float(v) for k, v in logged.items()},
           "baseline_minus_logged": {k: base_m[k] - float(logged[k]) for k in base_m
                                     if k in logged},
           "attribution": attr, "screen_certificate": cert,
           "published_screen_reproduction": screen_cmp,
           "arms": res, "arm_docs": ARMS, "bracket": BRACKET}
    path = Path(args.out) if args.out else (ROOT / "results" / "_scratch" /
                                            f"exact_ko_mgcn_{args.dataset}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
