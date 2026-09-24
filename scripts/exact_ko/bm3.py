"""BM3 (Zhou et al., WWW'23) -- NOT a knockout: an ID-ONLY CERTIFICATE.

==============================================================================
1. What the architecture actually does
==============================================================================
BM3's scoring function is, verbatim (`/workspace/Recsys/src/models/bm3.py`):

    def full_sort_predict(self, interaction):
        users = interaction["user"]
        u_e, i_e = self._propagate()          # user_embedding, item_id_embedding, norm_adj
        u_e = self.predictor(u_e)
        i_e = self.predictor(i_e)
        return u_e[users] @ i_e.t()

and `_propagate` reads exactly three things: `self.user_embedding.weight`,
`self.item_id_embedding.weight` and the interaction graph `self.norm_adj`
(LightGCN-style layer mean, plus an extra `+ item_id_embedding.weight` on the
item side). Neither `v_feat`/`t_feat` nor the trained content tables
(`image_embedding`, `text_embedding`) nor their projections (`image_trs`,
`text_trs`) appear anywhere downstream of a score.

Content enters ONLY `calculate_loss`, as four bootstrap alignment terms
(loss_t, loss_v, loss_tv, loss_vt, weighted by cl_weight=2.0). That is a
training-time objective, not an inference-time pathway.

Consequence: **an inference-time modality knockout is not merely small on BM3 --
it is UNDEFINED.** There is no content term to delete. Any protocol that
perturbs v_feat/t_feat and re-scores must return exactly 0.00000 for every
metric, on every dataset, for every checkpoint, whatever the encoder, whatever
the data. That is a property of the code path, not a measurement of the model.

==============================================================================
2. Why this matters for the paper's counting (the reader-facing point)
==============================================================================
The input-mean screen (`results/phasex_crossarch/crossarch_knockout.json`) reports
BM3 as image_knockout = text_knockout = both_knockout = 0.0000 on baby, sports,
clothing AND elec -- all 16 metrics, all four datasets, bit-for-bit. It was read
as the strongest possible "image is ignored" cell. It is the opposite: it is a
cell with ZERO information content. A protocol that cannot move the score cannot
distinguish "the model ignores image" from "the model depends on image entirely";
the likelihood is flat over both hypotheses. Counting it as supporting evidence
inflates the denominator with a tautology.

  => BM3 must be EXCLUDED from the 12-architecture knockout count, not counted
     as a 0-effect observation. Same for any other model this certificate covers.

The honest statement about BM3 is narrower and still interesting:

  (i)  at inference BM3 is an ID-only graph recommender -- it is literally
       LightGCN + an extra residual + a learned linear predictor head; and
  (ii) content is not thereby "unused": it shapes `item_id_embedding` DURING
       training through the cl_weight-weighted alignment losses. The only
       well-posed counterfactual for BM3 is a RETRAIN with cl_weight=0, which
       is a different (Phase-2 style) experiment and is not attempted here.

Section 5 quantifies the training-time imprint (ii) descriptively -- alignment,
explicitly NOT causal -- so the paper can say something true about BM3 instead of
either an empty zero or nothing.

==============================================================================
3. The certificate (three independent proofs, all run on the real checkpoint)
==============================================================================
P0  DETERMINISM CONTROL. Two consecutive unmodified `full_sort_predict` calls
    must be BITWISE identical. Without this control, "bitwise identical after a
    swap" would be unfalsifiable: if repeated calls already differ, bitwise
    equality can never be reached, and a tolerance test would let a tiny real
    dependence hide under the noise. P0 makes P2 sharp.

    P0 FAILS ON CUDA, AND THAT IS A REAL FINDING (measured here, 2026-09-04).
    `_propagate` uses `torch.sparse.mm` on a COO adjacency; the CUDA kernel
    accumulates with atomics, so two identical calls differ by ~1.2e-07 in the
    scores (~6e-08 in the propagation itself). Measured on baby:

        cuda:0  score bitwise-equal False, max|diff| 1.19e-07
        cpu     score bitwise-equal True,  max|diff| 0.0

    Consequently the FIRST version of this certificate reported
    "P2_scores_bitwise_identical: false" with max diff 8.9e-08 -- SMALLER than
    the model's own repeat-call noise, i.e. a false alarm produced entirely by
    the CUDA kernel. The certificate therefore runs P0/P2 on a CPU copy of the
    same weights (bitwise deterministic, verified by P0_cpu) and additionally
    reports the device-side pair as a noise-floor bracket. Two consequences
    beyond BM3: (a) never claim "bitwise" for a graph model on CUDA without a
    repeat-call control; (b) a `recon_error` of ~1e-07 on any sparse-graph
    architecture is this floor, not an algebraic residual.

P1  AUTOGRAD REACHABILITY, over the FULL parameter set. With grad enabled, take
    d(sum of scores)/d(theta) for EVERY named parameter and for v_feat/t_feat
    (re-bound as leaves that require grad), `allow_unused=True`. Content ->
    None. Non-content -> not None with a non-zero gradient norm. The positive
    control is the load-bearing half: it proves the graph was live and the Nones
    mean "structurally disconnected", not "we forgot to enable grad".

P2  DESTRUCTIVE RANDOM SWAP. Overwrite v_feat, t_feat, image_embedding.weight,
    text_embedding.weight, image_trs.{weight,bias}, text_trs.{weight,bias} with
    fresh randn of matched scale, then re-score. Bitwise identical (given P0).
    This catches any dependence autograd would miss -- non-differentiable use,
    an index built from features, a cached buffer.

P3  DETERMINISTIC RECONSTRUCTION. On CPU, our `baseline` arm must equal
    `full_sort_predict` BITWISE, not just to 1e-07 (see P0).

P4  RE-INSTANTIATION WITH RANDOM FEATURES. P1-P3 only prove content is not READ
    at inference. They are NOT sufficient: a model that consumed content once at
    `__init__` into a frozen kNN graph passes all three, because that graph is a
    non-persistent buffer rebuilt at construction and never touched afterwards.
    P4 closes the hole: build BM3 from scratch with RANDOM v_feat/t_feat, load the
    SAME checkpoint, and compare scores bitwise on CPU. Identical => content never
    influences inference through ANY route, construction included.

    This is not hypothetical. Running P1-P3 unchanged across the whole registry on
    baby (results/_scratch/idonly_sweep_baby.json) labels FREEDOM "ID-only", which
    is wrong -- FREEDOM's content lives in `mm_adj`, built in `__init__`. P4 is what
    separates them (results/_scratch/t3_reinit_baby.json):

        model     P1-P3 "not read at inference"   P4 re-init max|dscore|   truth
        bm3       pass                            0.0        (bitwise)     ID-ONLY
        lightgcn  pass                            0.0        (bitwise)     ID-ONLY (no content at all)
        freedom   pass                           20.81                     C1 construction-time
        mgcn      pass (raw feats dead)           0.291                    C1 + trained tables
        smore     pass (raw feats dead)           3.450                    C1 + trained tables
        gume      pass (raw feats dead)           2.898                    C1 + trained tables

    So of everything cheaply testable, **BM3 is the only multimodal architecture in
    the study whose knockout zero is structural.** LightGCN is ID-only by design and
    is the paper's control, not one of the 12. A second, separate lesson from that
    sweep: for MGCN/SMORE/GUME the gradient w.r.t. the RAW v_feat/t_feat is None --
    their content lives in `from_pretrained(..., freeze=False)` tables restored from
    the checkpoint -- so the input-mean screen only reaches them through the
    construction-time graph, not through the substituted features it thinks it is
    perturbing. (CONTRACT rule 6 territory; flagged, not handled here.)

Note on the checkpoint: `image_embedding`/`text_embedding` are
`nn.Embedding.from_pretrained(feat, freeze=False)` and ARE trained and stored, so
P2's swap destroys genuinely-learned state, not just a copy of the raw feature.
`attribution` reports the drift cosine (CONTRACT rule 6) to show how far those
tables moved -- and they still cannot touch a score.

==============================================================================
4. Contract compliance
==============================================================================
CLASS      = "C4"   content only in training-time objectives; off the inference
                    path. (C5 in the mmgcn.py legend is reserved for models with
                    no content parameters at all, e.g. LightGCN; BM3 HAS content
                    parameters, they are simply inference-dead.)
EXACTNESS  = "UNDEFINED_ID_ONLY"
variants() = {"baseline"} only. CONTRACT rule 7: do not fabricate a knockout.
             The random-swap arm is deliberately NOT returned as a variant --
             it would score exactly 0.0 by construction and any downstream
             aggregator would silently re-create the bogus 0-effect cell this
             module exists to remove. It lives in the certificate instead.
recon_error() = max |u@i.T - full_sort_predict| (still a real check: our baseline
             must reproduce the model's own scoring path).

Writes only under MechInterp; /workspace/Recsys is read-only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path("/workspace/MechInterp")
RECSYS = Path("/workspace/Recsys")
# Both repos use the top-level package name `src`; `src` MUST resolve to Recsys.
for _p in (str(RECSYS), str(ROOT / "src" / "models"), str(ROOT / "src" / "interp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CLASS = "C4"
CLASS_DESC = ("C4 = content is consumed ONLY by training-time objectives (BM3's "
              "cl_weight-weighted bootstrap alignment losses). The scoring function is a "
              "function of user_embedding, item_id_embedding and the interaction graph alone.")
EXACTNESS = "UNDEFINED_ID_ONLY"

ARMS = {
    "baseline": "model as trained, re-derived from parts (asserted == full_sort_predict). "
                "The ONLY arm: BM3 has no content term at inference, so no knockout exists.",
}

# Names of the content-side parameters. Every one of these must be autograd-unreachable
# from a score and destroyable without changing a score.
_CONTENT_PARAM_PREFIXES = ("image_embedding", "text_embedding", "image_trs", "text_trs")
# What the score is actually allowed to depend on.
_ONPATH_PARAM_PREFIXES = ("user_embedding", "item_id_embedding", "predictor")

_USER_CHUNK = 2048
_CERT_USERS = 512          # users used for the autograd / bitwise probes


# ------------------------------------------------------------------ decomposition
@torch.no_grad()
def _streams(model) -> dict:
    """BM3's inference-time parts. There is no content stream to split out.

    _propagate():  out = mean_l A^l [U;I]  ->  (u_e, i_e); item side gets `+ h`
                   where h = item_id_embedding.weight (a second, un-propagated copy).
    scoring:       predictor(u_e) @ predictor(i_e + h).T
    """
    u_g, i_g = model._propagate()                  # i_g already includes `+ h`
    h = model.item_id_embedding.weight.detach()
    return {"u_graph": u_g.detach(), "i_graph_plus_h": i_g.detach(), "h_id": h,
            "i_graph_only": (i_g - h).detach(),
            "u": model.predictor(u_g).detach(),
            "i": model.predictor(i_g).detach()}


# ------------------------------------------------------------------ contract API
@torch.no_grad()
def variants(model, dataset, device) -> dict:
    """ONLY 'baseline'. See CONTRACT rule 7 and section 2 of the module docstring."""
    s = _streams(model)
    return {"baseline": (s["u"], s["i"])}


@torch.no_grad()
def recon_error(model, dataset, device) -> float:
    """max |baseline_scores - BM3.full_sort_predict| over ALL users x items."""
    s = _streams(model)
    return _max_score_err(model, s["u"], s["i"])


def attribution(model, dataset, device) -> dict:
    """Stream norms + the three-part ID-only certificate + training-imprint alignment."""
    out = {
        "interface": "content enters calculate_loss ONLY (4 bootstrap alignment terms); "
                     "full_sort_predict is a function of user_embedding, item_id_embedding "
                     "and norm_adj alone",
        "fusion": "none at inference -- there is no modality fusion on the scoring path",
        "class": CLASS, "class_desc": CLASS_DESC, "exactness": EXACTNESS,
        "cl_weight": float(getattr(model, "cl_weight", float("nan"))),
        "n_layers": int(getattr(model, "n_layers", -1)),
        "knockout_defined": False,
        "screen_zero_is_structural": True,
        "screen_zero_note": (
            "The published input-mean screen reports exactly 0.0000 for image/text/both on "
            "all 4 datasets. That is forced by the code path, not measured: v_feat/t_feat "
            "are not read by full_sort_predict, so ANY input perturbation returns 0. The "
            "cell carries no evidence in either direction and must be EXCLUDED from the "
            "12-architecture count, not counted as support for 'image is ignored'."),
        "well_posed_counterfactual": (
            "retrain with cl_weight=0 (content losses removed) and compare -- a Phase-2 "
            "style retraining experiment, NOT an inference-time knockout. Not run here."),
    }
    with torch.no_grad():
        s = _streams(model)
        def mn(x): return float(x.norm(dim=-1).mean())
        out.update({
            "||u_graph||": mn(s["u_graph"]),
            "||i_graph_only||": mn(s["i_graph_only"]),
            "||h_id (item_id_embedding)||": mn(s["h_id"]),
            "||i_graph_plus_h||": mn(s["i_graph_plus_h"]),
            "||u_scored (predictor)||": mn(s["u"]),
            "||i_scored (predictor)||": mn(s["i"]),
            "||h_img (image stream on scoring path)||": 0.0,
            "||h_txt (text stream on scoring path)||": 0.0,
        })
        out["stream_note"] = ("both content stream norms are 0.0 by CONSTRUCTION (no such "
                              "term exists), not by measurement -- see the certificate.")
    out["state_inventory"] = _state_inventory(model, dataset)
    out["certificate"] = certificate(model, device)
    out["training_imprint"] = _training_imprint(model)
    return out


# ------------------------------------------------------------------ the certificate
def certificate(model, device, n_users: int = _CERT_USERS, seed: int = 0) -> dict:
    """P0 determinism control, P1 autograd reachability, P2 destructive random swap.

    P0/P2 are run TWICE: on a CPU copy of the same weights (bitwise deterministic --
    this is the sharp test) and on the model's own device (reported as a noise-floor
    bracket, because CUDA sparse.mm is not bitwise reproducible; see the docstring).
    """
    import copy                                                       # noqa: PLC0415
    cert: dict = {"n_probe_users": int(min(n_users, model.n_users)), "seed": seed}

    # ---- P1: autograd reachability over the FULL parameter set (device is irrelevant)
    idx = torch.arange(min(n_users, model.n_users), device=device)
    cert.update(_autograd_probe(model, idx))

    # ---- P0 + P2 on the model's own device: noise-floor bracket -------------------
    dev_res = _p0_p2(model, device, n_users, seed)
    cert["P0_repeat_call_bitwise_identical"] = dev_res["p0_equal"]
    cert["P0_repeat_call_max_abs_diff"] = dev_res["p0_max"]
    cert["P2_scores_bitwise_identical"] = dev_res["p2_equal"]
    cert["P2_max_abs_score_diff"] = dev_res["p2_max"]
    cert["P2_swapped_tensors"] = dev_res["swapped"]
    cert["P2_diff_within_repeat_noise"] = bool(dev_res["p2_max"] <= dev_res["p0_max"])
    cert["device"] = str(device)

    # ---- P0 + P2 on a CPU copy: the SHARP bitwise test ---------------------------
    if str(device) == "cpu":
        cpu_res = dev_res
        cert["P3_cpu_recon_bitwise_identical"] = None
    else:
        m_cpu = copy.deepcopy(model).to("cpu").eval()
        # P3: on the deterministic device our re-derivation should match the model's own
        # scoring path BITWISE, not merely to 1e-07. This separates "we reconstructed the
        # algebra exactly" from "the CUDA kernel is noisy".
        with torch.no_grad():
            s_cpu = _streams(m_cpu)
            j = torch.arange(min(n_users, m_cpu.n_users))
            ours = s_cpu["u"][j] @ s_cpu["i"].t()
            theirs = m_cpu.full_sort_predict({"user": j})
            cert["P3_cpu_recon_bitwise_identical"] = bool(torch.equal(ours, theirs))
            cert["P3_cpu_recon_max_abs_diff"] = float((ours - theirs).abs().max())
            del ours, theirs, s_cpu
        cpu_res = _p0_p2(m_cpu, "cpu", n_users, seed)
        del m_cpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    cert["P0_cpu_repeat_call_bitwise_identical"] = cpu_res["p0_equal"]
    cert["P0_cpu_repeat_call_max_abs_diff"] = cpu_res["p0_max"]
    cert["P2_cpu_scores_bitwise_identical"] = cpu_res["p2_equal"]
    cert["P2_cpu_max_abs_score_diff"] = cpu_res["p2_max"]

    cert["P3_note"] = (
        "reconstruction of the model's own scoring path on the deterministic device. "
        "Bitwise-equal => the ~1e-07 recon_error reported on CUDA is the sparse.mm noise "
        "floor, not an algebraic residual in this module.")
    cert["P0_note"] = (
        "control: bitwise equality of two identical calls. FALSE on CUDA -- torch.sparse.mm "
        "on a COO adjacency accumulates with atomics (~1.2e-07 on baby scores). TRUE on CPU. "
        "Without this control, P2's CUDA 'not bitwise identical' (8.9e-08, i.e. BELOW this "
        "noise floor) would have been misread as a content dependence.")
    cert["P2_note"] = (
        "v_feat/t_feat AND the trained (freeze=False) content tables AND their projections "
        "were replaced with noise. On CPU the scores are bitwise identical => no dependence "
        "of any kind, differentiable or not. On CUDA the residual is at/below the model's "
        "own repeat-call noise floor (P2_diff_within_repeat_noise).")

    sharp = bool(cpu_res["p0_equal"] and cpu_res["p2_equal"])
    grads_ok = bool(cert.get("P1_all_content_grads_none") and cert.get("P1_onpath_grads_live"))
    if sharp and grads_ok:
        cert["verdict"] = "CONTENT IS OFF THE INFERENCE PATH (bitwise, CPU-deterministic)"
    elif grads_ok and cert["P2_diff_within_repeat_noise"]:
        cert["verdict"] = ("CONTENT OFF-PATH by autograd; swap residual only bracketed by "
                           "the device noise floor (no deterministic control available)")
    else:
        cert["verdict"] = "CERTIFICATE FAILED -- do not report BM3 as ID-only"
    return cert


def _p0_p2(model, device, n_users: int, seed: int) -> dict:
    """Repeat-call control (P0) and destructive random swap (P2) on one device."""
    idx = torch.arange(min(n_users, model.n_users), device=device)
    with torch.no_grad():
        a = model.full_sort_predict({"user": idx}).clone()
        b = model.full_sort_predict({"user": idx}).clone()
    p0_equal, p0_max = bool(torch.equal(a, b)), float((a - b).abs().max())
    del b
    swap = _random_swap_probe(model, idx, a, seed)
    del a
    return {"p0_equal": p0_equal, "p0_max": p0_max,
            "p2_equal": swap["P2_scores_bitwise_identical"],
            "p2_max": swap["P2_max_abs_score_diff"],
            "swapped": swap["P2_swapped_tensors"]}


def _autograd_probe(model, idx) -> dict:
    """d(sum scores)/d(theta) for every parameter + v_feat/t_feat rebound as leaves."""
    was_training = model.training
    model.eval()
    saved = {"v_feat": getattr(model, "v_feat", None), "t_feat": getattr(model, "t_feat", None)}
    leaves: dict[str, torch.Tensor] = {}
    try:
        for k in ("v_feat", "t_feat"):
            if saved[k] is not None:
                leaf = saved[k].detach().clone().requires_grad_(True)
                setattr(model, k, leaf)
                leaves[k] = leaf

        names, params = zip(*[(n, p) for n, p in model.named_parameters()])
        with torch.enable_grad():
            scores = model.full_sort_predict({"user": idx})
            loss = scores.sum()
            grads = torch.autograd.grad(loss, list(params) + list(leaves.values()),
                                        allow_unused=True, retain_graph=False)
        all_names = list(names) + [f"<buffer>{k}" for k in leaves]
        gnorm = {n: (None if g is None else float(g.norm()))
                 for n, g in zip(all_names, grads)}
    finally:
        for k, v in saved.items():
            if v is not None:
                setattr(model, k, v)
        if was_training:
            model.train()

    def is_content(n: str) -> bool:
        return n.startswith(_CONTENT_PARAM_PREFIXES) or n.startswith("<buffer>")

    content = {n: g for n, g in gnorm.items() if is_content(n)}
    onpath = {n: g for n, g in gnorm.items() if n.startswith(_ONPATH_PARAM_PREFIXES)}
    other = {n: g for n, g in gnorm.items() if n not in content and n not in onpath}
    return {
        "P1_grad_norms": gnorm,
        "P1_content_params_checked": sorted(content),
        "P1_all_content_grads_none": all(v is None for v in content.values()),
        "P1_onpath_grads_live": (len(onpath) > 0
                                 and all(v is not None and v > 0 for v in onpath.values())),
        "P1_onpath_params": sorted(onpath),
        "P1_unclassified_params": {n: v for n, v in other.items()},
        "P1_note": ("None => the parameter is not in the score's autograd graph at all. The "
                    "live on-path gradients are the positive control that grad mode was on "
                    "and the graph was built, so the Nones are structural."),
    }


def _random_swap_probe(model, idx, ref_scores: torch.Tensor, seed: int) -> dict:
    """Overwrite every content tensor with randn of matched scale; re-score; compare."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    saved_buf = {k: getattr(model, k, None) for k in ("v_feat", "t_feat")}
    saved_par: dict[str, torch.Tensor] = {}
    swapped: list[str] = []
    try:
        with torch.no_grad():
            for k, v in saved_buf.items():
                if v is not None:
                    r = torch.randn(v.shape, generator=g).to(v.device, v.dtype)
                    r = r * (v.std() + 1.0) + v.mean()          # matched-ish scale, destroyed content
                    setattr(model, k, r)
                    swapped.append(k)
            for n, p in model.named_parameters():
                if n.startswith(_CONTENT_PARAM_PREFIXES):
                    saved_par[n] = p.detach().clone()
                    r = torch.randn(p.shape, generator=g).to(p.device, p.dtype)
                    p.copy_(r * (p.detach().std() + 1.0))
                    swapped.append(n)
            after = model.full_sort_predict({"user": idx})
            same = bool(torch.equal(ref_scores, after))
            maxd = float((ref_scores - after).abs().max())
            del after
    finally:
        with torch.no_grad():
            for k, v in saved_buf.items():
                if v is not None:
                    setattr(model, k, v)
            for n, p in model.named_parameters():
                if n in saved_par:
                    p.copy_(saved_par[n])
    return {"P2_swapped_tensors": swapped,
            "P2_scores_bitwise_identical": same,
            "P2_max_abs_score_diff": maxd,
            }


# ------------------------------------------------------------------ state provenance
@torch.no_grad()
def _state_inventory(model, dataset) -> dict:
    """Every tensor the module holds, classified -- plus proof that the ONE non-trivial
    buffer on the scoring path (`norm_adj`) is interaction-derived, not content-derived.

    This is the cheap, dataset-local half of P4: if no buffer other than norm_adj exists,
    there is nowhere for a construction-time content artifact (a kNN graph like FREEDOM's
    mm_adj) to hide. The full P4 re-instantiation lives in `p4_reinit_certificate`.
    """
    params = {n: list(p.shape) for n, p in model.named_parameters()}
    bufs = {n: list(b.shape) for n, b in model.named_buffers() if b is not None}
    content = sorted(n for n in list(params) + list(bufs)
                     if n.startswith(_CONTENT_PARAM_PREFIXES) or n in ("v_feat", "t_feat"))
    onpath = sorted(n for n in params if n.startswith(_ONPATH_PARAM_PREFIXES))
    other_bufs = sorted(n for n in bufs if n not in ("v_feat", "t_feat"))
    out = {"parameters": params, "buffers": bufs,
           "content_tensors": content, "onpath_parameters": onpath,
           "non_feature_buffers": other_bufs,
           "no_content_derived_buffer": other_bufs == ["norm_adj"]}
    try:
        from src.data.graph_utils import build_norm_adj                # noqa: PLC0415
        ref = build_norm_adj(dataset.train_matrix, dataset.n_users, dataset.n_items)
        ref = ref.coalesce().to(model.norm_adj.device)
        na = model.norm_adj.coalesce()
        out["norm_adj_is_interaction_derived"] = bool(
            torch.equal(na.indices(), ref.indices()) and torch.equal(na.values(), ref.values()))
        del ref
    except Exception as e:                                             # noqa: BLE001
        out["norm_adj_is_interaction_derived"] = f"uncheckable: {type(e).__name__}: {e}"
    out["note"] = ("norm_adj is rebuilt from dataset.train_matrix alone (no features). With "
                   "no other buffer present, BM3 has no construction-time content artifact -- "
                   "unlike FREEDOM's mm_adj. Confirmed end-to-end by p4_reinit_certificate.")
    return out


# ------------------------------------------------------------------ P4 re-instantiation
def p4_reinit_certificate(dataset_name: str = "baby", ckpt_path=None, n_users: int = _CERT_USERS,
                          seed_feat: int = 1234) -> dict:
    """Build BM3 twice on CPU -- real features vs. random features -- from the SAME
    checkpoint, and compare scores bitwise.

    Mirrors `recsys_bridge.load_frozen` step for step (same Config, same set_seed, same
    strict=False + missing-learnable assert); the ONLY difference between the two builds
    is the content features handed to __init__. Kept local because load_frozen has no
    feature-override hook and Recsys/ is read-only -- see the report note.
    """
    import numpy as np                                                 # noqa: PLC0415
    from src.utils import Config, set_seed                             # noqa: PLC0415
    from src.data.dataset import RecDataset                            # noqa: PLC0415
    from src.data.graph_utils import build_norm_adj                    # noqa: PLC0415
    from recsys_bridge import load_recsys_model_class, latest_ckpt, SCRATCH  # noqa: PLC0415

    def _build(randomize: bool):
        cfg = Config("bm3", dataset_name, cli_overrides={
            "ckpt_dir": str(SCRATCH / "ckpts"), "log_dir": str(SCRATCH / "logs")})
        set_seed(int(cfg.get("seed", 2024)),
                 deterministic=bool(cfg.get("cudnn_deterministic", True)))
        ds = RecDataset(cfg)
        adj = build_norm_adj(ds.train_matrix, ds.n_users, ds.n_items)
        v = torch.from_numpy(ds.v_feat[:].copy()) if ds.v_feat is not None else None
        t = torch.from_numpy(ds.t_feat[:].copy()) if ds.t_feat is not None else None
        if randomize:
            g = torch.Generator().manual_seed(seed_feat)
            if v is not None:
                v = torch.randn(v.shape, generator=g) * (v.std() + 1.0) + v.mean()
            if t is not None:
                t = torch.randn(t.shape, generator=g) * (t.std() + 1.0) + t.mean()
        m = load_recsys_model_class("bm3")(config=cfg, n_users=ds.n_users, n_items=ds.n_items,
                                           norm_adj=adj, v_feat=v, t_feat=t)
        ck = Path(ckpt_path) if ckpt_path else latest_ckpt("bm3", dataset_name)
        st = torch.load(ck, map_location="cpu", weights_only=False)
        inc = m.load_state_dict(st["model_state_dict"], strict=False)
        pn = {n for n, _ in m.named_parameters()}
        bad = [k for k in inc.missing_keys if k in pn]
        if bad:
            raise RuntimeError(f"missing LEARNABLE parameters {bad}")
        m.eval()
        return m, ck.name, list(inc.missing_keys)

    m0, ck_name, missing = _build(False)
    m1, _, _ = _build(True)
    j = torch.arange(min(n_users, m0.n_users))
    with torch.no_grad():
        a = m0.full_sort_predict({"user": j})
        a2 = m0.full_sort_predict({"user": j})
        b = m1.full_sort_predict({"user": j})
    res = {"ckpt": ck_name, "n_probe_users": int(j.numel()),
           "rebuilt_buffers_not_in_ckpt": sorted(missing),
           "cpu_determinism_control": bool(torch.equal(a, a2)),
           "P4_bitwise_identical": bool(torch.equal(a, b)),
           "P4_max_abs_score_diff": float((a - b).abs().max()),
           "note": ("both builds load the SAME checkpoint; only __init__'s v_feat/t_feat "
                    "differ. Bitwise equality rules out construction-time consumption "
                    "(the route by which FREEDOM/MGCN/SMORE/GUME pass P1-P3 yet are "
                    "content-dependent: 20.81 / 0.291 / 3.450 / 2.898 max|dscore|).")}
    res["verdict"] = ("TRUE ID-ONLY: content never reaches inference, even via __init__"
                      if res["P4_bitwise_identical"] and res["cpu_determinism_control"]
                      else "P4 FAILED -- content-derived state is on the path")
    del m0, m1, a, a2, b
    return res


# ------------------------------------------------------------------ training imprint
@torch.no_grad()
def _training_imprint(model) -> dict:
    """DESCRIPTIVE, NOT CAUSAL. Content is inference-dead but shaped training.

    Two numbers:
      * drift cosine  cos(trained content table row, raw v_feat/t_feat row)  -- CONTRACT
        rule 6: shows the freeze=False tables really did move off the encoder features.
      * alignment cos(predictor(i_e)_j , predictor(content_j)) vs. a shuffled-item control
        -- the residue of BM3's alignment losses in the ID embeddings. A gap over the
        control means "content left a fingerprint on the ID table during training"; it is
        NOT a causal claim about inference and must never be reported as an effect size.
    """
    out = {"is_causal": False,
           "note": "alignment, not causation; the only causal test is a cl_weight=0 retrain"}
    s = _streams(model)
    i_scored = torch.nn.functional.normalize(s["i"], dim=-1)
    perm = torch.randperm(i_scored.shape[0], device=i_scored.device)
    for mod, tab, trs, feat in (("image", "image_embedding", "image_trs", "v_feat"),
                                ("text", "text_embedding", "text_trs", "t_feat")):
        if getattr(model, feat, None) is None:
            continue
        W = getattr(model, tab).weight.detach()
        raw = getattr(model, feat).detach()
        drift = torch.nn.functional.cosine_similarity(W, raw, dim=-1)
        proj = model.predictor(getattr(model, trs)(W)).detach()
        proj = torch.nn.functional.normalize(proj, dim=-1)
        al = (i_scored * proj).sum(-1)
        al_ctrl = (i_scored * proj[perm]).sum(-1)
        out[mod] = {
            "drift_cos_mean": float(drift.mean()), "drift_cos_min": float(drift.min()),
            "align_cos_mean": float(al.mean()),
            "align_cos_shuffled_control": float(al_ctrl.mean()),
            "align_gap_over_control": float(al.mean() - al_ctrl.mean()),
        }
    return out


# ------------------------------------------------------------------ helpers
@torch.no_grad()
def _max_score_err(model, u_all: torch.Tensor, item_matrix: torch.Tensor) -> float:
    worst = 0.0
    n = u_all.shape[0]
    for lo in range(0, n, _USER_CHUNK):
        idx = torch.arange(lo, min(lo + _USER_CHUNK, n), device=u_all.device)
        ours = u_all[idx] @ item_matrix.t()
        theirs = model.full_sort_predict({"user": idx})
        worst = max(worst, float((ours - theirs).abs().max()))
        del ours, theirs
    return worst


# ------------------------------------------------------------------ standalone check
def _selfcheck(dataset_name: str = "baby", gpu: int = 0, out_dir: str | None = None) -> dict:
    from recsys_bridge import load_frozen                            # noqa: PLC0415
    from ranking_effects import evaluate_item_matrix                 # noqa: PLC0415

    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    pins = json.loads((ROOT / "results" / "phase_micro" / "ckpt_pins.json").read_text())
    pin = pins.get(f"bm3/{dataset_name}", {}).get("path")
    cfg, ds, model, loader = load_frozen("bm3", dataset_name, device, ckpt_path=pin)

    rec = recon_error(model, ds, device)
    attr = attribution(model, ds, device)
    arms = variants(model, ds, device)
    assert set(arms) == {"baseline"}, "ID-only module must expose exactly one arm"

    p4 = p4_reinit_certificate(dataset_name, ckpt_path=pin)
    assert p4["ckpt"] == model._ckpt_name, "P4 loaded a different checkpoint"
    attr["certificate"]["P4_reinit_random_features"] = p4

    res, _, _ = evaluate_item_matrix(*arms["baseline"], loader, device)
    base_r20 = float(res["Recall@20"])
    logged = json.loads((RECSYS / "logs" / model._ckpt_name.replace(".pt", "") /
                         "result.json").read_text())["test_result"]

    # The swap arm is evaluated ONLY as a certificate line, never as a variant:
    # its dR@20 is 0.0 by construction and must not enter any knockout tally.
    out = {"model": "bm3", "dataset": dataset_name, "class": CLASS, "exactness": EXACTNESS,
           "ckpt": model._ckpt_name, "ckpt_pinned": bool(pin),
           "recon_error": rec,
           "baseline_Recall@20": base_r20,
           "logged_test_Recall@20": float(logged["Recall@20"]),
           "baseline_minus_logged": base_r20 - float(logged["Recall@20"]),
           "baseline_all_metrics": {k: float(v) for k, v in res.items()},
           "arms": {"baseline": {k: float(v) for k, v in res.items()}},
           "knockout_arms": None,
           "knockout_arms_reason": ("UNDEFINED: no content term exists on BM3's scoring "
                                    "path. See attribution.certificate."),
           "certificate_verdict": attr["certificate"]["verdict"],
           "p4_verdict": p4["verdict"],
           "attribution": attr}
    d = Path(out_dir or (ROOT / "results" / "_scratch" / "exact_ko"))
    d.mkdir(parents=True, exist_ok=True)
    (d / f"bm3_{dataset_name}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="baby")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    _selfcheck(a.dataset, a.gpu, a.out)
