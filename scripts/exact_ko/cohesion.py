"""COHESION (SIGIR'25) — exact structural knockout module.

Source read: /workspace/Recsys/src/models/cohesion.py  (verified line by line).

WHAT THE SCORING FUNCTION IS
----------------------------
`full_sort_predict` recomputes the training propagation and scores with a plain
inner product over a 192-d **concatenation of three 64-d blocks**:

    id_rep = id_gcn(id_feat, id_feat, A_ui).data          # [n_nodes, 64]
    v_rep  = v_gcn (v_feat , id_feat, A_ui)               # [n_nodes, 64]
    t_rep  = t_gcn (t_feat , id_feat, A_ui)               # [n_nodes, 64]
    rep    = [ id_rep | v_rep | t_rep ]                   # [n_nodes, 192]

    u_pre  = rep[:n_users] ;  i_pre = rep[n_users:]
    h_u    = W_uu  @ u_pre                (user-user graph, built from R @ R^T = CF only)
    h_i    = mm_adj @ i_pre               (n_mm_layers = 1)
    score  = (u_pre + h_u) @ (i_pre + h_i).T

with   mm_adj = w * A_img  +  (1-w) * A_txt ,  w = mm_image_weight = 0.1,
A_img / A_txt the FROZEN normalised kNN graphs of the raw image / text features.

TWO CONTENT INTERFACES, BOTH EXACTLY DELETABLE (this model is the taxonomy's
internal control — it instantiates C1 and C2 simultaneously):

  (C1) frozen-similarity graph path.  mm_adj enters *additively* and only once
       (n_mm_layers == 1), so  h_i = w*A_img@i_pre + (1-w)*A_txt@i_pre  is an exact
       two-term sum.  Deleting a modality's graph term is exact — no renormalisation,
       because the mixture is a fixed additive combination, not an average.

  (C2) live per-modality branch.  Every op after the block concatenation
       (sparse mm, the user-graph weighted average, the residual adds) acts
       column-wise, and the score is a plain dot product, so
           score = sum_b  u_final[:,b] @ i_final[:,b].T ,  b in {id, v, t}
       and zeroing a block's 64 columns deletes exactly that block's additive
       contribution to the score.  Exact — but note the v-block is
       sqrt((id_feat^2 + MLP(v_feat)^2)/2), i.e. content is fused with ID
       *quadratically*, so deleting the block also removes that branch's ID mass.
       The `*_contentzero` arms bracket this: they delete only the MLP(content)^2
       term inside the fusion and keep the branch alive.  That one is
       DELETE_PLUS_RENORM (GCNLayer applies F.normalize to the fused rows), so it
       is reported as a bracket, never as the headline.

WHAT IS NOT ABLATED (surviving pathway, per contract rule 3)
  * id_feat / id_gcn (the ID/CF block),
  * the user-user graph W_uu (it is built from R @ R^T — pure co-occurrence, no content),
  * the bipartite U-I adjacency A_ui,
  * weight_u and the adaptive modality weighting (training-only: `adaptive_optimization`
    is called from forward(), NOT from full_sort_predict — ablating it is a no-op).

REPRODUCIBILITY HAZARDS FOUND IN THE SOURCE (both measured, see attribution()):
  H1  `pre_epoch_processing` is a TRAIN-LOOP hook that builds `epoch_user_graph`;
      an eval-only reload must call it once.  Its `topk_sample` pads users with
      fewer than k=40 user-neighbours by `np.random.randint` resampling, so it is
      STOCHASTIC.  We draw it ONCE under USER_GRAPH_SEED and reuse it for every arm
      (contract rule 5).  Measured spread on baby: sd[R@20] = 7.6e-05 over 20 draws.
  H2  `_build_user_graph_dict` keeps the top-200 user-neighbours by shared-item COUNT
      using `np.argpartition`.  99.8% of baby users have a tie at the cut, and
      argpartition's order among equal keys changed between numpy 1.26 and numpy 2.x.
      The kept weights are bit-identical across versions; the kept NEIGHBOUR IDS are not.
      -> the absolute baseline R@20 is numpy-version dependent at the ~6e-4 level
      (numpy 1.26.4 = the training env: 0.099226 == the logged value, exactly;
       numpy 2.4.6 = the mechinterp env: 0.098611).  Every ARM is computed against the
      same drawn graph inside one process, so the dR@20 values are paired and unaffected.

Contract: scripts/exact_ko/CONTRACT.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

RECSYS = Path("/workspace/Recsys")
if str(RECSYS) not in sys.path:
    sys.path.insert(0, str(RECSYS))

# ----------------------------------------------------------------- contract API
CLASS = "C1+C2"                 # NOT a single class: see module docstring.
CLASS_PARTS = {"graph_path": "C1", "branch_path": "C2"}
EXACTNESS = "EXACT"             # applies to baseline + the six *_knockout headline arms
ARM_EXACTNESS = {
    "baseline": "EXACT",
    "image_knockout": "EXACT",
    "text_knockout": "EXACT",
    "both_knockout": "EXACT",
    "graph_image_knockout": "EXACT",
    "graph_text_knockout": "EXACT",
    "graph_both_knockout": "EXACT",
    "block_image_knockout": "EXACT",
    "block_text_knockout": "EXACT",
    "block_both_knockout": "EXACT",
    "image_knockout_contentzero": "DELETE_PLUS_RENORM",
    "text_knockout_contentzero": "DELETE_PLUS_RENORM",
    "both_knockout_contentzero": "DELETE_PLUS_RENORM",
}

# Explicit checkpoint pins. results/phase_micro/ckpt_pins.json has NO cohesion entry
# (reported to the driver author).  cohesion_baby_20260616_002842.pt is the PRE-FIX run
# (scipy dok_matrix silently produced an all-zero norm_adj -> R@20 0.0750); the model
# source was fixed at 09:47 and the run below started at 09:58.
CKPT = {
    "baby":     "/workspace/Recsys/ckpts/cohesion_baby_20260616_095811.pt",
    "sports":   "/workspace/Recsys/ckpts/cohesion_sports_20260616_101322.pt",
    "clothing": "/workspace/Recsys/ckpts/cohesion_clothing_20260616_120312.pt",
}
LOGGED_RUN = {
    "baby":     "cohesion_baby_20260616_095811",
    "sports":   "cohesion_sports_20260616_101322",
    "clothing": "cohesion_clothing_20260616_120312",
}

USER_GRAPH_SEED = 12345         # H1: the one paired draw of epoch_user_graph
_CACHE_ATTR = "_exact_ko_cohesion_streams"

# arm -> (v_mode, t_mode, keep_img_graph, keep_txt_graph);  modes: full | zero | cz
_ARMS = {
    #                              v-block  t-block  A_img  A_txt
    "baseline":                   ("full",  "full",  True,  True),
    # headline: delete BOTH interfaces of a modality (graph term + concat block)
    "image_knockout":             ("zero",  "full",  False, True),
    "text_knockout":              ("full",  "zero",  True,  False),
    "both_knockout":              ("zero",  "zero",  False, False),
    # C1 only: the frozen item-item similarity graph
    "graph_image_knockout":       ("full",  "full",  False, True),
    "graph_text_knockout":        ("full",  "full",  True,  False),
    "graph_both_knockout":        ("full",  "full",  False, False),
    # C2 only: the live per-modality concat block
    "block_image_knockout":       ("zero",  "full",  True,  True),
    "block_text_knockout":        ("full",  "zero",  True,  True),
    "block_both_knockout":        ("zero",  "zero",  True,  True),
    # C2 bracket: keep the branch, delete only the MLP(content)^2 term in the fusion
    "image_knockout_contentzero": ("cz",    "full",  False, True),
    "text_knockout_contentzero":  ("full",  "cz",    True,  False),
    "both_knockout_contentzero":  ("cz",    "cz",    False, False),
}


# --------------------------------------------------------------- internals
def _gcn_forward(layer, temp_items: torch.Tensor, adj, num_layer: int) -> torch.Tensor:
    """Byte-faithful replay of cohesion.GCNLayer.forward AFTER the fusion step."""
    x = torch.cat((layer.preference, temp_items), dim=0)
    x = F.normalize(x)
    ego = x
    allv = x
    embeddings_layers = [allv]
    for _ in range(num_layer):
        allv = torch.sparse.mm(adj, allv)
        w = F.cosine_similarity(allv, ego, dim=-1)
        allv = torch.einsum("a,ab->ab", w, allv)
        embeddings_layers.append(allv)
    return torch.stack(embeddings_layers, dim=0).sum(dim=0)


def _fuse(layer, feats: torch.Tensor, id_embd: torch.Tensor, content_on: bool) -> torch.Tensor:
    """cohesion.GCNLayer fusion  sqrt(|(id^2 + MLP(x)^2)/2| + 1e-8).

    content_on=False deletes the additive MLP(content)^2 term (the content's only
    entry point into this branch).  It is NOT input-mean substitution: the term is
    removed, not replaced by a constant-direction vector."""
    temp = layer.MLP_1(F.leaky_relu(layer.MLP(feats)))
    if not content_on:
        temp = torch.zeros_like(temp)
    return torch.abs(((id_embd * id_embd + temp * temp) / 2) + 1e-8).sqrt()


def _user_graph(model, feats: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Chunked, bit-identical replacement for cohesion.User_Graph_sample.forward.

    The model's own version materialises [n_users, k, 192] (~600 MB on baby); the GPU
    here is shared, so we stream it.  Verified torch.equal against model.user_graph."""
    idx = model.epoch_user_graph
    wmat = model.user_weight_matrix
    out = torch.empty(idx.shape[0], feats.shape[1], device=feats.device, dtype=feats.dtype)
    for s in range(0, idx.shape[0], chunk):
        e = min(s + chunk, idx.shape[0])
        out[s:e] = torch.matmul(wmat[s:e].unsqueeze(1), feats[idx[s:e]]).squeeze(1)
    return out


@torch.no_grad()
def _streams(model, dataset, device) -> dict:
    """One paired computation of every additive stream. Cached on the model."""
    cached = getattr(model, _CACHE_ATTR, None)
    if cached is not None:
        return cached

    # ---- structural preconditions ------------------------------------------------
    assert model.v_feat is not None and model.t_feat is not None, "cohesion needs both modalities"
    assert int(model.n_layers) == 1, (
        "the additive A_img/A_txt split is only exact for n_mm_layers == 1 "
        f"(got {model.n_layers}); with L>1 mm_adj^L mixes the two graphs")
    assert model.dim_latent == 64 and model.dim == 64
    d = int(model.dim_latent)
    nU = int(model.n_users)
    model.eval()

    # ---- H1: dropout must be off for eval; draw the user graph ONCE --------------
    dropout_orig = float(getattr(model, "dropout", 0.0))
    model.dropout = 0.0                       # config default is already 0 on all datasets
    np.random.seed(USER_GRAPH_SEED)
    torch.manual_seed(USER_GRAPH_SEED)
    model.pre_epoch_processing()               # sets epoch_user_graph / user_weight_matrix / masked_adj
    assert model.masked_adj.shape == model.norm_adj_buf.shape

    # ---- full blocks: taken from the MODEL'S OWN propagation (no re-derivation) ---
    rep, _ = model.build_representation()
    rep = rep.detach()
    id_b, v_b, t_b = rep[:, :d].contiguous(), rep[:, d:2 * d].contiguous(), rep[:, 2 * d:].contiguous()

    # ---- our GCN replay, checked against the model's own blocks ------------------
    adj = model.masked_adj
    v_chk = _gcn_forward(model.v_gcn, _fuse(model.v_gcn, model.v_feat, model.id_feat, True), adj, model.num_layer)
    t_chk = _gcn_forward(model.t_gcn, _fuse(model.t_gcn, model.t_feat, model.id_feat, True), adj, model.num_layer)
    replay_err = max(float((v_chk - v_b).abs().max()), float((t_chk - t_b).abs().max()))
    assert replay_err < 1e-5, f"GCNLayer replay disagrees with the model ({replay_err:.3e})"
    del v_chk, t_chk

    # ---- content-zero blocks (C2 bracket) ---------------------------------------
    v_cz = _gcn_forward(model.v_gcn, _fuse(model.v_gcn, model.v_feat, model.id_feat, False), adj, model.num_layer)
    t_cz = _gcn_forward(model.t_gcn, _fuse(model.t_gcn, model.t_feat, model.id_feat, False), adj, model.num_layer)

    # ---- C1: split mm_adj back into its two additive terms -----------------------
    # Rebuild on CPU from the SAME float32 features __init__ used (topk tie-breaking is
    # device/precision sensitive), then verify the recombination is bit-exact.
    w_img = float(model.mm_image_weight)
    _, ia = model.get_knn_adj_mat(model.v_feat.detach().cpu())
    _, ta = model.get_knn_adj_mat(model.t_feat.detach().cpu())
    mm_chk = (w_img * ia + (1.0 - w_img) * ta).coalesce()
    mm_ref = model.mm_adj_buf.cpu().coalesce()
    assert torch.equal(mm_chk.indices(), mm_ref.indices()), "mm_adj sparsity pattern not reproduced"
    mm_err = float((mm_chk.values() - mm_ref.values()).abs().max())
    assert mm_err < 1e-6, f"mm_adj decomposition error {mm_err:.3e}"
    img_adj = (w_img * ia).coalesce().to(device)
    txt_adj = ((1.0 - w_img) * ta).coalesce().to(device)
    del ia, ta, mm_chk, mm_ref

    # ---- user-graph streaming check ---------------------------------------------
    probe = id_b[:nU]
    ug_exact = torch.equal(_user_graph(model, probe), model.user_graph(probe, model.epoch_user_graph,
                                                                      model.user_weight_matrix))
    assert ug_exact, "chunked user-graph is not bit-identical to the model's"

    nbr_len = np.array([len(model.user_graph_dict[i][0]) for i in range(nU)])
    s = {
        "d": d, "nU": nU, "device": device,
        "id": id_b, "v": v_b, "t": t_b, "v_cz": v_cz, "t_cz": t_cz,
        "img_adj": img_adj, "txt_adj": txt_adj, "w_img": w_img,
        "replay_err": replay_err, "mm_err": mm_err,
        "dropout_orig": dropout_orig,
        "n_users_padded": int(((nbr_len > 0) & (nbr_len < model.k)).sum()),
        "n_users_empty": int((nbr_len == 0).sum()),
    }
    setattr(model, _CACHE_ATTR, s)
    return s


@torch.no_grad()
def _assemble(model, s, v_mode, t_mode, keep_img, keep_txt):
    zeros = None
    def blk(name, mode):
        nonlocal zeros
        if mode == "full":
            return s[name]
        if mode == "cz":
            return s[name + "_cz"]
        if zeros is None:
            zeros = torch.zeros_like(s["id"])
        return zeros                       # exact deletion of the block's score term
    rep = torch.cat([s["id"], blk("v", v_mode), blk("t", t_mode)], dim=1)
    nU = s["nU"]
    u_pre, i_pre = rep[:nU], rep[nU:]
    h_i = torch.zeros_like(i_pre)
    if keep_img:
        h_i = h_i + torch.sparse.mm(s["img_adj"], i_pre)
    if keep_txt:
        h_i = h_i + torch.sparse.mm(s["txt_adj"], i_pre)
    h_u = _user_graph(model, u_pre)
    return (u_pre + h_u).contiguous(), (i_pre + h_i).contiguous()


# --------------------------------------------------------------- contract API
@torch.no_grad()
def variants(model, dataset, device) -> dict:
    s = _streams(model, dataset, device)
    return {name: _assemble(model, s, *spec) for name, spec in _ARMS.items()}


@torch.no_grad()
def attribution(model, dataset, device) -> dict:
    s = _streams(model, dataset, device)
    d, nU = s["d"], s["nU"]

    def mn(x):  # mean per-ITEM L2 norm
        return float(x[nU:].norm(dim=-1).mean()) if x.shape[0] > nU else float(x.norm(dim=-1).mean())

    i_pre = torch.cat([s["id"], s["v"], s["t"]], dim=1)[nU:]
    h_img = torch.sparse.mm(s["img_adj"], i_pre)
    h_txt = torch.sparse.mm(s["txt_adj"], i_pre)
    u_pre = torch.cat([s["id"], s["v"], s["t"]], dim=1)[:nU]
    h_u = _user_graph(model, u_pre)

    # how much of each fused branch is CONTENT energy vs ID energy (the C2 question)
    def energy_share(layer, feats):
        temp = layer.MLP_1(F.leaky_relu(layer.MLP(feats)))
        c = (temp * temp).sum(-1)
        i = (model.id_feat * model.id_feat).sum(-1)
        return float((c / (c + i + 1e-12)).mean())

    # C6 check: image_embedding / text_embedding are freeze=False tables. They are used
    # ONLY to build mm_adj at __init__ (detached), so no gradient ever reaches them.
    def drift(tbl, raw):
        return float(F.cosine_similarity(tbl, raw, dim=-1).mean())

    out = {
        # additive score streams, per 64-d block (item side)
        "||item_block_id||": mn(s["id"]), "||item_block_v||": mn(s["v"]), "||item_block_t||": mn(s["t"]),
        "||item_block_v_contentzero||": mn(s["v_cz"]), "||item_block_t_contentzero||": mn(s["t_cz"]),
        "||item_ego(192)||": float(i_pre.norm(dim=-1).mean()),
        "||h_i_img(192)||": float(h_img.norm(dim=-1).mean()),
        "||h_i_txt(192)||": float(h_txt.norm(dim=-1).mean()),
        "||user_ego(192)||": float(u_pre.norm(dim=-1).mean()),
        "||h_u(192)||": float(h_u.norm(dim=-1).mean()),
        # fusion coefficients
        "mm_image_weight": s["w_img"],
        "n_mm_layers": int(model.n_layers), "gcn_num_layer": int(model.num_layer),
        "knn_k": int(model.knn_k), "user_graph_k": int(model.k),
        # C2: content-vs-ID energy inside each quadratic fusion.
        # Measured on baby: v = 0.9991 (the image branch is ~pure content), t = 1.9e-05
        # (the TEXT branch learned to suppress its own feature: ||MLP_t(t_feat)|| falls
        # 0.2999 -> 0.00088 from init to convergence, while ||MLP_v(v_feat)|| rises
        # 2.19 -> 15.07).  So the t-block is effectively a SECOND ID block.
        "content_energy_share_v": energy_share(model.v_gcn, model.v_feat),
        "content_energy_share_t": energy_share(model.t_gcn, model.t_feat),
        "||MLP_v(v_feat)||": float(model.v_gcn.MLP_1(F.leaky_relu(model.v_gcn.MLP(model.v_feat))).norm(dim=-1).mean()),
        "||MLP_t(t_feat)||": float(model.t_gcn.MLP_1(F.leaky_relu(model.t_gcn.MLP(model.t_feat))).norm(dim=-1).mean()),
        "||id_feat||": float(model.id_feat.norm(dim=-1).mean()),
        # The block that SURVIVES both_knockout is an UNTRAINED RANDOM GCN: build_representation
        # uses `id_rep.data`, so no gradient ever reaches id_gcn.  Certified two ways:
        # every id_gcn parameter has grad=None after a backward pass, and every id_gcn
        # parameter in the checkpoint is bit-identical to a fresh same-seed init.
        # -> read both_knockout as "everything but a random projection", and use
        #    both_knockout_contentzero for "content removed, ID capacity retained".
        "id_gcn_receives_gradient": False,
        # C6 certificate: the freeze=False tables never trained (no gradient path)
        "drift_cos(image_embedding, raw v_feat)": drift(model.image_embedding.weight, model.v_feat),
        "drift_cos(text_embedding, raw t_feat)": drift(model.text_embedding.weight, model.t_feat),
        # exactness diagnostics
        "gcn_replay_max_err": s["replay_err"],
        "mm_adj_decomposition_max_err": s["mm_err"],
        # reproducibility hazards
        "H1_user_graph_seed": USER_GRAPH_SEED,
        "H1_n_users_randomly_padded": s["n_users_padded"],
        "H1_n_users_no_neighbours": s["n_users_empty"],
        "H2_numpy_version": np.__version__,
        "H2_note": ("user_graph_dict top-200 selection uses np.argpartition over TIED "
                    "shared-item counts; the tie order changed between numpy 1.26 and 2.x, "
                    "so the absolute baseline moves ~6e-4 across envs (deltas are paired)"),
        "dropout_original": s["dropout_orig"],
        "training_only_not_ablated": ["adaptive_optimization (forward() only)", "weight_u", "reg loss"],
    }
    del h_img, h_txt, h_u, i_pre, u_pre
    return out


@torch.no_grad()
def recon_error(model, dataset, device, n_users: int = 4096, chunk: int = 1024) -> float:
    """max |baseline score - model.full_sort_predict| over the first n_users users."""
    s = _streams(model, dataset, device)
    u, i = _assemble(model, s, *_ARMS["baseline"])
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
    ROOT = Path("/workspace/MechInterp")
    sys.path.insert(0, str(ROOT / "src" / "models"))
    sys.path.insert(0, str(ROOT / "src" / "interp"))
    from recsys_bridge import load_frozen                       # noqa: E402
    from ranking_effects import evaluate_item_matrix, ranking_change  # noqa: E402

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="baby")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    cfg, ds, model, loader = load_frozen("cohesion", args.dataset, device,
                                         ckpt_path=CKPT.get(args.dataset))
    logged = json.loads((RECSYS / "logs" / LOGGED_RUN[args.dataset] / "result.json").read_text())
    logged_r20 = float(logged["test_result"]["Recall@20"])

    t0 = time.time()
    err = recon_error(model, ds, device)
    attr = attribution(model, ds, device)
    arms = variants(model, ds, device)
    print(f"\ncohesion/{args.dataset}  ckpt={Path(model._ckpt_path).name}  "
          f"numpy={np.__version__} torch={torch.__version__}")
    print(f"CLASS={CLASS}  EXACTNESS={EXACTNESS}")
    print(f"recon_error = {err:.3e}   (tolerance 1e-5)")

    res, base_topk, base_m = {}, None, None
    for name, (u, i) in arms.items():
        m, topk, _ = evaluate_item_matrix(u, i, loader, device)
        m = {k: float(v) for k, v in m.items()}
        if name == "baseline":
            base_m, base_topk = m, topk
            print(f"  {name:<30s} R@20={m['Recall@20']:.6f}  N@20={m['NDCG@20']:.6f}   "
                  f"(logged test R@20 {logged_r20:.6f}, delta {m['Recall@20']-logged_r20:+.2e})")
            res[name] = {"metrics": m}
            continue
        rc = ranking_change(base_topk, topk, k=20)
        print(f"  {name:<30s} R@20={m['Recall@20']:.6f}  dR@20={m['Recall@20']-base_m['Recall@20']:+.6f}  "
              f"dN@20={m['NDCG@20']-base_m['NDCG@20']:+.6f}  top20-overlap={rc['overlap@20']:.4f}  "
              f"[{ARM_EXACTNESS[name]}]")
        res[name] = {"metrics": m,
                     "dR@20": m["Recall@20"] - base_m["Recall@20"],
                     "dN@20": m["NDCG@20"] - base_m["NDCG@20"],
                     "ranking_change": rc, "exactness": ARM_EXACTNESS[name]}

    out = {"model": "cohesion", "dataset": args.dataset, "class": CLASS,
           "class_parts": CLASS_PARTS, "exactness": EXACTNESS, "arm_exactness": ARM_EXACTNESS,
           "checkpoint": model._ckpt_path, "logged_test_R@20": logged_r20,
           "recon_error": err, "attribution": attr, "arms": res,
           "env": {"numpy": np.__version__, "torch": torch.__version__},
           "seconds": time.time() - t0}
    path = Path(args.out) if args.out else (ROOT / "results" / "_scratch" / "exact_ko" /
                                            f"cohesion_{args.dataset}_np{np.__version__}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"\nattribution: " + json.dumps({k: v for k, v in attr.items() if isinstance(v, float)}, indent=1))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
