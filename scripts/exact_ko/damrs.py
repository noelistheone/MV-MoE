"""Exact structural knockout for DA-MRS (KDD 2024).

Source of truth: /workspace/Recsys/src/models/damrs.py (READ-ONLY).

WHY DA-MRS IS EXACT
-------------------
`full_sort_predict` is, verbatim:

    user_embeddings, item_embeddings, h_t, h_v, h_s = self.forward()
    user_e       = user_embeddings[user, :]
    i_embedding  = (h_v + h_t + h_s) / 3.0
    all_item_e   = item_embeddings + i_embedding
    score        = user_e @ all_item_e.T

and `forward()` returns

    user_embeddings, item_embeddings = LightGCN mean-of-layers over norm_adj   (pure ID/CF)
    h_t = text_adj    ** n_mm_layers  @  item_id_embedding.weight
    h_v = image_adj   ** n_mm_layers  @  item_id_embedding.weight
    h_s = session_adj ** n_mm_layers  @  item_id_embedding.weight

Each modality contributes ONE additive term with a HARD-CODED fusion coefficient of
1/3 (no learned gate, no normaliser shared across streams, no renormalisation).
Deleting a term is therefore the model's own counterfactual, exactly:

    EXACTNESS = "EXACT"     (measured recon_error on baby = 0.0, BITWISE)

THE THIRD GRAPH IS NOT CONTENT
------------------------------
`session_adj` is built from item-item co-occurrence C = R^T R over the TRAIN
interaction matrix (`_build_item_graph_dict`, top-k=2, min co-occurrence 2) plus a
self-loop on every item.  It is a *behavioural* graph: no image or text feature
touches it.  So `both_knockout` (drop h_v and h_t) is NOT an ID-only floor -- it
still carries a second-order CF signal.  We therefore emit six arms, including a
true ID-only floor:

    baseline          cf + (h_v + h_t + h_s)/3
    image_knockout    cf + (      h_t + h_s)/3
    text_knockout     cf + (h_v +       h_s)/3
    both_knockout     cf + (            h_s)/3     <- content off, behaviour graph ON
    session_knockout  cf + (h_v + h_t      )/3     <- extra: price of the behavioural graph
    id_only_floor     cf                           <- true floor, all three graphs off

WHY THE PUBLISHED SCREEN IS BROKEN HERE (verified, see `screen_artifact_certificate`)
------------------------------------------------------------------------------------
`get_knn_adj_mat` couples the two content graphs through a shared mask and a shared
neighbour count:

    mask_v = v_sim < v_sim.mean();  mask_t = t_sim < t_sim.mean()
    t_sim[mask_v] = 0; v_sim[mask_t] = 0; t_sim[mask_t] = 0; v_sim[mask_v] = 0
    item_num = len(nonzero(t_sim[i]))          # <-- BOTH graphs' out-degree comes from t_sim
    topk(v_sim[i], item_num); topk(t_sim[i], item_num)

Replacing the visual feature by its per-dimension mean makes every row of
`image_embedding` identical, so after L2 normalisation every cosine in `v_sim` is the
same number c.  `v_sim.mean()` is a float32 reduction over n_items^2 entries and
lands strictly above c, so `mask_v` is ALL TRUE -> `t_sim[mask_v] = 0` zeroes the
TEXT similarity matrix as well -> `item_num == 0` for every item -> BOTH `image_adj`
and `text_adj` come out with zero edges -> h_v == h_t == 0.  That is bit-identical to
the screen's own `both_knockout`, which is exactly what
results/phasex_crossarch/crossarch_knockout.json shows for damrs/baby
(image_knockout == both_knockout == +0.0021247574744617664); this module's
`both_knockout` arm reproduces that number to absdiff EXACTLY 0.0.

The direction of the rounding accident decides which side fires: on baby, v_sim
under v_mean has ONE unique value 0.9999998212 whose float32 grand mean rounds UP
to 0.9999999404 -> mask_v ALL TRUE (49,702,500/49,702,500) -> both graphs empty;
t_sim under t_mean has one unique value whose grand mean does NOT round up ->
mask_t has ZERO true entries -> both graphs keep their full 70,500 edges but the
text graph degenerates to "topk of a constant row" = the lowest item ids. So the
screen's text arm on baby is a garbage-graph condition, not a deletion.

On SPORTS the accident fires from the other side, and the certificate reproduces
that too: under t_mean, mask_t is ALL TRUE (336,979,449 / 336,979,449 = 18357^2)
-> both graphs 0 edges; under v_mean, mask_v is NOT all true and both graphs keep
their 183,570 edges. That is exactly the input-mean screen's sports row, where
text_knockout == both_knockout bit-identically and image_knockout does not.

So the published damrs/baby "image knockout" number is not an image measurement at
all: it is a joint image+text deletion, and this module's `both_knockout` arm
reproduces it.

DEAD PARAMETERS
---------------
`image_trs` / `text_trs` are constructed in `__init__` and never referenced by
`forward`, `calculate_loss` or `full_sort_predict`.  They are in the checkpoint but
receive no gradient and are off every path.  `attribution` certifies this by
randomising them and checking the scores are bitwise identical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path("/workspace/MechInterp")
RECSYS = Path("/workspace/Recsys")
for _p in (str(RECSYS), str(ROOT / "src" / "models"), str(ROOT / "src" / "interp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------- contract fields
CLASS = "C1"
CLASS_DESC = ("C1 = content is consumed ONLY at construction time, to build a frozen "
              "item-item graph that propagates the ID embedding table; no content "
              "vector is ever an argument of the scoring function. Same interface as "
              "FREEDOM. (DA-MRS adds a third, behavioural co-occurrence graph on the "
              "same interface -- see session_* arms.)")
EXACTNESS = "EXACT"

# The published cross-architecture screen resolved checkpoints with latest_ckpt(),
# i.e. max timestamp. For damrs/baby that is 20260616_100604 (logged test R@20
# 0.08424910308685056), and the screen's baseline R@20 in
# results/phasex_crossarch/crossarch_knockout.json is 0.08424910308685057.
# We PIN it so this module measures the same model the paper reported.
# NOTE FOR THE DRIVER: results/phase_micro/ckpt_pins.json has no damrs entry; these
# pins should be added there (this module does not write to that file).
CKPT_PIN = {
    "baby":     str(RECSYS / "ckpts" / "damrs_baby_20260616_100604.pt"),
    "sports":   str(RECSYS / "ckpts" / "damrs_sports_20260616_105352.pt"),
    "clothing": str(RECSYS / "ckpts" / "damrs_clothing_20260616_111702.pt"),
    "microlens": str(RECSYS / "ckpts" / "damrs_microlens_20260621_051633.pt"),
}
LOGGED_RUN = {
    "baby":     str(RECSYS / "logs" / "damrs_baby_20260616_100604" / "result.json"),
    "sports":   str(RECSYS / "logs" / "damrs_sports_20260616_105352" / "result.json"),
    "clothing": str(RECSYS / "logs" / "damrs_clothing_20260616_111702" / "result.json"),
    "microlens": str(RECSYS / "logs" / "damrs_microlens_20260621_051633" / "result.json"),
}

FUSION_COEFF = 1.0 / 3.0   # hard-coded in DAMRS.full_sort_predict: (h_v + h_t + h_s)/3.0


# ---------------------------------------------------------------- decomposition
@torch.no_grad()
def _streams(model) -> dict:
    """Re-derive DAMRS.forward() term by term, exposing every additive stream.

    Uses the model's OWN frozen buffers (image_adj / text_adj / session_adj), so no
    CPU/GPU graph-rebuild reproducibility question arises (they were built on CPU in
    __init__ and moved by .to(device); we never rebuild them).
    """
    E = model.item_id_embedding.weight.detach()

    # --- LightGCN backbone, verbatim from forward() ---------------------------
    ego = torch.cat((model.user_embedding.weight, model.item_id_embedding.weight), dim=0).detach()
    all_emb = [ego]
    for _ in range(model.n_ui_layers):
        ego = torch.sparse.mm(model.norm_adj, ego)
        all_emb.append(ego)
    out = torch.stack(all_emb, dim=1).mean(dim=1, keepdim=False)
    u_g, i_cf = torch.split(out, [model.n_users, model.n_items], dim=0)

    # --- three graph streams, verbatim (n_layers == n_mm_layers) --------------
    def prop(adj):
        h = E.clone()
        for _ in range(model.n_layers):
            h = torch.sparse.mm(adj, h)
        return h

    h_t = prop(model.text_adj)
    h_v = prop(model.image_adj)
    h_s = prop(model.session_adj)

    fused = i_cf + (h_v + h_t + h_s) * FUSION_COEFF
    return {"u_all": u_g, "cf": i_cf, "h_img": h_v, "h_txt": h_t, "h_ses": h_s,
            "fused": fused}


# ---------------------------------------------------------------- contract API
@torch.no_grad()
def variants(model, dataset=None, device=None) -> dict:
    s = _streams(model)
    u, cf, w = s["u_all"], s["cf"], FUSION_COEFF
    hv, ht, hs = s["h_img"], s["h_txt"], s["h_ses"]
    return {
        "baseline":         (u, cf + (hv + ht + hs) * w),
        "image_knockout":   (u, cf + (     ht + hs) * w),   # delete h_v term
        "text_knockout":    (u, cf + (hv +      hs) * w),   # delete h_t term
        "both_knockout":    (u, cf + (           hs) * w),  # content off, session graph ON
        "session_knockout": (u, cf + (hv + ht     ) * w),   # extra: behavioural graph off
        "id_only_floor":    (u, cf),                        # extra: true ID-only floor
    }


@torch.no_grad()
def recon_error(model, dataset=None, device=None, chunk: int = 2048) -> float:
    """max |baseline_scores - DAMRS.full_sort_predict| over ALL users."""
    s = _streams(model)
    u_all, M = s["u_all"], s["fused"]
    dev = M.device
    err = 0.0
    for start in range(0, model.n_users, chunk):
        idx = torch.arange(start, min(start + chunk, model.n_users), device=dev)
        ref = model.full_sort_predict({"user": idx})
        mine = u_all[idx] @ M.t()
        err = max(err, float((ref - mine).abs().max()))
        del ref, mine
    return err


@torch.no_grad()
def attribution(model, dataset=None, device=None, heavy: bool = True) -> dict:
    s = _streams(model)
    w = FUSION_COEFF

    def mn(x):
        return float(x.norm(dim=-1).mean())

    out = {
        "fusion_coeff": w,
        "fusion_coeff_source": "hard-coded 1/3 in DAMRS.full_sort_predict; not learned",
        "n_mm_layers": int(model.n_layers),
        "n_ui_layers": int(model.n_ui_layers),
        "knn_k": int(model.knn_k),
        "||cf||": mn(s["cf"]),
        "||h_img/3||": mn(s["h_img"] * w),
        "||h_txt/3||": mn(s["h_txt"] * w),
        "||h_ses/3||": mn(s["h_ses"] * w),
        "||fused||": mn(s["fused"]),
        "graph_nnz": {"image_adj": int(model.image_adj._nnz()),
                      "text_adj": int(model.text_adj._nnz()),
                      "session_adj": int(model.session_adj._nnz()),
                      "session_adj_self_loops": int(model.n_items)},
        "session_graph_is_behavioural": True,
        "session_graph_source": "C = R^T R over TRAIN interactions, top-k=2, min co-occ 2, "
                                "+ self loop; NO image/text feature touches it",
    }
    # content tables are freeze=True -> raw features are still the on-path content
    for name, feat in (("image", "v_feat"), ("text", "t_feat")):
        emb = getattr(model, f"{name}_embedding").weight
        raw = getattr(model, feat)
        out[f"{name}_table_frozen"] = bool(not emb.requires_grad)
        out[f"{name}_table_equals_raw_feature"] = bool(torch.equal(emb, raw.to(emb.device)))

    out["dead_parameters"] = _dead_param_certificate(model)
    if heavy:
        out["screen_artifact_certificate"] = screen_artifact_certificate(model)
    return out


# ---------------------------------------------------------------- certificates
@torch.no_grad()
def _dead_param_certificate(model) -> dict:
    """image_trs / text_trs exist in the checkpoint but are referenced nowhere on any
    path. Prove it: randomise them, check full_sort_predict is BITWISE identical."""
    users = torch.arange(0, min(512, model.n_users), device=model.item_id_embedding.weight.device)
    before = model.full_sort_predict({"user": users}).clone()
    saved = {}
    for nm in ("image_trs", "text_trs"):
        if hasattr(model, nm):
            lin = getattr(model, nm)
            saved[nm] = (lin.weight.detach().clone(), lin.bias.detach().clone())
            lin.weight.copy_(torch.randn_like(lin.weight) * 10.0)
            lin.bias.copy_(torch.randn_like(lin.bias) * 10.0)
    after = model.full_sort_predict({"user": users})
    identical = bool(torch.equal(before, after))
    for nm, (wt, bs) in saved.items():
        lin = getattr(model, nm)
        lin.weight.copy_(wt)
        lin.bias.copy_(bs)
    restored = bool(torch.equal(before, model.full_sort_predict({"user": users})))
    return {"randomised": sorted(saved), "scores_bitwise_identical": identical,
            "restored_ok": restored,
            "note": "image_trs/text_trs are constructed in __init__ and never used"}


@torch.no_grad()
def screen_artifact_certificate(model) -> dict:
    """Reproduce, on the model's OWN get_knn_adj_mat, what the published input-mean
    screen did to the two content graphs. Runs on CPU (as __init__ did).

    For each of the screen's four conditions we report the two masks, the number of
    items whose neighbour count `item_num` collapses to 0, and the edge count of the
    graphs the model would actually use. An empty graph means h == 0, i.e. deletion
    of that content term -- so a condition in which BOTH graphs come out empty is a
    joint image+text deletion, not an ablation of one modality.
    """
    n_items = int(model.n_items)
    if n_items > 30000:
        return {"skipped": f"n_items={n_items} too large for the O(n^2) similarity matrix"}
    v = model.v_feat.detach().cpu().float()
    t = model.t_feat.detach().cpu().float()
    v_mean = v.mean(0, keepdim=True).expand_as(v).contiguous()
    t_mean = t.mean(0, keepdim=True).expand_as(t).contiguous()

    def cond(vv, tt):
        # verbatim head of DAMRS.get_knn_adj_mat
        vn = vv.div(torch.norm(vv, p=2, dim=-1, keepdim=True))
        v_sim = vn @ vn.t()
        tn = tt.div(torch.norm(tt, p=2, dim=-1, keepdim=True))
        t_sim = tn @ tn.t()
        mask_v = v_sim < v_sim.mean()
        mask_t = t_sim < t_sim.mean()
        # NOTE: integer counts, never a float32 .mean() over n^2 booleans -- that
        # reduction is itself rounded (it reports 0.99999994 for an ALL-TRUE mask).
        n = int(v_sim.numel())
        d = {"n_entries": n,
             "n_unique_v_sim": int(torch.unique(v_sim).numel()),
             "n_unique_t_sim": int(torch.unique(t_sim).numel()),
             "mask_v_n_true": int(mask_v.sum()), "mask_v_all_true": bool(int(mask_v.sum()) == n),
             "mask_t_n_true": int(mask_t.sum()), "mask_t_all_true": bool(int(mask_t.sum()) == n)}
        t_sim[mask_v] = 0
        v_sim[mask_t] = 0
        t_sim[mask_t] = 0
        v_sim[mask_v] = 0
        rows = (t_sim != 0).sum(1)          # item_num, the out-degree of BOTH graphs
        d["items_with_item_num_zero"] = int((rows == 0).sum())
        d["n_items"] = int(rows.numel())
        del v_sim, t_sim, mask_v, mask_t, vn, tn, rows
        va, ta = model.get_knn_adj_mat(vv, tt)
        d["image_adj_nnz"] = int(va._nnz())
        d["text_adj_nnz"] = int(ta._nnz())
        d["both_content_graphs_empty"] = bool(d["image_adj_nnz"] == 0 and d["text_adj_nnz"] == 0)
        del va, ta
        return d

    cert = {"conditions": {
        "baseline(v,t)": cond(v, t),
        "screen_image_KO(v_mean,t)": cond(v_mean, t),
        "screen_text_KO(v,t_mean)": cond(v, t_mean),
        "screen_both_KO(v_mean,t_mean)": cond(v_mean, t_mean),
    }, "live_graph_nnz": {"image_adj": int(model.image_adj._nnz()),
                          "text_adj": int(model.text_adj._nnz())}}
    c = cert["conditions"]
    cert["screen_image_KO_is_really_both_KO"] = bool(
        c["screen_image_KO(v_mean,t)"]["both_content_graphs_empty"]
        and c["screen_both_KO(v_mean,t_mean)"]["both_content_graphs_empty"])
    cert["screen_text_KO_is_really_both_KO"] = bool(
        c["screen_text_KO(v,t_mean)"]["both_content_graphs_empty"]
        and c["screen_both_KO(v_mean,t_mean)"]["both_content_graphs_empty"])
    cert["mechanism"] = (
        "per-dim-mean substitution makes every row of the content table identical, so "
        "every cosine in that modality's sim matrix is ONE bitwise-identical value c; the "
        "float32 mean over n_items^2 copies of c lands strictly above c, so the mask is "
        "ALL TRUE; the source then does t_sim[mask_v]=0, and item_num=len(nonzero(t_sim[i])) "
        "is the out-degree of BOTH graphs -> both graphs get zero edges -> h_v == h_t == 0.")
    return cert


# ---------------------------------------------------------------- self-test runner
def _main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="baby")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "_scratch" / "exact_ko"))
    args = ap.parse_args(argv)

    from recsys_bridge import load_frozen                            # noqa: E402
    from ranking_effects import evaluate_item_matrix, ranking_change  # noqa: E402

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    ck = CKPT_PIN.get(args.dataset)
    cfg, ds, model, test_loader = load_frozen("damrs", args.dataset, device, ckpt_path=ck)
    print(f"loaded ckpt={model._ckpt_name} pinned={model._ckpt_pinned}", flush=True)
    print(f"missing_keys={model._missing_keys}", flush=True)
    print(f"unexpected_keys={model._unexpected_keys}", flush=True)

    re_ = recon_error(model, args.dataset, device)
    print(f"recon_error = {re_:.3e}", flush=True)

    logged = None
    lp = LOGGED_RUN.get(args.dataset)
    if lp and Path(lp).is_file():
        logged = json.loads(Path(lp).read_text())["test_result"]["Recall@20"]

    vs = variants(model, args.dataset, device)
    res, base_topk = {}, None
    for name, (u, M) in vs.items():
        m, tk, _ = evaluate_item_matrix(u, M, test_loader, device)
        res[name] = {k: float(v) for k, v in m.items()}
        if name == "baseline":
            base_topk = tk
        else:
            res[name]["_ranking_change"] = ranking_change(base_topk, tk, k=20)
        print(f"  {name:18s} R@20={res[name]['Recall@20']:.8f} "
              f"dR@20={res[name]['Recall@20'] - res['baseline']['Recall@20']:+.8f}", flush=True)

    attr = attribution(model, args.dataset, device, heavy=True)
    out = {"model": "damrs", "dataset": args.dataset, "CLASS": CLASS, "CLASS_DESC": CLASS_DESC,
           "EXACTNESS": EXACTNESS, "ckpt": model._ckpt_path, "ckpt_pinned": model._ckpt_pinned,
           "recon_error": re_, "logged_test_Recall@20": logged,
           "baseline_minus_logged": (res["baseline"]["Recall@20"] - logged) if logged else None,
           "arms": res, "attribution": attr}
    op = Path(args.out); op.mkdir(parents=True, exist_ok=True)
    p = op / f"damrs_{args.dataset}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}", flush=True)
    print(json.dumps(attr, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
