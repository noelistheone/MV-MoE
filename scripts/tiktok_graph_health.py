"""Health of FREEDOM's frozen image and text item-item kNN graphs: TikTok vs MicroLens vs Baby (CPU).

Graphs are built by the harness's own builder, src.data.graph_utils.build_knn_graph, with
FREEDOM's config (knn_k = 10 from configs/model/freedom.yaml) on CPU from the raw features, exactly
as FREEDOM._build_mm_adj does at construction (phase1_knockout.freedom_streams notes the model
built it on CPU; the builder is CPU/GPU tie-sensitive):
  * directed graph = build_knn_graph(feat, k, sym=False): each item's top-k cosine neighbours
    (self excluded); in-degree (k-occurrence) = how many items list j among their k neighbours.
  * FREEDOM's graph = build_knn_graph(feat, k) (sym=True: max(A, A^T), binary, D^-1/2 A D^-1/2).
Reported per modality: in-degree Gini, max in-degree, share of items never chosen (in-degree 0),
share of directed edges pointing at the top-1% items, and degree Gini/max of the symmetrized graph.
Between modalities: overlap of the undirected edge sets of FREEDOM's two graphs (|I & T| / |I|,
Jaccard). Against behavior: fraction of undirected edges whose two items share at least one user
in the TRAINING split (x_label == 0) -- for the image graph, the text graph and, as a reference,
200,000 uniformly random item pairs (seed 0). Mean pairwise cosine of each raw feature is exact:
(||sum_i f_i||^2 - n) / (n (n - 1)) over L2-normalized rows.

Output -> results/phase_shortvideo2/graph_health.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

ROOT = Path("/workspace/MechInterp")
sys.path.insert(0, str(ROOT / "scripts"))
import tiktok_common as tc                           # noqa: E402

tc.install_overlay()
from src.data.graph_utils import build_knn_graph     # noqa: E402

OUT = tc.SV2 / "graph_health.json"
DATASETS = ("tiktok", "microlens", "baby")


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def edges_of(g: torch.Tensor) -> np.ndarray:
    i = g.coalesce().indices().numpy()
    return i


def undirected_set(idx: np.ndarray, n: int) -> np.ndarray:
    a, b = np.minimum(idx[0], idx[1]), np.maximum(idx[0], idx[1])
    keep = a != b
    return np.unique(a[keep].astype(np.int64) * n + b[keep])


def cointeracted(keys: np.ndarray, n: int, Rc: sp.csc_matrix) -> np.ndarray:
    """For undirected pairs encoded a*n+b, whether items a and b share a training user."""
    a, b = keys // n, keys % n
    out = np.empty(len(keys), dtype=bool)
    for s in range(0, len(keys), 50000):
        A = Rc[:, a[s:s + 50000]]
        B = Rc[:, b[s:s + 50000]]
        out[s:s + 50000] = np.asarray(A.multiply(B).sum(0)).ravel() > 0
    return out


def mean_pairwise_cos(X: torch.Tensor) -> float:
    f = F.normalize(X.double(), dim=-1)
    n = f.shape[0]
    s = f.sum(0)
    return float((s @ s - n) / (n * (n - 1)))


def modality(feat: torch.Tensor, k: int) -> dict:
    n = feat.shape[0]
    gd = build_knn_graph(feat, k, sym=False)
    gs = build_knn_graph(feat, k)
    di = edges_of(gd)
    indeg = np.bincount(di[1], minlength=n)
    outdeg = np.bincount(di[0], minlength=n)
    assert (outdeg == k).all(), "directed kNN graph should have exactly k out-edges per item"
    top = max(1, int(round(0.01 * n)))
    si = edges_of(gs)
    sdeg = np.bincount(si[0], minlength=n)
    und = undirected_set(si, n)
    return {"n_items": n, "dim": int(feat.shape[1]), "mean_pairwise_cosine": mean_pairwise_cos(feat),
            "directed": {"in_degree_gini": gini(indeg), "in_degree_max": int(indeg.max()),
                         "in_degree_mean": float(indeg.mean()),
                         "share_items_in_degree_0": float((indeg == 0).mean()),
                         "share_edges_into_top1pct_items": float(np.sort(indeg)[::-1][:top].sum() / indeg.sum()),
                         "top1pct_n_items": top},
            "symmetrized_freedom": {"degree_gini": gini(sdeg), "degree_max": int(sdeg.max()),
                                    "degree_mean": float(sdeg.mean()), "n_undirected_edges": int(len(und))},
            "_und": und}


def main() -> int:
    t0 = time.time()
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "builder": "src.data.graph_utils.build_knn_graph (CPU)",
           "datasets": {}}
    for dsn in DATASETS:
        cfg, ds = tc.load_dataset(dsn)
        k = int(cfg.get("knn_k", 10))
        n = ds.n_items
        v = torch.from_numpy(np.asarray(ds.v_feat[:]).copy()).float()
        t = torch.from_numpy(np.asarray(ds.t_feat[:]).copy()).float()
        R = ds.train_matrix.tocsc()
        mi, mt = modality(v, k), modality(t, k)
        ui, ut = mi.pop("_und"), mt.pop("_und")
        inter = np.intersect1d(ui, ut, assume_unique=True)
        rng = np.random.default_rng(0)
        ra, rb = rng.integers(0, n, 200000), rng.integers(0, n, 200000)
        keep = ra != rb
        rk = np.minimum(ra[keep], rb[keep]).astype(np.int64) * n + np.maximum(ra[keep], rb[keep])
        co_i, co_t, co_r = cointeracted(ui, n, R), cointeracted(ut, n, R), cointeracted(rk, n, R)
        co_both = cointeracted(inter, n, R) if len(inter) else np.array([], bool)
        img_only = np.setdiff1d(ui, ut, assume_unique=True)
        rec = {"knn_k": k, "n_users": ds.n_users, "n_items": n, "n_train_interactions": int(ds.train_matrix.nnz),
               "image": mi, "text": mt,
               "image_text_overlap": {"shared_undirected_edges": int(len(inter)),
                                      "share_of_image_edges_in_text_graph": float(len(inter) / len(ui)),
                                      "share_of_text_edges_in_image_graph": float(len(inter) / len(ut)),
                                      "jaccard": float(len(inter) / (len(ui) + len(ut) - len(inter)))},
               "co_interaction": {
                   "definition": "undirected edge (i,j) counts if some user has both i and j in the TRAINING split",
                   "image_edges": float(co_i.mean()), "text_edges": float(co_t.mean()),
                   "image_edges_not_in_text_graph": float(cointeracted(img_only, n, R).mean()) if len(img_only) else None,
                   "edges_in_both_graphs": float(co_both.mean()) if len(co_both) else None,
                   "random_pairs_reference": float(co_r.mean()), "random_pairs_n": int(keep.sum()),
                   "image_lift_over_random": float(co_i.mean() / co_r.mean()) if co_r.mean() > 0 else None,
                   "text_lift_over_random": float(co_t.mean() / co_r.mean()) if co_r.mean() > 0 else None}}
        # Added diagnostic (not part of the requested measure): a handful of very heavy users can
        # saturate "shares at least one user" (TikTok: max training history 603 items). Recompute
        # with the heaviest 1% of training users removed, and report the degree context.
        udeg = np.asarray(ds.train_matrix.sum(1)).ravel()
        cut = np.quantile(udeg[udeg > 0], 0.99)
        light = sp.diags((udeg <= cut).astype(np.float32)) @ ds.train_matrix
        Rl = light.tocsc()
        cl_i, cl_t, cl_r = cointeracted(ui, n, Rl), cointeracted(ut, n, Rl), cointeracted(rk, n, Rl)
        ideg = np.asarray(ds.train_matrix.sum(0)).ravel()
        rec["co_interaction_without_top1pct_users"] = {
            "user_degree_cut_99pct": float(cut), "n_users_removed": int((udeg > cut).sum()),
            "image_edges": float(cl_i.mean()), "text_edges": float(cl_t.mean()),
            "random_pairs_reference": float(cl_r.mean()),
            "image_lift_over_random": float(cl_i.mean() / cl_r.mean()) if cl_r.mean() > 0 else None,
            "text_lift_over_random": float(cl_t.mean() / cl_r.mean()) if cl_r.mean() > 0 else None}
        rec["training_degree_context"] = {
            "user_degree_max": int(udeg.max()), "user_degree_median": float(np.median(udeg[udeg > 0])),
            "user_degree_mean": float(udeg[udeg > 0].mean()),
            "share_items_without_training_interaction": float((ideg == 0).mean())}
        out["datasets"][dsn] = rec
        OUT.write_text(json.dumps(out, indent=2))
        print(f"{dsn}: cos img {mi['mean_pairwise_cosine']:.3f} txt {mt['mean_pairwise_cosine']:.3f} | "
              f"in-deg Gini img {mi['directed']['in_degree_gini']:.3f} (max {mi['directed']['in_degree_max']}) "
              f"txt {mt['directed']['in_degree_gini']:.3f} (max {mt['directed']['in_degree_max']}) | "
              f"overlap |I&T|/|I| {rec['image_text_overlap']['share_of_image_edges_in_text_graph']:.3f} | "
              f"co-int img {co_i.mean():.3f} txt {co_t.mean():.3f} rand {co_r.mean():.4f} | w/o top1% users "
              f"img {cl_i.mean():.3f} txt {cl_t.mean():.3f} rand {cl_r.mean():.4f}", flush=True)
    out["seconds"] = time.time() - t0
    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT} ({out['seconds']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
