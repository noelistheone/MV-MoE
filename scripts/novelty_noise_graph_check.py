"""Prior-audit check (CPU only): what does Ye et al.'s noise-replacement recipe do to
FREEDOM's frozen image kNN graph, compared with the real image graph and with
Pomo et al.'s N(0,1) recipe?

Recipes (from the released code, read 2026-09-23):
  Ye et al. (GAIR-Lab/MKF4MMRec, preprocessing-knockout-method.py, extract_noise_features):
      np.random.seed(42); image = np.random.normal(1, 0.1, (n_items, 4096))  -- fixed file, per item
  Pomo et al. (sisinflab/A-Comprehensive-Analysis-..., data/*/make_garbage.py):
      gaussian_noise = np.random.normal(0.0, 1.0, size=embs.shape)          -- fixed file, per item, unseeded

FREEDOM (MMRec) builds its item-item graph as top-k (k=10) cosine neighbours of the raw
features (self included), then freezes it. We build the same neighbour sets here and
report neighbour-set overlap and in-degree concentration. No training, no GPU.
Output: results/phase_novelty/noise_graph_check.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

DATA = Path("/workspace/Recsys/data")  # read-only
OUT = Path("/workspace/MechInterp/results/phase_novelty/noise_graph_check.json")
K = 10


def knn_sets(x: np.ndarray, k: int = K, chunk: int = 2048) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    t = t / t.norm(dim=1, keepdim=True).clamp_min(1e-12)
    out = np.empty((t.shape[0], k), dtype=np.int64)
    for s in range(0, t.shape[0], chunk):
        sim = t[s:s + chunk] @ t.T
        out[s:s + chunk] = torch.topk(sim, k, dim=1).indices.numpy()
    return out


def mean_pairwise_cos(x: np.ndarray, n: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=min(n, x.shape[0]), replace=False)
    t = torch.from_numpy(np.ascontiguousarray(x[idx], dtype=np.float32))
    t = t / t.norm(dim=1, keepdim=True).clamp_min(1e-12)
    s = t @ t.T
    m = s.shape[0]
    return float((s.sum() - s.diagonal().sum()) / (m * (m - 1)))


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Mean fraction of a row's k neighbours (self excluded) shared with the other graph."""
    tot = 0.0
    for i in range(a.shape[0]):
        sa = set(a[i].tolist()) - {i}
        sb = set(b[i].tolist()) - {i}
        tot += len(sa & sb) / max(len(sa), 1)
    return tot / a.shape[0]


def indegree_stats(nb: np.ndarray) -> dict:
    n = nb.shape[0]
    deg = np.bincount(nb.ravel(), minlength=n).astype(float)
    srt = np.sort(deg)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(srt) / (n * srt.sum()))
    return {"max_indegree": int(deg.max()), "gini_indegree": gini,
            "frac_items_indegree_ge_50": float((deg >= 50).mean())}


def main(datasets):
    res = {"script": "scripts/novelty_noise_graph_check.py", "k": K, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "recipes": {"ye_noise": "np.random.seed(42); N(mean=1, sd=0.1), 4096-d, per item, fixed",
                       "pomo_gauss": "N(0,1), 4096-d here (Pomo match the dim of their LVLM EOS embeddings), per item, fixed"},
           "datasets": {}}
    for d in datasets:
        v = np.load(DATA / d / "image_feat.npy")
        t = np.load(DATA / d / "text_feat.npy")
        n = v.shape[0]
        np.random.seed(42)
        ye = np.random.normal(1, 0.1, (n, v.shape[1]))
        pomo = np.random.default_rng(123).normal(0.0, 1.0, size=v.shape)
        g = {"real_image": knn_sets(v), "text": knn_sets(t), "ye_noise": knn_sets(ye), "pomo_gauss": knn_sets(pomo)}
        rng = np.random.default_rng(7)
        rand = np.stack([rng.choice(n, K, replace=False) for _ in range(n)])
        g["uniform_random"] = rand
        r = {"n_items": n,
             "mean_pairwise_cos": {"real_image": mean_pairwise_cos(v), "text": mean_pairwise_cos(t),
                                   "ye_noise": mean_pairwise_cos(ye), "pomo_gauss": mean_pairwise_cos(pomo)},
             "neighbour_overlap_with_real_image": {k: overlap(g[k], g["real_image"]) for k in g if k != "real_image"},
             "neighbour_overlap_with_text": {k: overlap(g[k], g["text"]) for k in g if k != "text"},
             "indegree": {k: indegree_stats(g[k]) for k in g}}
        res["datasets"][d] = r
        print(d, json.dumps(r, indent=1), flush=True)
        del v, t, ye, pomo, g
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main(sys.argv[1:] or ["baby", "sports"])
