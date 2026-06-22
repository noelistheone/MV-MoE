"""BA-I2I: Behavior-Aligned Image Item-Item recommender (self-built, image-centric).

Distinct from closed-form item-item models (e.g. EASE) in 3 ways: (1) IMAGE not text; (2) a LEARNED
co-purchase-supervised metric, NOT raw cosine similarity (this is exactly the blind spot that makes
every recommender ignore image: raw image cosine ~= chance 0.54 at predicting co-purchase, a learned
projection ~0.74); (3) NO EASE / ridge / Gram closed-form -- the kernel is a neural image metric,
used as item-kNN.

Pipeline:
  1. learn z = MLP(image_feats) so that co-purchased item pairs have high cosine (logsigmoid pos-neg).
  2. item-item kernel S[i,j] = topk cos(z_i, z_j).
  3. score(u,i) = sum_{j in train history(u)} S[i,j]   (item-kNN with the learned image kernel).
Saves results/bai/scores/baby_bai2i_<tag>.npy.  Also builds a RAW-image-kNN baseline (the blind spot).

Usage: python build_ba_i2i.py <feat: multi|cnn|dinov2|siglip2|text> [raw|learned] [topk]
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RECSYS = Path("/workspace/Recsys")
MECH = Path("/workspace/MechInterp")
SDIR = MECH / "results" / "bai" / "scores"
dev = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2024
torch.manual_seed(SEED); np.random.seed(SEED)
g = torch.Generator(device="cpu").manual_seed(SEED)


def load_feats(name, ds="baby"):
    if name == "multi":
        return np.load(MECH / "data" / "baby_multiimg.npy").astype(np.float32)   # baby only
    if name == "cnn":
        return np.load(RECSYS / "data" / ds / "image_feat.npy").astype(np.float32)
    if name == "dinov2":
        return np.load(MECH / "data" / "baby_dinov2_base.npy").astype(np.float32)
    if name == "siglip2":
        return np.load(MECH / "data" / "baby_siglip2_base_patch16_224.npy").astype(np.float32)
    if name == "text":
        return np.load(RECSYS / "data" / ds / "text_feat.npy").astype(np.float32)
    if name == "sig_text":            # SigLIP2 text tower of the title (VL-aligned)
        return np.load(MECH / "data" / f"{ds}_siglip2_text.npy").astype(np.float32)
    if name == "sig_img":             # SigLIP2 vision tower (image)
        return np.load(MECH / "data" / f"{ds}_siglip2_base_patch16_224.npy").astype(np.float32)
    if name == "sig_joint":           # VL joint = L2(image) (+) L2(title-text), same SigLIP2 space
        im = np.load(MECH / "data" / f"{ds}_siglip2_base_patch16_224.npy").astype(np.float32)
        tx = np.load(MECH / "data" / f"{ds}_siglip2_text.npy").astype(np.float32)
        l2 = lambda a: a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        return np.concatenate([l2(im), l2(tx)], axis=1)
    raise ValueError(name)


def load_train(ds="baby"):
    rows = np.loadtxt(RECSYS / "data" / ds / f"{ds}.inter", delimiter="\t", skiprows=1,
                      dtype=np.int64, usecols=(0, 1, 4))
    train = rows[rows[:, 2] == 0]
    nu = int(rows[:, 0].max()) + 1
    ni = int(rows[:, 1].max()) + 1
    return train, nu, ni


def copurchase_pairs(train):
    ub = defaultdict(list)
    for u, i in train[:, :2]:
        ub[int(u)].append(int(i))
    pairs = set()
    for items in ub.values():
        items = list(set(items))
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                x, y = items[a], items[b]
                pairs.add((x, y) if x < y else (y, x))
    return torch.tensor(sorted(pairs), dtype=torch.long)


class Proj(nn.Module):
    def __init__(self, din, d=128):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(din, 256), nn.GELU(), nn.Linear(256, d))

    def forward(self, x):
        return F.normalize(self.f(x), dim=1)


def learn_metric(F_, pairs, epochs=400, n_items=None):
    proj = Proj(F_.shape[1]).to(dev)
    opt = torch.optim.Adam(proj.parameters(), lr=1e-3, weight_decay=1e-4)
    P = pairs.to(dev)
    N = F_.shape[0]
    for ep in range(epochs):
        na = torch.randint(0, N, (P.shape[0],), device=dev)
        nb = torch.randint(0, N, (P.shape[0],), device=dev)
        z = proj(F_)
        pos = (z[P[:, 0]] * z[P[:, 1]]).sum(1)
        neg = (z[na] * z[nb]).sum(1)
        loss = -(F.logsigmoid(pos - neg)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    proj.eval()
    with torch.no_grad():
        return proj(F_)


def topk_kernel(z, k=20):
    """S[i,j] = cos(z_i,z_j) keep top-k per row (z already L2)."""
    N = z.shape[0]
    S = torch.zeros(N, N, device=dev)
    B = 1024
    for s in range(0, N, B):
        e = min(s + B, N)
        sim = z[s:e] @ z.t()
        sim[torch.arange(e - s), torch.arange(s, e)] = -2.0    # drop self
        kth = torch.topk(sim, k, dim=1).values[:, -1:]
        S[s:e] = torch.where(sim >= kth, sim, torch.zeros_like(sim))
    return S


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "baby"
    feat = sys.argv[2] if len(sys.argv) > 2 else "multi"
    mode = sys.argv[3] if len(sys.argv) > 3 else "learned"
    topk = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    SDIR.mkdir(parents=True, exist_ok=True)
    train, nu, ni = load_train(ds)
    feats = torch.tensor(load_feats(feat, ds), device=dev)
    R = torch.zeros(nu, ni, device=dev)
    R[train[:, 0], train[:, 1]] = 1.0

    if mode == "raw":
        z = F.normalize(feats, dim=1)
    else:
        pairs = copurchase_pairs(train)
        print(f"co-purchase pairs: {pairs.shape[0]}", flush=True)
        z = learn_metric(feats, pairs, n_items=ni)
    S = topk_kernel(z, k=topk)                                  # [ni, ni]
    scores = R @ S                                              # [nu, ni]
    out = SDIR / f"{ds}_bai2i_{feat}_{mode}_k{topk}.npy"
    np.save(out, scores.half().cpu().numpy())
    print(f"saved {out.name} shape={tuple(scores.shape)}", flush=True)


if __name__ == "__main__":
    main()
