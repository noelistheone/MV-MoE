"""Image-graph diffusion of GUME scores (a DIFFERENT image-utilization than item-kNN).

Instead of item-kNN over the user's binary history, propagate GUME's CONTINUOUS predicted scores
over the behavior-aligned learned-image item-item graph:
    diffused[u,i] = sum_j GUME_score[u,j] * S_img[j,i]
So items image-similar (learned metric) to items GUME already likes for user u get boosted -> image
acts where CF is confident-adjacent. Self-built, image-centric, learned-metric, NO EASE (not a closed-form item-item model).
Saves results/bai/scores/baby_imgdiff_<feat>.npy.

Usage: python build_image_diffusion.py <feat=multi> [topk=20]
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MECH = Path("/workspace/MechInterp")
SDIR = MECH / "results" / "bai" / "scores"
dev = "cuda" if torch.cuda.is_available() else "cpu"

# reuse the metric learner + kernel from build_ba_i2i
sys.path.insert(0, str(MECH / "scripts"))
from build_ba_i2i import load_feats, load_train, copurchase_pairs, learn_metric, topk_kernel  # noqa


def zr(S):
    return (S - S.mean(1, keepdim=True)) / (S.std(1, keepdim=True) + 1e-8)


def main():
    feat = sys.argv[1] if len(sys.argv) > 1 else "multi"
    topk = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    train, nu, ni = load_train()
    feats = torch.tensor(load_feats(feat), device=dev)
    pairs = copurchase_pairs(train)
    print(f"co-purchase pairs {pairs.shape[0]}; learning image metric...", flush=True)
    z = learn_metric(feats, pairs, n_items=ni)
    S = topk_kernel(z, k=topk)                                       # [ni, ni] learned-image kernel
    # GUME teacher scores: z-scored average of the 5 dumped GUME seeds
    G = torch.zeros(nu, ni, device=dev)
    seeds = [2024, 2025, 2026, 2027, 2028]
    for s in seeds:
        G = G + zr(torch.from_numpy(np.load(SDIR / f"baby_bprdump_s{s}.npy").astype(np.float32)).to(dev))
    G = G / len(seeds)
    diffused = G @ S                                                 # [nu, ni] propagate over image graph
    out = SDIR / f"baby_imgdiff_{feat}.npy"
    np.save(out, diffused.half().cpu().numpy())
    print(f"saved {out.name} shape={tuple(diffused.shape)}", flush=True)


if __name__ == "__main__":
    main()
