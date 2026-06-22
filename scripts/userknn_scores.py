"""Self-contained user-based kNN collaborative filtering score matrix (textbook CF, Sarwar 2001;
NOT item-item / EASE). score(u,i) = sum_v topk-cosine-sim(u,v) * R[v,i]. A different
inductive bias from GUME's item-graph GNN -> decorrelated ensemble partner.

Saves results/bai/scores/baby_userknn.npy [n_users, n_items] fp16.
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SDIR = Path("/workspace/MechInterp/results/bai/scores")
dev = "cuda" if torch.cuda.is_available() else "cpu"
DS = sys.argv[1] if len(sys.argv) > 1 else "baby"
TOPK_USERS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
DATA = Path(f"/workspace/Recsys/data/{DS}/{DS}.inter")


def main():
    SDIR.mkdir(parents=True, exist_ok=True)
    rows = np.loadtxt(DATA, delimiter="\t", skiprows=1, dtype=np.int64, usecols=(0, 1, 4))
    train = rows[rows[:, 2] == 0]
    nu = int(rows[:, 0].max()) + 1
    ni = int(rows[:, 1].max()) + 1
    R = torch.zeros(nu, ni, device=dev)
    R[train[:, 0], train[:, 1]] = 1.0
    Rn = torch.nn.functional.normalize(R, dim=1)               # cosine on interaction rows
    # user-user similarity in chunks, keep top-k neighbors, accumulate scores
    scores = torch.zeros(nu, ni, device=dev)
    B = 1024
    for s in range(0, nu, B):
        e = min(s + B, nu)
        sim = Rn[s:e] @ Rn.t()                                 # [b, nu]
        sim[torch.arange(e - s), torch.arange(s, e)] = 0.0     # zero self
        kth = torch.topk(sim, TOPK_USERS, dim=1).values[:, -1:]
        sim = torch.where(sim >= kth, sim, torch.zeros_like(sim))
        scores[s:e] = sim @ R                                  # [b, ni]
    np.save(SDIR / f"{DS}_userknn.npy", scores.half().cpu().numpy())
    print(f"saved user-kNN topk={TOPK_USERS} -> {DS}_userknn.npy shape={tuple(scores.shape)}")


if __name__ == "__main__":
    main()
