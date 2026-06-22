"""Does image help where CF is data-starved? Item-popularity-stratified recall.

Diagnosis: image carries strong item-item co-purchase signal but it is orthogonal to user-item ranking
once CF is present -- for HEAD items. CF, however, cannot rank low-degree (tail/cold) items well. Test:
split TEST items by their train-interaction count into popularity bins; measure Recall@K per bin for
GUME(bag) vs GUME(bag)+image-content. If image lifts TAIL-bin recall, image solves the cold-item problem
even though aggregate ranking is CF-dominated.

Usage: python tail_stratified.py <ds=baby> [content=sig_img]
Reads results/bai/scores/. GPU, fp16-lean.
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

RECSYS = Path("/workspace/Recsys")
SDIR = Path("/workspace/MechInterp/results/bai/scores")
dev = "cuda" if torch.cuda.is_available() else "cpu"
KS = [20]


def load_split(ds):
    rows = np.loadtxt(RECSYS / "data" / ds / f"{ds}.inter", delimiter="\t", skiprows=1,
                      dtype=np.int64, usecols=(0, 1, 4))
    train, test = defaultdict(list), defaultdict(list)
    for u, i, lab in rows:
        if lab == 0:
            train[int(u)].append(int(i))
        elif lab == 2:
            test[int(u)].append(int(i))
    nu = int(rows[:, 0].max()) + 1
    ni = int(rows[:, 1].max()) + 1
    pop = np.bincount(rows[rows[:, 2] == 0][:, 1], minlength=ni)   # train degree per item
    return train, test, nu, ni, pop


def zscore(S):
    m = S.mean(1, keepdim=True, dtype=torch.float32)
    var = (S * S).mean(1, keepdim=True, dtype=torch.float32) - m * m
    return (S - m.half()) / ((var.clamp(min=0) + 1e-8).sqrt()).half()


def load(name):
    return torch.from_numpy(np.load(SDIR / name)).to(dev).half()


def topk_recall_per_bin(S, train, test, pop, ni, bins, k=20):
    # mask train
    tu, ti = [], []
    for u, items in train.items():
        for it in items:
            tu.append(u); ti.append(it)
    S[torch.tensor(tu, device=dev), torch.tensor(ti, device=dev)] = -1e9
    topk = torch.topk(S, k, dim=1).indices.cpu().numpy()
    # per-test-item hit, grouped by that item's popularity bin
    bin_hit = {b: [0, 0] for b in range(len(bins) + 1)}   # [hits, total]
    logpop = np.log1p(pop)
    edges = np.quantile(logpop[pop > 0], bins)            # popularity quantile edges
    for u, items in test.items():
        tset = set(topk[u])
        for it in set(items):
            b = int(np.searchsorted(edges, logpop[it]))
            bin_hit[b][1] += 1
            if it in tset:
                bin_hit[b][0] += 1
    return bin_hit, edges


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "baby"
    content = sys.argv[2] if len(sys.argv) > 2 else "sig_img"
    train, test, nu, ni, pop = load_split(ds)
    seeds = [2024, 2025, 2026, 2027, 2028]
    gfiles = [f"{ds}_bprdump_s{s}.npy" for s in seeds if (SDIR / f"{ds}_bprdump_s{s}.npy").exists()]
    cfile = f"{ds}_bai2i_{content}_learned_k20.npy"
    bins = [0.5, 0.8, 0.95]   # tail / mid / head / top quantile edges of log-popularity

    # GUME bag (uniform z-scored)
    Sg = torch.zeros(nu, ni, device=dev, dtype=torch.float16)
    for f in gfiles:
        Sg += zscore(load(f))
    # + content (small weight, like the ensemble) + userKNN for a realistic strong base
    uk = load(f"{ds}_userknn.npy")
    base = Sg.clone() + 0.25 * zscore(uk)                      # GUME-bag + userKNN (the strong CF base)
    withc = base + 0.25 * zscore(load(cfile))                 # + image content

    print(f"{ds}: bins=log-pop quantiles {bins}; content={content}", flush=True)
    for label, S in [("CF-base (GUMEbag+userKNN)", base), ("CF-base + image", withc)]:
        bh, edges = topk_recall_per_bin(S.clone(), train, test, pop, ni, bins)
        line = " | ".join(f"bin{b}:{(h/t if t else 0):.4f}(n={t})" for b, (h, t) in sorted(bh.items()))
        print(f"  {label:28s} R@20 by pop-bin (bin0=tail..binN=head): {line}", flush=True)
    print("  pop-quantile edges (log1p):", [round(float(e), 2) for e in edges], flush=True)


if __name__ == "__main__":
    main()
