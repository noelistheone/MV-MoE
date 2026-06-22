"""Self-contained evaluator + score-level ensembler (no Recsys eval code).

Reads the baby split from baby.inter (x_label 0/1/2), loads one or more dumped score matrices
(results/bai/scores/*.npy, fp16 [n_users,n_items]), z-score-normalizes each, combines with weights,
masks training items, and computes Recall@K / NDCG@K for K in {5,10,20,50} on valid and test.

Usage:
  # sanity: eval a single dumped matrix, compare to the model's reported json
  python ensemble_eval.py --scores baby_bprdump_s2024.npy
  # search ensemble weights on VALID over several matrices, report TEST
  python ensemble_eval.py --search baby_bprdump_s2024.npy baby_ssmd_s2024.npy ...
"""
from __future__ import annotations
import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SDIR = Path("/workspace/MechInterp/results/bai/scores")
dev = "cuda" if torch.cuda.is_available() else "cpu"
KS = [5, 10, 20, 50]
DATA = Path("/workspace/Recsys/data/baby/baby.inter")  # overridden by --dataset in main()


def load_split():
    rows = np.loadtxt(DATA, delimiter="\t", skiprows=1, dtype=np.int64, usecols=(0, 1, 4))
    train, valid, test = defaultdict(list), defaultdict(list), defaultdict(list)
    for u, i, lab in rows:
        (train if lab == 0 else valid if lab == 1 else test)[int(u)].append(int(i))
    nu = int(rows[:, 0].max()) + 1
    ni = int(rows[:, 1].max()) + 1
    return train, valid, test, nu, ni


def zscore(S):
    # per-row z-score; reductions accumulate in fp32 (dtype=) so no full float32 copy of the big matrix
    m = S.mean(1, keepdim=True, dtype=torch.float32)
    var = (S * S).mean(1, keepdim=True, dtype=torch.float32) - m * m
    sd = (var.clamp(min=0) + 1e-8).sqrt()
    return (S - m.half()) / sd.half()


_CACHE = {}


def _prep(train, gt, nu, ni):
    """Precompute (once, cached) the train-mask index and the gt relevance matrix + per-user #rel."""
    key = (id(train), id(gt))
    if key in _CACHE:
        return _CACHE[key]
    tu, ti = [], []
    for u, items in train.items():
        for it in items:
            tu.append(u); ti.append(it)
    train_idx = (torch.tensor(tu, device=dev), torch.tensor(ti, device=dev))
    gu, gi = [], []
    for u, items in gt.items():
        for it in set(items):
            gu.append(u); gi.append(it)
    REL = torch.zeros(nu, ni, device=dev, dtype=torch.float16)
    REL[torch.tensor(gu, device=dev), torch.tensor(gi, device=dev)] = 1.0
    nrel = REL.sum(1).float()                                   # [nu]
    _CACHE[key] = (train_idx, REL, nrel)
    return _CACHE[key]


def evaluate(S, train, gt, nu, ni, ks=KS):
    """Vectorized: mask train, topk, gather hits from the relevance matrix, Recall/NDCG@k over GPU."""
    (tu, ti), REL, nrel = _prep(train, gt, nu, ni)
    # S is a freshly combined tensor (not reused by caller) -> mask in place, no clone (saves a big copy)
    S[tu, ti] = -1e9
    maxk = max(ks)
    topk = torch.topk(S, maxk, dim=1).indices                  # [nu, maxk]
    hits = torch.gather(REL, 1, topk).float()                  # [nu, maxk]
    del S
    discounts = 1.0 / torch.log2(torch.arange(2, maxk + 2, device=dev).float())
    idcg_cum = torch.cumsum(discounts, 0)                       # idcg_cum[k-1] = ideal DCG@k (all rel)
    keep = nrel > 0
    out = {}
    for k in ks:
        rec_k = hits[:, :k].sum(1) / nrel.clamp(min=1)
        dcg_k = (hits[:, :k] * discounts[:k]).sum(1)
        idcg_k = idcg_cum[(nrel.clamp(max=k).long() - 1).clamp(min=0)]
        ndcg_k = dcg_k / idcg_k
        prec_k = hits[:, :k].sum(1) / k
        # MAP@k, matching Recsys src/evaluation/metrics.py::map_at_k exactly
        cum_hits = torch.cumsum(hits[:, :k], dim=1)
        ranks = torch.arange(1, k + 1, device=dev).float()
        ap = ((cum_hits / ranks) * hits[:, :k]).sum(1)
        denom = nrel.clamp(max=k).clamp(min=1)
        map_k = ap / denom
        out[f"Recall@{k}"] = rec_k[keep].mean().item()
        out[f"NDCG@{k}"] = ndcg_k[keep].mean().item()
        out[f"Precision@{k}"] = prec_k[keep].mean().item()
        out[f"MAP@{k}"] = map_k[keep].mean().item()
    return out


def load_scores(names):
    mats = []
    for n in names:
        p = SDIR / n if not n.startswith("/") else Path(n)
        mats.append(torch.from_numpy(np.load(p)).to(dev).half())   # keep fp16 (big matrices)
    return mats


def combine(mats, weights):
    S = torch.zeros_like(mats[0])                                  # fp16
    for w, M in zip(weights, mats):
        S += w * zscore(M)                                         # in-place accumulate, fp16
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="*", default=[], help="single-matrix eval (sanity)")
    ap.add_argument("--search", nargs="*", default=[], help="matrices to ensemble (weight search on valid)")
    ap.add_argument("--dump_teacher", default=None, help="save the combined ensemble matrix to scores/<name>.npy")
    ap.add_argument("--dataset", default="baby")
    args = ap.parse_args()
    global DATA
    DATA = Path(f"/workspace/Recsys/data/{args.dataset}/{args.dataset}.inter")
    train, valid, test, nu, ni = load_split()
    print(f"users={nu} items={ni} | val users={len(valid)} test users={len(test)}")

    if args.scores:
        mats = load_scores(args.scores)
        S = combine(mats, [1.0] * len(mats))
        vr = evaluate(S, train, valid, nu, ni)
        tr = evaluate(S, train, test, nu, ni)
        print(f"[single] {args.scores}")
        print("  VALID:", {k: round(v, 5) for k, v in vr.items() if 'Recall' in k})
        print("  TEST :", {k: round(v, 5) for k, v in tr.items()})
        return

    if args.search:
        mats = load_scores(args.search)
        n = len(mats)
        grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        best, bestw = -1, None
        # coordinate-ish: full grid if small n, else greedy
        combos = itertools.product(grid, repeat=n) if n <= 3 else None
        if combos is None:
            # greedy: start all-1, refine each weight
            w = [1.0] * n
            for it in range(2):
                for j in range(n):
                    bj, bv = w[j], -1
                    for g in grid:
                        w[j] = g
                        vr = evaluate(combine(mats, w), train, valid, nu, ni, ks=[20])["Recall@20"]
                        if vr > bv:
                            bv, bj = vr, g
                    w[j] = bj
            bestw = w
        else:
            for w in combos:
                if sum(w) == 0:
                    continue
                vr = evaluate(combine(mats, list(w)), train, valid, nu, ni, ks=[20])["Recall@20"]
                if vr > best:
                    best, bestw = vr, list(w)
        Stest = combine(mats, bestw)
        vr = evaluate(Stest, train, valid, nu, ni)
        tr = evaluate(Stest, train, test, nu, ni)
        print(f"[ensemble] files={args.search}")
        print(f"  best weights (val-tuned): {bestw}")
        print("  VALID Recall@20:", round(vr['Recall@20'], 5))
        print("  TEST :", {k: round(v, 5) for k, v in tr.items()})
        if args.dump_teacher:
            np.save(SDIR / args.dump_teacher, Stest.half().cpu().numpy())
            print(f"  saved teacher matrix -> {SDIR / args.dump_teacher}")


if __name__ == "__main__":
    main()
