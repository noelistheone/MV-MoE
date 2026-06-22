"""Do items co-selected by the SAME user share generalizable structure in image space?

Three questions, on Baby (dev):
1. DESCRIPTIVE: are user baskets visually coherent? intra-basket mean cosine vs random baskets,
   for image vs text (L2-normalized features).
2. CRUX (generalization): does a learned projection of image features predict co-purchase on
   HELD-OUT item pairs? Build co-purchase positive pairs (item i,j co-occur in some user basket),
   split pairs 80/20, train a projection (linear + MLP) with a logistic-on-cosine objective vs
   random negatives, report TRAIN vs TEST AUC. A large test-AUC>0.5 => extractable generalizable
   image signal (problem is extraction). test-AUC~0.5 with high train-AUC => overfitting, no
   generalizable signal. Compare image vs text vs (optionally) raw cosine.
3. WHAT they share: top directions of the learned image metric + raw-cosine AUC.

Writes results/bai/image_basket_analysis.json. GPU.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RECSYS = Path("/workspace/Recsys")
OUT = Path("/workspace/MechInterp/results/bai")
DATA = RECSYS / "data" / "baby"
dev = "cuda" if torch.cuda.is_available() else "cpu"
g = torch.Generator(device="cpu").manual_seed(2024)


def load():
    v = np.load(DATA / "image_feat.npy").astype(np.float32)
    t = np.load(DATA / "text_feat.npy").astype(np.float32)
    inter = np.loadtxt(DATA / "baby.inter", delimiter="\t", skiprows=1, dtype=np.int64,
                       usecols=(0, 1, 4))   # user, item, x_label
    return v, t, inter


def l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def basket_coherence(feat, baskets, n_rand=20000):
    """mean pairwise cosine within baskets vs random pairs (feat already L2-normalized)."""
    F_ = torch.tensor(feat, device=dev)
    intra = []
    for items in baskets:
        if len(items) < 2:
            continue
        e = F_[items]
        sim = e @ e.t()
        n = len(items)
        intra.append((sim.sum() - n) / (n * (n - 1)))   # exclude diagonal (=1)
    intra = torch.stack(intra).mean().item()
    # random pairs
    N = F_.shape[0]
    a = torch.randint(0, N, (n_rand,), generator=g); b = torch.randint(0, N, (n_rand,), generator=g)
    rand = (F_[a] * F_[b]).sum(1).mean().item()
    return intra, rand


def build_pairs(train_inter, max_items=7050):
    """unique unordered co-purchase item pairs from train baskets."""
    from collections import defaultdict
    ub = defaultdict(list)
    for u, i in train_inter:
        ub[u].append(i)
    pairs = set()
    for u, items in ub.items():
        items = list(set(items))
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                x, y = items[a], items[b]
                pairs.add((x, y) if x < y else (y, x))
    return np.array(sorted(pairs), dtype=np.int64)


class Proj(nn.Module):
    def __init__(self, din, d=64, mlp=False):
        super().__init__()
        if mlp:
            self.f = nn.Sequential(nn.Linear(din, d), nn.GELU(), nn.Linear(d, d))
        else:
            self.f = nn.Linear(din, d)
    def forward(self, x):
        return F.normalize(self.f(x), dim=1)


def auc(scores_pos, scores_neg):
    # rank-based AUC = P(pos>neg)
    s = torch.cat([scores_pos, scores_neg])
    y = torch.cat([torch.ones_like(scores_pos), torch.zeros_like(scores_neg)])
    order = torch.argsort(s)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(len(s), device=s.device, dtype=torch.float)
    n_pos = scores_pos.numel(); n_neg = scores_neg.numel()
    return ((ranks[y == 1].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)).item()


def train_metric(feat, pairs_tr, pairs_te, mlp=False, epochs=300, wd=1e-4):
    F_ = torch.tensor(feat, device=dev)
    N = F_.shape[0]
    proj = Proj(F_.shape[1], 64, mlp=mlp).to(dev)
    opt = torch.optim.Adam(proj.parameters(), lr=1e-3, weight_decay=wd)
    ptr = torch.tensor(pairs_tr, device=dev); pte = torch.tensor(pairs_te, device=dev)
    for ep in range(epochs):
        proj.train()
        # negatives resampled each epoch
        na = torch.randint(0, N, (ptr.shape[0],), device=dev)
        nb = torch.randint(0, N, (ptr.shape[0],), device=dev)
        z = proj(F_)
        pos = (z[ptr[:, 0]] * z[ptr[:, 1]]).sum(1)
        neg = (z[na] * z[nb]).sum(1)
        loss = -(F.logsigmoid(pos - neg)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    proj.eval()
    with torch.no_grad():
        z = proj(F_)
        # eval AUC on held-out pairs vs fresh random negatives
        def pair_auc(pairs):
            na = torch.randint(0, N, (pairs.shape[0],), device=dev)
            nb = torch.randint(0, N, (pairs.shape[0],), device=dev)
            sp = (z[pairs[:, 0]] * z[pairs[:, 1]]).sum(1)
            sn = (z[na] * z[nb]).sum(1)
            return auc(sp, sn)
        return pair_auc(ptr), pair_auc(pte)


def raw_cosine_auc(feat, pairs_te):
    F_ = torch.tensor(feat, device=dev); N = F_.shape[0]
    pte = torch.tensor(pairs_te, device=dev)
    na = torch.randint(0, N, (pte.shape[0],), device=dev)
    nb = torch.randint(0, N, (pte.shape[0],), device=dev)
    sp = (F_[pte[:, 0]] * F_[pte[:, 1]]).sum(1)
    sn = (F_[na] * F_[nb]).sum(1)
    return auc(sp, sn)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v, t, inter = load()
    vN, tN = l2(v), l2(t)
    train = inter[inter[:, 2] == 0][:, :2]
    from collections import defaultdict
    ub = defaultdict(list)
    for u, i in train:
        ub[u].append(i)
    baskets = [list(set(x)) for x in ub.values()]

    res = {"dataset": "baby", "n_items": int(v.shape[0]),
           "image_dim": int(v.shape[1]), "text_dim": int(t.shape[1]),
           "image_raw_norm_mean": float(np.linalg.norm(v, axis=1).mean()),
           "text_raw_norm_mean": float(np.linalg.norm(t, axis=1).mean())}

    # 1. descriptive coherence
    iv, rv = basket_coherence(vN, baskets)
    it, rt = basket_coherence(tN, baskets)
    res["coherence"] = {
        "image_intra_basket_cos": iv, "image_random_cos": rv, "image_lift": iv - rv,
        "text_intra_basket_cos": it, "text_random_cos": rt, "text_lift": it - rt}

    # 2. crux: held-out co-purchase pair prediction
    pairs = build_pairs(train)
    perm = torch.randperm(len(pairs), generator=g).numpy()
    pairs = pairs[perm]
    cut = int(0.8 * len(pairs))
    ptr, pte = pairs[:cut], pairs[cut:]
    res["n_copurchase_pairs"] = int(len(pairs))
    res["pair_auc"] = {
        "image_raw_cos": raw_cosine_auc(vN, pte),
        "text_raw_cos": raw_cosine_auc(tN, pte),
    }
    for name, feat in [("image", vN), ("text", tN)]:
        tr_lin, te_lin = train_metric(feat, ptr, pte, mlp=False)
        tr_mlp, te_mlp = train_metric(feat, ptr, pte, mlp=True)
        res["pair_auc"][f"{name}_learned_linear_train"] = tr_lin
        res["pair_auc"][f"{name}_learned_linear_test"] = te_lin
        res["pair_auc"][f"{name}_learned_mlp_train"] = tr_mlp
        res["pair_auc"][f"{name}_learned_mlp_test"] = te_mlp

    (OUT / "image_basket_analysis.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
