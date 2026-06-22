"""Critical decomposition: is the multimodal 'image' co-purchase signal real beyond TEXT?

Held-out co-purchase pair prediction (learned metric) for SigLIP2 image-only / title-text-only /
VL-joint, plus the INCREMENTS:
   delta_image = AUC(joint) - AUC(text)   # what image adds BEYOND text
   delta_text  = AUC(joint) - AUC(image)  # what text adds BEYOND image
If delta_image ~ 0 while delta_text is large, the VL 'image' signal is actually text -> the basis of
the critical claim that VLM-enrichment gains (VLIF-class) are language, not vision.

Writes results/bai/vlm_decompose.json. GPU.  Usage: python vlm_decompose.py <ds=baby>
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RECSYS = Path("/workspace/Recsys")
MECH = Path("/workspace/MechInterp")
OUT = MECH / "results" / "bai"
dev = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2024
torch.manual_seed(SEED); np.random.seed(SEED)
g = torch.Generator(device="cpu").manual_seed(SEED)


def l2(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


def load(ds):
    im = np.load(MECH / "data" / f"{ds}_siglip2_base_patch16_224.npy").astype(np.float32)
    tx = np.load(MECH / "data" / f"{ds}_siglip2_text.npy").astype(np.float32)
    gtxt = np.load(RECSYS / "data" / ds / "text_feat.npy").astype(np.float32)   # GUME's text (SBERT)
    feats = {
        "sig_image": l2(im),
        "sig_text": l2(tx),
        "sig_joint": np.concatenate([l2(im), l2(tx)], 1),
        "gume_text": l2(gtxt),
        "gume_text+sig_image": np.concatenate([l2(gtxt), l2(im)], 1),
    }
    rows = np.loadtxt(RECSYS / "data" / ds / f"{ds}.inter", delimiter="\t", skiprows=1,
                      dtype=np.int64, usecols=(0, 1, 4))
    train = rows[rows[:, 2] == 0][:, :2]
    return feats, train, im.shape[0]


def pairs(train):
    ub = defaultdict(list)
    for u, i in train:
        ub[int(u)].append(int(i))
    P = set()
    for items in ub.values():
        items = list(set(items))
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                x, y = items[a], items[b]
                P.add((x, y) if x < y else (y, x))
    return np.array(sorted(P), dtype=np.int64)


class Proj(nn.Module):
    def __init__(self, din, d=128, mlp=True):
        super().__init__()
        self.f = (nn.Sequential(nn.Linear(din, 256), nn.GELU(), nn.Linear(256, d)) if mlp
                  else nn.Linear(din, d))

    def forward(self, x):
        return F.normalize(self.f(x), dim=1)


def auc(sp, sn):
    s = torch.cat([sp, sn]); y = torch.cat([torch.ones_like(sp), torch.zeros_like(sn)])
    order = torch.argsort(s); ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(len(s), device=s.device, dtype=torch.float)
    npos = sp.numel(); nneg = sn.numel()
    return ((ranks[y == 1].sum() - npos * (npos - 1) / 2) / (npos * nneg)).item()


def metric_auc(feat, ptr, pte, epochs=400):
    Fm = torch.tensor(feat, device=dev); N = Fm.shape[0]
    proj = Proj(Fm.shape[1]).to(dev)
    opt = torch.optim.Adam(proj.parameters(), lr=1e-3, weight_decay=1e-4)
    ptr = torch.tensor(ptr, device=dev); pte = torch.tensor(pte, device=dev)
    for _ in range(epochs):
        na = torch.randint(0, N, (ptr.shape[0],), device=dev)
        nb = torch.randint(0, N, (ptr.shape[0],), device=dev)
        z = proj(Fm)
        loss = -(F.logsigmoid((z[ptr[:, 0]] * z[ptr[:, 1]]).sum(1) - (z[na] * z[nb]).sum(1))).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    proj.eval()
    with torch.no_grad():
        z = proj(Fm)
        na = torch.randint(0, N, (pte.shape[0],), device=dev)
        nb = torch.randint(0, N, (pte.shape[0],), device=dev)
        return auc((z[pte[:, 0]] * z[pte[:, 1]]).sum(1), (z[na] * z[nb]).sum(1))


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "baby"
    OUT.mkdir(parents=True, exist_ok=True)
    feats, train, n = load(ds)
    P = pairs(train)
    perm = torch.randperm(len(P), generator=g).numpy(); P = P[perm]
    cut = int(0.8 * len(P)); ptr, pte = P[:cut], P[cut:]
    print(f"{ds}: {n} items, {len(P)} co-purchase pairs", flush=True)
    res = {"dataset": ds, "n_pairs": int(len(P)), "auc": {}}
    for k, fv in feats.items():
        a = np.mean([metric_auc(fv, ptr, pte) for _ in range(2)])
        res["auc"][k] = float(a)
        print(f"  held-out co-purchase AUC  {k:22s} = {a:.4f}", flush=True)
    a = res["auc"]
    res["delta_image_beyond_text"] = a["sig_joint"] - a["sig_text"]
    res["delta_text_beyond_image"] = a["sig_joint"] - a["sig_image"]
    res["delta_image_beyond_gumetext"] = a["gume_text+sig_image"] - a["gume_text"]
    print(f"\n  Δ image beyond SigLIP-text  = {res['delta_image_beyond_text']:+.4f}", flush=True)
    print(f"  Δ text  beyond SigLIP-image = {res['delta_text_beyond_image']:+.4f}", flush=True)
    print(f"  Δ image beyond GUME-text    = {res['delta_image_beyond_gumetext']:+.4f}", flush=True)
    (OUT / f"vlm_decompose_{ds}.json").write_text(json.dumps(res, indent=2))
    print(f"wrote vlm_decompose_{ds}.json", flush=True)


if __name__ == "__main__":
    main()
