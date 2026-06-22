"""Frugal generation of the MV-MoE content/CF views for MicroLens (98k x 17k), sized for a CONTENDED
GPU. The stock userknn_scores.py / build_ba_i2i.py allocate dense float32 R [nu,ni] (6.76 GB) plus
copies (~20 GB) and OOM. Here: sparse R, fp16, user-chunked matmuls, scores written straight to a CPU
array. Peak ~4 GB (user-kNN) / ~2 GB (i2i). Produces, in results/bai/scores/:
  microlens_userknn.npy, microlens_bai2i_cnn_learned_k20.npy, microlens_bai2i_text_learned_k20.npy
"""
import sys, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from pathlib import Path

RECSYS = Path("/workspace/Recsys"); SDIR = Path("/workspace/MechInterp/results/bai/scores")
dev = "cuda"; ds = "microlens"; CH = 1024


def load_train():
    rows = np.loadtxt(RECSYS / f"data/{ds}/{ds}.inter", delimiter="\t", skiprows=1,
                      dtype=np.int64, usecols=(0, 1, 4))
    tr = rows[rows[:, 2] == 0]
    return tr, int(rows[:, 0].max()) + 1, int(rows[:, 1].max()) + 1


def sparse_R(tr, nu, ni):
    idx = torch.tensor(np.stack([tr[:, 0], tr[:, 1]]), device=dev)
    return torch.sparse_coo_tensor(idx, torch.ones(idx.shape[1], device=dev), (nu, ni)).coalesce()


def save(name, cpu_arr):
    np.save(SDIR / name, cpu_arr); print(f"saved {name} {cpu_arr.shape}", flush=True)


# ---------- user-kNN ----------
def userknn(tr, nu, ni, topk=100):
    Rn = torch.zeros(nu, ni, device=dev, dtype=torch.float16)
    Rn[torch.tensor(tr[:, 0], device=dev), torch.tensor(tr[:, 1], device=dev)] = 1.0
    # R is binary -> L2 row-norm = sqrt(degree); scale in-place in fp16 (no 6.8GB float32 temp)
    deg = np.bincount(tr[:, 0], minlength=nu).astype(np.float32)
    inv = torch.tensor(1.0 / np.sqrt(np.clip(deg, 1.0, None)), device=dev, dtype=torch.float16)
    Rn.mul_(inv[:, None])                                      # 3.4 GB fp16, the only big tensor
    Rsp = sparse_R(tr, nu, ni)
    out = np.zeros((nu, ni), dtype=np.float16)
    for s in range(0, nu, CH):
        e = min(s + CH, nu)
        sim = (Rn[s:e] @ Rn.t()).float()                       # [b, nu]
        sim[torch.arange(e - s), torch.arange(s, e)] = 0.0
        kth = torch.topk(sim, topk, dim=1).values[:, -1:]
        sim = torch.where(sim >= kth, sim, torch.zeros_like(sim))
        sc = torch.sparse.mm(Rsp.t(), sim.t()).t()             # [b, ni] = sim @ R
        out[s:e] = sc.half().cpu().numpy(); del sim, sc
        if s % (CH * 20) == 0:
            torch.cuda.empty_cache(); print(f"  uknn {e}/{nu}", flush=True)
    del Rn; torch.cuda.empty_cache()
    save(f"{ds}_userknn.npy", out)


# ---------- behavior-aligned i2i ----------
class Proj(nn.Module):
    def __init__(self, din, d=128):
        super().__init__(); self.f = nn.Sequential(nn.Linear(din, 256), nn.GELU(), nn.Linear(256, d))
    def forward(self, x):
        return F.normalize(self.f(x), dim=1)


def copurchase_pairs(tr):
    from collections import defaultdict
    byu = defaultdict(list)
    for u, i in tr[:, :2]:
        byu[int(u)].append(int(i))
    P = set()
    for items in byu.values():
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                x, y = items[a], items[b]
                P.add((x, y) if x < y else (y, x))
    return torch.tensor(sorted(P), dtype=torch.long)


def learn_metric(Feat, pairs, epochs=400):
    proj = Proj(Feat.shape[1]).to(dev); opt = torch.optim.Adam(proj.parameters(), lr=1e-3, weight_decay=1e-4)
    Pz = pairs.to(dev); N = Feat.shape[0]
    for _ in range(epochs):
        na = torch.randint(0, N, (Pz.shape[0],), device=dev); nb = torch.randint(0, N, (Pz.shape[0],), device=dev)
        z = proj(Feat)
        loss = -(F.logsigmoid((z[Pz[:, 0]] * z[Pz[:, 1]]).sum(1) - (z[na] * z[nb]).sum(1))).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    proj.eval()
    with torch.no_grad():
        return proj(Feat)


def i2i(tr, nu, ni, featname, topk=20):
    fpath = RECSYS / "data" / ds / ("image_feat.npy" if featname == "cnn" else "text_feat.npy")
    Feat = torch.tensor(np.load(fpath).astype(np.float32), device=dev)
    z = learn_metric(Feat, copurchase_pairs(tr))               # [ni, d] L2
    # top-k item-item kernel S [ni, ni] fp16, row-chunked
    S = torch.zeros(ni, ni, device=dev, dtype=torch.float16)
    for s in range(0, ni, CH):
        e = min(s + CH, ni)
        sim = z[s:e] @ z.t()
        sim[torch.arange(e - s), torch.arange(s, e)] = -2.0
        kth = torch.topk(sim, topk, dim=1).values[:, -1:]
        S[s:e] = torch.where(sim >= kth, sim, torch.zeros_like(sim)).half()
    Rsp = sparse_R(tr, nu, ni)
    out = np.zeros((nu, ni), dtype=np.float16)
    for s in range(0, nu, CH):
        e = min(s + CH, nu)
        sc = torch.sparse.mm(_row_slice(Rsp, s, e), S.float())  # [b, ni]
        out[s:e] = sc.half().cpu().numpy(); del sc
    del S; torch.cuda.empty_cache()
    save(f"{ds}_bai2i_{featname}_learned_k20.npy", out)


def _row_slice(Rsp, s, e):
    """Rows [s:e] of a coalesced sparse [nu,ni] as a sparse [e-s, ni]."""
    idx = Rsp.indices(); val = Rsp.values()
    m = (idx[0] >= s) & (idx[0] < e)
    ii = torch.stack([idx[0][m] - s, idx[1][m]])
    return torch.sparse_coo_tensor(ii, val[m], (e - s, Rsp.shape[1])).coalesce()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    tr, nu, ni = load_train()
    print(f"microlens train={len(tr)} nu={nu} ni={ni}", flush=True)
    if which in ("all", "userknn"):
        userknn(tr, nu, ni)
    if which in ("all", "img"):
        i2i(tr, nu, ni, "cnn")
    if which in ("all", "txt"):
        i2i(tr, nu, ni, "text")
    print("VIEWS DONE", flush=True)
