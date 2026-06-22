"""MicroLens MV-MoE finalization, CHUNKED + memory-frugal (runs in <2 GB even on a contended GPU).
Score matrices are 98k x 17k fp16 (3.4 GB each); we never materialize a full [users,items] tensor on
GPU. Instead we stream USER-CHUNKS: per chunk, z-score each view's rows, weighted-sum, mask train,
topk, and accumulate Recall/NDCG/MAP exactly as ensemble_eval.evaluate defines them. MV-MoE = val-tuned
late fusion of 4 views {bagged GUME (mean of per-seed z-scored GUME), user-kNN, image-i2i, text-i2i}.
Outputs results/bai/finalize_microlens.json (bag, GUME single, +/-3SE band, per-user significance).
"""
import glob, json, math, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from scipy import stats
sys.path.insert(0, "/workspace/MechInterp/scripts")
import ensemble_eval as E

MET = ["Recall@10", "Recall@20", "NDCG@10", "NDCG@20", "MAP@10", "MAP@20"]
GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
SDIR = Path("/workspace/MechInterp/results/bai/scores")
ds = "microlens"; gpu = E.dev; CHUNK = 2048


def load_cpu(name):
    return torch.from_numpy(np.load(SDIR / name)).half()


def pairs(dct):
    us, iss = [], []
    for u, items in dct.items():
        for i in items:
            us.append(u); iss.append(i)
    return torch.tensor(us, device=gpu), torch.tensor(iss, device=gpu)


def chunk_eval(views, weights, tr_u, tr_i, gt_u, gt_i, nu, ni, ks=(10, 20), per_user=False):
    maxk = max(ks)
    disc = 1.0 / torch.log2(torch.arange(2, maxk + 2, device=gpu).float())
    idcg = torch.cumsum(disc, 0)
    keys = [f"{m}@{k}" for k in ks for m in ("Recall", "NDCG", "MAP")]
    sums = {k: 0.0 for k in keys}
    pu = {k: [] for k in keys} if per_user else None
    nkeep = 0
    for a in range(0, nu, CHUNK):
        b = min(a + CHUNK, nu); c = b - a
        S = torch.zeros((c, ni), device=gpu, dtype=torch.float16)
        for w, V in zip(weights, views):
            if w == 0:
                continue
            M = V[a:b].to(gpu).float()
            m = M.mean(1, keepdim=True); sd = M.std(1, keepdim=True).clamp_min(1e-6)
            S.add_(((M - m) / sd).half(), alpha=float(w)); del M
        tm = (tr_u >= a) & (tr_u < b); S[tr_u[tm] - a, tr_i[tm]] = -1e4
        gm = (gt_u >= a) & (gt_u < b)
        REL = torch.zeros((c, ni), device=gpu, dtype=torch.float16)
        REL[gt_u[gm] - a, gt_i[gm]] = 1.0
        nrel = REL.sum(1).float(); keep = nrel > 0
        topk = torch.topk(S, maxk, dim=1).indices; hits = torch.gather(REL, 1, topk).float()
        del S, REL
        for k in ks:
            rec = hits[:, :k].sum(1) / nrel.clamp(min=1)
            ndcg = (hits[:, :k] * disc[:k]).sum(1) / idcg[(nrel.clamp(max=k).long() - 1).clamp(min=0)]
            cum = torch.cumsum(hits[:, :k], 1); ranks = torch.arange(1, k + 1, device=gpu).float()
            mapk = ((cum / ranks) * hits[:, :k]).sum(1) / nrel.clamp(max=k).clamp(min=1)
            for nm, val in (("Recall", rec), ("NDCG", ndcg), ("MAP", mapk)):
                sums[f"{nm}@{k}"] += float(val[keep].sum().item())
                if per_user:
                    pu[f"{nm}@{k}"].append(val[keep].cpu().numpy())
        nkeep += int(keep.sum().item()); del hits; torch.cuda.empty_cache()
    res = {k: sums[k] / max(nkeep, 1) for k in keys}
    if per_user:
        return res, {k: np.concatenate(v) for k, v in pu.items()}
    return res


def val_tune(views, tru, tri, vu, vi, nu, ni):
    w = [1.0] * len(views)
    for _ in range(2):
        for j in range(len(views)):
            bj, bv = w[j], -1.0
            for g in GRID:
                w[j] = g
                v = chunk_eval(views, w, tru, tri, vu, vi, nu, ni, ks=(20,))["Recall@20"]
                if v > bv:
                    bv, bj = v, g
            w[j] = bj
    return w


def run():
    E.DATA = Path(f"/workspace/Recsys/data/{ds}/{ds}.inter"); E._CACHE.clear()
    train, valid, test, nu, ni = E.load_split()
    tru, tri = pairs(train); vu, vi = pairs(valid); teu, tei = pairs(test)
    seeds = sorted(int(re.search(r"_s(\d+)\.npy", p).group(1))
                   for p in glob.glob(str(SDIR / f"{ds}_bprdump_s*.npy")))
    print(f"seeds={seeds} nu={nu} ni={ni}", flush=True)

    # bagged GUME view = mean over seeds of z-scored GUME, computed CHUNK-wise on CPU output
    bagged = torch.zeros((nu, ni), dtype=torch.float16)
    for a in range(0, nu, CHUNK):
        b = min(a + CHUNK, nu)
        acc = torch.zeros((b - a, ni), device=gpu)
        for s in seeds:
            M = torch.from_numpy(np.load(SDIR / f"{ds}_bprdump_s{s}.npy", mmap_mode="r")[a:b]).to(gpu).float()
            m = M.mean(1, keepdim=True); sd = M.std(1, keepdim=True).clamp_min(1e-6)
            acc += (M - m) / sd; del M
        bagged[a:b] = (acc / len(seeds)).half().cpu(); del acc; torch.cuda.empty_cache()
    print("bagged GUME view built", flush=True)

    uk = load_cpu(f"{ds}_userknn.npy"); img = load_cpu(f"{ds}_bai2i_cnn_learned_k20.npy")
    txt = load_cpu(f"{ds}_bai2i_text_learned_k20.npy")
    views = [bagged, uk, img, txt]

    w = val_tune(views, tru, tri, vu, vi, nu, ni)
    bagres, bagpu = chunk_eval(views, w, tru, tri, teu, tei, nu, ni, per_user=True)
    bag = {m: round(bagres[m], 5) for m in MET}
    print("MV-MoE weights [bagGUME,uKNN,img,txt]:", w); print("MV-MoE bag:", bag, flush=True)

    # GUME single = seed_0 alone (z-score rank-preserving)
    g0 = [load_cpu(f"{ds}_bprdump_s{seeds[0]}.npy")]
    gres, gpu_ = chunk_eval(g0, [1.0], tru, tri, teu, tei, nu, ni, per_user=True)
    gume = {m: round(gres[m], 5) for m in MET}
    print("GUME single:", gume, flush=True)

    # per-seed band
    perseed = {m: [] for m in MET}
    for s in seeds:
        sv = [load_cpu(f"{ds}_bprdump_s{s}.npy"), uk, img, txt]
        ws = val_tune(sv, tru, tri, vu, vi, nu, ni)
        r = chunk_eval(sv, ws, tru, tri, teu, tei, nu, ni)
        for m in MET:
            perseed[m].append(r[m])
        print(f"  per-seed s{s} R@20={r['Recall@20']:.5f}", flush=True)
    three_SE = {m: round(3 * float(np.std(perseed[m], ddof=1)) / math.sqrt(len(seeds)), 5) for m in MET}

    sig = {}
    for m in MET:
        t, p2 = stats.ttest_rel(bagpu[m], gpu_[m]); p = p2 / 2 if t > 0 else 1 - p2 / 2
        sig[m] = {"p": float(p), "sig": bool(p < 0.05)}

    out = {"K": len(seeds), "seeds": seeds, "bag": bag, "gume_single_ensemble": gume,
           "three_SE": three_SE, "sig": sig, "bag_weights": w,
           "perseed": {m: [round(x, 5) for x in perseed[m]] for m in MET}}
    Path("/workspace/MechInterp/results/bai/finalize_microlens.json").write_text(json.dumps(out, indent=2))
    print("\n=== microlens MV-MoE ===")
    for m in MET:
        b_, e_, g_ = bag[m], three_SE[m], gume[m]
        print(f"  {m:10s} bag={b_:.4f} +/-{e_:.4f} band=[{b_-e_:.4f},{b_+e_:.4f}] GUME={g_:.4f} p={sig[m]['p']:.1e}")
    print("saved results/bai/finalize_microlens.json")


if __name__ == "__main__":
    run()
