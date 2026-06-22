"""Consolidated finalization (5 seeds/dataset): component ablation, 5-seed bagged MV-MoE point
estimate, per-user significance vs GUME, and the +/-3*SE seed band (SE = perseed_std/sqrt(K)).
Emits results/bai/finalize.json and LaTeX-ready rows. MV-MoE = bagged (the win depends on bagging:
a single-seed run does not reliably beat GUME). The band = bag +/- 3*SE; we verify it clears GUME.
"""
import glob, json, math, re
from pathlib import Path
import numpy as np, torch
from scipy import stats
import ensemble_eval as E

GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
MET = ["Recall@10", "Recall@20", "NDCG@10", "NDCG@20", "MAP@10", "MAP@20"]
SDIR = Path("/workspace/MechInterp/results/bai/scores")
seeds_of = lambda ds: sorted(int(re.search(r"_s(\d+)\.npy", p).group(1))
                             for p in glob.glob(str(SDIR / f"{ds}_bprdump_s*.npy")))
perseed = json.load(open("/workspace/MechInterp/results/bai/mvmoe_seeds.json"))


def val_tune(mats, train, valid, test, nu, ni):
    w = [1.0] * len(mats)
    for _ in range(2):
        for j in range(len(mats)):
            bj, bv = w[j], -1.0
            for g in GRID:
                w[j] = g
                v = E.evaluate(E.combine(mats, w), train, valid, nu, ni, ks=[20])["Recall@20"]
                if v > bv:
                    bv, bj = v, g
            w[j] = bj
    return E.combine(mats, w)


def peruser(S, train, gt, nu, ni):
    (tu, ti), REL, nrel = E._prep(train, gt, nu, ni)
    S = S.clone(); S[tu, ti] = -1e9
    topk = torch.topk(S, 20, dim=1).indices
    hits = torch.gather(REL, 1, topk).float()
    disc = 1.0 / torch.log2(torch.arange(2, 22, device=E.dev).float())
    idcg = torch.cumsum(disc, 0); keep = nrel > 0; out = {}
    for k in (10, 20):
        out[f"Recall@{k}"] = (hits[:, :k].sum(1) / nrel.clamp(min=1))[keep].cpu().numpy()
        out[f"NDCG@{k}"] = ((hits[:, :k] * disc[:k]).sum(1) / idcg[(nrel.clamp(max=k).long() - 1).clamp(min=0)])[keep].cpu().numpy()
        cum = torch.cumsum(hits[:, :k], 1); r = torch.arange(1, k + 1, device=E.dev).float()
        out[f"MAP@{k}"] = (((cum / r) * hits[:, :k]).sum(1) / nrel.clamp(max=k).clamp(min=1))[keep].cpu().numpy()
    return out


import sys
DATASETS = sys.argv[1:] or ["baby", "sports", "clothing"]
out = {}
for ds in DATASETS:
    E.DATA = Path(f"/workspace/Recsys/data/{ds}/{ds}.inter"); E._CACHE.clear(); torch.cuda.empty_cache()
    train, valid, test, nu, ni = E.load_split()
    sd = seeds_of(ds); sf = [f"{ds}_bprdump_s{s}.npy" for s in sd]
    uk, img, txt = f"{ds}_userknn.npy", f"{ds}_bai2i_sig_img_learned_k20.npy", f"{ds}_bai2i_sig_text_learned_k20.npy"
    rungs = {"GUME (single)": [sf[0]], "+ seed bagging": sf, "+ user-kNN": sf + [uk],
             "+ image-i2i": sf + [uk, img], "+ text-i2i (full)": sf + [uk, img, txt]}
    abl = {}
    bagS = None
    for name, files in rungs.items():
        mats = E.load_scores(files)
        S = val_tune(mats, train, valid, test, nu, ni)
        tr = E.evaluate(S.clone(), train, test, nu, ni)
        abl[name] = {m: round(tr[m], 5) for m in ["Recall@20", "NDCG@20"]}
        if name == "+ text-i2i (full)":
            bagS = S.clone(); bag = {m: round(tr[m], 5) for m in MET}
        del mats, S; torch.cuda.empty_cache()
    # per-user significance: bag vs single-seed GUME
    gS = E.combine(E.load_scores([sf[0]]), [1.0])
    pm, pg = peruser(bagS, train, test, nu, ni), peruser(gS, train, test, nu, ni)
    del bagS, gS; torch.cuda.empty_cache()
    sig, se3 = {}, {}
    for m in MET:
        t, p2 = stats.ttest_rel(pm[m], pg[m]); p = p2 / 2 if t > 0 else 1 - p2 / 2
        sig[m] = {"p": float(p), "sig": bool(p < 0.05)}
        se3[m] = round(perseed[ds]["metrics"][m]["3sigma"] / math.sqrt(len(sd)), 5)  # 3*SE = 3sigma/sqrt(K)
    out[ds] = {"K": len(sd), "bag": bag, "ablation": abl, "sig": sig, "three_SE": se3,
               "improv_vs_gume": {m: round(100 * (bag[m] - perseed[ds]["metrics"][m]["gume"]) /
                                          perseed[ds]["metrics"][m]["gume"], 1) for m in MET}}
    print(f"=== {ds} (K={len(sd)}) ===")
    for m in MET:
        g = perseed[ds]["metrics"][m]["gume"]; b = bag[m]; e = se3[m]
        print(f"  {m:10s} bag={b:.4f} +/-{e:.4f}  band=[{b-e:.4f},{b+e:.4f}] GUME={g:.4f} "
              f"{'CLEARS' if b-e>g else 'OVERLAP'} p={sig[m]['p']:.1e} improv={out[ds]['improv_vs_gume'][m]:+.1f}%")
    print("  ablation R@20:", {k: v["Recall@20"] for k, v in abl.items()})

Path("/workspace/MechInterp/results/bai/finalize.json").write_text(json.dumps(out, indent=2))
print("saved results/bai/finalize.json")
