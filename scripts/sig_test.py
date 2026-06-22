"""Per-user paired significance test: MV-MoE vs the strongest baseline (a GUME seed).
Standard recsys significance protocol: per-user metric vectors, one-sided paired t-test
(MV-MoE > GUME). Reports p-values for R@10/20, N@10/20, MAP@10/20 per dataset; a cell is
marked '*' in the table if p<0.05. Persists results/bai/sig_test.json.
"""
import json
from pathlib import Path
import numpy as np
import torch
from scipy import stats
import ensemble_eval as E

CFG = {
    "baby":     {"seeds": [2024, 2025, 2026, 2027, 2028], "w": [2.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.25, 0.0]},
    "sports":   {"seeds": [2024, 2025, 2026],             "w": [2.0, 2.0, 2.0, 1.5, 0.25, 0.0]},
    "clothing": {"seeds": [2024, 2025, 2026],             "w": [2.0, 2.0, 2.0, 1.0, 0.25, 0.25]},
}
KS = [10, 20]


def peruser(S, train, gt, nu, ni):
    """Per-user Recall@k, NDCG@k, MAP@k vectors (over users with >=1 test item)."""
    (tu, ti), REL, nrel = E._prep(train, gt, nu, ni)
    S = S.clone()
    S[tu, ti] = -1e9
    maxk = max(KS)
    topk = torch.topk(S, maxk, dim=1).indices
    hits = torch.gather(REL, 1, topk).float()
    disc = 1.0 / torch.log2(torch.arange(2, maxk + 2, device=E.dev).float())
    idcg_cum = torch.cumsum(disc, 0)
    keep = nrel > 0
    out = {}
    for k in KS:
        rec = (hits[:, :k].sum(1) / nrel.clamp(min=1))
        dcg = (hits[:, :k] * disc[:k]).sum(1)
        idcg = idcg_cum[(nrel.clamp(max=k).long() - 1).clamp(min=0)]
        ndcg = dcg / idcg
        cum = torch.cumsum(hits[:, :k], dim=1)
        ranks = torch.arange(1, k + 1, device=E.dev).float()
        ap = ((cum / ranks) * hits[:, :k]).sum(1) / nrel.clamp(max=k).clamp(min=1)
        out[f"Recall@{k}"] = rec[keep].cpu().numpy()
        out[f"NDCG@{k}"] = ndcg[keep].cpu().numpy()
        out[f"MAP@{k}"] = ap[keep].cpu().numpy()
    return out


res = {}
for ds, c in CFG.items():
    E.DATA = Path(f"/workspace/Recsys/data/{ds}/{ds}.inter")
    E._CACHE.clear()
    train, valid, test, nu, ni = E.load_split()
    seedfiles = [f"{ds}_bprdump_s{s}.npy" for s in c["seeds"]]
    views = seedfiles + [f"{ds}_userknn.npy", f"{ds}_bai2i_sig_img_learned_k20.npy", f"{ds}_bai2i_sig_text_learned_k20.npy"]
    mv = E.combine(E.load_scores(views), c["w"])
    gume = E.combine(E.load_scores([seedfiles[0]]), [1.0])   # single strongest baseline
    pm, pg = peruser(mv, train, test, nu, ni), peruser(gume, train, test, nu, ni)
    res[ds] = {}
    line = []
    for met in ["Recall@10", "Recall@20", "NDCG@10", "NDCG@20", "MAP@10", "MAP@20"]:
        a, b = pm[met], pg[met]
        t, p_two = stats.ttest_rel(a, b)
        p = p_two / 2 if t > 0 else 1 - p_two / 2   # one-sided MV-MoE > GUME
        res[ds][met] = {"p_onesided": float(p), "t": float(t), "sig_0.05": bool(p < 0.05)}
        line.append(f"{met}:p={p:.1e}{'*' if p < 0.05 else ''}")
    print(f"[{ds:8s}] vs GUME-seed | " + "  ".join(line))

Path("/workspace/MechInterp/results/bai/sig_test.json").write_text(json.dumps(res, indent=2))
print("saved results/bai/sig_test.json")
