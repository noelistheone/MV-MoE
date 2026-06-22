"""MV-MoE reported as mean +/- 3 sigma over multiple seeds.
Each run = one independently-seeded GUME backbone + (deterministic) user-kNN, img-i2i, txt-i2i,
fused with validation-tuned weights (greedy search on VALID R@20). We report mean and 3*std over
the K seeds for every metric, and flag whether mean - 3sigma exceeds the best baseline (GUME).
Also reports the K-seed bagged operating point. Persists results/bai/mvmoe_seeds.json.
"""
import glob
import json
import re
from pathlib import Path
import numpy as np
import ensemble_eval as E

GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
MET = ["Recall@10", "Recall@20", "NDCG@10", "NDCG@20", "MAP@10", "MAP@20"]
SDIR = Path("/workspace/MechInterp/results/bai/scores")

# best-baseline (GUME) numbers from the comparison table, to test the 3-sigma band against
import json as _j
cross = _j.load(open("/workspace/MechInterp/results/phasex_crossarch/crossarch_knockout.json"))
GUME = {r["dataset"]: r["baseline"] for r in cross if r["model"] == "gume"}


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
    return E.evaluate(E.combine(mats, w), train, test, nu, ni), w


out = {}
for ds in ["baby", "sports", "clothing"]:
    E.DATA = Path(f"/workspace/Recsys/data/{ds}/{ds}.inter")
    E._CACHE.clear()
    train, valid, test, nu, ni = E.load_split()
    seeds = sorted(int(re.search(r"_s(\d+)\.npy", p).group(1))
                   for p in glob.glob(str(SDIR / f"{ds}_bprdump_s*.npy")))
    views_det = [f"{ds}_userknn.npy", f"{ds}_bai2i_sig_img_learned_k20.npy", f"{ds}_bai2i_sig_text_learned_k20.npy"]
    perseed = {m: [] for m in MET}
    for s in seeds:
        tr, _ = val_tune(E.load_scores([f"{ds}_bprdump_s{s}.npy"] + views_det), train, valid, test, nu, ni)
        for m in MET:
            perseed[m].append(tr[m])
    # bagged operating point (all seeds)
    bag_tr, bag_w = val_tune(E.load_scores([f"{ds}_bprdump_s{s}.npy" for s in seeds] + views_det),
                             train, valid, test, nu, ni)
    ds_res = {"n_seeds": len(seeds), "seeds": seeds, "metrics": {}}
    print(f"=== {ds}  (K={len(seeds)} seeds: {seeds}) ===")
    for m in MET:
        a = np.array(perseed[m])
        mean, sd = float(a.mean()), float(a.std(ddof=1))
        three = 3 * sd
        gb = GUME[ds][m]
        clears = (mean - three) > gb
        ds_res["metrics"][m] = {"mean": round(mean, 5), "std": round(sd, 5), "3sigma": round(three, 5),
                                "per_seed": [round(x, 5) for x in a.tolist()],
                                "bag": round(bag_tr[m], 5), "gume": round(gb, 5),
                                "mean_minus_3sig_gt_gume": bool(clears)}
        print(f"  {m:10s} mean={mean:.5f} 3sig={three:.5f}  band=[{mean-three:.5f},{mean+three:.5f}]  "
              f"GUME={gb:.5f} {'CLEARS' if clears else 'overlaps'}  bag={bag_tr[m]:.5f}")
    out[ds] = ds_res

Path("/workspace/MechInterp/results/bai/mvmoe_seeds.json").write_text(json.dumps(out, indent=2))
print("saved results/bai/mvmoe_seeds.json")
