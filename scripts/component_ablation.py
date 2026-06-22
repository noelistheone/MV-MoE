"""Incremental component ablation for MV-MoE (validation-tuned weights, TEST split).

Each part added on top of GUME is given its own validation-tuned weight (greedy coordinate
ascent on VALID Recall@20, grid {0,.25,.5,.75,1,1.5,2}); we then report TEST. This is the SAME
protocol as the headline model, so a useless view simply receives weight ~0 and the metric stays
flat -- its honest marginal value -- instead of the artifact you get by forcing weight 1 on a weak
standalone view. Sequence: GUME(single) -> +seed-bagging -> +user-kNN -> +image-i2i -> +text-i2i(full).
Persists to results/bai/component_ablation.json.
"""
import json
from pathlib import Path

import ensemble_eval as E  # validated evaluator (load_split/load_scores/combine/evaluate)

GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def val_tuned_test(mats, train, valid, test, nu, ni):
    """Greedy coordinate-ascent weight search on VALID R@20; return TEST metrics + weights."""
    w = [1.0] * len(mats)
    for _ in range(2):
        for j in range(len(mats)):
            best_g, best_v = w[j], -1.0
            for g in GRID:
                w[j] = g
                v = E.evaluate(E.combine(mats, w), train, valid, nu, ni, ks=[20])["Recall@20"]
                if v > best_v:
                    best_v, best_g = v, g
            w[j] = best_g
    tr = E.evaluate(E.combine(mats, w), train, test, nu, ni)
    return tr, w


DATASETS = {
    "baby":     [f"baby_bprdump_s{s}.npy" for s in (2024, 2025, 2026, 2027, 2028)],
    "sports":   [f"sports_bprdump_s{s}.npy" for s in (2024, 2025, 2026)],
    "clothing": [f"clothing_bprdump_s{s}.npy" for s in (2024, 2025, 2026)],
}

out = {}
for ds, seed_files in DATASETS.items():
    E.DATA = Path(f"/workspace/Recsys/data/{ds}/{ds}.inter")
    E._CACHE.clear()
    train, valid, test, nu, ni = E.load_split()
    uk, img, txt = f"{ds}_userknn.npy", f"{ds}_bai2i_sig_img_learned_k20.npy", f"{ds}_bai2i_sig_text_learned_k20.npy"

    rungs = {
        "GUME (single)":     [seed_files[0]],
        "+ seed bagging":    seed_files,
        "+ user-kNN":        seed_files + [uk],
        "+ image-i2i":       seed_files + [uk, img],
        "+ text-i2i (full)": seed_files + [uk, img, txt],
    }

    ds_res, prev = {}, None
    for name, files in rungs.items():
        tr, w = val_tuned_test(E.load_scores(files), train, valid, test, nu, ni)
        r20, n20 = round(tr["Recall@20"], 5), round(tr["NDCG@20"], 5)
        dr = None if prev is None else round(100 * (r20 - prev[0]) / prev[0], 2)
        dn = None if prev is None else round(100 * (n20 - prev[1]) / prev[1], 2)
        ds_res[name] = {"R@20": r20, "N@20": n20, "dR%_vs_prev": dr, "dN%_vs_prev": dn,
                        "weights": [round(x, 2) for x in w]}
        print(f"[{ds:8s}] {name:20s} R@20={r20:.5f} N@20={n20:.5f}"
              + ("" if dr is None else f"  (ΔR {dr:+.2f}% ΔN {dn:+.2f}%)  w={ds_res[name]['weights']}"))
        prev = (r20, n20)
    g, f = ds_res["GUME (single)"], ds_res["+ text-i2i (full)"]
    ds_res["_total_vs_GUME_single"] = {"dR%": round(100 * (f["R@20"] - g["R@20"]) / g["R@20"], 2),
                                       "dN%": round(100 * (f["N@20"] - g["N@20"]) / g["N@20"], 2)}
    out[ds] = ds_res
    print()

dst = Path("/workspace/MechInterp/results/bai/component_ablation.json")
dst.write_text(json.dumps(out, indent=2))
print("saved", dst)
