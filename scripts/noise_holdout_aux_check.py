"""Aux checks for exp_noise_holdout: (1) directed top-10 in-degree hubness of Ye noise under the
derived per-run seeds vs Ye's constant seed 42 (Baby, CPU, harness convention = no self-loop);
(2) keepw no-op analysis record; (3) cost estimate from existing converged runs' train_min."""
import json, statistics, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path("/workspace/MechInterp"); sys.path.insert(0, str(ROOT / "scripts"))
import exp_noise_holdout as enh
out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "script": "scripts/noise_holdout_aux_check.py"}
n = int(np.load("/workspace/Recsys/data/baby/image_feat.npy", mmap_mode="r").shape[0])
def hub(x, k=10):
    x = torch.nn.functional.normalize(torch.from_numpy(x).float(), dim=-1)
    indeg = torch.zeros(x.shape[0])
    for s in range(0, x.shape[0], 1024):
        sims = x[s:s+1024] @ x.t(); sims[torch.arange(sims.shape[0]), torch.arange(s, s+sims.shape[0])] = -float("inf")
        indeg += torch.bincount(sims.topk(k, dim=-1).indices.reshape(-1), minlength=x.shape[0]).float()
    v = np.sort(indeg.numpy()); m = len(v)
    return {"max_indegree": int(v.max()), "gini_indegree": float((2*np.arange(1, m+1)-m-1).dot(v)/(m*v.sum())),
            "frac_indegree_ge_50": float((v >= 50).mean())}
h = {}
for label, ns in [("ye_seed42", 42)] + [(f"derived_s{s}", enh.derive_noise_seed(s, "derived")) for s in (2024, 2025, 2026)]:
    txt, img = enh.ye_noise(n, ns, 384, 4096)
    h[label] = {"noise_seed": ns, "image_4096": hub(img), "text_384": hub(txt)}
out["baby_directed_hubness"] = h
out["keepw_noop_analysis"] = {
    "verdict": "NOT a no-op; implemented",
    "evidence_code": [
        "graph_utils.build_knn_graph: binary symmetrized kNN, D^-1/2 A D^-1/2 per modality",
        "freedom.py:87-93: mixed = w*img + (1-w)*txt (coalesce sums duplicates); no renormalization after mixing",
        "freedom.py:94-96: single-modality branch returns the normalized graph at weight 1",
        "freedom.py:142-155: h = mm_adj @ item_id_embedding (n_mm_layers=1); item repr = LightGCN-mean(E) + h, the same E feeds both terms, so a scalar on mm_adj changes the function and cannot be absorbed by E",
    ],
    "evidence_smoke": "results/phase_noise/_smoke/smoke_check.json checks.keepw_vs_removal: same init as removal, epoch losses differ from epoch 0",
    "scales": {"no_image_keepw": "text graph x 0.9", "no_text_keepw": "image graph x 0.1"},
}
cost = {}
for ds, f in enh.REF_FILES.items():
    r = [x for x in json.loads((ROOT/"results/phase_holdout"/f).read_text()) if "test_result" in x and x.get("stopping_step")==100]
    per = {c: statistics.mean(x["train_min"] for x in r if x["condition"]==c) for c in ("full","no_image","no_text")}
    est = {"noise_image": per["full"], "noise_text": per["full"], "no_image_keepw": per["no_image"], "no_text_keepw": per["no_text"]}
    cost[ds] = {"existing_mean_train_min": per, "n_existing_runs": len(r),
                "est_min_per_run_proxy": est, "est_hours_8_seeds": {c: 8*m/60 for c, m in est.items()},
                "est_hours_all4x8": sum(8*m for m in est.values())/60}
cost["note"] = "proxy: noise_* ~ full (same architecture); keepw ~ plain removal. train_min excludes data loading/graph build/test eval; measured while other jobs may have shared the GPU."
out["cost_estimate"] = cost
(ROOT/"results/phase_noise/aux_check.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out["baby_directed_hubness"], indent=1)); print(json.dumps(cost, indent=1))
