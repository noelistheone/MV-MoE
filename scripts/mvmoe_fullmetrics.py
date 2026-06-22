"""Compute MV-MoE full metrics (Recall/NDCG/Precision @10,@20) with the headline weights,
to populate the expanded comparison table. Persists to results/bai/mvmoe_fullmetrics.json.
Headline weights are from results/bai/logs_final_model.log (val-tuned)."""
import json
from pathlib import Path
import ensemble_eval as E

# view order: GUME seeds..., user-kNN, image-i2i, text-i2i ; weights from logs_final_model.log
CFG = {
    "baby": {
        "seeds": [2024, 2025, 2026, 2027, 2028],
        "w": [2.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.25, 0.0],
    },
    "sports": {
        "seeds": [2024, 2025, 2026],
        "w": [2.0, 2.0, 2.0, 1.5, 0.25, 0.0],
    },
    "clothing": {
        "seeds": [2024, 2025, 2026],
        "w": [2.0, 2.0, 2.0, 1.0, 0.25, 0.25],
    },
}

out = {}
for ds, c in CFG.items():
    E.DATA = Path(f"/workspace/Recsys/data/{ds}/{ds}.inter")
    E._CACHE.clear()
    train, valid, test, nu, ni = E.load_split()
    # sanity: single GUME seed dump MAP should be ~crossarch GUME MAP (validates MAP impl)
    g = E.evaluate(E.combine(E.load_scores([f"{ds}_bprdump_s{c['seeds'][0]}.npy"]), [1.0]),
                   train, test, nu, ni)
    print(f"[{ds:8s}] GUME-seed sanity: R@20={g['Recall@20']:.5f} MAP@10={g['MAP@10']:.5f} MAP@20={g['MAP@20']:.5f}")
    files = [f"{ds}_bprdump_s{s}.npy" for s in c["seeds"]] + [
        f"{ds}_userknn.npy", f"{ds}_bai2i_sig_img_learned_k20.npy", f"{ds}_bai2i_sig_text_learned_k20.npy"]
    assert len(files) == len(c["w"]), (ds, len(files), len(c["w"]))
    S = E.combine(E.load_scores(files), c["w"])
    tr = E.evaluate(S, train, test, nu, ni)
    out[ds] = {k: round(v, 5) for k, v in tr.items()}
    print(f"[{ds:8s}] MV-MoE " + " ".join(f"{k}={tr[k]:.5f}" for k in
          ["Recall@10", "NDCG@10", "Recall@20", "NDCG@20", "MAP@10", "MAP@20"]))

Path("/workspace/MechInterp/results/bai/mvmoe_fullmetrics.json").write_text(json.dumps(out, indent=2))
print("saved results/bai/mvmoe_fullmetrics.json")
