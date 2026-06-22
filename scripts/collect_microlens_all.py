"""Collect MicroLens baseline test metrics for all 13 models from BOTH the concurrent Recsys runs
(/workspace/Recsys/logs) and my MechInterp runs (/workspace/MechInterp/logs_microlens),
keeping the best-validation run per model. Writes results/bai/microlens_baselines.json."""
import glob, json
from pathlib import Path

MET = ["Recall@10", "Recall@20", "NDCG@10", "NDCG@20", "MAP@10", "MAP@20"]
MODELS = ["lightgcn", "vbpr", "mmgcn", "lattice", "bm3", "mgcn", "mentor",
          "freedom", "lgmrec", "damrs", "smore", "cohesion", "gume"]
best = {}
for root in ("/workspace/Recsys/logs", "/workspace/MechInterp/logs_microlens"):
    for rj in glob.glob(f"{root}/*_microlens_*/result.json"):
        try:
            d = json.load(open(rj))
        except Exception:
            continue
        m = d.get("model"); tr = d.get("test_result", {})
        if m not in MODELS or "MAP@20" not in tr:
            continue
        vs = d.get("best_valid_score", 0.0) or 0.0
        if m not in best or vs > best[m]["_v"]:
            best[m] = {"_v": vs, **{k: round(float(tr[k]), 5) for k in MET if k in tr}}
out = {m: {k: v for k, v in best[m].items() if k != "_v"} for m in best}
Path("/workspace/MechInterp/results/bai/microlens_baselines.json").write_text(json.dumps(out, indent=2))
print(f"collected {len(out)}/13:", sorted(out))
miss = [m for m in MODELS if m not in out]
print("MISSING:", miss if miss else "none")
