"""Collect MicroLens baseline test metrics from logs_microlens/*/result.json into one JSON.
For each model keep the run with the best validation score (handles re-runs / the smoke run).
Writes results/bai/microlens_baselines.json: {model: {Recall@10,...,MAP@20}}.
"""
import glob, json
from pathlib import Path

MET = ["Recall@10", "Recall@20", "NDCG@10", "NDCG@20", "MAP@10", "MAP@20"]
best = {}
for rj in glob.glob("/workspace/MechInterp/logs_microlens/*/result.json"):
    try:
        d = json.load(open(rj))
    except Exception:
        continue
    m = d.get("model")
    tr = d.get("test_result", {})
    if not m or "Recall@20" not in tr:
        continue
    vs = d.get("best_valid_score", 0.0) or 0.0
    if m not in best or vs > best[m]["_valid"]:
        best[m] = {"_valid": vs, **{k: round(float(tr[k]), 5) for k in MET if k in tr}}

out = {m: {k: v for k, v in r.items() if k != "_valid"} for m, r in best.items()}
Path("/workspace/MechInterp/results/bai/microlens_baselines.json").write_text(json.dumps(out, indent=2))
for m in sorted(out):
    print(f"{m:10s} R@20={out[m].get('Recall@20')}  N@20={out[m].get('NDCG@20')}  MAP@20={out[m].get('MAP@20')}")
print(f"\n{len(out)} models -> results/bai/microlens_baselines.json")
