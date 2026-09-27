"""Put the prior audits' FREEDOM single-run deltas on our converged seed-noise scale (read-only).

Reads results/phase_holdout/freedom_p100*_runs.json (8 converged seeds per arm) and, for the
top-10 metrics the audits report, computes our mean, level floor (2*SD of full arm, % of mean),
paired floor (2*SD of per-seed differences, % of mean) and our mean delta, then divides each
audit's reported delta by our floors. Output: results/phase_novelty/prior_audit_floor_scale.json
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

R = Path("/workspace/MechInterp/results")
FILES = {"baby": "freedom_p100_runs.json", "sports": "freedom_p100_sports_runs.json",
         "clothing": "freedom_p100_clothing_runs.json"}
# Reported FREEDOM relative deltas (percent) -- sources in results/phase_novelty/prior_audits.json
AUDIT = {
    ("ye", "baby", "no_image"): {"NDCG@10": -0.9, "Recall@10": -0.3, "Precision@10": 0.0},
    ("ye", "sports", "no_image"): {"NDCG@10": -6.0, "Recall@10": -5.0, "Precision@10": -4.9},
    ("ye", "clothing", "no_image"): {"NDCG@10": 0.0, "Recall@10": 0.7, "Precision@10": 0.0},
    ("ye", "baby", "no_text"): {"NDCG@10": -28.0, "Recall@10": -27.7, "Precision@10": -26.9},
    ("ye", "sports", "no_text"): {"NDCG@10": -30.0, "Recall@10": -29.3, "Precision@10": -28.4},
    ("ye", "clothing", "no_text"): {"NDCG@10": -53.8, "Recall@10": -52.8, "Precision@10": -53.2},
    ("zhou", "baby", "no_image"): {"Recall@10": 100 * (0.0622 / 0.0627 - 1)},
    ("zhou", "baby", "no_text"): {"Recall@10": 100 * (0.0501 / 0.0627 - 1)},
}
METRICS = ["Recall@10", "NDCG@10", "Precision@10", "Recall@20"]


def main():
    out = {"script": "scripts/novelty_prior_audit_floors.py", "ours": {}, "audit_on_our_scale": []}
    for ds, fn in FILES.items():
        runs = json.loads((R / "phase_holdout" / fn).read_text())
        arms = {}
        for r in runs:
            arms.setdefault(r["condition"], {})[r["seed"]] = r["test_result"]
        full = arms["full"]
        out["ours"][ds] = {}
        for m in METRICS:
            fv = [full[s][m] for s in sorted(full)]
            mu = st.mean(fv)
            row = {"n_full": len(fv), "full_mean": mu, "F_level_pct": 100 * 2 * st.stdev(fv) / mu}
            for c in ("no_image", "no_text"):
                sh = sorted(set(full) & set(arms[c]))
                d = [arms[c][s][m] - full[s][m] for s in sh]
                row[c] = {"n": len(d), "delta_pct": 100 * st.mean(d) / mu,
                          "F_paired_pct": 100 * 2 * st.stdev(d) / mu}
            out["ours"][ds][m] = row
    for (who, ds, c), mets in AUDIT.items():
        for m, v in mets.items():
            o = out["ours"][ds][m]
            out["audit_on_our_scale"].append({
                "audit": who, "dataset": ds, "condition": c, "metric": m, "audit_delta_pct": round(v, 2),
                "our_delta_pct": round(o[c]["delta_pct"], 2),
                "audit_over_our_F_level": round(abs(v) / o["F_level_pct"], 2),
                "audit_over_our_F_paired": round(abs(v) / o[c]["F_paired_pct"], 2)})
    p = R / "phase_novelty" / "prior_audit_floor_scale.json"
    p.write_text(json.dumps(out, indent=2))
    for r in out["audit_on_our_scale"]:
        print(r)
    print("wrote", p)


if __name__ == "__main__":
    main()
