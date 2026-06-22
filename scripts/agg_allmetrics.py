"""Compare a config's multi-seed runs against the GUME baseline on ALL metrics with paired
(same-seed) deltas, %-change, and a paired t-stat. Usage: python agg_allmetrics.py <tag_suffix> [tag2 ...]
Reads results/bai/baby_baseline_s<seed>.json and results/bai/baby_<variant>_s<seed>_<tag>.json."""
import glob
import json
import math
import sys
from pathlib import Path

OUT = Path("/workspace/MechInterp/results/bai")
SEEDS = [2024, 2025, 2026, 2027, 2028]
METRICS = ["Recall@5", "Recall@10", "Recall@20", "Recall@50",
           "NDCG@5", "NDCG@10", "NDCG@20", "NDCG@50"]


def load_baseline():
    out = {}
    for s in SEEDS:
        p = OUT / f"baby_baseline_s{s}.json"
        if p.exists():
            out[s] = json.loads(p.read_text())["test_result"]
    return out


def load_tag(tag):
    out = {}
    for s in SEEDS:
        hits = glob.glob(str(OUT / f"baby_*_s{s}_{tag}.json"))
        hits = [h for h in hits if "baseline" not in h]
        if hits:
            out[s] = json.loads(Path(hits[0]).read_text())["test_result"]
    return out


def paired_t(diffs):
    n = len(diffs)
    if n < 2:
        return float("nan")
    m = sum(diffs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1))
    return (m / (sd / math.sqrt(n))) if sd > 0 else float("inf")


def main():
    tags = sys.argv[1:]
    base = load_baseline()
    if not base:
        print("no baseline runs found"); return
    print(f"baseline seeds: {sorted(base)}  (n={len(base)})\n")
    # baseline means + MDE per metric
    print(f"{'metric':12s} {'GUME':>9s} {'+2%':>9s} {'+5%':>9s}")
    bmean = {}
    for m in METRICS:
        vals = [base[s][m] for s in base]
        bmean[m] = sum(vals) / len(vals)
        print(f"{m:12s} {bmean[m]:9.5f} {bmean[m]*1.02:9.5f} {bmean[m]*1.05:9.5f}")
    print()
    for tag in tags:
        d = load_tag(tag)
        common = [s for s in SEEDS if s in d and s in base]
        if not common:
            print(f"[{tag}] no runs found\n"); continue
        print(f"=== {tag}  (n={len(common)} seeds: {common}) ===")
        print(f"{'metric':12s} {'mean':>9s} {'dvsGUME':>9s} {'%':>7s} {'t':>6s} {'beat2%?':>8s}")
        allbeat2 = True
        for m in METRICS:
            vals = [d[s][m] for s in common]
            mean = sum(vals) / len(vals)
            diffs = [d[s][m] - base[s][m] for s in common]
            dm = sum(diffs) / len(diffs)
            pct = 100 * dm / bmean[m]
            t = paired_t(diffs)
            beat2 = (mean >= bmean[m] * 1.02)
            allbeat2 = allbeat2 and beat2
            print(f"{m:12s} {mean:9.5f} {dm:+9.5f} {pct:+6.1f}% {t:6.2f} {'YES' if beat2 else 'no':>8s}")
        print(f"  --> beats GUME by >=2% on ALL metrics: {'YES' if allbeat2 else 'NO'}\n")


if __name__ == "__main__":
    main()
