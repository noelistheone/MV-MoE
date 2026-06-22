"""Aggregate LOO user-image-profile channel results vs baseline. Paired (same-seed) diffs,
t-stat, and comparison to MDE = 2*std(baseline). Reads results/bai/*.json. No fabrication --
only reports seeds that actually have a saved artifact."""
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

OUT = Path("/workspace/MechInterp/results/bai")
SEEDS = [2024, 2025, 2026, 2027, 2028]


def load(tag_suffix):
    """return {seed: (R@20, N@20, valid)} for files baby_bai_s<seed>_<tag_suffix>.json
    or baby_baseline_s<seed>.json when tag_suffix=='baseline'."""
    out = {}
    for s in SEEDS:
        if tag_suffix == "baseline":
            p = OUT / f"baby_baseline_s{s}.json"
        else:
            p = OUT / f"baby_bai_s{s}_{tag_suffix}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        tr = d["test_result"]
        out[s] = (tr["Recall@20"], tr["NDCG@20"], d.get("best_valid_score"))
    return out


def paired_t(diffs):
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    m = sum(diffs) / n
    var = sum((x - m) ** 2 for x in diffs) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    t = m / se if se > 0 else float("nan")
    return m, t


def main():
    base = load("baseline")
    bvals = [base[s][0] for s in SEEDS if s in base]
    bmean = sum(bvals) / len(bvals)
    bstd = (sum((x - bmean) ** 2 for x in bvals) / (len(bvals) - 1)) ** 0.5
    mde = 2 * bstd
    print(f"baseline R@20: n={len(bvals)} mean={bmean:.5f} std={bstd:.5f} MDE(2std)={mde:.5f}")
    print(f"baseline seeds: {[(s, round(base[s][0],5)) for s in SEEDS if s in base]}\n")

    # discover all bai tag suffixes present
    tags = set()
    for f in glob.glob(str(OUT / "baby_bai_s*_*.json")):
        name = Path(f).stem  # baby_bai_s2024_<tag>
        parts = name.split("_", 3)
        if len(parts) == 4:
            tags.add(parts[3])
    print(f"{'config':28s} {'n':>2s} {'R@20 mean':>10s} {'dR@20':>9s} {'t':>6s} {'>MDE?':>6s} {'N@20':>9s} {'valid':>8s}")
    for tag in sorted(tags):
        d = load(tag)
        common = [s for s in SEEDS if s in d and s in base]
        if not common:
            continue
        rv = [d[s][0] for s in common]
        nv = [d[s][1] for s in common]
        vv = [d[s][2] for s in common if d[s][2] is not None]
        rmean = sum(rv) / len(rv)
        diffs = [d[s][0] - base[s][0] for s in common]
        dm, t = paired_t(diffs)
        nmean = sum(nv) / len(nv)
        vmean = sum(vv) / len(vv) if vv else float("nan")
        flag = "YES" if dm > mde else ("~" if dm > 0 else "no")
        print(f"{tag:28s} {len(common):2d} {rmean:10.5f} {dm:+9.5f} {t:6.2f} {flag:>6s} "
              f"{nmean:9.5f} {vmean:8.5f}")


if __name__ == "__main__":
    main()
