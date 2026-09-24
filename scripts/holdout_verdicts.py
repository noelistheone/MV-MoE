"""Adjudicate the converged train-time holdout (patience 100 / cap 3000) under the pre-registered
two-floor rule, for every dataset whose runs file is complete.

Previously `results/phase_holdout/final_verdicts_p100.json` was produced by an inline command and
covered only the three Amazon datasets. This script is the reproducible replacement: it recomputes
those three (and asserts they match the stored values to 1e-12) and adds any other dataset whose
full / no_image / no_text arms are all present on the same seeds.

Rule (PREREG_HOLDOUT.md): per seed s, d_s = R@20(cond, s) - R@20(full, s).
  F_paired = 2 * sd(d_s)            F_level = 2 * sd(R@20(full, s))
  SIGNIFICANT iff |mean d| > max(F_paired, F_level); NULL iff below both; MARGINAL otherwise.
Paired t-test on d_s (df = n - 1), two-sided.

Output -> results/phase_holdout/final_verdicts_p100.json (Amazon entries unchanged).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from scipy import stats

ROOT = Path("/workspace/MechInterp")
HOLD = ROOT / "results" / "phase_holdout"
OUT = HOLD / "final_verdicts_p100.json"


def cell(runs: list[dict], cond: str) -> dict | None:
    by = {}
    for r in runs:
        if "test_result" in r:
            by.setdefault(r["condition"], {})[r["seed"]] = r
    full, arm = by.get("full", {}), by.get(cond, {})
    seeds = sorted(set(full) & set(arm))
    if len(seeds) < 2:
        return None
    f = [full[s]["test_result"]["Recall@20"] for s in seeds]
    c = [arm[s]["test_result"]["Recall@20"] for s in seeds]
    d = [ci - fi for ci, fi in zip(c, f)]
    m = statistics.mean(d)
    fp, fl = 2 * statistics.stdev(d), 2 * statistics.stdev(f)
    t, p = stats.ttest_1samp(d, 0.0)
    return {
        "n": len(seeds), "full_mean": statistics.mean(f), "cond_mean": statistics.mean(c),
        "d": m, "d_pct": 100 * m / statistics.mean(f), "F_paired": fp, "F_level": fl,
        "x_Fp": abs(m) / fp, "x_Fl": abs(m) / fl, "t": float(t), "df": len(seeds) - 1,
        "p": float(p), "n_neg": sum(x < 0 for x in d),
        "verdict": ("SIGNIFICANT" if abs(m) > max(fp, fl) else
                    "NULL" if abs(m) < min(fp, fl) else "MARGINAL"),
        "seeds": seeds,
        "median_best_epoch": statistics.median(arm[s]["best_epoch"] for s in seeds),
    }


def main() -> int:
    old = json.loads(OUT.read_text()) if OUT.is_file() else {}
    new = {}
    # Baby's p100 runs live in freedom_p100_runs.json (no dataset suffix), so group by each
    # record's own `dataset` field, never by filename. A (dataset, condition, seed) key seen in
    # two files must carry the identical R@20, otherwise we refuse to guess which is right.
    pooled: dict[tuple, dict] = {}
    for f in sorted(HOLD.glob("freedom_p100*_runs.json")):
        for r in json.loads(f.read_text()):
            # the converged protocol is recorded as stopping_step / epochs_cap (no record carries a
            # `patience` key, so filtering on it would silently accept everything)
            if "test_result" not in r or r.get("stopping_step") != 100 or r.get("epochs_cap") != 3000:
                continue
            k = (r["dataset"], r["condition"], r["seed"])
            if k in pooled:
                a, b = pooled[k]["test_result"]["Recall@20"], r["test_result"]["Recall@20"]
                assert abs(a - b) < 1e-12, f"conflicting records for {k}: {a} vs {b} ({f.name})"
            pooled[k] = r
    for ds in sorted({k[0] for k in pooled}):
        runs = [r for k, r in pooled.items() if k[0] == ds]
        for cond in ("no_image", "no_text"):
            c = cell(runs, cond)
            if c is not None:
                new[f"{ds}/{cond}"] = c
    # the Amazon cells were already certified; a recomputation must reproduce them exactly
    for k, v in old.items():
        assert k in new, f"{k} vanished"
        for key in ("d", "F_paired", "F_level", "p"):
            assert abs(new[k][key] - v[key]) < 1e-12, (k, key, new[k][key], v[key])
        assert new[k]["verdict"] == v["verdict"], (k, new[k]["verdict"], v["verdict"])
    OUT.write_text(json.dumps(new, indent=1))
    for k, v in new.items():
        tag = "" if k in old else "   <-- NEW"
        print(f"{k:22s} n={v['n']} d={v['d_pct']:+.2f}% xFp={v['x_Fp']:.3f} xFl={v['x_Fl']:.3f} "
              f"p={v['p']:.4g} neg={v['n_neg']}/{v['n']} {v['verdict']}{tag}")
    print(f"\nWrote {OUT} ({len(old)} certified entries reproduced exactly, "
          f"{len(new) - len(old)} added)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
