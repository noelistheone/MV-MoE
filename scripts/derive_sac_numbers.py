"""Derived numbers quoted in the SAC 2027 draft that no other script stores.

Reads only finished artifacts; writes results/phase_paper/derived_for_paper.json.
 (1) Ye et al. (IJMIR 2026, Tables 5 and 8) FREEDOM/Sports baseline and image-noise knockout at
     @10 against our eight converged full runs and our eight noise_image runs.
 (2) Baby keep-weight shares and the (unregistered) paired contrast no_image_keepw - no_image.
 (3) Baby image-only models (retrained without text) vs one converged LightGCN seed.
 (4) Clothing keep-weight (registered N3, text only) and Clothing image-only models vs one converged
     LightGCN seed, computed exactly as (2) and (3).
Pooling follows scripts/holdout_verdicts.py: records filtered on stopping_step 100 / epochs_cap 3000.
"""
from __future__ import annotations
import json, statistics
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
YE = {  # International Journal of Multimedia Information Retrieval 15:14 (2026), FREEDOM, Sports
    "source": "Ye et al., IJMIR 15(2):14, 2026, Table 5 (baseline) and Table 8 (image knockout with noise)",
    "baseline": {"Recall@10": 0.0735, "NDCG@10": 0.0403},
    "image_noise": {"Recall@10": 0.0698, "NDCG@10": 0.0379},
}


def runs(path, ds):
    out = {}
    for r in json.loads(Path(path).read_text()):
        if r.get("dataset") != ds or "test_result" not in r:
            continue
        if r.get("stopping_step") != 100 or r.get("epochs_cap") != 3000:
            continue
        out.setdefault(r["condition"], {})[r["seed"]] = r["test_result"]
    return out


def main() -> int:
    out = {"_doc": __doc__}
    # (1) Ye vs ours, Sports
    sp = runs(R / "phase_holdout/freedom_p100_sports_runs.json", "sports")
    nz = runs(R / "phase_noise/runs_sports.json", "sports")
    ye = {"ye": YE, "ours": {}}
    for m in ("Recall@10", "NDCG@10"):
        full = [v[m] for v in sp["full"].values()]
        mu, sd = statistics.mean(full), statistics.stdev(full)
        noise = [nz["noise_image"][s][m] for s in sorted(nz["noise_image"])]
        per_seed = [100 * (nz["noise_image"][s][m] - sp["full"][s][m]) / sp["full"][s][m]
                    for s in sorted(nz["noise_image"]) if s in sp["full"]]
        ye["ours"][m] = {
            "full_mean": mu, "full_sd": sd, "n": len(full),
            "ye_baseline_minus_our_mean_pct": 100 * (YE["baseline"][m] - mu) / mu,
            "ye_baseline_z": (YE["baseline"][m] - mu) / sd,
            "ye_noise_minus_our_noise_mean_pct": 100 * (YE["image_noise"][m] - statistics.mean(noise)) / statistics.mean(noise),
            "ye_noise_change_pct": 100 * (YE["image_noise"][m] - YE["baseline"][m]) / YE["baseline"][m],
            "our_noise_change_pct_ratio_of_means": 100 * (statistics.mean(noise) - mu) / mu,
            "our_noise_per_seed_change_pct_min": min(per_seed), "our_noise_per_seed_change_pct_max": max(per_seed),
        }
    out["ye_sports"] = ye
    # (2) Baby keep-weight
    bb = runs(R / "phase_holdout/freedom_p100_runs.json", "baby")
    bn = runs(R / "phase_noise/runs_baby.json", "baby")
    full = bb["full"]; mfull = statistics.mean(v["Recall@20"] for v in full.values())
    def rel(arm):
        return 100 * (statistics.mean(v["Recall@20"] for v in arm.values()) - mfull) / mfull
    kw = {c: rel(bn[c]) for c in ("no_image_keepw", "no_text_keepw", "noise_image", "noise_text")}
    kw.update(no_image=rel(bb["no_image"]), no_text=rel(bb["no_text"]))
    kw["share_image_cost_removed_by_keepw"] = (kw["no_image"] - kw["no_image_keepw"]) / kw["no_image"]
    kw["share_text_cost_removed_by_keepw"] = (kw["no_text"] - kw["no_text_keepw"]) / kw["no_text"]
    seeds = sorted(set(bn["no_image_keepw"]) & set(bb["no_image"]))
    d = [bn["no_image_keepw"][s]["Recall@20"] - bb["no_image"][s]["Recall@20"] for s in seeds]
    fl = 2 * statistics.stdev(v["Recall@20"] for v in full.values()); fp = 2 * statistics.stdev(d)
    m = statistics.mean(d)
    kw["contrast_no_image_keepw_minus_no_image"] = {
        "registered": False, "n": len(d), "mean": m, "pct_of_full": 100 * m / mfull,
        "x_F_level": abs(m) / fl, "x_F_paired": abs(m) / fp, "p_two_sided": float(stats.ttest_1samp(d, 0).pvalue),
        "n_positive": sum(x > 0 for x in d),
        "verdict": "SIGNIFICANT" if abs(m) > max(fl, fp) else "BELOW" if abs(m) < min(fl, fp) else "MARGINAL"}
    out["baby_keepweight"] = kw
    # (3) Baby image-only vs LightGCN (one converged seed)
    lg = [r for r in json.loads((R / "phase_convergence/table_rerun.json").read_text())
          if r["model"] == "lightgcn" and r["dataset"] == "baby"][0]["test_result"]["Recall@20"]
    io = {}
    for name, arm in (("image_only_keepw_0.1", bn["no_text_keepw"]), ("image_only_weight1", bb["no_text"])):
        vals = [v["Recall@20"] for v in arm.values()]
        io[name] = {"mean": statistics.mean(vals), "pct_vs_lightgcn": 100 * (statistics.mean(vals) - lg) / lg,
                    "n_above_lightgcn": sum(x > lg for x in vals), "n": len(vals)}
    out["baby_image_only_vs_lightgcn"] = {"lightgcn_s2024_R20": lg, **io}
    # (4) Clothing keep-weight (text) and image-only vs LightGCN
    cb = runs(R / "phase_holdout/freedom_p100_clothing_runs.json", "clothing")
    cn = runs(R / "phase_noise/runs_clothing.json", "clothing")
    cfull = cb["full"]; cmfull = statistics.mean(v["Recall@20"] for v in cfull.values())
    crel = lambda arm: 100 * (statistics.mean(v["Recall@20"] for v in arm.values()) - cmfull) / cmfull
    ck = {"no_text": crel(cb["no_text"]), "no_text_keepw": crel(cn["no_text_keepw"])}
    seeds = sorted(set(cn["no_text_keepw"]) & set(cb["no_text"]))
    d = [cn["no_text_keepw"][s]["Recall@20"] - cb["no_text"][s]["Recall@20"] for s in seeds]
    fl = 2 * statistics.stdev(v["Recall@20"] for v in cfull.values()); fp = 2 * statistics.stdev(d)
    m = statistics.mean(d)
    ck["contrast_no_text_keepw_minus_no_text"] = {
        "registered": "N3 (PREREG_NOISE.md), predicted > 0", "n": len(d), "mean": m, "pct_of_full": 100 * m / cmfull,
        "x_F_level": abs(m) / fl, "x_F_paired": abs(m) / fp, "p_two_sided": float(stats.ttest_1samp(d, 0).pvalue),
        "n_negative": sum(x < 0 for x in d),
        "verdict": "SIGNIFICANT" if abs(m) > max(fl, fp) else "BELOW" if abs(m) < min(fl, fp) else "MARGINAL"}
    lgc = [r for r in json.loads((R / "phase_convergence/table_rerun.json").read_text())
           if r["model"] == "lightgcn" and r["dataset"] == "clothing"][0]["test_result"]["Recall@20"]
    ioc = {}
    for name, arm in (("image_only_keepw_0.1", cn["no_text_keepw"]), ("image_only_weight1", cb["no_text"])):
        vals = [v["Recall@20"] for v in arm.values()]
        ioc[name] = {"mean": statistics.mean(vals), "pct_vs_lightgcn": 100 * (statistics.mean(vals) - lgc) / lgc,
                     "n_above_lightgcn": sum(x > lgc for x in vals), "n": len(vals)}
    ck["image_only_vs_lightgcn"] = {"lightgcn_s2024_R20": lgc, **ioc}
    out["clothing_keepweight"] = ck
    (R / "phase_paper").mkdir(exist_ok=True)
    (R / "phase_paper/derived_for_paper.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "_doc"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
