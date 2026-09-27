"""Registered noise / keep-weight contrasts (results/phase_noise/PREREG_NOISE.md) -> verdicts.

Why this is not a thin wrapper around exp_noise_holdout.summarize(): that function (written to
refresh results/phase_noise/summary_<ds>.json after every run) was checked against the
registration and the paper's rule and falls short on four points:
  1. It reports no F_level (2 sd of the full arm's R@20), no two-floor verdict, no p-value and
     no sign count -- only mean, delta sd, 2*sd(d) and the t statistic.
  2. It reads one reference file per dataset by FILENAME (REF_FILES[ds]) and never checks the
     record's own `dataset` field; scripts/holdout_verdicts.py pools every
     results/phase_holdout/freedom_p100*_runs.json by the record's dataset field and refuses
     conflicting duplicates. (Today the two give the same records: every file holds one dataset.)
  3. Its a-vs-b relative change is 100*mean(d)/mean(b); for the registered noise-vs-removal and
     keepw-vs-removal contrasts the paper's Table 1 unit is percent of the FULL model's Recall@20,
     so we report both (d_pct_of_full = Table 1 unit; d_pct_of_ref = relative to the removal arm).
  4. It does not filter the new runs by protocol (stopping_step 100 / epochs_cap 3000) or dataset
     field, and silently keeps the last of duplicate (condition, seed) records.
So the contrasts are implemented here, directly and self-contained; the summary files are read
only as a cross-check (mean paired difference and 2*sd(d) must agree where both exist).

Rule (identical to scripts/holdout_verdicts.py / PREREG_HOLDOUT.md), metric Recall@20:
  per seed s: d_s = R@20(a, s) - R@20(b, s), seeds = seeds present in both arms (and in full)
  F_paired = 2 * sd(d_s)                          (the contrast's own per-seed differences)
  F_level  = 2 * sd(R@20(full, s)) over the full arm's 8 converged seeds (primary; this is the
             floor printed in Table 1). While a contrast is incomplete, the same floor over only
             the contrast's shared seeds is also stored (F_level_shared_seeds); with all 8 seeds
             the two are identical and equal to holdout_verdicts.cell()'s F_level.
  SIGNIFICANT iff |mean d| > max(F_paired, F_level); NULL iff below both; MARGINAL otherwise.
  Paired t-test on d_s, two-sided (scipy.stats.ttest_1samp), df = n - 1.
Registered directions: N1 noise_image - no_image < 0; N2 noise_text - no_text < 0;
N3 no_text_keepw - no_text > 0. `direction_as_predicted` is reported next to the verdict.

Also written: every new condition vs full (same units as Table 1, @20 and @10), our relative
costs at Recall@10 / NDCG@10 next to Ye et al.'s FREEDOM noise-knockout losses (parsed from
results/phase_novelty/prior_audits.json), and Pomo et al.'s derived image-only vs image+text
Recall@20 / nDCG@20 gaps next to our no_text and no_text_keepw costs.

Read-only on every input. Output: results/phase_noise/noise_verdicts.json
usage: python scripts/noise_verdicts.py [--noise-dir DIR] [--out PATH]
       (--noise-dir/--out exist only so the test fixture can be run outside results/)
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from scipy import stats

ROOT = Path("/workspace/MechInterp")
HOLD = ROOT / "results" / "phase_holdout"
NOISE = ROOT / "results" / "phase_noise"
PRIOR = ROOT / "results" / "phase_novelty" / "prior_audits.json"
CERTIFIED = HOLD / "final_verdicts_p100.json"
SEEDS = list(range(2024, 2032))
PRIMARY = "Recall@20"
METRICS = ("Recall@20", "NDCG@20", "Recall@10", "NDCG@10")
NEW_CONDS = ("noise_image", "noise_text", "no_image_keepw", "no_text_keepw")

REGISTERED = [  # (id, dataset, a, b, predicted sign of mean(a - b))
    ("N1", "baby", "noise_image", "no_image", -1),
    ("N1", "sports", "noise_image", "no_image", -1),
    ("N2", "baby", "noise_text", "no_text", -1),
    ("N2", "sports", "noise_text", "no_text", -1),
    ("N3", "baby", "no_text_keepw", "no_text", +1),
    ("N3", "clothing", "no_text_keepw", "no_text", +1),
]
PRED_TEXT = {"N1": "noise_image costs more than no_image (mean paired difference < 0)",
             "N2": "noise_text costs more than no_text (mean paired difference < 0)",
             "N3": "no_text_keepw costs less than no_text (mean paired difference > 0)"}


# ----------------------------------------------------------------------------- loading

def _converged(r: dict) -> bool:
    return "test_result" in r and r.get("stopping_step") == 100 and r.get("epochs_cap") == 3000


def _rel(f: Path) -> str:
    try:
        return str(f.relative_to(ROOT))
    except ValueError:
        return str(f)


def pool(files: list[Path], allowed_conditions=None) -> tuple[dict, list[str]]:
    """{(dataset, condition, seed): record}, grouped by each record's own dataset field.
    A key seen twice must carry the identical Recall@20, else we refuse to guess."""
    pooled, notes = {}, []
    for f in files:
        n_skip = 0
        for r in json.loads(f.read_text()):
            if not _converged(r):
                n_skip += 1
                continue
            if allowed_conditions is not None and r["condition"] not in allowed_conditions:
                n_skip += 1
                continue
            k = (r["dataset"], r["condition"], int(r["seed"]))
            if k in pooled:
                a, b = pooled[k]["test_result"][PRIMARY], r["test_result"][PRIMARY]
                assert abs(a - b) < 1e-12, f"conflicting records for {k}: {a} vs {b} ({f.name})"
                notes.append(f"duplicate identical record {k} in {f.name}")
            pooled[k] = r
        if n_skip:
            notes.append(f"{_rel(f)}: {n_skip} records skipped (no test_result, not "
                         f"patience-100/cap-3000, or non-registered label)")
    return pooled, notes


def arm(pooled: dict, ds: str, cond: str) -> dict:
    return {k[2]: r for k, r in pooled.items() if k[0] == ds and k[1] == cond}


# ----------------------------------------------------------------------------- statistics

def verdict(m: float, fp: float, fl: float) -> str:
    return "SIGNIFICANT" if abs(m) > max(fp, fl) else "NULL" if abs(m) < min(fp, fl) else "MARGINAL"


def contrast(a: dict, b: dict, full: dict, metric: str) -> dict | None:
    """Paired a - b by seed on `metric`; floors on the same metric."""
    seeds = sorted(set(a) & set(b) & set(full))
    if not seeds:
        return None
    d = [a[s]["test_result"][metric] - b[s]["test_result"][metric] for s in seeds]
    m = statistics.mean(d)
    full_all = [full[s]["test_result"][metric] for s in sorted(full)]
    full_sh = [full[s]["test_result"][metric] for s in seeds]
    mb = statistics.mean(b[s]["test_result"][metric] for s in seeds)
    mf_sh = statistics.mean(full_sh)
    e = {"n": len(seeds), "seeds": seeds, "missing_seeds": [s for s in SEEDS if s not in seeds],
         "complete": seeds == SEEDS,
         "mean_a": statistics.mean(a[s]["test_result"][metric] for s in seeds), "mean_b": mb,
         "full_mean_shared_seeds": mf_sh, "full_mean_all_seeds": statistics.mean(full_all),
         "n_full_seeds": len(full_all),
         "d": m, "d_pct_of_full": 100 * m / mf_sh, "d_pct_of_ref": 100 * m / mb,
         "n_neg": sum(x < 0 for x in d), "n_pos": sum(x > 0 for x in d),
         "per_seed_d": dict(zip(map(str, seeds), d))}
    if len(full_all) >= 2:
        e["F_level"] = 2 * statistics.stdev(full_all)
        e["F_level_pct_of_full"] = 100 * e["F_level"] / statistics.mean(full_all)
    if len(full_sh) >= 2:
        e["F_level_shared_seeds"] = 2 * statistics.stdev(full_sh)
    if len(d) >= 2:
        fp = 2 * statistics.stdev(d)
        e["F_paired"] = fp
        e["F_paired_pct_of_full"] = 100 * fp / mf_sh
        t, p = stats.ttest_1samp(d, 0.0)
        e.update({"t": float(t), "df": len(d) - 1, "p": float(p)})
        if "F_level" in e and fp > 0:
            e["x_Fp"] = abs(m) / fp
            e["x_Fl"] = abs(m) / e["F_level"]
            e["verdict"] = verdict(m, fp, e["F_level"])
            if not e["complete"]:
                e["verdict_status"] = f"PROVISIONAL ({len(d)}/8 seeds)"
    else:
        e["verdict_status"] = "insufficient seeds (need >= 2 for F_paired)"
    return e


def best_epochs(x: dict, seeds: list[int]) -> float | None:
    v = [x[s].get("best_epoch") for s in seeds if x[s].get("best_epoch") is not None]
    return statistics.median(v) if v else None


# ----------------------------------------------------------------------------- prior audits

_YE_RE = re.compile(r"(NDCG|R|P)@(\d+) ([0-9.]+) \(([+-]?[0-9.]+)%\)")
_YE_METRIC = {"NDCG": "NDCG", "R": "Recall", "P": "Precision"}


def parse_ye(prior: dict) -> dict:
    fr = prior["papers"]["modalitybenefit"]["freedom_numbers"]
    base = fr["baseline_table6"]
    out = {}
    for key, cond in (("image_knockout_noise_table9", "noise_image"),
                      ("text_knockout_noise_table8", "noise_text")):
        for dsname, s in fr[key].items():
            ds = dsname.lower()
            for m, k, val, pct in _YE_RE.findall(s):
                if k != "10" or m == "P":
                    continue
                met = f"{_YE_METRIC[m]}@{k}"
                b = base[dsname][f"{m if m == 'NDCG' else 'Recall'}@{k}"]
                out.setdefault(ds, {}).setdefault(cond, {})[met] = {
                    "reported_pct": float(pct), "baseline": b, "knockout": float(val),
                    "recomputed_pct_from_4dp": 100 * (float(val) / b - 1)}
    for ds in ("baby", "sports", "clothing"):
        for cond in ("noise_image", "noise_text"):
            assert set(out[ds][cond]) == {"Recall@10", "NDCG@10"}, (ds, cond, out[ds][cond])
    return out


def parse_pomo(prior: dict) -> dict:
    fr = prior["papers"]["pomo2025leverage"]["freedom_numbers"]
    t2 = fr["table2_top20_percent"]
    out = {}
    for ds in ("Baby", "Pets", "Clothing"):
        both, img = t2["RNet50-Sbert"][ds], t2["RNet50"][ds]
        out[ds.lower()] = {
            "RNet50_Sbert_R@20_pct": both[0], "RNet50_R@20_pct": img[0],
            "Recall@20_gap_pct": 100 * (img[0] / both[0] - 1),
            "RNet50_Sbert_nDCG@20_pct": both[1], "RNet50_nDCG@20_pct": img[1],
            "NDCG@20_gap_pct": 100 * (img[1] / both[1] - 1)}
    return {"per_dataset": out, "quoted_in_prior_audits": fr["derived_image_only_vs_image_plus_text_recall@20"]}


# ----------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise-dir", type=Path, default=NOISE)
    ap.add_argument("--out", type=Path, default=NOISE / "noise_verdicts.json")
    args = ap.parse_args()

    ref_files = sorted(HOLD.glob("freedom_p100*_runs.json"))
    new_files = sorted(args.noise_dir.glob("runs_*.json"))
    ref, ref_notes = pool(ref_files)
    new, new_notes = pool(new_files, allowed_conditions=set(NEW_CONDS))
    clash = {k[1] for k in new} & {k[1] for k in ref}
    assert not clash, f"new-run labels collide with reference labels: {clash}"
    allp = {**ref, **new}

    # sanity: our contrast() reproduces the certified Table 1 retraining cells exactly
    certified = json.loads(CERTIFIED.read_text())
    reproduced = {}
    for key, c in certified.items():
        ds, cond = key.split("/")
        full = arm(allp, ds, "full")
        e = contrast(arm(allp, ds, cond), full, full, PRIMARY)
        for fld, mine in (("d", e["d"]), ("d_pct", e["d_pct_of_full"]), ("F_paired", e["F_paired"]),
                          ("F_level", e["F_level"]), ("p", e["p"])):
            assert abs(mine - c[fld]) < 1e-12, (key, fld, mine, c[fld])
        assert e["verdict"] == c["verdict"], (key, e["verdict"], c["verdict"])
        reproduced[key] = {"d_pct": e["d_pct_of_full"], "verdict": e["verdict"]}

    out = {"script": "scripts/noise_verdicts.py", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "registration": "results/phase_noise/PREREG_NOISE.md",
           "rule": ("per-seed d = R@20(a) - R@20(b); F_paired = 2 sd(d); F_level = 2 sd(R@20(full)) "
                    "over the full arm's converged seeds; SIGNIFICANT iff |mean d| > max(F_paired, "
                    "F_level), NULL iff below both, else MARGINAL; paired t two-sided. "
                    "d_pct_of_full = 100 mean(d)/mean(full R@20) = Table 1 unit."),
           "summarize_audit": ("exp_noise_holdout.summarize() lacks F_level, verdict, p and sign "
                               "count, pools by filename not dataset field, applies no protocol "
                               "filter to new runs, and scales a-b contrasts by mean(b) instead of "
                               "full; contrasts are therefore implemented here (see module docstring)."),
           "inputs": {"reference_runs": [_rel(f) for f in ref_files],
                      "new_runs": [_rel(f) for f in new_files],
                      "filter": "test_result present, stopping_step == 100, epochs_cap == 3000; "
                                "pooled by each record's dataset field; new-run labels restricted "
                                "to " + ", ".join(NEW_CONDS) + " (variants such as *_seed42 / "
                                "*_matchdim are not registered and are ignored)",
                      "notes": ref_notes + new_notes},
           "sanity_certified_table1_reproduced": reproduced,
           "seed_counts": {}, "registered_contrasts": {}, "vs_full": {},
           "prior_audit_comparison": {}}

    for ds in ("baby", "sports", "clothing", "microlens"):
        counts = {c: sorted(arm(allp, ds, c)) for c in ("full", "no_image", "no_text") + NEW_CONDS}
        out["seed_counts"][ds] = {c: {"n": len(v), "seeds": v} for c, v in counts.items() if v}

    # ---- registered contrasts
    for cid, ds, a, b, sign in REGISTERED:
        A, B, F = arm(allp, ds, a), arm(allp, ds, b), arm(allp, ds, "full")
        row = {"id": cid, "dataset": ds, "contrast": f"{a} - {b}", "prediction": PRED_TEXT[cid],
               "predicted_sign": sign, "n_seeds_a": len(A), "n_seeds_b": len(B),
               "n_seeds_paired": len(set(A) & set(B) & set(F)), "complete": False, "metrics": {}}
        for met in METRICS:
            e = contrast(A, B, F, met)
            if e is not None:
                if met == PRIMARY:
                    e["direction_as_predicted"] = (e["d"] * sign) > 0
                    e["n_seeds_as_predicted"] = e["n_pos"] if sign > 0 else e["n_neg"]
                    e["median_best_epoch_a"] = best_epochs(A, e["seeds"])
                    e["median_best_epoch_b"] = best_epochs(B, e["seeds"])
                row["metrics"][met] = e
        if PRIMARY in row["metrics"]:
            p = row["metrics"][PRIMARY]
            row["complete"] = p["complete"]
            row["verdict"] = p.get("verdict")
            row["direction_as_predicted"] = p["direction_as_predicted"]
            row["p"] = p.get("p")
            row["status"] = "final" if p["complete"] else p.get("verdict_status", "partial")
        else:
            row["status"] = "no paired runs yet"
        out["registered_contrasts"][f"{cid}/{ds}"] = row

    # ---- each condition vs full (Table 1 units); existing removal arms included for reference
    for ds in ("baby", "sports", "clothing"):
        F = arm(allp, ds, "full")
        for c in ("no_image", "no_text") + NEW_CONDS:
            C = arm(allp, ds, c)
            if not C:
                continue
            row = {"n_seeds": len(C), "complete": sorted(set(C) & set(F)) == SEEDS, "metrics": {}}
            for met in METRICS:
                e = contrast(C, F, F, met)
                if e is not None:
                    row["metrics"][met] = e
            pm = row["metrics"].get(PRIMARY, {})
            row["table1_style"] = {k: pm.get(k) for k in
                                   ("d_pct_of_full", "x_Fl", "x_Fp", "n_neg", "n", "p", "verdict")}
            out["vs_full"][f"{ds}/{c}"] = row

    # ---- prior-audit comparison
    prior = json.loads(PRIOR.read_text())
    ye = parse_ye(prior)
    pomo = parse_pomo(prior)
    ye_map = {"noise_image": ["noise_image", "no_image", "no_image_keepw"],
              "noise_text": ["noise_text", "no_text", "no_text_keepw"]}
    ye_rows = []
    for ds in ("baby", "sports", "clothing"):
        for ye_cond, ours_conds in ye_map.items():
            for met in ("Recall@10", "NDCG@10"):
                y = ye[ds][ye_cond][met]
                r = {"dataset": ds, "ye_condition": f"{ye_cond} (N(1,0.1) replacement, single run, seed 999)",
                     "metric": met, "ye_reported_pct": y["reported_pct"],
                     "ye_recomputed_pct_from_4dp": y["recomputed_pct_from_4dp"], "ours": {}}
                for oc in ours_conds:
                    v = out["vs_full"].get(f"{ds}/{oc}", {}).get("metrics", {}).get(met)
                    if v is None:
                        continue
                    r["ours"][oc] = {"n_seeds": v["n"], "complete": v["complete"],
                                     "our_pct": v["d_pct_of_full"],
                                     "gap_ours_minus_ye_pp": v["d_pct_of_full"] - y["reported_pct"],
                                     "our_F_level_pct": v.get("F_level_pct_of_full"),
                                     "our_F_paired_pct": v.get("F_paired_pct_of_full"),
                                     "ye_over_our_F_level": (abs(y["reported_pct"]) / v["F_level_pct_of_full"]
                                                             if v.get("F_level_pct_of_full") else None),
                                     "our_verdict_vs_full_at_this_metric": v.get("verdict")}
                ye_rows.append(r)
    pomo_rows = []
    for ds in ("baby", "clothing"):
        for met in ("Recall@20", "NDCG@20"):
            r = {"dataset": ds, "metric": met,
                 "pomo_image_only_vs_image_plus_text_pct": pomo["per_dataset"][ds][f"{met}_gap_pct"],
                 "ours": {}}
            for oc in ("no_text", "no_text_keepw"):
                v = out["vs_full"].get(f"{ds}/{oc}", {}).get("metrics", {}).get(met)
                if v is None:
                    continue
                r["ours"][oc] = {"n_seeds": v["n"], "complete": v["complete"], "our_pct": v["d_pct_of_full"],
                                 "gap_ours_minus_pomo_pp": v["d_pct_of_full"] - r["pomo_image_only_vs_image_plus_text_pct"],
                                 "our_F_level_pct": v.get("F_level_pct_of_full"),
                                 "our_F_paired_pct": v.get("F_paired_pct_of_full")}
            pomo_rows.append(r)
    out["prior_audit_comparison"] = {
        "source": "results/phase_novelty/prior_audits.json",
        "ye_et_al": {"rows": ye_rows, "parsed": ye,
                     "caveats": ["Ye: single run, seed 999, best of 8 configs selected on TEST NDCG@5, "
                                 "MMRec patience 20, possibly regenerated (unseeded) split; @5/@10 only.",
                                 "Ye's graph builder is directed + self-including; ours symmetrized "
                                 "(see exp_noise_holdout.YE_RECIPE) -- our noise arms are recipe- but "
                                 "not builder-faithful.",
                                 "Ye's FREEDOM image-knockout numbers are from arXiv v1/v2 appendix "
                                 "Table 9 (not the ECIR text); text from Table 8.",
                                 "our_pct = 100 mean(cond - full)/mean(full) over shared seeds (Table 1 "
                                 "convention); gap_ours_minus_ye_pp in percentage points."]},
        "pomo_et_al": {"rows": pomo_rows, "parsed": pomo,
                       "caveats": ["Pomo's gap is OUR derivation (RNet50 vs RNet50-Sbert, Table 2); "
                                   "Amazon Reviews 2023 data (different users/items), Sbert "
                                   "all-mpnet-base-v2 text, Elliot, 200 epochs, single seed 123.",
                                   "Pomo's image-only FREEDOM keeps the image graph at weight 0.1 "
                                   "(mw (0.1,0.9)), which is the no_text_keepw instrument; our no_text "
                                   "uses weight 1.",
                                   "Pets has no counterpart here (not in our datasets)."]},
    }

    # ---- cross-checks (read-only): prior_audits' stored numbers, and the running summaries
    xc = {}
    pa = prior.get("our_side_for_comparison", {})
    for ds, key in (("baby", "baby_no_image_ours_pct"), ("sports", "sports_no_image_ours_pct")):
        for met, val in pa.get(key, {}).items():
            v = out["vs_full"].get(f"{ds}/no_image", {}).get("metrics", {}).get(met)
            if v is not None:
                ok = abs(round(v["d_pct_of_full"], 2) - val) < 0.006
                assert ok, (ds, met, v["d_pct_of_full"], val)
                xc[f"prior_audits.our_side/{ds}/no_image/{met}"] = {
                    "agree": ok, "stored": val, "recomputed": v["d_pct_of_full"]}
    for ds in ("baby", "sports", "clothing"):
        sp = args.noise_dir / f"summary_{ds}.json"
        if not sp.is_file():
            continue
        s = json.loads(sp.read_text()).get("contrasts", {})
        rows = [(r["contrast"], r["metrics"]) for r in out["registered_contrasts"].values()
                if r["dataset"] == ds]
        rows += [(f"{k.split('/')[1]} - full", v["metrics"]) for k, v in out["vs_full"].items()
                 if k.split("/")[0] == ds]
        for name, mets in rows:
            if name not in s:
                continue
            for met, e in mets.items():
                se = s[name].get(met)
                if not se:
                    continue
                agree = (se["n"] == e["n"] and abs(se["delta_mean"] - e["d"]) < 1e-12 and
                         ("F_paired_2sd" not in se or abs(se["F_paired_2sd"] - e.get("F_paired", 0)) < 1e-12))
                xc[f"summary_{ds}/{name}/{met}"] = {"agree": agree, "summary_n": se["n"], "ours_n": e["n"]}
    out["cross_checks"] = xc

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(args.out)

    print("certified Table 1 retraining cells reproduced exactly:", len(reproduced))
    for k, r in out["registered_contrasts"].items():
        p = r["metrics"].get(PRIMARY)
        if p is None:
            print(f"{k:14s} {r['contrast']:28s} no paired runs yet")
            continue
        print(f"{k:14s} {r['contrast']:28s} n={p['n']}/8 d={p['d_pct_of_full']:+.2f}%full "
              f"xFp={p.get('x_Fp', float('nan')):.2f} xFl={p.get('x_Fl', float('nan')):.2f} "
              f"p={p.get('p', float('nan')):.3g} dir_ok={p['direction_as_predicted']} "
              f"{p.get('verdict', '-')} [{r['status']}]")
    for k, r in out["vs_full"].items():
        t = r["table1_style"]
        print(f"vs full {k:24s} n={t['n']} d={t['d_pct_of_full']:+.2f}% verdict={t['verdict']}")
    bad = [k for k, v in xc.items() if not v.get("agree")]
    print(f"cross-checks: {len(xc)} ({len(bad)} disagreements{': ' + str(bad) if bad else ''})")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
