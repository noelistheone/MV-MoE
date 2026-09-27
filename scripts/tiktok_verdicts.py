"""Registered TikTok retraining verdicts (results/phase_shortvideo2/PREREG_TIKTOK.md, T1-T4), with
exactly the statistics used for the other datasets.

Statistics come from the certified code, not a re-implementation:
  * gen_facts.py's exact-arith block (executed verbatim by tiktok_common.exact_arith()):
      holdout_arm  -> per-arm paired stats for Recall@20 / NDCG@20 / Recall@10 / NDCG@10
                      (mean d, d %, F_paired, F_level, x-floors, t(n-1), p, 95% CI, d<0, verdict)
      ttest1, dsd, two_floor -> the registered T3 contrast
  * holdout_verdicts.cell (float/scipy; imported) -> the Recall@20 cells in the same form as
    results/phase_holdout/final_verdicts_p100.json; cross-checked against the exact values.
Before anything TikTok is computed, an identity self-test re-derives the MicroLens (and Amazon)
numbers already certified in results/FACTS.md and final_verdicts_p100.json with the same
wrappers and refuses to continue if any line differs.

Registered T3: per seed, c_s = R(no_text, s) - R(no_image, s); paired floor F_p = 2 sd(c_s); level
floor F_l = 2 sd(R(full, s)) (seeds common to all three arms); verdict by the two-floor rule; paired
t and 95% CI. T3 FAILS iff mean c < 0 (text removal costs more) and |mean c| > max(F_p, F_l).

Inputs:  results/phase_shortvideo2/freedom_p100_tiktok_runs.json (read only)
         results/phase_shortvideo2/lightgcn_p100_tiktok_runs.json (read only)
         results/phase_shortvideo2/tiktok_knockout.json, alignment_tiktok.json, graph_health.json
         (optional; summarized if present)
Output:  results/phase_shortvideo2/tiktok_verdicts.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/MechInterp")
sys.path.insert(0, str(ROOT / "scripts"))
import tiktok_common as tc            # noqa: E402
import holdout_verdicts as hv         # noqa: E402  (imports scipy only; main() not run)

OUT = tc.SV2 / "tiktok_verdicts.json"
METRICS = ("Recall@20", "NDCG@20", "Recall@10", "NDCG@10")
FACTS_METS = ("Recall@20", "NDCG@20", "Recall@10")      # gen_facts _HOLDMETS
HV_NAME = {"BELOW-FLOOR": "NULL"}                        # holdout_verdicts' name for BELOW-FLOOR


def arm_json(a: dict | None) -> dict | None:
    if a is None:
        return None
    o = tc.jsonable(a)
    o["ci95_pct"] = [100 * x / float(tc.exact_arith()["dmean"](a["full"])) for x in o["ci"]]
    return o


def holdout_row(ds, cond, met, a) -> str:
    ns = tc.exact_arith()
    sci, fx, ptab = ns["sci"], ns["fx"], ns["ptab"]
    return tc.src_row(ds, cond, met, sci(a["mean"], 3), fx(a["pct"], 2, True), fx(a["pct"], 1, True),
                      fx(a["xfl"], 2), fx(a["xfp"], 2), fx(a["t"], 2, True), sci(a["p"], 2),
                      ptab(a["p"]), f"{a['neg']}/{a['n']}", a["verdict"], "HO")


def contrast_row(ds, met, c2) -> str:
    ns = tc.exact_arith()
    sci, fx, ptab = ns["sci"], ns["fx"], ns["ptab"]
    return tc.src_row(ds, met, sci(c2["mean"], 3), sci(c2["fp"], 4), fx(c2["xfp"], 2),
                      fx(c2["t"], 2, True), ptab(c2["p"]), sci(c2["p"], 3), f"{c2['neg']}/{c2['n']}", "HO")


def t3_contrast(pooled: dict, ds: str, metric: str = "Recall@20") -> dict | None:
    """Registered T3: c_s = R(no_text) - R(no_image); F_p from c_s, F_l from the full arm."""
    ns = tc.exact_arith()
    D, dsd, dmean, ttest1, two_floor = ns["D"], ns["dsd"], ns["dmean"], ns["ttest1"], ns["two_floor"]
    arms = {c: {k[2]: r for k, r in pooled.items() if k[0] == ds and k[1] == c}
            for c in ("full", "no_image", "no_text")}
    seeds = sorted(set(arms["full"]) & set(arms["no_image"]) & set(arms["no_text"]))
    if len(seeds) < 2:
        return None
    f = [D(arms["full"][s]["test_result"][metric]) for s in seeds]
    ni = [D(arms["no_image"][s]["test_result"][metric]) for s in seeds]
    nt = [D(arms["no_text"][s]["test_result"][metric]) for s in seeds]
    c = [a - b for a, b in zip(nt, ni)]
    tt = ttest1(c)
    fp, fl = 2 * tt["sd"], 2 * dsd(f)
    mu = dmean(f)
    v = two_floor(tt["mean"], fp, fl)
    return {"seeds": seeds, "n": len(seeds), "metric": metric,
            "definition": "c_s = R(no_text,s) - R(no_image,s); F_p = 2 sd(c_s); F_l = 2 sd(R(full,s))",
            "c": c, "mean": tt["mean"], "pct_of_full": 100 * tt["mean"] / mu, "sd": tt["sd"],
            "F_paired": fp, "F_level": fl, "x_F_paired": abs(tt["mean"]) / fp, "x_F_level": abs(tt["mean"]) / fl,
            "t": tt["t"], "df": len(seeds) - 1, "p": tt["p"], "ci95": tt["ci"],
            "ci95_pct_of_full": tuple(100 * x / mu for x in tt["ci"]),
            "n_negative": sum(1 for x in c if x < 0), "n_positive": sum(1 for x in c if x > 0),
            "verdict": v,
            "text_cost_significantly_exceeds_image_cost": bool(v == "SIGNIFICANT" and tt["mean"] < 0)}


def hv_cells(pooled: dict, ds: str) -> dict:
    """holdout_verdicts.cell on this dataset's pooled runs (the final_verdicts_p100.json form)."""
    runs = [r for k, r in pooled.items() if k[0] == ds]
    return {f"{ds}/{c}": hv.cell(runs, c) for c in ("no_image", "no_text") if hv.cell(runs, c)}


def crosscheck_hv(pooled: dict, ds: str) -> dict:
    """Float (holdout_verdicts, scipy) vs exact (gen_facts) Recall@20 cells must agree."""
    ns = tc.exact_arith()
    mx, dis = 0.0, []
    for c in ("no_image", "no_text"):
        runs = [r for k, r in pooled.items() if k[0] == ds]
        h = hv.cell(runs, c)
        a = ns["holdout_arm"](pooled, ds, c, "Recall@20")
        if h is None or a is None:
            continue
        mx = max(mx, abs(h["d"] - float(a["mean"])), abs(h["F_paired"] - float(a["fp"])),
                 abs(h["F_level"] - float(a["fl"])), abs(h["p"] - float(a["p"])))
        if h["verdict"] != HV_NAME.get(a["verdict"], a["verdict"]):
            dis.append(c)
    return {"max_abs_diff_d_Fp_Fl_p": mx, "verdict_disagreements": dis}


# ----------------------------------------------------------------------------- identity self-test
def selftest() -> dict:
    """Re-derive already-certified numbers with the same wrappers; any mismatch aborts."""
    ns = tc.exact_arith()
    D, fx = ns["D"], ns["fx"]
    facts = (ROOT / "results" / "FACTS.md").read_text().splitlines()
    fset = set(facts)
    res = {"gen_facts_exact_arith_sha256": ns["_block_sha256"]}

    # (a) run pooling identical to gen_facts.load_p100_holdout
    ref, files = ns["load_p100_holdout"]()
    mine = tc.pool_runs(sorted((ROOT / "results" / "phase_holdout").glob("freedom_p100*_runs.json")))
    assert set(ref) == set(mine) and all(ref[k] == mine[k] for k in ref), "pool_runs != load_p100_holdout"
    res["pooling_identical_to_load_p100_holdout"] = {"n_runs": len(ref), "files": files}

    # (b) holdout rows (21b table) + image-minus-text contrast rows, byte-identical to FACTS.md
    rows_ok, rows_missing = [], []
    for ds in ("baby", "sports", "clothing", "microlens"):
        for cond in ("no_image", "no_text"):
            for met in FACTS_METS:
                a = ns["holdout_arm"](ref, ds, cond, met)
                if a:
                    (rows_ok if holdout_row(ds, cond, met, a) in fset else rows_missing).append(f"{ds}/{cond}/{met}")
        for met in FACTS_METS:
            c2 = ns["holdout_arm"](ref, ds, "no_image", met, base_cond="no_text")
            if c2:
                (rows_ok if contrast_row(ds, met, c2) in fset else rows_missing).append(f"contrast {ds}/{met}")
    res["facts_rows_reproduced"] = {"n_identical": len(rows_ok), "missing_or_different": rows_missing}
    assert not rows_missing, f"FACTS rows not reproduced: {rows_missing}"

    # (c) holdout_verdicts.cell == stored final_verdicts_p100.json (to 1e-12, verdict equal)
    fv = json.loads((ROOT / "results" / "phase_holdout" / "final_verdicts_p100.json").read_text())
    mx = 0.0
    for k, v in fv.items():
        ds, c = k.split("/")
        h = hv_cells(ref, ds)[k]
        for key in ("d", "F_paired", "F_level", "p"):
            mx = max(mx, abs(h[key] - v[key]))
        assert h["verdict"] == v["verdict"], k
    assert mx < 1e-12, mx
    res["holdout_verdicts_cell_vs_final_verdicts_p100"] = {"n_cells": len(fv), "max_abs_diff": mx}
    res["float_vs_exact_R20"] = {ds: crosscheck_hv(ref, ds) for ds in ("baby", "sports", "clothing", "microlens")}
    for ds, cc in res["float_vs_exact_R20"].items():
        assert cc["max_abs_diff_d_Fp_Fl_p"] < 1e-12 and not cc["verdict_disagreements"], (ds, cc)

    # (d) the T3 wrapper is the FACTS contrast with the sign flipped (same F_p, |t|, p)
    t3m = {}
    for ds in ("baby", "sports", "clothing", "microlens"):
        for met in FACTS_METS:
            t3, c2 = t3_contrast(ref, ds, met), ns["holdout_arm"](ref, ds, "no_image", met, base_cond="no_text")
            if not (t3 and c2):
                continue
            assert t3["seeds"] == c2["seeds"]
            assert t3["mean"] == -c2["mean"] and t3["F_paired"] == c2["fp"] and t3["p"] == c2["p"] \
                and t3["t"] == -c2["t"], (ds, met)
            t3m[f"{ds}/{met}"] = {"c_mean": float(t3["mean"]), "verdict": t3["verdict"],
                                  "F_level_from_full": float(t3["F_level"]), "x_F_paired": float(t3["x_F_paired"]),
                                  "x_F_level": float(t3["x_F_level"]), "p": float(t3["p"])}
    res["t3_wrapper_equals_facts_contrast_sign_flipped"] = True
    res["t3_same_statistic_on_existing_datasets"] = t3m

    # (e) the deletion wrappers reproduce converged_knockout.json streams and FACTS s.21a rows
    ck = json.loads((ROOT / "results" / "phase_convergence" / "converged_knockout.json").read_text())
    del_ok, del_bad, mxs = [], [], 0.0
    for ds in ("baby", "sports", "clothing", "microlens"):
        if ds not in ck:
            continue
        cell = tc.ck_cell_float([dict(r) for r in ck[ds]["per_seed"]])
        for st, sv in ck[ds]["streams"].items():
            for key in ("mean_delta", "sd", "F_paired", "x_F_paired", "x_F_level"):
                mxs = max(mxs, abs(cell["streams"][st][key] - sv[key]))
            assert cell["streams"][st]["verdict"] == sv["verdict"] and \
                cell["streams"][st]["n_negative"] == sv["n_negative"], (ds, st)
        mxs = max(mxs, abs(cell["F_level"] - ck[ds]["F_level"]), abs(cell["mean_R20"] - ck[ds]["mean_R20"]))
        ex = tc.deletion_exact(ck[ds]["per_seed"])
        for st in ("image", "text", "both"):
            (del_ok if tc.deletion_row(ds, st, ex[st]) in fset else del_bad).append(f"{ds}/{st}")
    assert mxs == 0.0, mxs
    assert not del_bad, del_bad
    res["deletion_float_cell_vs_converged_knockout_json_max_abs_diff"] = mxs
    res["deletion_facts_rows_reproduced"] = {"n_identical": len(del_ok), "missing_or_different": del_bad}

    # (f) the MicroLens FREEDOM-vs-LightGCN line
    tr = json.loads((ROOT / "results" / "phase_convergence" / "table_rerun.json").read_text())
    trt = {(r["model"], r["dataset"]): r["test_result"] for r in tr if "test_result" in r}
    fm, lm = trt[("freedom", "microlens")], trt[("lightgcn", "microlens")]
    gap = 100 * (1 - D(fm["Recall@20"]) / D(lm["Recall@20"]))
    line = (f"- MicroLens, converged: FREEDOM {fx(fm['Recall@20'], 6)} vs LightGCN {fx(lm['Recall@20'], 6)}: "
            f"FREEDOM is {fx(gap, 2)}% ({fx(gap, 1)}%) below [TR].")
    assert line in fset, line
    res["microlens_lightgcn_line_reproduced"] = line
    return res


# ----------------------------------------------------------------------------- LightGCN comparison
def lightgcn_gap(pooled: dict) -> dict:
    ns = tc.exact_arith()
    D, fx = ns["D"], ns["fx"]
    out = {"definition": "gap % = 100 * (1 - R_FREEDOM / R_LightGCN), one converged seed each "
                         "(same formula and convention as the FACTS MicroLens line; positive = FREEDOM below)"}
    lg = {}
    if tc.LIGHTGCN_RUNS.is_file():
        for r in json.loads(tc.LIGHTGCN_RUNS.read_text()):
            if "test_result" in r and r.get("stopping_step") == tc.PATIENCE and r.get("epochs_cap") == tc.CAP:
                lg[r["seed"]] = r
    fr = {k[2]: r for k, r in pooled.items() if k[0] == "tiktok" and k[1] == "full"}
    common = sorted(set(lg) & set(fr))
    tr = json.loads((ROOT / "results" / "phase_convergence" / "table_rerun.json").read_text())
    trt = {(r["model"], r["dataset"]): r for r in tr if "test_result" in r}
    ml = {}
    for met in METRICS:
        f_, l_ = trt[("freedom", "microlens")]["test_result"][met], trt[("lightgcn", "microlens")]["test_result"][met]
        ml[met] = {"freedom": f_, "lightgcn": l_, "gap_pct": float(100 * (1 - D(f_) / D(l_))),
                   "gap_pct_2dp": fx(100 * (1 - D(f_) / D(l_)), 2)}
    out["microlens_reference"] = {"source": "results/phase_convergence/table_rerun.json (seed 2024 each)",
                                  "by_metric": ml}
    if not common:
        out["tiktok"] = None
        out["status"] = (f"PENDING: LightGCN TikTok runs {sorted(lg)}; FREEDOM full runs {sorted(fr)}; "
                         "no seed has both")
        return out
    s = 2024 if 2024 in common else common[0]
    tk = {}
    for met in METRICS:
        f_, l_ = fr[s]["test_result"][met], lg[s]["test_result"][met]
        tk[met] = {"freedom": f_, "lightgcn": l_, "gap_pct": float(100 * (1 - D(f_) / D(l_))),
                   "gap_pct_2dp": fx(100 * (1 - D(f_) / D(l_)), 2)}
    out["tiktok"] = {"seed": s, "freedom_best_epoch": fr[s]["best_epoch"], "lightgcn_best_epoch": lg[s]["best_epoch"],
                     "lightgcn_hit_cap": lg[s].get("hit_cap"), "by_metric": tk,
                     "facts_style_line": (f"- TikTok, converged: FREEDOM {fx(fr[s]['test_result']['Recall@20'], 6)} vs "
                                          f"LightGCN {fx(lg[s]['test_result']['Recall@20'], 6)}: FREEDOM is "
                                          f"{fx(100 * (1 - D(fr[s]['test_result']['Recall@20']) / D(lg[s]['test_result']['Recall@20'])), 2)}% "
                                          f"below (negative = above) [TT].")}
    out["status"] = "OK"
    return out


# ----------------------------------------------------------------------------- T4 (alignment)
def t4_block(t3: dict | None) -> dict:
    al = tc.SV2 / "alignment_tiktok.json"
    out = {"rule": ("T4 applies only if T3 holds. It holds iff TikTok's text-minus-image alignment gap "
                    "does not exceed MicroLens's text-minus-image gap. Primary: raw features, absolute gap g "
                    "(Table 3 headline, uniform null). Also reported: propagated streams (needs a converged "
                    "TikTok checkpoint), dispersion-normalized gap, popularity-matched null.")}
    if not al.is_file():
        out["status"] = "PENDING: alignment_tiktok.json missing (run scripts/tiktok_alignment.py)"
        return out
    a = json.loads(al.read_text())
    out["t3_holds"] = None if t3 is None else (not t3["text_cost_significantly_exceeds_image_cost"])
    out["comparison"] = a.get("t4_comparison")
    prim = (a.get("t4_comparison") or {}).get("uniform", {}).get("raw", {}).get("gap_abs")
    if prim:
        out["primary_point_estimate_holds"] = prim["tiktok_minus_microlens"] <= 0
        out["primary"] = prim
    if out["t3_holds"] is False:
        out["status"] = "T4 NOT APPLICABLE (T3 failed); comparison reported for completeness"
    elif out["t3_holds"] is None:
        out["status"] = "T3 not yet decidable; T4 comparison reported provisionally"
    else:
        out["status"] = "OK"
    return out


def main() -> int:
    t0 = time.time()
    st = selftest()
    print(f"self-test passed: {st['facts_rows_reproduced']['n_identical']} FACTS holdout/contrast rows, "
          f"{st['deletion_facts_rows_reproduced']['n_identical']} deletion rows, final_verdicts max diff "
          f"{st['holdout_verdicts_cell_vs_final_verdicts_p100']['max_abs_diff']:.1e}", flush=True)

    pooled = tc.pool_runs([tc.TIKTOK_RUNS])
    pooled = {k: r for k, r in pooled.items() if k[0] == "tiktok"}
    ns = tc.exact_arith()
    have = {c: sorted(k[2] for k in pooled if k[1] == c) for c in ("full", "no_image", "no_text")}
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
           "registration": "results/phase_shortvideo2/PREREG_TIKTOK.md",
           "runs_file": str(tc.TIKTOK_RUNS.relative_to(ROOT)),
           "protocol_filter": "stopping_step == 100 and epochs_cap == 3000",
           "seeds_finished": have,
           "complete": all(len(v) == 8 for v in have.values()),
           "identity_selftest": st}

    # retraining arms, every metric, exact
    arms, rows = {}, []
    for cond in ("no_image", "no_text"):
        for met in METRICS:
            a = ns["holdout_arm"](pooled, "tiktok", cond, met)
            arms[f"{cond}/{met}"] = arm_json(a)
            if a:
                rows.append(holdout_row("tiktok", cond, met, a))
    out["retraining"] = arms
    out["final_verdicts_p100_form"] = hv_cells(pooled, "tiktok")
    out["float_vs_exact_R20"] = crosscheck_hv(pooled, "tiktok")
    # FACTS-style contrast (no_image - no_text; F_p only) for parity with the other datasets' table
    con = {}
    for met in METRICS:
        c2 = ns["holdout_arm"](pooled, "tiktok", "no_image", met, base_cond="no_text")
        con[met] = arm_json(c2)
        if c2 and met in FACTS_METS:
            rows.append(contrast_row("tiktok", met, c2))
    out["facts_contrast_no_image_minus_no_text"] = con
    t3 = {met: t3_contrast(pooled, "tiktok", met) for met in METRICS}
    out["T3_contrast"] = {m: (tc.jsonable(v) if v else None) for m, v in t3.items()}
    out["facts_rows"] = rows

    # registered predictions (Recall@20, as registered)
    a_nt, a_ni, c = arms.get("no_text/Recall@20"), arms.get("no_image/Recall@20"), t3["Recall@20"]
    pred = {}
    pred["T1"] = ({"statement": "retraining without text costs less than 10% of Recall@20",
                   "no_text_d_pct": a_nt["pct"], "ci95_pct": a_nt["ci95_pct"],
                   "holds_point_estimate": (-a_nt["pct"]) < 10, "n": a_nt["n"]} if a_nt else "PENDING")
    pred["T2"] = ({"statement": "retraining without images is not SIGNIFICANT (BELOW or MARGINAL)",
                   "verdict": a_ni["verdict"], "d_pct": a_ni["pct"], "x_F_level": a_ni["xfl"],
                   "x_F_paired": a_ni["xfp"], "p": a_ni["p"], "n_negative": a_ni["neg"],
                   "holds": a_ni["verdict"] != "SIGNIFICANT", "n": a_ni["n"]} if a_ni else "PENDING")
    pred["T3"] = ({"statement": "text's cost does not significantly exceed image's (c = no_text - no_image)",
                   "mean_c": float(c["mean"]), "pct_of_full": float(c["pct_of_full"]), "verdict": c["verdict"],
                   "x_F_paired": float(c["x_F_paired"]), "x_F_level": float(c["x_F_level"]), "t": float(c["t"]),
                   "p": float(c["p"]), "n_negative": c["n_negative"],
                   "holds": not c["text_cost_significantly_exceeds_image_cost"], "n": c["n"]} if c else "PENDING")
    pred["T4"] = t4_block(c)
    partial = not out["complete"]
    for k in ("T1", "T2", "T3"):
        if isinstance(pred[k], dict):
            pred[k]["provisional"] = partial
    out["predictions"] = pred

    out["lightgcn_comparison"] = lightgcn_gap(pooled)

    kf = tc.SV2 / "tiktok_knockout.json"
    if kf.is_file():
        k = json.loads(kf.read_text())
        out["deletion"] = {"source": str(kf.relative_to(ROOT)), "n_seeds": (k.get("tiktok") or {}).get("n_seeds"),
                           "exact": k.get("exact"), "facts_rows": k.get("facts_rows")}
    else:
        out["deletion"] = "PENDING: run scripts/tiktok_converged_knockout.py"
    gh = tc.SV2 / "graph_health.json"
    out["graph_health"] = str(gh.relative_to(ROOT)) if gh.is_file() else "PENDING"

    OUT.write_text(json.dumps(out, indent=1, default=str))
    print(f"TikTok seeds finished: {have}")
    for r in rows:
        print(r)
    if c:
        print(f"T3 c=no_text-no_image: mean {float(c['mean']):+.6f} ({float(c['pct_of_full']):+.2f}% of full) "
              f"xFp={float(c['x_F_paired']):.2f} xFl={float(c['x_F_level']):.2f} t={float(c['t']):+.2f} "
              f"p={float(c['p']):.3g} -> {c['verdict']}")
    print(f"Wrote {OUT} ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
