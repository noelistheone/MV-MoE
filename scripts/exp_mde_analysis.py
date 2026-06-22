"""Exp A1 (analysis) — re-judge every modality-knockout significance call against the
correct per-(model,dataset) MDE instead of the single reused Baby MDE (0.000614).

Sources (all already on disk; FREEDOM needs NO new training):
  * FREEDOM per-seed R@20/N@20 at the operating point (mm_image_weight=0.1):
      - baby : results/phase0/seed_variance.json   (canonical 5-seed FREEDOM noise floor)
      - sports/clothing/elec : results/phase2/weight_sweep_retrain.json  (weight==0.1 seeds)
  * LGMRec per-seed R@20/N@20 : results/phase_mde/lgmrec_mde.json   (this run, NEW)
  * clean STRUCTURAL knockout deltas (image/text/both) : results/phase1/knockout_modality.json
  * trackA CLIP image-knockout deltas : results/trackA_clip/clip_freedom.json

MDE := 2 * sample-std of the metric across seeds, computed per (model,dataset).
A knockout is "significant" iff |delta| > that model+dataset's OWN MDE.
We print OLD verdict (reused Baby MDE) vs NEW verdict (own MDE) and FLAG every flip.

Output: results/phase_mde/significance_recheck.json + results/phase_mde/SUMMARY.md
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path("/workspace/MechInterp")
R = ROOT / "results"
OUT = R / "phase_mde"
BABY_REUSED_MDE_R20 = 0.0006135722885431936  # the single value previously reused everywhere
BABY_REUSED_MDE_N20 = 0.0005005  # FREEDOM baby N@20 MDE (phase0)


def mde(vals):
    if len(vals) < 2:
        return {"n": len(vals), "mean": (vals[0] if vals else None), "std": None, "MDE": None, "values": vals}
    s = statistics.stdev(vals)
    return {"n": len(vals), "mean": statistics.mean(vals), "std": s, "MDE": 2 * s, "values": [round(v, 6) for v in vals]}


def freedom_per_dataset_mde():
    out = {}
    # baby: canonical 5-seed from phase0
    p0 = json.loads((R / "phase0" / "seed_variance.json").read_text())
    fb = p0["summary"]["freedom"]
    out["baby"] = {"Recall@20": mde(fb["Recall@20"]["values"]),
                   "NDCG@20": mde(fb["NDCG@20"]["values"]),
                   "source": "phase0 5-seed canonical"}
    # sports/clothing/elec: weight==0.1 seeds from the retrain sweep
    ws = json.loads((R / "phase2" / "weight_sweep_retrain.json").read_text())
    by = {}
    for run in ws["runs"]:
        if abs(run["weight"] - 0.1) < 1e-9:
            by.setdefault(run["dataset"], {"Recall@20": [], "NDCG@20": []})
            by[run["dataset"]]["Recall@20"].append(run["test_result"]["Recall@20"])
            by[run["dataset"]]["NDCG@20"].append(run["test_result"]["NDCG@20"])
    for ds, d in by.items():
        if ds == "baby":
            continue
        out[ds] = {"Recall@20": mde(d["Recall@20"]), "NDCG@20": mde(d["NDCG@20"]),
                   "source": "phase2 weight_sweep_retrain (w=0.1)"}
    return out


def lgmrec_per_dataset_mde():
    p = OUT / "lgmrec_mde.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text()).get("summary", {}).get("lgmrec", {})
    out = {}
    for ds, mm in d.items():
        out[ds] = {"Recall@20": mde(mm["Recall@20"]["values"]) if "Recall@20" in mm else {"n": 0},
                   "NDCG@20": mde(mm["NDCG@20"]["values"]) if "NDCG@20" in mm else {"n": 0},
                   "source": "phase_mde new multi-seed"}
    return out


def load_knockouts():
    """clean structural knockout dR@20/dN@20 per (model,dataset)."""
    d = json.loads((R / "phase1" / "knockout_modality.json").read_text())
    out = {}
    for e in d:
        v = e["variants"]
        if "image_knockout" not in v:  # lightgcn control has none
            continue
        out[(e["model"], e["dataset"])] = {
            "image": {"dR@20": v["image_knockout"]["dR@20"], "dN@20": v["image_knockout"]["dN@20"],
                      "old_sig": v["image_knockout"].get("significant_vs_MDE")},
            "text": {"dR@20": v["text_knockout"]["dR@20"], "dN@20": v["text_knockout"]["dN@20"],
                     "old_sig": v["text_knockout"].get("significant_vs_MDE")},
            "both": {"dR@20": v["both_knockout"]["dR@20"], "dN@20": v["both_knockout"]["dN@20"],
                     "old_sig": v["both_knockout"].get("significant_vs_MDE")},
            "baseline_R@20": v["baseline"]["metrics"]["Recall@20"],
        }
    return out


def load_trackA():
    d = json.loads((R / "trackA_clip" / "clip_freedom.json").read_text())
    out = []
    for key, v in d.items():
        ds = key.split("/")[0]
        out.append({"dataset": ds, "variant": v["variant"],
                    "image_dR@20": v["image_knockout_dR@20"], "text_dR@20": v["text_knockout_dR@20"]})
    return out


def verdict(delta, mde_val):
    if mde_val is None:
        return None
    return abs(delta) > mde_val


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fmde = freedom_per_dataset_mde()
    lmde = lgmrec_per_dataset_mde()
    mde_by = {"freedom": fmde, "lgmrec": lmde}
    knock = load_knockouts()
    trackA = load_trackA()

    rows = []
    flips = []
    for (model, ds), kk in sorted(knock.items()):
        own = mde_by.get(model, {}).get(ds, {})
        own_r = own.get("Recall@20", {}).get("MDE")
        own_n = own.get("NDCG@20", {}).get("MDE")
        own_n_seeds = own.get("Recall@20", {}).get("n")
        for stream in ("image", "text", "both"):
            dr = kk[stream]["dR@20"]; dn = kk[stream]["dN@20"]; old = kk[stream]["old_sig"]
            new_r = verdict(dr, own_r); new_n = verdict(dn, own_n)
            row = {"model": model, "dataset": ds, "stream": stream,
                   "dR@20": dr, "dN@20": dn,
                   "own_MDE_R@20": own_r, "own_MDE_N@20": own_n, "own_n_seeds": own_n_seeds,
                   "reused_baby_MDE_R@20": BABY_REUSED_MDE_R20,
                   "old_sig_vs_baby_MDE": old, "new_sig_vs_own_MDE_R@20": new_r,
                   "new_sig_vs_own_MDE_N@20": new_n}
            rows.append(row)
            if old is not None and new_r is not None and bool(old) != bool(new_r):
                flips.append({**row, "flip": f"{model}/{ds}/{stream}: {old} -> {new_r}"})

    # trackA CLIP re-judged against FREEDOM own per-dataset MDE (proxy: same arch+dataset)
    trackA_rows = []
    for t in trackA:
        ds = t["dataset"]
        own_r = fmde.get(ds, {}).get("Recall@20", {}).get("MDE")
        old = abs(t["image_dR@20"]) > BABY_REUSED_MDE_R20
        new = verdict(t["image_dR@20"], own_r)
        tr = {**t, "old_sig_vs_baby_MDE": old, "new_sig_vs_own_freedom_MDE": new,
              "own_freedom_MDE_R@20": own_r}
        trackA_rows.append(tr)
        if new is not None and bool(old) != bool(new):
            flips.append({"trackA": True, "dataset": ds, "variant": t["variant"],
                          "image_dR@20": t["image_dR@20"],
                          "flip": f"trackA {ds}/{t['variant']} image-KO: {old} -> {new}"})

    out = {"freedom_per_dataset_mde": fmde, "lgmrec_per_dataset_mde": lmde,
           "knockout_significance": rows, "trackA_clip_significance": trackA_rows,
           "flips_vs_reused_baby_mde": flips,
           "note": ("MDE=2*std(metric across seeds) per (model,dataset). 'old' used the single "
                    "reused Baby FREEDOM MDE (0.000614) as the threshold everywhere; 'new' uses each "
                    "model+dataset's own MDE. trackA CLIP uses FREEDOM's own per-dataset MDE as proxy.")}
    (OUT / "significance_recheck.json").write_text(json.dumps(out, indent=2))

    # ---- SUMMARY.md ----
    L = ["# Exp A1 — per-(model,dataset) MDE & significance re-check", "",
         "Closes the 'reused Baby MDE' gap. MDE = 2*std(R@20) across seed-retrains, measured per",
         "(model,dataset). FREEDOM uses existing retrains (no new training); LGMRec is newly trained.", "",
         "## Measured per-dataset noise floors (MDE on R@20)", "",
         "| model | dataset | n seeds | mean R@20 | std | **MDE(R@20)** | vs reused Baby MDE 0.00061 |",
         "|---|---|---|---|---|---|---|"]
    for model, mm in (("freedom", fmde), ("lgmrec", lmde)):
        for ds, d in mm.items():
            r = d.get("Recall@20", {})
            if r.get("MDE") is None:
                continue
            ratio = r["MDE"] / BABY_REUSED_MDE_R20
            L.append(f"| {model} | {ds} | {r['n']} | {r['mean']:.4f} | {r['std']:.5f} | "
                     f"**{r['MDE']:.5f}** | {ratio:.1f}x |")
    L += ["", "## Knockout significance — OLD (reused Baby MDE) vs NEW (own MDE)", "",
          "| model | dataset | stream | dR@20 | own MDE | old sig | **new sig** |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        om = f"{r['own_MDE_R@20']:.5f}" if r["own_MDE_R@20"] is not None else "n/a"
        L.append(f"| {r['model']} | {r['dataset']} | {r['stream']} | {r['dR@20']:+.5f} | {om} | "
                 f"{r['old_sig_vs_baby_MDE']} | **{r['new_sig_vs_own_MDE_R@20']}** |")
    L += ["", "## trackA CLIP image-knockout — re-judged vs FREEDOM own per-dataset MDE", "",
          "| dataset | variant | image dR@20 | own FREEDOM MDE | old sig (Baby MDE) | **new sig** |",
          "|---|---|---|---|---|---|"]
    for t in trackA_rows:
        om = f"{t['own_freedom_MDE_R@20']:.5f}" if t["own_freedom_MDE_R@20"] is not None else "n/a"
        L.append(f"| {t['dataset']} | {t['variant']} | {t['image_dR@20']:+.5f} | {om} | "
                 f"{t['old_sig_vs_baby_MDE']} | **{t['new_sig_vs_own_freedom_MDE']}** |")
    L += ["", f"## FLIPS vs reused-Baby-MDE ({len(flips)})", ""]
    if not flips:
        L.append("_No verdict changed._")
    for f in flips:
        L.append(f"- {f['flip']}")
    (OUT / "SUMMARY.md").write_text("\n".join(L))

    print("=== per-dataset MDE (R@20) ===")
    for model, mm in (("freedom", fmde), ("lgmrec", lmde)):
        for ds, d in mm.items():
            r = d.get("Recall@20", {})
            if r.get("MDE") is not None:
                print(f"  {model}/{ds}: MDE={r['MDE']:.5f} (n={r['n']}, {r['MDE']/BABY_REUSED_MDE_R20:.1f}x Baby)")
    print(f"\nFLIPS: {len(flips)}")
    for f in flips:
        print("  -", f["flip"])
    print(f"\nWrote {OUT/'significance_recheck.json'} + SUMMARY.md")


if __name__ == "__main__":
    main()
