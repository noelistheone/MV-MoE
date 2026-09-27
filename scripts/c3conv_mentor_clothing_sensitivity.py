"""Post hoc sensitivity (NOT a registered verdict): MENTOR on Clothing without the early-stopped seeds.

Under the registered converged protocol (patience 100, cap 3000), 3 of 8 MENTOR/Clothing seeds stop
at an early validation peak (best epochs 13-15, test Recall@20 ~0.049); the other five peak at
epochs 1447-2142 (~0.080). Validation curves exist only for seeds 2028-2031 (logging started
2026-09-25): all four peak at epoch 14-17 and dip; the three that recovered first beat that peak 94-99
epochs later (epochs 108-115), i.e. at the edge of the 100-epoch patience window, and s2029 never did.
The bimodal outcome inflates the level floor (2 sd of the logged full-model Recall@20; the same for
both arms) to 39.8x the paired floor for image and 7.4x for text, and to 9.9x the level floor over
the five late-peaking seeds. The registered verdict stands; this script
recomputes the same cell statistics on the five late-peaking seeds only, and reports the
early-peak mechanism from the recorded validation curves.
-> results/phase_c3conv/sensitivity_mentor_clothing.json
"""
import json, sys
from pathlib import Path
ROOT = Path("/workspace/MechInterp")
sys.path.insert(0, str(ROOT / "scripts"))
import exp_c3conv_score as sc  # noqa: E402

rows = [r for r in json.loads((sc.C3 / "scored_clothing.json").read_text())["rows"].values()
        if r["model"] == "mentor" and r["guard"]["ok"]]
early = sorted(r["seed"] for r in rows if r["best_epoch_logged"] < 100)
late = [r for r in rows if r["best_epoch_logged"] >= 100]
out = {"note": __doc__.strip().splitlines()[0], "all_seeds": sorted(r["seed"] for r in rows),
       "early_stopped_seeds": early,
       "best_epochs": {r["seed"]: r["best_epoch_logged"] for r in rows},
       "R20_logged": {r["seed"]: r["R@20_logged"] for r in rows}, "cells": {}}
for s in ("image", "text"):
    full, part = sc.cell_stats(rows, s), sc.cell_stats(late, s)
    keep = ("n", "mean_base_R20", "mean_delta", "rel_delta_pct", "F_level", "F_paired", "x_F_level",
            "x_F_paired", "verdict", "t", "p", "ci95", "n_negative")
    out["cells"][s] = {"registered_all_8": {k: full[k] for k in keep},
                       "sensitivity_late_peak_only": {k: part[k] for k in keep}}
curves = {}
for r in json.loads((sc.C3 / "runs_clothing.json").read_text()):
    if r.get("model") == "mentor" and r.get("valid_curve"):
        vc = [v for _, v in r["valid_curve"]]
        pk = max(range(min(40, len(vc))), key=lambda i: vc[i])
        after = next((i for i in range(pk + 1, len(vc)) if vc[i] > vc[pk]), None)
        curves[r["seed"]] = {"early_peak_epoch": pk, "early_peak_valid_R20": vc[pk],
                             "first_epoch_beating_early_peak": after,
                             "gap_epochs": (after - pk) if after is not None else None,
                             "final_best_valid_R20": max(vc)}
out["valid_curve_mechanism"] = curves
(sc.C3 / "sensitivity_mentor_clothing.json").write_text(json.dumps(out, indent=1))
for s, c in out["cells"].items():
    a, b = c["registered_all_8"], c["sensitivity_late_peak_only"]
    print(f"{s:5s} all8: d={a['mean_delta']:+.5f} ({a['rel_delta_pct']:+.2f}%) Fl={a['F_level']:.5f} Fp={a['F_paired']:.5f} {a['verdict']} p={a['p']:.3g}"
          f" | late5: d={b['mean_delta']:+.5f} ({b['rel_delta_pct']:+.2f}%) Fl={b['F_level']:.5f} Fp={b['F_paired']:.5f} {b['verdict']} p={b['p']:.3g}")
print("early:", early, "curves:", curves)
