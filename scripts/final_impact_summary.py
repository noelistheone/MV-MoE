"""Final summary of the risk-reduction experiments and of what they change in the SAC paper.

Read-only on every input; CPU only. Writes results/phase_paper/final_impact_summary.json and
prints a readable digest. Inputs:
  results/phase_c3conv/verdicts.json (AMEND2: damrs_g11 replaces the void defective DAMRS cells),
  results/phase_c3conv/verdicts_preAMEND2.json (the verdicts computed before the DAMRS repair),
  results/phase_exact/exact_crossarch.json + screen_vs_exact.json (the preliminary screen),
  results/phase_c3conv_g11/screen_g11.json (repaired DAMRS screen cells),
  results/phase_noise/noise_verdicts.json, results/phase_c3conv/sensitivity_mentor_clothing.json.
Multiplicity uses the scorer's own _bh/_holm so every count matches its implementation.
"""
import json, sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
ROOT = Path("/workspace/MechInterp")
sys.path.insert(0, str(ROOT / "scripts"))
from exp_c3conv_score import _bh, _holm, Q_BH, ALPHA_HOLM  # noqa: E402

R = ROOT / "results"
V = json.loads((R / "phase_c3conv/verdicts.json").read_text())
V0 = json.loads((R / "phase_c3conv/verdicts_preAMEND2.json").read_text())
ARCH = ["lattice", "damrs", "vbpr", "mentor", "cohesion", "lgmrec", "mmgcn", "mgcn", "smore", "gume"]
MARK = {"SIGNIFICANT": "beyond noise", "MARGINAL": "marginal", "BELOW": "within noise"}
r2 = lambda x: float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def cells(v, repaired):
    out = {}
    for k, c in v["cells"].items():
        m, ds, s = k.split("/")
        if s not in ("image", "text"):
            continue
        if repaired and m == "damrs":
            continue
        if not repaired and m == "damrs_g11":
            continue
        out[("damrs" if m == "damrs_g11" else m, ds, s)] = c
    return out


def family(cs, datasets):
    keys = [k for k, c in cs.items() if k[1] in datasets and c.get("judged", True) and c.get("p") is not None
            and (c.get("judged") or c.get("dataset_admitted") is not False)]
    p = [cs[k]["p"] for k in keys]
    bh, hm = _bh(p), _holm(p)
    return {k: {"p_bh": b, "p_holm": h, "bh": b <= Q_BH, "holm": h <= ALPHA_HOLM} for k, b, h in zip(keys, bh, hm)}


def counts(cs, datasets, fam):
    img = [k for k in cs if k[1] in datasets and k[2] == "image"]
    txt = [k for k in cs if k[1] in datasets and k[2] == "text"]
    beyond = lambda ks: [k for k in ks if cs[k]["verdict"] == "SIGNIFICANT"]
    bh_img = [k for k in img if fam.get(k, {}).get("bh")]
    holm_img = [k for k in img if fam.get(k, {}).get("holm")]
    return {"image_cells": len(img), "image_beyond": sorted("/".join(k[:2]) for k in beyond(img)),
            "text_cells": len(txt), "text_beyond": sorted("/".join(k[:2]) for k in beyond(txt)),
            "text_beyond_architectures": sorted({k[0] for k in beyond(txt)}),
            "image_bh": len(bh_img), "image_holm": len(holm_img),
            "image_bh_gains": sum(cs[k]["mean_delta"] > 0 for k in bh_img),
            "image_bh_within_noise": sorted("/".join(k[:2]) for k in bh_img if cs[k]["verdict"] == "BELOW"),
            "family_m": len(fam)}


out = {"arch_table": {}, "counts": {}}
for label, v, rep in (("paper_now_defective", V0, False), ("repaired", V, True)):
    cs = cells(v, rep)
    tab = {}
    for m in ARCH:
        for ds in ("baby", "sports", "clothing", "microlens"):
            for s in ("image", "text"):
                c = cs.get((m, ds, s))
                if c:
                    tab[f"{m}/{ds}/{s}"] = {"pct": r2(c["rel_delta_pct"]), "verdict": MARK[c["verdict"]],
                                             "x_F_level": c["x_F_level"], "x_F_paired": c["x_F_paired"],
                                             "p": c["p"], "n": c["n"], "judged": c["judged"]}
    out["arch_table"][label] = tab
    admitted = [d for d, a in v["_meta"]["admitted"].items() if a]
    fams = {"baby+sports only (as in the current paper)": ["baby", "sports"],
            f"registered: all admitted datasets {admitted}": admitted}
    out["counts"][label] = {}
    for fl, dss in fams.items():
        fam = family(cs, dss)
        out["counts"][label][fl] = {"scope baby+sports": counts(cs, ["baby", "sports"], fam),
                                    "scope all in family": counts(cs, dss, fam)}

# preliminary screen (32-cell basis as gen_facts section 21j) and screen vs re-test agreement
EX = json.loads((R / "phase_exact/exact_crossarch.json").read_text())
SG = {c["dataset"]: c for c in json.loads((R / "phase_c3conv_g11/screen_g11.json").read_text())["cells"]}
basis = []
for r in EX:
    if "error" in r:
        continue
    a = (r.get("arms") or {}).get("image_knockout")
    if not a or not r.get("MDE_R@20"):
        continue
    d, f = a["dR@20"], r["MDE_R@20"]
    if r["model"] == "damrs":
        g = SG[r["dataset"]] if r["dataset"] in SG else None
        rep = (g["exact"], g["MDE_R@20"]) if g and g.get("status") == "OK" else None
    else:
        rep = (d, f)
    basis.append({"m": r["model"], "d": r["dataset"], "old_x": abs(d) / f,
                  "rep_x": abs(rep[0]) / rep[1] if rep else None})
n_elec = 2   # FREEDOM and LGMRec Electronics cells (gen_facts adds them from K1+SR; unaffected)
out["screen"] = {"basis_cells": len(basis) + n_elec,
                 "clears_old": sum(c["old_x"] > 1 for c in basis) + 0,
                 "clears_repaired": sum((c["rep_x"] or 0) > 1 for c in basis) + 0,
                 "note": "Electronics cells never clear (FACTS 21j); DAMRS cells from screen_g11.json"}
cs_old, cs_rep = cells(V0, False), cells(V, True)
agree = {}
for lab, cs, key in (("old", cs_old, "old_x"), ("repaired", cs_rep, "rep_x")):
    judg = [c for c in basis if c["d"] in ("baby", "sports") and c["m"] in ARCH and (c[key] is not None)
            and (c["m"], c["d"], "image") in cs]
    agree[lab] = {"judgeable": len(judg),
                  "agree": sum((c[key] > 1) == ((cs[(c["m"], c["d"], "image")]["x_F_level"] or 0) > 1) for c in judg),
                  "disagree": sorted(f"{c['m']}/{c['d']}" for c in judg
                                     if (c[key] > 1) != ((cs[(c["m"], c["d"], "image")]["x_F_level"] or 0) > 1))}
out["screen_vs_retest_baby_sports"] = agree

# averaging vs deletion (tab:subst), band 0.0026
SV = json.loads((R / "phase_exact/screen_vs_exact.json").read_text())["cells"]
band = 0.0026
def tally(cs_):
    return {"cells": len(cs_), "equal": sum(c["screen"] == c["exact"] for c in cs_),
            "flipped": sum(c["screen"] * c["exact"] < 0 for c in cs_),
            "across": sum((abs(c["screen"]) > band) != (abs(c["exact"]) > band) for c in cs_)}
rep_cells = [c for c in SV if c["model"] != "damrs"] + [
    {"model": "damrs", "dataset": d, "screen": g["screen"], "exact": g["exact"]} for d, g in SG.items() if g.get("status") == "OK"]
out["averaging_table"] = {"old_total": tally(SV), "old_damrs": tally([c for c in SV if c["model"] == "damrs"]),
                          "repaired_total": tally(rep_cells),
                          "repaired_damrs": tally([c for c in rep_cells if c["model"] == "damrs"]),
                          "repaired_baby_img_eq_both": SG.get("baby", {}).get("screen_artifact_img_eq_both")}

NV = json.loads((R / "phase_noise/noise_verdicts.json").read_text())
out["noise"] = NV.get("contrasts", NV.get("registered", None)) if isinstance(NV, dict) else None
out["mentor_clothing_sensitivity"] = json.loads((R / "phase_c3conv/sensitivity_mentor_clothing.json").read_text())["cells"]
(R / "phase_paper").mkdir(exist_ok=True)
(R / "phase_paper/final_impact_summary.json").write_text(json.dumps(out, indent=1, default=str))

# digest
for lab in ("paper_now_defective", "repaired"):
    for fl, v in out["counts"][lab].items():
        bs = v["scope baby+sports"]
        print(f"[{lab}] family={fl} m={bs['family_m']}: B+S image beyond {len(bs['image_beyond'])}/{bs['image_cells']} "
              f"{bs['image_beyond']}; text beyond {len(bs['text_beyond'])}/{bs['text_cells']} over "
              f"{len(bs['text_beyond_architectures'])} archs; image BH {bs['image_bh']} (gains {bs['image_bh_gains']}, "
              f"within noise {bs['image_bh_within_noise']}), Holm {bs['image_holm']}")
print("screen:", out["screen"]); print("screen vs re-test:", out["screen_vs_retest_baby_sports"])
print("averaging:", out["averaging_table"])
print("wrote", R / "phase_paper/final_impact_summary.json")
