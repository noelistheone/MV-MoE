"""Checks of scripts/exp_c3conv_score.py (read-only on every existing file).

  --equivalence   re-score a few seed-2024 checkpoints through the ORIGINAL drivers
                  (exp_exact_crossarch.run_exact; phase1_knockout.run_model for LGMRec) and
                  compare dR@20 / dN@20 of image_knockout / text_knockout with scored_<ds>.json.
  --lattice-probe LATTICE/baby s2024: evaluate the model's own _propagate() with modal_weight
                  shifted by small amounts, to test whether the trainer-logged R@20 (which used
                  the last epoch's cached item graph) lies inside the range such shifts produce.
Output -> results/phase_c3conv/scorer_checks.json (merged by key).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
ROOT = Path("/workspace/MechInterp")
for _p in ("/workspace/Recsys", str(ROOT / "src/models"), str(ROOT / "src/interp"), str(ROOT / "scripts")):
    sys.path.insert(0, _p)
import torch
OUT = ROOT / "results/phase_c3conv/scorer_checks.json"
CK = ROOT / "results/phase0/_scratch/ckpts"


def equivalence(device, pairs):
    from exp_exact_crossarch import run_exact
    from phase1_knockout import run_model
    res = {}
    for m, ds in pairs:
        sc = json.loads((ROOT / f"results/phase_c3conv/scored_{ds}.json").read_text())["rows"][f"{m}/s2024"]
        # PREREG_C3CONV_AMEND2.md: damrs_g11 = DAMRS re-trained on the repaired port (g11 prefix);
        # the legacy damrs checkpoints cannot be rebuilt by the repaired model file.
        arch, prefix = ("damrs", "g11pat100e3000") if m == "damrs_g11" else (m, "pat100e3000")
        ck = str(CK / f"{prefix}_{arch}_{ds}_s2024.pt")
        t0 = time.time()
        r = run_model(arch, ds, device, ckpt_path=ck) if arch == "lgmrec" else run_exact(arch, ds, device, ck)
        arms = r["variants"] if m == "lgmrec" else r["arms"]
        base = arms["baseline"]["metrics"]["Recall@20"]
        row = {"orig_driver": "phase1_knockout.run_model" if m == "lgmrec" else "exp_exact_crossarch.run_exact",
               "base_R20_orig": base, "base_R20_scorer": sc["baseline"]["Recall@20"], "seconds": time.time() - t0}
        worst = abs(base - sc["baseline"]["Recall@20"])
        for s, arm in (("image", "image_knockout"), ("text", "text_knockout")):
            for key, met in (("dR@20", "Recall@20"), ("dN@20", "NDCG@20")):
                a, b = arms[arm][key], sc["arms"][s]["delta"][met]
                row[f"{s}_{key}"] = [a, b]
                worst = max(worst, abs(a - b))
        row["max_abs_diff"] = worst
        res[f"{m}/{ds}"] = row
        print(m, ds, "max|orig - scorer| =", worst, flush=True)
        torch.cuda.empty_cache()
    return res


@torch.no_grad()
def lattice_probe(device):
    from recsys_bridge import load_frozen
    from ranking_effects import evaluate_item_matrix
    tr = [r for r in json.loads((ROOT / "results/phase_convergence/table_rerun.json").read_text())
          if r["model"] == "lattice" and r["dataset"] == "baby"][0]
    logged = tr["test_result"]["Recall@20"]
    _, _, model, loader = load_frozen("lattice", "baby", device, ckpt_path=tr["ckpt_path"])
    w0 = model.modal_weight.detach().clone()
    scan = []
    for k in range(-25, 26):
        dx = k * 2e-4
        model.modal_weight.data = w0 + torch.tensor([dx, -dx], device=w0.device) / 2
        model._build_item_graph = True
        u, i = model._propagate()
        m, _, _ = evaluate_item_matrix(u, i, loader, device)
        scan.append({"dlogit": dx, "R@20": float(m["Recall@20"])})
    vals = [x["R@20"] for x in scan]
    return {"logged_R20": logged, "ckpt_modal_weight": w0.tolist(),
            "rebuilt_R20_at_ckpt_weights": scan[25]["R@20"], "scan": scan,
            "scan_min": min(vals), "scan_max": max(vals),
            "logged_inside_scan_range": min(vals) <= logged <= max(vals),
            "exact_match_in_scan": any(abs(v - logged) < 1e-12 for v in vals),
            "note": "modal_weight is LATTICE's only trained graph parameter (image_trs/text_trs are frozen, F5); "
                    "the trainer's test used the item graph cached at the start of the LAST epoch. A bracket, not a proof."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equivalence", action="store_true")
    ap.add_argument("--lattice-probe", action="store_true")
    ap.add_argument("--pairs", nargs="+", default=["vbpr/baby", "mentor/baby", "mgcn/baby", "smore/baby",
                                                    "gume/baby", "lgmrec/baby", "damrs_g11/baby", "mmgcn/baby"])
    a = ap.parse_args()
    device = "cuda:0"
    tot = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(3.3 / tot, 0)
    doc = json.loads(OUT.read_text()) if OUT.is_file() else {}
    if a.equivalence:
        # merge by pair, so a run over a subset of pairs keeps the other pairs' earlier results
        doc.setdefault("equivalence_vs_original_drivers", {}).update(
            equivalence(device, [tuple(p.split("/")) for p in a.pairs]))
    if a.lattice_probe:
        doc["lattice_baby_s2024_graph_state_probe"] = lattice_probe(device)
    tmp = OUT.with_suffix(".json.tmp"); tmp.write_text(json.dumps(doc, indent=1)); tmp.replace(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
