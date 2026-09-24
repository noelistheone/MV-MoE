"""The paper's core experiment, redone on CONVERGED FREEDOM checkpoints.

The single-checkpoint knockout table was computed on checkpoints trained with the harness default
stopping_step=20. The 2026-09-19/21 convergence re-run showed FREEDOM itself is 1.9-2.6%
undertrained under that setting, so the knockout (and the floor it is judged against) must be
re-measured on models trained to convergence.

The holdout's `full` condition at patience 100 / cap 3000 is exactly that: eight independently
seeded, converged FREEDOM models per dataset. For each seed we run the same exact structural
knockout and report the paired statistics:
    F_level  = 2 sd of R@20 across seeds          (the paper's convention)
    F_paired = 2 sd of the per-seed paired delta  (tight; the knockout is deterministic
                                                   within a checkpoint)
Outputs -> results/phase_convergence/converged_knockout.json
"""
from __future__ import annotations

import json, statistics, sys
from pathlib import Path
import torch

ROOT = Path("/workspace/MechInterp")
sys.path.insert(0, "/workspace/Recsys")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "interp"))
from phase1_knockout import run_model  # noqa: E402

CK = ROOT / "results" / "phase_holdout" / "_scratch" / "ckpts"
OUT = ROOT / "results" / "phase_convergence" / "converged_knockout.json"


def main() -> int:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    res = json.loads(OUT.read_text()) if OUT.is_file() else {}
    for ds in ["baby", "sports", "clothing", "microlens"]:
        cks = sorted(CK.glob(f"hold_p100e3000_freedom_{ds}_full_s*.pt"))
        if len(cks) < 2:
            print(f"skip {ds}: {len(cks)} converged checkpoints"); continue
        # STALENESS GUARD (added 2026-09-21). The glob above has no completion check, so a
        # checkpoint still being written is picked up as if it were converged. That happened:
        # microlens seed 2028 was scored at base_R20=0.0970329 while the finished run logged
        # 0.1002106, and because one low baseline widens the spread it inflated F_level 3.62x
        # (0.003421 vs 0.000946 over the four clean seeds) -- which moved the image level ratio
        # from 0.98x (one seed short of SIGNIFICANT) to a comfortable 0.28x. The error ran in the
        # paper's favour and `if seed in have: continue` meant no re-run could ever repair it.
        logged = {}
        runs_f = ROOT / "results" / "phase_holdout" / f"freedom_p100_{ds}_runs.json"
        if runs_f.is_file():
            for r in json.loads(runs_f.read_text()):
                if r.get("condition") == "full" and "test_result" in r:
                    logged[r["seed"]] = r["test_result"]["Recall@20"]
        rows = res.get(ds, {}).get("per_seed", [])
        stale = [r["seed"] for r in rows
                 if r["seed"] in logged and abs(r["base_R20"] - logged[r["seed"]]) > 1e-9]
        if stale:
            print(f"  {ds}: DROPPING stale rows {stale} (baseline != the finished run's logged R@20)")
            rows = [r for r in rows if r["seed"] not in stale]
        have = {r["seed"] for r in rows}
        for ck in cks:
            seed = int(ck.stem.split("_s")[-1])
            if seed in have:
                continue
            if seed not in logged:
                print(f"  {ds} s{seed}: SKIP, no finished run logged yet (checkpoint may be mid-write)")
                continue
            r = run_model("freedom", ds, device, ckpt_path=str(ck))
            v = r["variants"]
            rows.append({"seed": seed, "ckpt": ck.name,
                         "base_R20": v["baseline"]["metrics"]["Recall@20"],
                         "d_image": v["image_knockout"]["dR@20"],
                         "d_text": v["text_knockout"]["dR@20"],
                         "d_both": v["both_knockout"]["dR@20"],
                         "dN_image": v["image_knockout"]["dN@20"],
                         "rbo_image": v["image_knockout"]["ranking_change"]["rbo"],
                         "rbo_text": v["text_knockout"]["ranking_change"]["rbo"],
                         "recon_max_err": r["recon_max_err"],
                         "R@20_logged_by_trainer": logged[seed]})
            got, exp = rows[-1]["base_R20"], logged[seed]
            if abs(got - exp) > 1e-9:
                rows.pop()
                print(f"  {ds} s{seed}: REJECTED, recomputed {got:.8f} != logged {exp:.8f}")
                continue
            print(f"  {ds} s{seed}: base={rows[-1]['base_R20']:.5f} "
                  f"d_img={rows[-1]['d_image']:+.6f} d_txt={rows[-1]['d_text']:+.6f}", flush=True)
        rows.sort(key=lambda x: x["seed"])
        base = [r["base_R20"] for r in rows]
        cell = {"n_seeds": len(rows), "mean_R20": statistics.mean(base),
                "F_level": 2 * statistics.stdev(base) if len(base) > 1 else None,
                "per_seed": rows, "streams": {}}
        for s in ["image", "text", "both"]:
            dd = [r[f"d_{s}"] for r in rows]
            if len(dd) < 2: continue
            m, sd = statistics.mean(dd), statistics.stdev(dd)
            fp, fl = 2 * sd, cell["F_level"]
            hi, lo = max(fp, fl), min(fp, fl)
            cell["streams"][s] = {"mean_delta": m, "sd": sd, "F_paired": fp,
                                  "x_F_paired": abs(m) / fp if fp else None,
                                  "x_F_level": abs(m) / fl if fl else None,
                                  "verdict": ("SIGNIFICANT" if abs(m) > hi else
                                              "NULL" if abs(m) < lo else "MARGINAL"),
                                  "n_negative": sum(x < 0 for x in dd)}
        res[ds] = cell
        OUT.write_text(json.dumps(res, indent=2))
        print(f"{ds}: n={cell['n_seeds']} mean={cell['mean_R20']:.5f} F_level={cell['F_level']:.5f}")
        for s, v in cell["streams"].items():
            print(f"   {s:6s} mean d={v['mean_delta']:+.6f}  xFp={v['x_F_paired']:.2f} "
                  f"xFl={v['x_F_level']:.2f} -> {v['verdict']}  ({v['n_negative']}/{cell['n_seeds']} neg)")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
