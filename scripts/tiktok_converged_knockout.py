"""Exact FREEDOM image/text term deletion on every converged TikTok full-condition checkpoint.

Same instrument as scripts/exp_converged_knockout.py (results/phase_convergence/
converged_knockout.json): phase1_knockout.run_model("freedom", ...) rebuilds the image and text kNN
graphs, decomposes the fused item embedding as cf + h_img + h_txt (reconstruction asserted
< 1e-3 against the model's own propagation, max error recorded), and evaluates the model with h_img,
h_txt or both deleted. Guards carried over unchanged:
  * a checkpoint is used only once its training run is LOGGED as finished in the runs file
    (a checkpoint on disk without a record may be mid-write);
  * STALENESS GUARD: the recomputed baseline R@20 must equal the trainer-logged R@20 to 1e-9,
    otherwise the row is rejected; stored rows whose baseline no longer matches are dropped;
  * the checkpoint file name must be the one the logged run names (gen_facts s.21a check).
Differences from exp_converged_knockout.py, all forced by where TikTok lives: the dataset overlay
is installed first, checkpoints/runs come from results/phase_shortvideo2, and the runs file is
filtered to the converged protocol (stopping_step 100, epochs_cap 3000) as gen_facts does.

The per-dataset cell (n_seeds, mean_R20, F_level, per_seed, streams) is computed by
tiktok_common.ck_cell_float, which reproduces the stored converged_knockout.json cells exactly
(checked by tiktok_verdicts.py's self-test); the paper-grade numbers (t, p, CI, verdict with the
two-floor rule) come from gen_facts' exact-arith code via tiktok_common.deletion_exact and are
stored under "exact" together with the s.21a-format table rows.

Incremental and resumable: finished seeds are kept, new ones appended, file rewritten per seed.
Output -> results/phase_shortvideo2/tiktok_knockout.json

--verify-microlens SEED additionally re-runs the instrument on that MicroLens converged checkpoint
and compares the row with the stored results/phase_convergence/converged_knockout.json row (a check
that the overlay/wrapper does not perturb the instrument; stored under "instrument_check").
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/workspace/MechInterp")
sys.path.insert(0, str(ROOT / "scripts"))
import tiktok_common as tc                   # noqa: E402

tc.install_overlay()                         # before any Config is constructed
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "interp"))
from phase1_knockout import run_model        # noqa: E402

OUT = tc.SV2 / "tiktok_knockout.json"
DS = "tiktok"


def row_from(r: dict, seed: int, ck: Path, logged: float) -> dict:
    """Identical fields to exp_converged_knockout.py's per-seed row."""
    v = r["variants"]
    return {"seed": seed, "ckpt": ck.name,
            "base_R20": v["baseline"]["metrics"]["Recall@20"],
            "d_image": v["image_knockout"]["dR@20"],
            "d_text": v["text_knockout"]["dR@20"],
            "d_both": v["both_knockout"]["dR@20"],
            "dN_image": v["image_knockout"]["dN@20"],
            "rbo_image": v["image_knockout"]["ranking_change"]["rbo"],
            "rbo_text": v["text_knockout"]["ranking_change"]["rbo"],
            "recon_max_err": r["recon_max_err"],
            "R@20_logged_by_trainer": logged}


def extra_from(r: dict) -> dict:
    """Additional metrics (not in the CK row form) kept alongside it for the paper's other columns."""
    v = r["variants"]
    o = {"baseline": v["baseline"]["metrics"]}
    for k in ("image_knockout", "text_knockout", "both_knockout"):
        o[k] = {"delta": v[k]["delta"], "rbo": v[k]["ranking_change"].get("rbo"),
                "rbo_n_sampled": v[k]["ranking_change"].get("n_sampled")}
    o["attribution"] = r.get("attribution")
    return o


def verify_microlens(seed: int, device: str) -> dict:
    ck = ROOT / "results" / "phase_holdout" / "_scratch" / "ckpts" / f"hold_p100e3000_freedom_microlens_full_s{seed}.pt"
    stored = json.loads((ROOT / "results" / "phase_convergence" / "converged_knockout.json").read_text())
    srow = next(r for r in stored["microlens"]["per_seed"] if r["seed"] == seed)
    t0 = time.time()
    r = run_model("freedom", "microlens", device, ckpt_path=str(ck))
    new = row_from(r, seed, ck, srow.get("R@20_logged_by_trainer"))
    diffs = {k: abs(new[k] - srow[k]) for k in ("base_R20", "d_image", "d_text", "d_both", "dN_image",
                                                   "rbo_image", "rbo_text") if k in srow}
    return {"dataset": "microlens", "seed": seed, "stored_row": srow, "recomputed_row": new,
            "abs_diffs": diffs, "max_abs_diff": max(diffs.values()), "seconds": time.time() - t0,
            "peak_gpu_mem_MiB": (torch.cuda.max_memory_allocated() / 2**20) if device.startswith("cuda") else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="run on CPU (the staleness guard may then reject rows "
                    "if CPU and GPU evaluation differ; the trainer evaluated on GPU)")
    ap.add_argument("--verify-microlens", type=int, default=None, metavar="SEED")
    args = ap.parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}"
    res = json.loads(OUT.read_text()) if OUT.is_file() else {}
    res.pop("status", None)

    pooled = tc.pool_runs([tc.TIKTOK_RUNS])
    logged = {k[2]: r for k, r in pooled.items() if k[0] == DS and k[1] == "full"}
    rows = (res.get(DS) or {}).get("per_seed", [])
    extras = res.get("per_seed_extra", {})
    stale = [r["seed"] for r in rows
             if r["seed"] not in logged
             or abs(r["base_R20"] - logged[r["seed"]]["test_result"]["Recall@20"]) > 1e-9
             or Path(logged[r["seed"]]["ckpt_path"]).name != r["ckpt"]]
    if stale:
        print(f"  {DS}: DROPPING stale rows {stale} (baseline or checkpoint != the finished run's record)")
        rows = [r for r in rows if r["seed"] not in stale]
    have = {r["seed"] for r in rows}
    rejected = res.get("rejected", {})
    cks = sorted(tc.CKPT_DIR.glob(f"hold_p{tc.PATIENCE}e{tc.CAP}_freedom_{DS}_full_s*.pt"))
    skipped = []
    for ck in cks:
        seed = int(ck.stem.split("_s")[-1])
        if seed in have:
            continue
        if seed not in logged:
            print(f"  {DS} s{seed}: SKIP, no finished run logged yet (checkpoint may be mid-write)")
            skipped.append(seed)
            continue
        if Path(logged[seed]["ckpt_path"]).name != ck.name:
            print(f"  {DS} s{seed}: SKIP, logged run names {logged[seed]['ckpt_path']}")
            skipped.append(seed)
            continue
        exp = logged[seed]["test_result"]["Recall@20"]
        r = run_model("freedom", DS, device, ckpt_path=str(ck))
        row = row_from(r, seed, ck, exp)
        if abs(row["base_R20"] - exp) > 1e-9:
            rejected[str(seed)] = {"recomputed": row["base_R20"], "logged": exp}
            print(f"  {DS} s{seed}: REJECTED, recomputed {row['base_R20']:.8f} != logged {exp:.8f}")
            continue
        rejected.pop(str(seed), None)
        rows.append(row)
        extras[str(seed)] = extra_from(r)
        print(f"  {DS} s{seed}: base={row['base_R20']:.5f} d_img={row['d_image']:+.6f} "
              f"d_txt={row['d_text']:+.6f} recon={row['recon_max_err']:.1e}", flush=True)
        res[DS] = tc.ck_cell_float(rows)
        res["per_seed_extra"] = extras
        OUT.write_text(json.dumps(res, indent=2, default=str))

    rows.sort(key=lambda x: x["seed"])
    res[DS] = tc.ck_cell_float(rows)
    res["per_seed_extra"] = extras
    res["rejected"] = rejected
    res["skipped_unfinished"] = skipped
    res["device"] = device
    res["gen_facts_exact_arith_sha256"] = tc.exact_arith()["_block_sha256"]
    if len(rows) >= 2:
        ex = tc.deletion_exact(rows)
        res["exact"] = {st: tc.jsonable(v) for st, v in ex.items()}
        res["facts_rows"] = [tc.deletion_row(DS, st, ex[st], tag="TK") for st in ("image", "text", "both")]
        ns = tc.exact_arith()
        res["recon_max_err_over_seeds"] = max(r["recon_max_err"] for r in rows)
        res["level_floor_pct_of_mean_R20"] = float(100 * ex["image"]["fl"] / ex["image"]["mu"])
        res["status"] = f"{len(rows)} seeds" + ("" if len(rows) == 8 else " (INCOMPLETE: registered n = 8)")
        _ = ns
    else:
        res["exact"], res["facts_rows"] = None, []
        res["status"] = f"{len(rows)} seed(s): statistics need >= 2"
    if args.verify_microlens is not None:
        res["instrument_check"] = verify_microlens(args.verify_microlens, device)
        print(f"  microlens s{args.verify_microlens} re-run vs stored row: max |diff| "
              f"{res['instrument_check']['max_abs_diff']:.2e}")
    OUT.write_text(json.dumps(res, indent=2, default=str))
    c = res[DS]
    print(f"{DS}: n={c['n_seeds']} mean={c['mean_R20']:.5f} F_level={c['F_level']}")
    for s, v in c["streams"].items():
        print(f"   {s:6s} mean d={v['mean_delta']:+.6f}  xFp={v['x_F_paired']:.2f} "
              f"xFl={v['x_F_level']:.2f} -> {v['verdict']}  ({v['n_negative']}/{c['n_seeds']} neg)")
    for line in res["facts_rows"]:
        print(line)
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
