"""Step 3 — the PAIRED noise floor and the pre-registered verdict.

PREREG s.3 requires TWO floors and demands they agree:
    F_paired = 2 * sd_s[ d_x(s) ]      per-seed paired delta   (tight; the correct floor,
                                        since a structural knockout is deterministic
                                        WITHIN a checkpoint, so seed-to-seed variation in
                                        the DELTA is the quantity that matters)
    F_level  = 2 * sd_s[ R@20(s) ]     the group's published convention (Exp A1)
Verdict: SIGNIFICANT iff |mean d| > max(F_paired, F_level); NULL iff below both; else
MARGINAL -- reported as MARGINAL, never rounded.

F_paired had no implementation anywhere in the repo (caught by the 2026-09-04 audit,
ADDENDUM 1 s.A1.4): exp_mde_perdataset.py computes only F_level, so the floor run alone
would have delivered half the decision rule and invited single-floor shopping.

Admissibility gate (PREREG s.3): the cell yields a verdict only if the TEXT knockout clears
both floors. Its known weakness is stated in the output: text carries 0.9 of FREEDOM's
mm_adj by construction, so the informative quantity is the RATIO |d_txt|/|d_img|.

Outputs -> results/phase_micro/{knockout_perseed,significance}_microlens.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path("/workspace/MechInterp")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "interp"))
from phase1_knockout import run_model  # noqa: E402

OUT = ROOT / "results" / "phase_micro"
MDE_DIR = ROOT / "results" / "phase_mde"
SCRATCH_CKPT = ROOT / "results" / "_scratch" / "ckpts"


def seed_checkpoints(model: str, dataset: str, tag: str) -> list:
    """Prefer the ckpt_path recorded by train_one; fall back to its deterministic name."""
    runs_f = MDE_DIR / f"{tag}_seed_runs.json"
    out = []
    if runs_f.is_file():
        for r in json.loads(runs_f.read_text()):
            if r.get("model") != model or r.get("dataset") != dataset or "test_result" not in r:
                continue
            p = r.get("ckpt_path") or str(SCRATCH_CKPT / f"nf_{model}_{dataset}_s{r['seed']}.pt")
            if Path(p).is_file():
                out.append({"seed": r["seed"], "ckpt": p,
                            "R@20": float(r["test_result"]["Recall@20"]),
                            "N@20": float(r["test_result"].get("NDCG@20", float("nan"))),
                            "best_epoch": r.get("best_epoch")})
    return sorted(out, key=lambda x: x["seed"])


def verdict(mean_d: float, f_paired: float, f_level: float) -> str:
    hi, lo = max(f_paired, f_level), min(f_paired, f_level)
    a = abs(mean_d)
    if a > hi:
        return "SIGNIFICANT"
    if a < lo:
        return "NULL"
    return "MARGINAL"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="freedom")
    ap.add_argument("--dataset", default="microlens")
    ap.add_argument("--tag", default="microlens")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    ck = seed_checkpoints(args.model, args.dataset, args.tag)
    if len(ck) < 2:
        print(f"need >=2 completed seed runs, found {len(ck)} (tag={args.tag}). "
              f"Is results/phase_mde/{args.tag}_seed_runs.json populated?")
        return 1
    print(f"{len(ck)} seed checkpoints for {args.model}/{args.dataset}: "
          f"{[c['seed'] for c in ck]}", flush=True)

    per = []
    for c in ck:
        r = run_model(args.model, args.dataset, device, ckpt_path=c["ckpt"])
        base = r["variants"]["baseline"]["metrics"]["Recall@20"]
        row = {"seed": c["seed"], "ckpt": Path(c["ckpt"]).name,
               "R@20_knockout_baseline": float(base),
               "R@20_logged_by_trainer": c["R@20"],
               "best_epoch": c["best_epoch"]}
        for k in ("image_knockout", "text_knockout", "both_knockout"):
            if k in r["variants"]:
                row[f"d_{k.split('_')[0]}"] = r["variants"][k]["dR@20"]
                row[f"dN_{k.split('_')[0]}"] = r["variants"][k]["dN@20"]
        per.append(row)
        print(f"  seed {c['seed']}: base={base:.6f} d_img={row.get('d_image'):+.6f} "
              f"d_txt={row.get('d_text'):+.6f}", flush=True)
        (OUT / f"knockout_perseed_{args.model}_{args.dataset}.json").write_text(json.dumps(per, indent=2))

    levels = [p["R@20_knockout_baseline"] for p in per]
    f_level = 2 * statistics.stdev(levels)
    res = {"model": args.model, "dataset": args.dataset, "n_seeds": len(per),
           "seeds": [p["seed"] for p in per],
           "F_level": f_level, "sd_R@20": statistics.stdev(levels),
           "mean_R@20": statistics.mean(levels),
           "note": ("F_level = 2*sd(R@20) across seeds (published convention). "
                    "F_paired = 2*sd(per-seed delta) -- the correct floor for a knockout "
                    "that is deterministic within a checkpoint. PREREG s.3 requires the "
                    "verdict to hold against max(F_paired, F_level)."),
           "streams": {}}
    for s in ("image", "text", "both"):
        ds = [p[f"d_{s}"] for p in per if f"d_{s}" in p]
        if len(ds) < 2:
            continue
        m, sd = statistics.mean(ds), statistics.stdev(ds)
        f_paired = 2 * sd
        res["streams"][s] = {
            "mean_delta": m, "sd_delta": sd, "F_paired": f_paired,
            "abs_mean_over_F_paired": abs(m) / f_paired if f_paired else None,
            "abs_mean_over_F_level": abs(m) / f_level if f_level else None,
            "verdict": verdict(m, f_paired, f_level),
            "per_seed": ds,
        }
    img, txt = res["streams"].get("image"), res["streams"].get("text")
    if img and txt:
        res["admissibility_gate"] = {
            "rule": "cell yields a verdict only if the TEXT knockout clears both floors",
            "text_verdict": txt["verdict"],
            "passes": txt["verdict"] == "SIGNIFICANT",
            "caveat": ("text carries 0.9 of FREEDOM's mm_adj by construction, so a passing "
                       "gate is partly mechanical; the informative quantity is the ratio"),
            "abs_ratio_txt_over_img": abs(txt["mean_delta"]) / abs(img["mean_delta"]),
        }
    (OUT / f"significance_{args.model}_{args.dataset}.json").write_text(json.dumps(res, indent=2))

    print(f"\n=== VERDICT ({args.model}/{args.dataset}, {len(per)} seeds) ===")
    print(f"  F_level  = {f_level:.6f}   (2*sd of R@20 across seeds)")
    for s, v in res["streams"].items():
        print(f"  {s:6s} mean d={v['mean_delta']:+.6f}  F_paired={v['F_paired']:.6f}  "
              f"|d|/F_paired={v['abs_mean_over_F_paired']:.2f}  "
              f"|d|/F_level={v['abs_mean_over_F_level']:.2f}  -> {v['verdict']}")
    if "admissibility_gate" in res:
        g = res["admissibility_gate"]
        print(f"  admissibility: text={g['text_verdict']} passes={g['passes']} "
              f"| ratio |d_txt|/|d_img| = {g['abs_ratio_txt_over_img']:.2f}")
    print(f"\nWrote {OUT/f'significance_{args.dataset}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
