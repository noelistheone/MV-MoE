"""Per-(model,dataset) noise floor (MDE) — closes the "MDE reuse" gap.

The project previously measured MDE=2*std(R@20) only on Baby (FREEDOM+LightGCN, 5
seeds) and reused it as the significance threshold for Sports/Clothing/Electronics
AND for LGMRec. This driver measures each model's OWN per-dataset MDE so every
"image is used / falls within noise" claim becomes a formal significance statement.

It REUSES phase0_noisefloor.train_one verbatim (same Trainer, configs, determinism,
checkpoint format) and writes ONLY to results/phase_mde/ — it never touches the
existing results/phase0/seed_variance*.json.

Resumable: skips (model,dataset,seed) triples already present in the runs file.

Run (background):
    conda run -n mechinterp python MechInterp/scripts/exp_mde_perdataset.py \
        --models lgmrec --datasets baby sports clothing --seeds 2024 2025 2026 --tag lgmrec
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/workspace/MechInterp")
sys.path.insert(0, str(ROOT / "scripts"))
from phase0_noisefloor import train_one  # noqa: E402  reuse exact training fn

OUT = ROOT / "results" / "phase_mde"


def summarize(runs: list) -> dict:
    summary: dict = {}
    models = sorted({r["model"] for r in runs if "test_result" in r})
    for m in models:
        dss = sorted({r["dataset"] for r in runs if r.get("model") == m and "test_result" in r})
        for ds in dss:
            rs = [r for r in runs if r.get("model") == m and r.get("dataset") == ds and "test_result" in r]
            for metric in ("Recall@20", "NDCG@20", "Recall@10", "NDCG@10"):
                vals = [r["test_result"][metric] for r in rs if metric in r["test_result"]]
                if len(vals) >= 2:
                    std = statistics.stdev(vals)
                    summary.setdefault(m, {}).setdefault(ds, {})[metric] = {
                        "n": len(vals), "mean": statistics.mean(vals), "std": std,
                        "MDE_2std": 2 * std, "values": vals,
                    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lgmrec"])
    ap.add_argument("--datasets", nargs="+", default=["baby", "sports", "clothing"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025, 2026])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="lgmrec")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    runs_path = OUT / f"{args.tag}_seed_runs.json"
    runs = json.loads(runs_path.read_text()) if runs_path.exists() else []
    done = {(r["model"], r["dataset"], r["seed"]) for r in runs if "test_result" in r}

    t_start = time.time()
    for m in args.models:
        for ds in args.datasets:
            for s in args.seeds:
                if (m, ds, s) in done:
                    print(f"skip {m}/{ds}/{s} (already done)", flush=True)
                    continue
                print(f"\n=== train {m}/{ds} seed={s} | elapsed {(time.time()-t_start)/60:.1f}m ===", flush=True)
                try:
                    r = train_one(m, ds, s, device)
                    runs.append(r)
                    print(f"  R@20={r['test_result'].get('Recall@20'):.4f} "
                          f"N@20={r['test_result'].get('NDCG@20'):.4f} "
                          f"(epoch {r['best_epoch']}, {r['train_min']:.1f} min)", flush=True)
                except Exception as e:  # noqa: BLE001
                    import traceback; traceback.print_exc()
                    runs.append({"model": m, "dataset": ds, "seed": s, "error": repr(e)})
                runs_path.write_text(json.dumps(runs, indent=2))  # checkpoint each step
                # refresh summary each step too, so it is always queryable
                (OUT / f"{args.tag}_mde.json").write_text(json.dumps(
                    {"summary": summarize(runs), "runs": runs,
                     "note": "MDE_2std=2*std(metric across seeds); per (model,dataset)."}, indent=2))

    summary = summarize(runs)
    print("\n=== per-(model,dataset) NOISE FLOOR (MDE = 2*std of R@20) ===")
    for m, dd in summary.items():
        for ds, mm in dd.items():
            r20 = mm.get("Recall@20", {})
            print(f"  {m}/{ds}: R@20 mean={r20.get('mean',0):.4f} std={r20.get('std',0):.5f} "
                  f"MDE={r20.get('MDE_2std',0):.5f} n={r20.get('n')}")
    print(f"Wrote {OUT/(args.tag+'_mde.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
