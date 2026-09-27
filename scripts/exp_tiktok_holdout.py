"""FREEDOM retraining without a modality on TikTok (second short-video dataset), converged protocol.

Registration: results/phase_shortvideo2/PREREG_TIKTOK.md. Same instrument and protocol as the
MicroLens/Amazon runs (scripts/exp_modality_holdout.py: v_feat=None / t_feat=None, patience 100,
cap 3000, seeds 2024-2031); only the dataset differs. TikTok lives under MechInterp/data and is
loaded through the runtime overlay in scripts/recsys_extra_datasets.py, so Recsys stays read-only.

One process may run any subset of (condition, seed); several processes can share the output file
(appends are locked, finished runs are skipped). LightGCN (the behavior-only reference used in the
MicroLens comparison) is trained with --model lightgcn --conditions full.

Outputs -> results/phase_shortvideo2/freedom_p100_tiktok_runs.json (FREEDOM),
           results/phase_shortvideo2/lightgcn_p100_tiktok_runs.json (LightGCN).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/workspace/MechInterp")
sys.path.insert(0, str(ROOT / "scripts"))
import recsys_extra_datasets  # noqa: E402

recsys_extra_datasets.install()
import exp_modality_holdout as holdout  # noqa: E402
import phase0_noisefloor  # noqa: E402

OUT = ROOT / "results" / "phase_shortvideo2"
holdout.OUT = OUT              # train_holdout writes checkpoints/logs under OUT/_scratch
phase0_noisefloor.OUT_DIR = OUT


def _locked_merge(path: Path, record: dict) -> None:
    lock = path.with_suffix(path.suffix + ".lock")
    with open(lock, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        runs = json.loads(path.read_text()) if path.is_file() else []
        runs.append(record)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(runs, indent=2))
        tmp.replace(path)
        fcntl.flock(lf, fcntl.LOCK_UN)


def _done(path: Path) -> set:
    if not path.is_file():
        return set()
    return {(r["condition"], r["seed"]) for r in json.loads(path.read_text()) if "test_result" in r}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="freedom", choices=("freedom", "lightgcn"))
    ap.add_argument("--dataset", default="tiktok")
    ap.add_argument("--conditions", nargs="+", default=list(holdout.CONDITIONS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(2024, 2032)))
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.model}_p{args.patience}_{args.dataset}_runs.json"

    for cond in args.conditions:
        for s in args.seeds:
            if (cond, s) in _done(path):
                print(f"skip {args.model}/{args.dataset}/{cond}/s{s}", flush=True)
                continue
            print(f"\n=== {args.model}/{args.dataset} {cond} seed={s} ===", flush=True)
            t0 = time.time()
            if args.model == "freedom":
                r = holdout.train_holdout("freedom", args.dataset, s, cond, device,
                                          args.patience, args.epochs)
            else:
                assert cond == "full", "LightGCN has no content modality to withhold"
                r = phase0_noisefloor.train_one(
                    "lightgcn", args.dataset, s, device,
                    extra_overrides={"stopping_step": args.patience, "epochs": args.epochs},
                    run_prefix=f"pat{args.patience}e{args.epochs}")
                r["condition"] = "full"
            r["stopping_step"], r["epochs_cap"] = args.patience, args.epochs
            r["hit_cap"] = bool(r["best_epoch"] >= 0.95 * args.epochs)
            _locked_merge(path, r)
            print(f"  R@20={r['test_result']['Recall@20']:.5f} best_epoch={r['best_epoch']} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
