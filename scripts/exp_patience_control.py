"""Control: is LATTICE's 50% training collapse a property of the MODEL, or of our
early-stopping patience?

Ten seeds of LATTICE/baby under the published config give a bimodal outcome: 5 runs peak at
epoch 2-3 and stop (R@20 ~0.068), 5 train to epoch 133-183 (R@20 ~0.086-0.089). Before
reporting "LATTICE fails to train on half its seeds" we must rule out the alternative that
`stopping_step: 20` simply fires before LATTICE's slow start recovers.

This re-runs the COLLAPSED seeds with a large patience. If they recover, the collapse is a
harness artifact and the floor must be re-measured with the larger patience. If they still
collapse, it is the model.

Outputs -> results/phase_mde/patience_control.json
"""
from __future__ import annotations

import argparse, json, statistics, sys, time
from pathlib import Path
import torch

ROOT = Path("/workspace/MechInterp")
sys.path.insert(0, str(ROOT / "scripts"))
from phase0_noisefloor import train_one  # noqa: E402

OUT = ROOT / "results" / "phase_mde"


def _locked_merge(path: Path, record: dict) -> list:
    """Append one record under an exclusive lock, RE-READING the file inside the lock.

    The previous read-once/write-whole pattern let two concurrent jobs each hold a stale copy
    and overwrite the other's appends; on 2026-09-06 that silently lost 10 completed runs
    (all of lattice/clothing, mentor/clothing and the vbpr/baby re-measurement)."""
    import fcntl
    lock = path.with_suffix(path.suffix + ".lock")
    with open(lock, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        runs = json.loads(path.read_text()) if path.is_file() else []
        runs.append(record)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(runs, indent=2))
        tmp.replace(path)
        fcntl.flock(lf, fcntl.LOCK_UN)
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lattice")
    ap.add_argument("--dataset", default="baby")
    ap.add_argument("--seeds", nargs="+", type=int, default=[2025, 2026, 2027])
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the 1000-epoch cap; several runs at patience 100 peaked at "
                         "epoch 977-999, i.e. were still improving when the cap stopped them")
    ap.add_argument("--out", default=None, help="results JSON (default phase_mde/patience_control.json)")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    f = Path(args.out) if args.out else OUT / "patience_control.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    runs = json.loads(f.read_text()) if f.is_file() else []
    done = {(r["model"], r["dataset"], r["seed"], r["patience"], r.get("epochs_cap")) for r in runs if "test_result" in r}

    for s in args.seeds:
        if (args.model, args.dataset, s, args.patience, args.epochs) in done:
            print(f"skip {args.model}/{args.dataset}/s{s}/p{args.patience}"); continue
        print(f"\n=== {args.model}/{args.dataset} seed={s} patience={args.patience} ===", flush=True)
        t0 = time.time()
        try:
            r = train_one(args.model, args.dataset, s, device,
                          extra_overrides=({"stopping_step": args.patience} if args.epochs is None
                                           else {"stopping_step": args.patience, "epochs": args.epochs}),
                          run_prefix=f"pat{args.patience}" + (f"e{args.epochs}" if args.epochs else ""))
            r["patience"] = args.patience
            r["epochs_cap"] = args.epochs
            r["hit_cap"] = bool(args.epochs and r["best_epoch"] >= 0.95 * args.epochs)
            runs = _locked_merge(f, r)
            print(f"  R@20={r['test_result']['Recall@20']:.5f} best_epoch={r['best_epoch']} "
                  f"({r['train_min']:.1f} min)", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            runs = _locked_merge(f, {"model": args.model, "dataset": args.dataset, "seed": s,
                                     "patience": args.patience, "error": repr(e)})
    print(f"\nWrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
