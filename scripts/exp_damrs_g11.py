"""DAMRS re-training with the repaired port (results/phase_c3conv/PREREG_C3CONV_AMEND2.md).

The DAMRS port's user-item adjacency was filled with `dict.update(A, ...)` on a scipy dok_matrix. In
the installed scipy (1.17.1) dok_matrix keeps its entries in a private `_dict`, so that call writes
into the unused dict base and leaves the matrix empty: the user-item propagation graph was empty in every
DAMRS run whose process started before the upstream repair (model file mtime 2026-09-25 20:40:12
-0700). This wrapper re-trains DAMRS with the repaired file through the SAME trainer (train_one),
config and protocols as the original runs, and records code provenance per run:

  --protocol converged   patience 100, cap 3000 (the C3 re-test)      -> runs_<ds>.json
  --protocol default     patience 20,  cap 1000 (the screen + floors) -> screen_runs_<ds>.json

Checkpoints get NEW prefixes (g11pat100e3000_ / g11nf_), so no original checkpoint is overwritten,
and output goes to results/phase_c3conv_g11/ only. A run counts only if the model file's sha256 at
start and at end both equal the hash registered in AMEND2 (checked again by the scorer).
"""
import argparse, hashlib, json, platform, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path("/workspace/MechInterp")
sys.path.insert(0, str(ROOT / "scripts"))
MODEL_FILE = Path("/workspace/Recsys/src/models/damrs.py")
REGISTERED_SHA256 = "d001139f3cbafff1eac8e3a1c82019cbc96deb4fec7db56ae01bc51731a4f5ed"
# Hashes a run may start from: the registered file (a code release may add the hash of its own
# copy of that file, which differs only in scrubbed comments and paths).
ACCEPTED_SHA256 = (REGISTERED_SHA256, "b0bd3670c78d3f90a2deec378f0182880876a9fdbb0994f447bb76a7ce690e10")  # + this release's scrubbed copy
OUT = ROOT / "results" / "phase_c3conv_g11"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--protocol", choices=["converged", "default"], required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    h0 = sha256(MODEL_FILE)
    if h0 not in ACCEPTED_SHA256:
        print(f"REFUSE: {MODEL_FILE} sha256 {h0} not in {ACCEPTED_SHA256}", flush=True)
        return 3
    import torch, scipy
    from exp_patience_control import _locked_merge
    from phase0_noisefloor import train_one
    import src.models.damrs as damrs_mod
    if Path(damrs_mod.__file__).resolve() != MODEL_FILE.resolve():
        print(f"REFUSE: imported {damrs_mod.__file__}, expected {MODEL_FILE}", flush=True)
        return 3

    if args.protocol == "converged":
        ov, prefix, patience, cap = {"stopping_step": 100, "epochs": 3000}, "g11pat100e3000", 100, 3000
        f = OUT / f"runs_{args.dataset}.json"
    else:
        ov, prefix, patience, cap = None, "g11nf", 20, 1000
        f = OUT / f"screen_runs_{args.dataset}.json"
    OUT.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    done = set()
    if f.is_file():
        done = {r["seed"] for r in json.loads(f.read_text()) if "test_result" in r}

    for s in args.seeds:
        if s in done:
            print(f"skip damrs/{args.dataset}/s{s}/{args.protocol}", flush=True)
            continue
        started = datetime.now().astimezone().isoformat(timespec="seconds")
        print(f"\n=== damrs(g11)/{args.dataset} seed={s} protocol={args.protocol} ===", flush=True)
        try:
            r = train_one("damrs", args.dataset, s, device, extra_overrides=ov, run_prefix=prefix)
            h1 = sha256(MODEL_FILE)
            r.update({
                "protocol": args.protocol, "patience": patience, "epochs_cap": cap,
                "hit_cap": bool(r["best_epoch"] >= 0.95 * cap),
                "code": "damrs_g11_repaired", "code_sha256_start": h0, "code_sha256_end": h1,
                "code_ok": h0 == h1 and h0 in ACCEPTED_SHA256, "started_at": started,
                "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "device": device, "host": platform.node(),
                "versions": {"torch": torch.__version__, "scipy": scipy.__version__},
            })
            _locked_merge(f, r)
            print(f"  R@20={r['test_result']['Recall@20']:.5f} best_epoch={r['best_epoch']} "
                  f"({r['train_min']:.1f} min) code_ok={r['code_ok']}", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            _locked_merge(f, {"model": "damrs", "dataset": args.dataset, "seed": s,
                              "protocol": args.protocol, "error": repr(e), "started_at": started})
    print(f"\nWrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
