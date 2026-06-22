"""Phase 0.3 — Multi-seed noise floor (MDE).

Retrain FREEDOM and LightGCN on Baby across several seeds to quantify run-to-run
variance, so later intervention Δ's can be judged against a measured noise floor.
MDE (minimum detectable effect) := 2 * std(R@20). Any later Δ < MDE is reported
as "indistinguishable from training noise".

Everything stays under MechInterp: checkpoints/logs go to results/phase0/_scratch,
the summary to results/phase0/seed_variance.json. Recsys is read-only source.

Run (background):
    conda run -n mechinterp python MechInterp/scripts/phase0_noisefloor.py \
        --models lightgcn freedom --dataset baby --seeds 2024 2025 2026 2027 2028
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path("/workspace/MechInterp/scripts")))
from phase0_repro import (  # noqa: E402  (reuse loader + paths)
    RECSYS, MECHINTERP, OUT_DIR, build_model, load_recsys_model,
)

sys.path.insert(0, str(RECSYS))
from src.utils import Config, set_seed                       # noqa: E402
from src.data.dataset import RecDataset                      # noqa: E402
from src.data.dataloader import EvalDataLoader, TrainDataLoader  # noqa: E402
from src.data.graph_utils import build_norm_adj              # noqa: E402
from src.common.trainer import Trainer                       # noqa: E402


def train_one(model_name: str, dataset_name: str, seed: int, device: str) -> dict:
    scratch = OUT_DIR / "_scratch"
    cfg = Config(model_name, dataset_name, cli_overrides={
        "seed": seed,
        "ckpt_dir": str(scratch / "ckpts"),
        "log_dir": str(scratch / "logs"),
        "show_progress": False,
    })
    set_seed(seed, deterministic=bool(cfg.get("cudnn_deterministic", True)))

    dataset = RecDataset(cfg)
    norm_adj = build_norm_adj(dataset.train_matrix, dataset.n_users, dataset.n_items)
    train_loader = TrainDataLoader(dataset, batch_size=int(cfg["train_batch_size"]),
                                   num_workers=int(cfg.get("num_workers", 4)),
                                   max_neg_tries=int(cfg.get("neg_sampling_max_tries", 100)))
    valid_loader = EvalDataLoader(dataset, phase="valid",
                                  batch_size=int(cfg.get("eval_batch_size_users", 1024)))
    test_loader = EvalDataLoader(dataset, phase="test",
                                 batch_size=int(cfg.get("eval_batch_size_users", 1024)))

    model = build_model(model_name, cfg, dataset, norm_adj, device)
    trainer = Trainer(cfg, model, train_loader, valid_loader, test_loader,
                      run_name=f"nf_{model_name}_{dataset_name}_s{seed}")
    t0 = time.time()
    result = trainer.fit()
    return {
        "model": model_name, "dataset": dataset_name, "seed": seed,
        "best_epoch": int(result["best_epoch"]),
        "train_min": (time.time() - t0) / 60.0,
        "test_result": {k: float(v) for k, v in result["test_result"].items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lightgcn", "freedom"])
    ap.add_argument("--dataset", default="baby")
    ap.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025, 2026, 2027, 2028])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runs, summary = [], {}
    for m in args.models:
        for s in args.seeds:
            print(f"\n=== train {m}/{args.dataset} seed={s} ===", flush=True)
            try:
                r = train_one(m, args.dataset, s, device)
                runs.append(r)
                print(f"  R@20={r['test_result'].get('Recall@20'):.4f} "
                      f"N@20={r['test_result'].get('NDCG@20'):.4f} "
                      f"(epoch {r['best_epoch']}, {r['train_min']:.1f} min)", flush=True)
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                runs.append({"model": m, "seed": s, "error": repr(e)})
            # checkpoint the partial results each step
            (OUT_DIR / "seed_variance_runs.json").write_text(json.dumps(runs, indent=2))

    for m in args.models:
        rs = [r for r in runs if r.get("model") == m and "test_result" in r]
        for metric in ("Recall@20", "NDCG@20", "Recall@10", "NDCG@10"):
            vals = [r["test_result"][metric] for r in rs if metric in r["test_result"]]
            if len(vals) >= 2:
                mean = statistics.mean(vals)
                std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                std_s = statistics.stdev(vals)
                summary.setdefault(m, {})[metric] = {
                    "n": len(vals), "mean": mean, "std": std_s,
                    "MDE_2std": 2 * std_s, "values": vals,
                }

    out = {"runs": runs, "summary": summary,
           "note": "MDE_2std = minimum detectable effect; later |Δ| < MDE is within training noise."}
    (OUT_DIR / "seed_variance.json").write_text(json.dumps(out, indent=2))
    print("\n=== NOISE FLOOR (MDE = 2*std of R@20) ===")
    for m, mm in summary.items():
        r20 = mm.get("Recall@20", {})
        print(f"  {m}: R@20 mean={r20.get('mean',0):.4f} std={r20.get('std',0):.5f} "
              f"MDE={r20.get('MDE_2std',0):.5f}")
    print(f"Wrote {OUT_DIR/'seed_variance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
