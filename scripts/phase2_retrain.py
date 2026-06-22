"""Phase 2(B) — FREEDOM mm_image_weight RETRAIN sweep (behavioral ground truth).

The frozen sweep (2A) holds representations fixed. Here we RETRAIN FREEDOM from
scratch at each image weight to test whether the model can LEARN to use image when
given more weight (H1: ignored-by-construction) vs cannot (H2: intrinsically weak).
>=3 seeds per weight so Δ's are read against the Phase-0 noise floor.

Heavy (each Baby train ~5 min). Outputs -> results/phase2/weight_sweep_retrain.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/workspace/MechInterp")
RECSYS = Path("/workspace/Recsys")
sys.path.insert(0, str(RECSYS))
sys.path.insert(0, str(ROOT / "src" / "models"))
from recsys_bridge import load_recsys_model_class, SCRATCH    # noqa: E402
from src.utils import Config, set_seed                        # noqa: E402
from src.data.dataset import RecDataset                       # noqa: E402
from src.data.dataloader import EvalDataLoader, TrainDataLoader  # noqa: E402
from src.data.graph_utils import build_norm_adj               # noqa: E402
from src.common.trainer import Trainer                        # noqa: E402

OUT = ROOT / "results" / "phase2"


def train_one(dataset_name: str, w: float, seed: int, device: str) -> dict:
    cfg = Config("freedom", dataset_name, cli_overrides={
        "mm_image_weight": w, "seed": seed,
        "ckpt_dir": str(SCRATCH / "ckpts"), "log_dir": str(SCRATCH / "logs"),
        "show_progress": False})
    set_seed(seed, deterministic=bool(cfg.get("cudnn_deterministic", True)))
    dataset = RecDataset(cfg)
    norm_adj = build_norm_adj(dataset.train_matrix, dataset.n_users, dataset.n_items)
    train_loader = TrainDataLoader(dataset, batch_size=int(cfg["train_batch_size"]),
                                   num_workers=int(cfg.get("num_workers", 4)),
                                   max_neg_tries=int(cfg.get("neg_sampling_max_tries", 100)))
    valid_loader = EvalDataLoader(dataset, phase="valid", batch_size=int(cfg.get("eval_batch_size_users", 1024)))
    test_loader = EvalDataLoader(dataset, phase="test", batch_size=int(cfg.get("eval_batch_size_users", 1024)))
    ModelCls = load_recsys_model_class("freedom")
    v_feat = torch.from_numpy(dataset.v_feat[:].copy())
    t_feat = torch.from_numpy(dataset.t_feat[:].copy())
    model = ModelCls(config=cfg, n_users=dataset.n_users, n_items=dataset.n_items, norm_adj=norm_adj,
                     train_user_idx=torch.from_numpy(np.asarray(dataset.train_users)),
                     train_item_idx=torch.from_numpy(np.asarray(dataset.train_items)),
                     v_feat=v_feat, t_feat=t_feat).to(device)
    assert abs(model.mm_image_weight - w) < 1e-9, f"weight override failed: {model.mm_image_weight} != {w}"
    trainer = Trainer(cfg, model, train_loader, valid_loader, test_loader,
                      run_name=f"w2_freedom_{dataset_name}_w{w}_s{seed}")
    t0 = time.time()
    res = trainer.fit()
    return {"dataset": dataset_name, "weight": w, "seed": seed,
            "best_epoch": int(res["best_epoch"]), "train_min": (time.time() - t0) / 60.0,
            "test_result": {k: float(v) for k, v in res["test_result"].items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="baby")
    ap.add_argument("--weights", nargs="+", type=float, default=[0.0, 0.1, 0.5, 0.9, 1.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025, 2026])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "weight_sweep_retrain.json"
    runs = json.loads(out_path.read_text())["runs"] if out_path.is_file() else []

    for w in args.weights:
        for s in args.seeds:
            print(f"\n=== retrain freedom/{args.dataset} w={w} seed={s} ===", flush=True)
            try:
                r = train_one(args.dataset, w, s, device)
                runs.append(r)
                print(f"  R@20={r['test_result']['Recall@20']:.4f} N@20={r['test_result']['NDCG@20']:.4f}"
                      f" (epoch {r['best_epoch']}, {r['train_min']:.1f}m)", flush=True)
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                runs.append({"dataset": args.dataset, "weight": w, "seed": s, "error": repr(e)})
            # summarize + checkpoint each step
            summary = {}
            for ww in sorted({r.get("weight") for r in runs if "test_result" in r}):
                vals = [r["test_result"]["Recall@20"] for r in runs
                        if r.get("weight") == ww and r.get("dataset") == args.dataset and "test_result" in r]
                if vals:
                    summary[str(ww)] = {"n": len(vals), "mean_R@20": statistics.mean(vals),
                                        "std_R@20": statistics.stdev(vals) if len(vals) > 1 else 0.0}
            out_path.write_text(json.dumps({"runs": runs, "summary_by_weight": summary,
                                            "dataset": args.dataset}, indent=2))
    print("\n=== RETRAIN SWEEP SUMMARY (mean R@20 by weight) ===")
    s = json.loads(out_path.read_text())["summary_by_weight"]
    for w in sorted(s, key=float):
        print(f"  w={w}: R@20={s[w]['mean_R@20']:.4f} ± {s[w]['std_R@20']:.4f} (n={s[w]['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
