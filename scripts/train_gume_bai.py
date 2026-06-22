"""Train GUME (baseline) or GUME+BAI in the MechInterp workspace.

Drives the read-only Recsys training stack (Config / RecDataset / dataloaders /
Trainer) but instantiates our GUME_BAI class. Baseline (bai_enable off) reproduces
GUME exactly; the BAI variant adds the behavior-aligned image residual.

Usage:
  python scripts/train_gume_bai.py --dataset baby --variant baseline --seed 2024
  python scripts/train_gume_bai.py --dataset baby --variant bai --seed 2024 --bai_alpha 1.0
Writes results/bai/<dataset>_<variant>_s<seed>[_<tag>].json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/workspace/MechInterp")
RECSYS = Path("/workspace/Recsys")
sys.path.insert(0, str(RECSYS))                       # so "src.*" resolves to Recsys
sys.path.insert(0, str(ROOT / "src" / "models"))      # so we can import gume_bai by filename

from src.utils import Config, set_seed, configure_runtime           # noqa: E402
from src.data.dataset import RecDataset                              # noqa: E402
from src.data.dataloader import TrainDataLoader, EvalDataLoader      # noqa: E402
from src.data.graph_utils import build_norm_adj                      # noqa: E402
from src.common.trainer import Trainer                               # noqa: E402
from gume_bai import GUME_BAI                                        # noqa: E402

OUT = ROOT / "results" / "bai"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="baby")
    ap.add_argument("--variant", choices=["baseline", "bai"], default="baseline")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--bai_alpha", type=float, default=1.0)
    ap.add_argument("--bai_align", type=float, default=0.0)
    ap.add_argument("--bai_cop", type=float, default=0.0)
    ap.add_argument("--bai_reg", type=float, default=0.0)
    ap.add_argument("--bai_channel", action="store_true",
                    help="separate image scoring channel (u_img . z_img) instead of residual")
    ap.add_argument("--bai_chan_scale", type=float, default=1.0)
    ap.add_argument("--bai_chan_reg", type=float, default=0.0)
    ap.add_argument("--bai_user_img_mode", choices=["content", "free", "loo"], default="content")
    ap.add_argument("--bai_loo_weight", type=float, default=1.0)
    ap.add_argument("--bai_cotrain", action="store_true",
                    help="loo mode: co-train base embeddings with the channel (fold into BPR)")
    ap.add_argument("--bai_loss", choices=["bpr", "ssm"], default="bpr",
                    help="ssm = in-batch sampled-softmax objective (replaces BPR)")
    ap.add_argument("--ssm_temp", type=float, default=0.15)
    ap.add_argument("--ssm_debias", type=float, default=1.0)
    ap.add_argument("--ssm_no_norm", action="store_true", help="disable cosine normalization in SSM")
    ap.add_argument("--ssm_popbias", type=float, default=0.0, help="popularity prior added to logits+eval")
    ap.add_argument("--ssm_aux", type=float, default=0.0, help="weight of in-batch softmax ADDED to GUME's BPR")
    ap.add_argument("--dump_scores", action="store_true", help="save full [users,items] score matrix (fp16) for ensembling")
    ap.add_argument("--n_layers", type=int, default=None, help="override GUME n_layers (architecture diversity)")
    ap.add_argument("--knn_k", type=int, default=None, help="override GUME knn_k (mm-graph density)")
    ap.add_argument("--distill_teacher", default=None, help="path to teacher ensemble score matrix .npy")
    ap.add_argument("--distill_weight", type=float, default=0.0)
    ap.add_argument("--distill_temp", type=float, default=1.0)
    ap.add_argument("--bai_img_npy", default=None, help="external image feature .npy (e.g. multi-encoder)")
    ap.add_argument("--bai_gate_bias", type=float, default=-2.0)
    ap.add_argument("--bai_graph_mode", choices=["none", "cooc"], default="none",
                    help="rebuild GUME image graph behavior-aligned (independent of residual)")
    ap.add_argument("--epochs", type=int, default=None, help="override max epochs")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    overrides = {"seed": args.seed, "bai_graph_mode": args.bai_graph_mode}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.n_layers is not None:
        overrides["n_layers"] = args.n_layers
    if args.knn_k is not None:
        overrides["knn_k"] = args.knn_k
    if args.variant == "bai":
        overrides.update(bai_enable=(not args.bai_channel), bai_channel=args.bai_channel,
                         bai_alpha=args.bai_alpha, bai_align=args.bai_align,
                         bai_cop=args.bai_cop, bai_reg=args.bai_reg,
                         bai_chan_scale=args.bai_chan_scale, bai_chan_reg=args.bai_chan_reg,
                         bai_user_img_mode=args.bai_user_img_mode, bai_img_npy=args.bai_img_npy,
                         bai_loo_weight=args.bai_loo_weight, bai_cotrain=args.bai_cotrain,
                         bai_gate_bias=args.bai_gate_bias)
    # objective upgrade applies to plain GUME too (no channel needed)
    overrides.update(bai_loss=args.bai_loss, ssm_temp=args.ssm_temp,
                     ssm_debias=args.ssm_debias, ssm_norm=(not args.ssm_no_norm),
                     ssm_popbias=args.ssm_popbias, ssm_aux=args.ssm_aux,
                     distill_teacher=args.distill_teacher, distill_weight=args.distill_weight,
                     distill_temp=args.distill_temp)
    # Load GUME's per-dataset hyperparameters (configs/model/gume.yaml).
    cfg = Config("gume", args.dataset, cli_overrides=overrides)
    configure_runtime(cfg)
    set_seed(int(cfg.get("seed", 2024)), deterministic=bool(cfg.get("cudnn_deterministic", True)))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    ds = RecDataset(cfg)
    norm_adj = build_norm_adj(ds.train_matrix, ds.n_users, ds.n_items)
    train_loader = TrainDataLoader(ds, batch_size=int(cfg["train_batch_size"]),
                                   num_workers=int(cfg.get("num_workers", 0)),
                                   max_neg_tries=int(cfg.get("neg_sampling_max_tries", 100)))
    valid_loader = EvalDataLoader(ds, phase="valid", batch_size=int(cfg.get("eval_batch_size_users", 1024)))
    test_loader = EvalDataLoader(ds, phase="test", batch_size=int(cfg.get("eval_batch_size_users", 1024)))

    v_feat = torch.from_numpy(ds.v_feat[:].copy()) if ds.v_feat is not None else None
    t_feat = torch.from_numpy(ds.t_feat[:].copy()) if ds.t_feat is not None else None
    model = GUME_BAI(config=cfg, n_users=ds.n_users, n_items=ds.n_items, norm_adj=norm_adj,
                     v_feat=v_feat, t_feat=t_feat,
                     train_user_idx=torch.from_numpy(np.asarray(ds.train_users)),
                     train_item_idx=torch.from_numpy(np.asarray(ds.train_items))).to(device)

    run_name = f"gumebai_{args.dataset}_{args.variant}_s{args.seed}"
    trainer = Trainer(cfg, model, train_loader, valid_loader, test_loader, run_name=run_name)
    t0 = time.time()
    result = trainer.fit()
    mins = (time.time() - t0) / 60.0

    if args.dump_scores:
        sdir = OUT / "scores"; sdir.mkdir(parents=True, exist_ok=True)
        model.eval()
        with torch.no_grad():
            chunks = []
            for s in range(0, ds.n_users, 2048):
                us = torch.arange(s, min(s + 2048, ds.n_users), device=device)
                chunks.append(model.full_sort_predict({"user": us}).half().cpu().numpy())
        np.save(sdir / f"{args.dataset}_{args.tag}_s{args.seed}.npy", np.concatenate(chunks, 0))
        print(f"  dumped scores -> {sdir / f'{args.dataset}_{args.tag}_s{args.seed}.npy'}")

    summary = {
        "dataset": args.dataset, "variant": args.variant, "seed": args.seed,
        "bai_enable": bool(args.variant == "bai"), "bai_alpha": args.bai_alpha,
        "bai_align": args.bai_align, "bai_cop": args.bai_cop, "bai_reg": args.bai_reg,
        "bai_graph_mode": args.bai_graph_mode, "bai_gate_bias": args.bai_gate_bias,
        "bai_loss": args.bai_loss, "ssm_temp": args.ssm_temp, "ssm_debias": args.ssm_debias,
        "best_valid_metric": trainer.valid_metric,
        "best_valid_score": float(result["best_valid"]),
        "best_epoch": int(result["best_epoch"]),
        "train_time_min": float(mins),
        "test_result": {k: float(v) for k, v in result["test_result"].items()},
        "best_valid_result": {k: float(v) for k, v in result.get("best_valid_result", {}).items()},
    }
    if args.variant == "bai":
        with torch.no_grad():
            if args.bai_channel:
                cs = model.bai_cs
                summary["bai_cs"] = float(cs.detach().cpu().item()) if torch.is_tensor(cs) else float(cs)
                summary["z_img_norm"] = float(model._img_proj_norm().norm(dim=1).mean().cpu().item())
                if getattr(model, "bai_user_img_mode", "content") == "free":
                    summary["u_img_norm"] = float(model.bai_user_img.weight.norm(dim=1).mean().cpu().item())
            else:
                summary["learned_bai_scale"] = float(model.bai_scale.detach().cpu().item())
                g = model.bai_gate(model.item_id_embedding.weight)
                summary["mean_gate"] = float(g.mean().cpu().item())
                res = model._img_residual()
                item_emb = model.forward(model.gume_norm_adj)[ds.n_users:]
                summary["residual_norm_frac"] = float(
                    (res.norm(dim=1).mean() / item_emb.norm(dim=1).mean()).cpu().item())
    tag = f"_{args.tag}" if args.tag else ""
    out_path = OUT / f"{args.dataset}_{args.variant}_s{args.seed}{tag}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    tr = summary["test_result"]
    print(f"[{args.variant}/{args.dataset}/s{args.seed}] test R@20={tr.get('Recall@20'):.5f} "
          f"N@20={tr.get('NDCG@20'):.5f} | best_valid={summary['best_valid_score']:.5f} "
          f"@ep{summary['best_epoch']} | {mins:.1f}min -> {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
