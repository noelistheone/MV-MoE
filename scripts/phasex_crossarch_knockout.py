"""Cross-architecture modality knockout across PUBLISHED multimodal recommenders.

Applies test-time mean replacement (replace a modality's input feature with its
per-dimension mean over items, then re-run inference) uniformly across many
published models, via each model's own full_sort_predict. This tests how universal
"image ignored" is across the published multimodal-recsys literature — no architecture
invention; every model is a searchable paper. Re-instantiating with the knocked-out
feature captures BOTH live-feature use and frozen-graph construction.

Models (published): VBPR(AAAI'16), MMGCN(MM'19), LATTICE(MM'21), BM3(WWW'23),
MGCN(MM'23), MENTOR(AAAI'24), + FREEDOM/LGMRec as in-study references.
Outputs -> results/phasex_crossarch/. Recsys read-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/workspace/MechInterp")
RECSYS = Path("/workspace/Recsys")
sys.path.insert(0, str(RECSYS))
sys.path.insert(0, str(ROOT / "src" / "models"))
from recsys_bridge import (load_recsys_model_class, latest_ckpt, SCRATCH,  # noqa: E402
                           Config, set_seed, RecDataset, EvalDataLoader, build_norm_adj)
from src.common.trainer import Trainer                        # noqa: E402

OUT = ROOT / "results" / "phasex_crossarch"
# typical noise floor from the measured models (freedom 0.0006, lightgcn 0.0026)
NOISE_HINT = 0.0026


def build_eval(model_name, cfg, dataset, norm_adj, device, v_feat, t_feat, test_loader):
    ModelCls = load_recsys_model_class(model_name)
    kwargs = {"config": cfg, "n_users": dataset.n_users, "n_items": dataset.n_items, "norm_adj": norm_adj}
    ml = model_name.lower()
    if ml not in ("lightgcn", "mllmrec", "falcon"):
        kwargs.update(v_feat=v_feat, t_feat=t_feat)
    if ml in ("freedom", "mllmrec", "histllm", "grcn", "dragon", "smore", "gume", "damrs", "cohesion"):
        kwargs.update(train_user_idx=torch.from_numpy(np.asarray(dataset.train_users)),
                      train_item_idx=torch.from_numpy(np.asarray(dataset.train_items)))
    model = ModelCls(**kwargs).to(device)
    ck = latest_ckpt(model_name, cfg["dataset"])
    model.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["model_state_dict"], strict=False)
    model.eval()
    # COHESION builds epoch_user_graph / masked_adj inside pre_epoch_processing (the training loop
    # calls it every epoch); an eval-only reload must call it once. Force the full graph (dropout=0)
    # for deterministic inference, matching the unmasked-graph eval convention used by FREEDOM.
    if ml == "cohesion" and hasattr(model, "pre_epoch_processing"):
        if hasattr(model, "dropout"):
            model.dropout = 0.0
        model.pre_epoch_processing()
    tr = Trainer(cfg, model, None, None, test_loader)
    return tr.evaluate(test_loader)


def run_model(model_name, dataset_name, device):
    scratch = OUT / "_scratch"
    cfg = Config(model_name, dataset_name, cli_overrides={"ckpt_dir": str(scratch / "ckpts"), "log_dir": str(scratch / "logs")})
    set_seed(int(cfg.get("seed", 2024)), deterministic=True)
    ds = RecDataset(cfg)
    norm_adj = build_norm_adj(ds.train_matrix, ds.n_users, ds.n_items)
    test_loader = EvalDataLoader(ds, phase="test", batch_size=int(cfg.get("eval_batch_size_users", 1024)))
    v = torch.from_numpy(ds.v_feat[:].copy()); t = torch.from_numpy(ds.t_feat[:].copy())
    v_mean = v.mean(0, keepdim=True).expand_as(v).contiguous()
    t_mean = t.mean(0, keepdim=True).expand_as(t).contiguous()

    base = build_eval(model_name, cfg, ds, norm_adj, device, v, t, test_loader)
    img = build_eval(model_name, cfg, ds, norm_adj, device, v_mean, t, test_loader)
    txt = build_eval(model_name, cfg, ds, norm_adj, device, v, t_mean, test_loader)
    both = build_eval(model_name, cfg, ds, norm_adj, device, v_mean, t_mean, test_loader)

    def dl(m): return {k: float(m[k]) - float(base[k]) for k in m}
    return {"model": model_name, "dataset": dataset_name,
            "baseline": {k: float(v) for k, v in base.items()},
            "image_knockout": dl(img), "text_knockout": dl(txt), "both_knockout": dl(both),
            "noise_hint": NOISE_HINT}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["vbpr", "mmgcn", "lattice", "bm3", "mgcn", "mentor", "freedom", "lgmrec"])
    ap.add_argument("--datasets", nargs="+", default=["baby"])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "crossarch_knockout.json"
    existing = {f"{r['model']}/{r['dataset']}": r for r in json.loads(out_path.read_text())} if out_path.is_file() else {}

    for ds in args.datasets:
        for m in args.models:
            print(f"\n=== {m} / {ds} ===", flush=True)
            try:
                r = run_model(m, ds, device)
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc(); existing[f"{m}/{ds}"] = {"model": m, "dataset": ds, "error": repr(e)}; continue
            existing[f"{m}/{ds}"] = r
            b = r["baseline"]["Recall@20"]
            di, dt = r["image_knockout"]["Recall@20"], r["text_knockout"]["Recall@20"]
            print(f"  base R@20={b:.4f} | image-KO dR@20={di:+.4f} ({100*di/b:+.1f}%)"
                  f" | text-KO dR@20={dt:+.4f} ({100*dt/b:+.1f}%)"
                  f" | image_used={'NO(~noise)' if abs(di)<NOISE_HINT else 'YES'}", flush=True)
            out_path.write_text(json.dumps(list(existing.values()), indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
