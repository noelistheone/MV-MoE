"""Track A — retrain FREEDOM with CLIP features + analyze. THE confound test:
is "image ignored" an artifact of weak CNN-4096 features, or does it survive strong
CLIP ViT-L/14 features?

Variants: (cnn,bert)=baseline sanity, (clip,bert)=strong image only, (clip,clip)=fully CLIP.
For each: train FREEDOM (features injected directly; Recsys read-only), then run the exact
structural modality knockout + behavioral-alignment on the resulting model.

Output: results/trackA_clip/clip_freedom.json
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
sys.path.insert(0, str(RECSYS))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "interp"))
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_bridge import load_recsys_model_class, SCRATCH   # noqa: E402
from ranking_effects import evaluate_item_matrix             # noqa: E402
from phase1_knockout import freedom_streams                  # noqa: E402
from phase_gaps3_knockouts_alignment import copurchase_pairs, alignment_report  # noqa: E402
from src.utils import Config, set_seed                       # noqa: E402
from src.data.dataset import RecDataset                      # noqa: E402
from src.data.dataloader import EvalDataLoader, TrainDataLoader  # noqa: E402
from src.data.graph_utils import build_norm_adj              # noqa: E402
from src.common.trainer import Trainer                       # noqa: E402

OUT = ROOT / "results" / "trackA_clip"
CLIP = ROOT / "data" / "clip"


def get_feats(ds, dataset, image_src, text_src):
    if image_src == "cnn":
        v = torch.from_numpy(ds.v_feat[:].copy()).float()
    else:
        v = torch.from_numpy(np.load(CLIP / f"{dataset}_image_clip.npy")).float()
    if text_src == "bert":
        t = torch.from_numpy(ds.t_feat[:].copy()).float()
    else:
        t = torch.from_numpy(np.load(CLIP / f"{dataset}_text_clip.npy")).float()
    return v, t


def run_variant(dataset, image_src, text_src, device, seed=2024):
    cfg = Config("freedom", dataset, cli_overrides={
        "seed": seed, "ckpt_dir": str(SCRATCH / "ckpts"), "log_dir": str(SCRATCH / "logs"),
        "show_progress": False})
    set_seed(seed, deterministic=True)
    ds = RecDataset(cfg)
    norm_adj = build_norm_adj(ds.train_matrix, ds.n_users, ds.n_items)
    train_loader = TrainDataLoader(ds, batch_size=int(cfg["train_batch_size"]),
                                   num_workers=int(cfg.get("num_workers", 4)),
                                   max_neg_tries=int(cfg.get("neg_sampling_max_tries", 100)))
    valid_loader = EvalDataLoader(ds, phase="valid", batch_size=int(cfg.get("eval_batch_size_users", 1024)))
    test_loader = EvalDataLoader(ds, phase="test", batch_size=int(cfg.get("eval_batch_size_users", 1024)))
    v, t = get_feats(ds, dataset, image_src, text_src)

    FREEDOM = load_recsys_model_class("freedom")
    model = FREEDOM(config=cfg, n_users=ds.n_users, n_items=ds.n_items, norm_adj=norm_adj,
                    train_user_idx=torch.from_numpy(np.asarray(ds.train_users)),
                    train_item_idx=torch.from_numpy(np.asarray(ds.train_items)),
                    v_feat=v, t_feat=t).to(device)
    tag = f"{image_src}img_{text_src}txt"
    trainer = Trainer(cfg, model, train_loader, valid_loader, test_loader, run_name=f"clipfree_{dataset}_{tag}")
    t0 = time.time()
    res = trainer.fit()
    train_min = (time.time() - t0) / 60

    # --- exact structural knockout on the trained model
    model.eval()
    s = freedom_streams(model)
    u, fused, h_img, h_txt, cf = s["u_all"], s["fused"], s["h_img"], s["h_txt"], s["cf"]
    def R20(item):
        m, _, _ = evaluate_item_matrix(u, item, test_loader, device); return float(m["Recall@20"])
    base = R20(fused)
    image_ko = R20(cf + h_txt) - base
    text_ko = R20(cf + h_img) - base
    both_ko = R20(cf) - base
    def mn(x): return float(x.norm(dim=-1).mean())

    # --- behavioral alignment on the raw features + graph streams
    ia, ib = copurchase_pairs(ds, 20000)
    ia, ib = ia.to(device), ib.to(device)
    al = {"raw_image": alignment_report(v.to(device), ia, ib),
          "raw_text": alignment_report(t.to(device), ia, ib),
          "h_img(graph)": alignment_report(h_img, ia, ib),
          "h_txt(graph)": alignment_report(h_txt, ia, ib),
          "cf": alignment_report(cf, ia, ib)}

    return {"variant": tag, "image_src": image_src, "text_src": text_src,
            "v_dim": int(v.shape[1]), "t_dim": int(t.shape[1]),
            "train_min": train_min, "best_epoch": int(res["best_epoch"]),
            "baseline_R@20": base,
            "image_knockout_dR@20": image_ko, "text_knockout_dR@20": text_ko, "both_knockout_dR@20": both_ko,
            "norms": {"h_img": mn(h_img), "h_txt": mn(h_txt), "cf": mn(cf), "fused": mn(fused)},
            "alignment_behavioral_gap": {k: v2["behavioral_gap(random-co)"] for k, v2 in al.items()},
            "alignment_relative": {k: v2["relative_gap"] for k, v2 in al.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="baby")
    ap.add_argument("--variants", nargs="+", default=["cnn:bert", "clip:bert", "clip:clip"])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "clip_freedom.json"
    results = json.loads(out_path.read_text()) if out_path.is_file() else {}

    for vspec in args.variants:
        img, txt = vspec.split(":")
        print(f"\n===== FREEDOM {args.dataset}: image={img}, text={txt} =====", flush=True)
        r = run_variant(args.dataset, img, txt, device)
        results[f"{args.dataset}/{r['variant']}"] = r
        mde = 0.00061
        sig = "SIG" if abs(r["image_knockout_dR@20"]) > mde else "~noise"
        print(f"  base R@20={r['baseline_R@20']:.4f} | image-KO dR@20={r['image_knockout_dR@20']:+.4f} [{sig}]"
              f" | text-KO={r['text_knockout_dR@20']:+.4f} | norms img={r['norms']['h_img']:.2f} txt={r['norms']['h_txt']:.2f}")
        print(f"  align behavioral gap: raw_image={r['alignment_behavioral_gap']['raw_image']:.4f}"
              f" raw_text={r['alignment_behavioral_gap']['raw_text']:.4f} cf={r['alignment_behavioral_gap']['cf']:.4f}")
        out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
