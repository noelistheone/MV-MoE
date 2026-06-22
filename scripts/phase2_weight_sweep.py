"""Phase 2(A) — FREEDOM mm_image_weight sweep, FROZEN weights (the cheap, mechanistic half).

FREEDOM scores on fused = cf + w*(imgKNN·E) + (1-w)*(txtKNN·E). We hold the trained
embeddings fixed and only re-mix the image/text graph contribution at inference,
sweeping w in [0,1]. This isolates the *inference-time* sensitivity of ranking to the
modality mix. If raising w from 0.1 toward 0.9 doesn't revive image (R@20 flat or down),
the image signal is intrinsically weak / inert at fixed representations (H2); the
retrain half (Phase 2B) tests whether the model could LEARN to use image.

Outputs -> MechInterp/results/phase2/. Recsys read-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/workspace/MechInterp")
RECSYS = Path("/workspace/Recsys")
sys.path.insert(0, str(RECSYS))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "interp"))
from recsys_bridge import load_frozen                        # noqa: E402
from ranking_effects import evaluate_item_matrix, ranking_change  # noqa: E402
from src.data.graph_utils import build_knn_graph             # noqa: E402

OUT = ROOT / "results" / "phase2"


@torch.no_grad()
def freedom_components(model):
    """cf, g_img = imgKNN·E (UNWEIGHTED), g_txt = txtKNN·E, and u_all. n_mm_layers=1."""
    assert model.n_mm_layers == 1
    E = model.item_id_embedding.weight.detach()
    dev = E.device
    img = build_knn_graph(model.v_feat.cpu(), model.knn_k).to(dev)
    txt = build_knn_graph(model.t_feat.cpu(), model.knn_k).to(dev)
    g_img = torch.sparse.mm(img, E)
    g_txt = torch.sparse.mm(txt, E)
    ego = torch.cat([model.user_embedding.weight, model.item_id_embedding.weight], 0).detach()
    embs = [ego]
    for _ in range(model.n_layers):
        ego = torch.sparse.mm(model.norm_adj, ego); embs.append(ego)
    out = torch.stack(embs, 1).mean(1)
    user_e, item_e = torch.split(out, [model.n_users, model.n_items], 0)
    return user_e, item_e, g_img, g_txt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="baby")
    ap.add_argument("--weights", nargs="+", type=float,
                    default=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    cfg, ds, model, test_loader = load_frozen("freedom", args.dataset, device)
    trained_w = model.mm_image_weight
    u, cf, g_img, g_txt = freedom_components(model)

    # reference top-K at the trained weight, for ranking-change vs each swept weight
    base_item = cf + trained_w * g_img + (1 - trained_w) * g_txt
    base_metrics, base_topk, _ = evaluate_item_matrix(u, base_item, test_loader, device)

    rows = []
    for w in args.weights:
        item = cf + w * g_img + (1 - w) * g_txt
        m, topk, _ = evaluate_item_matrix(u, item, test_loader, device)
        rc = ranking_change(base_topk, topk, k=20)
        rows.append({"weight": w,
                     "Recall@20": float(m["Recall@20"]), "NDCG@20": float(m["NDCG@20"]),
                     "Recall@10": float(m["Recall@10"]), "NDCG@10": float(m["NDCG@10"]),
                     "overlap@20_vs_trained": rc["overlap@20"], "rbo_vs_trained": rc["rbo"]})
        print(f"  w={w:.2f}  R@20={m['Recall@20']:.4f}  N@20={m['NDCG@20']:.4f}"
              f"  overlap_vs_trained={rc['overlap@20']:.3f}", flush=True)

    mde = None
    sv = ROOT / "results" / "phase0" / "seed_variance.json"
    if sv.is_file():
        mde = json.loads(sv.read_text()).get("summary", {}).get("freedom", {}).get("Recall@20", {}).get("MDE_2std")

    r20 = [r["Recall@20"] for r in rows]
    out = {"model": "freedom", "dataset": args.dataset, "trained_weight": trained_w,
           "checkpoint": getattr(model, "_ckpt_name", None), "MDE_R@20": mde,
           "sweep": rows,
           "interpretation": {
               "R@20_at_w0_textonly": rows[0]["Recall@20"] if rows[0]["weight"] == 0.0 else None,
               "R@20_at_w1_imageonly": rows[-1]["Recall@20"] if rows[-1]["weight"] == 1.0 else None,
               "R@20_range": max(r20) - min(r20),
               "argmax_weight": args.weights[int(torch.tensor(r20).argmax())],
               "image_helps_when_upweighted": bool(max(r20) - rows[1]["Recall@20"] > (mde or 0)) if len(rows) > 1 else None,
           }}
    out_path = OUT / "weight_sweep_frozen.json"
    existing = {}
    if out_path.is_file():
        for r in json.loads(out_path.read_text()):
            existing[r["dataset"]] = r
    existing[args.dataset] = out
    out_path.write_text(json.dumps(list(existing.values()), indent=2))
    print(f"\n  trained_w={trained_w}  argmax_w={out['interpretation']['argmax_weight']}"
          f"  R@20 range={out['interpretation']['R@20_range']:.4f}  MDE={mde}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
