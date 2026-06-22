"""Phase 1 — Modality knockout localization (WHERE, behavioral), ranking-native.

FREEDOM: modality enters scoring ONLY through the frozen item-item graph, and with
n_mm_layers=1 the fused item embed decomposes EXACTLY linearly:
    fused = item_e(CF) + h_img + h_txt,   h_img = w·(imgKNN·E),  h_txt = (1-w)·(txtKNN·E)
so structural knockout = drop h_img / h_txt (exact, deterministic).
LGMRec: modality is used as live features at inference (+ stochastic gumbel hypergraph),
so knockout = replace v_feat/t_feat with their per-dim mean, with the gumbel draw PAIRED
across conditions (same seed) for a clean Δ.
LightGCN: no modality -> negative control (knockout undefined; baseline only).

Outputs -> MechInterp/results/phase1/. Recsys is read-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/workspace/MechInterp")
RECSYS = Path("/workspace/Recsys")
# Both repos use the top-level package name `src`. `src` MUST mean Recsys here (its
# modules do `from src...`); import MechInterp helpers as TOP-LEVEL modules from
# their subdirs (they themselves only use `from src...` = Recsys, no relative imports).
sys.path.insert(0, str(RECSYS))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "interp"))
from recsys_bridge import load_frozen                        # noqa: E402
from ranking_effects import evaluate_item_matrix, ranking_change  # noqa: E402
from src.data.graph_utils import build_knn_graph             # noqa: E402

OUT = ROOT / "results" / "phase1"
GUMBEL_SEED = 12345


# --------------------------------------------------------------- FREEDOM
@torch.no_grad()
def freedom_streams(model) -> dict:
    assert model.n_mm_layers == 1, "linear decomposition assumes n_mm_layers=1"
    E = model.item_id_embedding.weight.detach()
    dev = E.device
    # The model built mm_adj on CPU at __init__ (v_feat was CPU then). build_knn_graph
    # is CPU/GPU-float-sensitive (cosine-sim topk tie-breaking), so we MUST rebuild on
    # CPU from v_feat.cpu() to byte-reproduce the model's frozen graph, then move to dev.
    img_knn = build_knn_graph(model.v_feat.cpu(), model.knn_k).to(dev)
    txt_knn = build_knn_graph(model.t_feat.cpu(), model.knn_k).to(dev)
    h_img = model.mm_image_weight * torch.sparse.mm(img_knn, E)
    h_txt = (1.0 - model.mm_image_weight) * torch.sparse.mm(txt_knn, E)
    ego = torch.cat([model.user_embedding.weight, model.item_id_embedding.weight], 0).detach()
    embs = [ego]
    for _ in range(model.n_layers):
        ego = torch.sparse.mm(model.norm_adj, ego)
        embs.append(ego)
    out = torch.stack(embs, 1).mean(1)
    user_e, item_e = torch.split(out, [model.n_users, model.n_items], 0)
    fused = item_e + h_img + h_txt
    # Verify the reconstruction against the model's own propagation.
    mu, mi = model._propagate(model.norm_adj)
    max_i = (fused - mi).abs().max().item()
    max_u = (user_e - mu).abs().max().item()
    assert max_i < 1e-3 and max_u < 1e-3, f"reconstruction mismatch: item {max_i}, user {max_u}"
    return {"u_all": user_e, "cf": item_e, "h_img": h_img, "h_txt": h_txt,
            "fused": fused, "recon_max_err": max(max_i, max_u)}


def freedom_variants(streams):
    cf, h_img, h_txt = streams["cf"], streams["h_img"], streams["h_txt"]
    u = streams["u_all"]
    return {
        "baseline":       (u, streams["fused"]),
        "image_knockout": (u, cf + h_txt),     # drop h_img (structural)
        "text_knockout":  (u, cf + h_img),     # drop h_txt
        "both_knockout":  (u, cf),             # CF only
    }


def freedom_attribution(streams) -> dict:
    def mn(x): return float(x.norm(dim=-1).mean())
    return {"||cf||": mn(streams["cf"]), "||h_img||": mn(streams["h_img"]),
            "||h_txt||": mn(streams["h_txt"]), "||fused||": mn(streams["fused"]),
            "image_weight": None}  # filled by caller


# --------------------------------------------------------------- LGMRec
@torch.no_grad()
def lgmrec_components(model, seed: int = GUMBEL_SEED) -> dict:
    """Replicate LGMRec._forward_views, exposing each additive component so we can
    zero a modality's CONTRIBUTION cleanly (no constant-leakage). Gumbel is drawn
    once (fixed seed) and reused across all conditions -> paired, clean Δ.
    all = cge + norm(v_feats) + norm(t_feats) + alpha*norm(av_hyper + at_hyper)."""
    import torch.nn.functional as F
    torch.manual_seed(seed)
    nU = model.n_users
    iv_hyper = model.v_feat @ model.v_hyper
    it_hyper = model.t_feat @ model.t_hyper
    uv_hyper = torch.sparse.mm(model.R, iv_hyper)
    ut_hyper = torch.sparse.mm(model.R, it_hyper)
    iv_hyper = F.gumbel_softmax(iv_hyper, model.tau, dim=1, hard=False)
    uv_hyper = F.gumbel_softmax(uv_hyper, model.tau, dim=1, hard=False)
    it_hyper = F.gumbel_softmax(it_hyper, model.tau, dim=1, hard=False)
    ut_hyper = F.gumbel_softmax(ut_hyper, model.tau, dim=1, hard=False)
    cge = model._cge()
    nv = F.normalize(model._mge("v"), dim=-1)
    nt = F.normalize(model._mge("t"), dim=-1)
    uv_emb, iv_emb = model.hgnnLayer(model.drop(iv_hyper), model.drop(uv_hyper), cge[nU:])
    ut_emb, it_emb = model.hgnnLayer(model.drop(it_hyper), model.drop(ut_hyper), cge[nU:])
    av = torch.cat([uv_emb, iv_emb], 0)
    at = torch.cat([ut_emb, it_emb], 0)
    return {"nU": nU, "cge": cge, "nv": nv, "nt": nt, "av": av, "at": at, "alpha": model.alpha}


def _lgmrec_combine(s, keep_v: bool, keep_t: bool):
    import torch.nn.functional as F
    allv = s["cge"].clone()
    if keep_v:
        allv = allv + s["nv"]
    if keep_t:
        allv = allv + s["nt"]
    ghe_terms = []
    if keep_v:
        ghe_terms.append(s["av"])
    if keep_t:
        ghe_terms.append(s["at"])
    if ghe_terms:
        ghe = ghe_terms[0] if len(ghe_terms) == 1 else ghe_terms[0] + ghe_terms[1]
        allv = allv + s["alpha"] * F.normalize(ghe, dim=-1)
    u, i = allv[:s["nU"]], allv[s["nU"]:]
    return u, i


def lgmrec_variants(model):
    s = lgmrec_components(model)
    return {
        "baseline":       _lgmrec_combine(s, keep_v=True,  keep_t=True),
        "image_knockout": _lgmrec_combine(s, keep_v=False, keep_t=True),   # drop image contribution
        "text_knockout":  _lgmrec_combine(s, keep_v=True,  keep_t=False),  # drop text contribution
        "both_knockout":  _lgmrec_combine(s, keep_v=False, keep_t=False),  # CF only
    }


def lgmrec_attribution(model):
    s = lgmrec_components(model)
    nU = s["nU"]
    def mn(x): return float(x[nU:].norm(dim=-1).mean())
    return {"||cge_i||": mn(s["cge"]), "||norm(v)_i||": mn(s["nv"]),
            "||norm(t)_i||": mn(s["nt"]), "||av_i||": mn(s["av"]), "||at_i||": mn(s["at"]),
            "alpha": s["alpha"]}


# --------------------------------------------------------------- LightGCN
@torch.no_grad()
def lightgcn_variants(model):
    # LightGCN.forward()/full_sort_predict gives user/item all-embeddings.
    u, i = model.forward() if hasattr(model, "forward") else (None, None)
    if u is None:
        # fall back: reconstruct from full_sort_predict internals not exposed; skip.
        raise RuntimeError("LightGCN forward() not available")
    return {"baseline": (u.detach(), i.detach())}


def load_mde(model_name: str) -> float | None:
    f = ROOT / "results" / "phase0" / "seed_variance.json"
    if not f.is_file():
        return None
    summ = json.loads(f.read_text()).get("summary", {})
    return summ.get(model_name, {}).get("Recall@20", {}).get("MDE_2std")


def run_model(model_name: str, dataset: str, device: str) -> dict:
    cfg, ds, model, test_loader = load_frozen(model_name, dataset, device)
    if model_name == "freedom":
        streams = freedom_streams(model)
        variants = freedom_variants(streams)
        attribution = freedom_attribution(streams)
        attribution["image_weight"] = model.mm_image_weight
        recon_err = streams["recon_max_err"]
    elif model_name == "lgmrec":
        variants = lgmrec_variants(model)
        attribution, recon_err = lgmrec_attribution(model), None
    elif model_name == "lightgcn":
        variants = lightgcn_variants(model)
        attribution, recon_err = {}, None
    else:
        raise ValueError(model_name)

    # Evaluate every variant; compare to baseline.
    base_metrics, base_topk, _ = evaluate_item_matrix(*variants["baseline"], test_loader, device)
    mde = load_mde(model_name)
    rows = {"baseline": {"metrics": {k: float(v) for k, v in base_metrics.items()}}}
    for name, (u, i) in variants.items():
        if name == "baseline":
            continue
        m, topk, _ = evaluate_item_matrix(u, i, test_loader, device)
        delta = {k: float(m[k]) - float(base_metrics[k]) for k in m}
        rc = ranking_change(base_topk, topk, k=20)
        rows[name] = {
            "metrics": {k: float(v) for k, v in m.items()},
            "delta": delta,
            "dR@20": delta.get("Recall@20"), "dN@20": delta.get("NDCG@20"),
            "MDE_R@20": mde,
            "significant_vs_MDE": (abs(delta.get("Recall@20", 0)) > mde) if mde else None,
            "ranking_change": rc,
        }
    return {"model": model_name, "dataset": dataset, "checkpoint": getattr(model, "_ckpt_name", None),
            "n_users": ds.n_users, "n_items": ds.n_items,
            "MDE_R@20": mde, "attribution": attribution, "recon_max_err": recon_err,
            "variants": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["freedom", "lgmrec", "lightgcn"])
    ap.add_argument("--dataset", default="baby")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    out_path = OUT / "knockout_modality.json"
    existing = {}
    if out_path.is_file():
        for r in json.loads(out_path.read_text()):
            existing[f"{r['model']}/{r['dataset']}"] = r

    for m in args.models:
        print(f"\n=== {m} / {args.dataset} ===", flush=True)
        try:
            r = run_model(m, args.dataset, device)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            existing[f"{m}/{args.dataset}"] = {"model": m, "dataset": args.dataset, "error": repr(e)}
            continue
        existing[f"{m}/{args.dataset}"] = r
        base = r["variants"]["baseline"]["metrics"]
        print(f"  baseline R@20={base['Recall@20']:.4f} N@20={base['NDCG@20']:.4f} | MDE={r['MDE_R@20']}")
        for name, v in r["variants"].items():
            if name == "baseline":
                continue
            sig = v["significant_vs_MDE"]
            print(f"  {name:16} dR@20={v['dR@20']:+.4f} dN@20={v['dN@20']:+.4f}"
                  f" overlap@20={v['ranking_change']['overlap@20']:.3f}"
                  f" rbo={v['ranking_change']['rbo']:.3f}"
                  f" {'[SIG]' if sig else '[~noise]' if sig is not None else ''}", flush=True)
        if r["attribution"]:
            print(f"  attribution: {r['attribution']}")
        out_path.write_text(json.dumps(list(existing.values()), indent=2))

    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
