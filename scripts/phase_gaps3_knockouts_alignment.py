"""Plan gaps (Phase 1/3):

(A) LGMRec finer per-stream knockouts: zero ONLY the hypergraph term (ghe) keeping
    modality streams, and zero ONLY the modality streams (mge) keeping ghe — the two
    conditions the plan listed that phase1 didn't run separately.
(B) FREEDOM aux-projection sanity check (plan: "expect ~0 effect at inference"):
    zero image_trs/text_trs weights and verify full_sort_predict is unchanged.
(C) ALIGNMENT metric (Wang & Isola): E‖f(x)−f(x⁺)‖² over CO-PURCHASED item pairs vs
    random pairs, per stream — both on the model's 64-d streams AND on the RAW
    4096-d image / 384-d text features. The behavioral-relevance gap
    (random − copurchased) measures how much a stream's similarity predicts
    co-purchase. Prediction from the retrain collapse (H2): image gap << text gap.

Outputs -> results/phase1/knockout_finer.json, results/phase3/alignment.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/workspace/MechInterp")
RECSYS = Path("/workspace/Recsys")
sys.path.insert(0, str(RECSYS))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "interp"))
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_bridge import load_frozen                        # noqa: E402
from ranking_effects import evaluate_item_matrix             # noqa: E402
from phase1_knockout import freedom_streams, lgmrec_components  # noqa: E402


# ---------------------------------------------------------------- (A) LGMRec finer
@torch.no_grad()
def lgmrec_finer(dataset: str, device: str) -> dict:
    cfg, ds, model, test_loader = load_frozen("lgmrec", dataset, device)
    s = lgmrec_components(model)
    nU, alpha = s["nU"], s["alpha"]

    def combine(use_mge: bool, use_ghe: bool):
        allv = s["cge"].clone()
        if use_mge:
            allv = allv + s["nv"] + s["nt"]
        if use_ghe:
            allv = allv + alpha * F.normalize(s["av"] + s["at"], dim=-1)
        return allv[:nU], allv[nU:]

    conds = {"baseline": combine(True, True),
             "ghe_knockout(keep mge)": combine(True, False),
             "mge_knockout(keep ghe)": combine(False, True),
             "cge_only": combine(False, False)}
    base_m, _, _ = evaluate_item_matrix(*conds["baseline"], test_loader, device)
    out = {"baseline_R@20": float(base_m["Recall@20"])}
    for nm, (u, i) in conds.items():
        if nm == "baseline":
            continue
        m, _, _ = evaluate_item_matrix(u, i, test_loader, device)
        out[nm] = {"R@20": float(m["Recall@20"]),
                   "dR@20": float(m["Recall@20"]) - float(base_m["Recall@20"])}
    return out


# ---------------------------------------------------------------- (B) FREEDOM aux sanity
@torch.no_grad()
def freedom_aux_sanity(dataset: str, device: str) -> dict:
    cfg, ds, model, test_loader = load_frozen("freedom", dataset, device)
    def evl():
        s = freedom_streams(model)
        m, _, _ = evaluate_item_matrix(s["u_all"], s["fused"], test_loader, device)
        return float(m["Recall@20"])
    base = evl()
    model.image_trs.weight.zero_(); model.image_trs.bias.zero_()
    model.text_trs.weight.zero_(); model.text_trs.bias.zero_()
    after = evl()
    return {"baseline_R@20": base, "aux_zeroed_R@20": after, "dR@20": after - base,
            "expected": "exactly 0 (aux projections are not on the inference path)"}


# ---------------------------------------------------------------- (C) alignment
@torch.no_grad()
def copurchase_pairs(ds, n_pairs: int, seed: int = 0):
    """Sample item pairs co-purchased by the same user from train interactions."""
    rng = np.random.default_rng(seed)
    users = np.asarray(ds.train_users); items = np.asarray(ds.train_items)
    order = np.argsort(users, kind="stable")
    u_sorted, i_sorted = users[order], items[order]
    starts = np.searchsorted(u_sorted, np.unique(u_sorted))
    bounds = np.append(starts, len(u_sorted))
    pairs = []
    uids = rng.permutation(len(starts))
    for k in uids:
        lo, hi = bounds[k], bounds[k + 1]
        if hi - lo < 2:
            continue
        a, b = rng.choice(np.arange(lo, hi), size=2, replace=False)
        if i_sorted[a] != i_sorted[b]:
            pairs.append((i_sorted[a], i_sorted[b]))
        if len(pairs) >= n_pairs:
            break
    p = np.array(pairs)
    return torch.from_numpy(p[:, 0]).long(), torch.from_numpy(p[:, 1]).long()


@torch.no_grad()
def alignment_report(X: torch.Tensor, ia, ib, n_rand: int = 20000, seed: int = 0) -> dict:
    f = F.normalize(X.float(), dim=-1)
    co = (f[ia] - f[ib]).pow(2).sum(-1).mean().item()
    g = torch.Generator().manual_seed(seed)
    ra = torch.randint(0, X.shape[0], (n_rand,), generator=g)
    rb = torch.randint(0, X.shape[0], (n_rand,), generator=g)
    keep = ra != rb
    rand = (f[ra[keep]] - f[rb[keep]]).pow(2).sum(-1).mean().item()
    return {"align_copurchased": co, "align_random": rand,
            "behavioral_gap(random-co)": rand - co,
            "relative_gap": (rand - co) / max(rand, 1e-9)}


@torch.no_grad()
def alignment(dataset: str, device: str, n_pairs: int = 20000) -> dict:
    cfg, ds, model, _ = load_frozen("freedom", dataset, device)
    ia, ib = copurchase_pairs(ds, n_pairs)
    s = freedom_streams(model)
    img_raw64 = model.image_trs(model.image_embedding.weight).detach()
    txt_raw64 = model.text_trs(model.text_embedding.weight).detach()
    streams = {
        "raw_image_4096d": model.v_feat.to(device),
        "raw_text_384d": model.t_feat.to(device),
        "img_proj_64d(trained aux)": img_raw64,
        "txt_proj_64d(trained aux)": txt_raw64,
        "h_img(graph)": s["h_img"], "h_txt(graph)": s["h_txt"],
        "cf": s["cf"], "fused": s["fused"],
    }
    ia, ib = ia.to(device), ib.to(device)
    return {"n_copurchase_pairs": int(ia.numel()),
            "per_stream": {nm: alignment_report(X, ia, ib) for nm, X in streams.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["baby", "sports", "clothing", "elec"])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    p1 = ROOT / "results" / "phase1" / "knockout_finer.json"
    p3 = ROOT / "results" / "phase3" / "alignment.json"
    e1 = json.loads(p1.read_text()) if p1.is_file() else {}
    e3 = json.loads(p3.read_text()) if p3.is_file() else {}

    for dsname in args.datasets:
        print(f"\n===== {dsname} =====", flush=True)
        try:
            r = lgmrec_finer(dsname, device)
            e1.setdefault(dsname, {})["lgmrec_finer"] = r
            print("  [A] LGMRec finer:", {k: (v if isinstance(v, float) else v["dR@20"])
                                          for k, v in r.items()})
        except Exception as ex:  # noqa: BLE001
            print("  [A] failed:", ex)
        try:
            r = freedom_aux_sanity(dsname, device)
            e1.setdefault(dsname, {})["freedom_aux_sanity"] = r
            print(f"  [B] FREEDOM aux-zero sanity: dR@20={r['dR@20']:+.6f} (expect 0)")
        except Exception as ex:  # noqa: BLE001
            print("  [B] failed:", ex)
        p1.write_text(json.dumps(e1, indent=2))
        try:
            r = alignment(dsname, device)
            e3[dsname] = r
            print("  [C] behavioral alignment gap (random − co-purchased), higher = more behaviorally relevant:")
            for nm, a in r["per_stream"].items():
                print(f"      {nm:28} gap={a['behavioral_gap(random-co)']:.4f} (rel {a['relative_gap']:.3f})")
        except Exception as ex:  # noqa: BLE001
            print("  [C] failed:", ex)
        p3.write_text(json.dumps(e3, indent=2))
    print(f"\nWrote {p1} and {p3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
