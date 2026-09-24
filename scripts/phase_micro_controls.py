"""Step-1b controls demanded by the adversarial audit of 2026-09-04.

Three questions, none of which needs training, all of which the MicroLens headline
("the on-path architecture flips to image-dominant, the frozen-graph one does not")
depends on:

  A. ALLOCATION vs ROUTING (audit Attack 1).  FREEDOM allocates image:text = 1:10 by the
     hard-coded lambda=0.1; LGMRec allocates 1:1 by construction (F.normalize(v)+F.normalize(t),
     both unit norm to 7 digits).  So the observed contrast is confounded: it may be
     allocation, not routing.  We sweep the ALLOCATION of both architectures at inference
     and trace |dR_img| / |dR_txt| as a function of it.  If FREEDOM reaches image-dominance
     when given LGMRec's 1:1 split, the headline is allocation and must be restated.  If
     LGMRec stays image-dominant when squeezed to FREEDOM's 1:10 split, routing is real.

  B. GUMBEL SPREAD (audit Attack 2).  LGMRec's hypergraph is resampled every forward
     (F.gumbel_softmax is a functional, so model.eval() does not disable it).  The reported
     flip rests on ONE draw (GUMBEL_SEED=12345).  We repeat the knockout across many draws
     and report sd[d_img], sd[d_txt] and sd[d_img - d_txt].  If that sd is comparable to the
     0.00345 gap, the flip is not established.

  C. CONTINUOUS RANK STATISTIC (PREREG s.5 bullet 2, not previously implemented).  R@20 on
     MicroLens is near-binary per user, so the null is weakly resolved.  We report the
     per-user paired mean delta log2(rank) of held-out items at K=1000, which has orders of
     magnitude more effective n.

Outputs -> results/phase_micro/controls_microlens.json.  Recsys stays READ-ONLY.
"""
from __future__ import annotations

import argparse
import json
import statistics
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
from recsys_bridge import load_frozen                            # noqa: E402
from ranking_effects import evaluate_item_matrix                 # noqa: E402
from phase1_knockout import freedom_streams, lgmrec_components   # noqa: E402
from src.data.graph_utils import build_knn_graph                 # noqa: E402

OUT = ROOT / "results" / "phase_micro"
RANK_K = 1000


# ------------------------------------------------------------------ allocation sweeps
@torch.no_grad()
def freedom_allocation_sweep(model, test_loader, device, lambdas) -> list:
    """Re-mix FREEDOM's frozen graph at each lambda, then knock out each stream AT that
    lambda. Confound to state: the embeddings were TRAINED at lambda=0.1, so this is an
    inference-time re-allocation, not a retrained one (that is Step 4)."""
    E = model.item_id_embedding.weight.detach()
    img = build_knn_graph(model.v_feat.cpu(), model.knn_k).to(device)
    txt = build_knn_graph(model.t_feat.cpu(), model.knn_k).to(device)
    g_img, g_txt = torch.sparse.mm(img, E), torch.sparse.mm(txt, E)
    ego = torch.cat([model.user_embedding.weight, model.item_id_embedding.weight], 0).detach()
    embs = [ego]
    for _ in range(model.n_layers):
        ego = torch.sparse.mm(model.norm_adj, ego); embs.append(ego)
    u, cf = torch.split(torch.stack(embs, 1).mean(1), [model.n_users, model.n_items], 0)

    rows = []
    for lam in lambdas:
        hi, ht = lam * g_img, (1 - lam) * g_txt
        base, _, _ = evaluate_item_matrix(u, cf + hi + ht, test_loader, device)
        ki, _, _ = evaluate_item_matrix(u, cf + ht, test_loader, device)       # drop image
        kt, _, _ = evaluate_item_matrix(u, cf + hi, test_loader, device)       # drop text
        di = float(ki["Recall@20"]) - float(base["Recall@20"])
        dt = float(kt["Recall@20"]) - float(base["Recall@20"])
        rows.append({"allocation_lambda": lam,
                     "alloc_ratio_img_over_txt": lam / (1 - lam) if lam < 1 else None,
                     "R@20": float(base["Recall@20"]),
                     "d_img": di, "d_txt": dt,
                     "abs_ratio_img_over_txt": abs(di) / abs(dt) if dt else None,
                     "image_dominant": abs(di) > abs(dt),
                     "norm_h_img": float(hi.norm(dim=-1).mean()),
                     "norm_h_txt": float(ht.norm(dim=-1).mean())})
        print(f"  FREEDOM lam={lam:.2f} R@20={rows[-1]['R@20']:.5f} d_img={di:+.6f} "
              f"d_txt={dt:+.6f} |img|/|txt|={rows[-1]['abs_ratio_img_over_txt']:.3f} "
              f"{'IMAGE-DOM' if rows[-1]['image_dominant'] else ''}", flush=True)
    return rows


@torch.no_grad()
def lgmrec_combine_weighted(s, keep_v, keep_t, w_img, w_txt):
    """LGMRec's own combination with an EXPLICIT modality allocation. w_img=w_txt=1 is the
    published model exactly (F.normalize(v) + F.normalize(t) -> equal unit norms)."""
    allv = s["cge"].clone()
    if keep_v:
        allv = allv + w_img * s["nv"]
    if keep_t:
        allv = allv + w_txt * s["nt"]
    terms = ([w_img * s["av"]] if keep_v else []) + ([w_txt * s["at"]] if keep_t else [])
    if terms:
        ghe = terms[0] if len(terms) == 1 else terms[0] + terms[1]
        allv = allv + s["alpha"] * F.normalize(ghe, dim=-1)
    return allv[:s["nU"]], allv[s["nU"]:]


@torch.no_grad()
def lgmrec_allocation_sweep(model, test_loader, device, w_imgs, gumbel_seed) -> list:
    s = lgmrec_components(model, seed=gumbel_seed)
    rows = []
    for wi in w_imgs:
        wt = 2.0 - wi   # hold total content mass fixed at the published 1+1
        base, _, _ = evaluate_item_matrix(*lgmrec_combine_weighted(s, True, True, wi, wt),
                                          test_loader, device)
        ki, _, _ = evaluate_item_matrix(*lgmrec_combine_weighted(s, False, True, wi, wt),
                                        test_loader, device)
        kt, _, _ = evaluate_item_matrix(*lgmrec_combine_weighted(s, True, False, wi, wt),
                                        test_loader, device)
        di = float(ki["Recall@20"]) - float(base["Recall@20"])
        dt = float(kt["Recall@20"]) - float(base["Recall@20"])
        rows.append({"w_img": wi, "w_txt": wt,
                     "alloc_ratio_img_over_txt": wi / wt,
                     "R@20": float(base["Recall@20"]), "d_img": di, "d_txt": dt,
                     "abs_ratio_img_over_txt": abs(di) / abs(dt) if dt else None,
                     "image_dominant": abs(di) > abs(dt)})
        print(f"  LGMRec w_img={wi:.2f}:w_txt={wt:.2f} R@20={rows[-1]['R@20']:.5f} "
              f"d_img={di:+.6f} d_txt={dt:+.6f} "
              f"|img|/|txt|={rows[-1]['abs_ratio_img_over_txt']:.3f} "
              f"{'IMAGE-DOM' if rows[-1]['image_dominant'] else ''}", flush=True)
    return rows


# ------------------------------------------------------------------ gumbel spread
@torch.no_grad()
def lgmrec_gumbel_spread(model, test_loader, device, seeds) -> dict:
    per = []
    for sd in seeds:
        s = lgmrec_components(model, seed=sd)
        base, _, _ = evaluate_item_matrix(*lgmrec_combine_weighted(s, True, True, 1.0, 1.0),
                                          test_loader, device)
        ki, _, _ = evaluate_item_matrix(*lgmrec_combine_weighted(s, False, True, 1.0, 1.0),
                                        test_loader, device)
        kt, _, _ = evaluate_item_matrix(*lgmrec_combine_weighted(s, True, False, 1.0, 1.0),
                                        test_loader, device)
        b = float(base["Recall@20"])
        di = float(ki["Recall@20"]) - b
        dt = float(kt["Recall@20"]) - b
        per.append({"gumbel_seed": sd, "R@20": b, "d_img": di, "d_txt": dt,
                    "gap_img_minus_txt": abs(di) - abs(dt), "image_dominant": abs(di) > abs(dt)})
        print(f"  gumbel seed={sd:<7d} R@20={b:.5f} d_img={di:+.6f} d_txt={dt:+.6f} "
              f"gap={per[-1]['gap_img_minus_txt']:+.6f} "
              f"{'IMAGE-DOM' if per[-1]['image_dominant'] else 'text-dom'}", flush=True)
    def sd_of(k):
        return statistics.stdev([r[k] for r in per]) if len(per) > 1 else None
    gaps = [r["gap_img_minus_txt"] for r in per]
    return {"per_seed": per, "n_seeds": len(per),
            "mean_R@20": statistics.mean([r["R@20"] for r in per]), "sd_R@20": sd_of("R@20"),
            "mean_d_img": statistics.mean([r["d_img"] for r in per]), "sd_d_img": sd_of("d_img"),
            "mean_d_txt": statistics.mean([r["d_txt"] for r in per]), "sd_d_txt": sd_of("d_txt"),
            "mean_gap": statistics.mean(gaps), "sd_gap": sd_of("gap_img_minus_txt"),
            "gap_over_sd": (statistics.mean(gaps) / sd_of("gap_img_minus_txt"))
                           if sd_of("gap_img_minus_txt") else None,
            "n_image_dominant": sum(r["image_dominant"] for r in per)}


# ------------------------------------------------------------------ continuous rank stat
@torch.no_grad()
def paired_rank_stat(u_all, item_base, item_cond, test_loader, device, K=RANK_K) -> dict:
    """Mean per-positive delta log2(rank) = log2(rank_knockout) - log2(rank_base), ranks
    capped at K. POSITIVE = the knockout pushed held-out items DOWN the list, i.e. the
    stream was HELPING. Paired across the identical user set."""
    d_all, n_capped_or, n_capped_and, n_pos = [], 0, 0, 0
    for batch in test_loader:
        uid = batch["user_ids"].to(device, non_blocking=True)
        hi = batch["history_indices"].to(device, non_blocking=True)
        hv = batch["history_values"].to(device, non_blocking=True)
        # positive_items is a LIST of np.ndarray, one per user in the batch (variable length)
        pos_list = batch["positive_items"]
        ranks = []
        for item_m in (item_base, item_cond):
            sc = u_all[uid] @ item_m.t()
            if hi.numel() > 0:
                m = hv.bool()
                row = torch.arange(sc.size(0), device=device).unsqueeze(1).expand_as(hi)
                safe = torch.where(hi >= 0, hi, torch.zeros_like(hi))
                sc[row[m], safe[m]] = float("-inf")
            # rank of an item = 1 + #items scoring strictly higher
            rows_r = []
            for bi, arr in enumerate(pos_list):
                if len(arr) == 0:
                    continue
                items = torch.as_tensor(np.asarray(arr), dtype=torch.long, device=device)
                s_pos = sc[bi, items]
                rr = 1 + (sc[bi].unsqueeze(0) > s_pos.unsqueeze(1)).sum(1)
                rows_r.append(rr)
            ranks.append(torch.cat(rows_r) if rows_r else torch.empty(0, device=device))
        rb, rc = ranks
        if rb.numel() == 0:
            continue
        n_pos += rb.numel()
        # OR = censored in at least one condition; AND = censored in BOTH, and only the
        # AND set contributes exactly 0 after clamping. The earlier record conflated them.
        n_capped_or += int(((rb > K) | (rc > K)).sum().item())
        n_capped_and += int(((rb > K) & (rc > K)).sum().item())
        rbc = rb.clamp(max=K).double(); rcc = rc.clamp(max=K).double()
        d_all.append((rcc.log2() - rbc.log2()).cpu())
    d = torch.cat(d_all)
    n = d.numel()
    mean = float(d.mean()); sd = float(d.std())
    se = sd / (n ** 0.5)
    return {"K_cap": K, "n_positives": int(n),
            "n_censored_either_condition": int(n_capped_or),
            "n_censored_both_conditions": int(n_capped_and),
            "frac_contributing_exactly_zero": n_capped_and / n if n else None,
            "mean_delta_log2_rank": mean, "sd": sd, "se": se,
            "t_stat": mean / se if se else None,
            "ci95": [mean - 1.96 * se, mean + 1.96 * se],
            "mean_rank_degradation_pct": 100.0 * (2.0 ** mean - 1.0),
            "note": ("POSITIVE = knockout pushed held-out items DOWN in rank "
                     "(the stream was helping). Only the both-censored set contributes "
                     "exactly 0; one-sided censoring truncates the effect toward 0, so the "
                     "estimate is conservative.")}


@torch.no_grad()
def rank_stat_null(u_all, fused, stream, base_item, test_loader, device, seed=0) -> dict:
    """The control the audit demanded: with n~1e5 positives, ANY additive stream that is not
    exactly orthogonal to the ranking gives a hugely significant t. So t=30 on h_img only
    establishes h_img != 0 unless we show a CONTENT-FREE stream of the same size does less.

    Two nulls, both preserving h_img's size:
      permuted : h_img's rows shuffled across items -- identical norm distribution and
                 identical marginal geometry, item<->content correspondence destroyed.
      gaussian : isotropic noise rescaled to h_img's exact per-item norms.
    We knock each null out of a fused representation built with it in h_img's place."""
    g = torch.Generator(device=stream.device).manual_seed(seed)
    norms = stream.norm(dim=-1, keepdim=True)
    perm = stream[torch.randperm(stream.shape[0], generator=g, device=stream.device)]
    gau = torch.randn(stream.shape, generator=g, device=stream.device, dtype=stream.dtype)
    gau = gau / gau.norm(dim=-1, keepdim=True).clamp_min(1e-9) * norms
    out = {}
    for nm, alt in (("permuted_h_img", perm), ("gaussian_matched_norm", gau)):
        out[nm] = paired_rank_stat(u_all, base_item + alt, base_item, test_loader, device)
        r = out[nm]
        print(f"  NULL {nm:22s} mean dlog2(rank)={r['mean_delta_log2_rank']:+.5f} "
              f"t={r['t_stat']:+.2f}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="microlens")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpt-map", default=str(OUT / "ckpt_pins.json"))
    ap.add_argument("--gumbel-seeds", type=int, default=12)
    ap.add_argument("--skip", nargs="*", default=[])
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    raw = json.loads(Path(args.ckpt_map).read_text())
    pins = {k: (v["path"] if isinstance(v, dict) else v) for k, v in raw.items() if not k.startswith("_")}

    out = {"dataset": args.dataset,
           "purpose": "controls demanded by the 2026-09-04 adversarial audit (Attacks 1-2, PREREG s.5)",
           "rbo_params": {"p": 0.9, "k": 20, "identical_list_ceiling": 1 - 0.9 ** 20}}
    out_path = OUT / "controls_microlens.json"

    if "freedom" not in args.skip:
        print("\n=== A1. FREEDOM allocation sweep (inference-time re-allocation) ===", flush=True)
        _, _, fm, tl = load_frozen("freedom", args.dataset, device,
                                   ckpt_path=pins.get(f"freedom/{args.dataset}"))
        out["freedom_checkpoint"] = getattr(fm, "_ckpt_name", None)
        out["freedom_allocation_sweep"] = freedom_allocation_sweep(
            fm, tl, device, [0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
        print("\n=== C. FREEDOM continuous rank statistic (K=1000) ===", flush=True)
        s = freedom_streams(fm)
        out["freedom_rank_stat"] = {
            "image_knockout": paired_rank_stat(s["u_all"], s["fused"], s["cf"] + s["h_txt"], tl, device),
            "text_knockout": paired_rank_stat(s["u_all"], s["fused"], s["cf"] + s["h_img"], tl, device)}
        for k, v in out["freedom_rank_stat"].items():
            print(f"  {k:16s} mean dlog2(rank)={v['mean_delta_log2_rank']:+.5f} "
                  f"CI95[{v['ci95'][0]:+.5f},{v['ci95'][1]:+.5f}] t={v['t_stat']:+.2f} "
                  f"n={v['n_positives']}", flush=True)
        print("\n  -- matched-size nulls (does ANY stream of this norm give t~30?) --", flush=True)
        out["freedom_rank_stat_nulls"] = rank_stat_null(
            s["u_all"], s["fused"], s["h_img"], s["cf"] + s["h_txt"], tl, device)
        out_path.write_text(json.dumps(out, indent=2))
        del fm, tl; torch.cuda.empty_cache()

    if "lgmrec" not in args.skip:
        _, _, lm, tl = load_frozen("lgmrec", args.dataset, device,
                                   ckpt_path=pins.get(f"lgmrec/{args.dataset}"))
        out["lgmrec_checkpoint"] = getattr(lm, "_ckpt_name", None)
        print("\n=== B. LGMRec gumbel-draw spread (the flip rests on ONE draw) ===", flush=True)
        seeds = [12345] + [1000 + i for i in range(args.gumbel_seeds - 1)]
        out["lgmrec_gumbel_spread"] = lgmrec_gumbel_spread(lm, tl, device, seeds)
        g = out["lgmrec_gumbel_spread"]
        print(f"  --> mean gap={g['mean_gap']:+.6f} sd={g['sd_gap']:.6f} "
              f"gap/sd={g['gap_over_sd']:.1f} | image-dominant in "
              f"{g['n_image_dominant']}/{g['n_seeds']} draws", flush=True)
        print("\n=== A2. LGMRec allocation sweep (2.0=all image ... 0.2=FREEDOM's 1:9) ===", flush=True)
        out["lgmrec_allocation_sweep"] = lgmrec_allocation_sweep(
            lm, tl, device, [0.2, 0.5, 1.0, 1.5, 1.8], gumbel_seed=12345)
        out_path.write_text(json.dumps(out, indent=2))

    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
