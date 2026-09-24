"""Exact structural knockout for MENTOR (Xu et al., AAAI 2024) — clean-room impl at
`/workspace/Recsys/src/models/mentor.py`.

WHY IT IS EXACT
---------------
MENTOR's inference path is `full_sort_predict -> _propagate`, and `_propagate` ends with

    mm  = F.normalize(image_item, dim=-1) + F.normalize(text_item, dim=-1)
    i_e = i_id + 0.5 * mm

i.e.

    i_e   = i_id_lightgcn  +  0.5*normalize(image_item)  +  0.5*normalize(text_item)
          = cf             +  h_img                     +  h_txt
    u_all = u_lightgcn                                   (no modality anywhere)

The two content streams are normalised SEPARATELY (per-modality), never jointly, and
the ID/CF term sits outside every normaliser. So each content stream is an *additive,
independently removable* term: deleting one leaves the other two bit-identical.
That makes the knockout EXACT — no renormalisation bracket needed, no input-mean
substitution, no touching of the surviving pathway (u_all and cf are untouched in
every arm).

THE SHARP CONSEQUENCE (verified in `attribution`)
-------------------------------------------------
||h_img|| = ||0.5 * normalize(image_item)|| = 0.5 for EVERY item, and identically for
text. MENTOR therefore *clamps the two content streams to exactly equal magnitude by
construction*, per item, with no learnable rescaling. Any image/text asymmetry measured
in MENTOR is therefore **purely DIRECTIONAL** (where the stream points relative to the
CF geometry), and cannot be a magnitude/allocation effect — unlike FREEDOM, whose
`mm_image_weight` λ is exactly an allocation knob. `attribution` reports
min/max/std of both stream norms so the reader can check the clamp holds numerically.

Because the clamp makes the permutation null EXACTLY norm-matched (a permuted h_img has
the same 0.5 norm as the true h_img, for every row), we also emit paired, fixed-seed
`image_permute` / `text_permute` control arms. They are CONTROLS, not knockouts: they
keep the mass and destroy only the item-specific direction.

  WARNING — a permute arm is ONE DRAW of a random variable and must not be read alone.
  Measured on baby over 8 permutation seeds (results/_scratch/mentor_diag.json):
      image_permute dR@20 = -0.00253 +- 0.00076 (sd)
      text_permute  dR@20 = -0.00236 +- 0.00062 (sd)
  The single PERMUTE_SEED draw (-0.00142 / -0.00279) sits ~1.5 sd either side of those
  means, so the apparent "image direction is worthless, text direction matters" reading
  of a single draw is NOT supported. Average over seeds before interpreting.

WHY THE COARSE SCREEN HAPPENED TO BE RIGHT HERE (and why that is luck)
---------------------------------------------------------------------
On baby this exact deletion is BIT-IDENTICAL to the published per-dimension input-mean
screen (dR@20 -0.0013443901602811226 / -0.0016678911214781655 / -0.0031275024795209827
for image/text/both, equal to 17 significant figures). Reason, measured in
results/_scratch/mentor_diag.json: feeding a constant feature makes every item's
image_item collinear, and the per-item unit normalisation then erases the only thing
that still differed (the kNN row-sum scale), so h_img becomes the SAME vector for every
item (max |h_img_i - mean| = 7.6e-07, min cos to the mean = 1.0). A constant added to
every item is a per-user constant score offset, which cannot reorder a ranking. So
mean-substitution is rank-equivalent to deletion *for MENTOR* — by accident of the
normaliser, not by design; on DAMRS the same degeneracy instead produced empty content
graphs and image_knockout == both_knockout. This module removes the dependence on that
accident.

OFF-PATH BRANCHES
-----------------
`align_loss` (cross-modal cosine gap) and `ssl_loss` (ID<->modality InfoNCE) live only
in `calculate_loss`; they shape training but are absent from `full_sort_predict`, so
they are correctly left untouched (CONTRACT rule 4). `image_trs` / `text_trs` are NOT
training-only: they are learnable Linears evaluated inside `_propagate`, hence on-path
and correctly included in the deleted stream. `image_adj` / `text_adj` are frozen
non-persistent kNN buffers rebuilt at `__init__` from the raw features; deleting a
stream deletes its graph propagation with it, so the arm removes the modality's whole
contribution (live features + its content graph), not just one of the two.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/workspace/MechInterp")
RECSYS = Path("/workspace/Recsys")
for _p in (str(RECSYS), str(ROOT / "src" / "models"), str(ROOT / "src" / "interp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- contract fields -------------------------------------------------------
# Content reaches the scoring function as a LIVE, additive, per-modality-normalised
# residual on top of the ID/CF item embedding (projected raw features propagated
# through a frozen content kNN graph, then unit-normalised and scaled by a fixed 0.5).
# On-path, additive, exactly deletable.
CLASS = "C2"
CLASS_DESC = ("live content features, additive per-modality-normalised residual on the "
              "ID embedding, fixed (non-learned) fusion coefficient 0.5")
EXACTNESS = "EXACT"

PERMUTE_SEED = 12345


# --------------------------------------------------------------------------- streams
@torch.no_grad()
def streams(model) -> dict:
    """Reproduce MENTOR._propagate, exposing every additive term separately.

    Uses the model's OWN frozen buffers (`image_adj`, `text_adj`, `norm_adj`) — we do
    not rebuild the kNN graphs, so there is no CPU/GPU tie-breaking hazard.
    """
    # modality streams: project, then propagate through the frozen content kNN graphs
    image_item = model.image_trs(model.v_feat)
    text_item = model.text_trs(model.t_feat)
    for _ in range(model.n_layers):
        image_item = torch.sparse.mm(model.image_adj, image_item)
        text_item = torch.sparse.mm(model.text_adj, text_item)

    # user-item LightGCN (modality-free)
    ego = torch.cat([model.user_embedding.weight, model.item_id_embedding.weight], dim=0)
    all_embs = [ego]
    for _ in range(model.n_ui_layers):
        ego = torch.sparse.mm(model.norm_adj, ego)
        all_embs.append(ego)
    out = torch.stack(all_embs, dim=1).mean(dim=1)
    u_e, i_id = torch.split(out, [model.n_users, model.n_items], dim=0)

    h_img = 0.5 * F.normalize(image_item, dim=-1)
    h_txt = 0.5 * F.normalize(text_item, dim=-1)
    fused = i_id + h_img + h_txt
    return {"u_all": u_e.detach(), "cf": i_id.detach(),
            "h_img": h_img.detach(), "h_txt": h_txt.detach(), "fused": fused.detach(),
            "image_item": image_item.detach(), "text_item": text_item.detach()}


# --------------------------------------------------------------------------- variants
@torch.no_grad()
def variants(model, dataset=None, device=None) -> dict:
    s = streams(model)
    u, cf, h_img, h_txt = s["u_all"], s["cf"], s["h_img"], s["h_txt"]

    g = torch.Generator(device="cpu").manual_seed(PERMUTE_SEED)
    n = cf.shape[0]
    perm_i = torch.randperm(n, generator=g).to(cf.device)
    perm_t = torch.randperm(n, generator=g).to(cf.device)

    return {
        # --- required arms: delete the modality's additive term (exact) ---------
        "baseline":       (u, s["fused"]),
        "image_knockout": (u, cf + h_txt),          # h_img deleted
        "text_knockout":  (u, cf + h_img),          # h_txt deleted
        "both_knockout":  (u, cf),                  # CF/ID only
        # --- controls: mass kept EXACTLY (norm is 0.5 for every row), direction
        #     scrambled. Only meaningful because MENTOR clamps the norms. ---------
        "image_permute":  (u, cf + h_img[perm_i] + h_txt),
        "text_permute":   (u, cf + h_img + h_txt[perm_t]),
    }


# --------------------------------------------------------------------------- attribution
@torch.no_grad()
def attribution(model, dataset=None, device=None) -> dict:
    s = streams(model)
    ni = s["h_img"].norm(dim=-1)
    nt = s["h_txt"].norm(dim=-1)
    ncf = s["cf"].norm(dim=-1)

    def cosm(a, b):
        return float(F.cosine_similarity(a, b, dim=-1).mean())

    return {
        # mean per-item norms of every additive stream
        "||cf||_mean": float(ncf.mean()),
        "||h_img||_mean": float(ni.mean()),
        "||h_txt||_mean": float(nt.mean()),
        "||fused||_mean": float(s["fused"].norm(dim=-1).mean()),
        # fusion coefficients (fixed by the architecture, NOT learned)
        "fusion_coeff_image": 0.5,
        "fusion_coeff_text": 0.5,
        "fusion_is_learned": False,
        # --- the magnitude clamp: these must be exactly 0.5 with std 0 ----------
        "||h_img||_min": float(ni.min()), "||h_img||_max": float(ni.max()),
        "||h_img||_std": float(ni.std()),
        "||h_txt||_min": float(nt.min()), "||h_txt||_max": float(nt.max()),
        "||h_txt||_std": float(nt.std()),
        "n_zero_norm_image_rows": int((s["image_item"].norm(dim=-1) == 0).sum()),
        "n_zero_norm_text_rows": int((s["text_item"].norm(dim=-1) == 0).sum()),
        "magnitude_clamped_equal": bool(torch.allclose(ni, nt, atol=1e-6)),
        # pre-normalisation scale (what the clamp throws away)
        "||image_item||_mean_prenorm": float(s["image_item"].norm(dim=-1).mean()),
        "||text_item||_mean_prenorm": float(s["text_item"].norm(dim=-1).mean()),
        # --- so the asymmetry can only be directional: report the directions ----
        "cos(h_img,h_txt)_mean": cosm(s["h_img"], s["h_txt"]),
        "cos(h_img,cf)_mean": cosm(s["h_img"], s["cf"]),
        "cos(h_txt,cf)_mean": cosm(s["h_txt"], s["cf"]),
        "content_to_cf_norm_ratio": float((ni + nt).mean() / ncf.mean()),
        # off-path certificate: training-only heads
        "training_only_branches": ["align_loss", "ssl_loss (InfoNCE)"],
        "on_path_content_params": ["image_trs", "text_trs", "image_adj", "text_adj"],
    }


# --------------------------------------------------------------------------- recon
@torch.no_grad()
def recon_error(model, dataset=None, device=None, chunk: int = 512) -> float:
    """max |baseline_scores - MENTOR.full_sort_predict| over ALL users."""
    s = variants(model)["baseline"]
    u_all, item_mat = s
    dev = item_mat.device
    worst = 0.0
    for lo in range(0, model.n_users, chunk):
        uid = torch.arange(lo, min(lo + chunk, model.n_users), device=dev)
        mine = u_all[uid] @ item_mat.t()
        theirs = model.full_sort_predict({"user": uid})
        worst = max(worst, float((mine - theirs).abs().max()))
        del mine, theirs
    return worst


# --------------------------------------------------------------------------- self-test
def _selftest(dataset: str = "baby", gpu: int = 0, ckpt: str | None = None) -> int:
    import json
    from recsys_bridge import load_frozen                                  # noqa: E402
    from ranking_effects import evaluate_item_matrix, ranking_change       # noqa: E402

    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    cfg, ds, model, test_loader = load_frozen("mentor", dataset, device, ckpt_path=ckpt)
    print(f"ckpt = {model._ckpt_path}  (pinned={model._ckpt_pinned})", flush=True)

    err = recon_error(model, ds, device)
    print(f"recon_error (vs full_sort_predict, all users) = {err:.3e}", flush=True)

    attr = attribution(model, ds, device)
    print("attribution:", json.dumps(attr, indent=2), flush=True)

    vs = variants(model, ds, device)
    res, base_topk = {}, None
    for name, (u, im) in vs.items():
        m, tk, _ = evaluate_item_matrix(u, im, test_loader, device)
        m = {k: float(v) for k, v in m.items()}
        if name == "baseline":
            base_topk = tk
            res[name] = m
        else:
            res[name] = {"delta": {k: m[k] - res["baseline"][k] for k in m},
                         "abs": m,
                         "rank": ranking_change(base_topk, tk, k=20)}
        print(f"  {name:16s} R@20={m['Recall@20']:.6f}"
              + ("" if name == "baseline"
                 else f"  dR@20={m['Recall@20']-res['baseline']['Recall@20']:+.6f}"),
              flush=True)

    logged = json.loads((RECSYS / "logs" / Path(model._ckpt_name).stem /
                         "result.json").read_text())["test_result"]
    out = {"model": "mentor", "dataset": dataset, "CLASS": CLASS, "CLASS_DESC": CLASS_DESC,
           "EXACTNESS": EXACTNESS, "ckpt": model._ckpt_path,
           "recon_error": err, "attribution": attr, "arms": res,
           "logged_test_result": logged,
           "baseline_vs_logged_R@20": res["baseline"]["Recall@20"] - logged["Recall@20"]}
    outp = ROOT / "results" / "_scratch" / f"exact_ko_mentor_{dataset}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nlogged test R@20 = {logged['Recall@20']:.12f}")
    print(f"arm  baseline R@20 = {res['baseline']['Recall@20']:.12f}")
    print(f"diff             = {out['baseline_vs_logged_R@20']:.3e}")
    print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="baby")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()
    raise SystemExit(_selftest(a.dataset, a.gpu, a.ckpt))
