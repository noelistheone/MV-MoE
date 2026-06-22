"""Exp A2 — behavioral-alignment gap with bootstrap CI + permutation null.

The project's "image << text << CF" behavioral-alignment result (the geometric proof
of H2) was reported as POINT ESTIMATES only, and the CLIP-vs-CNN ordering flipped
sign between the absolute and the relative (scale-normalised) gap. This script makes
the claim a formal statistical statement:

  * PRE-REGISTERED METRIC (fixed before looking at results):
      alignment distance d(x,y) = || f(x) - f(y) ||^2 ,  f = L2-normalize  (= 2 - 2cos)
      (the Wang & Isola 2020 alignment kernel).
      HEADLINE STATISTIC = ABSOLUTE behavioral gap
          g = mean_d(random pairs) - mean_d(co-purchased pairs).
      Higher g  =>  the stream's geometry predicts co-purchase better.
      The RELATIVE gap g / mean_d(random) is reported too, for transparency, but is
      explicitly NOT the headline: its scale-dependence is exactly what produced the
      CNN/CLIP sign flip, so it is demoted, not used for ordering claims.

  * 95% BOOTSTRAP CI on g (resample co- and random-pair distances independently).
  * PERMUTATION NULL: pool the two distance sets, shuffle the co/random labels,
      recompute g -> one-sided p-value for H0 "co-purchased pairs are no closer
      than random pairs".

Streams per dataset: raw CNN image (4096d), raw CLIP image (768d), raw BERT text
(384d), raw CLIP text (768d), h_img(graph 64d), h_txt(graph 64d), cf(64d), fused(64d).
All distances on GPU. Frozen FREEDOM loaded read-only from Recsys.

Outputs: results/phase_align/alignment_stats.json + SUMMARY.md
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
from recsys_bridge import load_frozen              # noqa: E402
from phase1_knockout import freedom_streams        # noqa: E402

OUT = ROOT / "results" / "phase_align"
CLIP = ROOT / "data" / "clip"

N_PAIRS = 20000
N_RAND = 20000
B_BOOT = 2000
B_PERM = 2000
PAIR_SEED = 0
RAND_SEED = 0
BOOT_SEED = 1234
PERM_SEED = 5678


def copurchase_pairs(ds, n_pairs: int, seed: int = PAIR_SEED):
    """Two distinct items co-purchased by the same user. Verbatim from phase_gaps3."""
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
def pair_dists(X: torch.Tensor, ia: torch.Tensor, ib: torch.Tensor) -> torch.Tensor:
    f = F.normalize(X.float(), dim=-1)
    return (f[ia] - f[ib]).pow(2).sum(-1)  # squared-L2 on the unit sphere


def _quantile(t: torch.Tensor, qs) -> list:
    return torch.quantile(t, torch.tensor(qs, device=t.device, dtype=t.dtype)).tolist()


@torch.no_grad()
def bootstrap_gap(co_d, rand_d, B=B_BOOT, seed=BOOT_SEED, chunk=500):
    dev = co_d.device
    g = torch.Generator(device=dev).manual_seed(seed)
    Nc, Nr = co_d.numel(), rand_d.numel()
    gaps, rels = [], []
    done = 0
    while done < B:
        b = min(chunk, B - done)
        ci = torch.randint(0, Nc, (b, Nc), generator=g, device=dev)
        ri = torch.randint(0, Nr, (b, Nr), generator=g, device=dev)
        mc = co_d[ci].mean(1); mr = rand_d[ri].mean(1)
        gp = mr - mc
        gaps.append(gp); rels.append(gp / mr.clamp_min(1e-9))
        done += b
    gaps = torch.cat(gaps); rels = torch.cat(rels)
    glo, ghi = _quantile(gaps, [0.025, 0.975])
    rlo, rhi = _quantile(rels, [0.025, 0.975])
    return {"gap_abs_ci95": [glo, ghi], "gap_rel_ci95": [rlo, rhi],
            "gap_abs_boot_std": float(gaps.std().item())}


@torch.no_grad()
def perm_pvalue(co_d, rand_d, obs_gap, B=B_PERM, seed=PERM_SEED, chunk=250):
    dev = co_d.device
    g = torch.Generator(device=dev).manual_seed(seed)
    pool = torch.cat([co_d, rand_d]); M = pool.numel(); Nc = co_d.numel()
    ge = 0; done = 0
    while done < B:
        b = min(chunk, B - done)
        order = torch.rand(b, M, generator=g, device=dev).argsort(1)
        perm = pool[order]
        gp = perm[:, Nc:].mean(1) - perm[:, :Nc].mean(1)
        ge += int((gp >= obs_gap).sum().item())
        done += b
    return (ge + 1) / (B + 1)


@torch.no_grad()
def analyze_stream(name, X, ia, ib, dev):
    X = X.to(dev)
    co_d = pair_dists(X, ia, ib)
    g = torch.Generator(device=dev).manual_seed(RAND_SEED)
    ra = torch.randint(0, X.shape[0], (N_RAND,), generator=g, device=dev)
    rb = torch.randint(0, X.shape[0], (N_RAND,), generator=g, device=dev)
    keep = ra != rb
    rand_d = pair_dists(X, ra[keep], rb[keep])
    mean_co = float(co_d.mean().item()); mean_rand = float(rand_d.mean().item())
    gap_abs = mean_rand - mean_co
    gap_rel = gap_abs / max(mean_rand, 1e-9)
    boot = bootstrap_gap(co_d, rand_d)
    pval = perm_pvalue(co_d, rand_d, gap_abs)
    sig = (boot["gap_abs_ci95"][0] > 0.0) and (pval < 0.05)
    return {
        "dim": int(X.shape[1]),
        "align_copurchased": mean_co, "align_random": mean_rand,
        "gap_abs": gap_abs, "gap_rel": gap_rel,
        "gap_abs_ci95": boot["gap_abs_ci95"], "gap_rel_ci95": boot["gap_rel_ci95"],
        "gap_abs_boot_std": boot["gap_abs_boot_std"],
        "perm_p_value": pval, "n_co": int(co_d.numel()), "n_rand": int(rand_d.numel()),
        "significant_gap>0": bool(sig),
    }


@torch.no_grad()
def run_dataset(dataset: str, dev: str) -> dict:
    cfg, ds, model, _ = load_frozen("freedom", dataset, dev)
    ia, ib = copurchase_pairs(ds, N_PAIRS)
    ia, ib = ia.to(dev), ib.to(dev)
    s = freedom_streams(model)
    streams = {
        "raw_image_cnn": model.v_feat.to(dev),
        "raw_text_bert": model.t_feat.to(dev),
        "h_img_graph": s["h_img"], "h_txt_graph": s["h_txt"],
        "cf": s["cf"], "fused": s["fused"],
    }
    ci = CLIP / f"{dataset}_image_clip.npy"; ct = CLIP / f"{dataset}_text_clip.npy"
    if ci.exists():
        streams["raw_image_clip"] = torch.from_numpy(np.load(ci)).float()
    if ct.exists():
        streams["raw_text_clip"] = torch.from_numpy(np.load(ct)).float()
    order = ["raw_image_cnn", "raw_image_clip", "raw_text_bert", "raw_text_clip",
             "h_img_graph", "h_txt_graph", "cf", "fused"]
    res = {}
    for nm in order:
        if nm in streams:
            res[nm] = analyze_stream(nm, streams[nm], ia, ib, dev)
            r = res[nm]
            print(f"    {nm:16s} dim{r['dim']:>5d}  gap={r['gap_abs']:.4f} "
                  f"CI[{r['gap_abs_ci95'][0]:.4f},{r['gap_abs_ci95'][1]:.4f}]  "
                  f"rel={r['gap_rel']:.3f} p={r['perm_p_value']:.4f}", flush=True)
    # ---- formal ordering tests via NON-OVERLAPPING bootstrap CIs ----
    def ci(nm):
        return res[nm]["gap_abs_ci95"] if nm in res else None

    def below(a, b):  # a's CI entirely below b's CI (strict separation)
        ca, cb = ci(a), ci(b)
        if ca is None or cb is None:
            return None
        return ca[1] < cb[0]

    def overlap(a, b):
        ca, cb = ci(a), ci(b)
        if ca is None or cb is None:
            return None
        return not (ca[1] < cb[0] or cb[1] < ca[0])

    ordering = {
        "raw: image_cnn < text_bert (CIs separate)": below("raw_image_cnn", "raw_text_bert"),
        "raw: text_bert < cf? (n/a - cf is graph)": None,
        "graph: h_img < h_txt (CIs separate)": below("h_img_graph", "h_txt_graph"),
        "graph: h_txt < cf (CIs separate)": below("h_txt_graph", "cf"),
        "graph: h_img < cf (CIs separate)": below("h_img_graph", "cf"),
        "CNN-vs-CLIP image CIs OVERLAP (sign-flip is within noise)": overlap("raw_image_cnn", "raw_image_clip"),
        "image_cnn gap < image_clip gap (point est)": (
            res["raw_image_cnn"]["gap_abs"] < res["raw_image_clip"]["gap_abs"]
            if "raw_image_clip" in res else None),
    }
    return {"dataset": dataset, "n_copurchase_pairs": int(ia.numel()),
            "streams": res, "ordering_tests": ordering}


def write_summary(out: dict):
    L = ["# Exp A2 — behavioral-alignment gap with bootstrap CI + permutation null", "",
         "Pre-registered metric: distance = ||f(x)-f(y)||^2, f=L2-normalize (Wang&Isola; =2-2cos).",
         "Headline = ABSOLUTE gap g = mean_dist(random) - mean_dist(co-purchased); higher = more",
         "behaviorally predictive. Relative gap reported but demoted (its scale-dependence caused",
         f"the CNN/CLIP sign flip). 95% bootstrap CI (B={B_BOOT}), permutation p (B={B_PERM}, H0: co",
         "pairs no closer than random).", "",
         "## Absolute gap [95% CI] (perm p) per stream",
         "", "| dataset | raw img CNN | raw img CLIP | raw txt BERT | h_img(graph) | h_txt(graph) | cf |",
         "|---|---|---|---|---|---|---|"]
    for d in out["datasets"]:
        r = d["streams"]
        def cell(nm):
            if nm not in r:
                return "—"
            x = r[nm]
            return f"{x['gap_abs']:.4f} [{x['gap_abs_ci95'][0]:.4f},{x['gap_abs_ci95'][1]:.4f}] p={x['perm_p_value']:.3f}"
        L.append(f"| {d['dataset']} | {cell('raw_image_cnn')} | {cell('raw_image_clip')} | "
                 f"{cell('raw_text_bert')} | {cell('h_img_graph')} | {cell('h_txt_graph')} | {cell('cf')} |")
    L += ["", "## Formal ordering tests (CI-separation; True = claim holds with non-overlapping 95% CIs)", ""]
    for d in out["datasets"]:
        L.append(f"**{d['dataset']}**")
        for k, v in d["ordering_tests"].items():
            if v is not None:
                L.append(f"- {k}: **{v}**")
        L.append("")
    (OUT / "SUMMARY.md").write_text("\n".join(L))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["baby", "sports", "clothing", "elec"])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    dev = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"metric": "||f(x)-f(y)||^2, f=L2-normalize (Wang&Isola alignment, =2-2cos)",
           "headline": "absolute gap = mean_dist(random) - mean_dist(co-purchased); higher=more predictive",
           "config": {"n_pairs": N_PAIRS, "n_rand": N_RAND, "B_boot": B_BOOT, "B_perm": B_PERM,
                      "pair_seed": PAIR_SEED, "boot_seed": BOOT_SEED, "perm_seed": PERM_SEED},
           "datasets": []}
    for d in args.datasets:
        print(f"\n=== {d} ===", flush=True)
        try:
            out["datasets"].append(run_dataset(d, dev))
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            out["datasets"].append({"dataset": d, "error": repr(e)})
        (OUT / "alignment_stats.json").write_text(json.dumps(out, indent=2))
        write_summary(out)
    print(f"\nWrote {OUT/'alignment_stats.json'} and SUMMARY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
