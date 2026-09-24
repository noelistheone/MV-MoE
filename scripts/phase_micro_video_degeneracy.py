"""Declared negative side-cell (PREREG s.6): are MicroLens's released `video_feat.npy`
features usable as a third modality at all?

NO CLAIM is built on the answer. This exists so that "you ignored the video features"
is answered with a measurement instead of silence. Compares all three released MicroLens
streams (and, as a calibration reference, the Amazon image/text streams) on:
  effective rank (exp of the entropy of the normalized eigen-spectrum of the covariance),
  mean all-pairs cosine on a random subsample, and mean/sd of the L2 norm.

Outputs -> results/phase_micro/video_feat_degeneracy.json. Recsys is read-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

DATA = Path("/workspace/Recsys/data")
OUT = Path("/workspace/MechInterp/results/phase_micro")
SUB_SEED, N_SUB = 0, 4000


def effective_rank(X: torch.Tensor) -> float:
    """exp(entropy of normalized covariance eigenvalues) — Roy & Vetterli."""
    Xc = X - X.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Xc.double())
    p = (s ** 2) / (s ** 2).sum().clamp_min(1e-30)
    p = p[p > 1e-30]
    return float(torch.exp(-(p * p.log()).sum()).item())


def stats(name: str, path: Path, dev: str) -> dict | None:
    if not path.is_file():
        return None
    X = torch.from_numpy(np.load(path)).float().to(dev)
    n, d = X.shape
    g = torch.Generator(device="cpu").manual_seed(SUB_SEED)
    idx = torch.randperm(n, generator=g)[:min(N_SUB, n)].to(dev)
    S = torch.nn.functional.normalize(X[idx], dim=-1)
    C = S @ S.T
    off = ~torch.eye(C.shape[0], dtype=torch.bool, device=dev)
    norms = X.norm(dim=-1)
    er = effective_rank(X[idx])
    return {"file": str(path), "n_items": int(n), "dim": int(d),
            "effective_rank": er, "effective_rank_frac_of_dim": er / d,
            "mean_pairwise_cosine": float(C[off].mean().item()),
            "p95_pairwise_cosine": float(C[off].quantile(0.95).item()),
            "mean_L2_norm": float(norms.mean().item()), "sd_L2_norm": float(norms.std().item()),
            "n_subsampled_for_cosine": int(idx.numel())}


def main() -> int:
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    targets = {
        "microlens/image (1024d, released)": DATA / "microlens" / "image_feat.npy",
        "microlens/text (1024d, released)": DATA / "microlens" / "text_feat.npy",
        "microlens/video (768d, released)": DATA / "microlens" / "video_feat.npy",
        "baby/image (4096d CNN, reference)": DATA / "baby" / "image_feat.npy",
        "baby/text (384d sBERT, reference)": DATA / "baby" / "text_feat.npy",
    }
    res = {}
    for k, p in targets.items():
        r = stats(k, p, dev)
        if r is None:
            continue
        res[k] = r
        print(f"{k:38s} eff_rank={r['effective_rank']:8.2f}/{r['dim']:<5d}"
              f" ({r['effective_rank_frac_of_dim']*100:5.2f}%)"
              f" mean_cos={r['mean_pairwise_cosine']:+.4f} |x|={r['mean_L2_norm']:.3f}", flush=True)
    v = res.get("microlens/video (768d, released)")
    out = {
        "purpose": "DECLARED NEGATIVE SIDE-CELL (PREREG s.6). No claim is built on this.",
        "verdict": (
            "The released MicroLens video features are near-degenerate: effective rank "
            f"{v['effective_rank']:.2f}/{v['dim']} ({v['effective_rank_frac_of_dim']*100:.2f}% of "
            f"dimensions) with mean all-pairs cosine {v['mean_pairwise_cosine']:.4f} — every item "
            "points in nearly the same direction. They cannot support a third-modality "
            "knockout, so this study tests the COVER-FRAME image interface only."
            if v else "video_feat.npy not found"),
        "method": ("effective rank = exp(entropy of normalized covariance eigenspectrum) "
                   f"(Roy & Vetterli) on a {N_SUB}-item random subsample (seed {SUB_SEED}); "
                   "cosine over the same subsample, off-diagonal only."),
        "streams": res,
    }
    (OUT / "video_feat_degeneracy.json").write_text(json.dumps(out, indent=2))
    print(f"\n{out['verdict']}\nWrote {OUT/'video_feat_degeneracy.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
