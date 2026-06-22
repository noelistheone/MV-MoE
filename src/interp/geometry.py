"""GPU representation-geometry diagnostics to explain "why is a modality weak?".

Each metric operates on an item-embedding matrix X [N, d] and is computed with
torch on GPU (singular values via torch.linalg.svdvals). The hypotheses these test:
a weak modality stream tends to have LOW effective rank / participation ratio
(little usable variance), HIGH anisotropy (cone effect), LOW per-item norm, and HIGH
redundancy (CKA) with the dominant stream.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _svals(X: torch.Tensor, center: bool = True) -> torch.Tensor:
    X = X.float()
    if center:
        X = X - X.mean(0, keepdim=True)
    return torch.linalg.svdvals(X)


def effective_rank(X: torch.Tensor) -> float:
    """RankMe / spectral effective rank = exp(entropy of normalized singular values)."""
    s = _svals(X)
    p = s / s.sum().clamp_min(1e-12)
    ent = -(p * p.clamp_min(1e-12).log()).sum()
    return float(ent.exp())


def participation_ratio(X: torch.Tensor) -> float:
    """(Σλ)²/Σλ² with λ = singular_value² (covariance eigenvalues)."""
    lam = _svals(X) ** 2
    return float((lam.sum() ** 2) / (lam ** 2).sum().clamp_min(1e-12))


@torch.no_grad()
def anisotropy(X: torch.Tensor, sample: int = 4000, seed: int = 0) -> float:
    """Mean off-diagonal cosine similarity (cone effect); high => embeddings cluster in a cone."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(X.shape[0], generator=g)[:sample].to(X.device)
    A = F.normalize(X[idx].float(), dim=-1)
    sims = A @ A.t()
    n = A.shape[0]
    return float((sims.sum() - sims.diag().sum()) / (n * n - n))


@torch.no_grad()
def uniformity(X: torch.Tensor, sample: int = 4000, t: float = 2.0, seed: int = 0) -> float:
    """Wang & Isola (2020): log E[exp(-t||x-y||²)] over unit-normalized embeddings."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(X.shape[0], generator=g)[:sample].to(X.device)
    A = F.normalize(X[idx].float(), dim=-1)
    d2 = torch.pdist(A) ** 2
    return float(d2.mul(-t).exp().mean().clamp_min(1e-12).log())


def mean_norm(X: torch.Tensor) -> float:
    return float(X.float().norm(dim=-1).mean())


def modality_gap(A: torch.Tensor, B: torch.Tensor) -> float:
    """Liang et al. (2022): ||mean(unit-normalized A) - mean(unit-normalized B)||."""
    a = F.normalize(A.float(), dim=-1).mean(0)
    b = F.normalize(B.float(), dim=-1).mean(0)
    return float((a - b).norm())


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA ||X^T Y||_F² / (||X^T X||_F ||Y^T Y||_F); redundancy in [0,1]."""
    X = X.float() - X.float().mean(0, keepdim=True)
    Y = Y.float() - Y.float().mean(0, keepdim=True)
    xy = (X.t() @ Y).pow(2).sum()
    xx = (X.t() @ X).pow(2).sum().sqrt()
    yy = (Y.t() @ Y).pow(2).sum().sqrt()
    return float(xy / (xx * yy).clamp_min(1e-12))


def full_report(X: torch.Tensor) -> dict:
    return {
        "n": int(X.shape[0]), "d": int(X.shape[1]),
        "effective_rank": effective_rank(X),
        "participation_ratio": participation_ratio(X),
        "anisotropy": anisotropy(X),
        "uniformity": uniformity(X),
        "mean_norm": mean_norm(X),
    }
