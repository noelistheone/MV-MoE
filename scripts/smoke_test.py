"""Smoke test: train an SAE on synthetic activations with known sparse structure.

Generates x = z @ D where z is k-sparse over a random unit dictionary D, then
checks the SAE recovers high fraction-of-variance-explained at the right L0.
Run: python scripts/smoke_test.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from src.sae import SAE, SAEConfig, SAETrainer, TrainConfig
from src.utils import set_seed


def make_synthetic(n=50_000, d=128, n_feats=512, k=8, device="cpu", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    D = torch.randn(n_feats, d, generator=g, device=device)
    D = D / D.norm(dim=1, keepdim=True)
    z = torch.zeros(n, n_feats, device=device)
    idx = torch.argsort(torch.rand(n, n_feats, generator=g, device=device), dim=1)[:, :k]
    vals = torch.rand(n, k, generator=g, device=device) + 0.5
    z.scatter_(1, idx, vals)
    x = z @ D
    return x, D


def main():
    set_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {dev}")
    d, n_feats, k = 128, 512, 8
    x, _ = make_synthetic(n=50_000, d=d, n_feats=n_feats, k=k, device="cpu", seed=0)
    print(f"synthetic activations: {tuple(x.shape)} | true n_feats={n_feats} k={k}")

    sae = SAE(SAEConfig(d_in=d, d_sae=n_feats, variant="topk", k=k))
    trainer = SAETrainer(sae, TrainConfig(lr=3e-4, batch_size=4096, epochs=30,
                                          device=dev, log_every=100))
    hist = trainer.fit(x)

    fve, l0, dead = hist["fve"][-1], hist["l0"][-1], hist["dead_frac"][-1]
    print(f"\nFINAL: FVE={fve:.4f}  L0={l0:.2f}  dead_frac={dead:.3f}")
    # Exact L0 recovery + no dead latents is the real correctness signal; FVE on a
    # coherent 512-atom-in-128-dim dictionary plateaus below 1.0 by construction.
    ok = fve > 0.85 and abs(l0 - k) < 2 and dead < 0.2
    print("SMOKE TEST:", "PASS ✅" if ok else "CHECK ⚠️  (low FVE/odd L0 — inspect)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
