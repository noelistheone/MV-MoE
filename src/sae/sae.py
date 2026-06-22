"""A compact, dependency-light Sparse Autoencoder.

Supports the two variants that dominate current practice:
  - "topk":  TopK SAE (Gao et al. 2024 / EleutherAI) with AuxK dead-feature revival.
  - "relu":  classic L1-penalised ReLU SAE (Anthropic "Towards Monosemanticity").

It trains on ANY ``[N, d_in]`` activation tensor, so it works equally on a
language-model residual stream, a CLIP embedding, or a recommender's user/item
embedding — which is the whole point of this project.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    d_in: int                      # width of the activation being explained
    d_sae: int                     # number of dictionary features (overcomplete: d_sae >> d_in)
    variant: str = "topk"          # "topk" | "relu"
    k: int = 32                    # active latents per example (topk only)
    l1_coeff: float = 1e-3         # sparsity penalty (relu only)
    aux_k: int = 256               # latents used by the AuxK dead-feature loss (topk only)
    aux_coeff: float = 1.0 / 32.0  # weight of the AuxK loss
    dead_steps_threshold: int = 1000  # steps a latent may be inactive before it counts as "dead"
    normalize_decoder: bool = True    # keep decoder columns unit-norm
    tied_init: bool = True            # initialise encoder as decoder transpose

    def to_dict(self) -> dict:
        return asdict(self)


class SAE(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        d_in, d_sae = cfg.d_in, cfg.d_sae

        # Decoder: columns are the dictionary atoms (directions in activation space).
        W_dec = torch.randn(d_sae, d_in)
        W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Encoder.
        if cfg.tied_init:
            self.W_enc = nn.Parameter(W_dec.t().clone())
        else:
            self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
            nn.init.kaiming_uniform_(self.W_enc, a=5 ** 0.5)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Bookkeeping for dead-feature detection (not a learnable param).
        self.register_buffer("steps_since_active", torch.zeros(d_sae, dtype=torch.long))

    # ------------------------------------------------------------------ encode
    def _pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the sparse latent activations for ``x`` (``[batch, d_sae]``)."""
        pre = self._pre_acts(x)
        if self.cfg.variant == "topk":
            acts = F.relu(pre)
            topv, topi = acts.topk(self.cfg.k, dim=-1)
            acts = torch.zeros_like(acts).scatter_(-1, topi, topv)
            return acts
        elif self.cfg.variant == "relu":
            return F.relu(pre)
        raise ValueError(f"unknown variant {self.cfg.variant!r}")

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        acts = self.encode(x)
        recon = self.decode(acts)
        return recon, acts

    # -------------------------------------------------------------------- loss
    def loss(self, x: torch.Tensor) -> dict:
        pre = self._pre_acts(x)
        if self.cfg.variant == "topk":
            relu_pre = F.relu(pre)
            topv, topi = relu_pre.topk(self.cfg.k, dim=-1)
            acts = torch.zeros_like(relu_pre).scatter_(-1, topi, topv)
        else:
            acts = F.relu(pre)

        recon = self.decode(acts)
        recon_loss = (recon - x).pow(2).sum(-1).mean()

        out = {"recon": recon, "acts": acts, "recon_loss": recon_loss}

        if self.cfg.variant == "relu":
            # L1 weighted by decoder norm (so it cannot be gamed by rescaling).
            dec_norm = self.W_dec.norm(dim=1)
            l1 = (acts.abs() * dec_norm).sum(-1).mean()
            out["sparsity_loss"] = self.cfg.l1_coeff * l1
            out["loss"] = recon_loss + out["sparsity_loss"]
        else:
            aux = self._auxk_loss(x, recon, F.relu(pre))
            out["aux_loss"] = self.cfg.aux_coeff * aux
            out["loss"] = recon_loss + out["aux_loss"]

        self._update_activity(acts)
        return out

    def _auxk_loss(self, x: torch.Tensor, recon: torch.Tensor, relu_pre: torch.Tensor) -> torch.Tensor:
        """Make dead latents reconstruct the residual (Gao et al. 2024)."""
        dead = self.steps_since_active > self.cfg.dead_steps_threshold
        n_dead = int(dead.sum())
        if n_dead == 0:
            return x.new_zeros(())
        k_aux = min(self.cfg.aux_k, n_dead)
        masked = relu_pre.masked_fill(~dead, 0.0)
        topv, topi = masked.topk(k_aux, dim=-1)
        aux_acts = torch.zeros_like(masked).scatter_(-1, topi, topv)
        aux_recon = aux_acts @ self.W_dec  # reconstruct the *residual*, no decoder bias
        residual = (x - recon).detach()
        return (aux_recon - residual).pow(2).sum(-1).mean()

    # ----------------------------------------------------------- housekeeping
    @torch.no_grad()
    def _update_activity(self, acts: torch.Tensor) -> None:
        fired = (acts > 0).any(dim=0)
        self.steps_since_active[fired] = 0
        self.steps_since_active[~fired] += 1

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        if self.cfg.normalize_decoder:
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))

    @torch.no_grad()
    def set_decoder_norm_grad_zero(self) -> None:
        """Remove the component of the decoder gradient parallel to each atom.

        Keeps unit-norm atoms stable under gradient descent (Anthropic trick).
        Call between ``backward()`` and ``step()`` for the L1 variant.
        """
        if self.W_dec.grad is None:
            return
        w = self.W_dec.data
        g = self.W_dec.grad
        parallel = (g * w).sum(dim=1, keepdim=True) * w
        g.sub_(parallel)

    @property
    def num_dead(self) -> int:
        return int((self.steps_since_active > self.cfg.dead_steps_threshold).sum())
