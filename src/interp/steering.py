"""Causal interventions on SAE latents.

The interpretability claim is only causal if editing a latent changes the model's
behaviour. These helpers produce *edited activations*; the downstream causal test
is to feed the edited activation back through the frozen recommender's scoring
function and measure ΔR@20 / ΔN@20 / Δcoverage / Δpopularity-bias.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from ..sae import SAE


@torch.no_grad()
def ablate_latents(sae: SAE, x: torch.Tensor, latent_ids: Sequence[int]) -> torch.Tensor:
    """Zero out the given latent(s) and reconstruct ``x`` without them.

    Returns an activation of the same shape as ``x`` with those features removed.
    """
    acts = sae.encode(x)
    acts[:, list(latent_ids)] = 0.0
    return sae.decode(acts)


@torch.no_grad()
def clamp_latent(sae: SAE, x: torch.Tensor, latent_id: int, value: float) -> torch.Tensor:
    """Force a latent to a fixed activation value and reconstruct.

    ``value`` can exceed the latent's natural range to over-express a concept
    (the "boost Children -> Pixar" style steer).
    """
    acts = sae.encode(x)
    acts[:, latent_id] = value
    return sae.decode(acts)


@torch.no_grad()
def steering_direction(sae: SAE, latent_id: int, strength: float = 1.0) -> torch.Tensor:
    """The activation-space direction of a latent (its decoder atom), scaled.

    Add this to a model's activation to steer it toward the concept without a full
    encode/decode round-trip: ``x_steered = x + steering_direction(sae, j, s)``.
    """
    return strength * sae.W_dec[latent_id].detach()
