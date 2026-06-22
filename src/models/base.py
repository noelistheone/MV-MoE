"""The contract between a frozen recommender and the SAE pipeline.

The whole project hinges on extracting FOUR representation streams per item from a
frozen multimodal recommender, so the SAE can be trained on each and latents can
be attributed to a modality of origin:

    fused : the recommender's internal fused image+text(+id) item embedding
    image : the raw image-branch item embedding
    text  : the raw text-branch item embedding
    cf    : the collaborative / graph node embedding (id-based)

Concrete wrappers for FREEDOM / LGMRec / LightGCN live alongside this file and
should adapt the user's existing models in ~/Recsys. They only need to (a) expose
these streams as ``[N_items, d]`` tensors and (b) expose a ``score`` function so
edited activations can be pushed back through the frozen scorer for causal tests.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import torch

STREAMS = ("fused", "image", "text", "cf")


@dataclass
class RepresentationStreams:
    """Container for the per-item representation streams (any subset may be None)."""
    fused: Optional[torch.Tensor] = None
    image: Optional[torch.Tensor] = None
    text: Optional[torch.Tensor] = None
    cf: Optional[torch.Tensor] = None

    def available(self) -> Dict[str, torch.Tensor]:
        return {s: getattr(self, s) for s in STREAMS if getattr(self, s) is not None}


class FrozenRecommender(ABC):
    """Interface a recommender must satisfy to be interpreted by this pipeline."""

    @abstractmethod
    def item_streams(self) -> RepresentationStreams:
        """Return the per-item representation streams as ``[N_items, d]`` tensors."""

    @abstractmethod
    def score(self, user_ids: torch.Tensor, item_embeds: torch.Tensor) -> torch.Tensor:
        """Score (user, items) given (possibly SAE-edited) item embeddings.

        Used by the causal tests: edit a fused embedding via an SAE intervention,
        re-score, and measure the change in ranking metrics. Must NOT update any
        model weights (the model is frozen).
        """

    def num_items(self) -> int:
        streams = self.item_streams().available()
        return next(iter(streams.values())).shape[0]


class PrecomputedItemEmbeddings(FrozenRecommender):
    """Zero-model wrapper over already-cached item features.

    Lets the first SAE experiments run directly on the precomputed Amazon
    image+text features (and any fused/cf tensors) without loading a recommender.
    ``score`` is unavailable here — use a real recommender wrapper for causal tests.
    """

    def __init__(self, fused=None, image=None, text=None, cf=None):
        self._streams = RepresentationStreams(fused=fused, image=image, text=text, cf=cf)

    def item_streams(self) -> RepresentationStreams:
        return self._streams

    def score(self, user_ids, item_embeds):
        raise NotImplementedError(
            "PrecomputedItemEmbeddings has no scorer; wrap a real recommender "
            "(FREEDOM/LGMRec/LightGCN) for causal ranking tests."
        )
