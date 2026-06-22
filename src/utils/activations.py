"""Model-agnostic activation capture and caching.

The SAE only ever sees a ``[N, d]`` tensor, so this module's job is to produce
such tensors from (a) any ``torch.nn.Module`` via forward hooks — works for a
custom recommender, a CLIP tower, anything — or (b) precomputed embedding files
(the multimodal item features the project already owns).
"""
from __future__ import annotations

import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------- capture
def _get_module(model: nn.Module, dotted: str) -> nn.Module:
    mod = model
    for part in dotted.split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


@torch.no_grad()
def collect_activations(
    model: nn.Module,
    layer_names: Sequence[str],
    inputs: Iterable,
    device: str = "cuda",
    forward_fn: Optional[Callable[[nn.Module, object], object]] = None,
    flatten: bool = True,
    max_samples: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Run ``inputs`` through ``model`` and cache the output of each named module.

    Args:
        model: any nn.Module (recommender, encoder, ...). Set to eval() & frozen.
        layer_names: dotted module paths to hook (e.g. "encoder.layers.3", "item_proj").
        inputs: iterable of batches. Each batch is passed to ``forward_fn``.
        forward_fn: how to call the model on a batch; default ``model(batch)``.
            Use a custom fn for recommenders with structured inputs
            (e.g. ``lambda m, b: m(b["users"], b["items"])``).
        flatten: reshape captured tensors to ``[-1, d]`` (last dim kept as feature dim).
        max_samples: stop after roughly this many rows per layer.

    Returns:
        dict layer_name -> tensor ``[N, d]`` on CPU.
    """
    model.eval().to(device)
    forward_fn = forward_fn or (lambda m, b: m(b))
    store: Dict[str, List[torch.Tensor]] = {ln: [] for ln in layer_names}
    counts: Dict[str, int] = {ln: 0 for ln in layer_names}

    handles = []
    def make_hook(name: str):
        def hook(_module, _inp, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(out):
                return
            t = out.detach()
            if flatten:
                t = t.reshape(-1, t.shape[-1])
            store[name].append(t.float().cpu())
            counts[name] += t.shape[0]
        return hook

    for ln in layer_names:
        handles.append(_get_module(model, ln).register_forward_hook(make_hook(ln)))

    try:
        for batch in inputs:
            if isinstance(batch, torch.Tensor):
                batch = batch.to(device)
            forward_fn(model, batch)
            if max_samples is not None and all(c >= max_samples for c in counts.values()):
                break
    finally:
        for h in handles:
            h.remove()

    result = {}
    for ln in layer_names:
        cat = torch.cat(store[ln], dim=0)
        if max_samples is not None:
            cat = cat[:max_samples]
        result[ln] = cat
    return result


# --------------------------------------------------------------- precomputed
def load_embedding_matrix(path: str, key: Optional[str] = None) -> torch.Tensor:
    """Load a precomputed ``[N, d]`` embedding tensor (.pt/.pth/.npy/.npz)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pt", ".pth"):
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            obj = obj[key] if key is not None else next(iter(obj.values()))
        t = obj if torch.is_tensor(obj) else torch.as_tensor(obj)
    elif ext == ".npy":
        t = torch.from_numpy(np.load(path))
    elif ext == ".npz":
        z = np.load(path)
        t = torch.from_numpy(z[key] if key is not None else z[z.files[0]])
    else:
        raise ValueError(f"unsupported embedding file extension: {ext}")
    return t.float()


# --------------------------------------------------------------------- cache
class ActivationCache:
    """Tiny on-disk cache so activations are harvested once and reused.

    Usage:
        cache = ActivationCache("activations/baby")
        if not cache.has("item_fused"):
            cache.put("item_fused", tensor)
        x = cache.get("item_fused")
    """

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _p(self, name: str) -> str:
        return os.path.join(self.root, f"{name}.pt")

    def has(self, name: str) -> bool:
        return os.path.exists(self._p(name))

    def put(self, name: str, tensor: torch.Tensor) -> None:
        torch.save(tensor.cpu(), self._p(name))

    def get(self, name: str, map_location: str = "cpu") -> torch.Tensor:
        return torch.load(self._p(name), map_location=map_location, weights_only=False)

    def names(self) -> List[str]:
        return [f[:-3] for f in os.listdir(self.root) if f.endswith(".pt")]
