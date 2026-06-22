"""How interpretable is each SAE latent? Score latents against item attributes.

Given latent activations ``Z [N, d_sae]`` and a binary attribute label ``y [N]``
(e.g. "item is in category Toys", "price is in the top band"), we quantify how
well a single latent encodes that attribute. These are the recsys analogues of
the protein/speech "latent vs ground-truth-property" metrics, and follow the
recsys-SAE papers (point-biserial r, ROC-AUC, precision/recall/F1).

All primary metrics are vectorised over latents and exact (tie-robust).
"""
from __future__ import annotations

from typing import Dict, List

import torch


def point_biserial(Z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Point-biserial correlation of every latent with a binary label.

    Equivalent to Pearson r between each latent column and ``y in {0,1}``;
    robust to the heavy ties at zero that sparse latents produce.

    Args:
        Z: ``[N, d_sae]`` latent activations.
        y: ``[N]`` binary labels (0/1).
    Returns:
        ``[d_sae]`` correlation per latent (nan->0 for constant latents).
    """
    Z = Z.float()
    y = y.float()
    zc = Z - Z.mean(0, keepdim=True)
    yc = y - y.mean()
    num = (zc * yc.unsqueeze(1)).sum(0)
    den = zc.pow(2).sum(0).sqrt() * yc.pow(2).sum().sqrt()
    r = num / den.clamp_min(1e-12)
    return torch.nan_to_num(r, nan=0.0)


def active_prf1(Z: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Treat "latent fires (>0)" as a binary predictor of ``y`` and score it.

    The most natural metric for sparse latents (latent active <-> attribute present).
    Returns precision/recall/F1 per latent, all ``[d_sae]``.
    """
    y = y.float()
    fired = (Z > 0).float()                       # [N, d_sae]
    tp = (fired * y.unsqueeze(1)).sum(0)
    fp = (fired * (1 - y).unsqueeze(1)).sum(0)
    fn = ((1 - fired) * y.unsqueeze(1)).sum(0)
    precision = tp / (tp + fp).clamp_min(1e-12)
    recall = tp / (tp + fn).clamp_min(1e-12)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def roc_auc_topk(Z: torch.Tensor, y: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
    """Exact ROC-AUC (sklearn) for a small set of candidate latents.

    AUC over *all* latents is wasteful; pre-rank with ``point_biserial`` and pass
    the top-k indices here for an exact score on the contenders.
    """
    from sklearn.metrics import roc_auc_score

    yy = y.cpu().numpy()
    out = []
    for j in candidate_ids.tolist():
        col = Z[:, j].cpu().numpy()
        try:
            out.append(roc_auc_score(yy, col))
        except ValueError:           # single-class label
            out.append(float("nan"))
    return torch.tensor(out)


def match_attributes_to_latents(
    Z: torch.Tensor,
    attribute_labels: Dict[str, torch.Tensor],
    topk: int = 5,
) -> List[dict]:
    """For each named attribute, find its best-matching latent(s).

    Args:
        Z: ``[N, d_sae]`` latent activations.
        attribute_labels: name -> ``[N]`` binary label tensor.
        topk: how many candidate latents (by |point-biserial|) to AUC-score.
    Returns:
        list of dicts, one per attribute, sorted by best latent F1.
    """
    results = []
    for name, y in attribute_labels.items():
        r = point_biserial(Z, y)
        prf = active_prf1(Z, y)
        cand = r.abs().topk(min(topk, Z.shape[1])).indices
        auc = roc_auc_topk(Z, y, cand)
        # Pick the candidate with the highest AUC (base-rate-robust, unlike F1).
        best_local = int(torch.nan_to_num(auc, nan=-1.0).argmax())
        best_latent = int(cand[best_local])
        results.append({
            "attribute": name,
            "best_latent": best_latent,
            "point_biserial": float(r[best_latent]),
            "precision": float(prf["precision"][best_latent]),
            "recall": float(prf["recall"][best_latent]),
            "f1": float(prf["f1"][best_latent]),
            "roc_auc": float(auc[best_local]),
            "candidate_latents": cand.tolist(),
        })
    # Rank attributes by how cleanly a single latent encodes them (AUC).
    results.sort(key=lambda d: (-1.0 if d["roc_auc"] != d["roc_auc"] else d["roc_auc"]), reverse=True)
    return results
