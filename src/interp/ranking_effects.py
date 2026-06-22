"""Ranking-native effect metrics for interventions on a recommender.

An intervention changes the item-embedding matrix the user vectors are scored
against. We score with the *unchanged* Recsys evaluation protocol (history-mask
to -inf, then top-K) so Recall@K/NDCG@K are exactly comparable to Phase 0, and we
add ranking-change metrics (top-K overlap, RBO) between pre/post-edit top lists.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

RECSYS = Path("/workspace/Recsys")
if str(RECSYS) not in sys.path:
    sys.path.insert(0, str(RECSYS))
from src.evaluation.topk_evaluator import TopKEvaluator      # noqa: E402  (reuse exact metric kernels)


@torch.no_grad()
def evaluate_item_matrix(u_all: torch.Tensor, item_matrix: torch.Tensor, test_loader,
                         device: str, topk: Sequence[int] = (10, 20),
                         metrics: Sequence[str] = ("Recall", "NDCG")):
    """Full-ranking eval of (frozen users) x (possibly-edited item matrix).

    Returns (metrics_dict, topk20 [N_users_eval, max(topk)] int tensor on CPU,
    user_order [N_users_eval] int tensor). The user order is fixed by the loader,
    so top-K lists from two calls are row-aligned and directly comparable.
    """
    evaluator = TopKEvaluator(metrics=list(metrics), topk=list(topk))
    evaluator.reset()
    topk_max = max(topk)
    all_topk, all_users = [], []
    for batch in test_loader:
        user_ids = batch["user_ids"].to(device, non_blocking=True)
        hist_idx = batch["history_indices"].to(device, non_blocking=True)
        hist_val = batch["history_values"].to(device, non_blocking=True)
        scores = u_all[user_ids] @ item_matrix.t()
        if hist_idx.numel() > 0:
            mask = hist_val.bool()
            row = torch.arange(scores.size(0), device=device).unsqueeze(1).expand_as(hist_idx)
            safe = torch.where(hist_idx >= 0, hist_idx, torch.zeros_like(hist_idx))
            scores[row[mask], safe[mask]] = float("-inf")
        _, topk_idx = torch.topk(scores, k=topk_max, dim=-1)
        evaluator.collect(topk_idx.cpu(), batch["positive_items"], batch["positive_lengths"])
        all_topk.append(topk_idx.cpu())
        all_users.append(user_ids.cpu())
    return evaluator.compute(), torch.cat(all_topk, 0), torch.cat(all_users, 0)


def _rbo(a: np.ndarray, b: np.ndarray, p: float = 0.9) -> float:
    """Rank-biased overlap of two ranked top-k lists (top-weighted, in [0,1])."""
    k = len(a)
    sa, sb = set(), set()
    overlap, rbo = 0, 0.0
    for d in range(k):
        sa.add(int(a[d])); sb.add(int(b[d]))
        # |intersection of first (d+1) of each|
        overlap = len(sa & sb)
        rbo += (overlap / (d + 1)) * (p ** d)
    return (1 - p) * rbo


def ranking_change(base_topk: torch.Tensor, cond_topk: torch.Tensor,
                   k: int = 20, sample: int = 4000, p: float = 0.9, seed: int = 0) -> dict:
    """Top-k overlap (Jaccard@k) and RBO between two row-aligned top-K list sets."""
    base = base_topk[:, :k].numpy()
    cond = cond_topk[:, :k].numpy()
    n = base.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    overlaps, rbos = [], []
    for i in idx:
        sa, sb = set(base[i].tolist()), set(cond[i].tolist())
        overlaps.append(len(sa & sb) / k)
        rbos.append(_rbo(base[i], cond[i], p))
    return {"overlap@%d" % k: float(np.mean(overlaps)),
            "rbo": float(np.mean(rbos)),
            "n_sampled": int(len(idx))}
