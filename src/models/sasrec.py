"""SASRec (Kang & McAuley, ICDM 2018) — self-attentive sequential recommender,
with explicit per-head attention so we can do circuit analysis (induction heads,
repeat-consumption mechanism, head ablation). Item id 0 = padding.

Deliberately a faithful, small reimplementation (d=64, 2 blocks, 2 heads) so every
attention head can be individually inspected and ablated.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalMHA(nn.Module):
    """Multi-head causal self-attention with per-head attention/output exposed."""

    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dh = n_heads, d // n_heads
        self.Wq = nn.Linear(d, d); self.Wk = nn.Linear(d, d); self.Wv = nn.Linear(d, d)
        self.Wo = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        # for introspection / ablation, filled on forward when requested
        self.last_attn: Optional[torch.Tensor] = None      # [B, H, L, L]
        self.ablate_heads: set[int] = set()                # heads to zero (per-head ablation)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor, store_attn: bool = False):
        B, L, _ = x.shape
        q = self.Wq(x).view(B, L, self.h, self.dh).transpose(1, 2)   # [B,H,L,dh]
        k = self.Wk(x).view(B, L, self.h, self.dh).transpose(1, 2)
        v = self.Wv(x).view(B, L, self.h, self.dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)          # [B,H,L,L]
        causal = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal, float("-inf"))
        # mask padding keys
        keypad = ~pad_mask.view(B, 1, 1, L)
        att = att.masked_fill(keypad, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = torch.nan_to_num(att)                                  # rows that are all -inf -> 0
        if store_attn:
            self.last_attn = att.detach()
        out = att @ v                                                # [B,H,L,dh]
        if self.ablate_heads:
            for hd in self.ablate_heads:
                out[:, hd] = 0.0
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.Wo(self.drop(out))


class SASBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalMHA(d, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask, store_attn=False):
        x = x + self.attn(self.ln1(x), pad_mask, store_attn=store_attn)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class SASRec(nn.Module):
    def __init__(self, n_items: int, d: int = 64, n_blocks: int = 2, n_heads: int = 2,
                 maxlen: int = 50, dropout: float = 0.2):
        super().__init__()
        self.n_items, self.d, self.maxlen, self.n_heads, self.n_blocks = n_items, d, maxlen, n_heads, n_blocks
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([SASBlock(d, n_heads, dropout) for _ in range(n_blocks)])
        self.last_ln = nn.LayerNorm(d)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def seq_repr(self, seq: torch.Tensor, store_attn: bool = False) -> torch.Tensor:
        B, L = seq.shape
        pad_mask = seq > 0
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) * math.sqrt(self.d) + self.pos_emb(pos)
        x = self.drop(x) * pad_mask.unsqueeze(-1)
        for blk in self.blocks:
            x = blk(x, pad_mask, store_attn=store_attn)
        return self.last_ln(x)                                      # [B,L,d]

    def last_hidden(self, seq: torch.Tensor, store_attn: bool = False) -> torch.Tensor:
        h = self.seq_repr(seq, store_attn=store_attn)
        lengths = (seq > 0).sum(1).clamp(min=1) - 1                  # last non-pad position
        return h[torch.arange(seq.size(0), device=seq.device), lengths]

    def score_items(self, seq: torch.Tensor, store_attn: bool = False) -> torch.Tensor:
        """Scores over all real items (1..n_items) from the last position. [B, n_items]."""
        h = self.last_hidden(seq, store_attn=store_attn)
        return h @ self.item_emb.weight[1:].t()

    # ---- training loss (SASRec BCE over all positions, 1 negative per position) ----
    def loss(self, seq, pos, neg):
        """seq/pos/neg: [B,L]; pos = next item per position, neg = sampled negative, 0 = pad."""
        h = self.seq_repr(seq)                                      # [B,L,d]
        pe = self.item_emb(pos); ne = self.item_emb(neg)
        pos_logit = (h * pe).sum(-1)
        neg_logit = (h * ne).sum(-1)
        mask = (pos > 0).float()
        loss = -(F.logsigmoid(pos_logit) * mask + F.logsigmoid(-neg_logit) * mask).sum() / mask.sum().clamp_min(1)
        return loss

    # ---- introspection helpers ----
    def set_ablation(self, block: int, heads):
        self.blocks[block].attn.ablate_heads = set(heads)

    def clear_ablation(self):
        for blk in self.blocks:
            blk.attn.ablate_heads = set()

    def attentions(self):
        return [blk.attn.last_attn for blk in self.blocks]          # list of [B,H,L,L]
