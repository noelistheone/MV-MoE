"""Track B — circuit discovery in a trained SASRec (direction 3: transfer the LLM
mech-interp toolkit to a sequential recommender).

Tests the central hypothesis from the landscape survey: is repeat-purchase implemented
by an INDUCTION-HEAD circuit ([A][B]...[A] -> predict B = "what followed A last time")?

Analyses (all on held-out test sequences, left-padded so the query is always position L-1):
  1. Repeat-consumption statistics — how often is the target a previously-seen item;
     model accuracy on repeat vs fresh targets; are seen items up-weighted?
  2. Per-head attention decomposition — recency (attend to previous item),
     repeat (attend to earlier occurrences of the current item), and INDUCTION
     (attend to the position AFTER the previous occurrence of the current item).
  3. Induction PREDICTION test — for sequences whose last item recurs, is the item
     that followed it last time ("B") boosted in the output?
  4. Causal head ablation — zero each (block,head), measure ΔRecall@20 and Δ on the
     induction-prediction metric -> which heads form the circuit.

Output: results/trackB_sasrec/circuits.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/workspace/MechInterp")
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "scripts"))
from sasrec import SASRec                                    # noqa: E402
from trackB_sasrec_train import build_sequences, pad_left, evaluate  # noqa: E402

OUT = ROOT / "results" / "trackB_sasrec"


def load_model(dataset, device):
    ck = torch.load(OUT / f"sasrec_{dataset}.pt", map_location=device, weights_only=False)
    c = ck["cfg"]
    m = SASRec(ck["n_items"], d=c["d"], n_blocks=c["blocks"], n_heads=c["heads"], maxlen=c["maxlen"])
    m.load_state_dict(ck["state"]); m.to(device).eval()
    return m, c["maxlen"]


@torch.no_grad()
def batched_test_inputs(seqs, maxlen, device, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    uids = list(seqs.keys())
    uids = [uids[i] for i in rng.choice(len(uids), min(n, len(uids)), replace=False)]
    S, tgt, inp_raw = [], [], []
    for u in uids:
        items = seqs[u]
        inp, t = items[:-1], items[-1]
        S.append(pad_left(inp, maxlen)); tgt.append(t); inp_raw.append(inp)
    return torch.tensor(S, device=device), np.array(tgt), inp_raw


@torch.no_grad()
def repeat_stats(model, seq, tgt, inp_raw, device):
    scores = model.score_items(seq)                          # [B, n_items]
    B = seq.size(0)
    is_repeat = np.array([t in set(r) for t, r in zip(tgt, inp_raw)])
    # rank of target
    tgt_col = torch.tensor(tgt - 1, device=device)
    rk = (scores > scores[torch.arange(B, device=device), tgt_col].unsqueeze(1)).sum(1).cpu().numpy()
    def recall20(mask): return float((rk[mask] < 20).mean()) if mask.any() else float("nan")
    # are SEEN items up-weighted? mean percentile of seen vs all
    seen_pct = []
    for j in range(B):
        seen = [it - 1 for it in set(inp_raw[j]) if it >= 1]
        if seen:
            sc = scores[j]
            seen_pct.append(float((sc < sc[seen].mean()).float().mean()))
    return {"repeat_target_rate": float(is_repeat.mean()),
            "recall@20_overall": float((rk < 20).mean()),
            "recall@20_repeat_targets": recall20(is_repeat),
            "recall@20_fresh_targets": recall20(~is_repeat),
            "mean_percentile_of_seen_items": float(np.mean(seen_pct))}


@torch.no_grad()
def attention_decomposition(model, seq, device):
    """Per (block,head): mean attention from last pos to (a) previous item [recency],
    (b) earlier occurrences of the current item [repeat], (c) position-after-previous-
    occurrence of current item [INDUCTION]."""
    model.score_items(seq, store_attn=True)
    atts = model.attentions()                                # list of [B,H,L,L]
    B, L = seq.shape
    cur = seq[:, -1]                                          # current (last) item
    # previous occurrence index of current item (most recent j<L-1 with seq[:,j]==cur)
    eq = (seq[:, :-1] == cur.unsqueeze(1))                    # [B, L-1]
    has_prev = eq.any(1)
    prev_idx = torch.where(has_prev, (L - 1) - torch.flip(eq, [1]).float().argmax(1), torch.full_like(cur, -1))
    out = {}
    for bi, att in enumerate(atts):
        H = att.size(1)
        for h in range(H):
            a = att[:, h, L - 1, :]                           # [B, L] attention from last pos
            recency = a[:, L - 2].mean().item()
            # repeat: attention mass on all earlier occurrences of current item
            rep = (a[:, :-1] * eq.float()).sum(1)
            repeat = rep[has_prev].mean().item() if has_prev.any() else float("nan")
            # induction: attention to position-after-previous-occurrence
            ind_pos = (prev_idx + 1).clamp(0, L - 1)
            ind = a[torch.arange(B, device=device), ind_pos]
            induction = ind[has_prev & (prev_idx < L - 2)].mean().item() if has_prev.any() else float("nan")
            out[f"block{bi}.head{h}"] = {"recency": recency, "repeat": repeat, "induction": induction}
    return out, has_prev.float().mean().item()


@torch.no_grad()
def induction_prediction(model, seq, device):
    """For sequences whose last item recurs at position j, is item at j+1 ('B' = what
    followed it last time) boosted? Report mean rank of B and its top-20 hit-rate."""
    B, L = seq.shape
    cur = seq[:, -1]
    eq = (seq[:, :-1] == cur.unsqueeze(1))
    has_prev = eq.any(1)
    prev_idx = (L - 1) - torch.flip(eq, [1]).float().argmax(1).long()
    valid = has_prev & (prev_idx < L - 2)                    # there is a 'next' item after prev occurrence
    if valid.sum() == 0:
        return {"n_valid": 0}
    bpos = (prev_idx + 1)
    Bitem = seq[torch.arange(B, device=device), bpos]        # the item that followed last time
    scores = model.score_items(seq)
    vidx = torch.where(valid)[0]
    bcol = (Bitem[vidx] - 1)
    rk = (scores[vidx] > scores[vidx, bcol].unsqueeze(1)).sum(1).float()
    return {"n_valid": int(valid.sum()), "mean_rank_of_B": float(rk.mean()),
            "B_hit@20": float((rk < 20).float().mean()), "B_hit@1": float((rk < 1).float().mean()),
            "median_rank_of_B": float(rk.median())}


@torch.no_grad()
def head_ablation(model, seqs, seq, device, maxlen):
    base = induction_prediction(model, seq, device)
    base_r20 = evaluate(model, seqs, device, maxlen, phase="test", n_eval=6000)["Recall@20"]
    rows = {}
    for bi in range(model.n_blocks):
        for h in range(model.n_heads):
            model.clear_ablation(); model.set_ablation(bi, [h])
            ind = induction_prediction(model, seq, device)
            r20 = evaluate(model, seqs, device, maxlen, phase="test", n_eval=6000)["Recall@20"]
            rows[f"block{bi}.head{h}"] = {
                "dRecall@20": r20 - base_r20,
                "dB_hit@20": ind.get("B_hit@20", float("nan")) - base.get("B_hit@20", float("nan")),
                "dmean_rank_B": ind.get("mean_rank_of_B", float("nan")) - base.get("mean_rank_of_B", float("nan"))}
    model.clear_ablation()
    return {"baseline": {"Recall@20": base_r20, **base}, "per_head_ablation": rows}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="baby"); ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    model, maxlen = load_model(args.dataset, device)
    seqs, _ = build_sequences(args.dataset)
    seq, tgt, inp_raw = batched_test_inputs(seqs, maxlen, device)

    rep = repeat_stats(model, seq, tgt, inp_raw, device)
    print("[1] repeat-consumption:", {k: round(v, 4) for k, v in rep.items()})
    attn, prev_rate = attention_decomposition(model, seq, device)
    print(f"[2] attention decomposition (frac seqs with a prior occurrence of last item = {prev_rate:.3f}):")
    for hd, v in attn.items():
        print(f"    {hd}: recency={v['recency']:.3f} repeat={v['repeat']:.3f} induction={v['induction']:.3f}")
    ind = induction_prediction(model, seq, device)
    print("[3] induction prediction:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ind.items()})
    abl = head_ablation(model, seqs, seq, device, maxlen)
    print("[4] head ablation (dRecall@20 | dB_hit@20):")
    for hd, v in abl["per_head_ablation"].items():
        print(f"    ablate {hd}: dR@20={v['dRecall@20']:+.4f}  dB_hit@20={v['dB_hit@20']:+.4f}  dmeanRankB={v['dmean_rank_B']:+.1f}")

    out = {"dataset": args.dataset, "n_test_seqs": int(seq.size(0)),
           "frac_with_prior_occurrence": prev_rate,
           "repeat_consumption": rep, "attention_decomposition": attn,
           "induction_prediction": ind, "head_ablation": abl}
    (OUT / "circuits.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT/'circuits.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
