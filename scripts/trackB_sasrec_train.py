"""Track B — train SASRec on Amazon sequences (temporal leave-one-out), reproduce a
reasonable next-item ranking, save the model. Foundation for the circuit analysis.

Sequences are built from the timestamped .inter file (sorted per user); last item =
test target, 2nd-last = val, the rest = train sequence. This is the canonical SASRec
setup (temporal split — note it is NOT the same split as FREEDOM's random 8:1:1, so
absolute metrics are not directly comparable; Track B is about the MECHANISM).

Output: results/trackB_sasrec/sasrec_<ds>.pt + metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/workspace/MechInterp")
RECSYS = Path("/workspace/Recsys")
sys.path.insert(0, str(ROOT / "src" / "models"))
from sasrec import SASRec                                    # noqa: E402

OUT = ROOT / "results" / "trackB_sasrec"


def build_sequences(dataset: str):
    df = pd.read_csv(RECSYS / "data" / dataset / f"{dataset}.inter", sep="\t")
    df = df.sort_values(["userID", "timestamp"], kind="stable")
    seqs = {}
    for uid, g in df.groupby("userID", sort=False):
        items = (g["itemID"].to_numpy() + 1).tolist()        # +1: reserve 0 for padding
        if len(items) >= 3:
            seqs[int(uid)] = items
    n_items = int(df["itemID"].max()) + 1
    return seqs, n_items


def pad_left(seq, maxlen):
    seq = seq[-maxlen:]
    return [0] * (maxlen - len(seq)) + seq


@torch.no_grad()
def evaluate(model, seqs, device, maxlen, phase="test", topk=(10, 20), n_eval=10000, seed=0):
    rng = np.random.default_rng(seed)
    uids = list(seqs.keys())
    if len(uids) > n_eval:
        uids = [uids[i] for i in rng.choice(len(uids), n_eval, replace=False)]
    hits = {k: 0 for k in topk}; ndcg = {k: 0.0 for k in topk}; n = 0
    B = 512
    for s0 in range(0, len(uids), B):
        chunk = uids[s0:s0 + B]
        inputs, targets, seen = [], [], []
        for u in chunk:
            items = seqs[u]
            if phase == "test":
                inp, tgt = items[:-1], items[-1]
            else:
                inp, tgt = items[:-2], items[-2]
            inputs.append(pad_left(inp, maxlen)); targets.append(tgt); seen.append(set(inp))
        seq = torch.tensor(inputs, device=device)
        scores = model.score_items(seq)                       # [B, n_items]
        for j, u in enumerate(chunk):
            sc = scores[j].clone()
            for it in seen[j]:
                if it >= 1:
                    sc[it - 1] = -1e9                          # mask seen (item ids are 1-based -> col it-1)
            tgt_col = targets[j] - 1
            rank = int((sc > sc[tgt_col]).sum().item())        # 0-based rank of target
            for k in topk:
                if rank < k:
                    hits[k] += 1; ndcg[k] += 1.0 / np.log2(rank + 2)
            n += 1
    return {f"Recall@{k}": hits[k] / n for k in topk} | {f"NDCG@{k}": ndcg[k] / n for k in topk}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="baby")
    ap.add_argument("--maxlen", type=int, default=50)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2024); np.random.seed(2024)

    seqs, n_items = build_sequences(args.dataset)
    print(f"{args.dataset}: {len(seqs)} users (>=3 items), {n_items} items, "
          f"mean train-len {np.mean([len(v)-2 for v in seqs.values()]):.1f}")
    train = {u: v[:-2] for u, v in seqs.items() if len(v) >= 3}
    train_users = [u for u, v in train.items() if len(v) >= 2]

    model = SASRec(n_items, d=args.d, n_blocks=args.blocks, n_heads=args.heads,
                   maxlen=args.maxlen, dropout=0.2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    rng = np.random.default_rng(0)

    best = {"Recall@20": -1}; best_state = None; t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        rng.shuffle(train_users)
        tot = 0.0; nb = 0
        for s0 in range(0, len(train_users), args.bs):
            chunk = train_users[s0:s0 + args.bs]
            S, P, N = [], [], []
            for u in chunk:
                items = train[u]
                inp, pos = items[:-1], items[1:]
                inp = pad_left(inp, args.maxlen); pos = pad_left(pos, args.maxlen)
                neg = [(int(rng.integers(1, n_items + 1)) if p > 0 else 0) for p in pos]
                S.append(inp); P.append(pos); N.append(neg)
            loss = model.loss(torch.tensor(S, device=device),
                              torch.tensor(P, device=device), torch.tensor(N, device=device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            model.eval()
            val = evaluate(model, seqs, device, args.maxlen, phase="val")
            print(f"  ep {ep+1:3d} loss {tot/nb:.4f} | val R@20={val['Recall@20']:.4f} N@20={val['NDCG@20']:.4f}", flush=True)
            if val["Recall@20"] > best["Recall@20"]:
                best = val; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    test = evaluate(model, seqs, device, args.maxlen, phase="test")
    print(f"\nBEST val R@20={best['Recall@20']:.4f} | TEST R@20={test['Recall@20']:.4f} "
          f"N@20={test['NDCG@20']:.4f}  ({(time.time()-t0)/60:.1f} min)")
    torch.save({"state": model.state_dict(), "n_items": n_items,
                "cfg": {"d": args.d, "blocks": args.blocks, "heads": args.heads, "maxlen": args.maxlen}},
               OUT / f"sasrec_{args.dataset}.pt")
    metrics = {"dataset": args.dataset, "n_items": n_items, "n_users": len(seqs),
               "val": best, "test": test, "cfg": vars(args)}
    p = OUT / "metrics.json"
    e = json.loads(p.read_text()) if p.is_file() else {}
    e[args.dataset] = metrics; p.write_text(json.dumps(e, indent=2))
    print(f"saved model + metrics -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
