#!/usr/bin/env python
"""
Independent verification of the GUME-ensemble result on Amazon 'baby'.

This is a CLEAN-ROOM evaluator. It does NOT import or reuse any existing
evaluation code in the repo. Everything below (split parsing, masking,
z-score normalization, uniform ensemble, top-K, Recall@K, NDCG@K) is
implemented from scratch here.

Definitions used:
  - Split: baby.inter TSV with x_label 0=train, 1=valid, 2=test.
  - Train items are masked to -inf before ranking.
  - Recall@K = (# of held-out TEST items in top-K) / (# of TEST items for user),
    i.e. the standard recall denominator is the number of relevant (test) items.
  - NDCG@K: gains are binary (1 if a top-K item is a test item, else 0),
    DCG = sum 1/log2(rank+1) over hits; IDCG = sum_{i=1..min(#rel,K)} 1/log2(i+1).
  - z-score per row: (x - mean_row) / std_row  (std computed over items, ddof=0).
  - Ensemble: uniform (equal-weight) mean of the 6 row-z-scored matrices.
  - Evaluated on TEST users only (users with >=1 test item).
"""
import numpy as np
import json, os, sys, time

DATA = "/workspace/Recsys/data/baby/baby.inter"
SCORE_DIR = "/workspace/MechInterp/results/bai/scores/"
OUT_DIR = "/workspace/MechInterp/results/bai/"
GUME_FILES = [f"baby_bprdump_s{s}.npy" for s in (2024, 2025, 2026, 2027, 2028)]
KNN_FILE = "baby_userknn.npy"
ENSEMBLE_FILES = GUME_FILES + [KNN_FILE]
SINGLE_GUME = "baby_bprdump_s2024.npy"
KS = [5, 10, 20, 50]

NEG_INF = -1e30  # used as -inf substitute (works in float32)


def parse_split(path):
    """Return per-user sets of train items (label 0) and test items (label 2)."""
    train = {}  # uid -> set(iid)
    test = {}   # uid -> set(iid)
    n_users = 0
    n_items = 0
    with open(path) as fh:
        header = fh.readline()  # skip header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            uid = int(parts[0]); iid = int(parts[1]); lab = int(parts[4])
            n_users = max(n_users, uid + 1)
            n_items = max(n_items, iid + 1)
            if lab == 0:
                train.setdefault(uid, set()).add(iid)
            elif lab == 2:
                test.setdefault(uid, set()).add(iid)
            # label 1 (valid) is ignored for masking AND for relevance here.
    return train, test, n_users, n_items


def row_zscore(mat):
    """Per-row (per-user) z-score over items. mat: float32 [U, I]. In-place safe copy."""
    m = mat.astype(np.float32, copy=True)
    mu = m.mean(axis=1, keepdims=True)
    sd = m.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)  # guard constant rows
    return (m - mu) / sd


def evaluate(scores, train, test, test_users, ks):
    """
    scores: float32 [U, I] (already the final ranking scores, NOT yet masked).
    Masks each user's train items to -inf, ranks, computes Recall@K / NDCG@K.
    Returns dict {('recall', k): val, ('ndcg', k): val}.
    """
    maxk = max(ks)
    log2 = np.log2(np.arange(2, maxk + 2))  # log2(rank+1) for ranks 1..maxk
    inv_log = 1.0 / log2  # discount per position

    # accumulators
    rec_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    n_eval = 0

    for uid in test_users:
        rel = test[uid]
        if not rel:
            continue
        row = scores[uid].astype(np.float32, copy=True)
        # mask train items
        tr = train.get(uid)
        if tr:
            tr_idx = np.fromiter(tr, dtype=np.int64, count=len(tr))
            row[tr_idx] = NEG_INF
        # top-maxk via argpartition then sort those
        if maxk < row.shape[0]:
            part = np.argpartition(-row, maxk - 1)[:maxk]
            top = part[np.argsort(-row[part])]
        else:
            top = np.argsort(-row)[:maxk]
        # binary hit vector over the top-maxk ranking
        hits = np.array([1.0 if i in rel else 0.0 for i in top], dtype=np.float32)
        nrel = len(rel)
        for k in ks:
            hk = hits[:k]
            n_hit = hk.sum()
            rec_sum[k] += n_hit / nrel
            dcg = (hk * inv_log[:k]).sum()
            idcg = inv_log[:min(nrel, k)].sum()
            ndcg_sum[k] += (dcg / idcg) if idcg > 0 else 0.0
        n_eval += 1

    out = {}
    for k in ks:
        out[("recall", k)] = rec_sum[k] / n_eval
        out[("ndcg", k)] = ndcg_sum[k] / n_eval
    out["_n_eval_users"] = n_eval
    return out


def main():
    t0 = time.time()
    print("Parsing split...", flush=True)
    train, test, n_users, n_items = parse_split(DATA)
    test_users = sorted(test.keys())
    print(f"n_users={n_users} n_items={n_items} #test_users={len(test_users)}", flush=True)

    # ---- single GUME ----
    print(f"Loading single GUME {SINGLE_GUME}...", flush=True)
    g = np.load(SCORE_DIR + SINGLE_GUME).astype(np.float32)
    assert g.shape == (n_users, n_items), (g.shape, (n_users, n_items))
    single_res = evaluate(g, train, test, test_users, KS)
    print("Single GUME done:", {f"R@{k}": round(single_res[('recall',k)],4) for k in KS}, flush=True)
    del g

    # ---- ensemble: row z-score each, uniform average ----
    print("Building ensemble (row z-score + uniform mean of 6)...", flush=True)
    acc = np.zeros((n_users, n_items), dtype=np.float32)
    for f in ENSEMBLE_FILES:
        print(f"  + {f}", flush=True)
        m = np.load(SCORE_DIR + f)
        assert m.shape == (n_users, n_items), (f, m.shape)
        acc += row_zscore(m)
        del m
    acc /= len(ENSEMBLE_FILES)
    ens_res = evaluate(acc, train, test, test_users, KS)
    print("Ensemble done:", {f"R@{k}": round(ens_res[('recall',k)],4) for k in KS}, flush=True)
    del acc

    # ---- report ----
    metrics = []
    for k in KS:
        for name in ("recall", "ndcg"):
            metrics.append((name, k))

    table = []
    all_beat_2pct = True
    for (name, k) in metrics:
        s = single_res[(name, k)]
        e = ens_res[(name, k)]
        delta_pct = (e - s) / s * 100.0 if s != 0 else float('nan')
        beats = delta_pct >= 2.0
        if not beats:
            all_beat_2pct = False
        table.append({
            "metric": f"{name.upper()[0]}@{k}",
            "single_gume": round(float(s), 6),
            "ensemble": round(float(e), 6),
            "delta_pct": round(float(delta_pct), 3),
            "beats_2pct": bool(beats),
        })

    r20_single = single_res[("recall", 20)]
    r20_ens = ens_res[("recall", 20)]
    r20_delta = (r20_ens - r20_single) / r20_single * 100.0

    print("\n==== RESULTS ====")
    print(f"{'metric':8s} {'single_GUME':>12s} {'ensemble':>12s} {'delta_%':>9s} {'>=2%?':>6s}")
    for row in table:
        print(f"{row['metric']:8s} {row['single_gume']:12.6f} {row['ensemble']:12.6f} "
              f"{row['delta_pct']:9.3f} {str(row['beats_2pct']):>6s}")
    print(f"\nSingle GUME R@20 = {r20_single:.6f} (expected ~0.1027)")
    print(f"Ensemble    R@20 = {r20_ens:.6f}")
    print(f"R@20 delta       = {r20_delta:.3f}%")
    print(f"ALL 8 metrics beat single by >=2%: {all_beat_2pct}")
    print(f"#eval users single={single_res['_n_eval_users']} ensemble={ens_res['_n_eval_users']}")

    result = {
        "n_users": n_users,
        "n_items": n_items,
        "n_test_users": len(test_users),
        "n_eval_users": single_res["_n_eval_users"],
        "single_gume_file": SINGLE_GUME,
        "ensemble_files": ENSEMBLE_FILES,
        "single_gume_R20": float(r20_single),
        "ensemble_R20": float(r20_ens),
        "R20_delta_pct": float(r20_delta),
        "all_metrics_beat_2pct": bool(all_beat_2pct),
        "table": table,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = OUT_DIR + "ensemble_verify_independent.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nWrote {out_path}  (elapsed {result['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
