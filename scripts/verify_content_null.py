"""
INDEPENDENT verifier (written from scratch, not reusing ensemble_eval core metric).

Claim under test (RANKING level): adding the content (image/text item-kNN) expert
to the GUME-seeds + userKNN ensemble changes TEST Recall@20 by ~0 (best content
weight tuned on VALID is near 0).

Design:
 - Split from <ds>.inter via x_label: 0=train, 1=valid, 2=test.
 - Per-row (per-user) z-score each score matrix BEFORE summing (mean/std in fp32,
   matrix stays fp16 on GPU).
 - Mask train items to -inf IN PLACE on fp16 combined matrix (no float32 copy).
 - My OWN topk + gather Recall@20 / NDCG@20. Only users with >=1 gt item counted.
 - Base = mean(GUME seeds 2024/25/26, each z-scored) + w_uk * z(userKNN).
   Then sweep an added content weight w_c for z(sig_img) [also text/joint] on VALID,
   report TEST R@20 / N@20 at the valid-best weight.
"""
import sys, os, json
from collections import defaultdict
import numpy as np
import torch

SDIR = "/workspace/MechInterp/results/bai/scores"
DATADIR = "/workspace/Recsys/data"
DEV = "cuda"
K = 20


def load_split(ds):
    path = f"{DATADIR}/{ds}/{ds}.inter"
    # columns: userID itemID rating timestamp x_label  -> use 0,1,4
    arr = np.loadtxt(path, delimiter="\t", skiprows=1, dtype=np.int64, usecols=(0, 1, 4))
    nu = int(arr[:, 0].max()) + 1
    ni = int(arr[:, 1].max()) + 1
    train_u, train_i = [], []
    val = defaultdict(set)
    tst = defaultdict(set)
    for u, i, lab in arr:
        u = int(u); i = int(i); lab = int(lab)
        if lab == 0:
            train_u.append(u); train_i.append(i)
        elif lab == 1:
            val[u].add(i)
        else:
            tst[u].add(i)
    return (np.array(train_u), np.array(train_i)), val, tst, nu, ni


def gt_matrix(gt, nu, ni):
    gu, gi = [], []
    for u, items in gt.items():
        for it in items:
            gu.append(u); gi.append(it)
    REL = torch.zeros(nu, ni, device=DEV, dtype=torch.bool)
    REL[torch.tensor(gu, device=DEV), torch.tensor(gi, device=DEV)] = True
    nrel = REL.sum(1)  # int per user
    return REL, nrel


def load_mat(name):
    a = np.load(f"{SDIR}/{name}")
    return torch.from_numpy(a).to(DEV)  # fp16


def zscore_rows(S):
    """Per-row z-score, IN PLACE on the fp16 matrix (no full float32 copy).
    Reductions (mean/var) accumulate in fp32 for numerical safety, but those are
    only [nu,1] vectors. The big [nu,ni] matrix is never upcast to float32 -- needed
    for clothing (1.8GB fp16 each) to avoid OOM."""
    nu = S.shape[0]
    m = S.mean(1, keepdim=True, dtype=torch.float32)               # [nu,1] fp32
    # chunked E[x^2] so the transient float32 slice is small (cap clothing OOM)
    sq = torch.empty(nu, 1, device=S.device, dtype=torch.float32)
    cs = 4096
    for a in range(0, nu, cs):
        b = min(a + cs, nu)
        sq[a:b] = (S[a:b].float() ** 2).mean(1, keepdim=True)
    var = sq - m * m                                               # [nu,1] fp32
    sd = var.clamp_min(0).sqrt().clamp_min(1e-6)                   # [nu,1] fp32
    S -= m.half()                                                   # in place fp16
    S /= sd.half()                                                  # in place fp16
    return S


def evaluate(S, train_idx, REL, nrel, k=K):
    """S fp16 [nu,ni] is consumed (masked in place). Returns (recall@k, ndcg@k)."""
    tu, ti = train_idx
    S[tu, ti] = float("-inf")
    topk = torch.topk(S, k, dim=1).indices            # [nu,k]
    hits = torch.gather(REL, 1, topk).float()         # [nu,k] 0/1, in topk order
    # recall
    nrel_f = nrel.float()
    keep = nrel > 0
    recall = (hits.sum(1) / nrel_f.clamp_min(1))[keep].mean().item()
    # ndcg
    disc = 1.0 / torch.log2(torch.arange(2, k + 2, device=DEV).float())  # [k]
    dcg = (hits * disc).sum(1)
    idcg_cum = torch.cumsum(disc, 0)                  # idcg_cum[j] = ideal dcg with j+1 rel
    idx = nrel.clamp(max=k).long().clamp_min(1) - 1
    idcg = idcg_cum[idx]
    ndcg = (dcg / idcg)[keep].mean().item()
    return recall, ndcg


def main():
    ds = sys.argv[1]
    train_idx_np, val, tst, nu, ni = load_split(ds)
    tu = torch.tensor(train_idx_np[0], device=DEV)
    ti = torch.tensor(train_idx_np[1], device=DEV)
    train_idx = (tu, ti)
    REL_v, nrel_v = gt_matrix(val, nu, ni)
    REL_t, nrel_t = gt_matrix(tst, nu, ni)
    print(f"[{ds}] users={nu} items={ni} train_inter={len(tu)} "
          f"val_users={int((nrel_v>0).sum())} test_users={int((nrel_t>0).sum())}", flush=True)

    # --- z-scored components (build once, free raw asap) ---
    gume = None
    for s in (2024, 2025, 2026):
        z = zscore_rows(load_mat(f"{ds}_bprdump_s{s}.npy"))
        gume = z if gume is None else gume + z
    gume = gume / 3.0  # mean of seed-z-scored matrices
    z_uk = zscore_rows(load_mat(f"{ds}_userknn.npy"))

    def base(w_uk):
        return gume + w_uk * z_uk

    # ---- tune userKNN weight (uniform-ish) on VALID for the BASE ensemble ----
    # prompt says "userKNN uniform"; we confirm w_uk=1.0 is reasonable but also report sweep
    print("  --- base GUME+userKNN: userKNN weight sweep (VALID R@20) ---", flush=True)
    best_uk, best_uk_r = 1.0, -1
    for w in [0.0, 0.5, 1.0, 1.5, 2.0]:
        r, _ = evaluate(base(w).clone(), train_idx, REL_v, nrel_v)
        print(f"    w_uk={w:<4} valid R@20={r:.5f}", flush=True)
        if r > best_uk_r:
            best_uk_r, best_uk = r, w
    # use uniform (w_uk=1.0) as the canonical base per the prompt ("userKNN uniform")
    w_uk = 1.0
    base_mat = base(w_uk)

    # base TEST
    br_v, bn_v = evaluate(base_mat.clone(), train_idx, REL_v, nrel_v)
    br_t, bn_t = evaluate(base_mat.clone(), train_idx, REL_t, nrel_t)
    print(f"  BASE (GUME+userKNN, w_uk=1.0): VALID R@20={br_v:.5f}  "
          f"TEST R@20={br_t:.5f} N@20={bn_t:.5f}", flush=True)

    results = {"dataset": ds, "n_users": nu, "n_items": ni,
               "base": {"valid_R20": br_v, "test_R20": br_t, "test_N20": bn_t,
                        "best_uk_w_on_valid": best_uk, "best_uk_valid_R20": best_uk_r}}

    # ---- for each content expert: sweep added weight on VALID, report TEST ----
    grid = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    for tag in ["img", "text", "joint"]:
        torch.cuda.empty_cache()
        z_c = zscore_rows(load_mat(f"{ds}_bai2i_sig_{tag}_learned_k20.npy"))
        sweep = []
        best_w, best_r = 0.0, -1
        for w_c in grid:
            S = base_mat + w_c * z_c
            r, _ = evaluate(S, train_idx, REL_v, nrel_v)
            sweep.append((w_c, r))
            if r > best_r:
                best_r, best_w = r, w_c
        # TEST at valid-best weight
        St = base_mat + best_w * z_c
        rt, nt = evaluate(St, train_idx, REL_t, nrel_t)
        dr = rt - br_t
        print(f"  CONTENT[{tag}] valid sweep: " +
              " ".join(f"{w}:{r:.5f}" for w, r in sweep), flush=True)
        print(f"  CONTENT[{tag}] best_w(valid)={best_w}  valid R@20={best_r:.5f}  "
              f"=> TEST R@20={rt:.5f} (Δ vs base {dr:+.5f}, {100*dr/br_t:+.2f}%) "
              f"N@20={nt:.5f}", flush=True)
        del z_c
        results[tag] = {"valid_sweep": sweep, "best_w_valid": best_w,
                        "best_valid_R20": best_r, "test_R20": rt, "test_N20": nt,
                        "delta_test_R20": dr, "pct_delta": 100 * dr / br_t}

    out = f"/workspace/Recsys/results/bai/content_null_verify_{ds}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
