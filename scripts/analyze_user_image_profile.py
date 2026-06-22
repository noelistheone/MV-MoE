"""What do ALL items of a SINGLE user share in image space? (per-user basket structure)

Goes beyond the earlier item-item pair analysis. The recommender scores user-item, so the
actionable object is a per-USER image profile. The leak-free formalization of "the common image
point of a user's items" is the LEAVE-ONE-OUT centroid: hold out item j, average the rest, ask
whether that centroid ranks j above negatives. Vectorized (precompute per-user sum S; LOO centroid
= (S[u]-z[j])/(n_u-1)); runs in ~1-2 min on GPU.

Questions (Baby, dev):
  A. RAW LOO retrieval  -- per encoder (CNN/DINOv2/SigLIP2/text/random); controls = global-mean
     centroid (popularity) and user-mismatch (score centroid against a random held item ~ 0.5).
  B. TRAIN->TEST transfer -- profile from a user's TRAIN items ranks their held-out TEST item vs
     negatives = recommendation-relevant ceiling of an image-only user profile (raw + learned).
  C. LEARNED LOO metric -- train projection P so the LOO centroid in P-space ranks held-out
     positives high; TRAIN vs held-out-interaction TEST AUC = headroom of a leak-free learned
     user-image profile. linear vs MLP.
  D. CATEGORY control -- is basket image-coherence ABOVE same-category? basket same/diff-cat cos vs
     intra-category cos vs random.
  E. GLOBAL vs IDIOSYNCRATIC -- remove top-m global PCs, re-measure raw LOO AUC.
  F. STRATIFY by basket size.

Writes results/bai/user_image_profile_analysis.json. GPU.
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RECSYS = Path("/workspace/Recsys")
MECH = Path("/workspace/MechInterp")
DATA = RECSYS / "data" / "baby"
OUT = MECH / "results" / "bai"
dev = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2024
torch.manual_seed(SEED)
np.random.seed(SEED)
gcpu = torch.Generator(device="cpu").manual_seed(SEED)


def load_inter():
    return np.loadtxt(DATA / "baby.inter", delimiter="\t", skiprows=1, dtype=np.int64,
                      usecols=(0, 1, 4))   # user, item, x_label (0 train / 1 valid / 2 test)


def load_feats():
    feats = {}
    feats["image_cnn"] = np.load(DATA / "image_feat.npy").astype(np.float32)
    feats["text"] = np.load(DATA / "text_feat.npy").astype(np.float32)
    d = MECH / "data"
    feats["image_dinov2"] = np.load(d / "baby_dinov2_base.npy").astype(np.float32)
    feats["image_siglip2"] = np.load(d / "baby_siglip2_base_patch16_224.npy").astype(np.float32)
    n = feats["image_cnn"].shape[0]
    feats["random"] = np.random.RandomState(SEED).randn(n, 256).astype(np.float32)
    return feats, n


def load_categories(n_items):
    cat = ["?"] * n_items
    with open(DATA / "meta-baby.csv", newline="") as f:
        for row in csv.DictReader(f):
            try:
                iid = int(row["itemID"])
            except (ValueError, KeyError):
                continue
            if 0 <= iid < n_items:
                cat[iid] = row.get("categories", "?") or "?"
    return cat


def l2np(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def l2(x):
    return F.normalize(x, dim=1)


def flat_interactions(user_items, min_basket=2):
    """flat tensors item_idx, user_compact (0..nU-1) for users with >= min_basket distinct items."""
    items_all, users_all = [], []
    nU = 0
    for u, items in user_items.items():
        items = list(dict.fromkeys(items))
        if len(items) < min_basket:
            continue
        for it in items:
            items_all.append(it); users_all.append(nU)
        nU += 1
    return (torch.tensor(items_all, device=dev), torch.tensor(users_all, device=dev), nU)


def user_sum(Z, items_all, users_all, nU):
    S = torch.zeros(nU, Z.shape[1], device=dev).index_add_(0, users_all, Z[items_all])
    n_u = torch.zeros(nU, device=dev).index_add_(
        0, users_all, torch.ones_like(users_all, dtype=torch.float))
    return S, n_u


def auc_vs_negs(C, P, Z, n_neg=500):
    """C [M,d] centroids, P [M,d] positive embeds (rows of Z), shared negative sample."""
    N = Z.shape[0]
    pos = (C * P).sum(1)
    neg_idx = torch.randint(0, N, (n_neg,), generator=gcpu).to(dev)
    aucs = []
    for s in range(0, C.shape[0], 50000):
        ns = C[s:s + 50000] @ Z[neg_idx].t()
        aucs.append((pos[s:s + 50000, None] > ns).float().mean(1))
    return torch.cat(aucs).mean().item()


# ---------------------------------------------------------------- A. raw LOO
def raw_loo_auc(feat_t, items_all, users_all, nU, mode="loo", n_neg=500):
    Z = l2(feat_t)
    S, n_u = user_sum(Z, items_all, users_all, nU)
    if mode == "loo":
        C = (S[users_all] - Z[items_all]) / (n_u[users_all] - 1).clamp(min=1).unsqueeze(1)
        P = Z[items_all]
    elif mode == "global_mean":
        C = l2(Z.mean(0, keepdim=True)).expand(items_all.shape[0], -1)
        P = Z[items_all]
    elif mode == "mismatch":   # LOO centroid scored against a RANDOM held item (sanity ~0.5)
        C = (S[users_all] - Z[items_all]) / (n_u[users_all] - 1).clamp(min=1).unsqueeze(1)
        perm = torch.randperm(items_all.shape[0], generator=gcpu).to(dev)
        P = Z[items_all[perm]]
    C = l2(C)
    return auc_vs_negs(C, P, Z, n_neg), C.shape[0]


# ---------------------------------------------------------------- B. train->test
def train_test_transfer(feat_t, train_items, test_items, n_neg=500):
    Z = l2(feat_t)
    cents, poss = [], []
    for u, titems in test_items.items():
        tr = list(dict.fromkeys(train_items.get(u, [])))
        if not tr:
            continue
        prof = l2(Z[torch.tensor(tr, device=dev)].mean(0, keepdim=True)).squeeze(0)
        for j in dict.fromkeys(titems):
            cents.append(prof); poss.append(j)
    if not cents:
        return float("nan"), 0
    C = torch.stack(cents); P = Z[torch.tensor(poss, device=dev)]
    return auc_vs_negs(C, P, Z, n_neg), C.shape[0]


# ---------------------------------------------------------------- C. learned LOO
class Proj(nn.Module):
    def __init__(self, din, d=64, mlp=False):
        super().__init__()
        self.f = (nn.Sequential(nn.Linear(din, d), nn.GELU(), nn.Linear(d, d))
                  if mlp else nn.Linear(din, d))

    def forward(self, x):
        return F.normalize(self.f(x), dim=1)


def learned_loo_metric(feat_t, train_items, test_items, mlp=False, epochs=400, n_neg=500):
    """P trained on TRAIN-basket LOO ranking; eval = (i) held-out TRAIN interactions (LOO) and
    (ii) TEST items ranked by the user's full TRAIN profile. Vectorized."""
    Z0 = feat_t; N = Z0.shape[0]
    items_all, users_all, nU = flat_interactions(train_items, 2)
    # map compact user id -> original user for the test phase
    comp_of = {}
    nU2 = 0
    for u, items in train_items.items():
        if len(set(items)) >= 2:
            comp_of[u] = nU2; nU2 += 1
    n_u = torch.zeros(nU, device=dev).index_add_(
        0, users_all, torch.ones_like(users_all, dtype=torch.float))
    M = items_all.shape[0]
    # 80/20 split of TRAIN interactions for the learned-metric generalization estimate
    rng = np.random.RandomState(SEED)
    tr_mask = torch.tensor(rng.rand(M) < 0.8, device=dev)
    proj = Proj(Z0.shape[1], 64, mlp=mlp).to(dev)
    opt = torch.optim.Adam(proj.parameters(), lr=1e-3, weight_decay=1e-4)
    tr_pos = tr_mask.nonzero().squeeze(1)
    for ep in range(epochs):
        proj.train()
        Z = proj(Z0)
        S = torch.zeros(nU, Z.shape[1], device=dev).index_add_(0, users_all, Z[items_all])
        sel = tr_pos[torch.randint(0, tr_pos.shape[0], (4096,), device=dev)]
        u = users_all[sel]; it = items_all[sel]
        C = l2((S[u] - Z[it]) / (n_u[u] - 1).clamp(min=1).unsqueeze(1))
        pos = (C * Z[it]).sum(1)
        na = torch.randint(0, N, (sel.shape[0],), device=dev)
        neg = (C * Z[na]).sum(1)
        loss = -(F.logsigmoid(pos - neg)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    proj.eval()
    with torch.no_grad():
        Z = proj(Z0)
        S = torch.zeros(nU, Z.shape[1], device=dev).index_add_(0, users_all, Z[items_all])

        def loo_auc(idx):
            u = users_all[idx]; it = items_all[idx]
            C = l2((S[u] - Z[it]) / (n_u[u] - 1).clamp(min=1).unsqueeze(1))
            return auc_vs_negs(C, Z[it], Z, n_neg)
        tr_auc = loo_auc(tr_pos)
        te_auc = loo_auc((~tr_mask).nonzero().squeeze(1))
        # (ii) TEST-item transfer: rank actual held-out test items by full TRAIN profile
        cents, poss = [], []
        prof_full = l2(S / n_u.clamp(min=1).unsqueeze(1))
        for uu, titems in test_items.items():
            c = comp_of.get(uu)
            if c is None:
                continue
            p = prof_full[c]
            for j in dict.fromkeys(titems):
                cents.append(p); poss.append(j)
        test_transfer = (auc_vs_negs(torch.stack(cents), Z[torch.tensor(poss, device=dev)], Z, n_neg)
                         if cents else float("nan"))
    return tr_auc, te_auc, test_transfer


# ---------------------------------------------------------------- D. category control
def category_control(feat_t, user_items, cat, n_pairs=40000):
    Z = l2(feat_t); N = Z.shape[0]
    cat = np.array(cat, dtype=object)
    same, diff = [], []
    seen = 0
    for u, items in user_items.items():
        items = list(dict.fromkeys(items))
        if len(items) < 2:
            continue
        idx = torch.tensor(items, device=dev); E = Z[idx]; sim = E @ E.t()
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                (same if cat[items[a]] == cat[items[b]] else diff).append(sim[a, b].item())
        seen += 1
        if len(same) + len(diff) > 300000:
            break
    cat2items = defaultdict(list)
    for i in range(N):
        cat2items[cat[i]].append(i)
    bigcats = [c for c, v in cat2items.items() if len(v) >= 2 and c != "?"]
    rng = np.random.RandomState(SEED)
    ia, ib = [], []
    for _ in range(n_pairs):
        v = cat2items[bigcats[rng.randint(len(bigcats))]]
        x, y = v[rng.randint(len(v))], v[rng.randint(len(v))]
        if x != y:
            ia.append(x); ib.append(y)
    ia = torch.tensor(ia, device=dev); ib = torch.tensor(ib, device=dev)
    intra_cat = (Z[ia] * Z[ib]).sum(1).mean().item()
    ra = torch.randint(0, N, (n_pairs,), generator=gcpu).to(dev)
    rb = torch.randint(0, N, (n_pairs,), generator=gcpu).to(dev)
    return {
        "basket_same_cat_cos": float(np.mean(same)) if same else None,
        "basket_diff_cat_cos": float(np.mean(diff)) if diff else None,
        "n_same_cat_pairs": len(same), "n_diff_cat_pairs": len(diff),
        "intra_category_cos": intra_cat, "random_cos": (Z[ra] * Z[rb]).sum(1).mean().item(),
        "frac_basket_pairs_same_cat": len(same) / max(1, len(same) + len(diff)),
    }


# ---------------------------------------------------------------- E. global vs idiosyncratic
def remove_top_pcs(feat_np, m):
    X = l2np(feat_np); Xc = X - X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:m].T
    return (X - X @ V @ V.T).astype(np.float32)


# ---------------------------------------------------------------- F. stratify
def stratified_loo(feat_t, train_items, bins=((2, 3), (4, 6), (7, 12), (13, 9999))):
    Z = l2(feat_t)
    out = {}
    for lo, hi in bins:
        sub = {u: it for u, it in train_items.items() if lo <= len(set(it)) <= hi}
        ia, ua, nU = flat_interactions(sub, 2)
        if nU == 0:
            out[f"{lo}-{hi}"] = {"auc": None, "n": 0}; continue
        a, m = raw_loo_auc(Z, ia, ua, nU)   # Z already normalized; raw_loo re-normalizes (idempotent)
        out[f"{lo}-{hi}"] = {"auc": a, "n": m}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_inter()
    feats, n = load_feats()
    cat = load_categories(n)
    train_items, test_items = defaultdict(list), defaultdict(list)
    for u, i, lab in rows:
        (train_items if lab == 0 else test_items)[int(u)].append(int(i)) if lab in (0, 2) else None
    n_b2 = sum(1 for v in train_items.values() if len(set(v)) >= 2)
    res = {"dataset": "baby", "n_items": int(n), "n_users": len(train_items),
           "n_users_basket>=2": n_b2, "n_distinct_categories": len(set(cat))}
    print(f"users={len(train_items)} (>=2 items: {n_b2}) items={n} cats={len(set(cat))}", flush=True)

    ft = {k: torch.tensor(v, device=dev) for k, v in feats.items()}
    items_all, users_all, nU = flat_interactions(train_items, 2)
    print(f"flat train interactions (basket>=2): {items_all.shape[0]} over {nU} users", flush=True)

    # A
    res["A_raw_loo_auc"] = {}
    for k in ft:
        a, m = raw_loo_auc(ft[k], items_all, users_all, nU)
        res["A_raw_loo_auc"][k] = {"auc": a, "n": m}
        print(f"[A] raw LOO {k:14s} AUC={a:.4f} (n={m})", flush=True)
    gm, _ = raw_loo_auc(ft["image_cnn"], items_all, users_all, nU, mode="global_mean")
    mm, _ = raw_loo_auc(ft["image_cnn"], items_all, users_all, nU, mode="mismatch")
    res["A_controls_cnn"] = {"global_mean_centroid": gm, "user_mismatch": mm}
    print(f"[A] CNN controls: global-mean={gm:.4f}  user-mismatch={mm:.4f}", flush=True)

    # B
    res["B_train2test_auc"] = {}
    for k in ["image_cnn", "image_dinov2", "image_siglip2", "text", "random"]:
        a, m = train_test_transfer(ft[k], train_items, test_items)
        res["B_train2test_auc"][k] = {"auc": a, "n": m}
        print(f"[B] train->test {k:14s} AUC={a:.4f} (n={m})", flush=True)

    # C
    res["C_learned_loo"] = {}
    for k in ["image_cnn", "image_dinov2", "image_siglip2", "text"]:
        tl, tel, tt_l = learned_loo_metric(ft[k], train_items, test_items, mlp=False)
        tm, tem, tt_m = learned_loo_metric(ft[k], train_items, test_items, mlp=True)
        res["C_learned_loo"][k] = {"linear_train": tl, "linear_test": tel, "linear_test_transfer": tt_l,
                                   "mlp_train": tm, "mlp_test": tem, "mlp_test_transfer": tt_m}
        print(f"[C] learned LOO {k:14s} lin tr/te={tl:.4f}/{tel:.4f} transfer={tt_l:.4f} | "
              f"mlp tr/te={tm:.4f}/{tem:.4f} transfer={tt_m:.4f}", flush=True)

    # D (only if categories aligned)
    res["D_category_control"] = {}
    if res["n_distinct_categories"] > 1:
        for k in ["image_cnn", "image_dinov2", "text"]:
            d = category_control(ft[k], train_items, cat); res["D_category_control"][k] = d
            sc = d['basket_same_cat_cos']; dc = d['basket_diff_cat_cos']
            print(f"[D] {k:14s} basket same/diff-cat={sc}/{dc} intra-cat={d['intra_category_cos']:.3f} "
                  f"rand={d['random_cos']:.3f} (same-cat frac={d['frac_basket_pairs_same_cat']:.2f})", flush=True)
    else:
        print("[D] skipped: categories not aligned (n_distinct=1)", flush=True)

    # E
    base = res["A_raw_loo_auc"]["image_cnn"]["auc"]
    res["E_global_vs_idiosyncratic"] = {"m=0_base": base}
    for m in [1, 5, 20]:
        fr = torch.tensor(remove_top_pcs(feats["image_cnn"], m), device=dev)
        a, _ = raw_loo_auc(fr, items_all, users_all, nU)
        res["E_global_vs_idiosyncratic"][f"m={m}_removed"] = a
        print(f"[E] CNN remove top-{m} PCs -> raw LOO AUC={a:.4f} (base {base:.4f})", flush=True)

    # F
    res["F_stratified_cnn"] = stratified_loo(ft["image_cnn"], train_items)
    print(f"[F] stratified (CNN): {res['F_stratified_cnn']}", flush=True)

    (OUT / "user_image_profile_analysis.json").write_text(json.dumps(res, indent=2))
    print("wrote " + str(OUT / "user_image_profile_analysis.json"), flush=True)


if __name__ == "__main__":
    main()
