#!/usr/bin/env python
"""
INDEPENDENT verification (written from scratch) of the pair-level claim:
  IMAGE is a stronger item-item co-purchase predictor than TEXT.

For each dataset {baby, sports, clothing}:
  1. Load SigLIP image + SigLIP title-text features ([n_items, 768]).
  2. Build co-purchase item pairs from TRAIN rows (x_label==0) of <ds>.inter.
     A pair = two distinct items co-purchased by the same user.
  3. Split pairs 80/20 (train/eval).
  4. Train a small projection (Linear->GELU->Linear, L2-normalized) per modality
     with a logsigmoid(pos_pair_cos - random_neg_cos) objective.
  5. Eval held-out pair AUC: P(cos(pos_i,pos_j) > cos(pos_i, random_neg)).
Report AUC(image) vs AUC(text) and whether image>text per dataset.

No existing project code is imported. Pure numpy/torch.
"""
import sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA = '/workspace/MechInterp/data'
INTER = '/workspace/Recsys/data/{ds}/{ds}.inter'

SEED = 2024
MAX_PAIRS_PER_USER = 200   # cap combinatorial blow-up from power users
PROJ_HID = 256
PROJ_OUT = 128
EPOCHS = 30
BATCH = 8192
LR = 1e-3
EVAL_NEG_PER_POS = 1       # AUC = P(pos > 1 random neg)


def load_train_pairs(ds, rng):
    """Return array of [i,j] co-purchase pairs from x_label==0 rows."""
    path = INTER.format(ds=ds)
    users = []
    items = []
    with open(path) as f:
        header = f.readline()  # skip header
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) < 5:
                continue
            if p[4] != '0':       # TRAIN only
                continue
            users.append(int(p[0]))
            items.append(int(p[1]))
    users = np.asarray(users)
    items = np.asarray(items)
    # group items by user
    order = np.argsort(users, kind='stable')
    users = users[order]; items = items[order]
    boundaries = np.where(np.diff(users) != 0)[0] + 1
    groups = np.split(items, boundaries)
    pairs = []
    for g in groups:
        g = np.unique(g)
        n = len(g)
        if n < 2:
            continue
        # all unordered pairs, capped
        ii, jj = np.triu_indices(n, k=1)
        if len(ii) > MAX_PAIRS_PER_USER:
            sel = rng.choice(len(ii), MAX_PAIRS_PER_USER, replace=False)
            ii, jj = ii[sel], jj[sel]
        pairs.append(np.stack([g[ii], g[jj]], axis=1))
    pairs = np.concatenate(pairs, axis=0)
    return pairs


class Proj(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.GELU(),
            nn.Linear(d_hid, d_out),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)


def train_eval(feat_np, train_pairs, eval_pairs, n_items, tag, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    feat = torch.from_numpy(np.ascontiguousarray(feat_np)).float().to(DEV)
    feat = F.normalize(feat, dim=-1)  # normalize raw input
    d_in = feat.shape[1]
    model = Proj(d_in, PROJ_HID, PROJ_OUT).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    tp = torch.from_numpy(train_pairs).long().to(DEV)
    ep = torch.from_numpy(eval_pairs).long().to(DEV)
    n_tr = tp.shape[0]

    g = torch.Generator(device=DEV); g.manual_seed(seed)

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_tr, device=DEV, generator=g)
        total = 0.0
        for s in range(0, n_tr, BATCH):
            idx = perm[s:s+BATCH]
            pi = tp[idx, 0]; pj = tp[idx, 1]
            # random negatives (item ids), independent of pi/pj
            neg = torch.randint(0, n_items, (idx.shape[0],), device=DEV, generator=g)
            zi = model(feat[pi]); zj = model(feat[pj]); zn = model(feat[neg])
            pos_cos = (zi * zj).sum(-1)
            neg_cos = (zi * zn).sum(-1)
            loss = -F.logsigmoid(pos_cos - neg_cos).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * idx.shape[0]
        # no print spam; keep last loss
        last_loss = total / n_tr

    # ---- eval AUC on held-out pairs ----
    model.eval()
    with torch.no_grad():
        # project all items once
        zall = []
        for s in range(0, n_items, 16384):
            zall.append(model(feat[s:s+16384]))
        zall = torch.cat(zall, 0)
        pi = ep[:, 0]; pj = ep[:, 1]
        ge = torch.Generator(device=DEV); ge.manual_seed(seed + 777)
        neg = torch.randint(0, n_items, (pi.shape[0],), device=DEV, generator=ge)
        pos_cos = (zall[pi] * zall[pj]).sum(-1)
        neg_cos = (zall[pi] * zall[neg]).sum(-1)
        # AUC = P(pos > neg) with ties counted as 0.5
        wins = (pos_cos > neg_cos).float()
        ties = (pos_cos == neg_cos).float()
        auc = (wins.sum() + 0.5 * ties.sum()).item() / pos_cos.shape[0]
        # also report RAW (untrained-projection) cosine AUC for reference
        rfeat = feat
        rpos = (rfeat[pi] * rfeat[pj]).sum(-1)
        rneg = (rfeat[pi] * rfeat[neg]).sum(-1)
        raw_auc = ((rpos > rneg).float().sum() + 0.5*(rpos==rneg).float().sum()).item() / rpos.shape[0]
    return auc, raw_auc, last_loss


def main():
    out = {}
    for ds in ['baby', 'sports', 'clothing']:
        t0 = time.time()
        rng = np.random.default_rng(SEED)
        img = np.load(f'{DATA}/{ds}_siglip2_base_patch16_224.npy')
        txt = np.load(f'{DATA}/{ds}_siglip2_text.npy')
        assert img.shape == txt.shape, (img.shape, txt.shape)
        n_items = img.shape[0]

        pairs = load_train_pairs(ds, rng)
        # 80/20 split of the pairs
        perm = rng.permutation(len(pairs))
        pairs = pairs[perm]
        n_split = int(0.8 * len(pairs))
        train_pairs = pairs[:n_split]
        eval_pairs = pairs[n_split:]

        img_auc, img_raw, img_loss = train_eval(img, train_pairs, eval_pairs, n_items, 'image')
        txt_auc, txt_raw, txt_loss = train_eval(txt, train_pairs, eval_pairs, n_items, 'text')

        rec = {
            'n_items': int(n_items),
            'n_pairs_total': int(len(pairs)),
            'n_train_pairs': int(len(train_pairs)),
            'n_eval_pairs': int(len(eval_pairs)),
            'auc_image': round(img_auc, 4),
            'auc_text': round(txt_auc, 4),
            'image_gt_text': bool(img_auc > txt_auc),
            'auc_gap_image_minus_text': round(img_auc - txt_auc, 4),
            'raw_cos_auc_image': round(img_raw, 4),
            'raw_cos_auc_text': round(txt_raw, 4),
            'final_loss_image': round(img_loss, 4),
            'final_loss_text': round(txt_loss, 4),
            'secs': round(time.time() - t0, 1),
        }
        out[ds] = rec
        print(json.dumps({ds: rec}), flush=True)

    out['_config'] = dict(SEED=SEED, MAX_PAIRS_PER_USER=MAX_PAIRS_PER_USER,
                          PROJ_HID=PROJ_HID, PROJ_OUT=PROJ_OUT, EPOCHS=EPOCHS,
                          BATCH=BATCH, LR=LR, device=DEV)
    res_path = '/workspace/Recsys/results/bai/independent_pair_image_vs_text.json'
    import os
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    with open(res_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('WROTE', res_path, flush=True)


if __name__ == '__main__':
    main()
