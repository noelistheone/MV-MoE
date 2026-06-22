"""Track A — extract REAL CLIP features for Amazon-Baby items.

Tests the biggest confound in the whole study: is "image ignored" only because the
standard 4096-d CNN features are weak, or does it survive STRONG VLM (CLIP ViT-L/14)
features? We extract CLIP image features from data/baby/images/{itemID}.jpg and CLIP
text features from item titles (meta-baby.csv). Saved into MechInterp (Recsys stays
read-only).

Output: MechInterp/data/clip/baby_{image,text}_clip.npy  ([n_items, 768] each).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

RECSYS = Path("/workspace/Recsys")
OUT = Path("/workspace/MechInterp/data/clip")
OUT.mkdir(parents=True, exist_ok=True)
DATASET = sys.argv[1] if len(sys.argv) > 1 else "baby"
MODEL, PRETRAINED = "ViT-L-14", "openai"   # strong, widely-used published CLIP


def main() -> int:
    import open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    tokenizer = open_clip.get_tokenizer(MODEL)
    model = model.to(device).eval()
    dim = model.visual.output_dim
    print(f"CLIP {MODEL}/{PRETRAINED} loaded, dim={dim}, device={device}")

    data_dir = RECSYS / "data" / DATASET
    # n_items from the standard feature file (canonical item count)
    n_items = np.load(data_dir / "image_feat.npy", mmap_mode="r").shape[0]
    img_dir = data_dir / "images"
    meta = pd.read_csv(data_dir / f"meta-{DATASET}.csv")
    titles = meta.set_index("itemID")["title"].to_dict()

    # ---------------- image features
    img_feats = np.zeros((n_items, dim), dtype=np.float32)
    have = np.zeros(n_items, dtype=bool)
    batch_imgs, batch_ids = [], []

    @torch.no_grad()
    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        f = model.encode_image(x).float().cpu().numpy()
        for j, iid in enumerate(batch_ids):
            img_feats[iid] = f[j]; have[iid] = True
        batch_imgs.clear(); batch_ids.clear()

    missing = 0
    for iid in range(n_items):
        p = img_dir / f"{iid}.jpg"
        if not p.is_file():
            missing += 1; continue
        try:
            batch_imgs.append(preprocess(Image.open(p).convert("RGB")))
            batch_ids.append(iid)
        except Exception:
            missing += 1; continue
        if len(batch_imgs) >= 256:
            flush()
            if iid % 2048 == 0:
                print(f"  image {iid}/{n_items}", flush=True)
    flush()
    # fill missing with mean of present
    if have.any():
        img_feats[~have] = img_feats[have].mean(0)
    print(f"image features done: {have.sum()}/{n_items} extracted, {missing} missing (mean-filled)")

    # ---------------- text features (titles)
    txt_feats = np.zeros((n_items, dim), dtype=np.float32)

    @torch.no_grad()
    def encode_text_batch(ids):
        texts = [str(titles.get(i, "")) or "product" for i in ids]
        toks = tokenizer(texts).to(device)
        return model.encode_text(toks).float().cpu().numpy()

    for s0 in range(0, n_items, 256):
        ids = list(range(s0, min(s0 + 256, n_items)))
        txt_feats[ids] = encode_text_batch(ids)
        if s0 % 2048 == 0:
            print(f"  text {s0}/{n_items}", flush=True)
    print("text features done")

    np.save(OUT / f"{DATASET}_image_clip.npy", img_feats)
    np.save(OUT / f"{DATASET}_text_clip.npy", txt_feats)
    # quick sanity: cosine-sim spread
    def spread(a):
        an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        idx = np.random.default_rng(0).choice(len(an), 2000, replace=False)
        s = an[idx] @ an[idx].T
        return float(s[np.triu_indices(2000, 1)].mean())
    print(f"saved -> {OUT}; mean cos-sim image={spread(img_feats):.3f} text={spread(txt_feats):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
