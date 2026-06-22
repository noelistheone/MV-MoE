"""Encode item TITLES with the SigLIP2 TEXT tower -> a VL-aligned text embedding that lives in the
SAME joint space as the SigLIP2 IMAGE embedding I already have (baby_siglip2_base_patch16_224.npy).

Gives a clean image-vs-text decomposition of the SAME VLM's joint semantics:
  Z_img  = SigLIP2 vision (image only)   [already have]
  Z_text = SigLIP2 text of the title     [this script]
  Z_joint= combine(Z_img, Z_text)        [VL joint semantic]
to test whether the multimodal 'image' contribution is actually text. Self-built (not Qwen2-VL).

Saves MechInterp/data/<ds>_siglip2_text.npy [n_items, d].
Usage: python encode_siglip_text.py <ds=baby>
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import numpy as np
import torch

RECSYS = Path("/workspace/Recsys")
OUT = Path("/workspace/MechInterp/data")
MODEL = "google/siglip2-base-patch16-224"
dev = "cuda" if torch.cuda.is_available() else "cpu"


def n_items(ds):
    rows = np.loadtxt(RECSYS / "data" / ds / f"{ds}.inter", delimiter="\t", skiprows=1,
                      dtype=np.int64, usecols=(1,))
    return int(rows.max()) + 1


def load_titles(ds, n):
    titles = [""] * n
    with open(RECSYS / "data" / ds / f"meta-{ds}.csv", newline="") as f:
        for row in csv.DictReader(f):
            try:
                iid = int(row["itemID"])
            except (ValueError, KeyError):
                continue
            if 0 <= iid < n:
                t = (row.get("title") or "").strip()
                titles[iid] = t if t else (row.get("brand") or "item")
    return titles


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "baby"
    OUT.mkdir(parents=True, exist_ok=True)
    n = n_items(ds)
    titles = load_titles(ds, n)
    n_empty = sum(1 for t in titles if not t)
    print(f"{ds}: {n} items, {n_empty} empty titles", flush=True)

    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev).eval()

    feats = None
    B = 256
    for s in range(0, n, B):
        batch = [t if t else "item" for t in titles[s:min(s + B, n)]]
        inp = proc(text=batch, return_tensors="pt", padding="max_length", max_length=64,
                   truncation=True).to(dev)
        with torch.no_grad():
            o = model.text_model(**inp)
            t = o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state[:, 0]
            z = t.float().cpu().numpy()
        if feats is None:
            feats = np.zeros((n, z.shape[1]), dtype=np.float32)
        feats[s:s + z.shape[0]] = z
        if s % 2048 == 0:
            print(f"  {s}/{n}", flush=True)
    out = OUT / f"{ds}_siglip2_text.npy"
    np.save(out, feats)
    print(f"saved {out} shape={feats.shape} norm_mean={np.linalg.norm(feats,axis=1).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
