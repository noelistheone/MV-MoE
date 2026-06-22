"""Re-encode Baby item images with DINOv2 (stronger self-supervised visual features than the
dataset's old CNN image_feat.npy). Tests the hypothesis that a modern encoder has a LARGER
co-purchase-predictive subspace. Saves to MechInterp/data/baby_<model>.npy [n_items, d].
Recsys read-only (images are only read)."""
from __future__ import annotations
import sys, numpy as np, torch
from pathlib import Path
from PIL import Image

MODEL = sys.argv[1] if len(sys.argv) > 1 else "facebook/dinov2-base"
DS = sys.argv[2] if len(sys.argv) > 2 else "baby"
IMG = Path(f"/workspace/Recsys/data/{DS}/images")
OUT = Path("/workspace/MechInterp/data"); OUT.mkdir(parents=True, exist_ok=True)
N = int(np.loadtxt(f"/workspace/Recsys/data/{DS}/{DS}.inter", delimiter="\t", skiprows=1,
                   dtype=np.int64, usecols=(1,)).max()) + 1
dev = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoModel, AutoImageProcessor
proc = AutoImageProcessor.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(dev).eval().half()

def load(i):
    p = IMG / f"{i}.jpg"
    if not p.exists():
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None

feats = None   # allocated lazily from the first batch's output dim
missing = []
B = 64
batch_imgs, batch_ids = [], []

def flush():
    global feats
    if not batch_imgs:
        return
    inp = proc(images=batch_imgs, return_tensors="pt").to(dev)
    inp = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inp.items()}
    with torch.no_grad():
        if "siglip" in MODEL.lower():                      # SigLIP: pooled image embedding
            o = model.vision_model(**inp)
            t = o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)
        else:                                              # DINOv2 style: CLS token
            t = model(**inp).last_hidden_state[:, 0]
        cls = t.float().cpu().numpy()
    if feats is None:
        feats = np.zeros((N, cls.shape[1]), dtype=np.float32)
    for j, iid in enumerate(batch_ids):
        feats[iid] = cls[j]
    batch_imgs.clear(); batch_ids.clear()

for i in range(N):
    im = load(i)
    if im is None:
        missing.append(i); continue
    batch_imgs.append(im); batch_ids.append(i)
    if len(batch_imgs) >= B:
        flush()
    if i % 1000 == 0:
        print(f"  encoded {i}/{N}", flush=True)
flush()

# fallback for missing images: mean of encoded features
if missing:
    present = [i for i in range(N) if i not in set(missing)]
    feats[missing] = feats[present].mean(0)
print(f"missing/fallback: {len(missing)} items")

name = MODEL.split("/")[-1].replace("-", "_")
out = OUT / f"{DS}_{name}.npy"
np.save(out, feats)
print(f"saved {out}  shape={feats.shape}  norm_mean={np.linalg.norm(feats,axis=1).mean():.3f}")
