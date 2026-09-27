"""Verify the encoder behind MicroLens's released image features on a handful of covers (CPU).

Encodes the covers saved by microlens_cover_probe.py with OpenAI CLIP RN50 (open_clip) and compares
to the official MicroLens-100k_image_features_CLIPRN50.npy rows (row = raw_video_id - 1), which are
bit-identical to Recsys/data/microlens/image_feat.npy (see microlens_feature_provenance_check.json).
Also times per-image encoding to size a full re-encode. Output:
results/phase_shortvideo2/microlens_clip_verify.json
"""
import json, time
from pathlib import Path
import numpy as np, torch, open_clip
from PIL import Image

ROOT = Path("/workspace/MechInterp")
S = ROOT / "data/microlens_raw/cover_samples"
OFF = np.load(ROOT / "data/microlens_raw/official_features/MicroLens-100k_image_features_CLIPRN50.npy")
t0 = time.time()
model, _, pre = open_clip.create_model_and_transforms("RN50", pretrained="openai")
model.eval(); load_s = time.time() - t0
files = sorted(S.glob("*.jpg"))
ids = [int(f.stem) for f in files]
x = torch.stack([pre(Image.open(f).convert("RGB")) for f in files])
t1 = time.time()
with torch.no_grad():
    e = model.encode_image(x).float().numpy()
enc_s = time.time() - t1
ref = OFF[np.array(ids) - 1]
def cos(a, b): return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
c = cos(e, ref)
# control: cosine to the official rows of OTHER sampled items (should be clearly lower)
perm = np.roll(np.arange(len(ids)), 1)
cc = cos(e, ref[perm])
res = {"model": "open_clip RN50 pretrained=openai", "n_images": len(ids), "raw_video_ids": ids,
       "cos_to_official_same_item": c.round(5).tolist(), "cos_to_official_other_item": cc.round(5).tolist(),
       "norm_ratio_ours_over_official": (np.linalg.norm(e, axis=1) / np.linalg.norm(ref, axis=1)).round(4).tolist(),
       "max_abs_diff": float(np.abs(e - ref).max()), "cpu_encode_s_per_image": enc_s / len(ids),
       "model_load_s": load_s}
out = ROOT / "results/phase_shortvideo2/microlens_clip_verify.json"
out.write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=1))
