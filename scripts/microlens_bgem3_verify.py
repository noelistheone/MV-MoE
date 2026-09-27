"""Verify the encoder behind MicroLens's released text features on a handful of titles (CPU).

Encodes English titles (MicroLens-100k_title_en.csv) with BAAI/bge-m3 dense embedding (CLS token of
the last hidden state, L2-normalized, as in FlagEmbedding) and compares to the official
MicroLens-100k_title_en_text_features_BgeM3.npy rows (row = raw_video_id - 1), which are bit-identical
to Recsys/data/microlens/text_feat.npy. Output: results/phase_shortvideo2/microlens_bgem3_verify.json
"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path("/workspace/MechInterp")
OFF = np.load(ROOT / "data/microlens_raw/official_features/MicroLens-100k_title_en_text_features_BgeM3.npy")
titles = pd.read_csv(ROOT / "data/microlens_raw/MicroLens-100k_title_en.csv", header=None,
                     names=["item", "title"], engine="python", quotechar='"')
ids = [9580, 14631, 7315, 3072, 10502, 6314, 9110, 13385]
tt = titles.set_index("item").loc[ids, "title"].astype(str).tolist()
t0 = time.time()
tok = AutoTokenizer.from_pretrained("BAAI/bge-m3"); m = AutoModel.from_pretrained("BAAI/bge-m3").eval()
load_s = time.time() - t0
res = {"model": "BAAI/bge-m3 (transformers, CLS pooling)", "raw_video_ids": ids, "titles": tt, "model_load_s": load_s,
       "official_row_norms": np.linalg.norm(OFF[np.array(ids) - 1], axis=1).round(4).tolist()}
for variant, texts in [("title_as_is", tt), ("title_stripped", [t.strip() for t in tt])]:
    b = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        h = m(**b).last_hidden_state[:, 0].float().numpy()
    ref = OFF[np.array(ids) - 1]
    cos = lambda a, r: (a * r).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(r, axis=1))
    res[variant] = {"cos_same_item": cos(h, ref).round(5).tolist(),
                    "cos_other_item": cos(h, ref[np.roll(np.arange(len(ids)), 1)]).round(5).tolist()}
out = ROOT / "results/phase_shortvideo2/microlens_bgem3_verify.json"
out.write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=1))
