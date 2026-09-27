"""Convert the DiffMM/MMSSL-preprocessed TikTok dataset to the MMRec layout the Recsys harness reads.

Source (downloaded verbatim into data/tiktok/raw_diffmm/):
    https://github.com/HKUDS/DiffMM/tree/main/Datasets/tiktok
    trnMat.pkl / valMat.pkl / tstMat.pkl : scipy COO [n_users, n_items], value 1.0
    image_feat.npy (6710x128 f16), audio_feat.npy (6710x128 f16), text_feat.npy (6710x768 f16)

Output (data/tiktok/):
    tiktok.inter   TSV  userID itemID rating timestamp x_label   (x_label 0=train 1=valid 2=test)
    image_feat.npy float32, row i = item i   (same row order as source; item ids are unchanged)
    text_feat.npy  float32, row i = item i
    audio_feat.npy float32 (kept for completeness; FREEDOM in the harness reads only image/text)
    conversion_manifest.json (checksums + counts)

The source split is KEPT: row in trnMat -> x_label 0, valMat -> 1, tstMat -> 2. User and item ids are
the source matrix indices (no re-indexing). No timestamps exist in the source, so timestamp = 0 and
rating = 1 (the harness ignores both).

Run:  python scripts/convert_tiktok_mmrec.py
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace/MechInterp")
SRC = ROOT / "data" / "tiktok" / "raw_diffmm"
DST = ROOT / "data" / "tiktok"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    mats = {}
    for name in ("trnMat", "valMat", "tstMat"):
        with (SRC / f"{name}.pkl").open("rb") as f:
            mats[name] = pickle.load(f).tocoo()
    shapes = {k: m.shape for k, m in mats.items()}
    assert len(set(shapes.values())) == 1, shapes
    n_users, n_items = mats["trnMat"].shape

    frames = []
    for label, name in enumerate(("trnMat", "valMat", "tstMat")):
        m = mats[name]
        assert np.all(m.data == 1.0), name
        frames.append(pd.DataFrame({"userID": m.row.astype(np.int64), "itemID": m.col.astype(np.int64),
                                    "rating": 1, "timestamp": 0, "x_label": label}))
    df = pd.concat(frames, ignore_index=True)
    # sanity: no duplicate (u,i) within or across splits
    dup = int(df.duplicated(["userID", "itemID"]).sum())
    assert dup == 0, f"{dup} duplicate (user,item) pairs"
    df = df.sort_values(["userID", "x_label", "itemID"], kind="stable").reset_index(drop=True)
    df.to_csv(DST / "tiktok.inter", sep="\t", index=False)

    feats = {}
    for f in ("image_feat", "text_feat", "audio_feat"):
        a = np.load(SRC / f"{f}.npy")
        assert a.shape[0] == n_items, (f, a.shape, n_items)
        a32 = a.astype(np.float32)
        assert np.isfinite(a32).all(), f
        np.save(DST / f"{f}.npy", a32)
        feats[f] = list(a32.shape)

    max_u, max_i = int(df.userID.max()), int(df.itemID.max())
    counts = df.x_label.value_counts().sort_index().to_dict()
    tr = df[df.x_label == 0]
    manifest = {
        "source": "https://github.com/HKUDS/DiffMM/tree/main/Datasets/tiktok (commit 1337447834aee475c1b1a59d291087e9ded9771a)",
        "matrix_shape": [n_users, n_items],
        "harness_n_users(max_id+1)": max_u + 1, "harness_n_items(max_id+1)": max_i + 1,
        "n_interactions": int(len(df)),
        "split_counts": {"train(0)": int(counts.get(0, 0)), "valid(1)": int(counts.get(1, 0)), "test(2)": int(counts.get(2, 0))},
        "users_with_zero_train": int(n_users - tr.userID.nunique()),
        "items_with_zero_train": int(n_items - tr.itemID.nunique()),
        "items_with_zero_any": int(n_items - df.itemID.nunique()),
        "users_in_valid": int(df[df.x_label == 1].userID.nunique()),
        "users_in_test": int(df[df.x_label == 2].userID.nunique()),
        "features": feats,
        "source_sha256": {p.name: sha256(p) for p in sorted(SRC.iterdir()) if p.is_file()},
        "output_sha256": {p.name: sha256(p) for p in sorted(DST.iterdir())
                          if p.is_file() and p.suffix in (".inter", ".npy")},
    }
    (DST / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if "sha256" not in k}, indent=2))


if __name__ == "__main__":
    main()
