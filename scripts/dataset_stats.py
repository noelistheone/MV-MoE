"""Dataset statistics the paper prints: users, items, interactions, density, split proportions,
time span, and how temporal the pre-assigned split is.

The split is the pre-assigned per-interaction `x_label` of the MMRec-format files (0 train,
1 validation, 2 test). It is not chronological, so we also report the share of users whose every test
interaction is at or after their last training interaction.

Output -> results/phase0/dataset_stats.json
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/workspace/Recsys/data")
OUT = ROOT / "results" / "phase0" / "dataset_stats.json"
DATASETS = ["baby", "sports", "clothing", "elec", "microlens"]


def stats(ds: str) -> dict:
    a = np.loadtxt(DATA / ds / f"{ds}.inter", delimiter="\t", skiprows=1, dtype=np.int64,
                   usecols=(0, 1, 3, 4))
    u, i, ts, lab = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    n_u, n_i, n = len(np.unique(u)), len(np.unique(i)), len(a)
    split = {k: int((lab == v).sum()) for k, v in (("train", 0), ("valid", 1), ("test", 2))}
    order = np.lexsort((ts, u))
    u_s, ts_s, lab_s = u[order], ts[order], lab[order]
    bounds = np.flatnonzero(np.diff(u_s)) + 1
    temporal = users_with_test = 0
    for seg_u, seg_ts, seg_lab in zip(np.split(u_s, bounds), np.split(ts_s, bounds), np.split(lab_s, bounds)):
        tr, te = seg_ts[seg_lab == 0], seg_ts[seg_lab == 2]
        if len(tr) and len(te):
            users_with_test += 1
            temporal += int(te.min() >= tr.max())
    # Amazon stores seconds, MicroLens milliseconds (13 digits); detect per file, never assume
    scale = 1000 if int(ts.max()) > 10**11 else 1
    utc = lambda s: dt.datetime.fromtimestamp(int(s) // scale, dt.timezone.utc).date().isoformat()
    return {
        "users": n_u, "items": n_i, "interactions": n,
        "density": n / (n_u * n_i),
        "split_counts": split,
        "split_share": {k: v / n for k, v in split.items()},
        "first_interaction_utc": utc(ts.min()), "last_interaction_utc": utc(ts.max()),
        "users_with_train_and_test": users_with_test,
        "share_users_test_after_last_train": temporal / users_with_test if users_with_test else None,
        "source": f"{ds}/{ds}.inter (x_label split)",
        "features": feature_sizes(ds),
    }


def feature_sizes(ds: str) -> dict:
    """Shapes and storage of the released content features: as stored, and at 32-bit floats
    (the paper's platform-cost comparison)."""
    out = {}
    for name in ("image_feat.npy", "text_feat.npy"):
        f = DATA / ds / name
        if not f.is_file():
            continue
        a = np.load(f, mmap_mode="r")
        out[name.split("_")[0]] = {"shape": list(a.shape), "dtype": str(a.dtype),
                                   "bytes_on_disk": f.stat().st_size,
                                   "bytes_float32": int(a.shape[0]) * int(np.prod(a.shape[1:])) * 4}
    return out


def freedom_config() -> dict:
    """FREEDOM's training configuration as used for every dataset (model yaml + framework defaults),
    and whether any dataset config overrides a model/training key (it would mean per-dataset tuning)."""
    import yaml
    cfg_dir = DATA.parent / "configs"
    model = yaml.safe_load((cfg_dir / "model" / "freedom.yaml").read_text())
    overall = yaml.safe_load((cfg_dir / "overall.yaml").read_text())
    keys = ["embedding_size", "n_layers", "n_mm_layers", "knn_k", "mm_image_weight", "dropout",
            "reg_weight", "learning_rate", "train_batch_size", "learner", "valid_metric"]
    merged = {k: model.get(k, overall.get(k)) for k in keys}
    overrides = {}
    for ds in DATASETS:
        dcfg = yaml.safe_load((cfg_dir / "dataset" / f"{ds}.yaml").read_text()) or {}
        hit = sorted(k for k in dcfg if k in keys or k in overall and k not in
                     ("dataset", "data_path", "inter_file_name", "USER_ID_FIELD", "ITEM_ID_FIELD"))
        overrides[ds] = hit
    return {"freedom": merged, "dataset_overrides_of_training_keys": overrides}


def main() -> int:
    out = {ds: stats(ds) for ds in DATASETS}
    (OUT.parent / "freedom_config.json").write_text(json.dumps(freedom_config(), indent=1))
    OUT.write_text(json.dumps(out, indent=1))
    for ds, s in out.items():
        sh = s["split_share"]
        print(f"{ds:9s} {s['users']:>7,} users {s['items']:>6,} items {s['interactions']:>9,} inter  "
              f"split {sh['train']:.3f}/{sh['valid']:.3f}/{sh['test']:.3f}  "
              f"{s['first_interaction_utc']}..{s['last_interaction_utc']}  "
              f"temporal-test {s['share_users_test_after_last_train']:.3f}")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
