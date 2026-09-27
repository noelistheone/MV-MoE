"""Share of test interactions whose item also occurs in training (warm items), per dataset.
Reads the MMRec-layout .inter files (x_label 0/1/2 = train/valid/test).
Output: results/phase_paper/warm_share.json"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILES = {"baby": "/workspace/Recsys/data/baby/baby.inter",
         "sports": "/workspace/Recsys/data/sports/sports.inter",
         "clothing": "/workspace/Recsys/data/clothing/clothing.inter",
         "microlens": "/workspace/Recsys/data/microlens/microlens.inter",
         "tiktok": str(ROOT / "data/tiktok/tiktok.inter")}
out = {}
for ds, f in FILES.items():
    df = pd.read_csv(f, sep="\t")
    item = [c for c in df.columns if c.lower().startswith("item")][0]
    train_items = set(df.loc[df["x_label"] == 0, item])
    test = df[df["x_label"] == 2]
    warm = test[item].isin(train_items)
    out[ds] = {"test_interactions": int(len(test)), "warm_share": float(warm.mean()),
               "cold_test_interactions": int((~warm).sum())}
(ROOT / "results/phase_paper").mkdir(exist_ok=True)
(ROOT / "results/phase_paper/warm_share.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
