"""Plan gaps (Phase 4/5):

(A) LATENT <-> ITEM-ATTRIBUTE interpretability: train the gate-passing SAE
    (d_sae=1024, k=48) on FREEDOM's fused stream, then match latents to binary item
    attributes from meta-<ds>.csv (top category tokens, top brands, price bands)
    via point-biserial -> exact ROC-AUC (src/interp/feature_metrics.py). For each
    attribute report its best latent + that latent's modality origin.
(B) LINEAR-PROBE ABLATION baseline (Phase 5): fit a linear probe from fused to a
    per-item image-dominance score (‖h_img‖/(‖h_img‖+‖h_txt‖+‖cf‖)), ablate fused
    along the probe direction, measure ΔR@20 — third simple baseline next to
    exact knockout and difference-in-means.

Outputs -> results/phase4/latent_attributes.json, results/phase5/probe_ablation.json.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/workspace/MechInterp")
RECSYS = Path("/workspace/Recsys")
sys.path.insert(0, str(RECSYS))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "interp"))
sys.path.insert(0, str(ROOT / "src" / "sae"))
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_bridge import load_frozen                        # noqa: E402
from ranking_effects import evaluate_item_matrix             # noqa: E402
from feature_metrics import match_attributes_to_latents      # noqa: E402
from sae import SAE, SAEConfig                               # noqa: E402
from trainer import SAETrainer, TrainConfig                  # noqa: E402
from phase1_knockout import freedom_streams                  # noqa: E402


def build_attributes(meta_path: Path, n_items: int, top_cats=20, top_brands=10):
    df = pd.read_csv(meta_path)
    attrs: dict[str, torch.Tensor] = {}
    # --- categories: parse [['Baby','Bath']...]-style strings into token sets
    cat_tokens: list[set] = [set() for _ in range(n_items)]
    counter: Counter = Counter()
    for _, row in df.iterrows():
        iid = int(row["itemID"])
        if iid >= n_items:
            continue
        try:
            lists = ast.literal_eval(str(row.get("categories", "[]")))
            toks = {t for sub in lists for t in (sub if isinstance(sub, list) else [sub])}
        except (ValueError, SyntaxError):
            toks = set()
        toks.discard("Baby")                       # root category is on ~every item
        cat_tokens[iid] = toks
        counter.update(toks)
    for cat, cnt in counter.most_common(top_cats):
        y = torch.tensor([1.0 if cat in cat_tokens[i] else 0.0 for i in range(n_items)])
        if 0.01 < y.mean() < 0.99:
            attrs[f"cat:{cat}"] = y
    # --- brands
    brand = df.set_index("itemID")["brand"].to_dict()
    bc = Counter(str(b) for b in brand.values() if isinstance(b, str) and b.strip())
    for b, cnt in bc.most_common(top_brands):
        y = torch.tensor([1.0 if str(brand.get(i, "")) == b else 0.0 for i in range(n_items)])
        if y.sum() >= 20:
            attrs[f"brand:{b}"] = y
    # --- price quartile bands
    price = df.set_index("itemID")["price"].to_dict()
    pv = np.array([price.get(i, np.nan) for i in range(n_items)], dtype=float)
    valid = ~np.isnan(pv)
    if valid.sum() > 100:
        qs = np.nanquantile(pv, [0.25, 0.5, 0.75])
        bands = {"price:Q1(low)": pv <= qs[0], "price:Q4(high)": pv >= qs[2]}
        for nm, m in bands.items():
            y = torch.tensor((m & valid).astype(np.float32))
            attrs[nm] = y
    # popularity (train interaction count) as a free non-content attribute
    return attrs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="baby")
    ap.add_argument("--d_sae", type=int, default=1024)
    ap.add_argument("--k", type=int, default=48)        # gate-passing config
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    cfg, ds, model, test_loader = load_frozen("freedom", args.dataset, device)
    s = freedom_streams(model)
    fused, u_all = s["fused"], s["u_all"]
    h_img, h_txt, cf = s["h_img"], s["h_txt"], s["cf"]

    # ---------------- (A) latent <-> attribute matching
    sae = SAE(SAEConfig(d_in=64, d_sae=args.d_sae, variant="topk", k=args.k))
    tr = SAETrainer(sae, TrainConfig(lr=3e-4, batch_size=2048, epochs=args.epochs,
                                     device=device, log_every=args.epochs))
    tr.fit(fused)
    with torch.no_grad():
        Z = sae.encode(tr._apply_norm(fused))
        # modality origin per latent (stream-removal)
        sens = {nm: (Z - sae.encode(tr._apply_norm(fused - st))).abs().mean(0)
                for nm, st in (("image", h_img), ("text", h_txt), ("cf", cf))}
        origin = torch.stack([sens["image"], sens["text"], sens["cf"]], 1).argmax(1)
        onames = np.array(["image", "text", "cf"])

    # popularity attribute from train interactions
    meta = RECSYS / "data" / args.dataset / f"meta-{args.dataset}.csv"
    attrs = build_attributes(meta, ds.n_items) if meta.is_file() else {}
    pop = np.zeros(ds.n_items); np.add.at(pop, np.asarray(ds.train_items), 1)
    attrs["popularity:top-quartile"] = torch.tensor((pop >= np.quantile(pop, 0.75)).astype(np.float32))
    print(f"built {len(attrs)} binary attributes")

    res = match_attributes_to_latents(Z.cpu(), {k: v for k, v in attrs.items()}, topk=5)
    for r in res:
        r["best_latent_origin"] = str(onames[int(origin[r["best_latent"]])])
    out = {"dataset": args.dataset, "d_sae": args.d_sae, "k": args.k,
           "n_attributes": len(attrs),
           "top_matches": res[:25],
           "auc_summary": {"n_auc>0.8": sum(1 for r in res if r["roc_auc"] > 0.8),
                           "n_auc>0.7": sum(1 for r in res if r["roc_auc"] > 0.7),
                           "origins_of_best_latents": dict(Counter(r["best_latent_origin"] for r in res))}}
    p = ROOT / "results" / "phase4" / "latent_attributes.json"
    existing = json.loads(p.read_text()) if p.is_file() else {}
    existing[args.dataset] = out
    p.write_text(json.dumps(existing, indent=2))
    print("top attribute-latent matches (AUC | origin):")
    for r in res[:12]:
        print(f"  {r['attribute']:38} AUC={r['roc_auc']:.3f} f1={r['f1']:.3f} latent#{r['best_latent']} [{r['best_latent_origin']}]")
    print("origin histogram of best latents:", out["auc_summary"]["origins_of_best_latents"])

    # ---------------- (B) linear-probe ablation baseline
    with torch.no_grad():
        dom = h_img.norm(dim=-1) / (h_img.norm(dim=-1) + h_txt.norm(dim=-1) + cf.norm(dim=-1)).clamp_min(1e-8)
    X = fused - fused.mean(0, keepdim=True)
    y = (dom - dom.mean()).unsqueeze(1)
    w = torch.linalg.lstsq(X, y).solution.squeeze(1)         # ridge-free least squares probe
    w_dir = w / w.norm().clamp_min(1e-12)
    probe_r2 = float(1 - ((X @ w.unsqueeze(1) - y) ** 2).sum() / (y ** 2).sum())
    fused_abl = fused - (fused @ w_dir).unsqueeze(1) * w_dir.unsqueeze(0)
    m_base, _, _ = evaluate_item_matrix(u_all, fused, test_loader, device)
    m_abl, _, _ = evaluate_item_matrix(u_all, fused_abl, test_loader, device)
    dR = float(m_abl["Recall@20"]) - float(m_base["Recall@20"])
    p5 = ROOT / "results" / "phase5" / "probe_ablation.json"
    e5 = json.loads(p5.read_text()) if p5.is_file() else {}
    e5[args.dataset] = {"probe_target": "image-dominance ||h_img||/(sum of stream norms)",
                        "probe_R2": probe_r2, "ablation_dR@20": dR,
                        "compare": {"exact_knockout": "see phase5/causal_steering.json",
                                    "note": "probe direction ablation = third simple baseline"}}
    p5.write_text(json.dumps(e5, indent=2))
    print(f"\n(B) linear probe: R²={probe_r2:.3f}; ablate probe direction -> dR@20={dR:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
