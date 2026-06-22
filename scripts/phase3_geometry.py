"""Phase 3 — Representation-geometry diagnostics (WHY is image weak?).

Extract per-item streams from each frozen model, cache them, and compute geometry:
effective rank, participation ratio, anisotropy, uniformity, per-item norm; the
modality gap (image vs text); and linear-CKA redundancy between streams.

FREEDOM streams (what scoring sees + the raw content): img_raw/txt_raw = projected
image/text content (image_trs·embed), img_struct/txt_struct = h_img/h_txt graph
contributions, cf, fused. LGMRec: nv/nt (normalized modality), cge (cf), fused.

Outputs -> results/phase3/. Cached streams -> activations/. Recsys read-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import torch

ROOT = Path("/workspace/MechInterp")
RECSYS = Path("/workspace/Recsys")
sys.path.insert(0, str(RECSYS))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "interp"))
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_bridge import load_frozen                        # noqa: E402
import geometry as G                                         # noqa: E402
from phase1_knockout import freedom_streams, lgmrec_components, _lgmrec_combine  # noqa: E402

OUT = ROOT / "results" / "phase3"
ACT = ROOT / "activations"


@torch.no_grad()
def extract_streams(model_name: str, model) -> dict:
    if model_name == "freedom":
        s = freedom_streams(model)
        img_raw = model.image_trs(model.image_embedding.weight).detach()
        txt_raw = model.text_trs(model.text_embedding.weight).detach()
        return {"img_raw": img_raw, "txt_raw": txt_raw,
                "img_struct": s["h_img"], "txt_struct": s["h_txt"],
                "cf": s["cf"], "fused": s["fused"]}
    if model_name == "lgmrec":
        c = lgmrec_components(model)
        nU = c["nU"]
        _, fused_i = _lgmrec_combine(c, True, True)
        return {"img": c["nv"][nU:], "txt": c["nt"][nU:],
                "cf": c["cge"][nU:], "fused": fused_i}
    raise ValueError(model_name)


# which (image-side, other) pairs to compute modality-gap / CKA for, per model
PAIRS = {
    "freedom": {"modality_gap": ("img_raw", "txt_raw"),
                "cka": [("img_raw", "txt_raw"), ("img_raw", "cf"), ("img_raw", "fused"),
                        ("txt_raw", "fused"), ("cf", "fused"), ("img_struct", "fused")]},
    "lgmrec": {"modality_gap": ("img", "txt"),
               "cka": [("img", "txt"), ("img", "cf"), ("img", "fused"),
                       ("txt", "fused"), ("cf", "fused")]},
}


def run(model_name: str, dataset: str, device: str) -> dict:
    cfg, ds, model, _ = load_frozen(model_name, dataset, device)
    streams = extract_streams(model_name, model)

    cache_dir = ACT / f"{model_name}_{dataset}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, X in streams.items():
        torch.save(X.cpu(), cache_dir / f"{name}.pt")

    per_stream = {name: G.full_report(X) for name, X in streams.items()}
    mg_a, mg_b = PAIRS[model_name]["modality_gap"]
    modality_gap = G.modality_gap(streams[mg_a], streams[mg_b])
    cka = {f"{a}~{b}": G.linear_cka(streams[a], streams[b])
           for a, b in PAIRS[model_name]["cka"]}

    return {"model": model_name, "dataset": dataset,
            "checkpoint": getattr(model, "_ckpt_name", None),
            "n_items": ds.n_items,
            "per_stream_geometry": per_stream,
            "modality_gap_%s_vs_%s" % (mg_a, mg_b): modality_gap,
            "cka_redundancy": cka}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["freedom", "lgmrec"])
    ap.add_argument("--datasets", nargs="+", default=["baby", "sports", "clothing"])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    out_path = OUT / "geometry.json"
    existing = {}
    if out_path.is_file():
        for r in json.loads(out_path.read_text()):
            existing[f"{r['model']}/{r['dataset']}"] = r

    for ds in args.datasets:
        for m in args.models:
            print(f"\n=== {m} / {ds} ===", flush=True)
            try:
                r = run(m, ds, device)
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                existing[f"{m}/{ds}"] = {"model": m, "dataset": ds, "error": repr(e)}
                continue
            existing[f"{m}/{ds}"] = r
            g = r["per_stream_geometry"]
            img_key = "img_raw" if m == "freedom" else "img"
            txt_key = "txt_raw" if m == "freedom" else "txt"
            print(f"  eff_rank  img={g[img_key]['effective_rank']:.2f}  txt={g[txt_key]['effective_rank']:.2f}"
                  f"  cf={g['cf']['effective_rank']:.2f}  fused={g['fused']['effective_rank']:.2f}")
            print(f"  anisotropy img={g[img_key]['anisotropy']:.3f} txt={g[txt_key]['anisotropy']:.3f}"
                  f" | mean_norm img={g[img_key]['mean_norm']:.3f} txt={g[txt_key]['mean_norm']:.3f}")
            mgk = [k for k in r if k.startswith("modality_gap")][0]
            print(f"  {mgk}={r[mgk]:.3f} | CKA img~txt={r['cka_redundancy'].get(f'{img_key}~{txt_key}'):.3f}"
                  f" img~fused={r['cka_redundancy'].get(f'{img_key}~fused'):.3f}")
            out_path.write_text(json.dumps(list(existing.values()), indent=2))

    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
