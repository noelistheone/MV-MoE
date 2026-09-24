"""Exact structural knockout across architectures — replaces the input-mean screen.

Why an exact instrument is needed: a test-time input-mean screen judged against one fixed
band treats every architecture alike, but models route content very differently, and on some
architectures the screen is structurally uninformative:
  * DAMRS/baby: image_knockout is BIT-IDENTICAL to both_knockout, and DAMRS/sports has
    text_knockout bit-identical to both_knockout -- the per-dimension mean equalises all
    cosines, the kNN masks go all-true and BOTH content graphs come out empty.
  * BM3: identically 0.0000 on all four datasets, because content never reaches
    full_sort_predict at all. A zero that is identical for image AND text carries no
    information in either direction and must be EXCLUDED from any count, not counted as
    evidence that "image is ignored".
  * freeze=False content tables (taxonomy C6): load_state_dict restores the TRAINED table
    over any mean-initialised one, so the perturbation never reaches the on-path tensor.

This driver runs the per-architecture exact modules in scripts/exact_ko/ (one per model,
each with its own reconstruction assert against the model's OWN scoring path) and judges
every delta against that (model, dataset)'s own measured noise floor rather than one
borrowed band.

Outputs -> results/phase_exact/. Recsys stays READ-ONLY.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path("/workspace/MechInterp")
RECSYS = Path("/workspace/Recsys")
sys.path.insert(0, str(RECSYS))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "interp"))
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_bridge import load_frozen                                  # noqa: E402
from ranking_effects import evaluate_item_matrix, ranking_change       # noqa: E402
from phase1_knockout import load_mde, verdict_reliability              # noqa: E402
from phase1_knockout import run_model as phase1_run_model              # noqa: E402

OUT = ROOT / "results" / "phase_exact"
PINS = ROOT / "results" / "phase_micro" / "ckpt_pins.json"
RECON_TOL = 1e-5
EXACT_MODELS = ["vbpr", "mmgcn", "lattice", "mgcn", "mentor", "gume", "damrs",
                "smore", "cohesion", "bm3"]
NATIVE_MODELS = ["freedom", "lgmrec"]     # already exact in phase1_knockout


def pins() -> dict:
    if not PINS.is_file():
        return {}
    raw = json.loads(PINS.read_text())
    return {k: (v["path"] if isinstance(v, dict) else v)
            for k, v in raw.items() if not k.startswith("_")}


def old_screen() -> dict:
    f = ROOT / "results" / "phasex_crossarch" / "crossarch_knockout.json"
    if not f.is_file():
        return {}
    out = {}
    for r in json.loads(f.read_text()):
        out[f"{r['model']}/{r['dataset']}"] = {
            k: r.get(k, {}).get("Recall@20")
            for k in ("image_knockout", "text_knockout", "both_knockout")}
    return out


def run_exact(model_name: str, dataset: str, device: str, ckpt: str | None) -> dict:
    mod = importlib.import_module(f"exact_ko.{model_name}")
    cfg, ds, model, loader = load_frozen(model_name, dataset, device, ckpt_path=ckpt)
    rec = float(mod.recon_error(model, ds, device))
    exactness = getattr(mod, "EXACTNESS", "UNKNOWN")
    if not str(exactness).startswith("UNDEFINED") and rec > RECON_TOL:
        raise AssertionError(f"{model_name}/{dataset}: reconstruction error {rec:.3e} > {RECON_TOL}")

    arms = mod.variants(model, ds, device)
    assert "baseline" in arms, f"{model_name}: no baseline arm"
    base_m, base_topk, _ = evaluate_item_matrix(*arms["baseline"], loader, device)
    mde, mde_src = load_mde(model_name, dataset)
    mde_n = None
    if mde_src and "(n=" in mde_src:
        mde_n = int(mde_src.split("(n=")[1].split(")")[0])
    elif mde_src:
        mde_n = 5

    rows = {"baseline": {"metrics": {k: float(v) for k, v in base_m.items()}}}
    for name, (u, i) in arms.items():
        if name == "baseline":
            continue
        m, topk, _ = evaluate_item_matrix(u, i, loader, device)
        d = {k: float(m[k]) - float(base_m[k]) for k in m}
        rows[name] = {
            "dR@20": d.get("Recall@20"), "dN@20": d.get("NDCG@20"),
            "significant_vs_MDE": (abs(d.get("Recall@20", 0)) > mde) if mde else None,
            "verdict_reliability": verdict_reliability(d.get("Recall@20", 0.0), mde, mde_n),
            "ranking_change": ranking_change(base_topk, topk, k=20),
        }
        del topk
    try:
        attrib = mod.attribution(model, ds, device)
    except Exception as e:  # noqa: BLE001
        attrib = {"error": repr(e)}

    return {"model": model_name, "dataset": dataset,
            "taxonomy_class": getattr(mod, "CLASS", "?"),
            "exactness": exactness,
            "checkpoint": getattr(model, "_ckpt_name", None),
            "checkpoint_pinned": getattr(model, "_ckpt_pinned", False),
            "recon_error": rec, "recon_tol": RECON_TOL,
            "MDE_R@20": mde, "MDE_source": mde_src, "MDE_n_seeds": mde_n,
            "attribution": attrib, "arms": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=EXACT_MODELS + NATIVE_MODELS)
    ap.add_argument("--datasets", nargs="+", default=["baby", "sports", "clothing"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "exact_crossarch.json"))
    ap.add_argument("--skip-done", action="store_true",
                    help="resume: skip (model,dataset) cells already present without an error")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    P, OLD = pins(), old_screen()

    out_path = Path(args.out)
    existing = {}
    if out_path.is_file():
        for r in json.loads(out_path.read_text()):
            existing[f"{r['model']}/{r['dataset']}"] = r

    for ds in args.datasets:
        for m in args.models:
            key = f"{m}/{ds}"
            if args.skip_done and key in existing and "error" not in existing[key]:
                print(f"skip {key} (already computed)", flush=True)
                continue
            print(f"\n=== {key} ===", flush=True)
            try:
                if m in NATIVE_MODELS:
                    r = phase1_run_model(m, ds, device, ckpt_path=P.get(key))
                    r = {"model": m, "dataset": ds, "taxonomy_class": "C1" if m == "freedom" else "C3",
                         "exactness": "EXACT" if m == "freedom" else "DELETE_PLUS_RENORM",
                         "checkpoint": r.get("checkpoint"), "checkpoint_pinned": r.get("checkpoint_pinned"),
                         "recon_error": r.get("recon_max_err"), "recon_tol": RECON_TOL,
                         "MDE_R@20": r.get("MDE_R@20"), "MDE_source": r.get("MDE_source"),
                         "MDE_n_seeds": r.get("MDE_n_seeds"), "attribution": r.get("attribution"),
                         "arms": {k: v for k, v in r["variants"].items()}}
                else:
                    r = run_exact(m, ds, device, P.get(key))
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                existing[key] = {"model": m, "dataset": ds, "error": repr(e)}
                out_path.write_text(json.dumps(list(existing.values()), indent=2))
                continue

            o = OLD.get(key, {})
            r["input_mean_screen"] = o
            img = r["arms"].get("image_knockout", {}).get("dR@20")
            if img is not None and o.get("image_knockout") is not None:
                r["screen_vs_exact"] = {
                    "screen_dR@20": o["image_knockout"], "exact_dR@20": img,
                    "screen_over_exact": (o["image_knockout"] / img) if img else None,
                    "screen_artifact_img_eq_both": (
                        o.get("both_knockout") is not None
                        and abs(o["image_knockout"] - o["both_knockout"]) < 1e-12),
                }
            existing[key] = r
            rec = r.get("recon_error")
            rec_s = f"{rec:.2e}" if isinstance(rec, (int, float)) else str(rec)
            print(f"  class={r['taxonomy_class']} exactness={str(r['exactness'])[:40]} "
                  f"recon={rec_s} MDE={r['MDE_R@20']}")
            for nm, v in r["arms"].items():
                if nm == "baseline" or "dR@20" not in v:
                    continue
                print(f"    {nm:28s} dR@20={v['dR@20']:+.6f} "
                      f"{'[SIG]' if v.get('significant_vs_MDE') else '[~noise]' if v.get('significant_vs_MDE') is not None else '[unjudged]'}")
            out_path.write_text(json.dumps(list(existing.values()), indent=2))
            torch.cuda.empty_cache()

    print(f"\nWrote {out_path}")
    n_ok = sum(1 for v in existing.values() if "error" not in v)
    print(f"{n_ok}/{len(existing)} cells computed, {len(existing)-n_ok} errored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
