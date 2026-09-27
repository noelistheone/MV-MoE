"""SMORE bracket ends for the C3 re-test (reporting only; no verdict uses these).

SMORE's module (scripts/exact_ko/smore.py) is BRACKET_ONLY: its default image/text arms alias
the bracket's HIGH end (branch + the shared bilinear joint branch), which is what the scorer
(scripts/exp_c3conv_score.py) records. The screen reported both ends; this script adds, for every
scored SMORE checkpoint, the LOW end (image_knockout_lo / text_knockout_lo: the modality's own
branch only) and joint_knockout (the joint branch alone), through the same load/variants/evaluate
path as the scorer, and checks that its baseline and hi-end deltas reproduce the scored row.

Incremental. Output: results/phase_c3conv/smore_bracket.json
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/workspace/MechInterp")
sys.path.insert(0, str(ROOT / "scripts"))
import exp_c3conv_score as sc  # noqa: E402  (sets sys.path for src/ and Recsys)

OUT = sc.C3 / "smore_bracket.json"
EXTRA = {"image_lo": "image_knockout_lo", "text_lo": "text_knockout_lo", "joint": "joint_knockout",
         "image_hi": "image_knockout_hi", "text_hi": "text_knockout_hi"}


def main() -> int:
    import importlib
    import torch
    from recsys_bridge import load_frozen
    from ranking_effects import evaluate_item_matrix

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        tot = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, sc.GPU_CAP_GB / tot), 0)
    mod = importlib.import_module("exact_ko.smore")
    done = sc._read_json(OUT) if OUT.is_file() else {}
    for ds in sc.DATASETS:
        f = sc.C3 / f"scored_{ds}.json"
        if not f.is_file():
            continue
        rows = {k: r for k, r in sc._read_json(f)["rows"].items() if r["model"] == "smore"}
        for key, row in sorted(rows.items()):
            k = f"{ds}/{key}"
            if k in done and done[k]["ckpt"] == row["ckpt"]:
                continue
            cfg, ds_obj, model, loader = load_frozen("smore", ds, device, ckpt_path=row["ckpt"])
            arms = mod.variants(model, ds_obj, device)
            base_m, _, _ = evaluate_item_matrix(*arms["baseline"], loader, device)
            base = sc._metrics(base_m)
            out = {"dataset": ds, "seed": row["seed"], "ckpt": row["ckpt"], "baseline": base,
                   "baseline_matches_scored_row": abs(base["Recall@20"] - row["baseline"]["Recall@20"]) < 1e-12}
            for name, arm in EXTRA.items():
                m, _, _ = evaluate_item_matrix(*arms[arm], loader, device)
                mm = sc._metrics(m)
                out[name] = {"arm": arm, "delta": {q: mm[q] - base[q] for q in sc.METRICS}}
            out["hi_matches_scored_image_delta"] = abs(
                out["image_hi"]["delta"]["Recall@20"] - row["arms"]["image"]["delta"]["Recall@20"]) < 1e-12
            done[k] = out
            print(f"{k}: base={base['Recall@20']:.5f} img lo={out['image_lo']['delta']['Recall@20']:+.5f} "
                  f"hi={out['image_hi']['delta']['Recall@20']:+.5f} joint={out['joint']['delta']['Recall@20']:+.5f} "
                  f"match={out['baseline_matches_scored_row']}/{out['hi_matches_scored_image_delta']}", flush=True)
            sc._locked_update(OUT, lambda d, k=k, v=out: (d.pop("rows", None), d.pop("refused", None), d.__setitem__(k, v)))
            del model, arms
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
