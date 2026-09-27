"""Re-screen DAMRS on the repaired port (results/phase_c3conv/PREREG_C3CONV_AMEND2.md B6).

Inference only. For Baby, Sports and Clothing, on the repaired default-protocol seed-2024
checkpoint (g11nf_damrs_<ds>_s2024.pt, trained by scripts/exp_damrs_g11.py --protocol default):

  floor     MDE = 2 sd(test Recall@20) over the repaired default-protocol seeds 2024-2026
            (the formula of exp_mde_perdataset.summarize, ddof = 1); NDCG@20 likewise.
  deletion  exact_ko.damrs variants, evaluated exactly as exp_exact_crossarch.run_exact does
            (same reconstruction rule: absolute error <= RECON_TOL = 1e-5, else the cell is
            refused), plus an exact baseline guard against the trainer-logged test Recall@20
            (1e-9). Whether the AMEND1 A5 relative rule would also pass is recorded, never used
            to admit a cell. A refused cell gets status "REFUSED" and no comparison fields.
  averaging every item's image (text) feature replaced by the per-dimension item mean, the model
            rebuilt and scored through its own full_sort_predict, with the SAME checkpoint
            (phasex_crossarch_knockout.build_eval, with the checkpoint pinned instead of
            latest_ckpt), in the same order under one set_seed.

Per cell, "screen" (averaging, image, dR@20), "exact" (deletion, image, dR@20), "sign_flip" and
"screen_artifact_img_eq_both" have the meaning of the same fields in
results/phase_exact/screen_vs_exact.json, which is all that gen_facts.py reads for tab:subst.
"materially_misreported" is not produced: no saved script defines it. Deletion arms also carry
run_exact's "verdict_reliability", and the cell carries run_exact's "attribution".

Configuration: the model yaml (train_batch_size 2048) for every dataset, as in the floor runs. The
original Baby screen checkpoint (damrs_baby_20260616_100604.pt) was trained with batch size 4096;
see PREREG_C3CONV_AMEND2_ERRATUM.md. Nothing outside /workspace/MechInterp is written.
-> results/phase_c3conv_g11/screen_g11.json
"""
from __future__ import annotations

import hashlib
import importlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/workspace/MechInterp")
RECSYS = Path("/workspace/Recsys")
for _p in (str(RECSYS), str(ROOT / "src" / "models"), str(ROOT / "src" / "interp"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from recsys_bridge import (load_frozen, load_recsys_model_class, Config, set_seed,  # noqa: E402
                           RecDataset, EvalDataLoader, build_norm_adj, SCRATCH)
from ranking_effects import evaluate_item_matrix, ranking_change                   # noqa: E402
from src.common.trainer import Trainer                                            # noqa: E402
from exp_exact_crossarch import RECON_TOL                                         # noqa: E402
from phase1_knockout import verdict_reliability                                   # noqa: E402

G11 = ROOT / "results" / "phase_c3conv_g11"
MODEL_FILE = RECSYS / "src" / "models" / "damrs.py"
REGISTERED_SHA256 = "d001139f3cbafff1eac8e3a1c82019cbc96deb4fec7db56ae01bc51731a4f5ed"
ACCEPTED_SHA256 = (REGISTERED_SHA256, "b0bd3670c78d3f90a2deec378f0182880876a9fdbb0994f447bb76a7ce690e10")  # + this release's scrubbed copy   # a code release may add the hash of its scrubbed copy
SEEDS = (2024, 2025, 2026)
BANDS = {"CX noise_hint": 0.0026}        # phasex_crossarch_knockout.NOISE_HINT (tab:subst band)
BASE_TOL = 1e-9


def _runs(ds: str) -> dict:
    f = G11 / f"screen_runs_{ds}.json"
    out = {}
    for r in json.loads(f.read_text()):
        if "test_result" in r and r.get("code_ok") and r["seed"] in SEEDS:
            out[r["seed"]] = r
    return out


def _averaging(ds: str, ckpt: str, device: str) -> dict:
    cfg = Config("damrs", ds, cli_overrides={"ckpt_dir": str(SCRATCH / "ckpts"), "log_dir": str(SCRATCH / "logs")})
    set_seed(int(cfg.get("seed", 2024)), deterministic=True)
    dso = RecDataset(cfg)
    norm_adj = build_norm_adj(dso.train_matrix, dso.n_users, dso.n_items)
    loader = EvalDataLoader(dso, phase="test", batch_size=int(cfg.get("eval_batch_size_users", 1024)))
    v = torch.from_numpy(dso.v_feat[:].copy()); t = torch.from_numpy(dso.t_feat[:].copy())
    v_mean = v.mean(0, keepdim=True).expand_as(v).contiguous()
    t_mean = t.mean(0, keepdim=True).expand_as(t).contiguous()
    Cls = load_recsys_model_class("damrs")
    state = torch.load(ckpt, map_location=device, weights_only=False)["model_state_dict"]

    def ev(vf, tf):
        model = Cls(config=cfg, n_users=dso.n_users, n_items=dso.n_items, norm_adj=norm_adj,
                    v_feat=vf, t_feat=tf,
                    train_user_idx=torch.from_numpy(np.asarray(dso.train_users)),
                    train_item_idx=torch.from_numpy(np.asarray(dso.train_items))).to(device)
        inc = model.load_state_dict(state, strict=False)
        bad = [k for k in inc.missing_keys if k in {n for n, _ in model.named_parameters()}]
        assert not bad, f"missing learnable parameters {bad}"
        model.eval()
        res = Trainer(cfg, model, None, None, loader).evaluate(loader)
        del model; torch.cuda.empty_cache()
        return {k: float(x) for k, x in res.items()}

    base = ev(v, t); img = ev(v_mean, t); txt = ev(v, t_mean); both = ev(v_mean, t_mean)
    dl = lambda m: {k: m[k] - base[k] for k in m}                      # noqa: E731
    return {"baseline": base, "image_knockout": dl(img), "text_knockout": dl(txt), "both_knockout": dl(both)}


def _exact(ds: str, ckpt: str, device: str, mde: float, mde_n: int) -> dict:
    mod = importlib.import_module("exact_ko.damrs")
    cfg, dso, model, loader = load_frozen("damrs", ds, device, ckpt_path=ckpt)
    rec = float(mod.recon_error(model, dso, device))
    arms = mod.variants(model, dso, device)
    # Screen rule (exp_exact_crossarch.run_exact): refuse if abs error > RECON_TOL. The AMEND1 A5
    # relative rule is recorded for information only.
    bu, bi = arms["baseline"]
    idx = torch.arange(min(2048, bu.shape[0]), device=bu.device)
    scale = float((bu[idx] @ bi.t()).abs().max())
    recon_ok = rec <= RECON_TOL
    recon_ok_rel_A5 = rec <= RECON_TOL or (scale > 0 and rec / scale <= 1e-6)
    base_m, base_topk, _ = evaluate_item_matrix(*arms["baseline"], loader, device)
    rows = {"baseline": {"metrics": {k: float(x) for k, x in base_m.items()}}}
    for name, (u, i) in arms.items():
        if name == "baseline":
            continue
        m, topk, _ = evaluate_item_matrix(u, i, loader, device)
        d = {k: float(m[k]) - float(base_m[k]) for k in m}
        rows[name] = {"dR@20": d.get("Recall@20"), "dN@20": d.get("NDCG@20"), "delta": d,
                      "significant_vs_MDE": abs(d.get("Recall@20", 0)) > mde,
                      "verdict_reliability": verdict_reliability(d.get("Recall@20", 0.0), mde, mde_n),
                      "ranking_change": ranking_change(base_topk, topk, k=20)}
        del topk
    try:
        attrib = mod.attribution(model, dso, device)
    except Exception as e:  # noqa: BLE001
        attrib = {"error": repr(e)}
    del base_topk, model; torch.cuda.empty_cache()
    return {"exactness": getattr(mod, "EXACTNESS", "?"), "taxonomy_class": getattr(mod, "CLASS", "?"),
            "recon_error": rec, "recon_tol": RECON_TOL, "score_scale_max_abs": scale,
            "recon_error_rel": rec / scale if scale else None, "recon_ok": recon_ok,
            "recon_ok_rel_A5_info_only": recon_ok_rel_A5, "attribution": attrib,
            "missing_keys_checked": True, "arms": rows}


def main() -> int:
    h = hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest()
    if h not in ACCEPTED_SHA256:
        print(f"REFUSE: model file sha256 {h} not accepted"); return 3
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):   # GPU shared with 5 training jobs: same cap as the scorer
        tot = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, 3.3 / tot), 0)
    out = {"_meta": {"prereg": "results/phase_c3conv/PREREG_C3CONV_AMEND2.md (B6)", "model_file_sha256": h,
                     "bands": BANDS, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "device": device},
           "cells": []}
    for ds in ("baby", "sports", "clothing"):
        runs = _runs(ds)
        if set(runs) != set(SEEDS):
            print(f"{ds}: repaired default-protocol seeds present {sorted(runs)}; need {SEEDS} -- skipped")
            out["cells"].append({"model": "damrs_g11", "dataset": ds, "status": f"incomplete seeds {sorted(runs)}"})
            continue
        r20 = [runs[s]["test_result"]["Recall@20"] for s in SEEDS]
        n20 = [runs[s]["test_result"]["NDCG@20"] for s in SEEDS]
        mde, mde_n = 2 * statistics.stdev(r20), 2 * statistics.stdev(n20)
        ck = runs[2024]["ckpt_path"]
        print(f"\n=== damrs_g11/{ds}: MDE={mde:.6f} ckpt={Path(ck).name} ===", flush=True)
        ex = _exact(ds, ck, device, mde, len(SEEDS))
        base = ex["arms"]["baseline"]["metrics"]["Recall@20"]
        logged = runs[2024]["test_result"]["Recall@20"]
        guard = {"base_R20": base, "logged_R20": logged, "diff": base - logged, "tol": BASE_TOL,
                 "ok": abs(base - logged) <= BASE_TOL and ex["recon_ok"]}
        av = _averaging(ds, ck, device)
        img_ex, img_av = ex["arms"]["image_knockout"]["dR@20"], av["image_knockout"]["Recall@20"]
        txt_ex, txt_av = ex["arms"]["text_knockout"]["dR@20"], av["text_knockout"]["Recall@20"]
        cell = {"model": "damrs_g11", "dataset": ds, "checkpoint": ck, "seeds_floor": list(SEEDS),
                "R20_logged_by_seed": dict(zip(map(str, SEEDS), r20)), "MDE_R@20": mde, "MDE_N@20": mde_n,
                "guard": guard, "deletion": ex, "averaging": av,
                "screen": img_av, "exact": img_ex,
                "sign_flip": img_av * img_ex < 0, "identical": img_av == img_ex,
                "screen_artifact_img_eq_both": abs(av["image_knockout"]["Recall@20"] - av["both_knockout"]["Recall@20"]) < 1e-12,
                "text_screen": txt_av, "text_exact": txt_ex,
                "clears_floor_image": abs(img_ex) > mde, "clears_floor_text": abs(txt_ex) > mde,
                "x_floor_image": abs(img_ex) / mde if mde else None,
                "band_side": {lab: {"exact_outside": abs(img_ex) > b, "averaging_outside": abs(img_av) > b,
                                    "across": (abs(img_ex) > b) != (abs(img_av) > b)} for lab, b in BANDS.items()}}
        if not guard["ok"]:
            for k in ("screen", "exact", "sign_flip", "identical", "screen_artifact_img_eq_both", "text_screen",
                      "text_exact", "clears_floor_image", "clears_floor_text", "x_floor_image", "band_side"):
                cell.pop(k, None)
            cell["status"] = ("REFUSED: " + ("baseline != logged" if abs(base - logged) > BASE_TOL else "") +
                              (" reconstruction error > RECON_TOL" if not ex["recon_ok"] else ""))
        else:
            cell["status"] = "OK"
        out["cells"].append(cell)
        print(f"  guard ok={guard['ok']} base={base:.6f} logged={logged:.6f} recon={ex['recon_error']:.2e} "
              f"status={cell['status']}")
        if not guard["ok"]:
            continue
        print(f"  image: deletion {img_ex:+.6f} ({img_ex / base * 100:+.2f}%) averaging {img_av:+.6f} "
              f"x_floor={cell['x_floor_image']:.2f} flip={cell['sign_flip']} identical={cell['identical']}")
        print(f"  text : deletion {txt_ex:+.6f} averaging {txt_av:+.6f}", flush=True)
    G11.mkdir(parents=True, exist_ok=True)
    tmp = G11 / "screen_g11.json.tmp"
    tmp.write_text(json.dumps(out, indent=1)); tmp.replace(G11 / "screen_g11.json")
    print(f"wrote {G11 / 'screen_g11.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
