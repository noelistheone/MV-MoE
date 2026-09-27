"""Behavioral alignment (the Table 3 statistic) on TikTok, from training interactions only, plus the
registered T4 comparison with MicroLens.

The statistic is not re-implemented: exp_alignment_stats.run_dataset() is called unchanged
(co-consumption pairs = two distinct items in one user's TRAINING history, copurchase_pairs;
distance ||f(x)-f(y)||^2 on L2-normalized vectors; gap g = mean_dist(random) - mean_dist(co);
dispersion-normalized gap g / mean_dist(random); 95% bootstrap CI, B=2000; one-sided permutation p,
B=2000; same seeds; same ordering tests), once with the uniform null (the alignment_microlens.json /
alignment_stats.json form) and once with the popularity-matched null (alignment_degreematched.json
form). Only the dataset overlay is installed first. Device: CPU (so random-pair and bootstrap draws
come from the CPU generator; the Amazon/MicroLens artifacts were made on GPU -> same distribution,
different draws; the MicroLens CPU replica below measures how much that moves the numbers).

Streams: raw image (TikTok: 128-d key-frame feature, undocumented encoder), raw text (768-d
Sentence-BERT), and FREEDOM's propagated h_img / h_txt, behavioral (cf) and fused terms from a
converged TikTok full checkpoint (the lowest-seed FINISHED run; Baby/MicroLens used their
default-patience checkpoints -- see "checkpoint_note"). If no converged checkpoint has finished,
only the raw streams are computed (same analyze_stream function).

Leakage guard: asserts that the pairs come from the training split only (len(train_users) equals
the number of x_label == 0 rows in tiktok.inter, and every sampled pair is a training co-interaction).

T4 (PREREG_TIKTOK.md): text-minus-image gap on TikTok vs on MicroLens, per null, for raw and
propagated streams, absolute and dispersion-normalized; each dataset's difference gets a PAIRED
bootstrap CI (co and random pairs resampled once and applied to both streams; B=2000, BOOT_SEED),
and TikTok-minus-MicroLens gets the CI of the difference of the two independent bootstrap
distributions. MicroLens is recomputed on CPU for this with its pinned checkpoint (AM's).

Output -> results/phase_shortvideo2/alignment_tiktok.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/workspace/MechInterp")
sys.path.insert(0, str(ROOT / "scripts"))
import tiktok_common as tc                 # noqa: E402

tc.install_overlay()
import exp_alignment_stats as eas          # noqa: E402
from recsys_bridge import load_frozen      # noqa: E402
from phase1_knockout import freedom_streams  # noqa: E402

OUT = tc.SV2 / "alignment_tiktok.json"
AM = ROOT / "results" / "phase_micro" / "alignment_microlens.json"
AD = ROOT / "results" / "phase_micro" / "alignment_degreematched.json"
AL = ROOT / "results" / "phase_align" / "alignment_stats.json"
ML_PIN = json.loads((ROOT / "results" / "phase_micro" / "ckpt_pins.json").read_text())["freedom/microlens"]["path"]
DEV = "cpu"
NULLS = {"uniform": False, "popularity_matched": True}


def leakage_check(ds) -> dict:
    inter = pd.read_csv(ROOT / "data" / "tiktok" / "tiktok.inter", sep="\t")
    n0 = int((inter["x_label"] == 0).sum())
    ntr = len(ds.train_users)
    assert ntr == n0, f"train_users {ntr} != x_label==0 rows {n0}"
    ia, ib = eas.copurchase_pairs(ds, eas.N_PAIRS)
    tr = {}
    for u, i in zip(np.asarray(ds.train_users).tolist(), np.asarray(ds.train_items).tolist()):
        tr.setdefault(i, set()).add(u)
    bad = sum(1 for a, b in zip(ia.tolist(), ib.tolist()) if not (tr.get(a, set()) & tr.get(b, set())))
    assert bad == 0, f"{bad} sampled pairs are not training co-interactions"
    ev = inter[inter["x_label"] != 0]
    return {"train_rows_x_label_0": n0, "train_users_len": ntr, "n_pairs": int(ia.numel()),
            "pairs_not_training_cointeractions": bad, "valid_test_rows_excluded": int(len(ev))}


def raw_only(ds, degree_matched: bool) -> dict:
    """Fallback when no converged checkpoint has finished: the raw streams only, same function."""
    ia, ib = eas.copurchase_pairs(ds, eas.N_PAIRS)
    deg_p = eas.copurchase_item_marginal(ds, ds.n_items) if degree_matched else None
    streams = {"raw_image_cnn": torch.from_numpy(np.asarray(ds.v_feat[:])).float(),
               "raw_text_bert": torch.from_numpy(np.asarray(ds.t_feat[:])).float()}
    res = {nm: eas.analyze_stream(nm, X, ia, ib, DEV, deg_p=deg_p) for nm, X in streams.items()}
    below = lambda a, b: res[a]["gap_abs_ci95"][1] < res[b]["gap_abs_ci95"][0]  # noqa: E731
    return {"dataset": "tiktok", "n_copurchase_pairs": int(ia.numel()), "checkpoint": None,
            "streams": res,
            "ordering_tests": {"raw: image_cnn < text_bert (CIs separate)": below("raw_image_cnn", "raw_text_bert"),
                               "raw: image > text  [INVERSION, CIs separate]": below("raw_text_bert", "raw_image_cnn")}}


# ------------------------------------------------------------------------ T4 paired bootstrap
@torch.no_grad()
def _draws(n_items: int, deg_p):
    """The random pairs analyze_stream draws (same generator, seed and calls)."""
    g = torch.Generator(device=DEV).manual_seed(eas.RAND_SEED)
    if deg_p is None:
        ra = torch.randint(0, n_items, (eas.N_RAND,), generator=g, device=DEV)
        rb = torch.randint(0, n_items, (eas.N_RAND,), generator=g, device=DEV)
    else:
        pp = deg_p.to(DEV)
        ra = torch.multinomial(pp, eas.N_RAND, replacement=True, generator=g)
        rb = torch.multinomial(pp, eas.N_RAND, replacement=True, generator=g)
    keep = ra != rb
    return ra[keep], rb[keep]


@torch.no_grad()
def paired_boot(streams: dict, ia, ib, deg_p, pairs=(("raw_text_bert", "raw_image_cnn"),
                                                     ("h_txt_graph", "h_img_graph")), B=eas.B_BOOT):
    n_items = next(iter(streams.values())).shape[0]
    ra, rb = _draws(n_items, deg_p)
    co = {k: eas.pair_dists(X, ia, ib) for k, X in streams.items()}
    rd = {k: eas.pair_dists(X, ra, rb) for k, X in streams.items()}
    g = torch.Generator(device=DEV).manual_seed(eas.BOOT_SEED)
    Nc, Nr = ia.numel(), ra.numel()
    boots = {k: {"abs": [], "rel": []} for k in streams}
    done = 0
    while done < B:
        b = min(500, B - done)
        ci = torch.randint(0, Nc, (b, Nc), generator=g, device=DEV)
        ri = torch.randint(0, Nr, (b, Nr), generator=g, device=DEV)
        for k in streams:
            mc, mr = co[k][ci].mean(1), rd[k][ri].mean(1)
            boots[k]["abs"].append(mr - mc)
            boots[k]["rel"].append((mr - mc) / mr.clamp_min(1e-9))
        done += b
    boots = {k: {m: torch.cat(v) for m, v in d.items()} for k, d in boots.items()}
    point = {k: {"abs": float(rd[k].mean() - co[k].mean()),
                 "rel": float((rd[k].mean() - co[k].mean()) / rd[k].mean())} for k in streams}
    out = {"point": point, "diff": {}, "_boot": {}}
    for t, i in pairs:
        if t in streams and i in streams:
            tag = "raw" if t.startswith("raw") else "propagated"
            for m in ("abs", "rel"):
                dist = boots[t][m] - boots[i][m]
                lo, hi = torch.quantile(dist, torch.tensor([0.025, 0.975])).tolist()
                out["diff"][f"{tag}/{m}"] = {"text_minus_image": point[t][m] - point[i][m], "ci95": [lo, hi]}
                out["_boot"][f"{tag}/{m}"] = dist
    return out


def stream_tensors(model, ds) -> dict:
    s = freedom_streams(model)
    return {"raw_image_cnn": model.v_feat.float(), "raw_text_bert": model.t_feat.float(),
            "h_img_graph": s["h_img"], "h_txt_graph": s["h_txt"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-microlens-replica", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(max(1, min(16, torch.get_num_threads())))
    cfg, ds = tc.load_dataset("tiktok")
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "device": DEV,
           "metric": "||f(x)-f(y)||^2, f=L2-normalize (Wang&Isola alignment, =2-2cos)",
           "headline": "absolute gap = mean_dist(random) - mean_dist(co-consumed); higher=more predictive",
           "config": {"n_pairs": eas.N_PAIRS, "n_rand": eas.N_RAND, "B_boot": eas.B_BOOT, "B_perm": eas.B_PERM,
                      "pair_seed": eas.PAIR_SEED, "boot_seed": eas.BOOT_SEED, "perm_seed": eas.PERM_SEED},
           "leakage_check": leakage_check(ds)}
    fin = tc.finished_full_ckpts()
    ck = fin[0] if fin else None
    out["checkpoint_note"] = (
        f"propagated streams from the converged TikTok full checkpoint {ck[1].name} (seed {ck[0]}, "
        f"logged R@20 {ck[2]['test_result']['Recall@20']:.6f}, best epoch {ck[2]['best_epoch']}). Baby "
        "(alignment_stats.json) and MicroLens (alignment_microlens.json) used default-patience "
        "checkpoints; raw-stream numbers do not depend on any checkpoint." if ck else
        "no converged TikTok full checkpoint finished yet: raw streams only")
    feat_note = ("TikTok (MMSSL/DiffMM release): 'raw_image_cnn' = 128-d video key-frame feature of "
                 "undocumented encoder, 'raw_text_bert' = 768-d Sentence-BERT caption feature; the keys "
                 "are the legacy Amazon names, NOT the encoders used here.")
    out["nulls"] = {}
    for nm, dm in NULLS.items():
        print(f"\n=== tiktok, {nm} null ===", flush=True)
        if ck:
            r = eas.run_dataset("tiktok", DEV, ckpt_path=str(ck[1]), degree_matched=dm)
        else:
            r = raw_only(ds, dm)
        r["feature_note"] = feat_note
        out["nulls"][nm] = r
        OUT.write_text(json.dumps(out, indent=2, default=str))

    # ---------------- side by side with the existing artifacts
    def pick(path, dsname):
        if not path.is_file():
            return None
        for d in json.loads(path.read_text())["datasets"]:
            if d.get("dataset") == dsname and "streams" in d:
                return d
        return None
    keys = ("raw_image_cnn", "raw_text_bert", "h_img_graph", "h_txt_graph", "cf", "fused")
    side = {}
    for nm, src in (("uniform", {"baby": (AL, "alignment_stats.json"), "microlens": (AM, "alignment_microlens.json")}),
                    ("popularity_matched", {"baby": (AD, "alignment_degreematched.json"),
                                            "microlens": (AD, "alignment_degreematched.json")})):
        tab = {}
        for dsn, (p, lab) in src.items():
            d = pick(p, dsn)
            if d:
                tab[dsn] = {"source": f"results/{p.parent.name}/{lab}", "checkpoint": d.get("checkpoint"),
                            "streams": {k: {x: d["streams"][k][x] for x in ("gap_abs", "gap_abs_ci95", "gap_rel", "gap_rel_ci95", "perm_p_value")}
                                        for k in keys if k in d["streams"]}}
        tk = out["nulls"][nm]["streams"]
        tab["tiktok"] = {"source": "this file", "checkpoint": out["nulls"][nm].get("checkpoint"),
                         "streams": {k: {x: tk[k][x] for x in ("gap_abs", "gap_abs_ci95", "gap_rel", "gap_rel_ci95", "perm_p_value")}
                                     for k in keys if k in tk}}
        side[nm] = tab
    out["side_by_side"] = side

    # ---------------- T4: text-minus-image, TikTok vs MicroLens, paired bootstrap per dataset
    t4 = {}
    ia_t, ib_t = eas.copurchase_pairs(ds, eas.N_PAIRS)
    if ck:
        _, _, mt, _ = load_frozen("freedom", "tiktok", DEV, ckpt_path=str(ck[1]))
        st_t = stream_tensors(mt, ds)
    else:
        st_t = {"raw_image_cnn": torch.from_numpy(np.asarray(ds.v_feat[:])).float(),
                "raw_text_bert": torch.from_numpy(np.asarray(ds.t_feat[:])).float()}
    ml = None
    if not args.skip_microlens_replica:
        _, ds_m, mm, _ = load_frozen("freedom", "microlens", DEV, ckpt_path=ML_PIN)
        ia_m, ib_m = eas.copurchase_pairs(ds_m, eas.N_PAIRS)
        st_m = stream_tensors(mm, ds_m)
        ml = (ds_m, ia_m, ib_m, st_m)
    for nm, dm in NULLS.items():
        dp_t = eas.copurchase_item_marginal(ds, ds.n_items) if dm else None
        bt = paired_boot(st_t, ia_t, ib_t, dp_t)
        # the paired bootstrap must see the SAME pairs analyze_stream saw (point gaps equal)
        for k, v in bt["point"].items():
            ref = out["nulls"][nm]["streams"][k]
            assert abs(v["abs"] - ref["gap_abs"]) < 1e-6, (nm, k, v["abs"], ref["gap_abs"])
        amd = pick(AM if nm == "uniform" else AD, "microlens")
        blk = {}
        bm = None
        if ml:
            ds_m, ia_m, ib_m, st_m = ml
            dp_m = eas.copurchase_item_marginal(ds_m, ds_m.n_items) if dm else None
            bm = paired_boot(st_m, ia_m, ib_m, dp_m)
        for key in bt["diff"]:
            tag, m = key.split("/")
            ti, ii = ("raw_text_bert", "raw_image_cnn") if tag == "raw" else ("h_txt_graph", "h_img_graph")
            gk = "gap_abs" if m == "abs" else "gap_rel"
            ml_stored = (amd["streams"][ti][gk] - amd["streams"][ii][gk]) if amd else None
            e = {"tiktok_text_minus_image": bt["diff"][key]["text_minus_image"],
                 "tiktok_ci95_paired_bootstrap": bt["diff"][key]["ci95"],
                 "microlens_text_minus_image_stored": ml_stored,
                 "microlens_source": f"results/phase_micro/{'alignment_microlens.json' if nm == 'uniform' else 'alignment_degreematched.json'}"}
            if ml_stored is not None:
                e["tiktok_minus_microlens"] = e["tiktok_text_minus_image"] - ml_stored
                e["t4_holds_point_estimate"] = e["tiktok_text_minus_image"] <= ml_stored
            if bm and key in bm["diff"]:
                e["microlens_cpu_replica_text_minus_image"] = bm["diff"][key]["text_minus_image"]
                e["microlens_cpu_replica_ci95"] = bm["diff"][key]["ci95"]
                dd = bt["_boot"][key] - bm["_boot"][key]
                lo, hi = torch.quantile(dd, torch.tensor([0.025, 0.975])).tolist()
                e["tiktok_minus_microlens_replica"] = e["tiktok_text_minus_image"] - e["microlens_cpu_replica_text_minus_image"]
                e["tiktok_minus_microlens_ci95"] = [lo, hi]
                e["t4_excess_resolved"] = ("TikTok text-image exceeds MicroLens (CI > 0)" if lo > 0 else
                                           "TikTok text-image below MicroLens (CI < 0)" if hi < 0 else "overlap")
            blk.setdefault(tag, {})[m.replace("abs", "gap_abs").replace("rel", "gap_rel")] = e
        if bm and amd:
            blk["microlens_cpu_replica_vs_stored_gap_abs"] = {
                k: {"cpu_replica": v["abs"], "stored_gpu": amd["streams"][k]["gap_abs"],
                    "stored_ci95": amd["streams"][k]["gap_abs_ci95"],
                    "replica_inside_stored_ci": amd["streams"][k]["gap_abs_ci95"][0] <= v["abs"] <= amd["streams"][k]["gap_abs_ci95"][1]}
                for k, v in bm["point"].items()}
        t4[nm] = blk
    out["t4_comparison"] = t4
    out["seconds"] = time.time() - t0
    OUT.write_text(json.dumps(out, indent=2, default=str))

    print("\nSide by side (uniform null), gap_abs [95% CI] (gap_rel):")
    for k in keys:
        cells = []
        for dsn in ("baby", "microlens", "tiktok"):
            s = side["uniform"].get(dsn, {}).get("streams", {}).get(k)
            cells.append(f"{s['gap_abs']:.4f} [{s['gap_abs_ci95'][0]:.4f},{s['gap_abs_ci95'][1]:.4f}] ({s['gap_rel']:.3f})" if s else "--")
        print(f"  {k:14s} " + " | ".join(cells))
    for nm in t4:
        for tag in ("raw", "propagated"):
            e = t4[nm].get(tag, {}).get("gap_abs")
            if e:
                print(f"T4 {nm} {tag}: TikTok text-image {e['tiktok_text_minus_image']:+.4f} vs MicroLens "
                      f"{e['microlens_text_minus_image_stored']:+.4f} -> holds(point)={e.get('t4_holds_point_estimate')} "
                      f"diff CI {e.get('tiktok_minus_microlens_ci95')}")
    print(f"Wrote {OUT} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
