"""Train-time modality holdout — separates training-time contribution from inference-time use.

Why this is needed: deleting the image pathway from a TRAINED model cannot show what image
contributed while the model was being trained -- visual information may already be baked into
the learned weights. A lambda-sweep does not answer that either. Setting
mm_image_weight=0 removes image from the item-item graph but LEAVES THE IMAGE AUXILIARY
BPR LOSS RUNNING (freedom.py:179-183 builds image_trs(image_embedding.weight) and adds a
BPR term on it), so the user/item embeddings are still shaped by image gradients. lambda=0
is a graph ablation, not a training ablation.

This script does the real thing: it withholds the modality from the constructor entirely
(v_feat=None), so
  * _build_mm_adj falls to the text-only branch (freedom.py:94-96)  -> no image in the graph
  * the image auxiliary loss is skipped (freedom.py:179)            -> no image in the loss
i.e. the model is trained as if the images never existed. Comparing a full retrain against
a modality-withheld retrain, across seeds and against the per-(model,dataset) MDE, is the
remove-and-retrain (ROAR-style) control. It is the LOGICAL COMPLEMENT of the inference-time
knockout: the knockout bounds what the pathway contributes at scoring time, this bounds what
the modality contributed over the whole of training.

Conditions: full | no_image | no_text  (no_text is the symmetric positive control -- if
withholding text also costs nothing, the instrument has no power on this dataset and the
cell is inadmissible).

Outputs -> results/phase_holdout/. Resumable. Recsys stays READ-ONLY.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/workspace/MechInterp")
RECSYS = Path("/workspace/Recsys")
sys.path.insert(0, str(RECSYS))
sys.path.insert(0, str(ROOT / "scripts"))
from src.utils import Config, set_seed                               # noqa: E402
from src.data.dataset import RecDataset                              # noqa: E402
from src.data.dataloader import EvalDataLoader, TrainDataLoader      # noqa: E402
from src.data.graph_utils import build_norm_adj                      # noqa: E402
from src.common.trainer import Trainer                               # noqa: E402
from phase0_repro import load_recsys_model                           # noqa: E402

OUT = ROOT / "results" / "phase_holdout"
CONDITIONS = ("full", "no_image", "no_text")


def train_holdout(model_name: str, dataset_name: str, seed: int, condition: str,
                  device: str, patience: int | None = None, epochs: int | None = None) -> dict:
    assert condition in CONDITIONS
    scratch = OUT / "_scratch"
    ov = {"seed": seed, "ckpt_dir": str(scratch / "ckpts"), "log_dir": str(scratch / "logs"),
          "show_progress": False}
    # The first 96 holdout runs used the harness default stopping_step=20, which the group's
    # truncation audit found cuts some models short; the audit of this experiment found a
    # truncation signature (ablated arms stop earlier and score lower). Patience/epochs are now
    # explicit so the holdout can be re-run to convergence.
    if patience is not None:
        ov["stopping_step"] = patience
    if epochs is not None:
        ov["epochs"] = epochs
    cfg = Config(model_name, dataset_name, cli_overrides=ov)
    set_seed(seed, deterministic=bool(cfg.get("cudnn_deterministic", True)))

    dataset = RecDataset(cfg)
    norm_adj = build_norm_adj(dataset.train_matrix, dataset.n_users, dataset.n_items)
    train_loader = TrainDataLoader(dataset, batch_size=int(cfg["train_batch_size"]),
                                   num_workers=int(cfg.get("num_workers", 4)),
                                   max_neg_tries=int(cfg.get("neg_sampling_max_tries", 100)))
    valid_loader = EvalDataLoader(dataset, phase="valid",
                                  batch_size=int(cfg.get("eval_batch_size_users", 1024)))
    test_loader = EvalDataLoader(dataset, phase="test",
                                 batch_size=int(cfg.get("eval_batch_size_users", 1024)))

    v = torch.from_numpy(dataset.v_feat[:].copy()) if dataset.v_feat is not None else None
    t = torch.from_numpy(dataset.t_feat[:].copy()) if dataset.t_feat is not None else None
    if condition == "no_image":
        v = None
    elif condition == "no_text":
        t = None
    assert v is not None or t is not None, "cannot withhold both modalities"

    ModelCls = load_recsys_model(model_name)
    kwargs = {"config": cfg, "n_users": dataset.n_users, "n_items": dataset.n_items,
              "norm_adj": norm_adj, "v_feat": v, "t_feat": t}
    if model_name.lower() in ("freedom", "grcn", "dragon"):
        kwargs.update(train_user_idx=torch.from_numpy(np.asarray(dataset.train_users)),
                      train_item_idx=torch.from_numpy(np.asarray(dataset.train_items)))
    model = ModelCls(**kwargs).to(device)

    # Assert the withheld modality really is absent from the built model, not merely unused.
    checks = {"has_v_feat": getattr(model, "v_feat", None) is not None,
              "has_t_feat": getattr(model, "t_feat", None) is not None,
              "has_image_trs": hasattr(model, "image_trs"),
              "has_text_trs": hasattr(model, "text_trs")}
    if condition == "no_image":
        assert not checks["has_v_feat"] and not checks["has_image_trs"], checks
    if condition == "no_text":
        assert not checks["has_t_feat"] and not checks["has_text_trs"], checks

    tag = "" if patience is None else f"_p{patience}" + ("" if epochs is None else f"e{epochs}")
    run = f"hold{tag}_{model_name}_{dataset_name}_{condition}_s{seed}"
    trainer = Trainer(cfg, model, train_loader, valid_loader, test_loader, run_name=run)
    # Record the per-epoch validation curve without touching the (read-only) Trainer: wrap its
    # validation method. Needed so convergence can be checked after the fact.
    curve = []
    _orig_valid = trainer._valid_epoch
    def _recording_valid():
        res = _orig_valid()
        curve.append([len(curve), float(res.get(cfg.get("valid_metric", "Recall@20"), float("nan")))])
        return res
    trainer._valid_epoch = _recording_valid
    t0 = time.time()
    result = trainer.fit()
    ck = scratch / "ckpts" / f"{run}.pt"
    return {"model": model_name, "dataset": dataset_name, "seed": seed,
            "condition": condition, "best_epoch": int(result["best_epoch"]),
            "train_min": (time.time() - t0) / 60.0,
            "ckpt_path": str(ck) if ck.is_file() else None,
            "modality_presence": checks,
            "stopping_step": int(cfg.get("stopping_step", -1)), "epochs_cap": int(cfg.get("epochs", -1)),
            "valid_curve": curve,
            "test_result": {k: float(v_) for k, v_ in result["test_result"].items()}}


def summarize(runs: list) -> dict:
    """Paired-by-seed condition contrasts against a floor measured from the FULL condition."""
    out: dict = {}
    for m in sorted({r["model"] for r in runs if "test_result" in r}):
        for ds in sorted({r["dataset"] for r in runs if r.get("model") == m and "test_result" in r}):
            cell = [r for r in runs if r.get("model") == m and r.get("dataset") == ds and "test_result" in r]
            by = {}
            for r in cell:
                by.setdefault(r["condition"], {})[r["seed"]] = r["test_result"]
            if "full" not in by or len(by["full"]) < 2:
                continue
            entry = {}
            for metric in ("Recall@20", "NDCG@20", "Recall@10", "NDCG@10"):
                fullv = {s: v[metric] for s, v in by["full"].items() if metric in v}
                if len(fullv) < 2:
                    continue
                f_level = 2 * statistics.stdev(list(fullv.values()))
                mrow = {"full_mean": statistics.mean(list(fullv.values())),
                        "full_sd": statistics.stdev(list(fullv.values())),
                        "F_level": f_level, "n_full_seeds": len(fullv)}
                for cond in ("no_image", "no_text"):
                    cv = {s: v[metric] for s, v in by.get(cond, {}).items() if metric in v}
                    if not cv:
                        continue
                    # PAIRED where seeds overlap -- the same seed is the same init/order
                    shared = sorted(set(cv) & set(fullv))
                    d = [cv[s] - fullv[s] for s in shared] if len(shared) >= 2 else None
                    mrow[cond] = {
                        "mean": statistics.mean(list(cv.values())), "n_seeds": len(cv),
                        "delta_unpaired": statistics.mean(list(cv.values())) - mrow["full_mean"],
                        "paired_seeds": shared,
                        "delta_paired_mean": statistics.mean(d) if d else None,
                        "F_paired": 2 * statistics.stdev(d) if d and len(d) >= 2 else None,
                    }
                    # Registered SECONDARY statistic (PREREG_HOLDOUT.md s.3): paired t-test.
                    # A retrain-vs-retrain contrast estimates a MEAN, and unlike the 2*sd
                    # band the t-test's power grows with seeds. Reported alongside, never
                    # instead of, the conservative two-floor verdict.
                    if d and len(d) >= 2:
                        n_d = len(d)
                        sd_d = statistics.stdev(d)
                        se_d = sd_d / (n_d ** 0.5)
                        tstat = statistics.mean(d) / se_d if se_d else None
                        mrow[cond]["paired_t"] = {
                            "t": tstat, "df": n_d - 1, "n": n_d, "sd": sd_d, "se": se_d,
                            "note": "two-sided; with df<=2 this is weak evidence whatever t is",
                        }
                    dm = mrow[cond]["delta_paired_mean"]
                    fp = mrow[cond]["F_paired"]
                    if dm is not None and fp is not None:
                        hi, lo = max(fp, f_level), min(fp, f_level)
                        mrow[cond]["verdict"] = ("SIGNIFICANT" if abs(dm) > hi
                                                 else "NULL" if abs(dm) < lo else "MARGINAL")
                        mrow[cond]["abs_over_F_paired"] = abs(dm) / fp if fp else None
                        mrow[cond]["abs_over_F_level"] = abs(dm) / f_level if f_level else None
                        # H-var (PREREG_HOLDOUT.md s.6): does withholding a modality
                        # DESTABILISE training rather than only shift its mean?
                        cvals = list(cv.values())
                        if len(cvals) >= 2 and mrow["full_sd"]:
                            mrow[cond]["sd_ratio_vs_full"] = statistics.stdev(cvals) / mrow["full_sd"]
                entry[metric] = mrow
            out.setdefault(m, {})[ds] = entry
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["freedom"])
    ap.add_argument("--datasets", nargs="+", default=["baby"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025, 2026])
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="freedom")
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    runs_path = OUT / f"{args.tag}_runs.json"
    runs = json.loads(runs_path.read_text()) if runs_path.exists() else []
    done = {(r["model"], r["dataset"], r["condition"], r["seed"]) for r in runs if "test_result" in r}

    t0 = time.time()
    for m in args.models:
        for ds in args.datasets:
            for cond in args.conditions:
                for s in args.seeds:
                    if (m, ds, cond, s) in done:
                        print(f"skip {m}/{ds}/{cond}/s{s}", flush=True)
                        continue
                    print(f"\n=== {m}/{ds} {cond} seed={s} | elapsed {(time.time()-t0)/60:.1f}m ===",
                          flush=True)
                    try:
                        r = train_holdout(m, ds, s, cond, device, args.patience, args.epochs)
                        runs.append(r)
                        print(f"  R@20={r['test_result']['Recall@20']:.5f} "
                              f"N@20={r['test_result']['NDCG@20']:.5f} "
                              f"(epoch {r['best_epoch']}, {r['train_min']:.1f} min)", flush=True)
                    except Exception as e:  # noqa: BLE001
                        import traceback; traceback.print_exc()
                        runs.append({"model": m, "dataset": ds, "condition": cond,
                                     "seed": s, "error": repr(e)})
                    runs_path.write_text(json.dumps(runs, indent=2))
                    (OUT / f"{args.tag}_summary.json").write_text(json.dumps(
                        {"summary": summarize(runs),
                         "note": ("Train-time modality holdout (ROAR-style). The withheld modality "
                                  "is absent from the constructor, so it enters neither the item-item "
                                  "graph nor the auxiliary loss. Verdict uses the same two-floor rule "
                                  "as results/phase_micro/PREREG.md s.3."),
                         "runs": runs}, indent=2))

    print("\n=== TRAIN-TIME HOLDOUT ===")
    for m, dd in summarize(runs).items():
        for ds, mm in dd.items():
            r20 = mm.get("Recall@20", {})
            print(f"  {m}/{ds}: full={r20.get('full_mean',0):.5f} F_level={r20.get('F_level',0):.5f}")
            for cond in ("no_image", "no_text"):
                c = r20.get(cond)
                if c:
                    print(f"    {cond:9s} mean={c['mean']:.5f} paired_d={c.get('delta_paired_mean')} "
                          f"-> {c.get('verdict')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
