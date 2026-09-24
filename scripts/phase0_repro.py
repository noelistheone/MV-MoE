"""Phase 0.1/0.2 — Reproduction validation.

Load existing FREEDOM/LGMRec/LightGCN checkpoints from the Recsys repo, recompute
full-ranking R@K/N@K with the *unchanged* Recsys evaluator, and compare against
each run's own logged test_result (the reproduction gate).

Run (from anywhere; the script chdirs into Recsys):
    conda run -n mechinterp python MechInterp/scripts/phase0_repro.py \
        --models freedom lgmrec lightgcn --datasets baby
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

RECSYS = Path("/workspace/Recsys")
MECHINTERP = Path("/workspace/MechInterp")
OUT_DIR = MECHINTERP / "results" / "phase0"
GATE_ABS = 5e-4  # ±0.0005 absolute on R@20/N@20 = PASS

# Treat Recsys as READ-ONLY source: import its code, load its checkpoints/data.
# RecDataset resolves data paths from its own module location (absolute), so we do
# NOT chdir into Recsys. Every output this script produces stays under MechInterp.
sys.path.insert(0, str(RECSYS))

import importlib.util                                        # noqa: E402

from src.utils import Config, set_seed                       # noqa: E402
from src.data.dataset import RecDataset                      # noqa: E402
from src.data.dataloader import EvalDataLoader               # noqa: E402
from src.data.graph_utils import build_norm_adj              # noqa: E402
from src.common.trainer import Trainer                       # noqa: E402

# Load only the model files we need DIRECTLY, bypassing src/models/__init__.py
# (which eagerly imports every model, incl. ones needing torch_geometric/transformers
# that we deliberately keep out of the mechinterp env). These 3 depend only on
# src.common / src.data (pure torch).
_MODEL_FILES = {"freedom": ("freedom.py", "FREEDOM"),
                "lgmrec": ("lgmrec.py", "LGMRec"),
                "lightgcn": ("lightgcn.py", "LightGCN"),
                "vbpr": ("vbpr.py", "VBPR"), "mmgcn": ("mmgcn.py", "MMGCN"),
                "lattice": ("lattice.py", "LATTICE"), "bm3": ("bm3.py", "BM3"),
                "mgcn": ("mgcn.py", "MGCN"), "mentor": ("mentor.py", "MENTOR"),
                "diffmm": ("diffmm.py", "DiffMM"),
                # 2024-25 models. Were MISSING here, so their noise floors silently
                # KeyError'd out of the screen-floor run (audit 2026-09-06).
                "smore": ("smore.py", "SMORE"), "gume": ("gume.py", "GUME"),
                "dragon": ("dragon.py", "DRAGON"), "damrs": ("damrs.py", "DAMRS"),
                "cohesion": ("cohesion.py", "COHESION"), "grcn": ("grcn.py", "GRCN")}


def load_recsys_model(model_name: str):
    fn, cls = _MODEL_FILES[model_name.lower()]
    path = RECSYS / "src" / "models" / fn
    spec = importlib.util.spec_from_file_location(f"_recsys_model_{model_name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, cls)

CKPT_RE = re.compile(r"^(?P<model>[a-z0-9]+)_(?P<ds>[a-z]+)_(?P<ts>\d{8}_\d{6})\.pt$")


def latest_ckpt(model: str, dataset: str) -> Path | None:
    cands = []
    for p in (RECSYS / "ckpts").iterdir():
        m = CKPT_RE.match(p.name)
        if m and m["model"] == model and m["ds"] == dataset:
            cands.append((m["ts"], p))
    if not cands:
        return None
    return sorted(cands)[-1][1]


def logged_test_result(ckpt: Path) -> dict | None:
    run_name = ckpt.stem
    rj = RECSYS / "logs" / run_name / "result.json"
    if rj.is_file():
        return json.loads(rj.read_text()).get("test_result")
    return None


def build_model(model_name: str, cfg, dataset, norm_adj, device):
    ModelCls = load_recsys_model(model_name)
    v_feat = (torch.from_numpy(dataset.v_feat[:].copy()) if dataset.v_feat is not None else None)
    t_feat = (torch.from_numpy(dataset.t_feat[:].copy()) if dataset.t_feat is not None else None)
    kwargs = {"config": cfg, "n_users": dataset.n_users, "n_items": dataset.n_items, "norm_adj": norm_adj}
    ml = model_name.lower()
    if ml not in ("lightgcn", "mllmrec", "falcon"):
        kwargs.update(v_feat=v_feat, t_feat=t_feat)
    # Models whose __init__ needs the raw interaction index. This list MUST stay in sync
    # across recsys_bridge.load_frozen, phase0_repro.build_model and
    # phasex_crossarch_knockout.build_eval -- it was previously short in two of the three,
    # making smore/gume/damrs/cohesion unloadable there (audit finding D3, 2026-09-04).
    if ml in ("freedom", "mllmrec", "histllm", "grcn", "dragon",
              "smore", "gume", "damrs", "cohesion"):
        kwargs.update(train_user_idx=torch.from_numpy(dataset.train_users),
                      train_item_idx=torch.from_numpy(dataset.train_items))
    return ModelCls(**kwargs).to(device)


def evaluate_ckpt(model_name: str, dataset_name: str, device: str) -> dict:
    ckpt = latest_ckpt(model_name, dataset_name)
    if ckpt is None:
        return {"model": model_name, "dataset": dataset_name, "error": "no checkpoint found"}

    # Route any incidental output dirs into MechInterp's workspace (eval-only never
    # writes them, but Trainer.__init__ ensure_dir's ckpt_dir/log_dir).
    scratch = OUT_DIR / "_scratch"
    cfg = Config(model_name, dataset_name, cli_overrides={
        "ckpt_dir": str(scratch / "ckpts"),
        "log_dir": str(scratch / "logs"),
    })
    set_seed(int(cfg.get("seed", 2024)), deterministic=bool(cfg.get("cudnn_deterministic", True)))

    dataset = RecDataset(cfg)
    norm_adj = build_norm_adj(dataset.train_matrix, dataset.n_users, dataset.n_items)
    test_loader = EvalDataLoader(dataset, phase="test",
                                 batch_size=int(cfg.get("eval_batch_size_users", 1024)))

    model = build_model(model_name, cfg, dataset, norm_adj, device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
    # persistent=False buffers (v_feat/t_feat/mm_adj/edge_*) are legitimately absent.
    benign = ("v_feat", "t_feat", "mm_adj", "edge_indices", "edge_values", "masked_adj",
              "R", "user_inv_deg")
    bad_missing = [k for k in missing if not k.endswith(benign)]

    trainer = Trainer(cfg, model, None, None, test_loader)
    recomputed = trainer.evaluate(test_loader)

    logged = logged_test_result(ckpt)
    rec = {
        "model": model_name,
        "dataset": dataset_name,
        "checkpoint": ckpt.name,
        "n_users": dataset.n_users,
        "n_items": dataset.n_items,
        "recomputed": {k: float(v) for k, v in recomputed.items()},
        "logged": logged,
        "missing_keys_unexpected": bad_missing,
        "unexpected_keys": list(unexpected),
    }
    if logged:
        deltas, gate = {}, "PASS"
        for key in ("Recall@20", "NDCG@20", "Recall@10", "NDCG@10"):
            if key in recomputed and key in logged:
                d = float(recomputed[key]) - float(logged[key])
                deltas[key] = d
                if key in ("Recall@20", "NDCG@20") and abs(d) > GATE_ABS:
                    gate = "INVESTIGATE" if abs(d) <= 0.05 * max(float(logged[key]), 1e-9) else "FAIL"
        rec["deltas_vs_logged"] = deltas
        rec["gate"] = gate
    else:
        rec["gate"] = "NO_LOG"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["freedom", "lgmrec", "lightgcn"])
    ap.add_argument("--datasets", nargs="+", default=["baby"])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for ds in args.datasets:
        for m in args.models:
            print(f"\n=== {m} / {ds} ===", flush=True)
            try:
                r = evaluate_ckpt(m, ds, device)
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                r = {"model": m, "dataset": ds, "error": repr(e)}
            results.append(r)
            if "recomputed" in r:
                rc, lg = r["recomputed"], (r.get("logged") or {})
                print(f"  recomputed R@20={rc.get('Recall@20'):.4f} N@20={rc.get('NDCG@20'):.4f}"
                      f" | logged R@20={lg.get('Recall@20')} N@20={lg.get('NDCG@20')}"
                      f" | gate={r.get('gate')}", flush=True)
            else:
                print(f"  {r.get('error')}", flush=True)

    # Merge into any existing artifact (keyed by model/dataset) so separate
    # invocations accumulate rather than overwrite.
    out = OUT_DIR / "repro_metrics.json"
    merged = {}
    if out.is_file():
        for r in json.loads(out.read_text()):
            merged[f"{r.get('model')}/{r.get('dataset')}"] = r
    for r in results:
        merged[f"{r.get('model')}/{r.get('dataset')}"] = r
    out.write_text(json.dumps(list(merged.values()), indent=2))
    print(f"\nWrote {out} ({len(merged)} records total)")
    # quick gate summary
    gates = {}
    for r in results:
        gates[r.get("gate", "ERROR")] = gates.get(r.get("gate", "ERROR"), 0) + 1
    print("Gate summary:", gates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
