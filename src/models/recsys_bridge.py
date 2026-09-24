"""Bridge to the (read-only) Recsys repo: load frozen recommenders + data.

Imports Recsys as a library WITHOUT triggering its models/__init__.py (which pulls
torch_geometric/transformers we keep out of the mechinterp env): we load the 3
model files we need directly. All outputs of any caller stay under MechInterp.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import torch

RECSYS = Path("/workspace/Recsys")
SCRATCH = Path("/workspace/MechInterp/results/_scratch")
if str(RECSYS) not in sys.path:
    sys.path.insert(0, str(RECSYS))

from src.utils import Config, set_seed                       # noqa: E402
from src.data.dataset import RecDataset                      # noqa: E402
from src.data.dataloader import EvalDataLoader               # noqa: E402
from src.data.graph_utils import build_norm_adj              # noqa: E402

_MODEL_FILES = {"freedom": ("freedom.py", "FREEDOM"),
                "lgmrec": ("lgmrec.py", "LGMRec"),
                "lightgcn": ("lightgcn.py", "LightGCN"),
                # additional PUBLISHED multimodal recommenders (cross-architecture track)
                "vbpr": ("vbpr.py", "VBPR"),
                "mmgcn": ("mmgcn.py", "MMGCN"),
                "lattice": ("lattice.py", "LATTICE"),
                "bm3": ("bm3.py", "BM3"),
                "mgcn": ("mgcn.py", "MGCN"),
                "mentor": ("mentor.py", "MENTOR"),
                "diffmm": ("diffmm.py", "DiffMM"),
                # recent multimodal recommenders (2024-2025)
                "smore": ("smore.py", "SMORE"),
                "gume": ("gume.py", "GUME"),
                "dragon": ("dragon.py", "DRAGON"),
                "damrs": ("damrs.py", "DAMRS"),
                "cohesion": ("cohesion.py", "COHESION")}
_CKPT_RE = re.compile(r"^(?P<model>[a-z0-9]+)_(?P<ds>[a-z]+)_(?P<ts>\d{8}_\d{6})\.pt$")


def load_recsys_model_class(model_name: str):
    fn, cls = _MODEL_FILES[model_name.lower()]
    spec = importlib.util.spec_from_file_location(f"_recsys_model_{model_name}", RECSYS / "src" / "models" / fn)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, cls)


def latest_ckpt(model: str, dataset: str) -> Path | None:
    cands = []
    for p in (RECSYS / "ckpts").iterdir():
        m = _CKPT_RE.match(p.name)
        if m and m["model"] == model and m["ds"] == dataset:
            cands.append((m["ts"], p))
    return sorted(cands)[-1][1] if cands else None


def load_frozen(model_name: str, dataset_name: str, device: str, load_ckpt: bool = True,
                ckpt_path: str | Path | None = None):
    """Returns (cfg, dataset, model, test_loader); model is eval() and on device.

    ckpt_path pins an EXPLICIT checkpoint. Without it we fall back to latest_ckpt(),
    which resolves by max timestamp and can therefore silently load an ABORTED run
    (verified on MicroLens/LGMRec, 2026-09-04). Pin whenever the dataset has more
    than one run on disk."""
    cfg = Config(model_name, dataset_name, cli_overrides={
        "ckpt_dir": str(SCRATCH / "ckpts"), "log_dir": str(SCRATCH / "logs")})
    set_seed(int(cfg.get("seed", 2024)), deterministic=bool(cfg.get("cudnn_deterministic", True)))

    dataset = RecDataset(cfg)
    norm_adj = build_norm_adj(dataset.train_matrix, dataset.n_users, dataset.n_items)
    test_loader = EvalDataLoader(dataset, phase="test",
                                 batch_size=int(cfg.get("eval_batch_size_users", 1024)))

    ModelCls = load_recsys_model_class(model_name)
    v_feat = torch.from_numpy(dataset.v_feat[:].copy()) if dataset.v_feat is not None else None
    t_feat = torch.from_numpy(dataset.t_feat[:].copy()) if dataset.t_feat is not None else None
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
        kwargs.update(train_user_idx=torch.from_numpy(np.asarray(dataset.train_users)),
                      train_item_idx=torch.from_numpy(np.asarray(dataset.train_items)))
    model = ModelCls(**kwargs).to(device)

    if load_ckpt:
        if ckpt_path is not None:
            ck = Path(ckpt_path)
            if not ck.is_file():
                raise FileNotFoundError(f"pinned checkpoint missing: {ck}")
        else:
            ck = latest_ckpt(model_name, dataset_name)
        if ck is None:
            raise FileNotFoundError(f"no checkpoint for {model_name}/{dataset_name}")
        state = torch.load(ck, map_location=device, weights_only=False)
        # strict=False is needed (frozen non-persistent buffers such as FREEDOM's mm_adj are
        # rebuilt, not stored), but an UNCHECKED strict=False silently leaves a renamed
        # parameter at random init and still returns a plausible number. Assert that every
        # missing key is a rebuildable buffer, never a learnable parameter.
        inc = model.load_state_dict(state["model_state_dict"], strict=False)
        param_names = {n for n, _ in model.named_parameters()}
        bad = [k for k in inc.missing_keys if k in param_names]
        if bad:
            raise RuntimeError(f"checkpoint {ck.name} is missing LEARNABLE parameters {bad} "
                               f"-- they would stay at random init. Refusing to proceed.")
        model._missing_keys = list(inc.missing_keys)
        model._unexpected_keys = list(inc.unexpected_keys)
        model._ckpt_name = ck.name
        model._ckpt_path = str(ck)
        model._ckpt_pinned = ckpt_path is not None
    model.eval()
    return cfg, dataset, model, test_loader
