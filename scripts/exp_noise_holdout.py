"""Prior-audit instruments on our converged FREEDOM retraining protocol.

Our retraining instrument (scripts/exp_modality_holdout.py) DROPS a modality: v_feat=None or
t_feat=None, so freedom.py:94-96 returns the surviving modality's normalized kNN graph at
weight 1, and that modality's auxiliary BPR term is skipped. Ye et al. (ECIR'26) instead
REPLACE the modality's feature file with Gaussian noise for both training and inference, which
keeps the architecture unchanged: a noise kNN graph at the original mixing weight plus an
auxiliary BPR term on (trainable) noise features. main.tex lists "noise versus removal" and
"graph reweighting" among the untested causes of the size gap. This script adds the conditions
that test those two causes, under exactly the converged protocol of the existing
full/no_image/no_text runs (stopping_step 100, epochs 3000, seeds 2024-2031), so that every new
run pairs by seed with results/phase_holdout/freedom_p100[_<ds>]_runs.json.

Conditions
  noise_image / noise_text  Ye et al.'s recipe (MKF4MMRec preprocessing-knockout-method.py::
      extract_noise_features, read from results/phase_novelty/_code/MKF4MMRec): a legacy
      numpy RandomState stream draws text = normal(1, 0.1, (n_items, 384)) FIRST and then
      image = normal(1, 0.1, (n_items, 4096)); float64, i.i.d. per item, generated once and
      fixed for the whole run (train and inference). The modality being knocked out gets its
      array; the other modality keeps its real features. Mixing weights unchanged (0.1 / 0.9).
      Deviation from Ye, as instructed: Ye's seed is a constant 42 for every run; here the
      noise seed is derived from the run seed (SeedSequence([run_seed, 42])), so each seed
      gets its own fixed noise draw. `--noise-seed-mode ye42` reproduces the constant-42
      stream instead (condition label gets suffix `_seed42`).
      Ambiguities (recorded in each run record): (a) the paper says only "Gaussian noise";
      mean 1 / sd 0.1 are from the code; (b) the code writes noise for BOTH modalities into
      one directory and does not show how the single-modality directories were assembled --
      we implement the literal reading (replace only the knocked-out file, keep the other
      real); (c) dimensions 4096/384 are hard-coded defaults in Ye's code; they equal our
      Amazon feature dims but NOT MicroLens (1024/1024). Default = Ye's literal 4096/384;
      `--noise-dims match` uses the real feature dims (label suffix `_matchdim`).
  no_image_keepw / no_text_keepw  Retrain without the modality (exactly as `no_image` /
      `no_text`) but scale the surviving graph by its ORIGINAL mixing weight
      (text: 1 - mm_image_weight = 0.9; image: mm_image_weight = 0.1) instead of 1.
      Not a no-op (see the no-op analysis below and results/phase_noise/aux_check.json).
  full  Same as exp_modality_holdout's `full`; only used by the smoke test to confirm this
      script reproduces the existing pipeline (the real full runs already exist).

No-op analysis for keepw (freedom.py): each modality graph is built by build_knn_graph as a
binary symmetrized kNN graph with D^-1/2 A D^-1/2 normalization; the two are mixed as
w*img + (1-w)*txt (lines 87-93) with NO renormalization after mixing; the single-modality
branch (lines 94-96) returns the normalized graph unscaled (weight 1). The mixed graph is used
numerically, not only structurally: _propagate computes h = mm_adj @ item_id_embedding
(n_mm_layers=1) and returns item_e + h, where item_e is the LightGCN mean over the SAME
item_id_embedding. Scaling mm_adj by c therefore scales h by c relative to item_e, and the
shared embedding table cannot absorb c. So the control is not a no-op.

Records -> results/phase_noise/runs_<dataset>.json (same fields as freedom_p100_*_runs.json plus
`instrument`, `feature_stats`, `mm_adj_stats`, `train_loss_curve`), appended under an fcntl lock,
resumable (finished (condition, seed) pairs are skipped; a live claim file prevents two
processes from training the same run). Checkpoints -> results/phase_noise/_scratch/ckpts/.
`--smoke` writes to results/phase_noise/_smoke/ instead and adds heavier diagnostics.
/workspace/Recsys stays READ-ONLY; exp_modality_holdout.py is imported, not edited.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/workspace/MechInterp")
sys.path.insert(0, str(ROOT / "scripts"))
import exp_modality_holdout as emh                                    # noqa: E402
from exp_modality_holdout import (Config, set_seed, RecDataset, EvalDataLoader,  # noqa: E402
                                  TrainDataLoader, build_norm_adj, Trainer,
                                  load_recsys_model)
from src.data.graph_utils import build_knn_graph                      # noqa: E402

OUT = ROOT / "results" / "phase_noise"
BASE_CONDITIONS = ("full", "noise_image", "noise_text", "no_image_keepw", "no_text_keepw")
DEFAULT_CONDITIONS = ("noise_image", "noise_text", "no_image_keepw", "no_text_keepw")
YE_MEAN, YE_SD, YE_TEXT_DIM, YE_IMAGE_DIM, YE_SEED = 1.0, 0.1, 384, 4096, 42
REF_FILES = {"baby": "freedom_p100_runs.json", "sports": "freedom_p100_sports_runs.json",
             "clothing": "freedom_p100_clothing_runs.json",
             "microlens": "freedom_p100_microlens_runs.json"}

YE_RECIPE = {
    "source": ("GAIR-Lab/MKF4MMRec @812e94e preprocessing/preprocessing-knockout-method.py::"
               "extract_noise_features (local copy results/phase_novelty/_code/MKF4MMRec)"),
    "generator": "numpy legacy RandomState (MT19937), np.random.normal",
    "distribution": f"Normal(mean={YE_MEAN}, sd={YE_SD}) i.i.d. per item and dimension",
    "draw_order": f"text ({YE_TEXT_DIM}-d) drawn first, then image ({YE_IMAGE_DIM}-d), same stream",
    "dtype": "float64 as generated; cast to float32 by MultimodalRecommender (.float())",
    "fixed": "generated once per run, used unchanged for training and inference; never resampled",
    "where_it_enters": ("replaces the modality's feature tensor passed to FREEDOM: its kNN graph "
                        "(kept at the original mixing weight) and the trainable "
                        "nn.Embedding.from_pretrained init of its auxiliary BPR branch"),
    "ambiguities": [
        "Paper text says only 'Gaussian noise'; mean 1 / sd 0.1 come from the released code.",
        ("Ye's seed is the constant 42 for every run/model; per the task this run derives the noise "
         "seed from the run seed unless --noise-seed-mode ye42."),
        ("Released code writes noise for BOTH modalities into one directory; how single-modality "
         "knockout directories were assembled is not in the code. Implemented literally: only the "
         "knocked-out modality is replaced, the other keeps its real features."),
        ("Dims 4096/384 are hard-coded defaults of Ye's function; they match our Amazon features but "
         "not MicroLens (real 1024/1024). Default keeps Ye's literal dims unless --noise-dims match."),
        ("Ye's knockout script (default --step all) also re-runs 5-core filtering and an unseeded "
         "split into each knockout directory; we deliberately keep OUR fixed split so runs pair "
         "by seed with the existing full run (not replicated)."),
        ("Graph builder differs from Ye's MMRec FREEDOM (MKF4MMRec src/models/freedom.py:79-99): "
         "theirs is a DIRECTED top-k incl. the item itself, D^-1/2 A D^-1/2 with row degrees = k, "
         "i.e. every row sums to 1 and a hub cannot inflate any row; ours "
         "(Recsys graph_utils.build_knn_graph) excludes self, SYMMETRIZES max(A, A^T) and normalizes "
         "by the symmetrized degree, so a noise hub item gets a high-degree row. We keep our "
         "harness's builder for every condition (so conditions pair), which means the noise graph "
         "is recipe-faithful but not graph-builder-faithful to Ye."),
        ("MMRec FREEDOM caches mm_adj_freedomdsp_{k}_{w}.pt in the dataset dir and reloads it if "
         "present; our harness rebuilds the graph from the passed features every run (no cache)."),
    ],
}


# ----------------------------------------------------------------------------- helpers

def condition_label(base: str, noise_seed_mode: str, noise_dims: str) -> str:
    lab = base
    if base.startswith("noise_"):
        if noise_seed_mode == "ye42":
            lab += "_seed42"
        if noise_dims == "match":
            lab += "_matchdim"
    return lab


def derive_noise_seed(run_seed: int, mode: str) -> int:
    if mode == "ye42":
        return YE_SEED
    return int(np.random.SeedSequence([int(run_seed), YE_SEED]).generate_state(1)[0])


def ye_noise(n_items: int, noise_seed: int, text_dim: int, image_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Literal extract_noise_features: seed a legacy stream, draw text first, then image."""
    rs = np.random.RandomState(noise_seed)
    text = rs.normal(YE_MEAN, YE_SD, (n_items, text_dim))
    image = rs.normal(YE_MEAN, YE_SD, (n_items, image_dim))
    return text, image


def feat_stats(x: torch.Tensor | None, real: torch.Tensor | None = None, n_pairs: int = 20000,
               seed: int = 0) -> dict | None:
    if x is None:
        return None
    x = x.detach().float().cpu()
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    i = torch.randint(0, n, (n_pairs,), generator=g)
    j = torch.randint(0, n, (n_pairs,), generator=g)
    keep = i != j
    xn = torch.nn.functional.normalize(x, dim=-1)
    out = {"shape": list(x.shape), "mean": float(x.mean()), "sd": float(x.std()),
           "min": float(x.min()), "max": float(x.max()),
           "mean_pairwise_cos_sampled": float((xn[i[keep]] * xn[j[keep]]).sum(-1).mean()),
           "sum_fingerprint": float(x.double().sum())}
    if real is not None and real.shape == x.shape:
        out["identical_to_real_features"] = bool(torch.equal(x, real.detach().float().cpu()))
    return out


def adj_stats(a: torch.Tensor) -> dict:
    a = a.coalesce().cpu()
    v = a.values()
    rs = torch.zeros(a.shape[0]).index_add_(0, a.indices()[0], v)
    return {"nnz": int(v.numel()), "value_sum": float(v.double().sum()),
            "row_sum_mean": float(rs.mean()), "row_sum_max": float(rs.max())}


def edge_set(a: torch.Tensor) -> set:
    idx = a.coalesce().indices().cpu().numpy()
    return set(zip(idx[0].tolist(), idx[1].tolist()))


def graph_diagnostics(feat: torch.Tensor, real_img: torch.Tensor | None, real_txt: torch.Tensor | None,
                      k: int) -> dict:
    """Structure of the kNN graph built from `feat` vs the real image / text graphs (smoke only)."""
    g = build_knn_graph(feat, k).coalesce()
    idx = g.indices()
    indeg = torch.bincount(idx[1], minlength=g.shape[0]).double().numpy()
    s = np.sort(indeg)
    n = len(s)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(s) / (n * s.sum())) if s.sum() else 0.0
    es = edge_set(g)
    out = {"n_edges_sym": len(es), "max_indegree_sym": int(indeg.max()), "gini_indegree_sym": gini}
    for name, ref in (("real_image", real_img), ("real_text", real_txt)):
        if ref is not None:
            er = edge_set(build_knn_graph(ref, k))
            out[f"edge_jaccard_with_{name}"] = len(es & er) / max(1, len(es | er))
    return out


class RunsFile:
    """JSON list on disk; every read-modify-write happens under an exclusive fcntl lock."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = path.with_suffix(path.suffix + ".lock")

    def _read(self) -> list:
        return json.loads(self.path.read_text()) if self.path.exists() else []

    def read(self) -> list:
        with open(self.lock, "a") as lf:
            fcntl.flock(lf, fcntl.LOCK_SH)
            try:
                return self._read()
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def append(self, rec: dict) -> list:
        with open(self.lock, "a") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                runs = self._read()
                runs.append(rec)
                tmp = self.path.with_suffix(f".tmp{os.getpid()}")
                tmp.write_text(json.dumps(runs, indent=2))
                os.replace(tmp, self.path)
                return runs
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)


def done_keys(runs: list) -> set:
    return {(r["model"], r["dataset"], r["condition"], r["seed"], r.get("stopping_step"), r.get("epochs_cap"))
            for r in runs if "test_result" in r}


def try_claim(claim_dir: Path, key: str) -> bool:
    """Atomic claim file so two launches never train the same run; stale claims (dead pid) are taken over."""
    claim_dir.mkdir(parents=True, exist_ok=True)
    p = claim_dir / f"{key}.claim"
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            pid = int(p.read_text().strip() or "0")
            os.kill(pid, 0)
            return False                      # owner alive
        except (ProcessLookupError, ValueError):
            p.write_text(str(os.getpid()))    # stale claim
            return True
        except PermissionError:
            return False


def release_claim(claim_dir: Path, key: str) -> None:
    p = claim_dir / f"{key}.claim"
    try:
        if p.read_text().strip() == str(os.getpid()):
            p.unlink()
    except FileNotFoundError:
        pass


# ----------------------------------------------------------------------------- training

def train_condition(model_name: str, dataset_name: str, seed: int, base_cond: str, device: str,
                    patience: int, epochs: int, out_dir: Path, noise_seed_mode: str = "derived",
                    noise_dims: str = "ye", diagnostics: bool = False) -> dict:
    """Mirror of exp_modality_holdout.train_holdout (same config overrides, seeding order, data
    loaders and model kwargs) with the prior-audit feature/graph manipulations inserted."""
    assert base_cond in BASE_CONDITIONS, base_cond
    assert model_name.lower() == "freedom", "keepw/noise semantics are defined for FREEDOM only"
    label = condition_label(base_cond, noise_seed_mode, noise_dims)
    scratch = out_dir / "_scratch"
    ov = {"seed": seed, "ckpt_dir": str(scratch / "ckpts"), "log_dir": str(scratch / "logs"),
          "show_progress": False, "stopping_step": patience, "epochs": epochs}
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

    v_real = torch.from_numpy(dataset.v_feat[:].copy()) if dataset.v_feat is not None else None
    t_real = torch.from_numpy(dataset.t_feat[:].copy()) if dataset.t_feat is not None else None
    v, t = v_real, t_real
    w_img = float(cfg.get("mm_image_weight", 0.1))
    instrument: dict = {"base_condition": base_cond}
    adj_scale = None

    if base_cond in ("noise_image", "noise_text"):
        # Separate RandomState: the global numpy/torch streams seeded by set_seed are not
        # consumed, so negative sampling and init stay identical to the paired full run
        # (except init of the noised branch's Linear if its input dim differs from the real one).
        noise_seed = derive_noise_seed(seed, noise_seed_mode)
        td = YE_TEXT_DIM if noise_dims == "ye" else int(t_real.shape[1])
        idim = YE_IMAGE_DIM if noise_dims == "ye" else int(v_real.shape[1])
        n_text, n_image = ye_noise(dataset.n_items, noise_seed, td, idim)
        if base_cond == "noise_image":
            v = torch.from_numpy(n_image)
        else:
            t = torch.from_numpy(n_text)
        instrument.update({
            "kind": "noise_replacement (Ye et al. ECIR'26 recipe)", "recipe": YE_RECIPE,
            "noise_seed_mode": noise_seed_mode, "noise_seed": noise_seed,
            "noise_seed_derivation": ("constant 42 (Ye literal)" if noise_seed_mode == "ye42" else
                                      "np.random.SeedSequence([run_seed, 42]).generate_state(1)[0]"),
            "noise_dims_mode": noise_dims, "noise_text_dim": td, "noise_image_dim": idim,
            "real_image_dim": int(v_real.shape[1]), "real_text_dim": int(t_real.shape[1]),
            "noised_dim_differs_from_real": (idim != v_real.shape[1]) if base_cond == "noise_image"
            else (td != t_real.shape[1]),
            "mixing_weights_unchanged": {"image": w_img, "text": 1.0 - w_img},
        })
    elif base_cond == "no_image_keepw":
        v, adj_scale = None, 1.0 - w_img
    elif base_cond == "no_text_keepw":
        t, adj_scale = None, w_img
    if adj_scale is not None:
        instrument.update({
            "kind": "removal with original mixing weight (graph-reweighting control)",
            "surviving_graph_scale": adj_scale,
            "note": ("Model built exactly as no_image/no_text (modality absent from constructor, "
                     "no auxiliary loss); its single-modality mm_adj (weight 1 in freedom.py:94-96) "
                     "is then multiplied by the weight it carries in the full mix (freedom.py:88-91)."),
        })
    assert v is not None or t is not None

    ModelCls = load_recsys_model(model_name)
    kwargs = {"config": cfg, "n_users": dataset.n_users, "n_items": dataset.n_items,
              "norm_adj": norm_adj, "v_feat": v, "t_feat": t,
              "train_user_idx": torch.from_numpy(np.asarray(dataset.train_users)),
              "train_item_idx": torch.from_numpy(np.asarray(dataset.train_items))}
    model = ModelCls(**kwargs)
    adj_before = adj_stats(model.mm_adj)
    if adj_scale is not None:
        model.mm_adj = (model.mm_adj * adj_scale).coalesce()     # still a registered buffer
        assert "mm_adj" in dict(model.named_buffers()), "mm_adj must remain a buffer"
    mm_stats = {"before_scale": adj_before, "final": adj_stats(model.mm_adj), "scale": adj_scale}
    model = model.to(device)

    checks = {"has_v_feat": getattr(model, "v_feat", None) is not None,
              "has_t_feat": getattr(model, "t_feat", None) is not None,
              "has_image_trs": hasattr(model, "image_trs"),
              "has_text_trs": hasattr(model, "text_trs")}
    if base_cond == "no_image_keepw":
        assert not checks["has_v_feat"] and not checks["has_image_trs"], checks
    if base_cond == "no_text_keepw":
        assert not checks["has_t_feat"] and not checks["has_text_trs"], checks
    if base_cond.startswith("noise_") or base_cond == "full":
        assert all(checks.values()), checks

    fstats = {
        "model_v_feat": feat_stats(model.v_feat, v_real),
        "model_t_feat": feat_stats(model.t_feat, t_real),
        "model_image_embedding_init": feat_stats(model.image_embedding.weight) if checks["has_image_trs"] else None,
        "model_text_embedding_init": feat_stats(model.text_embedding.weight) if checks["has_text_trs"] else None,
    }
    if diagnostics:
        k = int(cfg.get("knn_k", 10))
        fstats["graph_diagnostics"] = {}
        if base_cond == "noise_image":
            fstats["graph_diagnostics"]["noise_image_graph"] = graph_diagnostics(v, v_real, t_real, k)
        if base_cond == "noise_text":
            fstats["graph_diagnostics"]["noise_text_graph"] = graph_diagnostics(t, v_real, t_real, k)
        # edge-set overlap of the model's actual mm_adj with the real mixed graph pieces
        es = edge_set(model.mm_adj)
        er_i, er_t = edge_set(build_knn_graph(v_real, k)), edge_set(build_knn_graph(t_real, k))
        fstats["graph_diagnostics"]["model_mm_adj_edges"] = {
            "n": len(es), "frac_in_real_image_graph": len(es & er_i) / max(1, len(es)),
            "frac_in_real_text_graph": len(es & er_t) / max(1, len(es))}

    tag = f"_p{patience}e{epochs}"
    run = f"noise{tag}_{model_name}_{dataset_name}_{label}_s{seed}"
    trainer = Trainer(cfg, model, train_loader, valid_loader, test_loader, run_name=run)
    curve, losses = [], []
    _orig_valid, _orig_train = trainer._valid_epoch, trainer._train_epoch

    def _recording_valid():
        res = _orig_valid()
        curve.append([len(curve), float(res.get(cfg.get("valid_metric", "Recall@20"), float("nan")))])
        return res

    def _recording_train(epoch):
        loss = _orig_train(epoch)
        losses.append([int(epoch), float(loss)])
        return loss

    trainer._valid_epoch = _recording_valid
    trainer._train_epoch = _recording_train
    t0 = time.time()
    result = trainer.fit()
    ck = scratch / "ckpts" / f"{run}.pt"
    return {"model": model_name, "dataset": dataset_name, "seed": seed,
            "condition": label, "best_epoch": int(result["best_epoch"]),
            "train_min": (time.time() - t0) / 60.0,
            "ckpt_path": str(ck) if ck.is_file() else None,
            "modality_presence": checks,
            "stopping_step": int(cfg.get("stopping_step", -1)), "epochs_cap": int(cfg.get("epochs", -1)),
            "valid_curve": curve,
            "test_result": {k_: float(v_) for k_, v_ in result["test_result"].items()},
            "instrument": instrument, "feature_stats": fstats, "mm_adj_stats": mm_stats,
            "train_loss_curve": losses, "pairs_with": f"results/phase_holdout/{REF_FILES.get(dataset_name)}",
            "script": "scripts/exp_noise_holdout.py", "created": time.strftime("%Y-%m-%d %H:%M:%S")}


# ----------------------------------------------------------------------------- summary

def summarize(ds: str, new_runs: list) -> dict:
    """Paired-by-seed contrasts vs the existing converged full run (read-only reference), plus
    noise-vs-removal and keepw-vs-removal contrasts against the existing no_image/no_text runs."""
    ref_path = ROOT / "results" / "phase_holdout" / REF_FILES[ds]
    ref = [r for r in (json.loads(ref_path.read_text()) if ref_path.exists() else [])
           if "test_result" in r and r.get("stopping_step") == 100 and r.get("epochs_cap") == 3000]
    by: dict = {}
    for r in ref:
        by.setdefault(r["condition"], {})[r["seed"]] = r["test_result"]
    for r in new_runs:
        if "test_result" in r and r["condition"] != "full":
            by.setdefault(r["condition"], {})[r["seed"]] = r["test_result"]
    pairs = [(c, "full") for c in by if c != "full"]
    for c in by:
        if c.startswith("noise_image"):
            pairs.append((c, "no_image"))
        if c.startswith("noise_text"):
            pairs.append((c, "no_text"))
    if "no_image_keepw" in by:
        pairs.append(("no_image_keepw", "no_image"))
    if "no_text_keepw" in by:
        pairs.append(("no_text_keepw", "no_text"))
    out = {"reference_file": str(ref_path.relative_to(ROOT)), "contrasts": {}}
    for a, b in pairs:
        if b not in by:
            continue
        row = {}
        for metric in ("Recall@20", "NDCG@20", "Recall@10", "NDCG@10"):
            shared = sorted(s for s in set(by[a]) & set(by[b]) if metric in by[a][s] and metric in by[b][s])
            if not shared:
                continue
            d = [by[a][s][metric] - by[b][s][metric] for s in shared]
            bm = statistics.mean(by[b][s][metric] for s in shared)
            e = {"seeds": shared, "n": len(d), "mean_a": statistics.mean(by[a][s][metric] for s in shared),
                 "mean_b": bm, "delta_mean": statistics.mean(d),
                 "delta_rel_pct": 100 * statistics.mean(d) / bm if bm else None}
            if len(d) >= 2:
                sd = statistics.stdev(d)
                e.update({"delta_sd": sd, "F_paired_2sd": 2 * sd,
                          "paired_t": statistics.mean(d) / (sd / len(d) ** 0.5) if sd else None,
                          "df": len(d) - 1})
            row[metric] = e
        out["contrasts"][f"{a} - {b}"] = row
    return out


# ----------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["baby"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(2024, 2032)))
    ap.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS), choices=BASE_CONDITIONS)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--noise-seed-mode", choices=("derived", "ye42"), default="derived")
    ap.add_argument("--noise-dims", choices=("ye", "match"), default="ye")
    ap.add_argument("--smoke", action="store_true",
                    help="write to results/phase_noise/_smoke/ (never runs_*.json) and add graph diagnostics")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    out_dir = OUT / "_smoke" if args.smoke else OUT
    if not args.smoke and (args.patience != 100 or args.epochs != 3000):
        raise SystemExit("non-smoke runs must use the converged protocol (patience 100, epochs 3000) "
                         "so they pair with results/phase_holdout/freedom_p100_*; use --smoke for short runs")
    if not args.smoke and "full" in args.conditions:
        raise SystemExit("`full` already exists in results/phase_holdout; it is a smoke-only condition here")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for ds in args.datasets:
        rf = RunsFile(out_dir / f"runs_{ds}.json")
        for base in args.conditions:
            label = condition_label(base, args.noise_seed_mode, args.noise_dims)
            for s in args.seeds:
                key = ("freedom", ds, label, s, args.patience, args.epochs)
                if key in done_keys(rf.read()):
                    print(f"skip {ds}/{label}/s{s} (done)", flush=True)
                    continue
                ckey = f"{ds}_{label}_s{s}_p{args.patience}e{args.epochs}"
                if not try_claim(out_dir / "_scratch" / "claims", ckey):
                    print(f"skip {ds}/{label}/s{s} (claimed by a live process)", flush=True)
                    continue
                try:
                    if key in done_keys(rf.read()):   # finished between check and claim
                        continue
                    print(f"\n=== freedom/{ds} {label} seed={s} | elapsed {(time.time()-t0)/60:.1f}m ===",
                          flush=True)
                    try:
                        r = train_condition("freedom", ds, s, base, device, args.patience, args.epochs,
                                            out_dir, args.noise_seed_mode, args.noise_dims,
                                            diagnostics=args.smoke)
                        print(f"  R@20={r['test_result']['Recall@20']:.5f} "
                              f"N@20={r['test_result']['NDCG@20']:.5f} "
                              f"(best epoch {r['best_epoch']}, {r['train_min']:.1f} min)", flush=True)
                    except Exception as e:  # noqa: BLE001
                        import traceback; traceback.print_exc()
                        r = {"model": "freedom", "dataset": ds, "condition": label, "seed": s,
                             "error": repr(e)}
                    runs = rf.append(r)
                finally:
                    release_claim(out_dir / "_scratch" / "claims", ckey)
                if not args.smoke and ds in REF_FILES:
                    summ = summarize(ds, runs)
                    tmp = out_dir / f".summary_{ds}.tmp{os.getpid()}"
                    tmp.write_text(json.dumps(summ, indent=2))
                    os.replace(tmp, out_dir / f"summary_{ds}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
