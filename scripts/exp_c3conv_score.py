"""Converged multi-seed deletion scorer for the C3 re-test (results/phase_c3conv/PREREG_C3CONV.md).

Scores every FINISHED converged run (patience 100 / cap 3000) of the ten non-FREEDOM
architectures with the SAME per-architecture deletion instrument as the default-patience
screen (scripts/exp_exact_crossarch.py + scripts/exact_ko/*; LGMRec through
phase1_knockout.lgmrec_variants, exactly as exp_exact_crossarch routes it), default arms only
(`image_knockout`, `text_knockout`, plus `both_knockout` recorded as an extra), and writes one
row per (model, seed) to results/phase_c3conv/scored_<ds>.json.

Nothing here trains. Nothing outside /workspace/MechInterp is written. The training
campaign's files (runs_<ds>.json, pat100e3000_*_s2025..s2031.pt) are only READ, and a seed is
scored only once its run record exists (the trainer appends the record after `fit()` returns,
so a checkpoint without a record may still be in training).

STALENESS / IDENTITY GUARD (every row, before anything is written as scored):
  G1  checkpoint identity: the checkpoint's own `epoch` equals the run record's `best_epoch`,
      and its `config_snapshot` carries the record's seed, stopping_step 100 and epochs 3000.
  G2  baseline match: the deletion baseline Recall@20 equals the trainer-logged test Recall@20
      to 1e-9 (the tolerance of exp_converged_knockout.py).
      For the two architectures whose OWN inference path is stochastic, G2 cannot hold
      exactly by construction, so they get G2' instead (a DEVIATION from the literal
      registration, flagged per row as guard_mode="distributional"):
        LGMRec   full_sort_predict draws fresh gumbel noise on every call (every user batch);
        COHESION the user-user graph is re-sampled with np.random in the train-loop hook
                 pre_epoch_processing, and the logged test uses whatever draw the last epoch left.
      G2' re-evaluates the model's own baseline under K independent draws and requires the
      logged R@20 to lie within Z_TOL sds of the draw mean. The deletion arms still use the one
      PAIRED draw of the instrument (GUMBEL_SEED / USER_GRAPH_SEED), as in the screen.
      LATTICE: the logged test read train-loop state absent from the checkpoint (see
      TRAINLOOP_STATE), so only G1 applies (guard_mode="identity_only"; also a DEVIATION).
  A row failing any guard is written to `refused` with the reason and never enters a verdict.

  --verdicts  computes per (model, dataset, modality) cell, on Recall@20 (PREREG_C3CONV.md as
      amended by PREREG_C3CONV_AMEND1.md):
      F_level = 2 sd(trainer-logged full-model R@20 across seeds) (A4), F_paired = 2 sd(per-seed
      delta)  (ddof = 1); SIGNIFICANT iff |mean d| > max(F_level, F_paired); BELOW iff below both;
      MARGINAL otherwise; two-sided one-sample t on the deltas (A2 for zero spread), n negative,
      95% t-CI; Benjamini-Hochberg (q=0.05) and Holm (alpha=0.05) over the PRIMARY family = all
      judged image AND text cells together (A1), plus a SECONDARY image-only family and an A3
      sensitivity family restricted to exact-guard rows' models. A cell is judged iff it has >= 5
      seeds AND its dataset is admitted (every model of the dataset's registered set has >= 5
      scored seeds; MicroLens's set per A6). Interpretation flags per cell (not verdicts):
      deletion_raises_accuracy, joint_term_not_identified (SMORE), branch_not_content (COHESION),
      guard_modes.
  -> results/phase_c3conv/verdicts.json
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/workspace/MechInterp")
RECSYS = Path("/workspace/Recsys")
for _p in (str(RECSYS), str(ROOT / "src" / "models"), str(ROOT / "src" / "interp"),
           str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

C3 = ROOT / "results" / "phase_c3conv"
TABLE_RERUN = ROOT / "results" / "phase_convergence" / "table_rerun.json"
MODELS = ["vbpr", "damrs", "mmgcn", "lattice", "mgcn", "mentor", "gume", "smore",
          "cohesion", "lgmrec", "damrs_g11"]
# PREREG_C3CONV_AMEND2.md: the DAMRS port trained with an EMPTY user-item graph (dict.update(A, ...) on a
# dok_matrix leaves it empty in the installed scipy, 1.17.1) until the upstream repair of 2026-09-25 20:40:12 -0700.
# "damrs_g11" is the scoring name of the same architecture re-trained on the repaired port; its runs
# come only from results/phase_c3conv_g11/ and must carry the registered model-file hash (B4). The
# defective-port "damrs" rows stay in the files but are never judged (B5/B7).
G11 = ROOT / "results" / "phase_c3conv_g11"
DAMRS_FILE = RECSYS / "src" / "models" / "damrs.py"
DAMRS_G11_SHA256 = "d001139f3cbafff1eac8e3a1c82019cbc96deb4fec7db56ae01bc51731a4f5ed"
DAMRS_G11_ACCEPTED = (DAMRS_G11_SHA256, "b0bd3670c78d3f90a2deec378f0182880876a9fdbb0994f447bb76a7ce690e10")  # + this release's scrubbed copy   # a code release may add the hash of its scrubbed copy
ARCH = {"damrs_g11": "damrs"}       # scoring name -> architecture (load_frozen, exact_ko module)
LEGACY_VOID = {"damrs": "defective DAMRS port (empty user-item graph); PREREG_C3CONV_AMEND2.md B5/B7"}
REGISTERED = ["damrs_g11" if m == "damrs" else m for m in MODELS if m != "damrs_g11"]
DATASETS = ["baby", "sports", "clothing", "microlens"]
ARMS = {"image": "image_knockout", "text": "text_knockout", "both": "both_knockout"}
JUDGED_STREAMS = ("image", "text")
METRICS = ("Recall@10", "Recall@20", "NDCG@10", "NDCG@20")
BASE_TOL = 1e-9          # exp_converged_knockout.py
STOCHASTIC = {"lgmrec": "gumbel noise redrawn in every full_sort_predict call",
              "cohesion": "user-user graph re-sampled (np.random) in pre_epoch_processing"}
# Train-loop state that is NOT in the checkpoint but IS read by the logged test evaluation.
# LATTICE caches `_item_adj` (a plain attribute) built at the START of the last training epoch
# from that epoch's `modal_weight`; the trainer then loads the best-epoch weights and tests with
# that stale graph, whereas any reload rebuilds the graph from the best-epoch weights (F6 in
# exact_ko/lattice.py). The logged R@20 is therefore not reproducible from the checkpoint.
# Measured on baby/s2024: rebuilt 0.087354602 vs logged 0.087232891; shifting modal_weight by
# |dlogit| <= 0.005 moves the rebuilt R@20 over [0.087192, 0.087740], bracketing the logged value.
# These rows get guard_mode="identity_only" (G1 only, diff recorded) -- a flagged DEVIATION.
TRAINLOOP_STATE = {"lattice": "cached _item_adj from the last training epoch is not in the checkpoint"}
K_DRAWS = 10
Z_TOL = 4.0
RECON_REL_TOL = 1e-6      # PREREG_C3CONV_AMEND1.md A5
MIN_SEEDS = 5
# PREREG_C3CONV_AMEND1.md A6: MicroLens covers eight models (MENTOR, MGCN not run: 10.0 h / 8.4 h per run)
ADMIT_MODELS = {"microlens": [m for m in REGISTERED if m not in ("mentor", "mgcn")]}
Q_BH = 0.05
ALPHA_HOLM = 0.05
FULL_HASH = False
RUN_DEVICE: dict = {}
GPU_CAP_GB = 3.3          # hard per-process allocator cap (CUDA context adds ~0.5 GB; GPU shared with training)


# ============================================================================ run records
def _read_json(path: Path, retries: int = 5):
    for k in range(retries):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if k == retries - 1:
                raise
            time.sleep(0.5)


def _is_converged(r: dict) -> bool:
    ov = r.get("overrides") or {}
    return ("test_result" in r and ov.get("stopping_step") == 100 and ov.get("epochs") == 3000
            and r.get("epochs_cap") == 3000 and r.get("patience", 100) == 100)


def collect_runs(ds: str) -> tuple[dict, list]:
    """(model, seed) -> run record, over table_rerun.json (seed 2024) + runs_<ds>.json.
    Returns (records, conflicts). A key present twice with different R@20 is a conflict and
    is excluded rather than guessed."""
    srcs = []
    if TABLE_RERUN.is_file():
        srcs += [("table_rerun.json", r) for r in _read_json(TABLE_RERUN)]
    rf = C3 / f"runs_{ds}.json"
    if rf.is_file():
        srcs += [(rf.name, r) for r in _read_json(rf)]
    out, conflicts = {}, []
    for src, r in srcs:
        if r.get("dataset") != ds or r.get("model") not in MODELS or not _is_converged(r):
            continue
        if r.get("model") in ARCH:      # AMEND2 B4: only results/phase_c3conv_g11/ may supply these
            continue
        k = (r["model"], int(r["seed"]))
        if k in out:
            a, b = out[k]["test_result"]["Recall@20"], r["test_result"]["Recall@20"]
            if abs(a - b) > 1e-12:
                conflicts.append({"model": k[0], "seed": k[1], "R@20": [a, b],
                                  "sources": [out[k]["_src"], src]})
            continue
        out[k] = dict(r, _src=src)
    for c in conflicts:
        out.pop((c["model"], c["seed"]), None)
    # PREREG_C3CONV_AMEND2.md B4: repaired-port DAMRS runs, admitted only with the registered hash
    gf = G11 / f"runs_{ds}.json"
    g11_seen, g11_conf = {}, []
    if gf.is_file():
        for r in _read_json(gf):
            if r.get("dataset") != ds or r.get("model") != "damrs" or not _is_converged(r):
                continue
            k = ("damrs_g11", int(r["seed"]))
            code_ok = (r.get("code_ok") is True and r.get("code_sha256_start") in DAMRS_G11_ACCEPTED
                       and r.get("code_sha256_end") == r.get("code_sha256_start"))
            if not code_ok:
                conflicts.append({"model": k[0], "seed": k[1], "reason": "code hash not registered (AMEND2 B4)",
                                  "sources": [f"phase_c3conv_g11/{gf.name}"]})
                continue
            if k in g11_seen:
                a, b = g11_seen[k]["test_result"]["Recall@20"], r["test_result"]["Recall@20"]
                if abs(a - b) > 1e-12:
                    g11_conf.append({"model": k[0], "seed": k[1], "R@20": [a, b],
                                     "sources": [f"phase_c3conv_g11/{gf.name}"] * 2})
                continue
            g11_seen[k] = dict(r, model="damrs_g11", _src=f"phase_c3conv_g11/{gf.name}")
    for c in g11_conf:      # excluded rather than guessed, as for the main source
        g11_seen.pop((c["model"], c["seed"]), None)
    out.update(g11_seen)
    return out, conflicts + g11_conf


# ============================================================================ output file
def _locked_update(path: Path, fn) -> dict:
    """Re-read `path` under an exclusive lock, apply fn(doc) in place, write atomically."""
    lock = path.with_suffix(path.suffix + ".lock")
    with open(lock, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        doc = json.loads(path.read_text()) if path.is_file() else {"rows": {}, "refused": {}}
        fn(doc)
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(doc, indent=1))
        tmp.replace(path)
        fcntl.flock(lf, fcntl.LOCK_UN)
    return doc


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _fingerprint(p: Path, span: int = 8 << 20) -> str:
    """sha256 over (size, first 8 MiB, last 8 MiB). Checkpoints are 0.1-0.5 GB on a disk shared
    with the training campaign (a full hash of one Clothing checkpoint took 197 s); identity is
    guarded by G1, this is provenance only. --full-hash records the full sha256 instead."""
    size = p.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with open(p, "rb") as f:
        h.update(f.read(span))
        if size > span:
            f.seek(max(span, size - span))
            h.update(f.read(span))
    return h.hexdigest()


# ============================================================================ scoring
def _metrics(m) -> dict:
    return {k: float(m[k]) for k in METRICS}


def _cfg_diff(cfg, snap: dict) -> dict:
    """Hyper-parameter keys whose value differs between the scoring-time Config and the
    training-time config_snapshot (run-control keys excluded)."""
    skip = {"seed", "ckpt_dir", "log_dir", "show_progress", "stopping_step", "epochs"}
    live = dict(cfg)
    diff = {}
    for k in sorted(set(live) | set(snap)):
        if k in skip:
            continue
        a, b = live.get(k, "<absent>"), snap.get(k, "<absent>")
        if a != b and str(a) != str(b):
            diff[k] = {"scoring": str(a), "training": str(b)}
    return diff


def _lgmrec_draws(model, loader, device, k: int) -> list:
    """Recall@20 of LGMRec's OWN full_sort_predict under k independent gumbel streams,
    evaluated batch by batch exactly as the trainer does (fresh draw per user batch)."""
    import torch
    from src.evaluation.topk_evaluator import TopKEvaluator
    vals = []
    for s in range(k):
        torch.manual_seed(1000 + s)
        ev = TopKEvaluator(metrics=["Recall"], topk=[20]); ev.reset()
        for batch in loader:
            uid = batch["user_ids"].to(device)
            hi = batch["history_indices"].to(device); hv = batch["history_values"].to(device)
            sc = model.full_sort_predict({"user": uid})
            if hi.numel() > 0:
                mask = hv.bool()
                row = torch.arange(sc.size(0), device=device).unsqueeze(1).expand_as(hi)
                safe = torch.where(hi >= 0, hi, torch.zeros_like(hi))
                sc[row[mask], safe[mask]] = float("-inf")
            ev.collect(torch.topk(sc, k=20, dim=-1)[1].cpu(), batch["positive_items"],
                       batch["positive_lengths"])
        vals.append(float(ev.compute()["Recall@20"]))
    return vals


def _cohesion_draws(mod, model, ds_obj, loader, device, k: int) -> list:
    """Recall@20 of COHESION's baseline under k independent user-graph draws, through the
    instrument's own chunked replay (recon_error certifies it equals full_sort_predict)."""
    from ranking_effects import evaluate_item_matrix
    vals, seed0 = [], mod.USER_GRAPH_SEED
    try:
        for s in range(k):
            mod.USER_GRAPH_SEED = 1000 + s
            if hasattr(model, mod._CACHE_ATTR):
                delattr(model, mod._CACHE_ATTR)
            st = mod._streams(model, ds_obj, device)
            u, i = mod._assemble(model, st, *mod._ARMS["baseline"])
            m, _, _ = evaluate_item_matrix(u, i, loader, device)
            vals.append(float(m["Recall@20"]))
            del u, i, st
    finally:
        mod.USER_GRAPH_SEED = seed0
        if hasattr(model, mod._CACHE_ATTR):
            delattr(model, mod._CACHE_ATTR)
    return vals


def score_one(model_name: str, ds: str, rec: dict, device: str, k_draws: int = K_DRAWS) -> dict:
    """Score one checkpoint. Returns a row with `guard` = {"ok": bool, ...}."""
    import torch
    from recsys_bridge import load_frozen
    from ranking_effects import evaluate_item_matrix, ranking_change
    from exp_exact_crossarch import RECON_TOL

    t = {"t0": time.time()}
    arch = ARCH.get(model_name, model_name)
    ck = Path(rec["ckpt_path"])
    row = {"model": model_name, "architecture": arch, "dataset": ds, "seed": int(rec["seed"]),
           "run_source": rec["_src"],
           "ckpt": str(ck), "best_epoch_logged": rec["best_epoch"],
           "R@20_logged": rec["test_result"]["Recall@20"],
           "logged_metrics": {k: rec["test_result"][k] for k in METRICS},
           "run_device": dict(RUN_DEVICE)}
    st = ck.stat()
    row["ckpt_size"], row["ckpt_mtime"] = st.st_size, st.st_mtime
    if FULL_HASH:
        row["ckpt_sha256"] = _sha256(ck)
    else:
        row["ckpt_fingerprint_head_tail_8MiB"] = _fingerprint(ck)

    # ---- G1: checkpoint identity -------------------------------------------------------
    raw = torch.load(ck, map_location="cpu", weights_only=False, mmap=True)
    snap = dict(raw.get("config_snapshot") or {})
    g1 = {"ckpt_epoch": raw.get("epoch"), "snapshot_seed": snap.get("seed"),
          "snapshot_stopping_step": snap.get("stopping_step"), "snapshot_epochs": snap.get("epochs")}
    g1["ok"] = (g1["ckpt_epoch"] == rec["best_epoch"] and g1["snapshot_seed"] == rec["seed"]
                and g1["snapshot_stopping_step"] == 100 and g1["snapshot_epochs"] == 3000)
    if model_name == "damrs_g11":   # AMEND2 B4: the model is rebuilt from the file now on disk
        g1["model_file_sha256_now"] = _sha256(DAMRS_FILE)
        g1["ok"] = g1["ok"] and g1["model_file_sha256_now"] in DAMRS_G11_ACCEPTED
    del raw
    t["identity"] = time.time()

    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    cfg, ds_obj, model, loader = load_frozen(arch, ds, device, ckpt_path=str(ck))
    row["config_diff_vs_training"] = _cfg_diff(cfg, snap)
    row["missing_keys"] = list(getattr(model, "_missing_keys", []))
    t["load"] = time.time()

    # ---- instrument: identical to exp_exact_crossarch.run_exact / phase1 lgmrec path -----
    if arch == "lgmrec":
        from phase1_knockout import lgmrec_variants, GUMBEL_SEED
        arms = lgmrec_variants(model)
        # recon (added here; the screen recorded none for LGMRec): the model's own
        # _forward_views under the same paired gumbel seed must reproduce the baseline arm.
        torch.manual_seed(GUMBEL_SEED)
        with torch.no_grad():
            mu, mi, _ = model._forward_views()
        bu, bi = arms["baseline"]
        idx = torch.arange(min(4096, model.n_users), device=bu.device)
        rec_err = float(((mu[idx] @ mi.t()) - (bu[idx] @ bi.t())).abs().max())
        del mu, mi
        row.update(taxonomy_class="C3", exactness="DELETE_PLUS_RENORM",
                   recon_source="exp_c3conv_score: _forward_views(GUMBEL_SEED) vs baseline arm")
        mod = None
    else:
        mod = importlib.import_module(f"exact_ko.{arch}")
        rec_err = float(mod.recon_error(model, ds_obj, device))
        row.update(taxonomy_class=getattr(mod, "CLASS", "?"),
                   exactness=getattr(mod, "EXACTNESS", "UNKNOWN"),
                   recon_source=f"exact_ko.{arch}.recon_error")
        arms = mod.variants(model, ds_obj, device)
    row["recon_error"], row["recon_tol"] = rec_err, RECON_TOL
    # exp_exact_crossarch.run_exact REFUSES a cell whose reconstruction error exceeds RECON_TOL.
    # PREREG_C3CONV_AMEND1.md A5: a row also passes if the error relative to the largest absolute
    # score is <= RECON_REL_TOL (about 8 float32 ulps); rows failing both are refused. Refused rows
    # still store their arms (never used by --verdicts).
    bu, bi = arms["baseline"]
    idx = torch.arange(min(2048, bu.shape[0]), device=bu.device)
    row["score_scale_max_abs"] = float((bu[idx] @ bi.t()).abs().max())
    row["recon_error_rel"] = rec_err / row["score_scale_max_abs"] if row["score_scale_max_abs"] else None
    row["recon_rel_tol"] = RECON_REL_TOL
    recon_fail = ((not str(row["exactness"]).startswith("UNDEFINED")) and rec_err > RECON_TOL
                  and not (row["recon_error_rel"] is not None and row["recon_error_rel"] <= RECON_REL_TOL))
    if recon_fail:
        row["recon_error_relative"] = rec_err / row["score_scale_max_abs"]
    arm_ex = getattr(mod, "ARM_EXACTNESS", {}) if mod else {}
    t["variants"] = time.time()

    base_m, base_topk, _ = evaluate_item_matrix(*arms["baseline"], loader, device)
    row["baseline"] = _metrics(base_m)
    row["arms"] = {}
    for stream, arm in ARMS.items():
        if arm not in arms:
            row["arms"][stream] = None
            continue
        m, topk, _ = evaluate_item_matrix(*arms[arm], loader, device)
        mm = _metrics(m)
        row["arms"][stream] = {"arm": arm, "arm_exactness": arm_ex.get(arm, row["exactness"]),
                               "metrics": mm,
                               "delta": {k: mm[k] - row["baseline"][k] for k in METRICS},
                               "ranking_change": ranking_change(base_topk, topk, k=20)}
        del topk
    del base_topk
    t["eval"] = time.time()

    # ---- G2 / G2' ----------------------------------------------------------------------
    base, logged = row["baseline"]["Recall@20"], row["R@20_logged"]
    g2 = {"base_R20": base, "logged_R20": logged, "diff": base - logged}
    if arch in STOCHASTIC:
        del arms
        draws = (_lgmrec_draws(model, loader, device, k_draws) if arch == "lgmrec"
                 else _cohesion_draws(mod, model, ds_obj, loader, device, k_draws))
        mu, sd = statistics.mean(draws), statistics.stdev(draws)
        z = abs(logged - mu) / sd if sd > 0 else (0.0 if logged == mu else math.inf)
        g2.update(mode="distributional", why=STOCHASTIC[arch], draws=draws,
                  draw_mean=mu, draw_sd=sd, z_logged=z, z_tol=Z_TOL,
                  logged_within_draw_range=min(draws) <= logged <= max(draws),
                  z_paired_baseline=abs(base - mu) / sd if sd > 0 else None,
                  ok=z <= Z_TOL)
    elif arch in TRAINLOOP_STATE:
        g2.update(mode="identity_only", why=TRAINLOOP_STATE[arch],
                  exact_match=abs(base - logged) <= BASE_TOL, ok=True)
    else:
        g2.update(mode="exact", tol=BASE_TOL, ok=abs(base - logged) <= BASE_TOL)
    t["guard"] = time.time()
    ok = bool(g1["ok"] and g2["ok"] and not recon_fail)
    reason = None if ok else ("G1 checkpoint identity" if not g1["ok"] else
                              f"reconstruction error {rec_err:.3e} > {RECON_TOL}" if recon_fail else
                              f"G2 baseline {base:.9f} vs logged {logged:.9f} ({g2['mode']})")
    row["guard"] = {"ok": ok, "reason": reason, "G1": g1, "G2": g2}
    row["guard_mode"] = g2["mode"]
    row["timing_s"] = {"identity+hash": t["identity"] - t["t0"], "load": t["load"] - t["identity"],
                       "recon+variants": t["variants"] - t["load"], "eval": t["eval"] - t["variants"],
                       "stochastic_guard": t["guard"] - t["eval"], "total": t["guard"] - t["t0"]}
    if str(device).startswith("cuda"):
        row["gpu_peak_GB"] = torch.cuda.max_memory_allocated() / 2**30
    _cleanup(model, arch)
    return row


def _cleanup(model, model_name):
    import gc, torch
    if model_name == "gume":     # module-level cache keyed by id(model): ids are reused after GC
        importlib.import_module("exact_ko.gume")._PLAIN_GRAPH_CACHE.clear()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_scoring(ds: str, models: list, seeds: list | None, device: str, rescore_refused: bool,
                k_draws: int, limit_min: float | None, retry_exceptions: bool = False) -> int:
    out = C3 / f"scored_{ds}.json"
    recs, conflicts = collect_runs(ds)
    doc = _locked_update(out, lambda d: d.setdefault("_meta", {}).update(
        {"conflicts": conflicts, "prereg": "results/phase_c3conv/PREREG_C3CONV.md",
         "arms": ARMS, "base_tol": BASE_TOL, "z_tol": Z_TOL, "k_draws": k_draws}))
    t_start = time.time()
    for (m, s), rec in sorted(recs.items(), key=lambda kv: (MODELS.index(kv[0][0]), kv[0][1])):
        if m not in models or (seeds and s not in seeds):
            continue
        key = f"{m}/s{s}"
        prev = doc["rows"].get(key) or (None if rescore_refused else doc["refused"].get(key))
        # --retry-exceptions: a row refused only because scoring raised (e.g. CUDA OOM on a GPU
        # shared with training) is retried; rows refused by a guard are not.
        if retry_exceptions and prev is not None and key in doc["refused"] \
                and str((prev.get("guard") or {}).get("reason", "")).startswith("exception:"):
            prev = None
        if prev and prev.get("R@20_logged") == rec["test_result"]["Recall@20"] \
                and prev.get("ckpt") == rec.get("ckpt_path"):
            print(f"skip {ds}/{key} (already {'scored' if key in doc['rows'] else 'refused'})")
            continue
        if not rec.get("ckpt_path") or not Path(rec["ckpt_path"]).is_file():
            row = {"model": m, "dataset": ds, "seed": s, "ckpt": rec.get("ckpt_path"),
                   "R@20_logged": rec["test_result"]["Recall@20"],
                   "guard": {"ok": False, "reason": "checkpoint file missing"}}
        else:
            if limit_min and (time.time() - t_start) / 60 > limit_min:
                print(f"time limit {limit_min} min reached; stopping before {ds}/{key}")
                break
            print(f"\n=== {ds}/{key}  logged R@20={rec['test_result']['Recall@20']:.6f} ===",
                  flush=True)
            try:
                row = score_one(m, ds, rec, device, k_draws)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                row = {"model": m, "dataset": ds, "seed": s, "ckpt": rec.get("ckpt_path"),
                       "R@20_logged": rec["test_result"]["Recall@20"],
                       "guard": {"ok": False, "reason": f"exception: {e!r}"}}
        row["scored_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

        def put(d, key=key, row=row):
            d["rows"].pop(key, None); d["refused"].pop(key, None)
            (d["rows"] if row["guard"]["ok"] else d["refused"])[key] = row
        doc = _locked_update(out, put)
        g = row["guard"]
        if g["ok"]:
            a = row["arms"]
            print(f"  OK  base={row['baseline']['Recall@20']:.6f} [{row['guard_mode']}] "
                  f"recon={row['recon_error']:.2e} "
                  f"dImg={a['image']['delta']['Recall@20']:+.6f} "
                  f"dTxt={a['text']['delta']['Recall@20']:+.6f} "
                  f"t={row['timing_s']['total']:.1f}s peak={row.get('gpu_peak_GB', 0):.2f}GB",
                  flush=True)
        else:
            print(f"  REFUSED: {g['reason']}", flush=True)
    return 0


# ============================================================================ verdicts
def _bh(p: list) -> list:
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adj, prev = [0.0] * m, 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, p[i] * m / rank)
        adj[i] = min(prev, 1.0)
    return adj


def _holm(p: list) -> list:
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def cell_stats(rows: list, stream: str) -> dict | None:
    from scipy import stats
    rows = sorted(rows, key=lambda r: r["seed"])
    rows = [r for r in rows if r["arms"].get(stream)]
    n = len(rows)
    if n < 2:
        return None
    base = [r["baseline"]["Recall@20"] for r in rows]
    logged = [r["R@20_logged"] for r in rows]
    d = [r["arms"][stream]["delta"]["Recall@20"] for r in rows]
    m, sd = statistics.mean(d), statistics.stdev(d)
    # PREREG_C3CONV_AMEND1.md A4: primary level floor on the trainer-logged test R@20
    fl, fp = 2 * statistics.stdev(logged), 2 * sd
    hi, lo = max(fl, fp), min(fl, fp)
    if sd > 0:
        t, p = stats.ttest_1samp(d, 0.0)
        t, p = float(t), float(p)
        half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    else:   # all deltas identical: t undefined. AMEND1 A2: p=1 if all exactly 0, else p=0 (kept in family)
        t, p, half = None, (1.0 if all(x == 0 for x in d) else 0.0), 0.0
    mb = statistics.mean(base)
    return {
        "n": n, "seeds": [r["seed"] for r in rows], "arm": rows[0]["arms"][stream]["arm"],
        "guard_modes": sorted({r["guard_mode"] for r in rows}),
        "mean_base_R20": mb, "mean_delta": m, "rel_delta_pct": 100 * m / mb, "sd_delta": sd,
        "F_level": fl, "F_paired": fp,
        "F_level_from_rebuilt_baseline": 2 * statistics.stdev(base),
        "x_F_level": abs(m) / fl if fl else None, "x_F_paired": abs(m) / fp if fp else None,
        "verdict": ("SIGNIFICANT" if abs(m) > hi else "BELOW" if abs(m) < lo else "MARGINAL"),
        "t": t, "df": n - 1, "p": p, "ci95": [m - half, m + half],
        "n_negative": sum(x < 0 for x in d), "deltas": d, "base": base,
        "mean_delta_other": {k: statistics.mean(r["arms"][stream]["delta"][k] for r in rows)
                             for k in METRICS if k != "Recall@20"},
    }


def run_verdicts() -> int:
    cells, counts = {}, {}
    for ds in DATASETS:
        f = C3 / f"scored_{ds}.json"
        if not f.is_file():
            continue
        doc = _read_json(f)
        by = {}
        for r in doc["rows"].values():
            if r["guard"]["ok"]:
                by.setdefault(r["model"], []).append(r)
        admit_models = ADMIT_MODELS.get(ds, REGISTERED)
        counts[ds] = {m: len(by.get(m, [])) for m in MODELS}
        admitted = all(counts[ds][m] >= MIN_SEEDS for m in admit_models)
        for m, rows in by.items():
            for s in ("image", "text", "both"):
                c = cell_stats(rows, s)
                if c is None:
                    continue
                c["dataset_admitted"] = admitted
                c["judged"] = bool(s in JUDGED_STREAMS and admitted and c["n"] >= MIN_SEEDS
                                   and m in admit_models)
                c["verdict_status"] = ("judged" if c["judged"] else
                                       f"VOID: {LEGACY_VOID[m]}" if m in LEGACY_VOID else
                                       "extra arm (not registered)" if s not in JUDGED_STREAMS else
                                       f"PROVISIONAL: n={c['n']}<{MIN_SEEDS} or dataset not admitted"
                                       " -- not a registered verdict")
                c["flags"] = {
                    "deletion_raises_accuracy": c["mean_delta"] > 0,
                    "joint_term_not_identified": m == "smore",   # image/text arms alias the bracket's hi end
                    "branch_not_content": m == "cohesion",       # arm deletes a trained branch, not content
                    "non_exact_guard": any(g != "exact" for g in c["guard_modes"]),
                }
                cells[f"{m}/{ds}/{s}"] = c
    # PREREG_C3CONV_AMEND1.md A1: PRIMARY family = all judged image AND text cells together;
    # SECONDARY = image cells only (the default-patience screen's convention).
    fam = {}
    for name, streams, exact_only in (("primary_image_and_text", JUDGED_STREAMS, False),
                                      ("secondary_image_only", ("image",), False),
                                      ("sensitivity_A3_exact_guard_only", JUDGED_STREAMS, True)):
        keys = [k for k, c in cells.items() if c["judged"] and k.rsplit("/", 1)[1] in streams
                and not (exact_only and c["flags"]["non_exact_guard"])]
        p = [cells[k]["p"] for k in keys]
        if p:
            for k, qb, qh in zip(keys, _bh(p), _holm(p)):
                cells[k][name] = {"p_bh": qb, "p_holm": qh, "bh_sig": qb <= Q_BH, "holm_sig": qh <= ALPHA_HOLM}
        fam[name] = {"m": len(keys), "cells": keys,
                     "bh_sig": [k for k in keys if cells[k][name]["bh_sig"]],
                     "holm_sig": [k for k in keys if cells[k][name]["holm_sig"]],
                     "significant_two_floor": [k for k in keys if cells[k]["verdict"] == "SIGNIFICANT"]}
    out = {"_meta": {"prereg": "results/phase_c3conv/PREREG_C3CONV.md",
                     "rule": "SIGNIFICANT iff |mean d| > max(F_level,F_paired); BELOW iff < both; "
                             "else MARGINAL. F_* = 2 sd (ddof=1). F_level on the trainer-logged "
                             "full-model R@20 (AMEND1 A4); the rebuilt-baseline version is stored "
                             "as F_level_from_rebuilt_baseline.",
                     "judged": f"n >= {MIN_SEEDS} and dataset admitted (every model of the dataset's "
                               f"registered set >= {MIN_SEEDS}; MicroLens set per AMEND1 A6)",
                     "amendment": "results/phase_c3conv/PREREG_C3CONV_AMEND1.md",
                     "multiplicity": f"BH q={Q_BH}, Holm alpha={ALPHA_HOLM}, two-sided paired t; primary "
                                     "family image+text jointly, secondary image only (AMEND1 A1)",
                     "seed_counts": counts,
                     "registered_models": {ds: ADMIT_MODELS.get(ds, REGISTERED) for ds in counts},
                     "void_models": LEGACY_VOID,
                     "amendment2": "results/phase_c3conv/PREREG_C3CONV_AMEND2.md",
                     "admitted": {ds: all(c[m] >= MIN_SEEDS for m in ADMIT_MODELS.get(ds, REGISTERED))
                                  for ds, c in counts.items()},
                     "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
           "families": fam, "cells": cells}
    tmp = C3 / "verdicts.json.tmp"
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(C3 / "verdicts.json")
    for k, c in cells.items():
        pp = f"{c['p']:.3g}" if c["p"] is not None else "n/a"
        print(f"{k:28s} n={c['n']} d={c['mean_delta']:+.6f} ({c['rel_delta_pct']:+.2f}%) "
              f"xFl={c['x_F_level'] or 0:.2f} xFp={c['x_F_paired'] or 0:.2f} {c['verdict']:11s} "
              f"p={pp} neg={c['n_negative']}/{c['n']} judged={c['judged']}")
    print(json.dumps(fam, indent=1))
    print(f"wrote {C3 / 'verdicts.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--k-draws", type=int, default=K_DRAWS)
    ap.add_argument("--rescore-refused", action="store_true",
                    help="retry rows previously refused (e.g. after an exception)")
    ap.add_argument("--retry-exceptions", action="store_true",
                    help="retry rows refused only because scoring raised an exception (e.g. CUDA OOM)")
    ap.add_argument("--time-limit-min", type=float, default=None,
                    help="stop starting new checkpoints after this many minutes")
    ap.add_argument("--verdicts", action="store_true", help="compute verdicts.json only")
    ap.add_argument("--cpu", action="store_true", help="score on CPU (slow; for cells whose "
                    "model-own forward exceeds the GPU cap, e.g. COHESION/microlens needs ~4.5 GB)")
    ap.add_argument("--gpu-cap-gb", type=float, default=GPU_CAP_GB,
                    help="per-process CUDA allocator cap in GiB (default %(default)s)")
    ap.add_argument("--full-hash", action="store_true",
                    help="record the full sha256 of each checkpoint (slow on the shared disk)")
    args = ap.parse_args()
    global FULL_HASH
    FULL_HASH = args.full_hash
    if args.verdicts:
        return run_verdicts()
    import torch
    use_gpu = torch.cuda.is_available() and not args.cpu
    device = f"cuda:{args.gpu}" if use_gpu else "cpu"
    if use_gpu:
        tot = torch.cuda.get_device_properties(args.gpu).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_cap_gb / tot), args.gpu)
    global RUN_DEVICE
    RUN_DEVICE = {"device": device, "gpu_cap_gb": args.gpu_cap_gb if use_gpu else None,
                  "torch": torch.__version__}
    C3.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        run_scoring(ds, args.models, args.seeds, device, args.rescore_refused, args.k_draws,
                    args.time_limit_min, args.retry_exceptions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
