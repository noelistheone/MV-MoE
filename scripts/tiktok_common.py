"""Shared plumbing for the TikTok analysis scripts (tiktok_verdicts / tiktok_converged_knockout /
tiktok_alignment / tiktok_graph_health). Nothing here computes a statistic of its own.

The paper's certified statistics live in the "exact-arith" block of scripts/gen_facts.py
(Decimal arithmetic, stdlib Student-t, two-floor rule, holdout_arm, ttest1, ...). gen_facts.py is a
top-level script that writes results/FACTS.md when imported, so it cannot be imported as a
module. `exact_arith()` instead executes ONLY the source between its `# >>> exact-arith` and
`# <<< exact-arith` markers, verbatim, in a private namespace, and returns that namespace. The
functions used are therefore byte-for-byte the certified ones (the block's sha256 is recorded in
every output so a later edit of gen_facts.py is visible).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/workspace/MechInterp")
SV2 = ROOT / "results" / "phase_shortvideo2"
TIKTOK_RUNS = SV2 / "freedom_p100_tiktok_runs.json"
LIGHTGCN_RUNS = SV2 / "lightgcn_p100_tiktok_runs.json"
CKPT_DIR = SV2 / "_scratch" / "ckpts"
PATIENCE, CAP = 100, 3000

_GF = ROOT / "scripts" / "gen_facts.py"
_NS = None


def exact_arith() -> dict:
    """Namespace holding gen_facts.py's exact-arith block (D, dmean, dsd, ttest1, two_floor,
    holdout_arm, load_p100_holdout, sci, fx, ptab, t_p2, ...), executed verbatim."""
    global _NS
    if _NS is None:
        src = _GF.read_text()
        a = src.index("# >>> exact-arith")
        b = src.index("# <<< exact-arith")
        block = src[a:b]
        ns = {"__name__": "gen_facts_exact_arith", "json": json, "Path": Path, "ROOT": ROOT}
        exec(compile(block, f"{_GF}#exact-arith", "exec"), ns)  # noqa: S102 - our own file
        ns["_block_sha256"] = hashlib.sha256(block.encode()).hexdigest()
        _NS = ns
    return _NS


def src_row(*cells) -> str:
    """gen_facts._src_row (outside the exact-arith block; one line, copied)."""
    return "| " + " | ".join(str(c) for c in cells) + " |"


def num(x):
    """JSON form of an exact Decimal (float), recursing into lists/tuples."""
    ns = exact_arith()
    if isinstance(x, ns["_Dc"]):
        return float(x)
    if isinstance(x, (list, tuple)):
        return [num(v) for v in x]
    return x


def pool_runs(files) -> dict:
    """(dataset, condition, seed) -> record, for the converged protocol only. Identical filter and
    conflict rule to holdout_verdicts.main() and gen_facts.load_p100_holdout() (the self-test in
    tiktok_verdicts.py asserts it returns exactly load_p100_holdout()'s dict on phase_holdout)."""
    ns = exact_arith()
    pooled = {}
    for f in files:
        f = Path(f)
        if not f.is_file():
            continue
        for r in json.loads(f.read_text()):
            if "test_result" not in r or r.get("stopping_step") != PATIENCE or r.get("epochs_cap") != CAP:
                continue
            k = (r["dataset"], r["condition"], r["seed"])
            if k in pooled:
                a_, b_ = ns["D"](pooled[k]["test_result"]["Recall@20"]), ns["D"](r["test_result"]["Recall@20"])
                assert abs(a_ - b_) < ns["_Dc"]("1e-12"), f"conflicting records for {k} ({f.name})"
            pooled[k] = r
    return pooled


def install_overlay() -> None:
    for p in (str(ROOT / "scripts"), "/workspace/Recsys"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import recsys_extra_datasets
    recsys_extra_datasets.install()


def load_dataset(name: str):
    """RecDataset exactly as the trainers build it (FREEDOM config; the overlay supplies TikTok)."""
    install_overlay()
    from src.utils import Config
    from src.data.dataset import RecDataset
    cfg = Config("freedom", name, cli_overrides={})
    return cfg, RecDataset(cfg)


def finished_full_ckpts() -> list:
    """(seed, checkpoint, logged run) for every converged TikTok full-condition run that has
    FINISHED (its record is in the runs file) and whose checkpoint file exists and is the one the
    record names. A checkpoint without a logged record may still be mid-write and is never used."""
    pooled = pool_runs([TIKTOK_RUNS])
    out = []
    for (ds, cond, seed), r in sorted(pooled.items(), key=lambda kv: kv[0][2]):
        if ds != "tiktok" or cond != "full":
            continue
        ck = CKPT_DIR / f"hold_p{PATIENCE}e{CAP}_freedom_tiktok_full_s{seed}.pt"
        if ck.is_file() and r.get("ckpt_path") and Path(r["ckpt_path"]).name == ck.name:
            out.append((seed, ck, r))
    return out


# ----------------------------------------------------------------------------- deletion statistics
def ck_cell_float(rows: list) -> dict:
    """The per-dataset cell of results/phase_convergence/converged_knockout.json, computed exactly
    as exp_converged_knockout.main() computes it (float `statistics`, same keys, same rule; the
    self-test reproduces the stored MicroLens/Amazon cells to 0)."""
    import statistics
    rows = sorted(rows, key=lambda x: x["seed"])
    base = [r["base_R20"] for r in rows]
    cell = {"n_seeds": len(rows), "mean_R20": statistics.mean(base),
            "F_level": 2 * statistics.stdev(base) if len(base) > 1 else None,
            "per_seed": rows, "streams": {}}
    for s in ["image", "text", "both"]:
        dd = [r[f"d_{s}"] for r in rows]
        if len(dd) < 2:
            continue
        m, sd = statistics.mean(dd), statistics.stdev(dd)
        fp, fl = 2 * sd, cell["F_level"]
        hi, lo = max(fp, fl), min(fp, fl)
        cell["streams"][s] = {"mean_delta": m, "sd": sd, "F_paired": fp,
                              "x_F_paired": abs(m) / fp if fp else None,
                              "x_F_level": abs(m) / fl if fl else None,
                              "verdict": ("SIGNIFICANT" if abs(m) > hi else
                                          "NULL" if abs(m) < lo else "MARGINAL"),
                              "n_negative": sum(x < 0 for x in dd)}
    return cell


def deletion_exact(per_seed: list) -> dict:
    """gen_facts s.21a, per stream, in exact arithmetic (the lines of s.21a, not a re-derivation)."""
    ns = exact_arith()
    D, dmean, dsd, ttest1, two_floor = ns["D"], ns["dmean"], ns["dsd"], ns["ttest1"], ns["two_floor"]
    _ps = per_seed
    _base = [D(r["base_R20"]) for r in _ps]
    _mu, _fl = dmean(_base), 2 * dsd(_base)
    out = {}
    for _st in ("image", "text", "both"):
        _dd = [D(r["d_" + _st]) for r in _ps]
        _tt = ttest1(_dd)
        _fp = 2 * _tt["sd"]
        _rb = dmean([r["rbo_" + _st] for r in _ps]) if all(("rbo_" + _st) in r for r in _ps) else None
        out[_st] = {"mean": _tt["mean"], "pct": 100 * _tt["mean"] / _mu, "fl": _fl, "fp": _fp,
                    "xfl": abs(_tt["mean"]) / _fl, "xfp": abs(_tt["mean"]) / _fp, "t": _tt["t"],
                    "p": _tt["p"], "ci_pct": tuple(100 * x / _mu for x in _tt["ci"]), "mu": _mu,
                    "neg": sum(1 for x in _dd if x < 0), "pos": sum(1 for x in _dd if x > 0),
                    "rbo": _rb, "verdict": two_floor(_tt["mean"], _fp, _fl), "d": _dd,
                    "base": _base, "n": len(_dd), "seeds": [r["seed"] for r in _ps]}
    return out


def deletion_row(ds: str, st: str, rec: dict, tag: str = "CK") -> str:
    """The s.21a table row, formatted by gen_facts' own sci/fx/ptab."""
    ns = exact_arith()
    sci, fx, ptab = ns["sci"], ns["fx"], ns["ptab"]
    _rb = rec["rbo"]
    return src_row(ds, st, sci(rec["mean"], 3), fx(rec["pct"], 2, True), fx(rec["pct"], 1, True),
                   f"[{fx(rec['ci_pct'][0], 2, True)}, {fx(rec['ci_pct'][1], 2, True)}]",
                   fx(rec["xfl"], 2), fx(rec["xfp"], 2), fx(rec["fl"] / rec["fp"], 2), fx(rec["t"], 2, True),
                   sci(rec["p"], 2), ptab(rec["p"]), f"{rec['neg']}/{rec['n']}",
                   f"{rec['pos']}/{rec['n']}", fx(_rb, 3) if _rb is not None else "--",
                   rec["verdict"], tag)


def jsonable(rec: dict) -> dict:
    return {k: num(v) for k, v in rec.items()}
