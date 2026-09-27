"""What would a SINGLE-RUN modality audit have concluded?  (zero-compute; reads existing artifacts)

Prior multimodal-recommender audits report one run per condition. Using only our existing
eight-seed FREEDOM artifacts we enumerate EXACTLY (no sampling) every single-run audit that the
eight seeds allow, and report how far each one lands from the eight-seed conclusion.

Inputs (read-only):
  results/phase_holdout/freedom_p100*_runs.json   converged retraining (patience 100 / cap 3000),
                                                  pooled by each record's `dataset` field with the
                                                  stopping_step==100 / epochs_cap==3000 filter,
                                                  exactly as scripts/holdout_verdicts.py does
  results/phase_holdout/final_runs.json           default-patience retraining (HD in FACTS.md);
                                                  cross-checked against the per-batch files
  results/phase_convergence/converged_knockout.json  exact deletion on the 8 converged `full`
                                                  checkpoints (R@20 for image/text; NDCG@20 image)

Checks (asserted before any analysis):
  * pooled converged runs reproduce results/phase_holdout/final_verdicts_p100.json (d, F_paired,
    F_level, p, verdict) to 1e-12 using holdout_verdicts.cell();
  * final_runs.json reproduces results/phase_holdout/final_verdicts.json the same way;
  * final_runs.json == union of freedom_runs.json + freedom_{baby,sc,ml}extra_runs.json;
  * every knockout base_R20 equals the matched converged `full` run's logged test Recall@20.

Definitions.  rel(cond, full) = 100 * (cond - full) / full   [% of the full model's metric]
  eight-seed reference = 100 * mean(d_s) / mean(full_s), as holdout_verdicts.py / FACTS d%.
  (a) matched pairs   : full seed s  vs ablated seed s              (8)
  (b) unmatched pairs : full seed s  vs ablated seed s', all s, s'  (64, includes the 8 matched)
  Deletion is within one checkpoint, so only matched pairs exist.
  "loss" = -rel (positive = the ablation hurt).

Output -> results/phase_novelty/single_seed_instability.json
"""
from __future__ import annotations

import decimal
import itertools
import json
import statistics
import sys
from pathlib import Path

ROOT = Path("/workspace/MechInterp")
sys.path.insert(0, str(ROOT / "scripts"))
from holdout_verdicts import cell  # noqa: E402  (pure function; main() is not run)

HOLD = ROOT / "results" / "phase_holdout"
CK_F = ROOT / "results" / "phase_convergence" / "converged_knockout.json"
OUT = ROOT / "results" / "phase_novelty" / "single_seed_instability.json"

DATASETS = ["baby", "sports", "clothing", "microlens"]
MODS = {"image": "no_image", "text": "no_text"}
METRICS = ["Recall@20", "NDCG@20", "Recall@10"]
SEEDS = list(range(2024, 2032))
THRESH = [1, 2, 5]


# ----------------------------------------------------------------------------- presentation
def q(x, nd=2):
    """Round the exact binary value of x half-up to nd decimals (Python-version independent)."""
    if x is None:
        return None
    return str(decimal.Decimal(x).quantize(decimal.Decimal(1).scaleb(-nd),
                                           rounding=decimal.ROUND_HALF_UP))


def frac(k, n):
    ctx = decimal.Context(prec=50)
    return {"k": k, "n": n,
            "pct": q(ctx.divide(decimal.Decimal(100 * k), decimal.Decimal(n)), 1) if n else None}


def dmedian(xs):
    """Exact median of floats (Decimal arithmetic), returned as float."""
    s = sorted(decimal.Decimal(x) for x in xs)
    n = len(s)
    m = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return float(m)


def dcorr(xs, ys):
    """Pearson r computed exactly (Fractions), sqrt in 50-digit Decimal; version independent."""
    from fractions import Fraction as F
    X, Y = [F(x) for x in xs], [F(y) for y in ys]
    mx, my = sum(X) / len(X), sum(Y) / len(Y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(X, Y))
    sxx = sum((a - mx) ** 2 for a in X)
    syy = sum((b - my) ** 2 for b in Y)
    ctx = decimal.Context(prec=50)
    num = ctx.divide(decimal.Decimal(sxy.numerator), decimal.Decimal(sxy.denominator))
    den2 = ctx.divide(decimal.Decimal((sxx * syy).numerator), decimal.Decimal((sxx * syy).denominator))
    return float(ctx.divide(num, ctx.sqrt(den2)))


def sgn(x):
    return (x > 0) - (x < 0)


def summarize(rels, ref):
    """rels: list of relative changes (%); ref: eight-seed relative change (%)."""
    n = len(rels)
    s_ref = sgn(ref)
    opp = sum(1 for r in rels if sgn(r) == -s_ref and s_ref != 0)
    zero = sum(1 for r in rels if r == 0)
    out = {
        "n_pairs": n,
        "min_pct": min(rels), "max_pct": max(rels), "median_pct": dmedian(rels),
        "eight_seed_ref_pct": ref,
        "opposite_sign_to_ref": frac(opp, n),
        "exactly_zero": zero,
    }
    for t in THRESH:
        out[f"abs_gt_{t}pct"] = frac(sum(1 for r in rels if abs(r) > t), n)
    out["display"] = (f"{q(min(rels))}% .. {q(max(rels))}% (median {q(dmedian(rels))}%, "
                      f"8-seed {q(ref)}%); opposite sign {opp}/{n}")
    return out


# ----------------------------------------------------------------------------- loading
def load_converged():
    """Pool exactly as holdout_verdicts.main() does."""
    pooled = {}
    for f in sorted(HOLD.glob("freedom_p100*_runs.json")):
        for r in json.loads(f.read_text()):
            if ("test_result" not in r or r.get("stopping_step") != 100
                    or r.get("epochs_cap") != 3000):
                continue
            k = (r["dataset"], r["condition"], r["seed"])
            if k in pooled:
                a, b = pooled[k]["test_result"]["Recall@20"], r["test_result"]["Recall@20"]
                assert abs(a - b) < 1e-12, f"conflicting records for {k}"
            pooled[k] = r
    return pooled


def load_default():
    fin = {}
    for r in json.loads((HOLD / "final_runs.json").read_text()):
        if "test_result" in r:
            k = (r["dataset"], r["condition"], r["seed"])
            assert k not in fin, f"duplicate {k} in final_runs.json"
            fin[k] = r
    # cross-check against the per-batch source files
    parts = {}
    for name in ["freedom_runs.json", "freedom_babyextra_runs.json",
                 "freedom_scextra_runs.json", "freedom_mlextra_runs.json"]:
        for r in json.loads((HOLD / name).read_text()):
            if "test_result" in r:
                k = (r["dataset"], r["condition"], r["seed"])
                assert k not in parts, f"{k} in two default-patience batch files"
                parts[k] = r
    assert set(parts) == set(fin), "final_runs.json != union of default-patience batch files"
    for k in fin:
        for m in METRICS:
            assert fin[k]["test_result"][m] == parts[k]["test_result"][m], (k, m)
    return fin


def reproduce(pooled, verdict_file):
    stored = json.loads(verdict_file.read_text())
    checked = []
    for ds in DATASETS:
        runs = [r for k, r in pooled.items() if k[0] == ds]
        for cond in MODS.values():
            key = f"{ds}/{cond}"
            if key not in stored:
                continue
            c = cell(runs, cond)
            for fld in ("d", "F_paired", "F_level", "p"):
                assert abs(c[fld] - stored[key][fld]) < 1e-12, (verdict_file.name, key, fld)
            assert c["verdict"] == stored[key]["verdict"], (verdict_file.name, key)
            checked.append({"cell": key, "d_pct": c["d_pct"], "verdict": c["verdict"],
                            "p": c["p"]})
    return checked


def arms(pooled, ds, metric):
    """{condition: {seed: value}} for one dataset / metric."""
    out = {}
    for (d, cond, s), r in pooled.items():
        if d == ds:
            out.setdefault(cond, {})[s] = r["test_result"][metric]
    for cond in ["full", *MODS.values()]:
        assert sorted(out[cond]) == SEEDS, (ds, cond, sorted(out.get(cond, {})))
    return out


def ref_pct(full, abl):
    d = [abl[s] - full[s] for s in SEEDS]
    return 100 * statistics.mean(d) / statistics.mean(full[s] for s in SEEDS)


def rel(c, f):
    return 100 * (c - f) / f


# ----------------------------------------------------------------------------- analyses
def retraining_block(pooled, tag):
    res = {}
    for ds in DATASETS:
        res[ds] = {}
        for metric in METRICS:
            a = arms(pooled, ds, metric)
            full = a["full"]
            cell_m = {}
            for mod, cond in MODS.items():
                abl = a[cond]
                ref = ref_pct(full, abl)
                matched = [rel(abl[s], full[s]) for s in SEEDS]
                unmatched = [rel(abl[s2], full[s1]) for s1 in SEEDS for s2 in SEEDS]
                full_vals = [full[s] for s in SEEDS]
                abl_vals = [abl[s] for s in SEEDS]
                cell_m[mod] = {
                    "matched": summarize(matched, ref),
                    "unmatched": summarize(unmatched, ref),
                    "matched_per_seed_pct": {str(s): v for s, v in zip(SEEDS, matched)},
                    # does seed-matching remove variance? (Pearson r of full vs ablated arm)
                    "pearson_full_vs_ablated": dcorr(full_vals, abl_vals),
                }
            # image-vs-text ordering (loss_image > loss_text ?) against the 8-seed ordering
            li_ref = -cell_m["image"]["matched"]["eight_seed_ref_pct"]
            lt_ref = -cell_m["text"]["matched"]["eight_seed_ref_pct"]
            ref_img_first = li_ref > lt_ref

            def order_stats(pairs):
                n = len(pairs)
                flips = sum(1 for li, lt in pairs if (li > lt) != ref_img_first)
                return {"n": n, "reversed_vs_8seed": frac(flips, n)}

            ai, at = a["no_image"], a["no_text"]
            m_pairs = [(-rel(ai[s], full[s]), -rel(at[s], full[s])) for s in SEEDS]
            shared_full = [(-rel(ai[i], full[f]), -rel(at[t], full[f]))
                           for f in SEEDS for i in SEEDS for t in SEEDS]
            indep_full = [(-rel(ai[i], full[f1]), -rel(at[t], full[f2]))
                          for f1 in SEEDS for i in SEEDS for f2 in SEEDS for t in SEEDS]
            cell_m["image_vs_text_ordering"] = {
                "eight_seed_loss_pct": {"image": li_ref, "text": lt_ref},
                "eight_seed_says": "image loss > text loss" if ref_img_first
                                   else "text loss > image loss",
                "matched_same_seed": order_stats(m_pairs),
                "one_full_run_any_ablation_runs": order_stats(shared_full),
                "independent_full_runs": order_stats(indep_full),
            }
            res[ds][metric] = cell_m
    return res


def deletion_block(conv_pooled):
    ck = json.loads(CK_F.read_text())
    res = {}
    for ds in DATASETS:
        rows = {r["seed"]: r for r in ck[ds]["per_seed"]}
        assert sorted(rows) == SEEDS, (ds, sorted(rows))
        # integrity: knockout baseline == the converged full run's logged R@20
        mism = {}
        for s in SEEDS:
            logged = conv_pooled[(ds, "full", s)]["test_result"]
            if abs(rows[s]["base_R20"] - logged["Recall@20"]) > 1e-9:
                mism[s] = (rows[s]["base_R20"], logged["Recall@20"])
        assert not mism, (ds, mism)
        base = [rows[s]["base_R20"] for s in SEEDS]
        out = {}
        for mod in ["image", "text"]:
            d = [rows[s][f"d_{mod}"] for s in SEEDS]
            ref = 100 * statistics.mean(d) / statistics.mean(base)
            matched = [100 * rows[s][f"d_{mod}"] / rows[s]["base_R20"] for s in SEEDS]
            out[mod] = {"Recall@20": {
                "matched": summarize(matched, ref),
                "matched_per_seed_pct": {str(s): v for s, v in zip(SEEDS, matched)},
                "stored_verdict": ck[ds]["streams"][mod]["verdict"]}}
        # NDCG@20 (image only is stored); base NDCG@20 = the matched converged full run's
        # logged test NDCG@20 (same checkpoint: base_R20 verified equal above)
        bN = [conv_pooled[(ds, "full", s)]["test_result"]["NDCG@20"] for s in SEEDS]
        dN = [rows[s]["dN_image"] for s in SEEDS]
        refN = 100 * statistics.mean(dN) / statistics.mean(bN)
        out["image"]["NDCG@20"] = {
            "matched": summarize([100 * x / b for x, b in zip(dN, bN)], refN),
            "note": "base NDCG@20 taken from the converged full run's logged test_result "
                    "(same checkpoint; R@20 baseline verified equal to 1e-9)"}
        # image-vs-text ordering under deletion (per seed)
        li = [-100 * rows[s]["d_image"] / rows[s]["base_R20"] for s in SEEDS]
        lt = [-100 * rows[s]["d_text"] / rows[s]["base_R20"] for s in SEEDS]
        ref_img_first = (-out["image"]["Recall@20"]["matched"]["eight_seed_ref_pct"]
                         > -out["text"]["Recall@20"]["matched"]["eight_seed_ref_pct"])
        flips = sum(1 for a, b in zip(li, lt) if (a > b) != ref_img_first)
        out["image_vs_text_ordering_R20"] = {
            "eight_seed_says": "image loss > text loss" if ref_img_first
                               else "text loss > image loss",
            "matched_same_seed": {"n": 8, "reversed_vs_8seed": frac(flips, 8)}}
        out["unavailable"] = ("Recall@10 for deletion and NDCG@20 for text deletion are not "
                              "stored in converged_knockout.json; not computed.")
        res[ds] = out
    return res


def dataset_ordering(pooled, tag):
    """For every dataset pair and modality: P(single-run audit orders the two datasets' loss
    opposite to the eight-seed ordering), Recall@20."""
    res = {}
    loss = {}
    for ds in DATASETS:
        a = arms(pooled, ds, "Recall@20")
        full = a["full"]
        for mod, cond in MODS.items():
            abl = a[cond]
            loss[(ds, mod)] = {
                "ref": -ref_pct(full, abl),
                "matched": [-rel(abl[s], full[s]) for s in SEEDS],
                "unmatched": [-rel(abl[s2], full[s1]) for s1 in SEEDS for s2 in SEEDS],
            }
    for mod in MODS:
        for d1, d2 in itertools.combinations(DATASETS, 2):
            L1, L2 = loss[(d1, mod)], loss[(d2, mod)]
            first = d1 if L1["ref"] > L2["ref"] else d2
            big, small = (L1, L2) if first == d1 else (L2, L1)
            ent = {"eight_seed_loss_pct": {d1: L1["ref"], d2: L2["ref"]},
                   "eight_seed_says_larger_loss": first}
            for kind in ["matched", "unmatched"]:
                n = len(big[kind]) * len(small[kind])
                rev = sum(1 for x in big[kind] for y in small[kind] if y >= x)
                ent[f"both_single_{kind}"] = frac(rev, n)
            # one side single-run, the other side at its eight-seed mean
            ent["single_run_of_larger_vs_8seed_of_smaller_matched"] = frac(
                sum(1 for x in big["matched"] if x <= small["ref"]), 8)
            ent["single_run_of_smaller_vs_8seed_of_larger_matched"] = frac(
                sum(1 for y in small["matched"] if y >= big["ref"]), 8)
            ent["single_run_of_larger_vs_8seed_of_smaller_unmatched"] = frac(
                sum(1 for x in big["unmatched"] if x <= small["ref"]), 64)
            ent["single_run_of_smaller_vs_8seed_of_larger_unmatched"] = frac(
                sum(1 for y in small["unmatched"] if y >= big["ref"]), 64)
            res[f"{mod}:{d1}_vs_{d2}"] = ent
    return res


def protocol_block(conv, dflt):
    """Same seed number, default patience vs converged: does a single run's conclusion change?"""
    res = {}
    for ds in DATASETS:
        ac, ad = arms(conv, ds, "Recall@20"), arms(dflt, ds, "Recall@20")
        res[ds] = {}
        for mod, cond in MODS.items():
            rc = [rel(ac[cond][s], ac["full"][s]) for s in SEEDS]
            rd = [rel(ad[cond][s], ad["full"][s]) for s in SEEDS]
            refc, refd = ref_pct(ac["full"], ac[cond]), ref_pct(ad["full"], ad[cond])
            per = {str(s): {"default_pct": x, "converged_pct": y} for s, x, y in zip(SEEDS, rd, rc)}
            sign_dis = sum(1 for x, y in zip(rd, rc) if sgn(x) != sgn(y))
            thr = {}
            for t in THRESH:
                thr[f"gt_{t}pct_disagree_same_seed"] = frac(
                    sum(1 for x, y in zip(rd, rc) if (abs(x) > t) != (abs(y) > t)), 8)
            # cross product of all single matched runs: P(default single-run loss > converged)
            gt = sum(1 for x in rd for y in rc if -x > -y)
            res[ds][mod] = {
                "eight_seed_pct": {"default": refd, "converged": refc},
                "matched_range_pct": {"default": [min(rd), max(rd)], "converged": [min(rc), max(rc)]},
                "same_seed_sign_disagreement": frac(sign_dis, 8),
                **thr,
                "abs_diff_same_seed_pct_median": dmedian([abs(x - y) for x, y in zip(rd, rc)]),
                "abs_diff_same_seed_pct_max": max(abs(x - y) for x, y in zip(rd, rc)),
                "P_default_single_loss_gt_converged_single_loss": frac(gt, 64),
                "per_seed": per,
                "display": (f"default {q(min(rd))}..{q(max(rd))}% (8-seed {q(refd)}%) vs "
                            f"converged {q(min(rc))}..{q(max(rc))}% (8-seed {q(refc)}%)"),
            }
        # best epochs, to document that the two protocols are different runs
        res[ds]["median_best_epoch_full"] = {
            "default": statistics.median(dflt[(ds, "full", s)]["best_epoch"] for s in SEEDS),
            "converged": statistics.median(conv[(ds, "full", s)]["best_epoch"] for s in SEEDS)}
        # is the default-patience `full` score reproduced inside the converged run's curve?
        same = []
        for s in SEEDS:
            same.append(dflt[(ds, "full", s)]["best_epoch"] <= conv[(ds, "full", s)]["best_epoch"])
        res[ds]["default_best_epoch_le_converged_full"] = frac(sum(same), 8)
    return res


def main() -> int:
    conv = load_converged()
    dflt = load_default()
    checks = {
        "converged_reproduces_final_verdicts_p100": reproduce(conv, HOLD / "final_verdicts_p100.json"),
        "default_reproduces_final_verdicts": reproduce(dflt, HOLD / "final_verdicts.json"),
        "final_runs_equals_union_of_batch_files": True,
        "knockout_base_R20_equals_logged_converged_full": True,
    }
    out = {
        "_doc": __doc__,
        "checks": checks,
        "retraining_converged": retraining_block(conv, "converged"),
        "retraining_default_patience": retraining_block(dflt, "default"),
        "deletion_converged": deletion_block(conv),
        "dataset_ordering_converged": dataset_ordering(conv, "converged"),
        "dataset_ordering_default_patience": dataset_ordering(dflt, "default"),
        "protocol_default_vs_converged_R20": protocol_block(conv, dflt),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=False))

    # --------------------------------------------------------------- console summary
    print("checks passed:", ", ".join(k for k in checks))
    for tag in ["retraining_converged", "retraining_default_patience"]:
        print(f"\n== {tag} (Recall@20)")
        for ds in DATASETS:
            for mod in MODS:
                c = out[tag][ds]["Recall@20"][mod]
                print(f"  {ds:9s} {mod:5s} matched   {c['matched']['display']}"
                      f"  |>1%| {c['matched']['abs_gt_1pct']['k']}/8")
                print(f"  {'':9s} {'':5s} unmatched {c['unmatched']['display']}"
                      f"  |>1%| {c['unmatched']['abs_gt_1pct']['k']}/64  r={c['pearson_full_vs_ablated']:+.2f}")
            o = out[tag][ds]["Recall@20"]["image_vs_text_ordering"]
            print(f"  {ds:9s} ordering ({o['eight_seed_says']}): reversed matched "
                  f"{o['matched_same_seed']['reversed_vs_8seed']['k']}/8, shared-full "
                  f"{o['one_full_run_any_ablation_runs']['reversed_vs_8seed']['k']}/512, indep "
                  f"{o['independent_full_runs']['reversed_vs_8seed']['k']}/4096")
    print("\n== deletion_converged (Recall@20)")
    for ds in DATASETS:
        for mod in ["image", "text"]:
            print(f"  {ds:9s} {mod:5s} {out['deletion_converged'][ds][mod]['Recall@20']['matched']['display']}")
    for tag in ["dataset_ordering_converged", "dataset_ordering_default_patience"]:
        e = out[tag]["image:baby_vs_microlens"]
        print(f"\n== {tag} image baby vs microlens: {e['eight_seed_says_larger_loss']} larger; "
              f"reversed matched {e['both_single_matched']['k']}/{e['both_single_matched']['n']}, "
              f"unmatched {e['both_single_unmatched']['k']}/{e['both_single_unmatched']['n']}")
    print("\n== protocol (same seed, default vs converged, R@20)")
    for ds in DATASETS:
        for mod in MODS:
            p = out["protocol_default_vs_converged_R20"][ds][mod]
            print(f"  {ds:9s} {mod:5s} {p['display']}; sign disagree "
                  f"{p['same_seed_sign_disagreement']['k']}/8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
