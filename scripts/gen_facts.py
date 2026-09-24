"""Emit every number the revised paper is allowed to cite, straight from the artifacts.

Writes paper/revision/FACTS.md. Any number in the manuscript that is not in here is either
wrong or needs a new artifact. Regenerate after every experiment lands.
"""
from __future__ import annotations
import json, glob, statistics, math, collections, re
from pathlib import Path

# The repository root is this file's parent's parent (scripts/..), so a checkout or release
# export reads ITS OWN results/ -- a hard-coded workspace path would silently read another tree.
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "FACTS.md"

# CANONICAL content-interface taxonomy, the single source of truth for the manuscript.
# The per-model modules in scripts/exact_ko/ carry LEGACY class strings from the survey that
# produced them ("C1+C2g", bm3="C4", ...); those numbers predate the paper's C1-C5 scheme and
# MUST NOT be used in the manuscript. Everything below overrides them.
TAXONOMY = {
    "freedom":  ("C1", "frozen similarity routing"),
    "lattice":  ("C1", "frozen similarity routing"),
    "damrs":    ("C1", "frozen similarity routing"),
    "vbpr":     ("C2", "live projected feature, constant-weight additive"),
    "mentor":   ("C2", "live projected feature, constant-weight additive"),
    "cohesion": ("C1+C2", "compound: frozen graph path AND a live on-path branch"),
    "lgmrec":   ("C3", "shared normaliser; deletion re-points the survivor"),
    "mmgcn":    ("C3", "shared normaliser (intra-tower); towers parametrically disjoint"),
    "mgcn":     ("C3", "shared normaliser; deletion re-points the survivor"),
    "smore":    ("C4", "irreducibly joint term; per-modality NOT identified"),
    "gume":     ("C4", "irreducibly joint kNN-intersection block; per-modality NOT identified"),
    "bm3":      ("C5", "content off the inference path; knockout UNDEFINED"),
}
def cls(m):
    return TAXONOMY.get(m, ("?", "unclassified"))[0]
L = []
def w(s=""): L.append(s)

def jload(p):
    p = ROOT / p
    return json.loads(p.read_text()) if p.is_file() else None

# >>> exact-arith (used by the s.19 MicroLens note and every s.21 subsection)
# Every number s.21 prints is computed in decimal arithmetic from the shortest-repr expansion of
# the stored float, Decimal(repr(x)), and rounded ROUND_HALF_UP only when printed. No float sum(),
# `statistics` or `math` function touches these values: Python 3.12 changed float sum() (and the
# rounding of statistics.stdev), which flips .5 cells between interpreters. The Student-t, F and
# regularized incomplete-beta values are computed here as well (stdlib only, no scipy), so the
# generator runs unchanged on a bare interpreter and prints byte-identical text on 3.11 and 3.12.
import decimal as _dec
from fractions import Fraction as _Fr
_dec.setcontext(_dec.Context(prec=60, rounding=_dec.ROUND_HALF_EVEN, Emin=-999999, Emax=999999,
                             traps=[_dec.InvalidOperation, _dec.DivisionByZero, _dec.Overflow]))
_Dc = _dec.Decimal


def D(x):
    """Exact decimal image of a stored number (floats via their shortest repr)."""
    if isinstance(x, _Dc):
        return x
    if isinstance(x, bool):
        raise TypeError("bool is not a number here")
    if isinstance(x, int):
        return _Dc(x)
    if isinstance(x, float):
        return _Dc(repr(x))
    if isinstance(x, str):
        return _Dc(x)
    if isinstance(x, _Fr):
        return _Dc(x.numerator) / _Dc(x.denominator)
    raise TypeError(f"cannot convert {type(x).__name__}")


def dsum(xs):
    s = _Dc(0)
    for x in xs:
        s += D(x)
    return s


def dmean(xs):
    xs = [D(x) for x in xs]
    return dsum(xs) / len(xs)


def dvar(xs):
    xs = [D(x) for x in xs]
    m = dmean(xs)
    return dsum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def dsd(xs):
    return dvar(xs).sqrt()


def dmedian(xs):
    s = sorted(D(x) for x in xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def dcorr(xs, ys):
    xs, ys = [D(x) for x in xs], [D(y) for y in ys]
    mx, my = dmean(xs), dmean(ys)
    sxy = dsum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = dsum((x - mx) ** 2 for x in xs)
    syy = dsum((y - my) ** 2 for y in ys)
    return sxy / (sxx * syy).sqrt()


def q(x, nd):
    """Round half-up to nd decimals (nd may be negative)."""
    return D(x).quantize(_Dc(1).scaleb(-nd), rounding=_dec.ROUND_HALF_UP)


def fx(x, nd, sign=False):
    """Fixed-point string, half-up; sign=True prints '+' on positives (exact zero stays '0')."""
    x = D(x)
    s = format(q(x, nd), "f")
    if sign and x != 0 and not s.startswith("-"):
        s = "+" + s
    return s


def sci(x, sig=2):
    """Scientific string with `sig` significant digits, half-up, e.g. 2.1e-05."""
    x = D(x)
    if x == 0:
        return "0"
    e = x.adjusted()
    m = x.scaleb(-e).quantize(_Dc(1).scaleb(-(sig - 1)), rounding=_dec.ROUND_HALF_UP)
    if abs(m) >= 10:
        e += 1
        m = x.scaleb(-e).quantize(_Dc(1).scaleb(-(sig - 1)), rounding=_dec.ROUND_HALF_UP)
    return f"{format(m, 'f')}e{e:+03d}"


def ptab(p):
    """The p-value form the SAC tables print, rounded once from the exact value:
    <1e-7; 1 significant digit below 1e-3; 2 significant digits in [1e-3, 1e-2); 2 decimals above."""
    p = D(p)
    if p < _Dc("1e-7"):
        return "<1e-7"
    if p < _Dc("0.001"):
        return sci(p, 1)
    if p < _Dc("0.01"):
        return format(p.quantize(_Dc("0.0001"), rounding=_dec.ROUND_HALF_UP), "f")
    return format(q(p, 2), "f")


def _gauss_legendre_pi():
    a, b, t, p = _Dc(1), _Dc(1) / _Dc(2).sqrt(), _Dc(1) / 4, _Dc(1)
    for _ in range(10):
        an = (a + b) / 2
        b = (a * b).sqrt()
        t -= p * (a - an) ** 2
        a = an
        p *= 2
    return (a + b) ** 2 / (4 * t)


_PI = _gauss_legendre_pi()


def _bernoulli(nmax):
    """Exact Bernoulli numbers B_0..B_nmax (Akiyama-Tanigawa); only the even ones are used."""
    A, B = [_Fr(0)] * (nmax + 1), []
    for m in range(nmax + 1):
        A[m] = _Fr(1, m + 1)
        for j in range(m, 0, -1):
            A[j - 1] = j * (A[j - 1] - A[j])
        B.append(A[0])
    return B


_BERN = _bernoulli(40)
_HALF_LN_2PI = (2 * _PI).ln() / 2
_LGAMMA_CACHE = {}


def dlgamma(x):
    """ln Gamma(x) for x > 0: upward shift to x >= 40, then a 20-term Stirling series."""
    x0 = D(x)
    if x0 in _LGAMMA_CACHE:
        return _LGAMMA_CACHE[x0]
    if x0 <= 0:
        raise ValueError("dlgamma needs x > 0")
    xx, shift = x0, _Dc(0)
    while xx < 40:
        shift += xx.ln()
        xx += 1
    s = (xx - _Dc("0.5")) * xx.ln() - xx + _HALF_LN_2PI
    xp, x2 = xx, xx * xx
    for k in range(1, 21):
        b = _BERN[2 * k]
        s += _Dc(b.numerator) / _Dc(b.denominator) / (2 * k * (2 * k - 1) * xp)
        xp *= x2
    _LGAMMA_CACHE[x0] = s - shift
    return _LGAMMA_CACHE[x0]


def _betacf(a, b, x):
    tiny, eps = _Dc("1e-300"), _Dc("1e-50")
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = _Dc(1), 1 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1 / d
    h = d
    for m in range(1, 20001):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        de = d * c
        h *= de
        if abs(de - 1) < eps:
            return h
    raise ArithmeticError("incomplete-beta continued fraction did not converge")


def ibeta(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    a, b, x = D(a), D(b), D(x)
    if x <= 0:
        return _Dc(0)
    if x >= 1:
        return _Dc(1)
    bt = (dlgamma(a + b) - dlgamma(a) - dlgamma(b) + a * x.ln() + b * (1 - x).ln()).exp()
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1 - bt * _betacf(b, a, 1 - x) / b


def t_p2(t, df):
    """Two-sided p of Student's t with df degrees of freedom (df may be fractional)."""
    t, df = D(t), D(df)
    return ibeta(df / 2, _Dc("0.5"), df / (df + t * t))


def f_sf(F, d1, d2):
    """Upper-tail P(F_{d1,d2} > F)."""
    F, d1, d2 = D(F), D(d1), D(d2)
    return ibeta(d2 / 2, d1 / 2, d2 / (d2 + d1 * F))


_TCRIT = {}


def t_crit(df, alpha="0.05"):
    """Two-sided critical value by bisection on t_p2 (220 halvings of [0, 1000])."""
    key = (D(df), D(alpha))
    if key not in _TCRIT:
        lo, hi = _Dc(0), _Dc(1000)
        for _ in range(220):
            mid = (lo + hi) / 2
            if t_p2(mid, df) > D(alpha):
                lo = mid
            else:
                hi = mid
        _TCRIT[key] = (lo + hi) / 2
    return _TCRIT[key]


def ttest1(ds):
    """One-sample (paired) t-test of the per-seed deltas against 0."""
    ds = [D(x) for x in ds]
    n = len(ds)
    m, sd = dmean(ds), dsd(ds)
    se = sd / _Dc(n).sqrt()
    t = m / se
    tc = t_crit(n - 1)
    return {"n": n, "mean": m, "sd": sd, "se": se, "t": t, "p": t_p2(t, n - 1),
            "ci": (m - tc * se, m + tc * se)}


def welch(xs, ys):
    """Welch two-sample t-test of mean(xs) - mean(ys); returns t, Satterthwaite df, p."""
    v1, v2 = dvar(xs) / len(xs), dvar(ys) / len(ys)
    t = (dmean(xs) - dmean(ys)) / (v1 + v2).sqrt()
    df = (v1 + v2) ** 2 / (v1 ** 2 / (len(xs) - 1) + v2 ** 2 / (len(ys) - 1))
    return t, df, t_p2(t, df)


def two_floor(m, fp, fl):
    """The two-floor rule: SIGNIFICANT above both floors, BELOW-FLOOR under both, else MARGINAL."""
    a = abs(D(m))
    fp, fl = D(fp), D(fl)
    return "SIGNIFICANT" if a > max(fp, fl) else ("BELOW-FLOOR" if a < min(fp, fl) else "MARGINAL")


def bh_reject(pk, qv="0.05"):
    """Benjamini-Hochberg step-up at level qv over [(p, key)]; ties broken by key."""
    s = sorted(pk, key=lambda z: (z[0], z[1]))
    m, k = len(s), 0
    for i, (p, _) in enumerate(s, 1):
        if p <= D(qv) * i / m:
            k = i
    return [key for _, key in s[:k]]


def holm_reject(pk, qv="0.05"):
    """Holm step-down at level qv over [(p, key)]; ties broken by key."""
    s = sorted(pk, key=lambda z: (z[0], z[1]))
    out = []
    for i, (p, key) in enumerate(s):
        if p <= D(qv) / (len(s) - i):
            out.append(key)
        else:
            break
    return out


def fisher_2x2(a, b, c, d):
    """Two-sided Fisher exact p for [[a, b], [c, d]], summed exactly over tables no likelier."""
    from math import comb
    r1, r2, c1 = a + b, c + d, a + c
    n = r1 + r2

    def pr(x):
        return _Fr(comb(r1, x) * comb(r2, c1 - x), comb(n, c1))
    pobs = pr(a)
    return sum(pr(x) for x in range(max(0, c1 - r2), min(r1, c1) + 1) if pr(x) <= pobs)


def load_p100_holdout():
    """Converged holdout runs (stopping_step 100, epochs_cap 3000), pooled by each record's own
    (dataset, condition, seed); a key seen twice must carry the same R@20 to 1e-12."""
    pooled, files = {}, sorted((ROOT / "results" / "phase_holdout").glob("freedom_p100*_runs.json"))
    for f in files:
        for r in json.loads(f.read_text()):
            if "test_result" not in r or r.get("stopping_step") != 100 or r.get("epochs_cap") != 3000:
                continue
            k = (r["dataset"], r["condition"], r["seed"])
            if k in pooled:
                a_, b_ = D(pooled[k]["test_result"]["Recall@20"]), D(r["test_result"]["Recall@20"])
                assert abs(a_ - b_) < _Dc("1e-12"), f"conflicting holdout records for {k}"
            pooled[k] = r
    return pooled, [f.name for f in files]


def holdout_arm(pooled, ds, cond, metric="Recall@20", base_cond="full"):
    """Paired statistics of `cond` minus `base_cond` on the seeds both arms share."""
    base = {k[2]: r for k, r in pooled.items() if k[0] == ds and k[1] == base_cond}
    arm = {k[2]: r for k, r in pooled.items() if k[0] == ds and k[1] == cond}
    seeds = sorted(set(base) & set(arm))
    if len(seeds) < 2:
        return None
    f = [D(base[s]["test_result"][metric]) for s in seeds]
    c = [D(arm[s]["test_result"][metric]) for s in seeds]
    dd = [ci - fi for ci, fi in zip(c, f)]
    tt = ttest1(dd)
    fp, fl = 2 * tt["sd"], 2 * dsd(f)
    return {"seeds": seeds, "n": len(seeds), "full": f, "arm": c, "d": dd, "mean": tt["mean"],
            "pct": 100 * tt["mean"] / dmean(f), "fp": fp, "fl": fl,
            "xfp": abs(tt["mean"]) / fp, "xfl": abs(tt["mean"]) / fl, "t": tt["t"], "p": tt["p"],
            "ci": tt["ci"], "neg": sum(1 for x in dd if x < 0), "pos": sum(1 for x in dd if x > 0),
            "verdict": two_floor(tt["mean"], fp, fl)}
# <<< exact-arith

w("# FACTS — every number the revised manuscript may cite")
w()
w("*Generated by `scripts/gen_facts.py` from the artifacts. A number not in this file is not")
w("citable. Regenerate after each experiment lands.*")
w()

# ---------- 1. exact cross-architecture knockout ----------
rows = jload("results/phase_exact/exact_crossarch.json") or []
ok = [r for r in rows if "error" not in r]
w("## 1. Exact structural knockout, cross-architecture")
w()
w(f"{len(ok)} (model,dataset) cells computed with an exact structural ablation; "
  f"{len([r for r in rows if 'error' in r])} errored.")
w()
w("| model | dataset | class | exactness | recon err | img dR@20 | txt dR@20 | MDE | img sig? |")
w("|---|---|---|---|---|---|---|---|---|")
for r in sorted(ok, key=lambda x: (x["dataset"], x["model"])):
    a = r["arms"]
    def g(k, f="dR@20"):
        return a.get(k, {}).get(f)
    i, t = g("image_knockout"), g("text_knockout")
    rec = r.get("recon_error")
    mde = r.get("MDE_R@20")
    sig = a.get("image_knockout", {}).get("significant_vs_MDE")
    w(f"| {r['model']} | {r['dataset']} | {cls(r['model'])} | {str(r['exactness'])[:24]} | "
      f"{(f'{rec:.1e}' if isinstance(rec,(int,float)) else 'n/a')} | "
      f"{(f'{i:+.6f}' if i is not None else 'n/a')} | {(f'{t:+.6f}' if t is not None else 'n/a')} | "
      f"{(f'{mde:.5f}' if mde else '—')} | {sig} |")
w()

# ---------- 1a2. adjudicable verdict summary ----------
def _verdict_block(rows):
    o = ["## 1a. Adjudicable verdicts: does image clear the cell's OWN floor?", "",
         "A cell is *adjudicable* when it has an image arm and a non-contaminated measured floor.",
         "This table SUPERSEDES the screen-based count of architectures that 'ignore image',",
         "which was produced by the input-mean screen against a single fixed band of 2.6e-3.", "",
         "| model | dataset | class | d_img | MDE | x MDE | verdict |", "|---|---|---|---|---|---|---|"]
    adj = [r for r in rows if "error" not in r and r.get("MDE_R@20")
           and "image_knockout" in r["arms"]]
    nsig = 0
    percls = {}
    for r in sorted(adj, key=lambda x: (cls(x["model"]), x["model"], x["dataset"])):
        d = r["arms"]["image_knockout"]["dR@20"]; mde = r["MDE_R@20"]; x = abs(d) / mde
        sig = x > 1; nsig += sig
        k = cls(r["model"]); a, b = percls.get(k, (0, 0)); percls[k] = (a + int(sig), b + 1)
        note = " (removing image HELPS)" if sig and d > 0 else ""
        o.append(f"| {r['model']} | {r['dataset']} | {k} | {d:+.6f} | {mde:.5f} | {x:.1f} | "
                 f"{'**SIG**' if sig else 'null'}{note} |")
    o += ["", f"**{nsig} of {len(adj)} adjudicable cells: the image knockout clears the cell's own floor.**",
          "", "By class (significant / adjudicable):", ""]
    for k in ["C1", "C2", "C1+C2", "C3", "C4"]:
        if k in percls:
            o.append(f"- {k}: {percls[k][0]}/{percls[k][1]}")
    fre = [r for r in adj if r["model"] == "freedom"]
    nf = sum(1 for r in fre if abs(r["arms"]["image_knockout"]["dR@20"]) > r["MDE_R@20"])
    o += ["", f"**FREEDOM specifically (the paper's main subject): {nf}/{len(fre)} "
          f"significant — the central claim about FREEDOM is unchanged.** What does NOT survive is",
          "the generalisation across architectures.", "",
          "Caveats that must travel with this table: two of the significant cells (DA-MRS baby and",
          "sports) have a POSITIVE delta, i.e. removing image *helps* — that is not evidence the",
          "model uses image well; several floors rest on three seeds; and VBPR/Clothing sits exactly",
          "at 1.0x on the tightest floor in the set (1.2e-4), so it is a boundary case, not a finding.",
          ""]
    return o

for _l in _verdict_block(rows):
    w(_l)

# ---------- 1c. image-dominant cells, and which are adjudicable ----------
def _imgdom_block(rows):
    out = []
    out.append("## 1c. Image-dominant cells: bare magnitude vs adjudicable")
    out.append("")
    out.append("A bare `|d_img| > |d_txt|` comparison is nearly vacuous when both effects sit below the")
    out.append("cell's own noise floor. Both counts are reported; only the second is citable as a finding.")
    out.append("")
    both = [r for r in rows if "error" not in r
            and r["arms"].get("image_knockout", {}).get("dR@20") is not None
            and r["arms"].get("text_knockout", {}).get("dR@20") is not None]
    dom, adj = [], []
    for r in both:
        i = r["arms"]["image_knockout"]["dR@20"]; t = r["arms"]["text_knockout"]["dR@20"]
        if abs(i) <= abs(t):
            continue
        mde = r.get("MDE_R@20")
        dom.append((r["model"], r["dataset"], i, t, mde))
        if mde and abs(i) > mde and abs(t) > mde:
            adj.append((r["model"], r["dataset"], i, t, mde))
    out.append(f"- Cells with both arms measured: **{len(both)}**")
    out.append(f"- Image-dominant by bare magnitude: **{len(dom)}**")
    out.append(f"- Image-dominant AND both arms clear the cell's own measured floor: **{len(adj)}**")
    out.append("")
    out.append("| model | dataset | d_img | d_txt | ratio | MDE | both clear floor? |")
    out.append("|---|---|---|---|---|---|---|")
    for m, ds, i, t, mde in sorted(dom):
        ok = bool(mde and abs(i) > mde and abs(t) > mde)
        out.append(f"| {m} | {ds} | {i:+.6f} | {t:+.6f} | {abs(i/t) if t else float('inf'):.2f} | "
                   f"{(f'{mde:.5f}' if mde else '—')} | {'**YES**' if ok else 'no' if mde else 'no floor'} |")
    out.append("")
    out.append("Restricted to the two architectures measured by exact knockout in the single-checkpoint analysis")
    out.append("(FREEDOM and LGMRec, 10 cells), LGMRec/MicroLens is the only image-dominant cell.")
    out.append("")
    return out

for _l in _imgdom_block(rows):
    w(_l)

# ---------- 1b. canonical taxonomy ----------
w("## 1b. CANONICAL content-interface taxonomy (overrides the legacy CLASS strings in scripts/exact_ko/)")
w()
w("| model | class | interface |")
w("|---|---|---|")
for m, (c, d) in sorted(TAXONOMY.items(), key=lambda x: (x[1][0], x[0])):
    w(f"| {m} | **{c}** | {d} |")
w()
w("The `class` column of every table in this file uses THIS mapping. The per-model modules in")
w("`scripts/exact_ko/` still carry the survey-era strings (`C1+C2g`, `bm3=C4`, ...); those are")
w("legacy and are not citable.")
w()

# ---------- 2. screen vs exact ----------
sv = jload("results/phase_exact/screen_vs_exact.json")
if sv:
    w("## 2. The input-mean screen vs the exact knockout")
    w()
    w("| interface class | cells | materially misreported | sign flips | offending models |")
    w("|---|---|---|---|---|")
    tot = mis = fl = 0
    regroup = {}
    for c in sv["cells"]:
        k = cls(c["model"])
        r = regroup.setdefault(k, {"n": 0, "misreported": 0, "sign_flips": 0, "cells": []})
        r["n"] += 1; r["misreported"] += int(c["materially_misreported"]); r["sign_flips"] += int(c["sign_flip"])
        if c["materially_misreported"]: r["cells"].append(f"{c['model']}/{c['dataset']}")
    sv["by_class"] = regroup
    for k in ["C1", "C2", "C1+C2", "C3", "C4", "C5"]:
        v = sv["by_class"].get(k)
        if not v: continue
        tot += v["n"]; mis += v["misreported"]; fl += v["sign_flips"]
        mem = ", ".join(sorted({c.split("/")[0] for c in v["cells"]})) or "—"
        w(f"| {k} | {v['n']} | {v['misreported']} | {v['sign_flips']} | {mem} |")
    w(f"| **total** | **{tot}** | **{mis}** ({100*mis/tot:.0f}%) | **{fl}** | |")
    w()
    w("Worst individual cells:")
    for c in sorted(sv["cells"], key=lambda x: -abs(x["exact"] - x["screen"]))[:8]:
        if not c["materially_misreported"]: continue
        w(f"- `{c['model']}/{c['dataset']}` ({c['class']}): screen {c['screen']:+.6f} -> "
          f"exact {c['exact']:+.6f}" + ("  **sign flip**" if c["sign_flip"] else "")
          + ("  **screen img==both artifact**" if c["screen_artifact_img_eq_both"] else ""))
    w()

# ---------- 3. MicroLens pre-registered verdicts ----------
w("## 3. MicroLens — pre-registered two-floor verdicts")
w()
w("| model | n seeds | stream | mean d | F_paired | F_level | x F_paired | x F_level | VERDICT |")
w("|---|---|---|---|---|---|---|---|---|")
for m in ["freedom", "lgmrec"]:
    s = jload(f"results/phase_micro/significance_{m}_microlens.json")
    if not s: continue
    for k, v in s["streams"].items():
        w(f"| {m} | {s['n_seeds']} | {k} | {v['mean_delta']:+.6f} | {v['F_paired']:.6f} | "
          f"{s['F_level']:.6f} | {v['abs_mean_over_F_paired']:.2f} | "
          f"{v['abs_mean_over_F_level']:.2f} | **{v['verdict']}** |")
    ps = jload(f"results/phase_micro/knockout_perseed_{m}_microlens.json") or []
    di = [p["d_image"] for p in ps if "d_image" in p]
    if di:
        neg = sum(1 for x in di if x < 0)
        w(f"| {m} | | *sign test* | {neg}/{len(di)} seeds negative | | | | | p={2**-len(di) if neg==len(di) else 'n/a'} |")
w()

# ---------- 4. train-time holdout ----------
allr = jload("results/phase_holdout/final_runs.json") or jload("results/phase_holdout/merged_runs.json") or []
d = collections.defaultdict(dict)
for r in allr: d[(r["dataset"], r["condition"])][r["seed"]] = r["test_result"]["Recall@20"]
w("## 4. Train-time modality holdout (retrain with the modality absent)")
w()
w("| dataset | n | condition | mean R@20 | paired d | F_paired | F_level | t (df) | verdict | sd ratio vs full |")
w("|---|---|---|---|---|---|---|---|---|---|")
train = {}
for ds in ["baby", "sports", "clothing", "microlens"]:
    full = d.get((ds, "full"), {})
    if len(full) < 2: continue
    fl = 2 * statistics.stdev(list(full.values()))
    w(f"| {ds} | {len(full)} | full | {statistics.mean(full.values()):.5f} | — | — | {fl:.5f} | — | — | — |")
    for c in ["no_image", "no_text"]:
        cv = d.get((ds, c), {})
        sh = sorted(set(cv) & set(full))
        if len(sh) < 2: continue
        dd = [cv[s] - full[s] for s in sh]
        m_, sd = statistics.mean(dd), statistics.stdev(dd)
        fp = 2 * sd; se = sd / math.sqrt(len(dd)); t = m_ / se if se else float("nan")
        hi, lo = max(fp, fl), min(fp, fl)
        v = "SIGNIFICANT" if abs(m_) > hi else "NULL" if abs(m_) < lo else "MARGINAL"
        sdr = statistics.stdev(list(cv.values())) / statistics.stdev(list(full.values()))
        train[(ds, c)] = m_
        w(f"| {ds} | {len(cv)} | {c} | {statistics.mean(cv.values()):.5f} | {m_:+.5f} | {fp:.5f} | "
          f"{fl:.5f} | {t:+.2f} ({len(dd)-1}) | **{v}** | {sdr:.1f}x |")
w()

# ---------- 5. inference pathway vs training contribution ----------
iv = jload("results/phase_holdout/inference_vs_train.json")
if iv:
    w("## 5. Inference PATHWAY vs total TRAINING contribution (FREEDOM)")
    w()
    w("| dataset | inf d_img | train d_img | ratio | inf d_txt | train d_txt | inf img/txt | train img/txt | compression |")
    w("|---|---|---|---|---|---|---|---|---|")
    for ds, r in iv.items():
        if ds.startswith("_"):
            continue
        ii, it, ti, tt = r["inf_img"], r["inf_txt"], r["train_img"], r["train_txt"]
        infr, trr = abs(ii / it), abs(ti / tt)
        w(f"| {ds} | {ii:+.5f} | {ti:+.5f} | {ti/ii:.1f}x | {it:+.5f} | {tt:+.5f} | "
          f"{infr:.3f} | {trr:.3f} | {infr/trr:.3f} |")
    w()
    w("`compression` = (inference img/txt ratio) / (training img/txt ratio). FREEDOM's hard-coded")
    w("lambda/(1-lambda) = 0.1/0.9 = 0.1111.")
    w()

# ---------- 6. alignment ----------
for tag, f in [("uniform null", "results/phase_micro/alignment_microlens.json"),
               ("co-consumption-marginal null", "results/phase_micro/alignment_degreematched.json")]:
    a = jload(f)
    if not a: continue
    w(f"## 6{'a' if 'uniform' in tag else 'b'}. Behavioural-alignment gap ({tag})")
    w()
    w("| dataset | raw img | raw txt | h_img | h_txt | cf | h_img<h_txt? |")
    w("|---|---|---|---|---|---|---|")
    for dd_ in a["datasets"]:
        if "streams" not in dd_: continue
        s = dd_["streams"]
        def c(k):
            x = s.get(k)
            return f"{x['gap_abs']:.4f} [{x['gap_abs_ci95'][0]:.4f},{x['gap_abs_ci95'][1]:.4f}]" if x else "—"
        ot = dd_["ordering_tests"].get("graph: h_img < h_txt (CIs separate)")
        w(f"| {dd_['dataset']} | {c('raw_image_cnn')} | {c('raw_text_bert')} | {c('h_img_graph')} | "
          f"{c('h_txt_graph')} | {c('cf')} | {ot} |")
    w()

# ---------- 7. floors ----------
w("## 7. Per-(model,dataset) noise floors, MDE = 2 sd of R@20")
w()
w("**TRUNCATION FLAG.** A floor built from runs that stopped early is INFLATED, and an inflated")
w("floor silently manufactures nulls. Cells flagged `CONTAMINATED` below have a best_epoch")
w("spread indicating premature early stopping (see results/phase_mde/TRUNCATION_AUDIT.md);")
w("they are **not citable** until re-measured at patience 100. Do not use them in any range,")
w("minimum, maximum or 'spans an order of magnitude' claim.")
w()
w("| model | dataset | n | mean R@20 | MDE | MDE % of mean | best-epoch range | flag | source |")
w("|---|---|---|---|---|---|---|---|---|")
byc = {}
for f in sorted(glob.glob(str(ROOT / "results/phase_mde/*_mde.json"))):
    nm = f.split("/")[-1]
    dj = json.loads(Path(f).read_text())
    eps = collections.defaultdict(list)
    for r in dj.get("runs", []):
        if "test_result" in r: eps[(r["model"], r["dataset"])].append(r.get("best_epoch"))
    for m, dd_ in dj.get("summary", {}).items():
        for ds, mm in dd_.items():
            r = mm.get("Recall@20")
            if not r: continue
            k = (m, ds)
            if k in byc and byc[k][0] >= r["n"]: continue
            e = [x for x in eps.get(k, []) if x is not None]
            byc[k] = (r["n"], r["mean"], r["MDE_2std"], (min(e), max(e)) if e else None, nm)
CONTAM = {("lattice", "baby"), ("mgcn", "baby"), ("mgcn", "clothing")}
clean = []
for (m, ds), (n, mu, mde, er, src) in sorted(byc.items()):
    ratio = er[1] / max(er[0], 1) if er else 1.0
    bad = (m, ds) in CONTAM or ratio >= 3
    if not bad:
        clean.append((100 * mde / mu, m, ds))
    w(f"| {m} | {ds} | {n} | {mu:.5f} | {mde:.5f} | {100*mde/mu:.2f}% | "
      f"{f'{er[0]}-{er[1]}' if er else '—'} | {'**CONTAMINATED**' if bad else 'ok'} | {src} |")
w()
if clean:
    clean.sort()
    w(f"**Citable floor range (CONTAMINATED cells excluded): {clean[0][0]:.2f}% "
      f"({clean[0][1]}/{clean[0][2]}) to {clean[-1][0]:.2f}% ({clean[-1][1]}/{clean[-1][2]}) "
      f"of the cell's own mean Recall@20**, over {len(clean)} clean cells.")
w()
w("The screen-based analysis judged the whole cross-architecture screen against a single fixed band")
w("of 2.6e-3. Compare that against the per-cell floors above.")
w()

# ---------- 8. rank statistic + nulls ----------
c = jload("results/phase_micro/controls_microlens.json")
if c:
    w("## 8. Continuous rank statistic at K=1000 (FREEDOM/MicroLens) and its matched-size nulls")
    w()
    w("| stream | mean dlog2(rank) | 95% CI | t | mean rank degradation | n positives |")
    w("|---|---|---|---|---|---|")
    for k, v in c.get("freedom_rank_stat", {}).items():
        w(f"| {k} | {v['mean_delta_log2_rank']:+.5f} | [{v['ci95'][0]:+.5f},{v['ci95'][1]:+.5f}] | "
          f"{v['t_stat']:+.2f} | {v['mean_rank_degradation_pct']:+.2f}% | {v['n_positives']} |")
    for k, v in c.get("freedom_rank_stat_nulls", {}).items():
        w(f"| NULL {k} | {v['mean_delta_log2_rank']:+.5f} | [{v['ci95'][0]:+.5f},{v['ci95'][1]:+.5f}] | "
          f"{v['t_stat']:+.2f} | {v['mean_rank_degradation_pct']:+.2f}% | {v['n_positives']} |")
    rs = c["freedom_rank_stat"]["image_knockout"]
    w()
    w(f"Censoring: {rs['n_censored_both_conditions']}/{rs['n_positives']} "
      f"({100*rs['frac_contributing_exactly_zero']:.1f}%) positives are capped at K in BOTH "
      f"conditions and contribute exactly zero, so the estimate is conservative.")
    w()
    w("### Allocation sweeps (audit Attack 1)")
    w()
    w("| FREEDOM lambda | " + " | ".join(f"{r['allocation_lambda']}" for r in c["freedom_allocation_sweep"]) + " |")
    w("|---|" + "---|" * len(c["freedom_allocation_sweep"]))
    w("| \\|d_img\\|/\\|d_txt\\| | " + " | ".join(f"{r['abs_ratio_img_over_txt']:.3f}" for r in c["freedom_allocation_sweep"]) + " |")
    w()
    w("| LGMRec w_img | " + " | ".join(f"{r['w_img']}" for r in c["lgmrec_allocation_sweep"]) + " |")
    w("|---|" + "---|" * len(c["lgmrec_allocation_sweep"]))
    w("| \\|d_img\\|/\\|d_txt\\| | " + " | ".join(f"{r['abs_ratio_img_over_txt']:.3f}" for r in c["lgmrec_allocation_sweep"]) + " |")
    g = c["lgmrec_gumbel_spread"]
    w()
    w(f"Gumbel-draw spread ({g['n_seeds']} draws): mean gap {g['mean_gap']:+.6f}, sd {g['sd_gap']:.6f}, "
      f"gap/sd {g['gap_over_sd']:.1f}, image-dominant in {g['n_image_dominant']}/{g['n_seeds']}.")
    w()

# ---------- 9. LGMRec cross-domain ----------
lg = jload("results/phase_micro/lgmrec_crossdomain.json")
if lg:
    w("## 9. LGMRec across five domains (identical decomposition), normalised by own base R@20")
    w()
    w("| dataset | base R@20 | image % of base | text % of base | ratio img/txt |")
    w("|---|---|---|---|---|")
    for ds, v in lg.items():
        w(f"| {ds} | {v['base_R@20']:.5f} | {v['img_pct_of_base']:.2f}% | "
          f"{v['txt_pct_of_base']:.2f}% | {v['ratio_img_over_txt']:.3f} |")
    w()

# ---------- 10. video degeneracy + frozen sweep ----------
vd = jload("results/phase_micro/video_feat_degeneracy.json")
if vd:
    w("## 10. Released MicroLens feature degeneracy (declared side-cell)")
    w()
    w("| stream | eff. rank | % of dim | mean pairwise cos |")
    w("|---|---|---|---|")
    for k, v in vd["streams"].items():
        w(f"| {k} | {v['effective_rank']:.2f} | {100*v['effective_rank_frac_of_dim']:.2f}% | "
          f"{v['mean_pairwise_cosine']:+.4f} |")
    w()
fp = jload("results/phase_micro/frozen_imageonly_penalty.json")
if fp:
    w("## 11. Frozen image-only vs text-only penalty (all five datasets)")
    w()
    w("| dataset | text-only (w=0) | image-only (w=1) | image below text |")
    w("|---|---|---|---|")
    for ds, v in sorted(fp.items(), key=lambda x: x[1]["pct_image_below_text"]):
        w(f"| {ds} | {v['text_only_w0']:.5f} | {v['image_only_w1']:.5f} | {v['pct_image_below_text']:.2f}% |")
    w()

# ---------- 12. baselines needed for disclosures ----------
w("## 12. Baselines cited in disclosures")
w()
base = {}
import os
for pat in ("results/phase_micro/knockout_microlens.json", "results/phase1/knockout_modality.json"):
    for r in (jload(pat) or []):
        if "variants" in r and "baseline" in r["variants"]:
            base[(r["model"], r["dataset"])] = r["variants"]["baseline"]["metrics"]["Recall@20"]
w("| model | dataset | R@20 (pinned checkpoint) |")
w("|---|---|---|")
for (m, ds), v in sorted(base.items()):
    w(f"| {m} | {ds} | {v:.5f} |")
if ("lightgcn", "microlens") in base and ("freedom", "microlens") in base:
    lg, fr = base[("lightgcn", "microlens")], base[("freedom", "microlens")]
    w()
    w(f"**MicroLens disclosure:** pure-CF LightGCN {lg:.5f} > LGMRec "
      f"{base.get(('lgmrec','microlens'),float('nan')):.5f} > FREEDOM {fr:.5f}; FREEDOM is "
      f"**{100*(1-fr/lg):.1f}% below pure CF**, the only one of five datasets where that holds.")
w()

# ---------- 13. patience re-measurements ----------
pc = jload("results/phase_mde/patience_control.json")
if pc:
    w("## 13. Patience-100 re-measurements (the truncation fix)")
    w()
    w("Floors and baselines re-measured with `stopping_step: 100` after the truncation audit")
    w("(results/phase_mde/TRUNCATION_AUDIT.md). These SUPERSEDE the patience-20 values in §7 for")
    w("the cells listed.")
    w()
    g = collections.defaultdict(list)
    for r in pc:
        if "test_result" in r: g[(r["model"], r["dataset"])].append(
            (r["seed"], r["test_result"]["Recall@20"], r["best_epoch"]))
    w("| model | dataset | n | mean R@20 | MDE | MDE % of mean | best-epoch range |")
    w("|---|---|---|---|---|---|---|")
    for k, v in sorted(g.items()):
        v = sorted(set(v)); vals = [x[1] for x in v]; eps = [x[2] for x in v]
        if len(vals) < 2: continue
        mu, sd = statistics.mean(vals), statistics.stdev(vals)
        w(f"| {k[0]} | {k[1]} | {len(vals)} | {mu:.5f} | {2*sd:.5f} | {200*sd/mu:.2f}% | "
          f"{min(eps)}-{max(eps)} |")
    w()

# ---------- 14. holdout: final verdicts, paired t, three-point decomposition ----------
fv = jload("results/phase_holdout/final_verdicts.json")
if fv:
    w("## 14. Train-time holdout — FINAL (8 paired seeds per cell)")
    w()
    w("| cell | d R@20 | d % | x F_paired | x F_level | verdict (two-floor) | paired t (df) | p | seeds negative |")
    w("|---|---|---|---|---|---|---|---|---|")
    for k, v in fv.items():
        if k.startswith("_"):
            continue
        w(f"| {k} | {v['d']:+.5f} | {v['d_pct']:+.2f}% | {v['x_Fp']:.2f} | {v['x_Fl']:.2f} | **{v['verdict']}** | "
          f"{v['t']:+.2f} ({v['df']}) | {v['p']:.4f} | {v['n_neg']}/{v['n']} |")
    w()
    w("Registered reading (PREREG_HOLDOUT s.3): where the two-floor verdict and the paired t-test")
    w("disagree, report the disagreement. MicroLens: the text-withheld arm is itself NULL, so by the")
    w("registered admissibility rule the MicroLens holdout has no positive control and supports no")
    w("conclusion about image.")
    w()
tp = jload("results/phase_holdout/three_point_decomposition.json")
if tp:
    w("## 15. Three-point decomposition of image's training-time contribution (3 shared seeds)")
    w()
    w("full(lambda=0.1) -> lambda=0 retrain [graph off, image aux loss ON] -> image withheld [graph AND loss off]")
    w()
    w("| dataset | seeds | total | via graph | via aux loss | aux share |")
    w("|---|---|---|---|---|---|")
    for ds, v in tp.items():
        w(f"| {ds} | {v['seeds']} | {v['total']:+.5f} | {v['via_graph']:+.5f} | {v['via_aux_loss']:+.5f} | {100*v['aux_share']:.0f}% |")
    w()
    w("Components are individually below the floors; report as a decomposition, not as significant effects.")
    w()
ps = jload("results/phase_mde/patience_summary.json")
if ps:
    w("## 16. Patience-100 re-measurement vs the submitted baseline table")
    w()
    w("| cell | p20 mean | p100 mean | change | p100 epochs | paper table | p100 vs paper |")
    w("|---|---|---|---|---|---|---|")
    for k, v in ps.items():
        w(f"| {k} | {v['mean_p20']:.5f} | {v['mean_p100']:.5f} | {100*(v['mean_p100']/v['mean_p20']-1):+.1f}% | "
          f"{v['epochs_p100'][0]}-{v['epochs_p100'][1]} | {v['paper'] if v['paper'] else '-'} | "
          f"{(f'{100*v[chr(114)+chr(101)+chr(108)+chr(95)+chr(99)+chr(104)+chr(97)+chr(110)+chr(103)+chr(101)+chr(95)+chr(118)+chr(115)+chr(95)+chr(112)+chr(97)+chr(112)+chr(101)+chr(114)]:+.1f}%' if v['paper'] else '-')} |")
    w()
    w("Several p100 runs peaked at epoch 977-999 of a 1000-epoch cap, i.e. were still improving; a")
    w("full-table re-run at patience 100 / cap 3000 is in progress (results/phase_convergence/).")
    w()

# ---------- 17. converged re-measurements ----------
ck = jload("results/phase_convergence/converged_knockout.json")
if ck:
    w("## 17. THE CORE EXPERIMENT ON CONVERGED MODELS (8 seeds, patience 100 / cap 3000)")
    w()
    w("> **REPAIRED 2026-09-21. The MicroLens rows are now citable; the earlier quarantine is**")
    w("> **lifted.** The cell had been scored partly from a checkpoint still being written (seed")
    w("> 2028 stored `base_R20` 0.09703291 against the finished run's 0.10021060), which inflated")
    w("> `F_level` 3.62x and made the image level ratio read a comfortable 0.28x. All eight")
    w("> patience-100 MicroLens checkpoints have since finished; `exp_converged_knockout.py` now")
    w("> drops any row whose baseline disagrees with the finished run and refuses seeds with no")
    w("> logged run, and the cell was re-scored over all eight. The repaired result is **larger**")
    w("> than both the contaminated five-seed figure (2.38x) and the patience-20 eight-seed figure")
    w("> the paper currently prints (1.71x): image is **2.67x the paired floor, 0.68x the level")
    w("> floor, MARGINAL, 8/8 seeds negative**. The verdict is unchanged; the effect is not. Do")
    w("> not print 1.71x as the converged number and do not print 2.38x at all.")
    w()
    w()
    w("The single-checkpoint knockout table is one checkpoint per dataset, trained at the harness")
    w("default patience 20, under which FREEDOM is 1.9-2.6% undertrained. Here the same exact")
    w("structural knockout is run on eight independently seeded CONVERGED FREEDOM models per")
    w("dataset, with both floors computed from those same eight runs.")
    w()
    w("| dataset | n | mean R@20 | F_level | stream | mean d | x F_paired | x F_level | verdict | seeds negative |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for ds, c in ck.items():
        for st, v in c["streams"].items():
            w(f"| {ds} | {c['n_seeds']} | {c['mean_R20']:.5f} | {c['F_level']:.5f} | {st} | "
              f"{v['mean_delta']:+.6f} | {v['x_F_paired']:.2f} | {v['x_F_level']:.2f} | "
              f"**{v['verdict']}** | {v['n_negative']}/{c['n_seeds']} |")
    w()
tr = jload("results/phase_convergence/table_rerun.json")
if tr:
    import re as _re
    # The paper source is only a FALLBACK for the submitted-table values (every one of the 52
    # cells resolves from a full-precision artifact below), so a checkout without the paper --
    # e.g. the code release -- must still run.
    _texf = ROOT / "paper" / "main.tex"
    _tex = _texf.read_text() if _texf.is_file() else ""
    _tab = (_tex[_tex.index("label{tab:main}"):_tex.index("end{tabular}", _tex.index("label{tab:main}"))]
            if "label{tab:main}" in _tex else "")
    _cols = ["baby", "sports", "clothing", "microlens"]; _main = {}
    for _l in _tab.split("\n"):
        _m = _re.match(r"\s*\\?t?e?x?t?b?f?\{?([A-Za-z\-]+)\}?.*?&(.*)\\\\", _l)
        if not _m: continue
        _v = [_re.sub(r"[^0-9.]", "", x) for x in _m.group(2).split("&")]
        if len(_v) == 16:
            for _i, _d in enumerate(_cols):
                try: _main[(_m.group(1).lower(), _d)] = float(_v[4*_i+1])
                except Exception: pass
    # The default-patience table's PRINTED values are 4-decimal; dividing by them mis-states the change
    # (mmgcn/clothing reads -5.9% against 0.0302 but -6.0% against the artifact's 0.03023931).
    # Prefer the full-precision artifacts the submitted numbers actually came from.
    _fp = {}
    for _r in (jload("results/phasex_crossarch/crossarch_knockout.json") or []):
        _b = (_r.get("baseline") or {}).get("Recall@20")
        if _b is not None: _fp[(_r["model"], _r["dataset"])] = _b
    for _r in (jload("results/phase0/repro_metrics.json") or []):
        _v = (_r.get("test_result") or _r.get("recomputed") or {}).get("Recall@20")
        if _v is not None: _fp.setdefault((_r.get("model"), _r.get("dataset")), _v)
    # NOTE: the MicroLens file stores 4-5 decimals, not full precision (lgmrec 0.10525 rounds
    # half-up to 0.1053 while the default-patience table printed 0.1052), so those rows are labelled 5dp.
    _ml = set()
    for _m, _v in (jload("results/bai/microlens_baselines.json") or {}).items():
        if isinstance(_v, dict) and "Recall@20" in _v:
            if (_m, "microlens") not in _fp:
                _fp[(_m, "microlens")] = _v["Recall@20"]; _ml.add((_m, "microlens"))
    w("## 18. Baseline table re-run to convergence (patience 100 / cap 3000, seed 2024)")
    w()
    w("| model/dataset | default-patience table | converged | change | src | best epoch |")
    w("|---|---|---|---|---|---|")
    rows = [r for r in tr if "test_result" in r]
    _n_fp = 0
    for r in sorted(rows, key=lambda x: (x["model"], x["dataset"])):
        k = (r["model"], r["dataset"])
        t = _fp.get(k); _src = "artifact5dp" if k in _ml else "artifact"
        if t is None:
            t = _main.get(k); _src = "printed4dp"
        else:
            _n_fp += 1
        v = r["test_result"]["Recall@20"]
        if t is None: continue
        w(f"| {r['model']}/{r['dataset']} | {t:.6f} | {v:.6f} | {100*(v/t-1):+.1f}% | {_src} | {r['best_epoch']} |")
    w()
    w("The `change` column divides by the full-precision artifact the submitted number came from,")
    w("not by the table's printed 4-decimal value (which read mmgcn/clothing as -5.9%). Rows")
    w("marked artifact5dp come from a file storing 4-5 decimals, so their change is good to")
    w("~0.1 pp, not exact.")
    w()
    w("No run hit the 3000-epoch cap.")
    w()
pc = jload("results/phase0/published_comparison.json")
if pc and tr:
    _cv = {(r["model"], r["dataset"]): r["test_result"] for r in tr if "test_result" in r}
    w("## 18a. Published-number check RECOMPUTED against the converged runs")
    w()
    w("The stored deltas in `published_comparison.json` are against the patience-20 numbers and")
    w("are superseded here; the file itself is left untouched. Source for BOTH models is the")
    w("FREEDOM paper's Table 4 (arXiv:2211.06924), i.e. LightGCN's published numbers are the")
    w("FREEDOM authors' reproduction, not LightGCN's own paper. These two are the ONLY models")
    w("for which a published number was verified, so any 'we match published numbers' claim in")
    w("the paper is scoped to them.")
    w()
    w("| model | dataset | metric | published | converged | rel |")
    w("|---|---|---|---|---|---|")
    _worst = []
    for c in pc.get("comparison", []):
        t = _cv.get((c["model"], c["dataset"]))
        if not t: continue
        for _k, _pk in [("Recall@20", "published_R20"), ("NDCG@20", "published_N20")]:
            _rel = 100 * (t[_k] - c[_pk]) / c[_pk]
            _worst.append(abs(_rel))
            w(f"| {c['model']} | {c['dataset']} | {_k} | {c[_pk]:.4f} | {t[_k]:.4f} | {_rel:+.2f}% |")
    if _worst:
        w()
        w(f"Range of |relative difference|: **{min(_worst):.2f}%** to **{max(_worst):.2f}%** "
          f"over {len(_worst)} comparisons (patience-20 range was 0.1-6.4%).")
    w()
hp = jload("results/phase_holdout/final_verdicts_p100.json")
if hp:
    w("## 19. Train-time holdout AT CONVERGENCE (patience 100 / cap 3000) — supersedes s.14")
    w()
    w("| cell | d R@20 | d % | x F_paired | x F_level | verdict | paired t (df) | p | seeds negative |")
    w("|---|---|---|---|---|---|---|---|---|")
    for k, v in hp.items():
        w(f"| {k} | {v['d']:+.5f} | {v['d_pct']:+.2f}% | {v['x_Fp']:.2f} | {v['x_Fl']:.2f} | "
          f"**{v['verdict']}** | {v['t']:+.2f} ({v['df']}) | {v['p']:.4f} | {v['n_neg']}/{v['n']} |")
    w()
    w("Compared with the patience-20 runs of s.14, the image effect SHRINKS at convergence")
    w("(Sports -1.2% -> -0.2%, Clothing -2.5% -> -0.7%, Baby -3.4% -> -3.0% and SIGNIFICANT ->")
    w("MARGINAL), confirming the audit's finding that truncation of the ablated arm inflated it.")
    w("Text is unchanged at -22% to -35%.")
    w()
    if "microlens/no_image" in hp:
        _mi, _mt = hp["microlens/no_image"], hp["microlens/no_text"]
        # image-minus-text contrast from the raw runs: R@20(no_image) - R@20(no_text) per seed,
        # judged against its own paired floor (2 sd of those per-seed differences)
        _pool19, _ = load_p100_holdout()
        _con = holdout_arm(_pool19, "microlens", "no_image", base_cond="no_text")
        _held = sorted({k.split("/")[0] for k in hp})
        _amz = ", ".join(d_ + " " + hp[d_ + "/no_image"]["verdict"]
                         for d_ in ("baby", "sports", "clothing") if d_ + "/no_image" in hp)
        w("**MicroLens (added 2026-09-23, `scripts/holdout_verdicts.py`, which first reproduces the six")
        w("Amazon entries above to 1e-12).** The converged holdout is complete on 8 paired seeds and")
        w(f"ADMISSIBLE: its text positive control clears both floors ({fx(_mt['d_pct'], 2, True)}%, "
          f"{fx(_mt['x_Fp'], 2)}x paired / {fx(_mt['x_Fl'], 2)}x level). Withholding IMAGE clears both floors")
        w(f"too: {fx(_mi['d_pct'], 2, True)}%, {fx(_mi['x_Fp'], 2)}x paired / {fx(_mi['x_Fl'], 2)}x level, "
          f"p = {sci(_mi['p'], 2)}, {_mi['n_neg']}/{_mi['n']} seeds negative -> **{_mi['verdict']}**, and it is")
        w(f"comparable to text: the per-seed image-minus-text difference is {sci(_con['mean'], 2)}, "
          f"{fx(_con['xfp'], 2)}x its paired floor ({sci(_con['fp'], 4)}; p = {fx(_con['p'], 2)}, "
          f"{_con['neg']}/{_con['n']} seeds negative), so the two arms are not distinguishable "
          "[phase_holdout/freedom_p100_microlens_runs.json].")
        _nw = {2: "two", 3: "three", 4: "four", 5: "five"}.get(len(_held), str(len(_held)))
        w(f"MicroLens is one of the {_nw} datasets held out ({', '.join(_held)}; Electronics "
          f"{'was not held out' if 'elec' not in _held else 'was held out'}); on the Amazon datasets held out "
          f"the image-withheld verdict is {_amz} (NULL = below both floors).")
        w("It supersedes the default-patience MicroLens holdout, which was inadmissible (its text arm")
        w("was below both floors).")
        w()

# ---------------------------------------------------------------- s.20 (added 2026-09-21)
# The cross-architecture EXACT adjudication the paper prints in C3, Sec.V-C, tab:exactcross and
# the Conclusion. s.1a computes the same thing on the 30-cell sweep basis; this adds the two
# Electronics cells whose floors live in phase_mde (the sweep OOM-ed after damrs/elec), giving the
# 32-cell basis the paper uses. Floors are read the way s.1a reads them (TOP-level MDE_R@20), and
# the two Elec cells take their per-dataset floors from significance_recheck.json -- NOT the
# borrowed Baby floor that phase1/knockout_modality.json stores in their MDE_R@20 field.
_ex = jload("results/phase_exact/exact_crossarch.json") or []
_sr = jload("results/phase_mde/significance_recheck.json") or {}
_km = jload("results/phase1/knockout_modality.json") or []
_sv = jload("results/phase_exact/screen_vs_exact.json") or []
if _ex:
    _rows, _err = [], 0
    for _r in _ex:
        if "error" in _r: _err += 1; continue
        _a = (_r.get("arms") or {}).get("image_knockout")
        _f = _r.get("MDE_R@20")
        if not _a or not _f: continue
        _rows.append({"m": _r["model"], "d": _r["dataset"], "dr": _a["dR@20"], "f": _f,
                      "n": _r.get("MDE_n_seeds"), "x": abs(_a["dR@20"]) / _f, "src": "sweep"})
    for _mo in ("freedom", "lgmrec"):
        _f = (((_sr.get(f"{_mo}_per_dataset_mde") or {}).get("elec") or {}).get("Recall@20") or {}).get("MDE")
        _rec = next((r for r in _km if r.get("model") == _mo and r.get("dataset") == "elec"), None)
        _d = ((_rec or {}).get("variants", {}).get("image_knockout") or {}).get("dR@20") if _rec else None
        if _f and _d is not None:
            _rows.append({"m": _mo, "d": "elec", "dr": _d, "f": _f,
                          "n": (((_sr.get(f"{_mo}_per_dataset_mde") or {})["elec"])["Recall@20"]).get("n"),
                          "x": abs(_d) / _f, "src": "phase_mde"})
    for _r in _rows:
        _r["sig"] = _r["x"] > 1
        _r["rel"] = not (_r["n"] is not None and _r["n"] < 5 and 0.5 <= _r["x"] <= 2.0)
    _sig = [r for r in _rows if r["sig"]]
    _arch = sorted({r["m"] for r in _rows})
    w("## 20. Cross-architecture EXACT adjudication, 32-cell basis (paper's C3 / Sec. V-C)")
    w()
    w(f"- `exact_crossarch.json`: **{len(_ex)}** records, **{_err}** with an `error` key; "
      f"**{len([r for r in _rows if r['src'] == 'sweep'])}** adjudicable there, plus the two "
      "Electronics cells from `phase1/knockout_modality.json` judged against their own "
      "`significance_recheck.json` floors.")
    w(f"- **Basis: {len(_rows)} adjudicable cells. Image clears its own floor in {len(_sig)}, "
      f"over {len({r['m'] for r in _sig})} of {len(_arch)} architectures with an image arm.**")
    w()
    w("| model | cells | clears | which |")
    w("|---|---|---|---|")
    for _m in _arch:
        _c = [r for r in _rows if r["m"] == _m]; _s = [r for r in _c if r["sig"]]
        w(f"| {_m} | {len(_c)} | **{len(_s)}** | {', '.join(sorted(r['d'] for r in _s)) or '--'} |")
    w()
    _pos = [r for r in _sig if r["dr"] > 0]
    # (joined outside the f-strings: nesting the same quote inside an f-string needs Python 3.12,
    # and this file must run on the 3.11 mechinterp interpreter too; output is unchanged)
    _pos_s = "; ".join("%s/%s %+.6f" % (r["m"], r["d"], r["dr"]) for r in _pos) or "none"
    w(f"- **Positive deltas among the {len(_sig)}: {len(_pos)}** "
      f"({_pos_s}). Removing image "
      "RAISES Recall@20 there -- evidence image is net-harmful, NOT that the model uses it.")
    _us = [r for r in _sig if r["rel"] is False]
    _un = [r for r in _rows if not r["sig"] and r["rel"] is False]
    _us_s = "; ".join("%s/%s %.2fx" % (r["m"], r["d"], r["x"]) for r in _us)
    _un_s = "; ".join("%s/%s %.2fx" % (r["m"], r["d"], r["x"]) for r in _un)
    w(f"- **Reliability flags, BOTH directions:** borderline (0.5-2x a floor of fewer than 5 seeds) on {len(_us)} of the clearing "
      f"({_us_s}) and {len(_un)} of the "
      f"non-clearing ({_un_s}). FREEDOM/baby "
      "is in the second set: disclosing only the first is the flattering half.")
    _n3 = sum(1 for r in _rows if r["n"] == 3)
    w(f"- **Floor thinness:** {_n3} of {len(_rows)} rest on n=3; distribution "
      f"{dict(sorted({r['n']: sum(1 for q in _rows if q['n'] == r['n']) for r in _rows}.items(), key=lambda kv: (kv[0] is None, kv[0])))}.")
    _B = 2.6e-3
    _rt = sorted((_B / r["f"], f"{r['m']}/{r['d']}") for r in _rows)
    w(f"- **Band vs floor:** the screen's single fixed band {_B:.1e} is {_rt[0][0]:.2f}x "
      f"the cell's own floor at one extreme ({_rt[0][1]}) and {_rt[-1][0]:.1f}x at the other "
      f"({_rt[-1][1]}); looser than the cell's own floor in {sum(1 for x, _ in _rt if x > 1)} of {len(_rt)}.")
    w()
if _sv:
    _c = _sv if isinstance(_sv, list) else _sv.get("cells", [])
    def _g(r, *k):
        for kk in k:
            if kk in r: return r[kk]
        return None
    _pairs = [(_g(r, "screen_dR@20", "screen"), _g(r, "exact_dR@20", "exact"), r.get("model"), r.get("dataset"))
              for r in _c]
    _pairs = [(a, b, m, d) for a, b, m, d in _pairs if a is not None and b is not None]
    _lg = sum(1 for a, b, *_ in _pairs if abs(a) > abs(b))
    _sm = sum(1 for a, b, *_ in _pairs if abs(a) < abs(b))
    _eq = sum(1 for a, b, *_ in _pairs if abs(a) == abs(b))
    _fl = sum(1 for a, b, *_ in _pairs if a * b < 0)
    _BAND = 2.6e-3
    _dis = [(m, d, a, b) for a, b, m, d in _pairs if (abs(a) > _BAND) != (abs(b) > _BAND)]
    _inside = [x for x in _dis if abs(x[2]) <= _BAND]
    w("## 20a. Screen vs exact: the direction of the error (refutes 'the screen over-states image')")
    w()
    w(f"- Cells measured both ways: **{len(_pairs)}**.")
    w(f"- |screen| **larger in {_lg}**, **smaller in {_sm}**, identical in {_eq}; **{_fl} sign flips**.")
    w(f"- Band disagreements: **{len(_dis)}**, of which **{len(_inside)}** put the cell wrongly "
      f"INSIDE the {_BAND:.1e} band. The earlier claim that the screen over-states image, "
      "so a within-band reading is conservative, is therefore false as a general statement.")
    w()
# ===================================================================== s.21 (added 2026-09-23)
# Additional certified numbers. Reads results/ only. Every printed value is recomputed here from the
# raw artifact named by the [TAG] on its line, in Decimal (exact-arith block at the top), and is
# rounded half-up once, at print time. The only literals below are rule parameters (alpha = q =
# 0.05, RBO persistence 0.9 at depth 20, the degenerate-floor threshold) and model-group metadata;
# each is labelled where it is defined.
_S21 = [
    ("CK", "results/phase_convergence/converged_knockout.json"),
    ("HO", "results/phase_holdout/freedom_p100*_runs.json (records with stopping_step 100, epochs_cap 3000)"),
    ("HV", "results/phase_holdout/final_verdicts_p100.json (derived from HO; cross-check only)"),
    ("HD", "results/phase_holdout/final_runs.json (default-patience holdout)"),
    ("EX", "results/phase_exact/exact_crossarch.json"),
    ("SV", "results/phase_exact/screen_vs_exact.json"),
    ("K1", "results/phase1/knockout_modality.json"),
    ("KF", "results/phase1/knockout_finer.json"),
    ("SR", "results/phase_mde/significance_recheck.json"),
    ("MDE", "results/phase_mde/*_mde.json (a cell's floor runs = the summary entry whose MDE equals the cell's floor exactly)"),
    ("OP", "results/phase_mde/freedom_opgpoint_mde.json"),
    ("MM", "results/phase_mde/microlens_mde.json"),
    ("PC", "results/phase_mde/patience_control.json"),
    ("TR", "results/phase_convergence/table_rerun.json"),
    ("AL", "results/phase_align/alignment_stats.json"),
    ("AM", "results/phase_micro/alignment_microlens.json"),
    ("AD", "results/phase_micro/alignment_degreematched.json"),
    ("CL", "results/trackA_clip/clip_freedom.json"),
    ("WF", "results/phase2/weight_sweep_frozen.json"),
    ("WM", "results/phase_micro/weightsweep_microlens.json"),
    ("WR", "results/phase2/weight_sweep_retrain.json"),
    ("RM", "results/phase0/repro_metrics.json"),
    ("PB", "results/phase0/published_comparison.json"),
    ("CX", "results/phasex_crossarch/crossarch_knockout.json"),
    ("SF", "results/phase_micro/significance_freedom_microlens.json"),
    ("KM", "results/phase_micro/knockout_microlens.json"),
    ("PR", "results/phase_holdout/PREREG_HOLDOUT.md"),
    ("PM", "results/phase_micro/PREREG.md"),
    ("CM", "results/phase_micro/controls_microlens.json"),
    ("VD", "results/phase_micro/video_feat_degeneracy.json"),
    ("DS", "results/phase0/dataset_stats.json"),
    ("CS", "results/_scratch/exact_ko/cohesion_baby_np*.json"),
]


def _rj(rel):
    return jload("results/" + rel)


def _src_row(*cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


_DS4 = ["baby", "sports", "clothing", "microlens"]
_AMZ3 = ["baby", "sports", "clothing"]
w("## 21. Additional certified numbers")
w()
w("Recomputed from the raw artifacts on every run, in decimal arithmetic (Decimal(repr(x))), and")
w("rounded half-up once at print time, so a 1-decimal figure below is never a re-rounding of a")
w("2-decimal one. p-values are two-sided unless marked and come from the stdlib Student-t / F /")
w("incomplete-beta code at the top of this file. Every line or table row names its artifact by tag:")
w()
for _t, _p in _S21:
    w(f"- `[{_t}]` {_p}")
w()
w("Two-floor rule: SIGNIFICANT if |d| exceeds both floors, BELOW-FLOOR if it is under both (the")
w("older artifacts store this as NULL), MARGINAL otherwise. F_l = 2 sd of R@20 across seeds (level")
w("floor); F_p = 2 sd of the per-seed deltas (paired floor). `[math]` marks a line that is a pure")
w("function of stated parameters.")
w()

# ------------------------------------------------------------------ 21a converged deletion
_CK = _rj("phase_convergence/converged_knockout.json") or {}
_HO, _HO_FILES = load_p100_holdout()
_DEL = {}
if _CK:
    w("### 21a. Deletion on the eight converged models per dataset (tab:freedom deletion half; R3, R21)")
    w()
    _nchk, _mx, _bad = 0, _Dc(0), []
    for _ds in _DS4:
        for _r in (_CK.get(_ds) or {}).get("per_seed", []):
            _h = _HO.get((_ds, "full", _r["seed"]))
            if _h is None:
                _bad.append(f"{_ds}/s{_r['seed']} (no holdout run)")
                continue
            _nchk += 1
            _dd = abs(D(_r["base_R20"]) - D(_h["test_result"]["Recall@20"]))
            _mx = max(_mx, _dd)
            if _dd > _Dc("1e-12") or Path(_h["ckpt_path"]).name != _r["ckpt"]:
                _bad.append(f"{_ds}/s{_r['seed']}")
    w(f"- The deletion baselines are the holdout's full-arm runs: {_nchk} seed-level pairs compared, max "
      f"|R@20 difference| {sci(_mx, 2)}, checkpoint file names compared too; mismatches: "
      f"{', '.join(_bad) or 'none'} [CK+HO].")
    w()
    w("| dataset | stream | mean d | d % | d % (1 dp) | 95% CI of d, % of R@20 | x F_l | x F_p | F_l/F_p | "
      "t(7) | p | p (table) | d<0 | d>0 | mean RBO@20 | verdict | src |")
    w("|" + "---|" * 17)
    for _ds in _DS4:
        _c = _CK.get(_ds)
        if not _c:
            continue
        _ps = _c["per_seed"]
        _base = [D(r["base_R20"]) for r in _ps]
        _mu, _fl = dmean(_base), 2 * dsd(_base)
        for _st in ("image", "text", "both"):
            _dd = [D(r["d_" + _st]) for r in _ps]
            _tt = ttest1(_dd)
            _fp = 2 * _tt["sd"]
            _rb = dmean([r["rbo_" + _st] for r in _ps]) if all(("rbo_" + _st) in r for r in _ps) else None
            _rec = {"mean": _tt["mean"], "pct": 100 * _tt["mean"] / _mu, "fl": _fl, "fp": _fp,
                    "xfl": abs(_tt["mean"]) / _fl, "xfp": abs(_tt["mean"]) / _fp, "t": _tt["t"],
                    "p": _tt["p"], "ci_pct": tuple(100 * x / _mu for x in _tt["ci"]), "mu": _mu,
                    "neg": sum(1 for x in _dd if x < 0), "pos": sum(1 for x in _dd if x > 0),
                    "rbo": _rb, "verdict": two_floor(_tt["mean"], _fp, _fl), "d": _dd,
                    "base": _base, "n": len(_dd), "seeds": [r["seed"] for r in _ps]}
            _DEL[(_ds, _st)] = _rec
            w(_src_row(_ds, _st, sci(_rec["mean"], 3), fx(_rec["pct"], 2, True), fx(_rec["pct"], 1, True),
                       f"[{fx(_rec['ci_pct'][0], 2, True)}, {fx(_rec['ci_pct'][1], 2, True)}]",
                       fx(_rec["xfl"], 2), fx(_rec["xfp"], 2), fx(_fl / _fp, 2), fx(_rec["t"], 2, True),
                       sci(_rec["p"], 2), ptab(_rec["p"]), f"{_rec['neg']}/{_rec['n']}",
                       f"{_rec['pos']}/{_rec['n']}", fx(_rb, 3) if _rb is not None else "--",
                       _rec["verdict"], "CK"))
    w()
    _mxd, _vdis = _Dc(0), []
    for (_ds, _st), _rec in _DEL.items():
        _s = ((_CK.get(_ds) or {}).get("streams") or {}).get(_st)
        if not _s:
            continue
        _mxd = max(_mxd, abs(_rec["mean"] - D(_s["mean_delta"])), abs(_rec["fp"] - D(_s["F_paired"])),
                   abs(_rec["fl"] - D(_CK[_ds]["F_level"])))
        if {"NULL": "BELOW-FLOOR"}.get(_s["verdict"], _s["verdict"]) != _rec["verdict"]:
            _vdis.append(f"{_ds}/{_st}")
    w(f"- Recomputed vs the stored `streams`: max |difference| over mean d, F_p, F_l = {sci(_mxd, 2)}; "
      f"verdict disagreements: {', '.join(_vdis) or 'none'} [CK].")
    _rbo_ceiling = 1 - _Dc("0.9") ** 20
    _wf0 = _rj("phase2/weight_sweep_frozen.json") or []
    _ident = sorted({fx(s["rbo_vs_trained"], 6) for c in _wf0 for s in c["sweep"]
                     if s["weight"] == c["trained_weight"]})
    w(f"- RBO of two identical top-20 lists at persistence 0.9 = 1 - 0.9^20 = {fx(_rbo_ceiling, 6)} "
      f"(3 dp: {fx(_rbo_ceiling, 3)}) [math]; the implementation's own identical-list value "
      f"(rbo_vs_trained at the trained weight): {', '.join(_ident) or 'n/a'} [WF].")
    _k1 = _rj("phase1/knockout_modality.json") or []
    _ns = sorted({((r.get("variants", {}).get(v) or {}).get("ranking_change") or {}).get("n_sampled")
                  for r in _k1 if r.get("model") == "freedom"
                  for v in ("image_knockout", "text_knockout")} - {None})
    _has_ns = any(("n_sampled" in r or "rbo_n_sampled" in r)
                  for _ds in _DS4 for r in (_CK.get(_ds) or {}).get("per_seed", []))
    w(f"- RBO sample size: CK's per-seed records {'record' if _has_ns else 'do NOT record'} how many users the "
      f"RBO averages over; the default-patience FREEDOM records written by the same run_model -> "
      f"ranking_change path record n_sampled = {', '.join(str(x) for x in _ns) or 'none'} [CK, K1]. "
      "The converged sample size is therefore not certified by CK itself.")
    _recs = [D(r["recon_max_err"]) for _ds in _DS4 for r in (_CK.get(_ds) or {}).get("per_seed", [])]
    if _recs:
        w(f"- Embedding reconstruction on the {len(_recs)} converged models: max recon_max_err "
          f"{sci(max(_recs), 2)} [CK].")
    _ext = []
    for _ds in _AMZ3:
        if (_ds, "image") not in _DEL:
            continue
        _rec = _DEL[(_ds, "image")]
        _i = max(range(_rec["n"]), key=lambda i: (abs(_rec["d"][i]), -i))
        _ext.append((abs(_rec["d"][_i]), _ds, _rec["seeds"][_i], _rec["d"][_i], _rec["base"][_i]))
    if _ext:
        w("- Largest single-seed |image deletion| on the converged Amazon models: " + "; ".join(
            f"{e[1]} {sci(e[3], 2)} (s{e[2]}, {fx(100 * e[3] / e[4], 2, True)}% of that seed's R@20)"
            for e in _ext) + f"; range of these maxima {sci(min(e[0] for e in _ext), 2)} to "
            f"{sci(max(e[0] for e in _ext), 2)} [CK].")
    w("- Level floor as a share of mean R@20: " + "; ".join(
        f"{_ds} {sci(_DEL[(_ds, 'image')]['fl'], 3)} = {fx(100 * _DEL[(_ds, 'image')]['fl'] / _DEL[(_ds, 'image')]['mu'], 2)}% "
        f"of {fx(_DEL[(_ds, 'image')]['mu'], 4)}" for _ds in _DS4 if (_ds, "image") in _DEL) + " [CK].")
    w(f"- A pure-noise delta judged against a level floor from n seeds exceeds it with probability "
      f"P(|t_(n-1)| > 2): n=2 {fx(t_p2(2, 1), 3)}, n=3 {fx(t_p2(2, 2), 3)}, n=8 {fx(t_p2(2, 7), 3)} [math].")
    w()

# ------------------------------------------------------------------ 21b converged holdout
_HOLD = {}
_HOLDMETS = ("Recall@20", "NDCG@20", "Recall@10")
if _HO:
    w("### 21b. Retraining without a modality, converged (tab:freedom retraining half; E.1; R18, R23, R34)")
    w()
    w(f"- Run files read: {', '.join(_HO_FILES)}; {len(_HO)} runs kept (stopping_step 100, epochs_cap "
      "3000), paired on seed [HO].")
    for _ds in _DS4:
        for _cond in ("no_image", "no_text"):
            for _met in _HOLDMETS:
                _a = holdout_arm(_HO, _ds, _cond, _met)
                if _a:
                    _HOLD[(_ds, _cond, _met)] = _a
    _hv = _rj("phase_holdout/final_verdicts_p100.json") or {}
    _hvd, _hvv = _Dc(0), []
    for _k, _v in _hv.items():
        _ds, _cond = _k.split("/")
        _a = _HOLD.get((_ds, _cond, "Recall@20"))
        if not _a:
            _hvv.append(f"{_k} (not recomputable)")
            continue
        _hvd = max(_hvd, abs(_a["mean"] - D(_v["d"])), abs(_a["fp"] - D(_v["F_paired"])),
                   abs(_a["fl"] - D(_v["F_level"])), abs(_a["p"] - D(_v["p"])))
        if {"NULL": "BELOW-FLOOR"}.get(_v["verdict"], _v["verdict"]) != _a["verdict"]:
            _hvv.append(_k)
    w(f"- Recomputed vs HV: {len(_hv)} cells, max |difference| over d, F_p, F_l, p = {sci(_hvd, 2)}; "
      f"verdict disagreements: {', '.join(_hvv) or 'none'} [HO, HV].")
    w()
    w("| dataset | arm | metric | mean d | d % | d % (1 dp) | x F_l | x F_p | t(7) | p | p (table) | d<0 | verdict | src |")
    w("|" + "---|" * 14)
    for _ds in _DS4:
        for _cond in ("no_image", "no_text"):
            for _met in _HOLDMETS:
                _a = _HOLD.get((_ds, _cond, _met))
                if not _a:
                    continue
                w(_src_row(_ds, _cond, _met, sci(_a["mean"], 3), fx(_a["pct"], 2, True), fx(_a["pct"], 1, True),
                           fx(_a["xfl"], 2), fx(_a["xfp"], 2), fx(_a["t"], 2, True), sci(_a["p"], 2),
                           ptab(_a["p"]), f"{_a['neg']}/{_a['n']}", _a["verdict"], "HO"))
    w()
    w("Image-minus-text contrast, per seed R(no_image) - R(no_text), judged against its own paired floor")
    w("(2 sd of those per-seed differences); d<0 counts seeds where the image-withheld arm scored lower:")
    w()
    w("| dataset | metric | mean difference | its F_p | x F_p | t(7) | p | p (3 sf) | d<0 | src |")
    w("|" + "---|" * 10)
    _CON = {}
    for _ds in _DS4:
        for _met in _HOLDMETS:
            _c2 = holdout_arm(_HO, _ds, "no_image", _met, base_cond="no_text")
            if not _c2:
                continue
            _CON[(_ds, _met)] = _c2
            w(_src_row(_ds, _met, sci(_c2["mean"], 3), sci(_c2["fp"], 4), fx(_c2["xfp"], 2),
                       fx(_c2["t"], 2, True), ptab(_c2["p"]), sci(_c2["p"], 3), f"{_c2['neg']}/{_c2['n']}", "HO"))
    w()
    _IOT = {}
    for _ds in _DS4:
        _ai, _at = _HOLD.get((_ds, "no_image", "Recall@20")), _HOLD.get((_ds, "no_text", "Recall@20"))
        if _ai and _at:
            _IOT[_ds] = 100 * (dmean(_at["arm"]) - dmean(_ai["arm"])) / dmean(_ai["arm"])
    w("- Image-only (text withheld) vs text-only (image withheld) retrained models, mean R@20, % relative "
      "to text-only: " + "; ".join(f"{_ds} {fx(v, 2, True)}% (1 dp {fx(v, 1, True)}; 0 dp {fx(v, 0, True)})"
                                  for _ds, v in _IOT.items()) + " [HO].")
    w("- Arm means, mean R@20 over the 8 seeds (full / image-withheld / text-withheld): " + "; ".join(
        f"{_ds} {fx(dmean(_HOLD[(_ds, 'no_image', 'Recall@20')]['full']), 6)} / "
        f"{fx(dmean(_HOLD[(_ds, 'no_image', 'Recall@20')]['arm']), 6)} / {fx(dmean(_HOLD[(_ds, 'no_text', 'Recall@20')]['arm']), 6)}"
        for _ds in _DS4 if (_ds, "no_image", "Recall@20") in _HOLD and (_ds, "no_text", "Recall@20") in _HOLD) + " [HO].")
    if all(_d in _IOT for _d in _AMZ3):
        _am = [abs(_IOT[_d]) for _d in _AMZ3]
        w(f"- Amazon image-only deficit range: {fx(min(_am), 1)}-{fx(max(_am), 1)}% (0 dp {fx(min(_am), 0)}-"
          f"{fx(max(_am), 0)}%) [HO].")
    w()
    w("| dataset | arm | n | median best epoch | min | max | arm longer than its full partner | last logged epoch - best epoch | max last epoch | epochs cap | src |")
    w("|" + "---|" * 11)
    for _ds in _DS4:
        _full = {k[2]: r for k, r in _HO.items() if k[0] == _ds and k[1] == "full"}
        for _cond in ("full", "no_image", "no_text"):
            _arm = {k[2]: r for k, r in _HO.items() if k[0] == _ds and k[1] == _cond}
            if not _arm:
                continue
            _ss = sorted(_arm)
            _be = [_arm[s]["best_epoch"] for s in _ss]
            _last = [_arm[s]["valid_curve"][-1][0] for s in _ss]
            _gaps = sorted({l_ - b_ for l_, b_ in zip(_last, _be)})
            _lon = ("--" if _cond == "full" else
                    f"{sum(1 for s in _ss if s in _full and _arm[s]['best_epoch'] > _full[s]['best_epoch'])}/{len(_ss)}")
            _caps = sorted({_arm[s].get("epochs_cap") for s in _ss})
            w(_src_row(_ds, _cond, len(_ss), format(dmedian(_be), "f"), min(_be), max(_be), _lon,
                       ", ".join(str(g) for g in _gaps), max(_last), ", ".join(str(c) for c in _caps), "HO"))
    w()
    _prh = (ROOT / "results" / "phase_holdout" / "PREREG_HOLDOUT.md")
    _prt = _prh.read_text() if _prh.is_file() else ""
    _m = re.search(r"sd\(no_image runs\) / sd\(full runs\) = ([0-9.]+)` on (\w+) \(([0-9.]+) vs ([0-9.]+)\)", _prt)
    if _m:
        w(f"- Registered H-var value, quoted from the artifact: sd(no_image)/sd(full) = {_m.group(1)} on "
          f"{_m.group(2)} ({_m.group(3)} vs {_m.group(4)}) [PR].")
    else:
        w("- Registered H-var value: pattern not found [PR].")
    w("- H-var at convergence, sd(arm R@20)/sd(full R@20) over the 8 seeds: " + "; ".join(
        f"{_ds} image-withheld {fx(dsd(_HOLD[(_ds, 'no_image', 'Recall@20')]['arm']) / dsd(_HOLD[(_ds, 'no_image', 'Recall@20')]['full']), 2)}, "
        f"text-withheld {fx(dsd(_HOLD[(_ds, 'no_text', 'Recall@20')]['arm']) / dsd(_HOLD[(_ds, 'no_text', 'Recall@20')]['full']), 2)}"
        for _ds in _DS4 if (_ds, "no_image", "Recall@20") in _HOLD) + " [HO].")
    w()
    w("| dataset | arm | Pearson r(full, arm) over seeds | Welch t | Welch df | Welch p | paired p | same call at 0.05 | src |")
    w("|" + "---|" * 9)
    _rs, _rsi = [], []
    for _ds in _DS4:
        for _cond in ("no_image", "no_text"):
            _a = _HOLD.get((_ds, _cond, "Recall@20"))
            if not _a:
                continue
            _r = dcorr(_a["full"], _a["arm"])
            _rs.append(_r)
            if _cond == "no_image":
                _rsi.append(_r)
            _wt, _wdf, _wp = welch(_a["arm"], _a["full"])
            _same = (_wp < _Dc("0.05")) == (_a["p"] < _Dc("0.05"))
            w(_src_row(_ds, _cond, fx(_r, 2, True), fx(_wt, 2, True), fx(_wdf, 2), sci(_wp, 2),
                       sci(_a["p"], 2), "yes" if _same else "NO", "HO"))
    w()
    if _rs:
        w(f"- Pearson r range: image-withheld arms {fx(min(_rsi), 2, True)} to {fx(max(_rsi), 2, True)}; "
          f"all eight arms {fx(min(_rs), 2, True)} to {fx(max(_rs), 2, True)} [HO].")
    w()

# ------------------------------------------------------------------ 21c default-patience holdout
_HDr = _rj("phase_holdout/final_runs.json") or []
_HD = {}
for _r in _HDr:
    if "test_result" not in _r:
        continue
    _k = (_r["dataset"], _r["condition"], _r["seed"])
    if _k in _HD:
        assert abs(D(_HD[_k]["test_result"]["Recall@20"]) - D(_r["test_result"]["Recall@20"])) < _Dc("1e-12"), _k
    _HD[_k] = _r
_HDA = {}
if _HD:
    w("### 21c. Retraining without a modality at the default patience (6.2 protocol comparison; E.1)")
    w()
    w("| dataset | arm | d % | x F_l | x F_p | verdict | F_l | median best epoch, arm | median best epoch, full | src |")
    w("|" + "---|" * 10)
    for _ds in _DS4:
        _fb = [_HD[k]["best_epoch"] for k in sorted(_HD) if k[0] == _ds and k[1] == "full"]
        for _cond in ("no_image", "no_text"):
            _a = holdout_arm(_HD, _ds, _cond)
            if not _a:
                continue
            _HDA[(_ds, _cond)] = _a
            _be = [_HD[(_ds, _cond, s)]["best_epoch"] for s in _a["seeds"]]
            w(_src_row(_ds, _cond, fx(_a["pct"], 2, True), fx(_a["xfl"], 2), fx(_a["xfp"], 2), _a["verdict"],
                       sci(_a["fl"], 3), format(dmedian(_be), "f"), format(dmedian(_fb), "f"), "HD"))
    w()
    _chg = []
    for _ds in _DS4:
        _o, _nw = _HDA.get((_ds, "no_image")), _HOLD.get((_ds, "no_image", "Recall@20"))
        if _o and _nw:
            _chg.append(f"{_ds} {fx(_o['pct'], 2, True)}% -> {fx(_nw['pct'], 2, True)}% "
                        f"(1 dp {fx(_o['pct'], 1, True)} -> {fx(_nw['pct'], 1, True)}; {_o['verdict']} -> {_nw['verdict']})")
    w("- Image-withheld effect, default patience -> converged: " + "; ".join(_chg) + " [HD, HO].")
    _chg = []
    for _ds in _DS4:
        _o, _nw = _HDA.get((_ds, "no_text")), _HOLD.get((_ds, "no_text", "Recall@20"))
        if _o and _nw:
            _chg.append(f"{_ds} {fx(_o['pct'], 2, True)}% -> {fx(_nw['pct'], 2, True)}% ({_o['verdict']} -> {_nw['verdict']})")
    w("- Text-withheld effect, default patience -> converged: " + "; ".join(_chg) + " [HD, HO].")
    _chg = []
    for _ds in _DS4:
        _o, _nw = _HDA.get((_ds, "no_image")), _HOLD.get((_ds, "no_image", "Recall@20"))
        if _o and _nw:
            _chg.append(f"{_ds} {sci(_o['fl'], 3)} -> {sci(_nw['fl'], 3)}")
    w("- Holdout level floor F_l, default patience -> converged: " + "; ".join(_chg) + " [HD, HO].")
    w()

# ------------------------------------------------------------------ 21d floors by protocol
_OP = _rj("phase_mde/freedom_opgpoint_mde.json") or {}
_SR = _rj("phase_mde/significance_recheck.json") or {}
_MM = _rj("phase_mde/microlens_mde.json") or {}
_FLOORS = {}   # (ds, label) -> (value, n, mean, tag)
for _ds, _mm in ((_OP.get("summary") or {}).get("freedom") or {}).items():
    _r = _mm.get("Recall@20") or {}
    if _r.get("MDE_2std") is not None:
        _FLOORS[(_ds, f"default, {_r['n']} seeds")] = (D(_r["MDE_2std"]), _r["n"], D(_r["mean"]), "OP")
_srb = ((((_SR.get("freedom_per_dataset_mde") or {}).get("baby") or {}).get("Recall@20")) or {})
if _srb.get("MDE") is not None:
    _FLOORS[("baby", f"default, {_srb['n']} seeds")] = (D(_srb["MDE"]), _srb["n"], D(_srb["mean"]), "SR")
for _ds in _DS4:
    _f = [D(_HD[k]["test_result"]["Recall@20"]) for k in sorted(_HD) if k[0] == _ds and k[1] == "full"]
    if len(_f) > 1:
        _FLOORS[(_ds, f"default, {len(_f)} seeds (holdout full arm)")] = (2 * dsd(_f), len(_f), dmean(_f), "HD")
for _ds, _mm in ((_MM.get("summary") or {}).get("freedom") or {}).items():
    _r = _mm.get("Recall@20") or {}
    if _r.get("MDE_2std") is not None:
        _FLOORS[(_ds, f"default, {_r['n']} seeds")] = (D(_r["MDE_2std"]), _r["n"], D(_r["mean"]), "MM")
for _ds in _DS4:
    if (_ds, "image") in _DEL:
        _FLOORS[(_ds, "converged, 8 seeds")] = (_DEL[(_ds, "image")]["fl"], _DEL[(_ds, "image")]["n"],
                                                _DEL[(_ds, "image")]["mu"], "CK")
if _FLOORS:
    w("### 21d. FREEDOM seed-noise floors by training protocol (6.2; R35)")
    w()
    w("| dataset | protocol | n | F_l | F_l % of mean R@20 | src |")
    w("|---|---|---|---|---|---|")
    for (_ds, _lab), (_v, _n, _mu, _tag) in sorted(_FLOORS.items(), key=lambda kv: (_DS4.index(kv[0][0]) if kv[0][0] in _DS4 else 9, kv[0][0], kv[0][1])):
        w(_src_row(_ds, _lab, _n, sci(_v, 3), fx(100 * _v / _mu, 2), _tag))
    w()
    _mmh = _FLOORS.get(("microlens", "default, 8 seeds (holdout full arm)"))
    _mmm = _FLOORS.get(("microlens", "default, 8 seeds"))
    if _mmh and _mmm:
        w(f"- The MicroLens default-patience floor in MM and the holdout's default full arm agree to "
          f"{sci(abs(_mmh[0] - _mmm[0]), 2)} [MM, HD].")
    _bfl = [(k[1], v) for k, v in _FLOORS.items() if k[0] == "baby" and k[1].startswith("default")]
    for _num in ("sports", "clothing"):
        for _nl, _nv in [(k[1], v) for k, v in _FLOORS.items() if k[0] == _num and k[1].startswith("default")]:
            for _bl, _bv in _bfl:
                _ratio = _nv[0] / _bv[0]
                _F = _ratio ** 2
                _p1 = f_sf(_F, _nv[1] - 1, _bv[1] - 1)
                _p2 = 2 * min(_p1, 1 - _p1)
                w(f"- {_num} [{_nl}] / baby [{_bl}] floor = {fx(_ratio, 2)}x; variance-ratio F({_nv[1] - 1},{_bv[1] - 1}) = "
                  f"{fx(_F, 2)}, one-sided p {sci(_p1, 2)}, two-sided p {sci(_p2, 2)} [{_nv[3]}, {_bv[3]}].")
    _conv = {k[0]: v for k, v in _FLOORS.items() if k[1].startswith("converged")}
    if _conv:
        _t = min(_conv, key=lambda d_: _conv[d_][0])
        w(f"- Converged floors: " + ", ".join(f"{d_} {sci(v[0], 3)}" for d_, v in _conv.items())
          + f"; tightest: {_t} [CK].")
    _ex0 = _rj("phase_exact/exact_crossarch.json") or []
    _fbx = next((r for r in _ex0 if r.get("model") == "freedom" and r.get("dataset") == "baby"), None)
    _fbk = next((r for r in (_rj("phase1/knockout_modality.json") or [])
                 if r.get("model") == "freedom" and r.get("dataset") == "baby"), None)
    _wfb = next((c for c in (_rj("phase2/weight_sweep_frozen.json") or []) if c.get("dataset") == "baby"), None)
    _lab = {str(v[0]): k[1] for k, v in _FLOORS.items() if k[0] == "baby"}
    w("- Which FREEDOM/Baby floor each artifact judges against: "
      + (f"EX {sci(_fbx['MDE_R@20'], 3)} ({_fbx.get('MDE_source')}); " if _fbx else "")
      + (f"K1 {sci(_fbk['MDE_R@20'], 3)} (= the {_lab.get(str(D(_fbk['MDE_R@20'])), '?')} floor); " if _fbk else "")
      + (f"WF {sci(_wfb['MDE_R@20'], 3)} (= the {_lab.get(str(D(_wfb['MDE_R@20'])), '?')} floor); " if _wfb else "")
      + "CK the converged 8-seed floor [EX, K1, WF, OP, SR, CK].")
    w()
    w("F_l / F_p on the converged FREEDOM cells (above 1 = the level floor is the looser one):")
    w()
    w("| dataset | deletion image | deletion text | deletion both | retraining image-withheld | retraining text-withheld | src |")
    w("|---|---|---|---|---|---|---|")
    _fam = {"del_image": [], "del_text": [], "del_both": [], "ret": []}
    for _ds in _DS4:
        _cells = []
        for _st in ("image", "text", "both"):
            _rr = _DEL.get((_ds, _st))
            _v = _rr["fl"] / _rr["fp"] if _rr else None
            if _v is not None:
                _fam["del_" + _st].append(_v)
            _cells.append(fx(_v, 2) if _v is not None else "--")
        for _cond in ("no_image", "no_text"):
            _a = _HOLD.get((_ds, _cond, "Recall@20"))
            _v = _a["fl"] / _a["fp"] if _a else None
            if _v is not None:
                _fam["ret"].append(_v)
            _cells.append(fx(_v, 2) if _v is not None else "--")
        w(_src_row(_ds, *_cells, "CK, HO"))
    w()
    w("- Ranges of F_l/F_p: " + "; ".join(f"{k} {fx(min(v), 2)}-{fx(max(v), 2)} ({sum(1 for x in v if x > 1)}/{len(v)} above 1)"
                                        for k, v in _fam.items() if v) + " [CK, HO].")
    w()

# ------------------------------------------------------------------ 21e instrument ratios
if _DEL and _HOLD:
    w("### 21e. The two instruments on the same eight converged models (6.2)")
    w()
    w("| dataset | modality | deletion d % | deletion verdict | retraining d % | retraining verdict | retraining d / deletion d | deletion d / retraining d | src |")
    w("|" + "---|" * 9)
    for _ds in _DS4:
        for _st, _cond in (("image", "no_image"), ("text", "no_text")):
            _dl, _rt = _DEL.get((_ds, _st)), _HOLD.get((_ds, _cond, "Recall@20"))
            if not (_dl and _rt):
                continue
            w(_src_row(_ds, _st, fx(_dl["pct"], 2, True), _dl["verdict"], fx(_rt["pct"], 2, True), _rt["verdict"],
                       fx(_rt["mean"] / _dl["mean"], 2, True), fx(_dl["mean"] / _rt["mean"], 2, True), "CK, HO"))
    w()

# ------------------------------------------------------------------ 21f Elec deletion rows
_K1 = _rj("phase1/knockout_modality.json") or []
_FE = next((r for r in _K1 if r.get("model") == "freedom" and r.get("dataset") == "elec"), None)
_ELEC = {}
_ope = ((((_OP.get("summary") or {}).get("freedom") or {}).get("elec") or {}).get("Recall@20")) or {}
if _FE and _ope:
    w("### 21f. Electronics deletion rows (tab:freedom Elec, one default-patience checkpoint)")
    w()
    _fl_e, _n_e = D(_ope["MDE_2std"]), _ope["n"]
    _sre = ((((_SR.get("freedom_per_dataset_mde") or {}).get("elec") or {}).get("Recall@20")) or {}).get("MDE")
    _base_e = D(_FE["variants"]["baseline"]["metrics"]["Recall@20"])
    w(f"- Checkpoint {_FE.get('checkpoint')}, R@20 {fx(_base_e, 5)}; floor F_l {sci(_fl_e, 3)} from {_n_e} "
      f"default-patience seeds (SR stores {sci(_sre, 3) if _sre is not None else 'n/a'}); no paired floor, "
      f"no p-value (one checkpoint) [K1, OP, SR].")
    w()
    w("| stream | d | d % | d % (1 dp) | x F_l | against the one floor | RBO@20 (3 dp) | RBO@20 (2 dp) | RBO users | src |")
    w("|" + "---|" * 10)
    for _st in ("image", "text", "both"):
        _v = _FE["variants"].get(_st + "_knockout")
        if not _v:
            continue
        _d = D(_v["dR@20"])
        _rc = _v.get("ranking_change") or {}
        _ELEC[_st] = {"d": _d, "pct": 100 * _d / _base_e, "x": abs(_d) / _fl_e}
        w(_src_row(_st, sci(_d, 3), fx(100 * _d / _base_e, 2, True), fx(100 * _d / _base_e, 1, True),
                   fx(abs(_d) / _fl_e, 2), "clears" if abs(_d) > _fl_e else "below",
                   fx(_rc["rbo"], 3) if "rbo" in _rc else "--", fx(_rc["rbo"], 2) if "rbo" in _rc else "--",
                   _rc.get("n_sampled", "--"), "K1, OP"))
    w()

# ------------------------------------------------------------------ 21g CLIP encoder control
_CL = _rj("trackA_clip/clip_freedom.json") or {}
_VAR = [("cnnimg_berttxt", "cnn:bert"), ("clipimg_berttxt", "clip:bert"), ("clipimg_cliptxt", "clip:clip")]
_DSC = ["baby", "sports", "clothing", "elec"]
if _CL:
    w("### 21g. Encoder control: FREEDOM retrained on CLIP ViT-L/14 features (tab:clip; 4.3)")
    w()

    def _clip_floors(ds, stream):
        out = []
        for (d_, lab), (v, n, mu, tag) in _FLOORS.items():
            if d_ == ds:
                out.append((f"{tag}:{lab}", v))
        rr = _DEL.get((ds, stream))
        if rr:
            out.append((f"CK:converged paired ({stream} deletion)", rr["fp"]))
        return out
    w("| encoders | dataset | image d (x1e-3, 2 dp) | image d / each available floor | image below every floor | text d (x1e-3, 2 dp) | min text d / floor | src |")
    w("|" + "---|" * 8)
    _below_all, _exc, _txt_all, _ncl = 0, [], 0, 0
    for _vk, _vl in _VAR:
        for _ds in _DSC:
            _c = _CL.get(f"{_ds}/{_vk}")
            if not _c:
                continue
            _ncl += 1
            _di, _dt = D(_c["image_knockout_dR@20"]), D(_c["text_knockout_dR@20"])
            _fi, _ft = _clip_floors(_ds, "image"), _clip_floors(_ds, "text")
            _ri = [(lab, abs(_di) / v) for lab, v in _fi]
            _rt = [(lab, abs(_dt) / v) for lab, v in _ft]
            _ball = all(x < 1 for _, x in _ri)
            _below_all += _ball
            if not _ball:
                _exc.append(f"{_ds} {_vl}: clears " + ", ".join(f"{lab} ({fx(x, 2)}x)" for lab, x in _ri if x > 1))
            _txt_all += all(x > 1 for _, x in _rt)
            w(_src_row(_vl, _ds, fx(1000 * _di, 2, True),
                       "; ".join(f"{lab} {fx(x, 2)}" for lab, x in _ri), "yes" if _ball else "NO",
                       fx(1000 * _dt, 2, True), fx(min(x for _, x in _rt), 2) if _rt else "--", "CL + floors"))
    w()
    w(f"- Image deletion is below every available floor in {_below_all} of {_ncl} cells; exceptions: "
      f"{'; '.join(_exc) or 'none'} [CL, OP, SR, HD, MM, CK].")
    w(f"- Text deletion exceeds every available floor in {_txt_all} of {_ncl} cells; its sign on Baby: "
      + ", ".join(f"{_vl} {fx(1000 * D(_CL[f'baby/{_vk}']['text_knockout_dR@20']), 2, True)}e-3"
                  for _vk, _vl in _VAR if f"baby/{_vk}" in _CL) + " [CL].")
    w("- Propagated image stream h_img(graph), absolute gap (relative gap), cnn:bert -> clip:bert -> clip:clip: " + "; ".join(
        f"{_ds} " + " -> ".join(f"{fx(_CL[f'{_ds}/{_vk}']['alignment_behavioral_gap']['h_img(graph)'], 3)} "
                                f"({fx(_CL[f'{_ds}/{_vk}']['alignment_relative']['h_img(graph)'], 3)})"
                                for _vk, _ in _VAR if f"{_ds}/{_vk}" in _CL)
        for _ds in _DSC) + " [CL].")
    _up = [_ds for _ds in _DSC if all(f"{_ds}/{_vk}" in _CL for _vk, _ in _VAR)
           and D(_CL[f"{_ds}/clipimg_cliptxt"]["alignment_behavioral_gap"]["h_img(graph)"])
           > D(_CL[f"{_ds}/cnnimg_berttxt"]["alignment_behavioral_gap"]["h_img(graph)"])
           and D(_CL[f"{_ds}/clipimg_berttxt"]["alignment_behavioral_gap"]["h_img(graph)"])
           > D(_CL[f"{_ds}/cnnimg_berttxt"]["alignment_behavioral_gap"]["h_img(graph)"])]
    w(f"- CLIP raises the propagated image alignment (both CLIP variants above cnn:bert) on {len(_up)} of "
      f"{len(_DSC)} datasets: {', '.join(_up)} [CL].")
    _prov = []
    for _ds in _DSC:
        _c = _CL.get(f"{_ds}/cnnimg_berttxt")
        _runs = [r for r in (_OP.get("runs") or []) if r.get("dataset") == _ds and "test_result" in r]
        _hit = [r["seed"] for r in _runs if D(r["test_result"]["Recall@20"]) == D(_c["baseline_R@20"])
                and r.get("best_epoch") == _c.get("best_epoch")] if _c else []
        _prov.append(f"{_ds} {'= seed ' + ','.join(map(str, _hit)) if _hit else 'matches no'} OP run")
    w("- Seed and patience are not recorded in CL; the cnn:bert arm's baseline R@20 and best epoch match a "
      "default-patience OP run exactly for: " + "; ".join(_prov) + " [CL, OP].")
    w("- Best epochs of the 12 CLIP-grid runs: " + ", ".join(
        f"{_ds}/{_vl} {_CL[f'{_ds}/{_vk}']['best_epoch']}" for _vk, _vl in _VAR for _ds in _DSC if f"{_ds}/{_vk}" in _CL) + " [CL].")
    w()

# ------------------------------------------------------------------ 21h behavioural alignment
_AL = _rj("phase_align/alignment_stats.json") or {}
_AM = _rj("phase_micro/alignment_microlens.json") or {}
_AD = _rj("phase_micro/alignment_degreematched.json") or {}
_ALIGN = {}
for _dd in (_AL.get("datasets") or []):
    _ALIGN[_dd["dataset"]] = (_dd["streams"], "AL")
for _dd in (_AM.get("datasets") or []):
    _ALIGN[_dd["dataset"]] = (_dd["streams"], "AM")
if _ALIGN:
    w("### 21h. Behavioural alignment (tab:align; R4, R10, R22, R29)")
    w()
    w(f"- AL config: {json.dumps(_AL.get('config'), sort_keys=True)}; the permutation p floor is "
      f"1/(B_perm+1) = {sci(_Dc(1) / (int((_AL.get('config') or {}).get('B_perm', 0)) + 1), 3)}; AL records no "
      f"checkpoint name; AM/AD record: " + ", ".join(sorted({f"{d_.get('dataset')} {d_.get('checkpoint')} (pinned "
      f"{d_.get('checkpoint_pinned')})" for d_ in (_AM.get('datasets') or []) + (_AD.get('datasets') or [])})) + " [AL, AM, AD].")
    w()
    _ROWS = [("raw_image_cnn", "image, CNN (MicroLens: released image)"), ("raw_image_clip", "image, CLIP"),
             ("raw_text_bert", "text, BERT (MicroLens: released text)"), ("raw_text_clip", "text, CLIP"),
             ("h_img_graph", "h_img"), ("h_txt_graph", "h_txt"), ("cf", "CF"), ("fused", "fused")]
    _dsl = [d_ for d_ in ("baby", "sports", "clothing", "elec", "microlens") if d_ in _ALIGN]
    w("Absolute gap (dispersion-normalised gap in parentheses), 3 dp:")
    w()
    w("| stream | " + " | ".join(_dsl) + " | src |")
    w("|---|" + "---|" * len(_dsl) + "---|")
    for _sk, _sl in _ROWS:
        _cells = []
        for _d in _dsl:
            _s = _ALIGN[_d][0].get(_sk)
            _cells.append(f"{fx(_s['gap_abs'], 3)} ({fx(_s['gap_rel'], 3)})" if _s else "--")
        w(_src_row(_sl, *_cells, "AL (Amazon), AM (MicroLens)"))
    w()

    def _sep(sa, sb, key):
        """'a<b' / 'a>b' if the 95% CIs separate, else 'overlap'."""
        la, ha = (D(x) for x in sa[key + "_ci95"])
        lb, hb = (D(x) for x in sb[key + "_ci95"])
        return "a<b" if ha < lb else ("a>b" if la > hb else "overlap")
    w("| dataset | CNN vs BERT, abs | CNN vs BERT, rel | h_img vs h_txt, abs | h_img vs h_txt, rel | h_img vs CF, abs | CLIP-img vs CNN, abs | CLIP-img vs CNN, rel | src |")
    w("|" + "---|" * 9)
    for _d in _dsl:
        _s = _ALIGN[_d][0]
        _c = [_sep(_s["raw_image_cnn"], _s["raw_text_bert"], "gap_abs"),
              _sep(_s["raw_image_cnn"], _s["raw_text_bert"], "gap_rel"),
              _sep(_s["h_img_graph"], _s["h_txt_graph"], "gap_abs"),
              _sep(_s["h_img_graph"], _s["h_txt_graph"], "gap_rel"),
              _sep(_s["h_img_graph"], _s["cf"], "gap_abs")]
        if "raw_image_clip" in _s:
            _c += [_sep(_s["raw_image_clip"], _s["raw_image_cnn"], "gap_abs"),
                   _sep(_s["raw_image_clip"], _s["raw_image_cnn"], "gap_rel")]
        else:
            _c += ["--", "--"]
        w(_src_row(_d, *_c, _ALIGN[_d][1]))
    w()
    w("(a<b / a>b = 95% bootstrap intervals separate in that direction; overlap = not resolved.)")
    for _d in _dsl:
        _s = _ALIGN[_d][0]
        _raw = [k for k in ("raw_image_cnn", "raw_image_clip", "raw_text_bert", "raw_text_clip") if k in _s]
        for _key in ("gap_abs", "gap_rel"):
            _lo = min(_raw, key=lambda k: D(_s[k][_key]))
            w(f"- {_d} {_key}: lowest point estimate among raw streams {', '.join(_raw)} is {_lo}"
              f"{'' if _lo == 'raw_image_cnn' else ' (NOT the CNN image stream)'} [{_ALIGN[_d][1]}].")
    for _src, _lab in ((_AM, "uniform null"), (_AD, "popularity-matched (co-consumption-marginal) null")):
        for _dd in (_src.get("datasets") or []):
            if _dd.get("dataset") != "microlens":
                continue
            _s = _dd["streams"]
            w(f"- MicroLens, {_lab}: raw image {fx(_s['raw_image_cnn']['gap_abs'], 3)} vs raw text "
              f"{fx(_s['raw_text_bert']['gap_abs'], 3)} ({_sep(_s['raw_image_cnn'], _s['raw_text_bert'], 'gap_abs')}); "
              f"h_img {fx(_s['h_img_graph']['gap_abs'], 3)} vs h_txt {fx(_s['h_txt_graph']['gap_abs'], 3)} "
              f"({_sep(_s['h_img_graph'], _s['h_txt_graph'], 'gap_abs')}); relative: raw image "
              f"{fx(_s['raw_image_cnn']['gap_rel'], 3)} vs text {fx(_s['raw_text_bert']['gap_rel'], 3)} "
              f"[{'AM' if _src is _AM else 'AD'}].")
    _r22 = [(_d, D(_ALIGN[_d][0]["raw_image_cnn"]["gap_abs"]), D(_ALIGN[_d][0]["raw_image_cnn"]["gap_rel"]),
             _HOLD[(_d, "no_image", "Recall@20")]["pct"]) for _d in _AMZ3
            if _d in _ALIGN and (_d, "no_image", "Recall@20") in _HOLD]
    if _r22:
        w("- Across datasets (R22): raw CNN image gap abs / rel vs the converged image-withheld retraining effect: "
          + "; ".join(f"{d_} {fx(a_, 3)} / {fx(r_, 3)} vs {fx(p_, 2, True)}%" for d_, a_, r_, p_ in _r22)
          + f". Lowest-alignment dataset among these: {min(_r22, key=lambda z: z[1])[0]}; highest: "
          f"{max(_r22, key=lambda z: z[1])[0]}; largest |retraining effect|: {max(_r22, key=lambda z: abs(z[3]))[0]} [AL, HO].")
        _allc = [(d_, D(_ALIGN[d_][0]["raw_image_cnn"]["gap_abs"])) for d_ in ("baby", "sports", "clothing", "elec") if d_ in _ALIGN]
        w(f"- Over all four Amazon datasets (Elec was not retrained) the lowest raw CNN image gap is "
          f"{min(_allc, key=lambda z: z[1])[0]} ({fx(min(z[1] for z in _allc), 3)}) [AL].")
    w()

# ------------------------------------------------------------------ 21i frozen lambda re-mix
_WF = _rj("phase2/weight_sweep_frozen.json") or []
_WM = _rj("phase_micro/weightsweep_microlens.json") or []
if _WF or _WM:
    w("### 21i. Re-mixing FREEDOM's image weight on trained models, no retraining (R24)")
    w()
    w("| dataset | checkpoint | trained weight | argmax weight | best alternative weight | its d R@20 | d (4 dp) | weights with d > 0 | stored floor (which it is) | d / stored floor | d / own default-patience floors | d / converged F_l | src |")
    w("|" + "---|" * 13)
    _sf = _rj("phase_micro/significance_freedom_microlens.json") or {}
    _MLREMIX = None
    for _c in list(_WF) + list(_WM):
        _ds = _c["dataset"]
        _tw = _c["trained_weight"]
        _tr = D(next(s["Recall@20"] for s in _c["sweep"] if s["weight"] == _tw))
        _alts = [(D(s["Recall@20"]) - _tr, s["weight"]) for s in _c["sweep"] if s["weight"] != _tw]
        _best = max(_alts, key=lambda z: (z[0], -z[1]))
        _amax = max(_c["sweep"], key=lambda s: (D(s["Recall@20"]), -s["weight"]))["weight"]
        _pos = [f"{wt} ({sci(d_, 2)})" for d_, wt in _alts if d_ > 0]
        _stf = _c.get("MDE_R@20")
        _conv = _DEL.get((_ds, "image"))
        _is = [f"{d_} {lab}" for (d_, lab), (v, n, mu, tag) in _FLOORS.items()
               if _stf is not None and v == D(_stf)]
        _own = [(lab, v) for (d_, lab), (v, n, mu, tag) in _FLOORS.items() if d_ == _ds and lab.startswith("default")]
        w(_src_row(_ds, _c.get("checkpoint"), _tw, _amax, _best[1], sci(_best[0], 3), fx(_best[0], 4, True),
                   f"{len(_pos)} of {len(_alts)}: {', '.join(_pos) or 'none'}",
                   (sci(_stf, 3) + f" ({'; '.join(_is) or 'unmatched'})") if _stf is not None else "none",
                   fx(_best[0] / D(_stf), 2, True) if _stf is not None else "--",
                   "; ".join(f"{lab} {fx(_best[0] / v, 2, True)}" for lab, v in _own) or "--",
                   fx(_best[0] / _conv["fl"], 2, True) if _conv else "--",
                   "WM" if _ds == "microlens" else "WF, OP, SR, HD, MM, CK"))
        if _ds == "microlens":
            _ff = [(lab, v) for (d_, lab), (v, n, mu, tag) in _FLOORS.items() if d_ == "microlens"]
            _fpi = ((_sf.get("streams") or {}).get("image") or {}).get("F_paired")
            if _fpi is not None:
                _ff.append(("SF default-patience paired floor, image stream", D(_fpi)))
            _MLREMIX = (_best, _ff)
    w()
    if _MLREMIX:
        _best, _ff = _MLREMIX
        w(f"- MicroLens best re-mix d = {sci(_best[0], 2)} at weight {_best[1]}, against each MicroLens floor: "
          + "; ".join(f"{lab} {fx(_best[0] / v, 2)}x" for lab, v in _ff) + " [WM, MM, HD, CK, SF].")
    _wr = _rj("phase2/weight_sweep_retrain.json") or {}
    _wrds = sorted({r.get("dataset") for r in (_wr.get("runs") or [])})
    w(f"- Datasets in the retrain lambda sweep: {', '.join(_wrds)}; MicroLens "
      f"{'is' if 'microlens' in _wrds else 'is NOT'} among them (the registered H3 retrain) [WR].")
    w()

# ------------------------------------------------------------------ 21j cross-architecture
# RULE, fixed here before the counts below are printed (the SAC brief's own example): a cell's
# floor is DEGENERATE if EVERY run the floor is computed from peaked at best_epoch <= 25.
DEGEN_MAX_BEST_EPOCH = 25
# Metadata, not results: the words for each canonical interface class (s.1b) used in tab:arch.
_IFACE_WORDS = {"C1": "frozen similarity graph", "C2": "live projected feature", "C1+C2": "graph + live branch",
                "C3": "shared normalizer", "C4": "joint term", "C5": "off-path"}
_TIER = {"EXACT": "exact", "DELETE_PLUS_RENORM": "renormalized", "BRACKET_ONLY": "bracketed",
         "UNDEFINED_ID_ONLY": "undefined"}
_EX = _rj("phase_exact/exact_crossarch.json") or []
_MDEF = {}
for _f in sorted((ROOT / "results" / "phase_mde").glob("*_mde.json")):
    _MDEF[_f.name] = json.loads(_f.read_text())


def _floor_runs(model, ds, floor):
    """Runs behind a floor: every *_mde.json summary entry whose R@20 MDE equals `floor` exactly."""
    hits = []
    for name, dj in _MDEF.items():
        s = (((dj.get("summary") or {}).get(model) or {}).get(ds) or {})
        r20 = s.get("Recall@20") or {}
        if r20.get("MDE_2std") is not None and D(r20["MDE_2std"]) == D(floor):
            runs = [r for r in dj.get("runs", []) if r.get("model") == model and r.get("dataset") == ds
                    and "test_result" in r]
            nd = (s.get("NDCG@20") or {}).get("MDE_2std")
            hits.append((name, runs, D(nd) if nd is not None else None))
    return hits


_XA = []
if _EX:
    for _r in _EX:
        if "error" in _r:
            continue
        _a = (_r.get("arms") or {}).get("image_knockout")
        if not _a or not _r.get("MDE_R@20"):
            continue
        _XA.append({"m": _r["model"], "d": _r["dataset"], "dr": D(_a["dR@20"]),
                    "dn": D(_a["dN@20"]) if _a.get("dN@20") is not None else None,
                    "f": D(_r["MDE_R@20"]), "n": _r.get("MDE_n_seeds"), "tier": _r["exactness"],
                    "src": "EX", "rec": _r, "screen": (_r.get("input_mean_screen") or {}).get("image_knockout"),
                    "base": D(_r["arms"]["baseline"]["metrics"]["Recall@20"]),
                    "pinned": _r.get("checkpoint_pinned"), "ckpt": _r.get("checkpoint")})
    for _mo in ("freedom", "lgmrec"):
        _fl = ((((_SR.get(f"{_mo}_per_dataset_mde") or {}).get("elec") or {}).get("Recall@20")) or {})
        _rec = next((r for r in _K1 if r.get("model") == _mo and r.get("dataset") == "elec"), None)
        _v = ((_rec or {}).get("variants") or {}).get("image_knockout")
        if _fl.get("MDE") is not None and _v:
            _tiers = {r["exactness"] for r in _EX if r.get("model") == _mo and "error" not in r}
            assert len(_tiers) == 1, (_mo, _tiers)
            _XA.append({"m": _mo, "d": "elec", "dr": D(_v["dR@20"]), "dn": D(_v["dN@20"]),
                        "f": D(_fl["MDE"]), "n": _fl.get("n"), "tier": _tiers.pop(), "src": "K1+SR",
                        "rec": _rec, "screen": None,
                        "base": D(_rec["variants"]["baseline"]["metrics"]["Recall@20"]),
                        "pinned": _rec.get("checkpoint_pinned"), "ckpt": _rec.get("checkpoint")})
    for _c in _XA:
        _c["x"] = abs(_c["dr"]) / _c["f"]
        _c["clears"] = _c["x"] > 1
        _c["borderline"] = _c["n"] is not None and _c["n"] < 5 and _Dc("0.5") <= _c["x"] <= 2
        _c["cls"] = cls(_c["m"])
        _c["t"] = 2 * abs(_c["dr"]) / _c["f"]
        _c["p"] = t_p2(_c["t"], _c["n"] - 1)
        _c["key"] = f"{_c['m']}/{_c['d']}"
        _h = _floor_runs(_c["m"], _c["d"], _c["f"])
        _c["floor_file"] = _h[0][0] if _h else None
        _c["floor_runs"] = _h[0][1] if _h else []
        _c["fN"] = _h[0][2] if _h else None
        _c["eps"] = [r.get("best_epoch") for r in _c["floor_runs"]]
        _c["degenA"] = bool(_c["eps"]) and all(e is not None and e <= DEGEN_MAX_BEST_EPOCH for e in _c["eps"])
    _PCr = _rj("phase_mde/patience_control.json") or []
    _PCG = collections.defaultdict(dict)
    for _r in _PCr:
        if "test_result" in _r:
            _PCG[(_r["model"], _r["dataset"])][_r["seed"]] = _r
    for _c in _XA:
        _pe = [r["best_epoch"] for r in _PCG.get((_c["m"], _c["d"]), {}).values()]
        _c["p100_eps"] = sorted(_pe)
        _c["degenBmax"] = _c["degenA"] and bool(_pe) and max(_pe) > DEGEN_MAX_BEST_EPOCH
        _c["degenBmed"] = _c["degenA"] and bool(_pe) and dmedian(_pe) > DEGEN_MAX_BEST_EPOCH
    _BH = set(bh_reject([(c["p"], c["key"]) for c in _XA]))
    _HOLM = set(holm_reject([(c["p"], c["key"]) for c in _XA]))
    _clr = [c for c in _XA if c["clears"]]

    w("### 21j. Deletion across eleven architectures, 32-cell basis (tab:arch; 3.2; R12-R15, R30-R32)")
    w()
    # tiers over all computed cells
    _allc = [r for r in _EX if "error" not in r]
    _tcount = collections.Counter(_TIER.get(r["exactness"], r["exactness"]) for r in _allc)
    for _c in _XA:
        if _c["src"] == "K1+SR":
            _tcount[_TIER.get(_c["tier"], _c["tier"])] += 1
    _errs = [r for r in _EX if "error" in r]
    w(f"- Cells computed: {len(_allc)} in EX plus {sum(1 for c in _XA if c['src'] == 'K1+SR')} Electronics cells in K1 = "
      f"{len(_allc) + sum(1 for c in _XA if c['src'] == 'K1+SR')}; tiers: "
      + ", ".join(f"{k} {v}" for k, v in sorted(_tcount.items(), key=lambda kv: -kv[1]))
      + f"; interval-valued (renormalized + bracketed): {_tcount.get('renormalized', 0) + _tcount.get('bracketed', 0)} [EX, K1].")
    w(f"- EX records with an error: {len(_errs)} (" + "; ".join(
        f"{r['model']}/{r['dataset']}: {str(r['error']).split('(')[0]}" for r in _errs)
      + f"); reconstruction tolerance recon_tol = {', '.join(sorted({sci(r['recon_tol'], 1) for r in _allc if r.get('recon_tol') is not None}))} [EX].")
    _rc = [(D(r["recon_error"]), r["model"], r["dataset"]) for r in _allc if r.get("recon_error") is not None]
    _rcm = collections.defaultdict(list)
    for _v, _mo, _d in _rc:
        _rcm[_mo].append(_v)
    _nulls = sorted({r["model"] for r in _allc if r.get("recon_error") is None})
    w(f"- Reconstruction error over the {len(_rc)} EX cells that record one: max {sci(max(_rc)[0], 2)} "
      f"({max(_rc)[1]}/{max(_rc)[2]}); per-model maxima " + ", ".join(f"{mo} {sci(max(v), 2)}" for mo, v in sorted(_rcm.items()))
      + f"; max excluding mmgcn {sci(max(v for v, mo, _ in _rc if mo != 'mmgcn'), 2)}; recon_error is null for: "
      f"{', '.join(_nulls) or 'none'} [EX].")
    _fe_rec = next((r for r in _K1 if r.get("model") == "freedom" and r.get("dataset") == "elec"), None)
    if _fe_rec:
        w(f"- FREEDOM Electronics (K1) recon_max_err {sci(_fe_rec['recon_max_err'], 2)} [K1].")
    w(f"- Adjudicable basis: {len(_XA)} cells ({sum(1 for c in _XA if c['src'] == 'EX')} from EX + "
      f"{sum(1 for c in _XA if c['src'] == 'K1+SR')} Electronics); {sum(1 for c in _XA if c['d'] != 'microlens')} "
      f"are Amazon; floor seeds: {dict(sorted(collections.Counter(c['n'] for c in _XA).items()))}; "
      f"architectures with an image arm: {len({c['m'] for c in _XA})} [EX, K1, SR].")
    w()
    w("| model | interface | tier | dataset | d R@20 | floor | n | d/floor (signed, 2 dp) | (1 dp) | clears | borderline | BH | degenerate floor (rule) | floor best epochs | dN@20/NDCG floor | src |")
    w("|" + "---|" * 16)
    _order = ["lgmrec", "cohesion", "mmgcn", "lattice", "vbpr", "damrs", "smore", "freedom", "mentor", "gume", "mgcn"]
    _dso = ["baby", "sports", "clothing", "elec", "microlens"]
    for _c in sorted(_XA, key=lambda c: (_order.index(c["m"]) if c["m"] in _order else 99, _dso.index(c["d"]))):
        _sr = _c["dr"] / _c["f"]
        _nr = (fx(_c["dn"] / _c["fN"], 2, True) if (_c["dn"] is not None and _c["fN"]) else "--")
        w(_src_row(_c["m"], _IFACE_WORDS.get(_c["cls"], _c["cls"]), _TIER.get(_c["tier"], _c["tier"]), _c["d"],
                   sci(_c["dr"], 3), sci(_c["f"], 3), _c["n"], fx(_sr, 2, True), fx(_sr, 1, True),
                   "**yes**" if _c["clears"] else "no", "yes" if _c["borderline"] else "no",
                   "*" if _c["key"] in _BH else "", "A" if _c["degenA"] else "",
                   ",".join(str(e) for e in _c["eps"]), _nr, f"{_c['src']}, MDE:{_c['floor_file']}"))
    w()
    w(f"- Clearing its own floor: **{len(_clr)} of {len(_XA)}** cells, over {len({c['m'] for c in _clr})} of "
      f"{len({c['m'] for c in _XA})} architectures: " + ", ".join(
          f"{m_} {sum(1 for c in _clr if c['m'] == m_)}/{sum(1 for c in _XA if c['m'] == m_)}"
          for m_ in _order if any(c["m"] == m_ for c in _XA)) + " [EX, K1, SR].")
    _bc = [c for c in _clr if c["borderline"]]
    _bn = [c for c in _XA if not c["clears"] and c["borderline"]]
    w(f"- Borderline (0.5-2x a floor from fewer than 5 seeds): {len(_bc)} of the {len(_clr)} clearing "
      f"({', '.join(c['key'] + ' ' + fx(c['x'], 2) + 'x' for c in _bc)}) and {len(_bn)} of the "
      f"{len(_XA) - len(_clr)} non-clearing ({', '.join(c['key'] + ' ' + fx(c['x'], 2) + 'x' for c in _bn)}) [EX, K1, SR].")
    w()
    w("Multiplicity. Reference, stated before the counts: each level floor is 2 sd, so a single delta")
    w("is read as t = 2|d|/F on n-1 degrees of freedom (the floor's own seeds); p is two-sided;")
    w("BH and Holm at 0.05 over all 32 cells.")
    w()
    _unadj = [c for c in _XA if c["p"] < _Dc("0.05")]
    _exp = dsum(t_p2(2, c["n"] - 1) for c in _XA)
    w(f"- Unadjusted p < 0.05: {len(_unadj)} cells ({', '.join(c['key'] for c in sorted(_unadj, key=lambda c: c['p']))}) [EX, K1, SR].")
    w(f"- Benjamini-Hochberg (q = 0.05): {len(_BH)} survive ({', '.join(sorted(_BH))}); Holm (0.05): {len(_HOLM)} "
      f"survive ({', '.join(sorted(_HOLM)) or 'none'}) [EX, K1, SR].")
    _sp = sorted(_XA, key=lambda c: (c["p"], c["key"]))
    w("- Smallest p-values vs their BH thresholds k*0.05/32: " + "; ".join(
        f"{i}. {c['key']} p={sci(c['p'], 3)} vs {sci(_Dc('0.05') * i / len(_XA), 3)}" for i, c in enumerate(_sp[:6], 1))
      + f"; Holm's first threshold 0.05/{len(_XA)} = {sci(_Dc('0.05') / len(_XA), 3)} [EX, K1, SR].")
    w(f"- Expected number of cells clearing by chance if all {len(_XA)} were null, sum of P(|t_(n-1)| > 2): "
      f"{fx(_exp, 2)} [EX, K1, SR].")
    w()
    # composition of the clearing cells
    _comp = collections.OrderedDict()
    for _c in _clr:
        _s = _c["screen"]
        if _c["dr"] > 0:
            _cat = "removing image RAISES R@20"
        elif _c["cls"] == "C4":
            _cat = "joint term: per-modality effect not identified"
        elif _c["cls"] == "C1+C2":
            _cat = "branch, not content (COHESION)"
        elif _s is None:
            _cat = "deletion-only (no averaging measurement)"
        elif D(_s) < 0 and abs(D(_s)) > _c["f"]:
            _cat = "negative; averaging also clears the floor"
        else:
            _cat = "negative; averaging does NOT clear the floor"
        _comp.setdefault(_cat, []).append(_c)
    w("Composition of the clearing cells (precedence top to bottom: sign, joint-term class C4, compound")
    w("class C1+C2, averaging measured or not, averaging clears or not). Averaging = the input-mean")
    w("screen stored in EX `input_mean_screen`:")
    w()
    w("| category | cells | members (d/floor; averaging d/floor) | src |")
    w("|---|---|---|---|")
    for _cat, _cs in _comp.items():
        w(_src_row(_cat, len(_cs), "; ".join(
            f"{c['key']} ({fx(c['dr'] / c['f'], 2, True)}; "
            f"{fx(D(c['screen']) / c['f'], 2, True) + ('' if D(c['screen']) != c['dr'] else ', identical to deletion') if c['screen'] is not None else 'none'})"
            for c in _cs), "EX"))
    w()
    w(f"- Composition counts: " + " / ".join(str(len(v)) for v in _comp.values()) + f" (total {sum(len(v) for v in _comp.values())}) [EX].")
    w()
    # degenerate floors
    _dA = [c for c in _XA if c["degenA"]]
    _dBx = [c for c in _XA if c["degenBmax"]]
    _dBm = [c for c in _XA if c["degenBmed"]]

    def _excl(lst):
        ks = {c["key"] for c in lst}
        keep = [c for c in _XA if c["key"] not in ks]
        return f"{sum(1 for c in keep if c['clears'])} of {len(keep)}"
    w(f"Degenerate-floor audit. RULE A (stated in the code above before any count; the rule this file adopts): "
      f"every floor run peaked at best_epoch <= {DEGEN_MAX_BEST_EPOCH}.")
    w()
    w(f"- Rule A flags {len(_dA)} cells: " + "; ".join(
        f"{c['key']} (best epochs {','.join(map(str, c['eps']))}; floor-run R@20 "
        f"{fx(min(D(r['test_result']['Recall@20']) for r in c['floor_runs']), 4)}-"
        f"{fx(max(D(r['test_result']['Recall@20']) for r in c['floor_runs']), 4)}; "
        f"{'clears' if c['clears'] else 'does not clear'}; patience-100 best epochs {','.join(map(str, c['p100_eps'])) or 'none'})"
        for c in _dA) + f". Excluding them: **{_excl(_dA)}** cells clear [EX, K1, SR, MDE, PC].")
    w(f"- Sensitivity only (written AFTER rule A's result was seen, so not pre-stated): rule B-max = rule A "
      f"and some patience-100 run of the same cell peaks after epoch {DEGEN_MAX_BEST_EPOCH} flags "
      f"{len(_dBx)} ({', '.join(c['key'] for c in _dBx)}), leaving {_excl(_dBx)}; rule B-median = rule A and the "
      f"median patience-100 best epoch exceeds {DEGEN_MAX_BEST_EPOCH} flags {len(_dBm)} "
      f"({', '.join(c['key'] for c in _dBm)}), leaving {_excl(_dBm)} [EX, MDE, PC].")
    w()
    # patience-100 floor basis
    _PCF = {}
    for (_mo, _d), _runs in _PCG.items():
        _vals = [D(r["test_result"]["Recall@20"]) for _, r in sorted(_runs.items())]
        if len(_vals) > 1:
            _PCF[(_mo, _d)] = (2 * dsd(_vals), len(_vals), sorted(r["best_epoch"] for r in _runs.values()))
    _p100 = []
    for _c in _XA:
        _pf = _PCF.get((_c["m"], _c["d"]))
        _p100.append({"key": _c["key"], "dr": _c["dr"], "f": _pf[0] if _pf else _c["f"],
                      "n": _pf[1] if _pf else _c["n"], "p100": bool(_pf), "eps": _pf[2] if _pf else None})
    _added = []
    for _r in _EX:
        if "error" in _r or not _r.get("floor_contaminated"):
            continue
        _a = (_r.get("arms") or {}).get("image_knockout")
        _pf = _PCF.get((_r["model"], _r["dataset"]))
        if _a and _pf:
            _p100.append({"key": f"{_r['model']}/{_r['dataset']}", "dr": D(_a["dR@20"]), "f": _pf[0], "n": _pf[1],
                          "p100": True, "eps": _pf[2]})
            _added.append(f"{_r['model']}/{_r['dataset']}")
    for _c in _p100:
        _c["x"] = abs(_c["dr"]) / _c["f"]
        _c["clears"] = _c["x"] > 1
        _c["p"] = t_p2(2 * _c["x"], _c["n"] - 1)
    _was = {c["key"]: c["clears"] for c in _XA}
    _left = [c for c in _p100 if _was.get(c["key"]) and not c["clears"]]
    _ent = [c for c in _p100 if c["clears"] and not _was.get(c["key"], False)]
    w(f"- Under patience-100 floors from PC (replacing {sum(1 for c in _p100 if c['p100'] and c['key'] in _was)} "
      f"floors, adding the {len(_added)} contaminated cells {', '.join(_added)}): {len(_p100)} adjudicable, "
      f"**{sum(1 for c in _p100 if c['clears'])} clear**; leaving: "
      + ", ".join(f"{c['key']} ({fx(c['x'], 2)}x)" for c in _left)
      + "; entering: " + ", ".join(f"{c['key']} ({fx(c['x'], 2)}x, d {sci(c['dr'], 2)})" for c in _ent)
      + f"; BH {len(bh_reject([(c['p'], c['key']) for c in _p100]))}, Holm {len(holm_reject([(c['p'], c['key']) for c in _p100]))} [EX, PC].")
    _sp100 = sorted(_p100, key=lambda c: (c["p"], c["key"]))
    w(f"- Same t reference on the {len(_p100)}-cell basis, smallest p-values vs BH thresholds k*0.05/{len(_p100)}: " + "; ".join(
        f"{i}. {c['key']} p={sci(c['p'], 3)} vs {sci(_Dc('0.05') * i / len(_p100), 3)}" for i, c in enumerate(_sp100[:5], 1))
      + f"; unadjusted p < 0.05: {sum(1 for c in _p100 if c['p'] < _Dc('0.05'))} [EX, PC].")
    _cap = [f"{k[0]}/{k[1]} (best epochs {','.join(map(str, v[2]))})" for k, v in sorted(_PCF.items())
            if any(e >= 975 for e in v[2]) or (max(v[2]) > 10 * max(1, min(v[2])))]
    w(f"- PC records carry no epochs cap; PC floors whose runs peak at 975+ or spread over 10x in best epoch: "
      f"{'; '.join(_cap) or 'none'} [PC].")
    w()
    # interval-valued cells
    _ivk = re.compile(r"^image_knockout(_(nonorm|hi|lo|fixsupport|massremoved|hi_massremoved|hi_attfree))?$")
    _iv = []
    for _c in _XA:
        if _c["tier"] not in ("DELETE_PLUS_RENORM", "BRACKET_ONLY") or _c["src"] != "EX":
            continue
        _vs = {k: D(v["dR@20"]) for k, v in _c["rec"]["arms"].items() if _ivk.match(k) and "dR@20" in v}
        _xs = {k: abs(v) / _c["f"] for k, v in _vs.items()}
        _iv.append((_c, _xs, any(x > 1 for x in _xs.values()) and any(x <= 1 for x in _xs.values())))
    _k1iv = [c for c in _XA if c["src"] == "K1+SR" and c["tier"] in ("DELETE_PLUS_RENORM", "BRACKET_ONLY")]
    _str = [c for c, _, s in _iv if s]
    w(f"- Interval-valued cells among the {len(_XA)}: {len(_iv) + len(_k1iv)}; with more than one recorded image-deletion "
      f"variant: {sum(1 for _, xs, _ in _iv if len(xs) > 1)}; straddling the floor (some variant clears, some does not): "
      f"**{len(_str)}**: " + "; ".join(
          f"{c['key']} (" + ", ".join(f"{k.replace('image_knockout', 'ko') or 'ko'} {fx(x, 2)}x" for k, x in sorted(xs.items())) + ")"
          for c, xs, s in _iv if s) + " [EX].")
    _n13 = sum(1 for c in _XA if c["clears"])
    _lose = sum(1 for c, xs, s in _iv if s and c["clears"])
    _gain = sum(1 for c, xs, s in _iv if s and not c["clears"])
    w(f"- Clearing count across recorded deletion variants: {_n13 - _lose} to {_n13 + _gain} of {len(_XA)} "
      f"(default variant: {_n13}; {_lose} straddling cells clear only at the default, {_gain} only at another "
      f"variant) [EX].")
    w("- Interval cells with a single recorded variant (no interval measured): " + ", ".join(
        c["key"] for c, xs, _ in _iv if len(xs) == 1) + (", " + ", ".join(c["key"] for c in _k1iv) if _k1iv else "") + " [EX, K1].")
    w()
    # checkpoints: pinning, default patience vs converged
    _TRr = _rj("phase_convergence/table_rerun.json") or []
    _TRv = {(r["model"], r["dataset"]): D(r["test_result"]["Recall@20"]) for r in _TRr if "test_result" in r}
    _pin = sorted({c["m"] for c in _XA if c["pinned"] is True})
    _nopin = sorted({c["m"] for c in _XA if c["pinned"] is None})
    w(f"- Pinned checkpoints among the {len(_XA)} cells: {sum(1 for c in _XA if c['pinned'] is True)} "
      f"(models: {', '.join(_pin)}); cells whose record carries no pin field: "
      f"{', '.join(c['key'] for c in _XA if c['pinned'] is None) or 'none'} [EX, K1].")
    _vs = []
    for _c in _clr:
        _cv = _TRv.get((_c["m"], _c["d"]))
        if _cv is not None:
            _vs.append((100 * (_c["base"] / _cv - 1), _c["key"]))
    if _vs:
        _below3 = [(v, k) for v, k in _vs if v <= -3]
        w(f"- Deletion checkpoint R@20 vs the converged seed-2024 re-run, for the {len(_vs)} clearing cells: "
          + "; ".join(f"{k} {fx(v, 1, True)}%" for v, k in sorted(_vs)) + f". At least 3% below converged: "
          f"{len(_below3)} ({', '.join(k for _, k in sorted(_below3))}); largest excess: {max(_vs)[1]} "
          f"{fx(max(_vs)[0], 1, True)}% [EX, TR].")
    w()
    # COHESION split
    _coh = [r for r in _EX if r.get("model") == "cohesion" and "error" not in r]
    if _coh:
        w("COHESION: frozen-graph path vs live block, and the averaging (input-mean) measurement:")
        w()
        w("| dataset | floor | image d / floor | graph image / floor | block image / floor | block share of image d | graph text / floor | averaging image / floor | src |")
        w("|" + "---|" * 9)
        for _r in sorted(_coh, key=lambda r: _dso.index(r["dataset"])):
            _A = _r["arms"]
            _f = D(_r["MDE_R@20"]) if _r.get("MDE_R@20") else None
            _im, _gi, _bi, _gt = (D(_A[k]["dR@20"]) for k in ("image_knockout", "graph_image_knockout",
                                                              "block_image_knockout", "graph_text_knockout"))
            _sc = (_r.get("input_mean_screen") or {}).get("image_knockout")

            def _rf(v):
                return fx(v / _f, 2, True) if _f else f"({sci(v, 2)})"
            w(_src_row(_r["dataset"], sci(_f, 3) if _f else "none", _rf(_im), _rf(_gi), _rf(_bi),
                       fx(100 * _bi / _im, 0) + "%", _rf(_gt),
                       (_rf(D(_sc)) if _sc is not None else "--"), "EX"))
        w()
        _cs = [f for f in sorted((ROOT / "results" / "_scratch" / "exact_ko").glob("cohesion_baby_np*.json"))]
        for _f in _cs:
            _dj = json.loads(_f.read_text())
            _b = D(_dj["arms"]["baseline"]["metrics"]["Recall@20"])
            w(f"- COHESION/Baby baseline in {_f.name} (numpy {(_dj.get('env') or {}).get('numpy')}): {fx(_b, 6)} vs "
              f"trainer-logged {fx(_dj['logged_test_R@20'], 6)}, difference {sci(_b - D(_dj['logged_test_R@20']), 2)}; "
              f"EX's COHESION/Baby baseline is {fx(D(next(r for r in _coh if r['dataset'] == 'baby')['arms']['baseline']['metrics']['Recall@20']), 6)} [CS, EX].")
        w("- Trainer-logged COHESION values for Sports, Clothing and MicroLens are not stored under results/, so "
          "the drift is certified on Baby only [CS].")
        w()
    # LGMRec
    _lgm = [c for c in _XA if c["m"] == "lgmrec"]
    if _lgm:
        w("LGMRec (renormalized deletion, one fixed Gumbel draw):")
        w()
        w("| dataset | image d / floor | text d / floor | image dN@20 / NDCG floor | text dN@20 / NDCG floor | recon_error | src |")
        w("|" + "---|" * 7)
        for _c in sorted(_lgm, key=lambda c: _dso.index(c["d"])):
            _arms = _c["rec"]["arms"] if _c["src"] == "EX" else _c["rec"]["variants"]
            _tk = _arms.get("text_knockout") or {}
            _tn = D(_tk["dN@20"]) if _tk.get("dN@20") is not None else None
            w(_src_row(_c["d"], fx(_c["dr"] / _c["f"], 2, True), fx(D(_tk["dR@20"]) / _c["f"], 2, True),
                       fx(_c["dn"] / _c["fN"], 2, True) if _c["fN"] else "--",
                       fx(_tn / _c["fN"], 2, True) if (_c["fN"] and _tn is not None) else "--",
                       _c["rec"].get("recon_error", _c["rec"].get("recon_max_err")),
                       f"{_c['src']}, MDE:{_c['floor_file']}"))
        w()
        _amz_n = [c for c in _lgm if c["d"] != "microlens" and c["fN"]]
        w(f"- LGMRec clears its NDCG@20 floor on {sum(1 for c in _amz_n if abs(c['dn']) > c['fN'])} of {len(_amz_n)} "
          f"Amazon datasets [EX, K1, MDE].")
        _RM = _rj("phase0/repro_metrics.json") or []
        _lg_log = []
        for _c in _lgm:
            _rm = next((r for r in _RM if r.get("model") == "lgmrec" and r.get("dataset") == _c["d"]), None)
            if _rm and _rm.get("checkpoint") == _c["ckpt"]:
                _lg_log.append((abs(_c["base"] - D(_rm["logged"]["Recall@20"])), _c["d"], _c["base"] - D(_rm["logged"]["Recall@20"])))
        if _lg_log:
            w("- LGMRec baseline used for deletion minus the trainer-logged R@20 of the same checkpoint: " + "; ".join(
                f"{d_} {sci(s_, 2)}" for _, d_, s_ in _lg_log) + f"; max |difference| {sci(max(_lg_log)[0], 3)} ({max(_lg_log)[1]}); "
                f"MicroLens has no logged value in RM [EX, K1, RM].")
        _rmd = [(abs(D(r["deltas_vs_logged"]["Recall@20"])), r["dataset"], D(r["deltas_vs_logged"]["Recall@20"]))
                for r in _RM if r.get("model") == "lgmrec" and (r.get("deltas_vs_logged") or {}).get("Recall@20") is not None]
        if _rmd:
            w("- RM's own reload of the same LGMRec checkpoints (recomputed minus logged R@20): " + "; ".join(
                f"{d_} {sci(s_, 2)}" for _, d_, s_ in _rmd) + f"; max |difference| {sci(max(_rmd)[0], 3)} ({max(_rmd)[1]}) [RM].")
        w()
    # interface contrast (why no interface rule is claimed)
    _g1 = [c for c in _XA if c["cls"] == "C1"]
    _g2 = [c for c in _XA if c["cls"] == "C2"]
    if _g1 and _g2:
        _a1, _a2 = sum(c["clears"] for c in _g1), sum(c["clears"] for c in _g2)
        _s1, _s2 = sum(c["clears"] and c["dr"] < 0 for c in _g1), sum(c["clears"] and c["dr"] < 0 for c in _g2)
        _pA = fisher_2x2(_a1, len(_g1) - _a1, _a2, len(_g2) - _a2)
        _pS = fisher_2x2(_s1, len(_g1) - _s1, _s2, len(_g2) - _s2)
        w(f"- Frozen-graph class (C1: {', '.join(sorted({c['m'] for c in _g1}))}) clears in {_a1} of {len(_g1)}, "
          f"live-projected class (C2: {', '.join(sorted({c['m'] for c in _g2}))}) in {_a2} of {len(_g2)}: Fisher exact "
          f"p = {fx(D(_pA), 3)}; counting only negative deltas: {_s1} of {len(_g1)} vs {_s2} of {len(_g2)}, p = "
          f"{fx(D(_pS), 3)} [EX, K1, SR].")
    w()

# ------------------------------------------------------------------ 21k averaging vs deletion
_SVj = _rj("phase_exact/screen_vs_exact.json") or {}
_SVc = _SVj.get("cells", []) if isinstance(_SVj, dict) else _SVj
# Metadata, not results: why averaging and deletion differ, per model (brief R6, from the code).
_CAUSE = [("rank-equivalent", ("vbpr", "mentor", "freedom", "lattice")),
          ("shared normalizer / renormalized", ("lgmrec", "mmgcn")),
          ("content table restored at reload", ("mgcn", "smore", "gume")),
          ("constant-feature graph", ("damrs",)),
          ("branch keeps its own parameters", ("cohesion",))]
if _SVc:
    _CX = _rj("phasex_crossarch/crossarch_knockout.json") or []
    _bands = sorted({D(r["noise_hint"]) for r in _CX if r.get("noise_hint") is not None})
    _lgc = next((r for r in _K1 if r.get("model") == "lightgcn" and r.get("MDE_R@20") is not None), None)
    _bl = [("CX noise_hint", b) for b in _bands]
    if _lgc:
        _bl.append(("K1 LightGCN floor", D(_lgc["MDE_R@20"])))
    w("### 21k. Test-time averaging vs deletion on the same checkpoints (tab:subst; R6)")
    w()
    w(f"- Cells measured both ways: {len(_SVc)}; bands: " + ", ".join(f"{lab} {sci(b, 4)}" for lab, b in _bl) + " [SV, CX, K1].")
    for _blab, _band in _bl:
        w()
        w(f"Band = {_blab} ({sci(_band, 4)}):")
        w()
        w("| cause (models) | cells | identical | sign flips | wrongly inside band | wrongly outside band | src |")
        w("|---|---|---|---|---|---|---|")
        _tot = [0, 0, 0, 0, 0]
        for _lab, _mods in _CAUSE:
            _cc = [c for c in _SVc if c["model"] in _mods]
            _row = [len(_cc),
                    sum(1 for c in _cc if D(c["screen"]) == D(c["exact"])),
                    sum(1 for c in _cc if D(c["screen"]) * D(c["exact"]) < 0),
                    sum(1 for c in _cc if abs(D(c["exact"])) > _band >= abs(D(c["screen"]))),
                    sum(1 for c in _cc if abs(D(c["screen"])) > _band >= abs(D(c["exact"])))]
            _tot = [a + b for a, b in zip(_tot, _row)]
            w(_src_row(f"{_lab} ({', '.join(_mods)})", *_row, "SV"))
        w(_src_row("**total**", *[f"**{x}**" for x in _tot], "SV"))
        _unc = [c for c in _SVc if not any(c["model"] in m for _, m in _CAUSE)]
        if _unc:
            w(f"- Cells with no cause group: {', '.join(c['model'] + '/' + c['dataset'] for c in _unc)} [SV].")
    w()
    _lgs = [(abs(D(c["screen"])) / abs(D(c["exact"])), c["dataset"]) for c in _SVc if c["model"] == "lgmrec"]
    if _lgs:
        w(f"- LGMRec |averaging| / |deletion|: " + ", ".join(f"{d_} {fx(v, 2)}" for v, d_ in _lgs)
          + f" (range {fx(min(_lgs)[0], 1)}-{fx(max(_lgs)[0], 1)}) [SV].")
    _mms = [c for c in _SVc if c["model"] == "mmgcn"]
    if _mms:
        w("- MMGCN averaging vs deletion: " + "; ".join(
            f"{c['dataset']} {sci(c['screen'], 3)} vs {sci(c['exact'], 3)} (same sign: "
            f"{'yes' if D(c['screen']) * D(c['exact']) > 0 else 'no'})" for c in _mms) + " [SV].")
    _cf = {c["d"]: c["f"] for c in _XA if c["m"] == "cohesion"}
    _chs = [c for c in _SVc if c["model"] == "cohesion" and c["dataset"] in _cf]
    if _chs:
        w("- COHESION averaging / floor vs deletion / floor: " + "; ".join(
            f"{c['dataset']} {fx(abs(D(c['screen'])) / _cf[c['dataset']], 2)} vs {fx(abs(D(c['exact'])) / _cf[c['dataset']], 2)}"
            for c in _chs) + f" (averaging range {fx(min(abs(D(c['screen'])) / _cf[c['dataset']] for c in _chs), 2)}-"
            f"{fx(max(abs(D(c['screen'])) / _cf[c['dataset']] for c in _chs), 2)}; deletion range "
            f"{fx(min(abs(D(c['exact'])) / _cf[c['dataset']] for c in _chs), 1)}-{fx(max(abs(D(c['exact'])) / _cf[c['dataset']] for c in _chs), 1)}) [SV, EX].")
    _vb = [r for r in _EX if r.get("model") == "vbpr" and "error" not in r
           and any(c["model"] == "vbpr" and c["dataset"] == r["dataset"] for c in _SVc)]
    if _vb:
        _vi = [(D(r["attribution"]["image_inputmean_rank_equiv"]), r["dataset"]) for r in _vb
               if "image_inputmean_rank_equiv" in (r.get("attribution") or {})]
        _vt = [(D(r["attribution"]["text_inputmean_rank_equiv"]), r["dataset"]) for r in _vb
               if "text_inputmean_rank_equiv" in (r.get("attribution") or {})]
        w(f"- VBPR: max over users of the spread of the per-item score shift under averaging, over the {len(_vb)} VBPR cells "
          f"in SV: image {sci(max(_vi)[0], 2)} ({max(_vi)[1]}), text {sci(max(_vt)[0], 2)} ({max(_vt)[1]}) [EX].")
    w()

# ------------------------------------------------------------------ 21l data, reproduction, datasets per instrument
w("### 21l. Data, reproduction, datasets per instrument, ranks (3.1, 3.6, R27)")
w()
_sizes = {}
for _r in _K1:
    if _r.get("model") == "freedom" and _r.get("n_users") is not None:
        _sizes[_r["dataset"]] = (_r["n_users"], _r["n_items"], "K1")
for _r in (_rj("phase_micro/knockout_microlens.json") or []):
    if _r.get("model") == "freedom" and _r.get("n_users") is not None:
        _sizes[_r["dataset"]] = (_r["n_users"], _r["n_items"], "KM")
w("- Users / items: " + "; ".join(f"{d_} {v[0]:,} / {v[1]:,} [{v[2]}]" for d_, v in _sizes.items()) + ".")
_dsm = (jload("results/phase0/dataset_stats.json") or {}).get("microlens")
_blm = bool(_dsm)
if _blm:
    _u, _i = _dsm["users"], _dsm["items"]
    _tr_, _va, _te = (_dsm["split_counts"][k] for k in ("train", "valid", "test"))
    _tot = _tr_ + _va + _te
    w(f"- MicroLens split: {_tr_:,} / {_va:,} / {_te:,} = {_tot:,} interactions; proportions "
      f"{fx(D(_tr_) / _tot, 3)} / {fx(D(_va) / _tot, 3)} / {fx(D(_te) / _tot, 3)} (users {_u:,}, items {_i:,}) [DS].")
    _cmj = _rj("phase_micro/controls_microlens.json") or {}
    _npos = sorted({v.get("n_positives") for v in (_cmj.get("freedom_rank_stat") or {}).values()} - {None})
    if _npos:
        w(f"- Cross-check: the K=1000 rank statistic on FREEDOM/MicroLens counts n_positives = "
          f"{', '.join(f'{x:,}' for x in _npos)}, {'equal to' if _npos == [_te] else 'NOT equal to'} the logged "
          f"test-split size [CM, DS].")
w("- NOT certifiable from results/: the Amazon interaction counts and the Baby split proportions "
  "(no artifact under results/ records n_train/n_valid/n_test for an Amazon dataset); the data "
  "release year and last timestamp are likewise not recorded under results/ [absent from results/].")
_dims = []
for _dd in (_AL.get("datasets") or [])[:1]:
    _dims.append("Amazon " + ", ".join(f"{k} {v['dim']}-d" for k, v in _dd["streams"].items() if k.startswith("raw_")) + " [AL]")
for _dd in (_AM.get("datasets") or [])[:1]:
    _dims.append("MicroLens " + ", ".join(f"{k} {v['dim']}-d" for k, v in _dd["streams"].items() if k.startswith("raw_")) + " [AM]")
if _dims:
    w("- Feature dimensions: " + "; ".join(_dims) + ".")
_VD = _rj("phase_micro/video_feat_degeneracy.json") or {}
if _VD.get("streams"):
    w("- Mean pairwise cosine of the raw features, 2 dp from full precision: " + "; ".join(
        f"{k} {fx(v['mean_pairwise_cosine'], 2)}" for k, v in _VD["streams"].items()
        if v.get("mean_pairwise_cosine") is not None) + " [VD].")
_KF = _rj("phase1/knockout_finer.json") or {}
_aux = {d_: v.get("freedom_aux_sanity") for d_, v in _KF.items() if isinstance(v, dict) and v.get("freedom_aux_sanity")}
if _aux:
    w("- Zeroing FREEDOM's auxiliary projections changes R@20 by: " + ", ".join(
        f"{d_} {format(D(v['dR@20']), 'f')}" for d_, v in _aux.items()) + " (default-patience pinned checkpoints) [KF].")
_PB = _rj("phase0/published_comparison.json") or {}
_TRt = {(r["model"], r["dataset"]): r["test_result"] for r in (_rj("phase_convergence/table_rerun.json") or []) if "test_result" in r}
_RMt = {(r["model"], r["dataset"]): r["recomputed"] for r in (_rj("phase0/repro_metrics.json") or []) if r.get("recomputed")}
for _lab, _src, _tag in (("converged (seed 2024)", _TRt, "TR"), ("default patience (pinned checkpoints)", _RMt, "RM")):
    _rel = []
    for _c in (_PB.get("comparison") or []):
        _t = _src.get((_c["model"], _c["dataset"]))
        if not _t:
            continue
        for _k, _pk in (("Recall@20", "published_R20"), ("NDCG@20", "published_N20")):
            _rel.append((abs(100 * (D(_t[_k]) - D(_c[_pk])) / D(_c[_pk])), f"{_c['model']}/{_c['dataset']} {_k}"))
    if _rel:
        w(f"- Published-number check, {_lab}: |relative difference| over {len(_rel)} comparisons "
          f"{fx(min(_rel)[0], 2)}-{fx(max(_rel)[0], 2)}% (1 dp {fx(min(_rel)[0], 1)}-{fx(max(_rel)[0], 1)}%); largest "
          f"{max(_rel)[1]} [{_tag}, PB].")
_stored = [abs(D(_c[k])) for _c in (_PB.get("comparison") or []) for k in ("rel_delta_R20_pct", "rel_delta_N20_pct") if k in _c]
if _stored:
    w(f"- The stored (1-dp) default-patience deltas in PB span {format(min(_stored), 'f')}-{format(max(_stored), 'f')}% [PB].")
_ranks = collections.defaultdict(list)
for (_mo, _d), _t in _TRt.items():
    _ranks[_d].append((D(_t["Recall@20"]), _mo))
_rk = []
for _d in _DS4:
    if _ranks.get(_d):
        _srt = sorted(_ranks[_d], key=lambda z: (-z[0], z[1]))
        _pos = [m_ for _, m_ in _srt].index("freedom") + 1 if any(m_ == "freedom" for _, m_ in _srt) else None
        _rk.append(f"{_d} {_pos} of {len(_srt)}")
w("- FREEDOM's rank by converged R@20 among the re-run models: " + "; ".join(_rk) + " [TR].")
_fm, _lm = _TRt.get(("freedom", "microlens")), _TRt.get(("lightgcn", "microlens"))
if _fm and _lm:
    _gap = 100 * (1 - D(_fm["Recall@20"]) / D(_lm["Recall@20"]))
    w(f"- MicroLens, converged: FREEDOM {fx(_fm['Recall@20'], 6)} vs LightGCN {fx(_lm['Recall@20'], 6)}: FREEDOM is "
      f"{fx(_gap, 2)}% ({fx(_gap, 1)}%) below [TR].")
_cov = {"deletion (converged FREEDOM)": sorted(d_ for d_ in _DS4 if (d_, "image") in _DEL),
        "deletion (default patience, Elec)": ["elec"] if _ELEC else [],
        "retraining (converged)": sorted({k[0] for k in _HOLD}),
        "encoder control": sorted({k.split("/")[0] for k in _CL}),
        "alignment": sorted(_ALIGN)}
w("- Datasets per instrument: " + "; ".join(f"{k}: {len(v)} ({', '.join(v)})" for k, v in _cov.items())
  + f"; cross-architecture: {len(_XA)} cells, {sum(1 for c in _XA if c['d'] != 'microlens')} Amazon "
  f"[CK, K1, HO, CL, AL, AM, EX].")
w()

# ------------------------------------------------------------------ 21m abstract and headline numbers
w("### 21m. Abstract and headline numbers (F; E.2; 6.3)")
w()
_ai = [(abs(_DEL[(d_, "image")]["pct"]), d_) for d_ in _AMZ3 if (d_, "image") in _DEL]
if _ai:
    w(f"- Mean image-deletion effect on the converged Amazon models, largest |d %|: {fx(max(_ai)[0], 2)}% "
      f"({max(_ai)[1]}; 1 dp {fx(max(_ai)[0], 1)}%) [CK].")
if "image" in _ELEC and _ai:
    _mx4 = max(max(_ai)[0], abs(_ELEC["image"]["pct"]))
    w(f"- Including Electronics (one default-patience checkpoint, |d %| {fx(abs(_ELEC['image']['pct']), 2)}%): largest "
      f"{fx(_mx4, 2)}% (1 dp {fx(_mx4, 1)}%) [CK, K1].")
_ri = [(abs(_HOLD[(d_, "no_image", "Recall@20")]["pct"]), d_) for d_ in _AMZ3 if (d_, "no_image", "Recall@20") in _HOLD]
_rt = [(abs(_HOLD[(d_, "no_text", "Recall@20")]["pct"]), d_) for d_ in _AMZ3 if (d_, "no_text", "Recall@20") in _HOLD]
if _ri:
    w(f"- Retraining without images, the three Amazon datasets: {fx(min(_ri)[0], 2)}-{fx(max(_ri)[0], 2)}% "
      f"(1 dp {fx(min(_ri)[0], 1)}-{fx(max(_ri)[0], 1)}%): " + ", ".join(f"{d_} {fx(v, 2)}%" for v, d_ in _ri)
      + "; Sports and Clothing only: " + "-".join(fx(v, 1) for v, d_ in sorted(_ri) if d_ != "baby") + "% [HO].")
if _rt:
    w(f"- Retraining without text, the three Amazon datasets: {fx(min(_rt)[0], 2)}-{fx(max(_rt)[0], 2)}% "
      f"(1 dp {fx(min(_rt)[0], 1)}-{fx(max(_rt)[0], 1)}%; 0 dp {fx(min(_rt)[0], 0)}-{fx(max(_rt)[0], 0)}%) [HO].")
_mi_r, _mt_r = _HOLD.get(("microlens", "no_image", "Recall@20")), _HOLD.get(("microlens", "no_text", "Recall@20"))
_mi_d = _DEL.get(("microlens", "image"))
if _mi_r and _mt_r and _mi_d:
    w(f"- MicroLens: retraining without images {fx(_mi_r['pct'], 2, True)}% ({fx(_mi_r['pct'], 1, True)}%, {_mi_r['verdict']}); "
      f"without text {fx(_mt_r['pct'], 2, True)}% ({fx(_mt_r['pct'], 1, True)}%, {_mt_r['verdict']}); deleting the trained "
      f"image term {fx(_mi_d['pct'], 2, True)}% ({fx(_mi_d['pct'], 1, True)}%, {_mi_d['verdict']}) [HO, CK].")
if _XA:
    w(f"- Cross-architecture: {len(_clr)} of {len(_XA)} cells clear their own floor; {len(_BH)} survive BH, "
      f"{len(_HOLM)} survive Holm [EX, K1, SR].")
w()

# ------------------------------------------------------------------ 21n registered hypotheses
_pmf = ROOT / "results" / "phase_micro" / "PREREG.md"
_pmt = _pmf.read_text() if _pmf.is_file() else ""


def _quote(pattern, text):
    m_ = re.search(pattern, text, flags=re.S)
    return " ".join(m_.group(0).split()) if m_ else None


w("### 21n. Outcomes of the registered MicroLens and holdout hypotheses (R18, R19; E.2)")
w()
_h1 = _quote(r"\*\*H1 \(primary, causal\)\.\*\*.*?clears it\.", _pmt)
_sfj = _rj("phase_micro/significance_freedom_microlens.json") or {}
_sfs = _sfj.get("streams") or {}
if _h1:
    w(f"- H1 as registered: \"{_h1}\" [PM].")
if ("microlens", "image") in _DEL and ("microlens", "text") in _DEL:
    w(f"- H1 outcome, converged models: image deletion {_DEL[('microlens', 'image')]['verdict']} "
      f"({fx(_DEL[('microlens', 'image')]['xfl'], 2)}x F_l, {fx(_DEL[('microlens', 'image')]['xfp'], 2)}x F_p), text deletion "
      f"{_DEL[('microlens', 'text')]['verdict']} [CK]" + (
          f"; default-patience models: image {_sfs['image']['verdict']} ({fx(_sfs['image']['abs_mean_over_F_level'], 2)}x F_l, "
          f"{fx(_sfs['image']['abs_mean_over_F_paired'], 2)}x F_p), text {_sfs['text']['verdict']} [SF]." if "image" in _sfs else "."))
_h2 = _quote(r"\*\*H2 \(mechanism, directional\)\.\*\*.*?preserved\.", _pmt)
_h2n = _quote(r"H2's direction \(`h_img > h_txt`\) was already visible in the dry run\.", _pmt)
if _h2:
    w(f"- H2 as registered: \"{_h2}\"" + (f" The file also states: \"{_h2n}\"" if _h2n else "") + " [PM].")
for _src, _tg in ((_AM, "AM"), (_AD, "AD")):
    for _dd in (_src.get("datasets") or []):
        if _dd.get("dataset") == "microlens":
            _s = _dd["streams"]
            _hi, _ht = _s["h_img_graph"], _s["h_txt_graph"]
            w(f"- H2 outcome ({_tg}): h_img {fx(_hi['gap_abs'], 3)} [{fx(_hi['gap_abs_ci95'][0], 3)}, {fx(_hi['gap_abs_ci95'][1], 3)}] "
              f"vs h_txt {fx(_ht['gap_abs'], 3)} [{fx(_ht['gap_abs_ci95'][0], 3)}, {fx(_ht['gap_abs_ci95'][1], 3)}]; "
              f"intervals {'separate, image higher' if D(_hi['gap_abs_ci95'][0]) > D(_ht['gap_abs_ci95'][1]) else 'do NOT separate with image higher'} [{_tg}].")
if ("microlens", "image") in _DEL and ("microlens", "text") in _DEL:
    w(f"- H2's second half, deletion ordering on the converged models: |d_image| {sci(abs(_DEL[('microlens', 'image')]['mean']), 2)} "
      f"vs |d_text| {sci(abs(_DEL[('microlens', 'text')]['mean']), 2)} [CK].")
_h3 = _quote(r"\"Image-only FREEDOM does NOT\s+collapse relative to text-only on MicroLens\" is a reachable falsifier", _pmt)
_wrj = _rj("phase2/weight_sweep_retrain.json") or {}
_wr_ml = any(r.get("dataset") == "microlens" for r in (_wrj.get("runs") or []))
w(f"- H3 (the lambda retrain on MicroLens): {'run' if _wr_ml else 'NOT run'} (WR {'has' if _wr_ml else 'has no'} "
  f"MicroLens entry) [WR]." + (f" Its registered falsifier: {_h3} [PM]." if _h3 else ""))
if "microlens" in _IOT and ("microlens", "Recall@20") in _CON:
    _cc = _CON[("microlens", "Recall@20")]
    w(f"- The falsifier on the holdout arms: image-only vs text-only retrained FREEDOM {fx(_IOT['microlens'], 2, True)}% "
      f"(per-seed difference {sci(_cc['mean'], 2)}, {fx(_cc['xfp'], 2)}x its paired floor, p = {sci(_cc['p'], 3)}) [HO].")
_hv_arms = [(d_, dsd(_HOLD[(d_, "no_image", "Recall@20")]["arm"]) / dsd(_HOLD[(d_, "no_image", "Recall@20")]["full"]))
            for d_ in _DS4 if (d_, "no_image", "Recall@20") in _HOLD]
if _hv_arms:
    w("- H-var at convergence (registered value in 21b): " + ", ".join(f"{d_} {fx(v, 2)}" for d_, v in _hv_arms)
      + f"; at the default patience, 8 seeds: " + ", ".join(
          f"{d_} {fx(dsd(_HDA[(d_, 'no_image')]['arm']) / dsd(_HDA[(d_, 'no_image')]['full']), 2)}"
          for d_ in _DS4 if (d_, "no_image") in _HDA) + " [HO, HD].")
w()

# ---------------------------------------------------------------- s.21o (added 2026-09-23)
_dsx = jload("results/phase0/dataset_stats.json")
if _dsx:
    w("## 21o. Dataset statistics (scripts/dataset_stats.py -> results/phase0/dataset_stats.json)")
    w()
    w("Split = the pre-assigned per-interaction x_label (0/1/2 = train/valid/test); NOT chronological.")
    w("'temporal-test' = share of users (with train and test items) whose every test interaction is at")
    w("or after their last training interaction.")
    w()
    w("| dataset | users | items | interactions | density | train/valid/test | first..last (UTC) | temporal-test |")
    w("|---|---|---|---|---|---|---|---|")
    for _n, _v in _dsx.items():
        _sh = _v["split_share"]
        w(f"| {_n} | {_v['users']:,} | {_v['items']:,} | {_v['interactions']:,} | "
          f"{_v['density'] * 100:.3f}% | {_sh['train']:.3f}/{_sh['valid']:.3f}/{_sh['test']:.3f} | "
          f"{_v['first_interaction_utc']}..{_v['last_interaction_utc']} | "
          f"{_v['share_users_test_after_last_train']:.3f} |")
    w()
    w("Released content features (storage as stored, and at 32-bit floats):")
    w()
    for _n, _v in _dsx.items():
        for _m, _f in (_v.get("features") or {}).items():
            w(f"- {_n} {_m}: shape {tuple(_f['shape'])}, stored {_f['dtype']} = "
              f"{_f['bytes_on_disk'] / 1e6:,.1f} MB; at float32 = {_f['bytes_float32'] / 1e6:,.1f} MB")
    w()

_fcfg = jload("results/phase0/freedom_config.json")
if _fcfg:
    w("## 21p. FREEDOM configuration (one configuration for every dataset)")
    w()
    w("- " + "; ".join(f"{k} = {v}" for k, v in _fcfg["freedom"].items()) + " [recsys configs].")
    _ov = {k: v for k, v in _fcfg["dataset_overrides_of_training_keys"].items() if v}
    w(f"- Dataset configs overriding a model/training key: {_ov or 'none (no per-dataset tuning)'}.")
    w()

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(L))
print(f"wrote {OUT}  ({len(L)} lines)")
