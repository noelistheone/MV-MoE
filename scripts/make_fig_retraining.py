"""Figure 1 of the SAC 2027 draft: per-seed FREEDOM retraining effects against the level floor.

For each dataset, one dot per seed: 100 * (R@20 without the modality - R@20 of the full model with
the same seed) / mean full R@20, so the dots' mean equals Table 1's Delta%. The grey band is
+/- the level floor (2 sd of full-model R@20 over the eight seeds, as % of its mean). Reads only
the converged run records (stopping_step 100, epochs_cap 3000), pooled as in holdout_verdicts.py.
Palette: reference categorical slots 1-2 (#2a78d6, #eb6834), validated all-pairs, light mode;
shape is a second channel (circle = image, triangle = text).
Output: results/phase_paper/fig-retraining.pdf and results/phase_paper/fig_retraining.json
"""
import json, statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
SRC = {"Baby": (R / "phase_holdout/freedom_p100_runs.json", "baby"),
       "Sports": (R / "phase_holdout/freedom_p100_sports_runs.json", "sports"),
       "Clothing": (R / "phase_holdout/freedom_p100_clothing_runs.json", "clothing"),
       "MicroLens": (R / "phase_holdout/freedom_p100_microlens_runs.json", "microlens"),
       "TikTok": (R / "phase_shortvideo2/freedom_p100_tiktok_runs.json", "tiktok")}
COL = {"no_image": "#2a78d6", "no_text": "#eb6834"}
MK = {"no_image": "o", "no_text": "^"}
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d3"


def load(path, ds):
    out = {}
    for r in json.loads(Path(path).read_text()):
        if r.get("dataset") != ds or "test_result" not in r:
            continue
        if r.get("stopping_step") != 100 or r.get("epochs_cap") != 3000:
            continue
        out.setdefault(r["condition"], {})[r["seed"]] = r["test_result"]["Recall@20"]
    return out


def main():
    data = {}
    for name, (p, ds) in SRC.items():
        runs = load(p, ds)
        full = runs["full"]; mu = statistics.mean(full.values())
        band = 100 * 2 * statistics.stdev(full.values()) / mu
        d = {c: [100 * (runs[c][s] - full[s]) / mu for s in sorted(runs[c]) if s in full]
             for c in ("no_image", "no_text")}
        data[name] = {"band_pct": band, **d, "mean": {c: statistics.mean(v) for c, v in d.items()}}
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.40), sharey=False)
    names = list(SRC)
    for ax, cond, title in ((axes[0], "no_image", "without images"), (axes[1], "no_text", "without text")):
        for i, n in enumerate(names):
            b = data[n]["band_pct"]
            ax.add_patch(plt.Rectangle((i - 0.34, -b), 0.68, 2 * b, color=BAND, lw=0, zorder=1))
            vals = data[n][cond]
            xs = [i - 0.21 + 0.42 * k / (len(vals) - 1) for k in range(len(vals))]
            ax.scatter(xs, vals, s=16, marker=MK[cond], color=COL[cond], edgecolor="white",
                       linewidth=0.4, zorder=3)
            ax.plot([i - 0.3, i + 0.3], [data[n]["mean"][cond]] * 2, color=INK, lw=1.0, zorder=4)
        ax.axhline(0, color=INK2, lw=0.5, zorder=2)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_xlim(-0.55, len(names) - 0.45)
        ax.set_title(f"Retraining {title}", fontsize=8.5, color=INK, pad=3)
        ax.yaxis.grid(True, color=GRID, lw=0.4, zorder=0)
        ax.tick_params(length=2, width=0.5, colors=INK2, pad=1.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(INK2)
    axes[0].set_ylabel(r"$\Delta$ Recall@20 (%)", color=INK, labelpad=2)
    fig.tight_layout(pad=0.3, w_pad=1.5)
    out = ROOT / "results/phase_paper/fig-retraining.pdf"
    fig.savefig(out)
    (R / "phase_paper").mkdir(exist_ok=True)
    (R / "phase_paper/fig_retraining.json").write_text(json.dumps(data, indent=1))
    print(out)
    for n in names:
        print(n, round(data[n]["band_pct"], 2), {c: round(data[n]["mean"][c], 2) for c in ("no_image", "no_text")},
              {c: (round(min(data[n][c]), 1), round(max(data[n][c]), 1)) for c in ("no_image", "no_text")})


if __name__ == "__main__":
    main()
