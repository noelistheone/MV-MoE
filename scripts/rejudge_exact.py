"""Re-judge the exact cross-architecture table against the CURRENT floors.

The knockout deltas are fixed once measured; only the verdicts move as noise floors land.
Re-running exp_exact_crossarch.py would recompute 50 cells of model evaluation for nothing.
This reloads results/phase_exact/exact_crossarch.json, re-reads every floor through
phase1_knockout.load_mde (which prefers the estimate built from the most seeds and now also
returns the seed count), and rewrites the MDE / significance / reliability fields in place.

Run after every floor lands, then regenerate paper/revision/FACTS.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/workspace/MechInterp")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "interp"))
sys.path.insert(0, "/workspace/Recsys")
from phase1_knockout import load_mde, verdict_reliability  # noqa: E402

F = ROOT / "results" / "phase_exact" / "exact_crossarch.json"
# Cells whose floor is contaminated by early-stopping truncation and must not adjudicate
# anything until re-measured at patience 100 (results/phase_mde/TRUNCATION_AUDIT.md).
CONTAMINATED = {("lattice", "baby"), ("mgcn", "baby"), ("mgcn", "clothing"), ("vbpr", "baby")}


def main() -> int:
    rows = json.loads(F.read_text())
    changed = 0
    for r in rows:
        if "error" in r:
            continue
        m, ds = r["model"], r["dataset"]
        mde, src = load_mde(m, ds)
        n = None
        if src and "(n=" in src:
            n = int(src.split("(n=")[1].split(")")[0])
        elif src:
            n = 5
        contaminated = (m, ds) in CONTAMINATED
        if contaminated:
            mde, src, n = None, (src + "  [CONTAMINATED-WITHHELD]" if src else None), n
        if r.get("MDE_R@20") != mde or r.get("MDE_source") != src:
            changed += 1
        r["MDE_R@20"], r["MDE_source"], r["MDE_n_seeds"] = mde, src, n
        r["floor_contaminated"] = contaminated
        for name, v in r["arms"].items():
            if name == "baseline" or "dR@20" not in v:
                continue
            d = v["dR@20"]
            v["significant_vs_MDE"] = (abs(d) > mde) if mde else None
            v["verdict_reliability"] = verdict_reliability(d, mde, n)
    F.write_text(json.dumps(rows, indent=2))

    ok = [r for r in rows if "error" not in r]
    with_floor = [r for r in ok if r.get("MDE_R@20")]
    print(f"re-judged {len(ok)} cells; {changed} MDE fields changed; "
          f"{len(with_floor)} now carry a floor, {len(ok)-len(with_floor)} PENDING "
          f"({len([r for r in ok if r.get('floor_contaminated')])} withheld as contaminated)")
    sig = [(r["model"], r["dataset"], r["arms"]["image_knockout"]["dR@20"], r["MDE_R@20"])
           for r in with_floor
           if r["arms"].get("image_knockout", {}).get("significant_vs_MDE")]
    print(f"\nimage knockout SIGNIFICANT vs its own floor in {len(sig)} cells:")
    for m, ds, d, mde in sorted(sig):
        print(f"  {m:9s}/{ds:10s} dR@20={d:+.6f}  MDE={mde:.5f}  ({abs(d)/mde:.1f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
