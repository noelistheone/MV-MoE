"""Seed-2024 converged deletion deltas (results/phase_c3conv/scored_<ds>.json) vs the
default-patience single-checkpoint screen (results/phase_exact/exact_crossarch.json = Table 4).
Different checkpoints (default patience vs patience 100 / cap 3000), so this is a descriptive
comparison, not a replication test. -> results/phase_c3conv/compare_s2024_vs_table4.json"""
import json
from pathlib import Path
ROOT = Path("/workspace/MechInterp")
t4 = {f"{r['model']}/{r['dataset']}": r for r in json.loads((ROOT / "results/phase_exact/exact_crossarch.json").read_text())}
out, lines = {}, []
for ds in ["baby", "sports", "clothing", "microlens"]:
    f = ROOT / f"results/phase_c3conv/scored_{ds}.json"
    if not f.is_file():
        continue
    doc = json.loads(f.read_text())
    for key, r in list(doc["rows"].items()) + list(doc["refused"].items()):
        if r["seed"] != 2024:
            continue
        cell = f"{r['model']}/{ds}"
        o = t4.get(cell, {})
        e = {"scored": key in doc["rows"], "refused_reason": None if key in doc["rows"] else r["guard"]["reason"],
             "conv_base_R20": r.get("baseline", {}).get("Recall@20"),
             "t4_base_R20": o.get("arms", {}).get("baseline", {}).get("metrics", {}).get("Recall@20"),
             "t4_error": o.get("error"), "t4_MDE_R@20": o.get("MDE_R@20"), "t4_MDE_source": o.get("MDE_source")}
        for s in ("image", "text"):
            conv = (r.get("arms") or {}).get(s)
            e[f"conv_d{s}"] = conv["delta"]["Recall@20"] if conv else None
            e[f"t4_d{s}"] = o.get("arms", {}).get(f"{s}_knockout", {}).get("dR@20")
            a, b = e[f"conv_d{s}"], e[f"t4_d{s}"]
            e[f"{s}_same_sign"] = (None if a is None or b is None else (a > 0) == (b > 0) if a and b else a == b)
        out[cell] = e
        lines.append(f"{cell:20s} conv img {e['conv_dimage'] if e['conv_dimage'] is not None else float('nan'):+.6f} "
                     f"t4 img {e['t4_dimage'] if e['t4_dimage'] is not None else float('nan'):+.6f} | "
                     f"conv txt {e['conv_dtext'] if e['conv_dtext'] is not None else float('nan'):+.6f} "
                     f"t4 txt {e['t4_dtext'] if e['t4_dtext'] is not None else float('nan'):+.6f} "
                     f"{'' if e['scored'] else '[REFUSED: ' + e['refused_reason'] + ']'}{' [T4 ERR]' if e['t4_error'] else ''}")
(ROOT / "results/phase_c3conv/compare_s2024_vs_table4.json").write_text(json.dumps(out, indent=1))
print("\n".join(lines))
