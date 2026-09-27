"""PREREG_C3CONV_AMEND1.md A7: COHESION/MicroLens seed 2024 was scored on CPU (GPU memory was full at
the time). Score the same checkpoint on GPU with the same instrument (exp_c3conv_score.score_one),
compare arms and guard with the stored CPU row, and record both. The scored row itself is NOT
replaced. -> results/phase_c3conv/a7_device_check.json"""
import json, sys, time
from pathlib import Path
ROOT = Path("/workspace/MechInterp")
sys.path.insert(0, str(ROOT / "scripts"))
import torch  # noqa: E402
import exp_c3conv_score as sc  # noqa: E402

tot = torch.cuda.get_device_properties(0).total_memory / 2**30
torch.cuda.set_per_process_memory_fraction(min(1.0, 12 / tot), 0)
sc.RUN_DEVICE = {"device": "cuda:0", "gpu_cap_gb": 12, "torch": torch.__version__}
recs, _ = sc.collect_runs("microlens")
rec = recs[("cohesion", 2024)]
cpu_row = json.loads((sc.C3 / "scored_microlens.json").read_text())["rows"]["cohesion/s2024"]
gpu_row = sc.score_one("cohesion", "microlens", rec, "cuda:0")
cmp = {}
for s in ("image", "text", "both"):
    a, b = cpu_row["arms"].get(s), gpu_row["arms"].get(s)
    if a and b:
        cmp[s] = {m: {"cpu": a["delta"][m], "gpu": b["delta"][m], "abs_diff": abs(a["delta"][m] - b["delta"][m])}
                  for m in sc.METRICS}
out = {"checkpoint": rec["ckpt_path"], "cpu_device": cpu_row.get("run_device"), "gpu_device": gpu_row.get("run_device"),
       "baseline_R20": {"cpu": cpu_row["baseline"]["Recall@20"], "gpu": gpu_row["baseline"]["Recall@20"],
                        "logged": rec["test_result"]["Recall@20"]},
       "guard": {"cpu": cpu_row["guard"]["ok"], "gpu": gpu_row["guard"]["ok"],
                 "gpu_G2": {k: gpu_row["guard"]["G2"].get(k) for k in ("mode", "draw_mean", "draw_sd", "z_logged", "ok")}},
       "deltas": cmp,
       "max_abs_diff_R20": max(v["Recall@20"]["abs_diff"] for v in cmp.values()),
       "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "gpu_row": gpu_row}
(sc.C3 / "a7_device_check.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: out[k] for k in ("baseline_R20", "guard", "max_abs_diff_R20")}, indent=1))
for s, v in cmp.items():
    print(s, {m: f"cpu {x['cpu']:+.6f} gpu {x['gpu']:+.6f}" for m, x in v.items() if m == "Recall@20"})
