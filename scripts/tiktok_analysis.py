"""One command for every registered TikTok number (PREREG_TIKTOK.md), in dependency order:

  1. scripts/tiktok_converged_knockout.py  exact image/text deletion on each finished converged
                                           full checkpoint (GPU, < 1 GB, incremental)
  2. scripts/tiktok_alignment.py           Table-3 alignment + T4 comparison (CPU)
  3. scripts/tiktok_graph_health.py        kNN-graph health, TikTok vs MicroLens vs Baby (CPU)
  4. scripts/tiktok_verdicts.py            retraining verdicts T1-T3, T4, LightGCN gap, identity
                                           self-test against the certified MicroLens/Amazon numbers
Each step reads the runs file as it is at that moment; nothing is trained, nothing in
results/_queue or the runs files is touched. Stops at the first failing step.
Usage: python scripts/tiktok_analysis.py [--skip-knockout]
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/workspace/MechInterp")
PY = sys.executable


def main() -> int:
    steps = [("tiktok_converged_knockout.py", {}),
             ("tiktok_alignment.py", {"CUDA_VISIBLE_DEVICES": ""}),
             ("tiktok_graph_health.py", {"CUDA_VISIBLE_DEVICES": ""}),
             ("tiktok_verdicts.py", {})]
    if "--skip-knockout" in sys.argv:
        steps = steps[1:]
    for script, env in steps:
        print(f"\n######## {script}", flush=True)
        rc = subprocess.run([PY, str(ROOT / "scripts" / script)], cwd=str(ROOT),
                            env={**os.environ, **env}).returncode
        if rc != 0:
            print(f"FAILED: {script} (exit {rc})")
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
