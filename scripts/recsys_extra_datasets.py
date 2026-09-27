"""Runtime overlay that lets the read-only Recsys harness load datasets living under MechInterp.

The harness's Config reads configs/dataset/<name>.yaml from /workspace/Recsys/configs. For a
dataset name that has NO yaml there, this overlay makes Config fall back to
/workspace/MechInterp/configs/datasets_extra/<name>.yaml. Existing datasets are untouched
(the Recsys yaml always wins when it exists). The extra yaml sets an ABSOLUTE data_path; RecDataset
computes project_root / data_path, and pathlib returns the absolute path unchanged.

Usage (must run before Config is constructed):
    import recsys_extra_datasets; recsys_extra_datasets.install()
    from phase0_noisefloor import train_one
    train_one("freedom", "tiktok", 2024, "cuda", extra_overrides={...})
Nothing is written into /workspace/Recsys.
"""
from __future__ import annotations

import sys
from pathlib import Path

RECSYS = Path("/workspace/Recsys")
EXTRA = Path("/workspace/MechInterp/configs/datasets_extra")
_installed = False


def install() -> None:
    global _installed
    if _installed:
        return
    if str(RECSYS) not in sys.path:
        sys.path.insert(0, str(RECSYS))
    from src.utils import configurator

    orig = configurator.Config._load_yaml

    def _load_yaml(path: Path):
        path = Path(path)
        if not path.is_file() and path.parent.name == "dataset":
            alt = EXTRA / path.name
            if alt.is_file():
                return orig(alt)
        return orig(path)

    configurator.Config._load_yaml = staticmethod(_load_yaml)
    _installed = True
