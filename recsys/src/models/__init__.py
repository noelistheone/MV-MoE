"""Model registry: the thirteen recommenders measured in the paper."""
from typing import Type

from src.common.abstract_recommender import AbstractRecommender
from src.models.freedom import FREEDOM
from src.models.lgmrec import LGMRec
from src.models.lightgcn import LightGCN
from src.models.vbpr import VBPR
from src.models.mmgcn import MMGCN
from src.models.lattice import LATTICE
from src.models.bm3 import BM3
from src.models.mgcn import MGCN
from src.models.mentor import MENTOR
from src.models.damrs import DAMRS
from src.models.smore import SMORE
from src.models.cohesion import COHESION
from src.models.gume import GUME

MODEL_REGISTRY: dict[str, Type[AbstractRecommender]] = {
    "freedom": FREEDOM,
    "lgmrec": LGMRec,
    "lightgcn": LightGCN,
    "vbpr": VBPR,
    "mmgcn": MMGCN,
    "lattice": LATTICE,
    "bm3": BM3,
    "mgcn": MGCN,
    "mentor": MENTOR,
    "damrs": DAMRS,
    "smore": SMORE,
    "cohesion": COHESION,
    "gume": GUME,
}


def get_model(name: str) -> Type[AbstractRecommender]:
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key]


__all__ = ["MODEL_REGISTRY", "get_model", "FREEDOM", "LGMRec", "LightGCN", "VBPR", "MMGCN", "LATTICE", "BM3", "MGCN", "MENTOR", "DAMRS", "SMORE", "COHESION", "GUME"]
