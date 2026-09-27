"""Read-only: build each measured model on Baby (CPU, no checkpoint) and report every sparse graph's nnz."""
import sys, json
sys.path.insert(0, '/workspace/MechInterp/scripts')
import phase0_noisefloor as p0
from src.utils import Config, set_seed
from src.data.dataset import RecDataset
from src.data.graph_utils import build_norm_adj
import torch
out = {}
for m in ["freedom","lgmrec","lightgcn","vbpr","mmgcn","lattice","bm3","mgcn","mentor","damrs","smore","cohesion","gume"]:
    try:
        cfg = Config(m, "baby", cli_overrides={"seed": 2024, "show_progress": False})
        set_seed(2024)
        ds = RecDataset(cfg)
        norm_adj = build_norm_adj(ds.train_matrix, ds.n_users, ds.n_items)
        model = p0.build_model(m, cfg, ds, norm_adj, "cpu")
        graphs = {}
        for name, t in list(model.named_buffers()) + [(k, v) for k, v in vars(model).items() if torch.is_tensor(v)]:
            if torch.is_tensor(t) and (t.is_sparse or getattr(t, "layout", None) in (torch.sparse_coo, torch.sparse_csr)):
                nnz = t._nnz() if t.is_sparse else t.values().numel()
                graphs[name] = {"shape": list(t.shape), "nnz": int(nnz)}
        out[m] = graphs
        empty = [k for k, v in graphs.items() if v["nnz"] == 0]
        print(f"{m:9s} graphs={len(graphs)} empty={empty} " + "; ".join(f"{k}:{v['nnz']}" for k, v in graphs.items()), flush=True)
    except Exception as e:
        out[m] = {"error": repr(e)[:300]}
        print(f"{m:9s} ERROR {repr(e)[:200]}", flush=True)
json.dump(out, open('/workspace/MechInterp/results/phase_c3conv_g11/probe_graphs_all_models_baby.json', 'w'), indent=1)
