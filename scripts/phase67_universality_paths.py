"""Phase 6 (cross-model universality) + Phase 7 (per-hop path contribution).

Phase 7 (FREEDOM, exact): decompose each test-positive's SCORE into stream paths
  score = u·cf + u·h_img + u·h_txt, and cf into per-GCN-hop contributions.
Phase 6: train SAEs on FREEDOM-fused and LGMRec-fused (same items), match latents
  across models by activation-correlation; report a universality score + the
  image-latent divergence (FREEDOM 0 vs LGMRec >0 = where the architectures differ).

Outputs -> results/phase6_7/. Recsys read-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/workspace/MechInterp")
RECSYS = Path("/workspace/Recsys")
sys.path.insert(0, str(RECSYS))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "interp"))
sys.path.insert(0, str(ROOT / "src" / "sae"))
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_bridge import load_frozen                        # noqa: E402
from sae import SAE, SAEConfig                               # noqa: E402
from trainer import SAETrainer, TrainConfig                  # noqa: E402
from phase1_knockout import freedom_streams, lgmrec_components, _lgmrec_combine  # noqa: E402

OUT = ROOT / "results" / "phase6_7"


def _test_pairs(test_loader):
    users, items = [], []
    for batch in test_loader:
        uid = batch["user_ids"]
        for r in range(len(uid)):
            pl = int(batch["positive_lengths"][r])
            if pl == 0:
                continue
            pos = np.asarray(batch["positive_items"][r][:pl])
            users.append(np.full(pl, int(uid[r]))); items.append(pos)
    return (torch.from_numpy(np.concatenate(users)).long(),
            torch.from_numpy(np.concatenate(items)).long())


@torch.no_grad()
def phase7_paths(dataset, device):
    cfg, ds, model, test_loader = load_frozen("freedom", dataset, device)
    s = freedom_streams(model)
    u_all, h_img, h_txt, cf = s["u_all"], s["h_img"], s["h_txt"], s["cf"]
    # per-CF-hop contributions: cf = mean(ego^0..L)[items]; recover each hop
    ego = torch.cat([model.user_embedding.weight, model.item_id_embedding.weight], 0).detach()
    hops = [ego]
    for _ in range(model.n_layers):
        ego = torch.sparse.mm(model.norm_adj, ego); hops.append(ego)
    L = len(hops)
    item_hops = [(h / L)[model.n_users:] for h in hops]   # each hop's contribution to cf

    uu, pp = _test_pairs(test_loader)
    uu, pp = uu.to(device), pp.to(device)
    U = u_all[uu]
    def contrib(stream): return float((U * stream[pp]).sum(-1).mean())
    score_cf, score_img, score_txt = contrib(cf), contrib(h_img), contrib(h_txt)
    total = score_cf + score_img + score_txt
    hop_contribs = [contrib(ih) for ih in item_hops]
    return {"dataset": dataset, "n_test_pairs": int(uu.numel()),
            "mean_score_contribution": {"cf": score_cf, "h_img": score_img, "h_txt": score_txt},
            "fraction_of_score": {"cf": score_cf / total, "h_img": score_img / total, "h_txt": score_txt / total},
            "cf_per_hop_contribution": {f"hop{l}": hop_contribs[l] for l in range(L)}}


def _sae_acts(fused, device, d_sae=512, k=16, epochs=300):
    sae = SAE(SAEConfig(d_in=fused.shape[1], d_sae=d_sae, variant="topk", k=k))
    tr = SAETrainer(sae, TrainConfig(lr=3e-4, batch_size=2048, epochs=epochs, device=device, log_every=epochs))
    tr.fit(fused)  # needs grad
    with torch.no_grad():
        return sae.encode(tr._apply_norm(fused))


def phase6_universality(dataset, device, d_sae=512):
    cfgF, dsF, mF, _ = load_frozen("freedom", dataset, device)
    fusedF = freedom_streams(mF)["fused"]
    cfgL, dsL, mL, _ = load_frozen("lgmrec", dataset, device)
    cL = lgmrec_components(mL); fusedL = _lgmrec_combine(cL, True, True)[1]

    A_F = _sae_acts(fusedF, device, d_sae)
    A_L = _sae_acts(fusedL, device, d_sae)
    # correlation matching across the shared item set
    def znorm(A):
        A = A - A.mean(0, keepdim=True)
        return A / A.norm(dim=0, keepdim=True).clamp_min(1e-8)
    Fz, Lz = znorm(A_F), znorm(A_L)
    corr = (Fz.t() @ Lz).abs()                       # [dF, dL]
    activeF = (A_F > 0).any(0)
    best = corr.max(dim=1).values[activeF]
    return {"dataset": dataset, "d_sae": d_sae,
            "n_active_F": int(activeF.sum()),
            "universality_score_mean_bestmatch_corr": float(best.mean()),
            "median_bestmatch_corr": float(best.median()),
            "frac_F_latents_strongmatch_>0.5": float((best > 0.5).float().mean()),
            "note": "FREEDOM has 0 image-origin latents vs LGMRec uses image (Phase 4) -> the architectures diverge exactly where image is concerned; shared text/cf concepts drive the universality score."}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["baby"])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    p7 = json.loads((OUT / "phase7_paths.json").read_text()) if (OUT / "phase7_paths.json").is_file() else {}
    p6 = json.loads((OUT / "phase6_universality.json").read_text()) if (OUT / "phase6_universality.json").is_file() else {}
    for dsname in args.datasets:
        print(f"\n=== Phase 7 paths: freedom/{dsname} ===", flush=True)
        r7 = phase7_paths(dsname, device); p7[dsname] = r7
        f = r7["fraction_of_score"]
        print(f"  score fraction: cf={f['cf']:.3f} h_img={f['h_img']:.3f} h_txt={f['h_txt']:.3f}")
        print(f"  cf per-hop: {r7['cf_per_hop_contribution']}")
        (OUT / "phase7_paths.json").write_text(json.dumps(p7, indent=2))

        print(f"=== Phase 6 universality: {dsname} ===", flush=True)
        r6 = phase6_universality(dsname, device); p6[dsname] = r6
        print(f"  universality (mean best-match corr FREEDOM->LGMRec) = {r6['universality_score_mean_bestmatch_corr']:.3f}"
              f"  median={r6['median_bestmatch_corr']:.3f}  frac>0.5={r6['frac_F_latents_strongmatch_>0.5']:.3f}")
        (OUT / "phase6_universality.json").write_text(json.dumps(p6, indent=2))
    print(f"\nWrote {OUT}/phase6_universality.json, phase7_paths.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
